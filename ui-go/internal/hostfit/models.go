package hostfit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit/hfmock"
	"github.com/geocine/aitoolkit/internal/netguard"
)

// ModelFiles is what ai-toolkit's loader for one arch downloads.
type ModelFiles struct {
	Arch  string
	Repos []RepoFiles // in download order
}

// RepoFiles is the part of one Hugging Face repo a loader fetches.
type RepoFiles struct {
	Repo       string   // "black-forest-labs/FLUX.1-dev"; NameOrPath = the job's own repo (Default when the job names none)
	Default    string   // the repo a NameOrPath entry falls back to
	Include    []string // "dir/" (everything under it), or a path glob ("flux2-dev.safetensors", "model-*.safetensors")
	Exclude    []string // basename globs: "*.bin"
	FallbackGB float64  // measured 2026-10-24 from the HF tree API; used when it is unreachable (0 = unknown)
}

// NameOrPath stands for the job's own model repo in RepoFiles.Repo.
const NameOrPath = "{name_or_path}"

// diffusers is the layout a diffusers pipeline downloads (the root
// single-file checkpoints are not fetched).
var diffusers = []string{"transformer/", "transformer_2/", "text_encoder/", "text_encoder_2/", "text_encoder_3/",
	"vae/", "tokenizer/", "tokenizer_2/", "tokenizer_3/", "scheduler/", "image_processor/", "processor/", "model_index.json"}

// weightsOnly drops the non-safetensors copies repos often carry.
var weightsOnly = []string{"*.bin", "*.pth", "*.pt", "*.ckpt", "*.msgpack", "*.h5", "*.onnx", "*.gguf"}

func pipeline(def string, gb float64) []RepoFiles {
	return []RepoFiles{{Repo: NameOrPath, Default: def, Include: diffusers, Exclude: weightsOnly, FallbackGB: gb}}
}

// modelFiles follows each loader's actual downloads (IMAGE-PLAN §3.5).
// FLUX.2 fetches the root transformer and ae from its repo plus the
// sharded Mistral text encoder from another (not the 48 GB consolidated
// copy). Sizes for flux_kontext, zimage, krea2 and ltx2 are UNCONFIRMED.
var modelFiles = map[string]ModelFiles{
	"flux":                 {"flux", pipeline("black-forest-labs/FLUX.1-dev", 33.7)},
	"flux_kontext":         {"flux_kontext", pipeline("black-forest-labs/FLUX.1-Kontext-dev", 33.7)},
	"qwen_image":           {"qwen_image", pipeline("Qwen/Qwen-Image", 57.7)},
	"qwen_image_edit":      {"qwen_image_edit", pipeline("Qwen/Qwen-Image-Edit", 57.7)},
	"qwen_image_edit_plus": {"qwen_image_edit_plus", pipeline("Qwen/Qwen-Image-Edit-2509", 57.7)},
	"wan22_14b":            {"wan22_14b", pipeline("Wan-AI/Wan2.2-T2V-A14B-Diffusers", 126.2)},
	"wan21":                {"wan21", pipeline("Wan-AI/Wan2.1-T2V-14B-Diffusers", 80.4)},
	"wan22_5b":             {"wan22_5b", pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 34.2)},
	"zimage":               {"zimage", pipeline("Tongyi-MAI/Z-Image-Turbo", 32.8)},
	"flux2": {"flux2", []RepoFiles{
		{Repo: NameOrPath, Default: "black-forest-labs/FLUX.2-dev", Include: []string{"flux2-dev.safetensors", "ae.safetensors"}, FallbackGB: 64.8},
		{Repo: "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
			Include:    []string{"model-*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tekken.json"},
			FallbackGB: 48.0},
	}},
	"krea2": {"krea2", []RepoFiles{{Repo: NameOrPath, Include: []string{"*.safetensors"}, FallbackGB: 36}}},
	"ltx2": {"ltx2", []RepoFiles{
		{Repo: NameOrPath, Default: "Lightricks/LTX-2", Include: []string{"*.safetensors"}},
		{Repo: "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized", Include: []string{"model-*.safetensors", "*.json"}},
	}},
}

// genericFiles is what an arch the table does not know downloads: the
// safetensors at the root of its own repo, when it names one.
var genericFiles = []RepoFiles{{Repo: NameOrPath, Include: []string{"*.safetensors"}}}

const (
	hfBase       = "https://huggingface.co"
	hfBodyCap    = 8 << 20
	hfTimeout    = 10 * time.Second
	hfCacheTTL   = 24 * time.Hour
	hfFailTTL    = 5 * time.Minute // a failed tree is retried after this, so an outage is not hammered
	hfMaxPages   = 10
	hfPathPrefix = "/api/models/"
)

// hfRepo is a Hugging Face repo id ("owner/name"), not a local path.
var hfRepo = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)

// SizerOptions configures NewSizer. The zero value is the production sizer.
type SizerOptions struct {
	Base string           // default https://huggingface.co
	HTTP *http.Client     // nil = netguard client (public addresses only, proxy honoured, 10 s)
	Now  func() time.Time // tests: fake clock
}

// ModelSizer sizes model downloads from the public Hugging Face tree API
// (no token; gated repos list sizes too), with a 24 h cache per repo and
// the table sizes as the fallback.
type ModelSizer struct {
	base string
	http *http.Client
	now  func() time.Time

	mu    sync.Mutex
	trees map[string]treeEntry
}

type treeEntry struct {
	files []hfFile
	err   error
	at    time.Time
}

type hfFile struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		Size int64 `json:"size"`
	} `json:"lfs,omitempty"`
}

// NewSizer builds a sizer.
func NewSizer(o SizerOptions) *ModelSizer {
	s := &ModelSizer{base: strings.TrimRight(o.Base, "/"), http: o.HTTP, now: o.Now, trees: map[string]treeEntry{}}
	if s.base == "" {
		s.base = hfBase
	}
	if s.http == nil {
		s.http = netguard.NewClient(netguard.Options{Proxy: true, Timeout: hfTimeout})
	}
	if s.now == nil {
		s.now = time.Now
	}
	return s
}

// DefaultSizer is the sizer every caller uses, read at call time (never
// copied). InitSizer sets it; tests swap it and restore it in t.Cleanup.
var DefaultSizer *ModelSizer

var (
	sizerMu  sync.Mutex
	sizerSim *hfmock.Server
)

// InitSizer sets DefaultSizer. It is idempotent: a later call replaces the
// earlier sizer (and stops the simulated Hugging Face it started). With
// simulate the tree API is an in-process hfmock, so nothing leaves the
// machine.
func InitSizer(simulate bool) {
	sizerMu.Lock()
	defer sizerMu.Unlock()
	if sizerSim != nil {
		sizerSim.Close()
		sizerSim = nil
	}
	if !simulate {
		DefaultSizer = NewSizer(SizerOptions{})
		return
	}
	sizerSim = hfmock.New(hfmock.Options{})
	sizerSim.Start()
	DefaultSizer = NewSizer(SizerOptions{Base: sizerSim.URL(), HTTP: netguard.NewClient(netguard.Options{AllowLoopback: true})})
}

// Size is the download size of model m in GB and where it came from: "hf"
// (the tree API), "table" (a repo was unreachable, or the job's model is a
// local path) or "none" (unknown). A nil sizer uses the table.
func (s *ModelSizer) Size(ctx context.Context, m ModelRef) (gb float64, source string) {
	repos := genericFiles
	if f, ok := modelFiles[m.Arch]; ok {
		repos = f.Repos
	}
	var fallback, total float64
	live := s != nil
	for _, rf := range repos {
		fallback += rf.FallbackGB
		repo := rf.Repo
		if repo == NameOrPath {
			repo = rf.Default
			if p := strings.TrimSpace(m.NameOrPath); p != "" {
				repo = p
			}
		}
		if !live || !hfRepo.MatchString(repo) {
			live = false
			continue
		}
		files, err := s.tree(ctx, repo)
		if err != nil {
			live = false
			continue
		}
		total += sumFiles(files, rf)
	}
	switch {
	case live && total > 0:
		return round1(total / 1e9), "hf"
	case fallback > 0:
		return fallback, "table"
	}
	return 0, "none"
}

// sumFiles adds the bytes of the files rf includes and does not exclude.
func sumFiles(files []hfFile, rf RepoFiles) float64 {
	var n int64
	for _, f := range files {
		if f.Type != "file" || !matchAny(f.Path, rf.Include) || matchAny(path.Base(f.Path), rf.Exclude) {
			continue
		}
		size := f.Size
		if f.LFS != nil && f.LFS.Size > 0 {
			size = f.LFS.Size
		}
		n += size
	}
	return float64(n)
}

// matchAny: a pattern ending in "/" matches everything under that
// directory; any other pattern is a path glob over the whole path.
func matchAny(p string, patterns []string) bool {
	for _, pat := range patterns {
		if strings.HasSuffix(pat, "/") {
			if strings.HasPrefix(p, pat) {
				return true
			}
			continue
		}
		if ok, _ := path.Match(pat, p); ok {
			return true
		}
	}
	return false
}

// tree is repo's recursive file list, cached 24 h (failures 5 min).
func (s *ModelSizer) tree(ctx context.Context, repo string) ([]hfFile, error) {
	s.mu.Lock()
	e, ok := s.trees[repo]
	s.mu.Unlock()
	now := s.now()
	if ok && (e.err == nil && now.Sub(e.at) < hfCacheTTL || e.err != nil && now.Sub(e.at) < hfFailTTL) {
		return e.files, e.err
	}
	if err := ctx.Err(); err != nil { // the caller is gone or out of time: nothing was asked
		return nil, err
	}
	files, err := s.fetch(ctx, repo)
	// A cancelled caller says nothing about the repo. A deadline does: a
	// blackholed huggingface.co times out, and caching that keeps every
	// NeedFor from stalling on it for the next five minutes.
	if errors.Is(ctx.Err(), context.Canceled) {
		return files, err
	}
	if err != nil {
		log.Printf("hostfit: model size for %s: %v", repo, err)
	}
	s.mu.Lock()
	s.trees[repo] = treeEntry{files: files, err: err, at: now}
	s.mu.Unlock()
	return files, err
}

// fetch reads every page of the tree API. Next pages come from the Link
// header and must stay on the sizer's own host.
func (s *ModelSizer) fetch(ctx context.Context, repo string) ([]hfFile, error) {
	owner, name, _ := strings.Cut(repo, "/")
	u := s.base + hfPathPrefix + url.PathEscape(owner) + "/" + url.PathEscape(name) + "/tree/main?recursive=true"
	base, err := url.Parse(s.base)
	if err != nil {
		return nil, err
	}
	var out []hfFile
	for page := 0; u != "" && page < hfMaxPages; page++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Accept", "application/json")
		resp, err := s.http.Do(req)
		if err != nil {
			return nil, err
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, hfBodyCap+1))
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
		}
		if len(body) > hfBodyCap {
			return nil, fmt.Errorf("tree listing over %d bytes", hfBodyCap)
		}
		var files []hfFile
		if err := json.Unmarshal(body, &files); err != nil {
			return nil, fmt.Errorf("tree listing: %w", err)
		}
		out = append(out, files...)
		u = nextPage(resp.Header.Get("Link"), base)
	}
	return out, nil
}

// nextPage is the rel="next" URL of a Link header, only on base's host.
func nextPage(link string, base *url.URL) string {
	for _, part := range strings.Split(link, ",") {
		target, params, ok := strings.Cut(strings.TrimSpace(part), ";")
		if !ok || !strings.Contains(params, `rel="next"`) {
			continue
		}
		next, err := base.Parse(strings.Trim(strings.TrimSpace(target), "<>"))
		if err != nil || next.Scheme != base.Scheme || next.Host != base.Host {
			return ""
		}
		return next.String()
	}
	return ""
}
