package v1

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/bundles"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Bootstrapping a fresh server from run bundles (docs/deploy.md):
// AITK_BOOTSTRAP=<ref>[,<ref>...] imports each bundle once at boot and
// continues training (AITK_BOOTSTRAP_START=0 only imports). A ref is an
// http(s) URL, <remote>:<key> of a configured remote, or a local path. Each
// import is an ordinary task (GET /v1/bundles/imports, bundle.import
// events) with `ref` set. While a bundle of this server's format downloads,
// its base model is fetched into the Hugging Face cache in parallel
// (AITK_BOOTSTRAP_PREFETCH=0 turns that off).

// bootstrapBackoff is the wait before each retry of a failed bootstrap.
var bootstrapBackoff = []time.Duration{30 * time.Second, 2 * time.Minute, 8 * time.Minute}

type bootstrapRecord struct {
	Ref     string    `json:"ref"`
	JobID   string    `json:"job_id"`
	JobName string    `json:"job_name"`
	Step    int       `json:"step"`
	At      time.Time `json:"at"`
}

func bootstrapKey(ref string) string {
	sum := sha256.Sum256([]byte(ref))
	return "bootstrap." + hex.EncodeToString(sum[:8])
}

// BootstrapRefs parses AITK_BOOTSTRAP.
func bootstrapRefs() []string {
	var out []string
	for _, r := range strings.Split(os.Getenv("AITK_BOOTSTRAP"), ",") {
		if r = strings.TrimSpace(r); r != "" {
			out = append(out, r)
		}
	}
	return out
}

func envOn(name string, def bool) bool {
	v, err := boolParam(strings.TrimSpace(os.Getenv(name)), def)
	if err != nil {
		return def
	}
	return v
}

// bootstrapLoop imports AITK_BOOTSTRAP's bundles one after another.
func bootstrapLoop(ctx context.Context) {
	refs := bootstrapRefs()
	if len(refs) == 0 || state.DB == nil || store.DB == nil {
		return
	}
	start := envOn("AITK_BOOTSTRAP_START", true)
	for _, ref := range refs {
		for attempt := 0; ; attempt++ {
			err := bootstrapOne(ctx, ref, start)
			if err == nil {
				break
			}
			if ctx.Err() != nil {
				return
			}
			if !retryable(err) || attempt >= len(bootstrapBackoff) {
				log.Printf("bootstrap %s: giving up: %v", ref, err)
				break
			}
			wait := bootstrapBackoff[attempt]
			log.Printf("bootstrap %s: attempt %d failed: %v; retrying in %s", ref, attempt+1, err, wait)
			select {
			case <-ctx.Done():
				return
			case <-time.After(wait):
			}
		}
	}
}

// retryable: transfers and the remote can fail transiently; a bundle that
// is invalid or conflicts with this server will not get better.
func retryable(err error) bool {
	var p *v1api.Problem
	if errors.As(err, &p) {
		switch p.Code {
		case v1api.CodeInvalidRequest, v1api.CodeInvalidConfig, v1api.CodeConflict, v1api.CodeDatasetExists, v1api.CodeTooLarge:
			return false
		}
	}
	return true
}

// localStep is how far a local job got: its row or its newest save.
func localStep(ctx context.Context, j *store.Job) int {
	step := j.Step
	if folder, err := jobFolder(ctx, j.Name); err == nil {
		if cps, err := artifacts.Checkpoints(folder); err == nil {
			for _, cp := range cps {
				if s, ok := bundles.CheckpointStep(cp.Path); ok && s > step {
					step = s
				} else if cp.Step != nil && *cp.Step > step {
					step = *cp.Step
				}
			}
		}
	}
	return step
}

// alreadyHere reports (and logs) a job of that name at the same or a later
// step: then the bundle is skipped. An older local job is left alone too,
// with a warning, since importing would need its name.
func alreadyHere(ctx context.Context, ref, name string, step int) (bool, error) {
	j, err := store.FindFirstJob(ctx, store.JobQuery{Where: `"name" = ?`, Args: []any{name}})
	if err != nil || j == nil {
		return false, err
	}
	have := localStep(ctx, j)
	if have >= step {
		log.Printf("bootstrap %s: job %s is already here at step %d (bundle: step %d); skipped", ref, name, have, step)
	} else {
		log.Printf("bootstrap %s: job %s exists at step %d, older than the bundle's %d; not replacing it (delete the job to import the bundle)", ref, name, have, step)
	}
	return true, state.SetSetting(ctx, bootstrapKey(ref), mustJSON(bootstrapRecord{Ref: ref, JobID: j.ID, JobName: name, Step: have, At: time.Now().UTC()}))
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// bootstrapOne imports one ref unless it was imported before (and its job
// still exists) or a job of its name is already here.
func bootstrapOne(ctx context.Context, ref string, start bool) error {
	if v, ok, err := state.Setting(ctx, bootstrapKey(ref)); err != nil {
		return err
	} else if ok {
		var rec bootstrapRecord
		if json.Unmarshal([]byte(v), &rec) == nil {
			if j, err := store.GetJob(ctx, rec.JobID); err == nil && j != nil {
				log.Printf("bootstrap %s: imported before as job %s; skipped", ref, j.Name)
				return nil
			}
		}
	}
	src, err := parseRef(ctx, ref)
	if err != nil {
		return err
	}

	// what the bundle is, before the big transfer: skip or prefetch early
	var prefetch <-chan string
	if m := src.peek(ctx); m != nil {
		if here, err := alreadyHere(ctx, ref, m.Name, m.Step); here || err != nil {
			return err
		}
		prefetch = prefetchBaseModel(ctx, m.BaseModel)
	}

	t := bundles.NewTask(ctx, bundles.KindImport)
	t.Ref, t.Source = &ref, src.kind
	if src.kind == "url" {
		t.URL = &src.shown
	} else if src.kind == "remote" {
		t.Remote, t.Key = &src.remoteName, &src.key
	}
	t.Save(ctx, true)

	zipPath, uploadID := src.path, ""
	switch src.kind {
	case "url":
		uploadID, err = stageURL(ctx, t, src.url)
	case "remote":
		uploadID, err = stageRemote(ctx, t, src.client, src.key, src.size)
	}
	if err != nil {
		failTask(ctx, t, err)
		return err
	}
	if uploadID != "" {
		t.UploadID = &uploadID
		if zipPath, err = bundles.UploadPath(ctx, uploadID); err != nil {
			failTask(ctx, t, err)
			return err
		}
	}
	t.Phase, t.Bytes, t.BytesDone = "reading", 0, 0
	t.Save(ctx, true)

	name, step, base, err := readBundleHead(zipPath)
	if err != nil {
		failTask(ctx, t, err)
		return err
	}
	if here, err := alreadyHere(ctx, ref, name, step); here || err != nil {
		if uploadID != "" {
			bundles.DeleteUpload(ctx, uploadID)
		}
		t.UploadID = nil
		t.State, t.Phase = bundles.TaskDone, ""
		t.Warnings = append(t.Warnings, "skipped: job "+name+" is already on this server")
		t.Save(ctx, true)
		return err
	}
	if prefetch == nil {
		prefetch = prefetchBaseModel(ctx, base)
	}

	prep, err := prepareZip(ctx, zipPath, uploadID, importOpts{})
	if err != nil {
		failTask(ctx, t, err)
		return err
	}
	err = restoreImport(ctx, t, prep, importOpts{})
	prep.close()
	if err != nil {
		failTask(ctx, t, err)
		return err
	}
	job, err := store.GetJob(ctx, *t.JobID)
	if err != nil || job == nil {
		return fmt.Errorf("imported job vanished: %v", err)
	}
	if err := state.SetSetting(ctx, bootstrapKey(ref), mustJSON(bootstrapRecord{Ref: ref, JobID: job.ID,
		JobName: job.Name, Step: job.Step, At: time.Now().UTC()})); err != nil {
		log.Printf("bootstrap %s: record: %v", ref, err)
	}
	log.Printf("bootstrap %s: imported job %s at step %d", ref, job.Name, job.Step)
	if !start {
		return nil
	}
	if prefetch != nil {
		// the trainer would download the same files; let the prefetch finish
		if msg := <-prefetch; msg != "" {
			t.Warnings = append(t.Warnings, msg)
		}
	}
	startImported(t, job)
	t.Save(ctx, true)
	log.Printf("bootstrap %s: job %s queued", ref, job.Name)
	return nil
}

// readBundleHead opens a staged bundle for its job name, step and base model.
func readBundleHead(zipPath string) (name string, step int, base string, err error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return "", 0, "", v1api.Errorf(v1api.CodeInvalidRequest, "not a zip archive: "+err.Error())
	}
	defer zr.Close()
	plan, err := bundles.Read(&zr.Reader)
	if err != nil {
		return "", 0, "", err
	}
	base = plan.Manifest.BaseModel
	if base == "" {
		base, _, _ = bundles.Describe(plan.Manifest.Config)
	}
	return plan.Manifest.Name, plan.Manifest.Step, base, nil
}

type bundleRef struct {
	kind            string // url | remote | local
	url, shown      string
	remoteName, key string
	client          remote.Client
	size            int64
	path            string
}

// parseRef: an http(s) URL, a local path (absolute or ./), or
// <remote>:<key> of a configured remote.
func parseRef(ctx context.Context, ref string) (*bundleRef, error) {
	if strings.HasPrefix(ref, "http://") || strings.HasPrefix(ref, "https://") {
		u := ref
		shown := u
		if i := strings.IndexAny(shown, "?#"); i >= 0 {
			shown = shown[:i]
		}
		return &bundleRef{kind: "url", url: u, shown: shown}, nil
	}
	if filepath.IsAbs(ref) || strings.HasPrefix(ref, "/") || strings.HasPrefix(ref, "./") || strings.HasPrefix(ref, "../") {
		st, err := os.Stat(ref)
		if err != nil || !st.Mode().IsRegular() {
			return nil, v1api.Errorf(v1api.CodeNotFound, "no bundle file at "+ref)
		}
		abs, _ := filepath.Abs(ref)
		return &bundleRef{kind: "local", path: abs}, nil
	}
	name, key, ok := strings.Cut(ref, ":")
	if !ok || name == "" || key == "" {
		return nil, v1api.Errorf(v1api.CodeInvalidRequest, "bootstrap ref "+ref+" is not a URL, a path or <remote>:<key>")
	}
	rem, c, err := openRemote(ctx, name)
	if err != nil {
		return nil, err
	}
	obj, err := c.Stat(ctx, key)
	if errors.Is(err, remote.ErrNotFound) {
		return nil, v1api.Errorf(v1api.CodeNotFound, "remote "+rem.Name+" has no "+key)
	} else if err != nil {
		return nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	return &bundleRef{kind: "remote", remoteName: rem.Name, key: key, client: c, size: obj.Size}, nil
}

// peek learns the bundle's job before the transfer: a library sidecar, the
// manifest at the head of a URL (one ranged GET), or the local file.
func (b *bundleRef) peek(ctx context.Context) *bundles.Manifest {
	switch b.kind {
	case "remote":
		if id, ok := strings.CutSuffix(b.key, ".zip"); ok {
			if raw, err := b.client.GetBytes(ctx, id+".json"); err == nil {
				var s bundles.Summary
				if json.Unmarshal(raw, &s) == nil && s.Name != "" {
					return &bundles.Manifest{Name: s.Name, Step: s.Step, BaseModel: s.BaseModel}
				}
			}
		}
	case "url":
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.url, nil)
		if err != nil {
			return nil
		}
		req.Header.Set("Range", "bytes=0-1048575")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
			return nil
		}
		head, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if m, ok := bundles.PeekManifest(head); ok {
			return m
		}
	case "local":
		if name, step, base, err := readBundleHead(b.path); err == nil {
			return &bundles.Manifest{Name: name, Step: step, BaseModel: base}
		}
	}
	return nil
}

// hubRepo reports whether a config's model.name_or_path is a Hugging Face
// repo id (org/name) rather than a local path or a single file.
func hubRepo(base string) bool {
	if base == "" || strings.ContainsAny(base, `\:`) || strings.HasPrefix(base, "/") || strings.HasPrefix(base, ".") {
		return false
	}
	org, name, ok := strings.Cut(base, "/")
	if !ok || org == "" || name == "" || strings.Contains(name, "/") || strings.Contains(name, ".safetensors") {
		return false
	}
	_, err := os.Stat(base)
	return err != nil
}

// prefetchScript downloads what diffusers' from_pretrained would for a
// diffusers-format repo (DiffusionPipeline.download: model_index.json and the
// component folders, not the single-file checkpoints beside them). Other
// repos are left to the trainer.
const prefetchScript = `import os, sys
repo = sys.argv[1]
tok = os.environ.get("HF_TOKEN") or None
from huggingface_hub import list_repo_files
if "model_index.json" not in list_repo_files(repo, token=tok):
    print("not a diffusers repo; left to the trainer")
    sys.exit(0)
from diffusers import DiffusionPipeline
DiffusionPipeline.download(repo, token=tok)
print("done")
`

// prefetchBaseModel starts fetching base into the HF cache; the channel
// yields "" when it finished (or was not needed) and a warning otherwise.
func prefetchBaseModel(ctx context.Context, base string) <-chan string {
	if !hubRepo(base) || !envOn("AITK_BOOTSTRAP_PREFETCH", true) {
		return nil
	}
	ch := make(chan string, 1)
	go func() {
		ctx, cancel := context.WithTimeout(ctx, 3*time.Hour)
		defer cancel()
		cmd := exec.CommandContext(ctx, paths.ResolvePython(), "-c", prefetchScript, base)
		cmd.Dir = paths.ToolkitRoot
		cmd.Env = os.Environ()
		if tok, err := settings.HFToken(ctx); err == nil && tok != "" {
			cmd.Env = append(cmd.Env, "HF_TOKEN="+tok)
		}
		var out bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &out
		log.Printf("bootstrap: prefetching base model %s", base)
		started := time.Now()
		if err := cmd.Run(); err != nil {
			tail := strings.TrimSpace(out.String())
			if len(tail) > 1000 {
				tail = "..." + tail[len(tail)-1000:]
			}
			msg := fmt.Sprintf("base model prefetch of %s failed (the trainer downloads it instead): %v %s", base, err, tail)
			log.Printf("bootstrap: %s", msg)
			ch <- msg
			return
		}
		log.Printf("bootstrap: base model %s ready in %s (%s)", base, time.Since(started).Round(time.Second), strings.TrimSpace(out.String()))
		ch <- ""
	}()
	return ch
}
