package v1

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/templates"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Preflight levels: block means the run cannot work as configured; warn is
// probably a mistake; info is worth knowing before paying for the GPU.
const (
	LevelBlock = "block"
	LevelWarn  = "warn"
	LevelInfo  = "info"
)

// trainerImages are the extensions ai-toolkit's dataset loader reads
// (toolkit/data_loader.py image_extensions); other images are skipped.
var trainerImages = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true}

type preflightCheck struct {
	Level   string `json:"level"`
	Message string `json:"message"`
	Fix     string `json:"fix,omitempty"`
}

// preflightJob is POST /v1/jobs/preflight {config | job_id}: what would go
// wrong or be wasted if this config ran now, before a GPU is paid for.
func preflightJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var b struct {
		Config json.RawMessage `json:"config"`
		JobID  string          `json:"job_id"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	raw := b.Config
	var pol state.Policy
	if b.JobID != "" {
		j, err := store.GetJob(ctx, b.JobID)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if j == nil {
			v1api.Fail(w, v1api.CodeNotFound, "job not found")
			return
		}
		if len(raw) == 0 {
			raw = json.RawMessage(j.JobConfig)
		}
		if pol, err = state.GetPolicy(ctx, j.ID); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	if len(raw) == 0 {
		v1api.Fail(w, v1api.CodeInvalidRequest, "config or job_id is required")
		return
	}
	if _, err := checkConfig(raw); err != nil {
		v1api.Respond(w, r, err)
		return
	}
	var cfg map[string]any
	json.Unmarshal(raw, &cfg)
	checks := runPreflight(ctx, cfg, pol, b.JobID != "")
	ok := true
	for _, c := range checks {
		if c.Level == LevelBlock {
			ok = false
		}
	}
	v1api.OK(w, map[string]any{"ok": ok, "checks": checks})
}

func asMap(v any) map[string]any { m, _ := v.(map[string]any); return m }

func asNum(v any) (float64, bool) { f, ok := v.(float64); return f, ok }

func runPreflight(ctx context.Context, cfg map[string]any, pol state.Policy, hasJob bool) []preflightCheck {
	checks := []preflightCheck{}
	add := func(level, msg, fix string) { checks = append(checks, preflightCheck{level, msg, fix}) }

	for _, p := range templates.Check(cfg).Problems {
		add(LevelWarn, p.Path+": "+p.Message, p.Fix)
	}
	procs, _ := asMap(cfg["config"])["process"].([]any)
	p0 := asMap(procs[0])
	train := asMap(p0["train"])
	trigger, _ := p0["trigger_word"].(string)

	// datasets: there, readable by the trainer, captioned
	datasets, _ := p0["datasets"].([]any)
	if len(datasets) == 0 {
		add(LevelBlock, "the config has no datasets", "add one under config.process[0].datasets")
	}
	for i, d := range datasets {
		ds := asMap(d)
		folder, _ := ds["folder_path"].(string)
		where := fmt.Sprintf("datasets[%d]", i)
		if folder == "" {
			add(LevelBlock, where+" has no folder_path", "")
			continue
		}
		if st, err := os.Stat(folder); err != nil || !st.IsDir() {
			add(LevelBlock, fmt.Sprintf("%s: %s does not exist on this server", where, folder), "upload the dataset, or import the run's bundle")
			continue
		}
		ext, _ := ds["caption_ext"].(string)
		if ext == "" {
			ext = "txt"
		}
		var images, uncaptioned int
		skipped := map[string]int{}
		flags := map[string]int{}
		walkDataset(folder, func(rel string, _ fs.FileInfo) {
			e := strings.ToLower(filepath.Ext(rel))
			if !isMedia(rel) {
				return
			}
			if !trainerImages[e] {
				if imageExts[e] {
					skipped[e]++
				}
				return
			}
			images++
			full := filepath.Join(folder, filepath.FromSlash(rel))
			b, err := os.ReadFile(strings.TrimSuffix(full, filepath.Ext(full)) + "." + strings.TrimLeft(ext, "."))
			var caption *string
			if err == nil {
				s := string(b)
				caption = &s
			}
			for _, f := range captionFlags(caption, trigger) {
				flags[f]++
			}
			if caption == nil || strings.TrimSpace(*caption) == "" {
				uncaptioned++
			}
		})
		if len(skipped) > 0 {
			var parts []string
			for e, n := range skipped {
				parts = append(parts, fmt.Sprintf("%d %s", n, e))
			}
			sort.Strings(parts)
			add(LevelWarn, fmt.Sprintf("%s: the trainer skips %s (it reads jpg, jpeg, png, webp)", where, strings.Join(parts, ", ")), "convert them to JPEG")
		}
		switch {
		case images == 0:
			add(LevelBlock, fmt.Sprintf("%s: %s has no images the trainer reads", where, folder), "")
			continue
		case images < 10:
			add(LevelWarn, fmt.Sprintf("%s: only %d images; a likeness rarely holds with fewer than ~15", where, images), "")
		default:
			add(LevelInfo, fmt.Sprintf("%s: %d images", where, images), "")
		}
		if uncaptioned > 0 {
			add(LevelWarn, fmt.Sprintf("%s: %d of %d images have no caption (the trainer uses the default caption)", where, uncaptioned, images), "datasets.caption, or datasets.fixCaptions with fill_missing")
		}
		if n := flags[FlagNoTrigger] + flags[FlagTriggerCase]; n > 0 && trigger != "" {
			add(LevelWarn, fmt.Sprintf("%s: %d captions lack the trigger %q or misspell it", where, n, trigger), "datasets.fixCaptions with fix_trigger_case and ensure_trigger")
		}
		if n := flags[FlagTruncated]; n > 0 {
			add(LevelWarn, fmt.Sprintf("%s: %d captions end mid-sentence (cut off)", where, n), "caption again with a higher caption.max_new_tokens")
		}
	}

	// a local model path that is not there
	if model := asMap(p0["model"]); model != nil {
		if path, _ := model["name_or_path"].(string); strings.HasPrefix(path, "/") || strings.HasPrefix(path, "./") {
			if _, err := os.Stat(path); err != nil {
				add(LevelBlock, "model.name_or_path "+path+" does not exist on this server", "")
			}
		}
	}

	// disk for checkpoints, samples and caches
	if root, err := settings.TrainingFolder(ctx); err == nil {
		probe := root
		for {
			if _, err := os.Stat(probe); err == nil || filepath.Dir(probe) == probe {
				break
			}
			probe = filepath.Dir(probe)
		}
		if free, _, err := diskSpace(probe); err == nil {
			gb := float64(free) / (1 << 30)
			switch {
			case gb < 2:
				add(LevelBlock, fmt.Sprintf("only %.1f GB free where checkpoints are written (%s)", gb, root), "free space or mount a larger volume")
			case gb < 10:
				add(LevelWarn, fmt.Sprintf("%.1f GB free where checkpoints are written (%s)", gb, root), "")
			}
		}
	}

	// checkpoints kept on disk while training
	steps, _ := asNum(train["steps"])
	save := asMap(p0["save"])
	every, _ := asNum(save["save_every"])
	keepN, hasKeep := asNum(save["max_step_saves_to_keep"])
	if steps > 0 && every > 0 && hasKeep && keepN*every < steps {
		add(LevelInfo, fmt.Sprintf("only the last %d checkpoints stay on disk (from about step %d of %d)", int(keepN), int(steps-keepN*every), int(steps)),
			"raise save.max_step_saves_to_keep to pick from earlier steps")
	}

	// where the results go, and whether when done can keep them
	syncRemote, uploadRemote := "", ""
	if hasJob {
		if pol.SyncRemote != nil {
			syncRemote = *pol.SyncRemote
		}
		if pol.AutoUploadRemote != nil {
			uploadRemote = *pol.AutoUploadRemote
		}
	}
	if !hasJob || pol.SyncRemote == nil {
		if v, ok, _ := state.Setting(ctx, state.SettingSyncRemoteDefault); ok {
			syncRemote = v
		}
	}
	if !hasJob || pol.AutoUploadRemote == nil {
		if v, ok, _ := state.Setting(ctx, state.SettingAutoUploadRemoteDefault); ok {
			uploadRemote = v
		}
	}
	if syncRemote != "" {
		if rem, c, err := remote.Open(ctx, syncRemote); err != nil {
			add(LevelWarn, "sync remote "+syncRemote+": "+err.Error(), "")
		} else if _, err := c.Stat(ctx, "aitk-preflight"); err != nil && !errors.Is(err, remote.ErrNotFound) {
			add(LevelWarn, "sync remote "+rem.Name+" is not reachable: "+err.Error(), "check its credentials and endpoint")
		} else if n, ok := state.KeepCheckpoints(ctx, pol); !ok || n == 0 {
			add(LevelInfo, "the sync remote keeps only the newest checkpoint", "set policy.keep_checkpoints (-1 all) to pick the best step after the server is gone")
		}
	}
	if life, err := lifecycle.Get(); err == nil && (life.Mode == lifecycle.ModeStop || life.Mode == lifecycle.ModeDestroy) {
		if syncRemote == "" && uploadRemote == "" && strings.TrimSpace(os.Getenv("AITK_WHEN_DONE_BUNDLE")) == "" {
			add(LevelWarn, "when done is "+life.Mode+" but the results go to no remote: it will hold the server up instead",
				"set a sync remote (policy.sync_remote) or AITK_WHEN_DONE_BUNDLE")
		}
	}
	return checks
}
