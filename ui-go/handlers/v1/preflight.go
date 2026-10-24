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
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/photocheck"
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

// photoExts are the image extensions a dataset may hold, so "the trainer
// skips" also counts photos straight off a phone or camera (heic, avif,
// jxl), which isMedia does not list.
var photoExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".bmp": true,
	".gif": true, ".tif": true, ".tiff": true, ".heic": true, ".heif": true, ".avif": true, ".jxl": true}

// minImages is the image count below which a likeness rarely holds.
const minImages = 15

// preflightPhotoWait is how long preflight waits, per dataset, for photos
// the photo check has not analysed yet; the rest is reported as not
// checked yet.
var preflightPhotoWait = 5 * time.Second

// reviewDataset and convertFormats are the photo check's calls, replaced by
// tests (a failing review; a server with or without a new ffmpeg).
var (
	reviewDataset  = photocheck.ReviewDataset
	convertFormats = photocheck.ConvertFormats
)

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
		// the policy the job will be created with (as on POST /v1/jobs)
		Policy json.RawMessage `json:"policy"`
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
	hasPolicy := b.JobID != ""
	if len(b.Policy) > 0 && string(b.Policy) != "null" {
		if err := policyPatch(ctx, b.Policy, &pol); err != nil {
			v1api.Respond(w, r, err)
			return
		}
		hasPolicy = true
	}
	if _, err := checkConfig(raw); err != nil {
		v1api.Respond(w, r, err)
		return
	}
	var cfg map[string]any
	json.Unmarshal(raw, &cfg)
	checks := runPreflight(ctx, cfg, pol, hasPolicy)
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
			if !trainerImages[e] {
				if photoExts[e] {
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
			add(LevelWarn, fmt.Sprintf("%s: the trainer skips %s (it reads jpg, jpeg, png, webp)", where, skippedList(skipped)), skippedFix(skipped))
		}
		switch {
		case images == 0:
			add(LevelBlock, fmt.Sprintf("%s: %s has no images the trainer reads", where, folder), "")
			continue
		case images < minImages:
			add(LevelWarn, fmt.Sprintf("%s: only %d images; a likeness rarely holds with fewer than ~15", where, images), "")
		default:
			add(LevelInfo, fmt.Sprintf("%s: %d images", where, images), "")
		}
		if uncaptioned > 0 {
			add(LevelWarn, fmt.Sprintf("%s: %d of %d images have no caption (the trainer uses the default caption)", where, uncaptioned, images), "caption them (Auto caption, or datasets.caption / datasets.fixCaptions fill_missing)")
		}
		if n := flags[FlagNoTrigger] + flags[FlagTriggerCase]; n > 0 && trigger != "" {
			add(LevelWarn, fmt.Sprintf("%s: %d captions lack the trigger %q or misspell it", where, n, trigger), "fix the trigger in the captions (Fix the trigger, or datasets.fixCaptions)")
		}
		if n := flags[FlagTruncated]; n > 0 {
			add(LevelWarn, fmt.Sprintf("%s: %d captions end mid-sentence (cut off)", where, n), "caption again with a higher caption.max_new_tokens")
		}
		// a regularization set shows other people on purpose, so the photo
		// check's identity and usable lines would only be noise
		if reg, _ := ds["is_reg"].(bool); !reg {
			photoChecks(ctx, add, where, folder, ds, ext, trigger, images)
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
			add(LevelInfo, "the sync remote keeps only the newest checkpoint", "keep all checkpoints (policy.keep_checkpoints -1) to pick the best step after the server is gone")
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

// skippedList is "1 .bmp, 2 .heic".
func skippedList(skipped map[string]int) string {
	var parts []string
	for e, n := range skipped {
		parts = append(parts, fmt.Sprintf("%d %s", n, e))
	}
	sort.Strings(parts)
	return strings.Join(parts, ", ")
}

// skippedFix names datasets.fixPhotos convert only for the formats this
// server converts (photocheck.ConvertFormats); convert leaves the others,
// such as .jxl, or .heic without ffmpeg 7.1, alone.
func skippedFix(skipped map[string]int) string {
	formats := convertFormats()
	var conv, other []string
	for e := range skipped {
		if serverConverts(formats, e) {
			conv = append(conv, e)
		} else {
			other = append(other, e)
		}
	}
	sort.Strings(conv)
	sort.Strings(other)
	them := func(exts []string) string {
		if len(conv) > 0 && len(other) > 0 {
			return "the " + strings.Join(exts, ", ")
		}
		return "them"
	}
	var fix []string
	if len(conv) > 0 {
		fix = append(fix, "convert "+them(conv)+" to JPEG (Images step, or datasets.fixPhotos convert)")
	}
	if len(other) > 0 {
		fix = append(fix, "export "+them(other)+" as JPEG on the phone or computer, or add them in the studio in Safari")
	}
	return strings.Join(fix, "; ")
}

// serverConverts reports whether formats (photocheck.ConvertFormats: "bmp",
// "tiff", "heic", ...) cover the extension, as the photo check's judge reads
// them.
func serverConverts(formats []string, ext string) bool {
	f := strings.TrimPrefix(ext, ".")
	switch f {
	case "tif":
		f = "tiff"
	case "heif":
		if slices.Contains(formats, "heif") {
			return true
		}
		f = "heic"
	}
	return slices.Contains(formats, f)
}

// photoQuality are the photo flags preflight sums up as "leave them out",
// with how one and several read after the count. Only warn-level ones count
// (qualityWarns).
var photoQuality = []struct {
	flag      photocheck.Flag
	one, many string
}{
	{photocheck.FlagDuplicate, "duplicate", "duplicates"},
	{photocheck.FlagUnreadable, "unreadable", "unreadable"},
	{photocheck.FlagBlurry, "blurry", "blurry"},
	{photocheck.FlagDark, "too dark", "too dark"},
	{photocheck.FlagBright, "blown out", "blown out"},
	{photocheck.FlagScreenshot, "screenshot", "screenshots"},
	{photocheck.FlagBars, "with black bars", "with black bars"},
	{photocheck.FlagWatermark, "with a watermark", "with watermarks"},
	{photocheck.FlagOthers, "with other people", "with other people"},
	{photocheck.FlagNoFace, "with no face", "with no face"},
	{photocheck.FlagSmallFace, "with a small face", "with small faces"},
	{photocheck.FlagFaceBlurry, "with a blurry face", "with blurry faces"},
}

// photoFormat are the flags of files the trainer reads, but reads wrongly.
var photoFormat = []photocheck.Flag{photocheck.FlagGray16, photocheck.FlagTransparent, photocheck.FlagHuge, photocheck.FlagSameStem}

// photoChecks adds the photo check's lines for one dataset (PHO 11.4): the
// review the Images step shows, judged with the run's resolution, caption
// extension and trigger. They are warn or info only: a photo line never
// blocks. images is preflight's own count of photos the trainer reads.
func photoChecks(ctx context.Context, add func(level, msg, fix string), where, folder string, ds map[string]any, ext, trigger string, images int) {
	var res []int
	if v, ok := ds["resolution"]; ok {
		if raw, err := json.Marshal(v); err == nil {
			res, _ = photocheck.ParseResolutions(raw) // a bad value judges at the default
		}
	}
	rv, err := reviewDataset(ctx, folder, photocheck.Options{Resolutions: res, CaptionExt: ext, Trigger: trigger, Wait: preflightPhotoWait})
	if err != nil {
		add(LevelInfo, fmt.Sprintf("%s: photos not checked (%v)", where, err), "")
		return
	}
	has := func(flags ...photocheck.Flag) int {
		n := 0
		for _, it := range rv.Items {
			if slices.ContainsFunc(flags, func(f photocheck.Flag) bool { return slices.Contains(it.Flags, f) }) {
				n++
			}
		}
		return n
	}
	finding := func(id string) bool {
		return slices.ContainsFunc(rv.Findings, func(f photocheck.Finding) bool { return f.ID == id })
	}

	var parts, ids []string
	for _, q := range photoQuality {
		if n := rv.Counts[q.flag]; n > 0 && qualityWarns(rv, q.flag) {
			parts = append(parts, countOf(n, q.one, q.many))
			ids = append(ids, string(q.flag))
		}
	}
	if len(parts) > 0 {
		// set_aside_flagged with no flags sets aside only DefaultSetAsideFlags,
		// so the fix names the flags this line counts
		add(LevelWarn, where+": "+strings.Join(parts, ", "),
			"leave them out (Images step, or datasets.fixPhotos set_aside_flagged flags ["+strings.Join(ids, ", ")+"])")
	}
	if n := has(photocheck.FlagSmall, photocheck.FlagTiny); n > 0 {
		add(LevelWarn, fmt.Sprintf("%s: %s below %d px (the trainer never scales photos up)", where, countOf(n, "photo trains", "photos train"), rv.Resolution),
			"use the original files, not copies from a chat app")
	}
	if n := has(photoFormat...); n > 0 {
		var kinds, fix []string
		convert := false
		for _, f := range photoFormat {
			if rv.Counts[f] > 0 {
				kinds = append(kinds, photocheck.Catalog[f].Label)
				convert = convert || f != photocheck.FlagSameStem
			}
		}
		if convert {
			fix = append(fix, "convert them (Images step, or datasets.fixPhotos convert)")
		}
		if rv.Counts[photocheck.FlagSameStem] > 0 {
			fix = append(fix, "rename files that share a name (Images step, or datasets.fixPhotos rename_same_stem)")
		}
		add(LevelWarn, fmt.Sprintf("%s: %s the trainer reads wrongly (%s)", where, countOf(n, "file", "files"), strings.Join(kinds, ", ")), strings.Join(fix, "; "))
	}
	if n := rv.Counts[photocheck.FlagNotThem]; n > 0 {
		add(LevelWarn, fmt.Sprintf("%s: %s show someone else", where, countOf(n, "photo may", "photos may")),
			"check them in the Images step (datasets.checkPhotos) and leave them out, or set a reference photo (datasets.photoSettings)")
	}
	if finding("two_people") {
		add(LevelWarn, where+": these photos seem to show two different people", "mark one photo with This is them (datasets.photoSettings reference)")
	}
	// Photos still pending count as unusable, so the usable count and the
	// grade wait for a complete review.
	if rv.Pending == 0 {
		if rv.Usable < minImages && images >= minImages {
			add(LevelWarn, fmt.Sprintf("%s: only %d usable of %d photos", where, rv.Usable, images), "")
		}
		msg := fmt.Sprintf("%s: grade %s, %s", where, rv.Grade, countOf(rv.Usable, "usable photo", "usable photos"))
		if rv.Provisional {
			msg += ", before the face check"
		}
		if len(rv.Hints) > 0 {
			msg += "; " + lowerFirst(strings.TrimSuffix(rv.Hints[0], "."))
		}
		add(LevelInfo, msg, "")
	}
	if finding("faces_unchecked") {
		add(LevelInfo, where+": faces not checked", "Check faces (Images step, or datasets.checkFaces)")
	}
	if rv.Pending > 0 {
		add(LevelInfo, fmt.Sprintf("%s: %s not checked yet", where, countOf(rv.Pending, "photo", "photos")), "")
	}
}

// qualityWarns reports whether the judge set flag at warn in this review.
// Counts holds no level, so the two info cases are worked out as the judge
// does: black bars on an object (judge.go), and at most 2 photos with no
// face, at most 10 % of those face-checked (identity.go).
func qualityWarns(rv *photocheck.Review, flag photocheck.Flag) bool {
	switch flag {
	case photocheck.FlagBars:
		return rv.Kind != photocheck.KindObject
	case photocheck.FlagNoFace:
		checked := 0
		for _, it := range rv.Items {
			if !it.Pending && it.Face != nil {
				checked++
			}
		}
		n := rv.Counts[flag]
		return n > 2 || float64(n) > 0.10*float64(checked)
	}
	return true
}

// countOf is "1 photo" / "3 photos".
func countOf(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return fmt.Sprintf("%d %s", n, many)
}

// lowerFirst lowers a sentence's first letter for use after a semicolon,
// and leaves a word in capitals ("JPEG") as it is.
func lowerFirst(s string) string {
	r, n := utf8.DecodeRuneInString(s)
	if !unicode.IsUpper(r) {
		return s
	}
	if next, _ := utf8.DecodeRuneInString(s[n:]); unicode.IsUpper(next) {
		return s
	}
	return string(unicode.ToLower(r)) + s[n:]
}
