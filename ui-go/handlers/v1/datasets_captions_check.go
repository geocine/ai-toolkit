package v1

import (
	"io/fs"
	"net/http"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/collation"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Caption flags (GET /v1/datasets/{name}/captions/check).
const (
	FlagMissing     = "missing"      // no caption (the trainer uses the default caption)
	FlagNoTrigger   = "no_trigger"   // neither [trigger] nor the trigger word: the trainer prepends it
	FlagTriggerCase = "trigger_case" // "[Trigger]", "Jihan" for "jihan": not replaced or matched exactly
	FlagTruncated   = "truncated"    // ends mid-sentence (the captioner ran out of tokens)
	FlagHedging     = "hedging"      // "appears to be", "possibly": weak, uncertain captions
	FlagPreamble    = "preamble"     // "Here is a caption", "This image shows"
	FlagDuplicate   = "duplicate"    // the same text as another file's caption
)

var (
	hedgingRe  = regexp.MustCompile(`(?i)\b(appears to( be)?|seems to( be)?|possibly|probably|likely|it looks like|might be|could be)\b`)
	preambleRe = regexp.MustCompile(`(?i)^\s*(here is|here's|sure[,!]|this image (shows|depicts|features)|the image (shows|depicts|features)|caption:)`)
	bracketRe  = regexp.MustCompile(`(?i)\[\s*trigger\s*\]`)
)

type captionCheck struct {
	File    string   `json:"file"`
	Caption *string  `json:"caption"`
	Flags   []string `json:"flags"`
}

// danglingRe: how a caption cut off by the token limit ends (on a word that
// needs a continuation, or a comma). Tag lists and sentences without a final
// period are fine.
var danglingRe = regexp.MustCompile(`(?i)(,|;|:|-|\b(a|an|the|of|in|on|with|and|or|to|for|by|at|from|as|its|their|his|her|is|are|was|were|that|which|while|featuring|wearing|holding|rendered|creating))\s*$`)

func truncated(c string) bool { return danglingRe.MatchString(c) }

// captionFlags judges one caption; trigger is the run's trigger word ("" when unknown).
func captionFlags(caption *string, trigger string) []string {
	flags := []string{}
	if caption == nil || strings.TrimSpace(*caption) == "" {
		return append(flags, FlagMissing)
	}
	c := strings.TrimSpace(*caption)
	exact := strings.Contains(c, "[trigger]") || strings.Contains(c, "[name]") || (trigger != "" && strings.Contains(c, trigger))
	if !exact {
		if bracketRe.MatchString(c) || (trigger != "" && strings.Contains(strings.ToLower(c), strings.ToLower(trigger))) {
			flags = append(flags, FlagTriggerCase)
		} else {
			flags = append(flags, FlagNoTrigger)
		}
	}
	if truncated(c) {
		flags = append(flags, FlagTruncated)
	}
	if hedgingRe.MatchString(c) {
		flags = append(flags, FlagHedging)
	}
	if preambleRe.MatchString(c) {
		flags = append(flags, FlagPreamble)
	}
	return flags
}

// datasetCaptions lists the dataset's media files with their captions, sorted.
func datasetCaptions(dir string) ([]captionCheck, error) {
	var out []captionCheck
	err := walkDataset(dir, func(rel string, _ fs.FileInfo) {
		if isMedia(rel) {
			out = append(out, captionCheck{File: rel, Caption: readCaption(filepath.Join(dir, filepath.FromSlash(rel)))})
		}
	})
	sort.SliceStable(out, func(i, j int) bool { return collation.Compare(out[i].File, out[j].File) < 0 })
	return out, err
}

// checkCaptions is GET /v1/datasets/{name}/captions/check?trigger=: every
// caption with what is wrong with it, and counts per flag.
func checkCaptions(w http.ResponseWriter, r *http.Request) {
	dir, ok := datasetDir(w, r, r.PathValue("name"), true)
	if !ok {
		return
	}
	trigger := strings.TrimSpace(r.URL.Query().Get("trigger"))
	items, err := datasetCaptions(dir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	seen := map[string][]int{}
	for i := range items {
		items[i].Flags = captionFlags(items[i].Caption, trigger)
		if c := items[i].Caption; c != nil && strings.TrimSpace(*c) != "" {
			k := strings.ToLower(strings.TrimSpace(*c))
			seen[k] = append(seen[k], i)
		}
	}
	for _, idx := range seen {
		if len(idx) > 1 {
			for _, i := range idx {
				items[i].Flags = append(items[i].Flags, FlagDuplicate)
			}
		}
	}
	counts := map[string]int{}
	flagged := 0
	for _, it := range items {
		if len(it.Flags) > 0 {
			flagged++
		}
		for _, f := range it.Flags {
			counts[f]++
		}
	}
	if items == nil {
		items = []captionCheck{}
	}
	v1api.OK(w, map[string]any{"items": items, "total": len(items), "flagged": flagged, "counts": counts, "trigger": trigger})
}

// CaptionFixOp is one bulk edit of POST /v1/datasets/{name}/captions/fix.
type CaptionFixOp struct {
	Op string `json:"op"` // ensure_trigger | fix_trigger_case | replace | fill_missing | strip
	// ensure_trigger: the text to put first when no trigger is there (default "[trigger]")
	Text string `json:"text"`
	// replace
	Find  string `json:"find"`
	With  string `json:"with"`
	Regex bool   `json:"regex"`
	// fill_missing
	Caption string `json:"caption"`
}

// fixCaptions is POST /v1/datasets/{name}/captions/fix {ops, trigger?,
// files?, apply?}: the edits to every caption (or only files), returned as
// before/after pairs; written only with apply: true.
func fixCaptions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	var b struct {
		Ops     []CaptionFixOp `json:"ops"`
		Trigger string         `json:"trigger"`
		Files   []string       `json:"files"`
		Apply   bool           `json:"apply"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	if len(b.Ops) == 0 {
		v1api.Fail(w, v1api.CodeInvalidRequest, "ops is required")
		return
	}
	trigger := strings.TrimSpace(b.Trigger)
	type fn func(string) string
	var steps []fn
	for _, op := range b.Ops {
		switch op.Op {
		case "ensure_trigger":
			text := op.Text
			if text == "" {
				text = "[trigger]"
			}
			steps = append(steps, func(c string) string {
				if c == "" || strings.Contains(c, "[trigger]") || strings.Contains(c, "[name]") || (trigger != "" && strings.Contains(c, trigger)) {
					return c
				}
				return text + " " + c
			})
		case "fix_trigger_case":
			var tr *regexp.Regexp
			if trigger != "" {
				tr = regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(trigger) + `\b`)
			}
			steps = append(steps, func(c string) string {
				c = bracketRe.ReplaceAllString(c, "[trigger]")
				if tr != nil {
					c = tr.ReplaceAllString(c, trigger)
				}
				return c
			})
		case "replace":
			if op.Find == "" {
				v1api.Fail(w, v1api.CodeInvalidRequest, "replace needs find")
				return
			}
			if op.Regex {
				re, err := regexp.Compile(op.Find)
				if err != nil {
					v1api.Fail(w, v1api.CodeInvalidRequest, "replace: "+err.Error())
					return
				}
				with := op.With
				steps = append(steps, func(c string) string { return re.ReplaceAllString(c, with) })
			} else {
				find, with := op.Find, op.With
				steps = append(steps, func(c string) string { return strings.ReplaceAll(c, find, with) })
			}
		case "fill_missing":
			if strings.TrimSpace(op.Caption) == "" {
				v1api.Fail(w, v1api.CodeInvalidRequest, "fill_missing needs caption")
				return
			}
			fill := op.Caption
			steps = append(steps, func(c string) string {
				if strings.TrimSpace(c) == "" {
					return fill
				}
				return c
			})
		case "strip":
			steps = append(steps, func(c string) string {
				return strings.Join(strings.Fields(preambleRe.ReplaceAllString(c, "")), " ")
			})
		default:
			v1api.Fail(w, v1api.CodeInvalidRequest, "unknown op "+op.Op+" (ensure_trigger, fix_trigger_case, replace, fill_missing, strip)")
			return
		}
	}
	items, err := datasetCaptions(dir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	only := map[string]bool{}
	for _, f := range b.Files {
		only[f] = true
	}
	type change struct {
		File   string  `json:"file"`
		Before *string `json:"before"`
		After  string  `json:"after"`
	}
	changes := []change{}
	for _, it := range items {
		if len(only) > 0 && !only[it.File] {
			continue
		}
		before := ""
		if it.Caption != nil {
			before = *it.Caption
		}
		after := before
		for _, s := range steps {
			after = s(after)
		}
		if after == before {
			continue
		}
		if b.Apply {
			full := filepath.Join(dir, filepath.FromSlash(it.File))
			if err := writeAtomic(captionPath(full), strings.NewReader(after)); err != nil {
				v1api.Internal(w, r, err)
				return
			}
		}
		changes = append(changes, change{File: it.File, Before: it.Caption, After: after})
	}
	if b.Apply && len(changes) > 0 {
		changed(ctx, name)
	}
	v1api.OK(w, map[string]any{"changes": changes, "applied": b.Apply})
}
