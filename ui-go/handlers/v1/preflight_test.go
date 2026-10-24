package v1

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func TestPreflight(t *testing.T) {
	e := setup(t)
	dir := filepath.Join(e.datasets, "faces")
	os.MkdirAll(dir, 0o755)
	for i, ext := range []string{".png", ".png", ".jpg", ".bmp", ".gif"} {
		os.WriteFile(filepath.Join(dir, "f"+string(rune('a'+i))+ext), pngBytes(2, 2), 0o644)
	}
	os.WriteFile(filepath.Join(dir, "fa.txt"), []byte("[trigger] smiling."), 0o644)
	os.WriteFile(filepath.Join(dir, "fb.txt"), []byte("Jihan by a window, wearing a"), 0o644)
	cfg := func(folder string) map[string]any {
		return map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
			"type": "diffusion_trainer", "trigger_word": "jihan",
			"train":    map[string]any{"steps": 3000},
			"save":     map[string]any{"save_every": 150, "max_step_saves_to_keep": 4},
			"model":    map[string]any{"name_or_path": "/nowhere/model"},
			"datasets": []any{map[string]any{"folder_path": folder}},
		}}}}
	}
	m := e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg(dir)}).json(t)
	var all []string
	for _, c := range m["checks"].([]any) {
		c := c.(map[string]any)
		all = append(all, c["level"].(string)+": "+c["message"].(string))
	}
	text := strings.Join(all, "\n")
	for _, want := range []string{
		"warn: datasets[0]: the trainer skips 1 .bmp, 1 .gif",
		"warn: datasets[0]: only 3 images",
		"warn: datasets[0]: 1 of 3 images have no caption",
		"warn: datasets[0]: 1 captions lack the trigger",
		"warn: datasets[0]: 1 captions end mid-sentence",
		"block: model.name_or_path /nowhere/model does not exist",
		"info: only the last 4 checkpoints stay on disk (from about step 2400 of 3000)",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in\n%s", want, text)
		}
	}
	if m["ok"] != false {
		t.Errorf("ok with a block: %v", m["ok"])
	}

	m = e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg(filepath.Join(e.datasets, "nope"))}).json(t)
	if !strings.Contains(m["checks"].([]any)[0].(map[string]any)["message"].(string)+" "+jsonOf(m), "does not exist on this server") {
		t.Errorf("missing dataset: %v", m)
	}
	want(t, e.call(t, "POST", "/v1/jobs/preflight", map[string]any{}), 400, "invalid_request")
	want(t, e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"job_id": "nope"}), 404, "not_found")
}

func jsonOf(v any) string {
	var b strings.Builder
	for _, c := range v.(map[string]any)["checks"].([]any) {
		b.WriteString(c.(map[string]any)["message"].(string) + "\n")
	}
	return b.String()
}

func TestPreflightPhotos(t *testing.T) {
	e := setup(t)
	faces := photocheck.FacesAvailable
	photocheck.FacesAvailable = true
	t.Cleanup(func() { photocheck.FacesAvailable = faces })
	noNewFFmpeg(t)

	// n distinct 800x600 photos, the last one a byte copy of the first,
	// plus a phone photo the trainer cannot read
	set := func(name string, n int) string {
		files := map[string][]byte{"phone.heic": []byte("\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1")}
		for i := 1; i < n; i++ {
			files[fmt.Sprintf("p%02d.jpg", i)] = photoScene(i)
		}
		files[fmt.Sprintf("p%02d.jpg", n)] = photoScene(1)
		return photoSet(t, e, name, files)
	}
	cfg := func(folder string, res []any) map[string]any {
		ds := map[string]any{"folder_path": folder}
		if res != nil {
			ds["resolution"] = res
		}
		return map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
			"type": "diffusion_trainer", "datasets": []any{ds},
		}}}}
	}
	type check struct{ level, message, fix string }
	run := func(folder string, res []any) []check {
		t.Helper()
		m := e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg(folder, res)}).json(t)
		var out []check
		for _, c := range m["checks"].([]any) {
			c := c.(map[string]any)
			fix, _ := c["fix"].(string)
			out = append(out, check{c["level"].(string), c["message"].(string), fix})
			if strings.HasPrefix(c["message"].(string), "datasets[0]") && c["level"] == LevelBlock {
				t.Errorf("a photo line blocks: %v", c)
			}
		}
		return out
	}
	find := func(cs []check, level, prefix string) *check {
		for i := range cs {
			if cs[i].level == level && strings.HasPrefix(cs[i].message, prefix) {
				return &cs[i]
			}
		}
		return nil
	}
	dump := func(cs []check) string {
		var b strings.Builder
		for _, c := range cs {
			fmt.Fprintf(&b, "%s: %s [%s]\n", c.level, c.message, c.fix)
		}
		return b.String()
	}

	few := set("few", 14)
	// before the analysis is done: what is missing is said, and no grade
	wait := preflightPhotoWait
	preflightPhotoWait = 0
	cs := run(few, nil)
	preflightPhotoWait = wait
	if c := find(cs, LevelInfo, "datasets[0]: "); c == nil || !strings.Contains(dump(cs), "not checked yet") {
		t.Errorf("no pending line in\n%s", dump(cs))
	}
	if find(cs, LevelInfo, "datasets[0]: grade") != nil {
		t.Errorf("a grade while photos are pending:\n%s", dump(cs))
	}

	// 14 photos at 1024: too few, all below the resolution, one copy
	cs = run(few, nil)
	text := dump(cs)
	for _, w := range []struct{ level, prefix string }{
		{LevelWarn, "datasets[0]: only 14 images"},
		{LevelWarn, "datasets[0]: the trainer skips 1 .heic"},
		{LevelWarn, "datasets[0]: 1 duplicate"},
		{LevelWarn, "datasets[0]: 14 photos train below 1024 px"},
		{LevelInfo, "datasets[0]: grade "},
		{LevelInfo, "datasets[0]: faces not checked"},
	} {
		if find(cs, w.level, w.prefix) == nil {
			t.Errorf("missing %s %q in\n%s", w.level, w.prefix, text)
		}
	}
	if c := find(cs, LevelWarn, "datasets[0]: 1 duplicate"); c != nil && !strings.Contains(c.fix, "datasets.fixPhotos set_aside_flagged flags [duplicate]") {
		t.Errorf("duplicate fix %q", c.fix)
	}
	// without ffmpeg 7.1 the server cannot convert HEIC, so the fix does not
	// send the user to convert
	if c := find(cs, LevelWarn, "datasets[0]: the trainer skips 1 .heic"); c != nil && (strings.Contains(c.fix, "fixPhotos convert") || !strings.Contains(c.fix, "export them as JPEG")) {
		t.Errorf("heic fix %q", c.fix)
	}
	if c := find(cs, LevelWarn, "datasets[0]: 14 photos train below"); c != nil && !strings.Contains(c.fix, "original files") {
		t.Errorf("low-res fix %q", c.fix)
	}
	if c := find(cs, LevelInfo, "datasets[0]: faces not checked"); c != nil && !strings.Contains(c.fix, "datasets.checkFaces") {
		t.Errorf("faces fix %q", c.fix)
	}
	if strings.Contains(text, "usable of") {
		t.Errorf("the usable line repeats the count warning:\n%s", text)
	}

	// 15 photos at 512: enough, none low-res, the copy leaves 14 usable
	cs = run(set("enough", 15), []any{512})
	text = dump(cs)
	for _, w := range []struct{ level, prefix string }{
		{LevelInfo, "datasets[0]: 15 images"},
		{LevelWarn, "datasets[0]: 1 duplicate"},
		{LevelWarn, "datasets[0]: only 14 usable of 15 photos"},
		{LevelInfo, "datasets[0]: grade "},
	} {
		if find(cs, w.level, w.prefix) == nil {
			t.Errorf("missing %s %q in\n%s", w.level, w.prefix, text)
		}
	}
	if strings.Contains(text, "only 15 images") || strings.Contains(text, "px (the trainer never") {
		t.Errorf("15 photos at 512 warned about count or resolution:\n%s", text)
	}
	if c := find(cs, LevelInfo, "datasets[0]: grade "); c != nil && !strings.Contains(c.message, "14 usable photos, before the face check") {
		t.Errorf("grade line %q", c.message)
	}
}

// noNewFFmpeg makes the server one whose ffmpeg cannot read HEIC, whatever
// is installed here.
func noNewFFmpeg(t *testing.T) {
	prev := convertFormats
	convertFormats = func() []string { return []string{"bmp", "gif", "tiff"} }
	t.Cleanup(func() { convertFormats = prev })
}

type pfCheck struct{ level, message, fix string }

// preflightOf runs preflight on one dataset entry.
func preflightOf(t *testing.T, e *env, ds map[string]any) []pfCheck {
	t.Helper()
	cfg := map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
		"type": "diffusion_trainer", "datasets": []any{ds},
	}}}}
	m := e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg}).json(t)
	var out []pfCheck
	for _, c := range m["checks"].([]any) {
		c := c.(map[string]any)
		fix, _ := c["fix"].(string)
		out = append(out, pfCheck{c["level"].(string), c["message"].(string), fix})
		// only the disk check may block here
		if c["level"] == LevelBlock && !strings.Contains(c["message"].(string), "free where checkpoints") {
			t.Errorf("blocks: %v", c)
		}
	}
	return out
}

func pfFind(cs []pfCheck, level, prefix string) *pfCheck {
	for i := range cs {
		if cs[i].level == level && strings.HasPrefix(cs[i].message, prefix) {
			return &cs[i]
		}
	}
	return nil
}

func pfDump(cs []pfCheck) string {
	var b strings.Builder
	for _, c := range cs {
		fmt.Fprintf(&b, "%s: %s [%s]\n", c.level, c.message, c.fix)
	}
	return b.String()
}

// toHalf is a float64 in [0.0001, 1) as binary16, enough for cosines.
func toHalf(f float64) uint16 {
	b := math.Float32bits(float32(f))
	return uint16(int(b>>23&0xff)-127+15)<<10 | uint16(b&0x7fffff>>13)
}

// seedFaces stores what a face check of dir would leave: one embedded
// frontal face per photo, and the identity matrix cos gives.
func seedFaces(t *testing.T, dir string, files []string, cos func(i, j int) float64) {
	t.Helper()
	ctx := context.Background()
	rows := map[string][]byte{}
	var shas, faces []string
	for _, f := range files {
		b, err := os.ReadFile(filepath.Join(dir, f))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(b)
		sha := hex.EncodeToString(sum[:])
		fd, _ := json.Marshal(photocheck.FaceData{Algo: photocheck.FaceAlgo, FrameSharp: 300, Faces: []photocheck.Face{{
			Box: [4]float64{0.4, 0.3, 0.2, 0.3}, Kps: [5][2]float64{{0.45, 0.40}, {0.55, 0.40}, {0.50, 0.45}, {0.46, 0.50}, {0.54, 0.50}},
			Score: 0.9, Side: 240, AlignSide: 240, Sharp: 100, MeanY: 120, ClipHi: 0.01, Embedded: true,
		}}})
		rows[sha] = fd
		shas = append(shas, sha)
		faces = append(faces, sha+":0")
	}
	if err := state.PutPhotoFaces(ctx, rows, photocheck.FaceAlgo); err != nil {
		t.Fatal(err)
	}
	var tri []byte
	for i := range faces {
		for j := i + 1; j < len(faces); j++ {
			tri = binary.LittleEndian.AppendUint16(tri, toHalf(cos(i, j)))
		}
	}
	id, _ := json.Marshal(photocheck.Identity{FaceMatrix: photocheck.FaceMatrix{Faces: faces, Cos: base64.StdEncoding.EncodeToString(tri)},
		SetSig: photocheck.SetSig(shas), Model: photocheck.FaceModel})
	if err := state.PutPhotoSet(ctx, &state.PhotoSet{Dir: dir, Identity: id}); err != nil {
		t.Fatal(err)
	}
}

func TestPreflightPhotoIdentity(t *testing.T) {
	e := setup(t)
	faces := photocheck.FacesAvailable
	photocheck.FacesAvailable = true
	t.Cleanup(func() { photocheck.FacesAvailable = faces })

	// 20 face-checked photos whose matrix cos gives
	faceSet := func(name string, cos func(i, j int) float64) string {
		files := map[string][]byte{}
		var names []string
		for i := range 20 {
			f := fmt.Sprintf("p%02d.jpg", i)
			files[f] = photoScene(i + 1)
			names = append(names, f)
		}
		dir := photoSet(t, e, name, files)
		settledReview(t, name, "")
		seedFaces(t, dir, names, cos)
		return dir
	}
	// the last photo is someone else
	stranger := faceSet("stranger", func(i, j int) float64 {
		if i == 19 || j == 19 {
			return 0.1
		}
		return 0.6
	})
	cs := preflightOf(t, e, map[string]any{"folder_path": stranger})
	if c := pfFind(cs, LevelWarn, "datasets[0]: 1 photo may show someone else"); c == nil {
		t.Errorf("no not_them line in\n%s", pfDump(cs))
	} else if !strings.Contains(c.fix, "datasets.checkPhotos") || !strings.Contains(c.fix, "datasets.photoSettings") {
		t.Errorf("not_them fix %q", c.fix)
	}

	// 12 photos of one person, 8 of another
	two := faceSet("two", func(i, j int) float64 {
		if (i < 12) == (j < 12) {
			return 0.6
		}
		return 0.1
	})
	cs = preflightOf(t, e, map[string]any{"folder_path": two})
	if c := pfFind(cs, LevelWarn, "datasets[0]: these photos seem to show two different people"); c == nil {
		t.Errorf("no two_people line in\n%s", pfDump(cs))
	} else if !strings.Contains(c.fix, "datasets.photoSettings reference") {
		t.Errorf("two_people fix %q", c.fix)
	}

	// a regularization set shows other people on purpose: no photo lines
	cs = preflightOf(t, e, map[string]any{"folder_path": two, "is_reg": true})
	if pfFind(cs, LevelWarn, "datasets[0]: these photos") != nil || pfFind(cs, LevelInfo, "datasets[0]: grade") != nil {
		t.Errorf("photo lines on a reg set:\n%s", pfDump(cs))
	}
	if pfFind(cs, LevelInfo, "datasets[0]: 20 images") == nil {
		t.Errorf("a reg set lost its count:\n%s", pfDump(cs))
	}

	// a review that fails is said, and blocks nothing
	prev := reviewDataset
	reviewDataset = func(context.Context, string, photocheck.Options) (*photocheck.Review, error) {
		return nil, errors.New("photo state unreadable")
	}
	t.Cleanup(func() { reviewDataset = prev })
	cs = preflightOf(t, e, map[string]any{"folder_path": two})
	if pfFind(cs, LevelInfo, "datasets[0]: photos not checked (photo state unreadable)") == nil {
		t.Errorf("no error line in\n%s", pfDump(cs))
	}
	if pfFind(cs, LevelWarn, "datasets[0]: these photos") != nil {
		t.Errorf("photo lines without a review:\n%s", pfDump(cs))
	}
}

func TestSkippedFix(t *testing.T) {
	noNewFFmpeg(t)
	for _, c := range []struct {
		skipped map[string]int
		want    string
	}{
		{map[string]int{".bmp": 1, ".tif": 2}, "convert them to JPEG (Images step, or datasets.fixPhotos convert)"},
		{map[string]int{".jxl": 1, ".heif": 1}, "export them as JPEG on the phone or computer, or add them in the studio in Safari"},
		{map[string]int{".gif": 1, ".heic": 3, ".jxl": 1}, "convert the .gif to JPEG (Images step, or datasets.fixPhotos convert); " +
			"export the .heic, .jxl as JPEG on the phone or computer, or add them in the studio in Safari"},
	} {
		if got := skippedFix(c.skipped); got != c.want {
			t.Errorf("skippedFix(%v) = %q, want %q", c.skipped, got, c.want)
		}
	}
	convertFormats = func() []string { return []string{"bmp", "gif", "tiff", "heic", "heif", "avif"} }
	if got := skippedFix(map[string]int{".heif": 1, ".avif": 1}); got != "convert them to JPEG (Images step, or datasets.fixPhotos convert)" {
		t.Errorf("with ffmpeg 7.1: %q", got)
	}
}

func TestQualityWarns(t *testing.T) {
	faced := func(n, noFace int) *photocheck.Review {
		rv := &photocheck.Review{Kind: photocheck.KindPerson, Counts: map[photocheck.Flag]int{photocheck.FlagNoFace: noFace, photocheck.FlagBars: 1}}
		for range n {
			rv.Items = append(rv.Items, photocheck.Item{Face: &photocheck.ItemFace{}})
		}
		return rv
	}
	for _, c := range []struct {
		rv   *photocheck.Review
		flag photocheck.Flag
		want bool
	}{
		{faced(20, 2), photocheck.FlagNoFace, false}, // 2 of 20: info
		{faced(20, 3), photocheck.FlagNoFace, true},
		{faced(10, 2), photocheck.FlagNoFace, true}, // 20 %
		{faced(20, 0), photocheck.FlagBars, true},
		{&photocheck.Review{Kind: photocheck.KindObject}, photocheck.FlagBars, false},
		{faced(20, 0), photocheck.FlagBlurry, true},
	} {
		if got := qualityWarns(c.rv, c.flag); got != c.want {
			t.Errorf("qualityWarns(%d faced, %v) = %v, want %v", len(c.rv.Items), c.flag, got, c.want)
		}
	}
}

func TestPreflightPhotoFormats(t *testing.T) {
	e := setup(t)
	dir := photoSet(t, e, "formats", map[string][]byte{
		"grey.png":  testutil.Gray16PNG(testutil.Scene(1, 640, 640)),
		"clear.png": testutil.TransparentPNG(testutil.Scene(2, 640, 640)),
		"a.jpg":     photoScene(3),
		"a.png":     testutil.PNG(testutil.Scene(4, 640, 640)),
	})
	cfg := map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
		"type": "diffusion_trainer", "datasets": []any{map[string]any{"folder_path": dir}},
	}}}}
	m := e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg}).json(t)
	var line map[string]any
	for _, c := range m["checks"].([]any) {
		if c := c.(map[string]any); strings.Contains(c["message"].(string), "the trainer reads wrongly") {
			line = c
		}
	}
	if line == nil {
		t.Fatalf("no format line in %v", m["checks"])
	}
	msg, fix := line["message"].(string), line["fix"].(string)
	if line["level"] != LevelWarn || !strings.HasPrefix(msg, "datasets[0]: 4 files the trainer reads wrongly (16-bit grey, transparent, sharing a caption)") {
		t.Errorf("format line %v", line)
	}
	if !strings.Contains(fix, "datasets.fixPhotos convert") || !strings.Contains(fix, "rename_same_stem") {
		t.Errorf("format fix %q", fix)
	}
}

func TestLowerFirst(t *testing.T) {
	for in, want := range map[string]string{"Add 3 more photos": "add 3 more photos", "JPEG only": "JPEG only", "a": "a", "": "", "É": "é"} {
		if got := lowerFirst(in); got != want {
			t.Errorf("lowerFirst(%q) = %q, want %q", in, got, want)
		}
	}
}
