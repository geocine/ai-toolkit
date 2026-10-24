package v1

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/fs"
	"mime/multipart"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
	"github.com/geocine/aitoolkit/internal/thumbs"
)

// uploadPart is one files[] part of a multipart upload, in stream order.
type uploadPart struct {
	name string
	data []byte
}

func uploadParts(t *testing.T, e *env, dataset, query string, parts ...uploadPart) resp {
	t.Helper()
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	for _, p := range parts {
		fw, err := mw.CreateFormFile("files", p.name)
		if err != nil {
			t.Fatal(err)
		}
		fw.Write(p.data)
	}
	mw.Close()
	target := "/v1/datasets/" + dataset + "/files"
	if query != "" {
		target += "?" + query
	}
	return e.do(t, "POST", target, &b, map[string]string{"Content-Type": mw.FormDataContentType()})
}

func uploadResult(t *testing.T, r resp) photocheck.UploadResult {
	t.Helper()
	if r.code != 200 {
		t.Fatalf("status %d: %s", r.code, r.body)
	}
	var res photocheck.UploadResult
	if err := json.Unmarshal(r.body, &res); err != nil || res.Items == nil || res.Renamed == nil || res.Skipped == nil {
		t.Fatalf("not an upload result (%v): %s", err, r.body)
	}
	return res
}

func uploadScene(seed int) []byte { return testutil.JPEG(testutil.Scene(seed, 640, 480), 90) }

// cacheFiles plants a photo's derived files: a thumb and one file in each of
// the trainer's caches, plus a cache file of a longer stem that must stay.
func cacheFiles(t *testing.T, dir, file string) (own, other []string) {
	t.Helper()
	stem := strings.TrimSuffix(file, filepath.Ext(file))
	const id = "AbCdEfGhIjKlMnOpQrStUv" // 22 chars, like the trainer's md5 tag
	base := thumbs.Base(filepath.Join(dir, file))
	own = append(own, base+".jpg")
	for _, c := range []string{"_latent_cache", "_t_e_cache", "_clip_vision_cache"} {
		own = append(own, filepath.Join(dir, c, stem+"_"+id+".safetensors"))
	}
	other = append(other, filepath.Join(dir, "_latent_cache", stem+"_2_"+id+".safetensors"))
	for _, p := range append(slices.Clone(own), other...) {
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return own, other
}

func readFile(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestUploadConflict(t *testing.T) {
	e := setup(t)
	want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": "set"}), 201, "")
	dir := filepath.Join(e.datasets, "set")
	one, two, three := uploadScene(1), uploadScene(2), uploadScene(3)

	res := uploadResult(t, uploadParts(t, e, "set", "", uploadPart{"a.jpg", one}, uploadPart{"a.txt", []byte("first")}))
	if !slices.Equal(res.Items, []string{"a.jpg", "a.txt"}) || len(res.Renamed) != 0 || len(res.Skipped) != 0 {
		t.Fatalf("first upload %+v", res)
	}

	// replace (the default, and by name) overwrites and drops the old photo's
	// thumb and caches; so does a PUT over the file
	for i, query := range []string{"", "on_conflict=replace"} {
		own, other := cacheFiles(t, dir, "a.jpg")
		content := uploadScene(10 + i)
		res = uploadResult(t, uploadParts(t, e, "set", query, uploadPart{"a.jpg", content}))
		if !slices.Equal(res.Items, []string{"a.jpg"}) || !bytes.Equal(readFile(t, filepath.Join(dir, "a.jpg")), content) {
			t.Fatalf("replace (query %q) %+v", query, res)
		}
		checkPurged(t, "replace (query "+strconv.Quote(query)+")", own, other)
	}
	own, other := cacheFiles(t, dir, "a.jpg")
	want(t, e.do(t, "PUT", "/v1/datasets/set/files/a.jpg", bytes.NewReader(two), nil), 200, "")
	if !bytes.Equal(readFile(t, filepath.Join(dir, "a.jpg")), two) {
		t.Fatal("PUT did not replace a.jpg")
	}
	checkPurged(t, "PUT", own, other)

	// rename: other content gets _2, and its caption follows it, after the image
	res = uploadResult(t, uploadParts(t, e, "set", "on_conflict=rename",
		uploadPart{"a.jpg", three}, uploadPart{"a.txt", []byte("third")}))
	if !slices.Equal(res.Items, []string{"a_2.jpg", "a_2.txt"}) ||
		!slices.Equal(res.Renamed, []photocheck.Renamed{{From: "a.jpg", To: "a_2.jpg"}, {From: "a.txt", To: "a_2.txt"}}) {
		t.Fatalf("rename %+v", res)
	}
	if !bytes.Equal(readFile(t, filepath.Join(dir, "a.jpg")), two) || !bytes.Equal(readFile(t, filepath.Join(dir, "a_2.jpg")), three) {
		t.Fatal("rename touched a.jpg or wrote a_2.jpg wrong")
	}
	if string(readFile(t, filepath.Join(dir, "a.txt"))) != "first" || string(readFile(t, filepath.Join(dir, "a_2.txt"))) != "third" {
		t.Fatal("the caption did not follow its image")
	}

	// ... and before the image in the stream; the next free name is _3
	four := uploadScene(4)
	res = uploadResult(t, uploadParts(t, e, "set", "on_conflict=rename",
		uploadPart{"a.txt", []byte("fourth")}, uploadPart{"a.jpg", four}))
	if !slices.Equal(res.Items, []string{"a_3.jpg", "a_3.txt"}) || len(res.Renamed) != 2 {
		t.Fatalf("rename, caption first %+v", res)
	}
	if string(readFile(t, filepath.Join(dir, "a_3.txt"))) != "fourth" || string(readFile(t, filepath.Join(dir, "a.txt"))) != "first" {
		t.Fatal("a caption sent before its image did not follow it")
	}

	// the same content is skipped, wherever it is stored; a caption alone replaces
	res = uploadResult(t, uploadParts(t, e, "set", "on_conflict=rename",
		uploadPart{"a.jpg", three}, uploadPart{"a.txt", []byte("third again")}, uploadPart{"b.txt", []byte("lone")}))
	if !slices.Equal(res.Skipped, []string{"a.jpg"}) || !slices.Equal(res.Items, []string{"a_2.txt", "b.txt"}) {
		t.Fatalf("same content %+v", res)
	}
	if string(readFile(t, filepath.Join(dir, "a_2.txt"))) != "third again" {
		t.Fatal("the caption of a skipped image did not follow the stored copy")
	}
	if _, err := os.Stat(filepath.Join(dir, "a_4.jpg")); err == nil {
		t.Fatal("same content was saved again")
	}

	// skip never overwrites
	res = uploadResult(t, uploadParts(t, e, "set", "on_conflict=skip",
		uploadPart{"a.jpg", four}, uploadPart{"a.txt", []byte("no")}, uploadPart{"c.jpg", four}))
	if !slices.Equal(res.Skipped, []string{"a.jpg", "a.txt"}) || !slices.Equal(res.Items, []string{"c.jpg"}) {
		t.Fatalf("skip %+v", res)
	}
	if !bytes.Equal(readFile(t, filepath.Join(dir, "a.jpg")), two) || string(readFile(t, filepath.Join(dir, "a.txt"))) != "first" {
		t.Fatal("skip overwrote a file")
	}
	// all skipped is still a 200 with the old shape
	res = uploadResult(t, uploadParts(t, e, "set", "on_conflict=skip", uploadPart{"c.jpg", one}))
	if len(res.Items) != 0 || !slices.Equal(res.Skipped, []string{"c.jpg"}) {
		t.Fatalf("all skipped %+v", res)
	}

	want(t, uploadParts(t, e, "set", "on_conflict=merge", uploadPart{"d.jpg", one}), 400, "invalid_request")
	want(t, uploadParts(t, e, "set", "on_conflict=rename",
		uploadPart{"d.txt", bytes.Repeat([]byte("x"), maxUploadCaption+1)}), 413, "too_large")
	want(t, uploadParts(t, e, "set", "on_conflict=rename"), 400, "invalid_request")

	// deleting a photo drops its thumb and caches as well
	own, _ = cacheFiles(t, dir, "c.jpg")
	want(t, e.call(t, "DELETE", "/v1/datasets/set/files/c.jpg", nil), 204, "")
	for _, p := range own {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("delete left %s", p)
		}
	}

	// rename: a stem another photo or a caption uses is taken, whatever the
	// extension, so an upload never takes or shares another photo's caption
	for i, tc := range []struct {
		name      string
		have      map[string][]byte
		up        []uploadPart
		items     []string
		unchanged []string
	}{
		{"a_2 used by a png", map[string][]byte{"a.jpg": one, "a_2.png": pngBytes(2, 2), "a_2.txt": []byte("png")},
			[]uploadPart{{"a.jpg", two}, {"a.txt", []byte("new")}}, []string{"a_3.jpg", "a_3.txt"}, []string{"a_2.txt"}},
		{"a used by a png", map[string][]byte{"a.png": pngBytes(2, 2), "a.txt": []byte("png")},
			[]uploadPart{{"a.jpg", two}, {"a.txt", []byte("new")}}, []string{"a_2.jpg", "a_2.txt"}, []string{"a.txt"}},
		{"a used by an orphan caption", map[string][]byte{"a.txt": []byte("orphan")},
			[]uploadPart{{"a.jpg", two}}, []string{"a_2.jpg"}, []string{"a.txt"}},
		{"two photos of one stem in a request", map[string][]byte{},
			[]uploadPart{{"a.jpg", one}, {"a.png", pngBytes(2, 2)}}, []string{"a.jpg", "a_2.png"}, nil},
	} {
		ds := "stems" + strconv.Itoa(i)
		want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": ds}), 201, "")
		sdir := filepath.Join(e.datasets, ds)
		if err := testutil.WriteTree(sdir, tc.have); err != nil {
			t.Fatal(err)
		}
		res := uploadResult(t, uploadParts(t, e, ds, "on_conflict=rename", tc.up...))
		if !slices.Equal(res.Items, tc.items) {
			t.Errorf("%s: items %v, want %v", tc.name, res.Items, tc.items)
		}
		for _, f := range tc.unchanged {
			if !bytes.Equal(readFile(t, filepath.Join(sdir, f)), tc.have[f]) {
				t.Errorf("%s: %s was overwritten", tc.name, f)
			}
		}
		if c := tc.items[len(tc.items)-1]; strings.HasSuffix(c, ".txt") && string(readFile(t, filepath.Join(sdir, c))) != "new" {
			t.Errorf("%s: the caption did not follow its image to %s", tc.name, c)
		}
	}
}

// checkPurged fails the test when a file of own is left or one of other is
// gone.
func checkPurged(t *testing.T, what string, own, other []string) {
	t.Helper()
	for _, p := range own {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("%s left %s", what, p)
		}
	}
	for _, p := range other {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s removed another stem's cache %s", what, p)
		}
	}
}

// TestUploadClaimName shows the rename upload never replaces a file written after
// it looked: the claim fails with fs.ErrExist and leaves the file alone.
func TestUploadClaimName(t *testing.T) {
	dir := t.TempDir()
	tmp, dst := filepath.Join(dir, ".upload-1"), filepath.Join(dir, "a.jpg")
	os.WriteFile(tmp, []byte("new"), 0o644)
	os.WriteFile(dst, []byte("old"), 0o644)
	if err := claimName(tmp, dst); !errors.Is(err, fs.ErrExist) {
		t.Fatalf("claim over a file: %v, want fs.ErrExist", err)
	}
	if string(readFile(t, dst)) != "old" {
		t.Fatal("the claim replaced the file")
	}
	free := filepath.Join(dir, "a_2.jpg")
	if err := claimName(tmp, free); err != nil || string(readFile(t, free)) != "new" {
		t.Fatalf("claim of a free name: %v", err)
	}
}

func TestWalkControls(t *testing.T) {
	e := setup(t)
	dir := filepath.Join(e.datasets, "set")
	if err := testutil.WriteTree(dir, map[string][]byte{
		"a.jpg":                   uploadScene(1),
		"a.txt":                   []byte("[trigger] by a window."),
		"_controls/a.depth.jpg":   uploadScene(2),
		"_controls/a.depth.txt":   []byte("depth"),
		"_controls/b.bmp":         pngBytes(2, 2),
		"sub/_controls/c.jpg":     uploadScene(3),
		"sub/_controls_not/d.jpg": uploadScene(4),
	}); err != nil {
		t.Fatal(err)
	}

	m := e.call(t, "GET", "/v1/datasets/set", nil).json(t)
	var files []string
	for _, it := range m["items"].([]any) {
		files = append(files, it.(map[string]any)["file"].(string))
	}
	if m["images"] != float64(2) || m["captions"] != float64(1) || !slices.Equal(files, []string{"a.jpg", "sub/_controls_not/d.jpg"}) {
		t.Fatalf("dataset counts _controls: images %v captions %v items %v", m["images"], m["captions"], files)
	}
	if v, ok := m["photo_job"]; !ok || v != nil {
		t.Errorf("photo_job = %v (present %v), want null", v, ok)
	}

	m = e.call(t, "GET", "/v1/datasets/set/captions/check", nil).json(t)
	if m["total"] != float64(2) {
		t.Errorf("captions/check counts _controls: %v", m["items"])
	}

	cfg := map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
		"type": "diffusion_trainer", "datasets": []any{map[string]any{"folder_path": dir}}}}}}
	m = e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg}).json(t)
	var lines []string
	for _, c := range m["checks"].([]any) {
		lines = append(lines, c.(map[string]any)["message"].(string))
	}
	text := strings.Join(lines, "\n")
	if !strings.Contains(text, "datasets[0]: only 2 images") || strings.Contains(text, "trainer skips") {
		t.Errorf("preflight counts _controls:\n%s", text)
	}
}

func TestDatasetPhotoJobAndOrientation(t *testing.T) {
	e := setup(t)
	dir := filepath.Join(e.datasets, "set")
	sideways := testutil.WithExif(testutil.JPEG(testutil.Scene(1, 64, 48), 90), 6, "2026:05:01 14:03:22", "Apple", "iPhone 15")
	if err := testutil.WriteTree(dir, map[string][]byte{"a.jpg": sideways, "b.jpg": testutil.JPEG(testutil.Scene(2, 64, 48), 90)}); err != nil {
		t.Fatal(err)
	}
	typ := photocheck.FaceJobType
	j, err := store.CreateJob(t.Context(), store.NewJob{Name: "set_faces", GpuIDs: "cpu", JobConfig: "{}", JobType: &typ, JobRef: &dir})
	if err != nil {
		t.Fatal(err)
	}

	m := e.call(t, "GET", "/v1/datasets/set", nil).json(t)
	size := map[string][2]any{}
	for _, it := range m["items"].([]any) {
		it := it.(map[string]any)
		size[it["file"].(string)] = [2]any{it["width"], it["height"]}
	}
	if size["a.jpg"] != [2]any{float64(48), float64(64)} || size["b.jpg"] != [2]any{float64(64), float64(48)} {
		t.Errorf("sizes after EXIF orientation %v", size)
	}
	pj, _ := m["photo_job"].(map[string]any)
	if pj == nil || pj["id"] != j.ID || pj["type"] != photocheck.FaceJobType {
		t.Errorf("photo_job %v", m["photo_job"])
	}

	// a running face check holds the dataset, like a caption job
	store.UpdateJob(t.Context(), j.ID, store.Fields{"status": "running"})
	want(t, e.call(t, "DELETE", "/v1/datasets/set", nil), 409, "conflict")
	store.UpdateJob(t.Context(), j.ID, store.Fields{"status": "stopped"})

	// deleting the dataset forgets its photo analysis
	style := photocheck.Kind("style")
	if _, err := photocheck.SaveSettings(t.Context(), dir, photocheck.SettingsPatch{Kind: &style}); err != nil {
		t.Fatal(err)
	}
	if set, err := state.GetPhotoSet(t.Context(), dir); err != nil || set == nil {
		t.Fatalf("no photo settings to forget (%v)", err)
	}
	want(t, e.call(t, "DELETE", "/v1/datasets/set", nil), 204, "")
	if _, err := os.Stat(dir); err == nil {
		t.Fatal("the dataset folder is still there")
	}
	if set, err := state.GetPhotoSet(t.Context(), dir); err != nil || set != nil {
		t.Errorf("the photo settings outlived the dataset: %+v (%v)", set, err)
	}
}
