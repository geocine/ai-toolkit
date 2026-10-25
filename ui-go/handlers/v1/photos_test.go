package v1

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
)

// The photo routes are registered by photos_routes.go, so these tests call
// the handlers directly, with the path value set by hand.

func photoCall(t *testing.T, h http.HandlerFunc, method, target, name string, body any) resp {
	t.Helper()
	var rd io.Reader
	switch b := body.(type) {
	case nil:
	case []byte:
		rd = bytes.NewReader(b)
	default:
		data, _ := json.Marshal(b)
		rd = bytes.NewReader(data)
	}
	req := httptest.NewRequest(method, target, rd)
	req.Header.Set("Content-Type", "application/json")
	if name != "" {
		req.SetPathValue("name", name)
	}
	rec := httptest.NewRecorder()
	h(rec, req)
	return resp{rec.Code, rec.Body.Bytes(), rec.Header()}
}

func photoReview(t *testing.T, r resp) *photocheck.Review {
	t.Helper()
	if r.code != http.StatusOK {
		t.Fatalf("status %d: %s", r.code, r.body)
	}
	var rv photocheck.Review
	if err := json.Unmarshal(r.body, &rv); err != nil {
		t.Fatalf("not a review: %s", r.body)
	}
	return &rv
}

// photoSet makes the dataset folder name with files in it.
func photoSet(t *testing.T, e *env, name string, files map[string][]byte) string {
	t.Helper()
	dir := filepath.Join(e.datasets, name)
	if err := testutil.WriteTree(dir, files); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { drainPhotos(t, dir) })
	return dir
}

// settledReview is the dataset's review once every photo is analysed.
func settledReview(t *testing.T, name, query string) *photocheck.Review {
	t.Helper()
	target := "/v1/datasets/" + name + "/photos/check?wait_ms=20000"
	if query != "" {
		target += "&" + query
	}
	rv := photoReview(t, photoCall(t, checkPhotos, "GET", target, name, nil))
	if rv.Pending != 0 {
		t.Fatalf("still %d pending", rv.Pending)
	}
	return rv
}

// drainPhotos waits (a few seconds at most) until the set-aside photos of
// dir, which a review only queues for warm-up, are analysed too, so no
// background analysis outlives the test's databases.
func drainPhotos(t *testing.T, dir string) {
	t.Helper()
	ctx := context.Background() // t.Context() is done once cleanup runs
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		done := true
		filepath.WalkDir(filepath.Join(dir, photocheck.SetAsideDir), func(p string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() || !photocheck.TrainerExts[strings.ToLower(filepath.Ext(p))] {
				return nil
			}
			info, err := d.Info()
			if err != nil {
				return nil
			}
			sha, ok, _ := state.LookupFileSHA256(ctx, p, info)
			if ok {
				rows, _ := state.GetPhotoMetrics(ctx, []string{sha}, photocheck.MetricsCacheVersion)
				ok = rows[sha] != nil
			}
			done = done && ok
			return nil
		})
		if done {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func photoScene(seed int) []byte { return testutil.JPEG(testutil.Scene(seed, 800, 600), 90) }

func reviewItem(rv *photocheck.Review, file string) *photocheck.Item {
	for i := range rv.Items {
		if rv.Items[i].File == file {
			return &rv.Items[i]
		}
	}
	return nil
}

func reviewFiles(rv *photocheck.Review) []string {
	out := []string{}
	for _, it := range rv.Items {
		out = append(out, it.File)
	}
	return out
}

func TestPhotoCheck(t *testing.T) {
	e := setup(t)
	dir := photoSet(t, e, "set", map[string][]byte{
		"b.jpg":              photoScene(1),
		"b.png":              testutil.PNG(testutil.Scene(2, 640, 640)),
		"c.heic":             []byte("\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"),
		"sub/d.jpg":          photoScene(3),
		"_controls/x.jpg":    photoScene(4),
		".set_aside/old.jpg": photoScene(5),
		"b.txt":              []byte("a caption"),
	})
	rv := photoReview(t, photoCall(t, checkPhotos, "GET", "/v1/datasets/set/photos/check?wait_ms=0", "set", nil))
	if want := []string{"b.jpg", "b.png", "c.heic", "sub/d.jpg"}; !slices.Equal(reviewFiles(rv), want) {
		t.Fatalf("items %v, want %v", reviewFiles(rv), want)
	}
	if rv.Pending == 0 {
		t.Error("wait_ms=0 should leave photos pending")
	}
	// a wait past the cap, even past the int range, waits the cap (the
	// first call settles the photos left pending above)
	for _, q := range []string{"wait_ms=10000000000000", "wait_ms=99999999999999999999999"} {
		if rv := photoReview(t, photoCall(t, checkPhotos, "GET", "/v1/datasets/set/photos/check?"+q, "set", nil)); rv.Pending != 0 {
			t.Errorf("%s: %d pending", q, rv.Pending)
		}
	}
	// the analysis can finish before a wait that came out as none, so the
	// cap itself is checked here: 1e13 ms overflows a Duration to negative
	for s, want := range map[string]time.Duration{"0": 0, "1500": 1500 * time.Millisecond, "20001": maxPhotoWait,
		"10000000000000": maxPhotoWait, "99999999999999999999999": maxPhotoWait} {
		if got, ok := photoWait(s); !ok || got != want {
			t.Errorf("wait_ms=%s: %v %v, want %v", s, got, ok, want)
		}
	}
	for _, s := range []string{"-5", "-99999999999999999999999", "soon", "1.5"} {
		if _, ok := photoWait(s); ok {
			t.Errorf("wait_ms=%s accepted", s)
		}
	}

	rv = settledReview(t, "set", "")
	if it := reviewItem(rv, "c.heic"); it == nil || !slices.Contains(it.Flags, photocheck.FlagUnsupported) {
		t.Errorf("c.heic: %+v", it)
	}
	for _, f := range []string{"b.jpg", "b.png"} {
		if it := reviewItem(rv, f); !slices.Contains(it.Flags, photocheck.FlagSameStem) {
			t.Errorf("%s flags %v, want same_stem", f, it.Flags)
		}
	}
	for _, f := range reviewFiles(rv) {
		if strings.Contains(f, "_controls") || strings.Contains(f, photocheck.SetAsideDir) {
			t.Errorf("%s is listed as a dataset photo", f)
		}
	}
	if len(rv.SetAside) != 1 || rv.SetAside[0].File != ".set_aside/old.jpg" {
		t.Errorf("set_aside %+v", rv.SetAside)
	}
	if rv.Resolution != photocheck.DefaultR || rv.Kind != photocheck.KindPerson || rv.Faces.Job != nil {
		t.Errorf("resolution %d kind %s faces %+v", rv.Resolution, rv.Kind, rv.Faces)
	}

	// query parameters
	rv = settledReview(t, "set", "resolution=512,768&kind=style&pick=2")
	if rv.Resolution != 768 || rv.Kind != photocheck.KindStyle || rv.Pick == nil {
		t.Errorf("resolution %d kind %s pick %+v", rv.Resolution, rv.Kind, rv.Pick)
	}
	for _, q := range []string{"resolution=huge", "resolution=1024,x", "kind=pet", "pick=-1", "pick=many", "wait_ms=soon", "wait_ms=-5"} {
		want(t, photoCall(t, checkPhotos, "GET", "/v1/datasets/set/photos/check?"+q, "set", nil), 400, "invalid_request")
	}
	want(t, photoCall(t, checkPhotos, "GET", "/v1/datasets/nope/photos/check", "nope", nil), 404, "not_found")
	want(t, photoCall(t, checkPhotos, "GET", "/v1/datasets/x/photos/check", "../x", nil), 400, "invalid_request")

	// the dataset's face check comes back as a /v1 job
	typ := photocheck.FaceJobType
	j, err := store.CreateJob(t.Context(), store.NewJob{Name: "set_faces", GpuIDs: "cpu", JobConfig: "{}", JobType: &typ, JobRef: &dir})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.UpdateJob(t.Context(), j.ID, store.Fields{"status": "queued"}); err != nil {
		t.Fatal(err)
	}
	r := photoCall(t, checkPhotos, "GET", "/v1/datasets/set/photos/check?wait_ms=20000", "set", nil)
	rv = photoReview(t, r)
	if rv.Faces.State != "queued" || rv.Faces.JobID != j.ID {
		t.Fatalf("faces %+v", rv.Faces)
	}
	job, _ := r.json(t)["faces"].(map[string]any)["job"].(map[string]any)
	if job == nil || job["id"] != j.ID || job["type"] != photocheck.FaceJobType {
		t.Errorf("faces.job %v", job)
	}
}

func TestPhotoAssess(t *testing.T) {
	e := setup(t)
	dir := photoSet(t, e, "set", map[string][]byte{"stored.jpg": photoScene(31), "other.jpg": photoScene(32)})
	stored := settledReview(t, "set", "")
	measure := func(rel string, data []byte) photocheck.Metrics {
		t.Helper()
		p := filepath.Join(t.TempDir(), filepath.Base(rel))
		if err := os.WriteFile(p, data, 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := photocheck.Measure(p, rel)
		if err != nil {
			t.Fatal(err)
		}
		m.Source = "browser"
		return m
	}

	// metrics alone give the flags the stored check gives, with the
	// browser's dHash a few bits off the server's
	m := measure("other.jpg", photoScene(32))
	h, err := photocheck.ParseHash(m.DHash)
	if err != nil {
		t.Fatal(err)
	}
	m.DHash = photocheck.HashHex(h ^ 0b101)
	rv := photoReview(t, photoCall(t, assessPhotos, "POST", "/v1/photos/assess", "", map[string]any{"items": []any{m}}))
	if got, want := reviewItem(rv, "other.jpg").Flags, reviewItem(stored, "other.jpg").Flags; !slices.Equal(got, want) {
		t.Errorf("assess flags %v, check flags %v", got, want)
	}

	// with dataset, the stored photos join the duplicate pass
	dup := measure("new.jpg", photoScene(31))
	rv = photoReview(t, photoCall(t, assessPhotos, "POST", "/v1/photos/assess", "",
		map[string]any{"items": []any{dup}, "dataset": "set", "resolution": []int{512, 768}}))
	it := reviewItem(rv, "new.jpg")
	if it == nil || !slices.Contains(it.Flags, photocheck.FlagDuplicate) || it.SameAs != "stored.jpg" {
		t.Fatalf("new.jpg %+v", it)
	}
	if reviewItem(rv, "stored.jpg") != nil || rv.Resolution != 768 || rv.Total != 1 {
		t.Fatalf("stored photos are not items: %v, resolution %d", reviewFiles(rv), rv.Resolution)
	}

	// no items: the formats the server converts
	r := photoCall(t, assessPhotos, "POST", "/v1/photos/assess", "", map[string]any{"items": []any{}})
	rv = photoReview(t, r)
	if !slices.Contains(rv.Convert, "bmp") || rv.Items == nil || rv.Total != 0 {
		t.Fatalf("empty assess: %s", r.body)
	}

	// limits and bad bodies
	many := make([]map[string]any, photocheck.MaxAssessItems+1)
	for i := range many {
		many[i] = map[string]any{"file": "a.jpg"}
	}
	want(t, photoCall(t, assessPhotos, "POST", "/v1/photos/assess", "", map[string]any{"items": many}), 400, "invalid_request")
	big := []byte(`{"items":[],"kind":"` + strings.Repeat("x", photocheck.MaxAssessBody) + `"}`)
	want(t, photoCall(t, assessPhotos, "POST", "/v1/photos/assess", "", big), 413, "too_large")
	for _, body := range []any{
		[]byte(""),
		[]byte("{"),
		map[string]any{"items": []any{map[string]any{"version": 99, "file": "a.jpg"}}},
		map[string]any{"items": []any{map[string]any{"file": "a.jpg"}, map[string]any{"file": "./a.jpg"}}},
		map[string]any{"kind": "pet"},
		map[string]any{"resolution": "huge"},
		map[string]any{"pick": -1},
	} {
		want(t, photoCall(t, assessPhotos, "POST", "/v1/photos/assess", "", body), 400, "invalid_request")
	}
	want(t, photoCall(t, assessPhotos, "POST", "/v1/photos/assess", "", map[string]any{"dataset": "nope"}), 404, "not_found")
	if _, err := os.Stat(dir); err != nil {
		t.Fatal(err)
	}
}

func TestPhotoFixGuard(t *testing.T) {
	e := setup(t)
	dir := photoSet(t, e, "set", map[string][]byte{"a.jpg": photoScene(41), "b.jpg": photoScene(42)})
	settledReview(t, "set", "")
	ctx := t.Context()
	fix := func(body map[string]any) resp {
		t.Helper()
		return photoCall(t, fixPhotos, "POST", "/v1/datasets/set/photos/fix", "set", body)
	}
	setAside := map[string]any{"ops": []any{map[string]any{"op": "set_aside", "files": []string{"b.jpg"}}}, "apply": true}
	running := func(j *store.Job) {
		t.Helper()
		if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"status": "running"}); err != nil {
			t.Fatal(err)
		}
		// t.Context() is cancelled before cleanups run
		t.Cleanup(func() { store.UpdateJob(context.Background(), j.ID, store.Fields{"status": "stopped"}) })
	}
	// changed lists the dataset.changed names emitted since its last call
	var seen int64
	changed := func() []string {
		t.Helper()
		evs, err := events.List(ctx, seen, events.Filter{Types: map[string]bool{events.DatasetChange: true}}, 100)
		if err != nil {
			t.Fatal(err)
		}
		names := []string{}
		for _, ev := range evs {
			seen = ev.ID
			var d struct{ Name string }
			if err := json.Unmarshal(ev.Data, &d); err != nil {
				t.Fatal(err)
			}
			names = append(names, d.Name)
		}
		return names
	}
	changed()

	// a training run reads the folder
	cfg, _ := json.Marshal(map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{
		map[string]any{"type": "diffusion_trainer", "datasets": []any{map[string]any{"folder_path": dir}}}}}})
	train, err := store.CreateJob(ctx, store.NewJob{Name: "run", GpuIDs: "0", JobConfig: string(cfg)})
	if err != nil {
		t.Fatal(err)
	}
	running(train)
	r := fix(setAside)
	want(t, r, 409, "conflict")
	if !strings.Contains(string(r.body), "run is training on this dataset") {
		t.Errorf("409 message: %s", r.body)
	}
	// a preview is fine, and an unknown op is still a 400
	preview := photoCall(t, fixPhotos, "POST", "/v1/datasets/set/photos/fix", "set",
		map[string]any{"ops": []any{map[string]any{"op": "set_aside", "files": []string{"b.jpg"}}}})
	if preview.code != 200 || len(preview.json(t)["changes"].([]any)) != 1 || preview.json(t)["applied"] != false {
		t.Fatalf("preview: %d %s", preview.code, preview.body)
	}
	if _, err := os.Stat(filepath.Join(dir, "b.jpg")); err != nil {
		t.Fatal("the preview moved b.jpg")
	}
	if got := changed(); len(got) != 0 {
		t.Errorf("the preview emitted dataset.changed %v", got)
	}
	r = fix(map[string]any{"ops": []any{map[string]any{"op": "shuffle"}}, "apply": true})
	want(t, r, 400, "invalid_request")
	if !strings.Contains(string(r.body), strings.Join(photocheck.FixOps, ", ")) {
		t.Errorf("400 does not list the ops: %s", r.body)
	}
	store.UpdateJob(ctx, train.ID, store.Fields{"status": "stopped"})

	// a caption job writes into it
	capType := "caption"
	caption, err := store.CreateJob(ctx, store.NewJob{Name: "set_caption", GpuIDs: "0", JobConfig: "{}", JobType: &capType, JobRef: &dir})
	if err != nil {
		t.Fatal(err)
	}
	running(caption)
	r = fix(setAside)
	want(t, r, 409, "conflict")
	if !strings.Contains(string(r.body), "Auto caption is running on this dataset") {
		t.Errorf("409 message: %s", r.body)
	}
	store.UpdateJob(ctx, caption.ID, store.Fields{"status": "stopped"})

	// the dataset's own face check never blocks
	faceType := photocheck.FaceJobType
	faces, err := store.CreateJob(ctx, store.NewJob{Name: "set_faces", GpuIDs: "cpu", JobConfig: "{}", JobType: &faceType, JobRef: &dir})
	if err != nil {
		t.Fatal(err)
	}
	running(faces)
	r = fix(setAside)
	if r.code != 200 {
		t.Fatalf("status %d: %s", r.code, r.body)
	}
	var out photocheck.Fix
	if err := json.Unmarshal(r.body, &out); err != nil {
		t.Fatal(err)
	}
	if !out.Applied || len(out.Changes) != 1 || out.Changes[0].To != ".set_aside/b.jpg" {
		t.Fatalf("fix %+v", out)
	}
	if _, err := os.Stat(filepath.Join(dir, photocheck.SetAsideDir, "b.jpg")); err != nil {
		t.Fatal("b.jpg was not set aside")
	}
	if got := changed(); !slices.Equal(got, []string{"set"}) {
		t.Errorf("the applied fix emitted dataset.changed %v, want [set]", got)
	}
	if out.Review == nil || reviewItem(out.Review, "b.jpg") != nil || out.Review.Faces.JobID != faces.ID {
		t.Fatalf("review after the fix: %+v", out.Review)
	}
	if job, _ := r.json(t)["review"].(map[string]any)["faces"].(map[string]any)["job"].(map[string]any); job == nil || job["id"] != faces.ID {
		t.Errorf("review.faces.job %v", job)
	}

	// an applied fix that changes nothing emits nothing
	if r := fix(map[string]any{"ops": []any{map[string]any{"op": "rename_same_stem"}}, "apply": true}); r.code != 200 || len(r.json(t)["changes"].([]any)) != 0 {
		t.Fatalf("no-op fix: %d %s", r.code, r.body)
	}
	if got := changed(); len(got) != 0 {
		t.Errorf("a fix with no changes emitted dataset.changed %v", got)
	}

	want(t, fix(map[string]any{"ops": []any{}}), 400, "invalid_request")
	want(t, photoCall(t, fixPhotos, "POST", "/v1/datasets/nope/photos/fix", "nope", setAside), 404, "not_found")
}

func TestPhotoSettings(t *testing.T) {
	e := setup(t)
	dir := photoSet(t, e, "set", map[string][]byte{"a.jpg": photoScene(51), "b.jpg": photoScene(52)})
	settledReview(t, "set", "")
	put := func(body any) resp {
		t.Helper()
		return photoCall(t, putPhotoSettings, "PUT", "/v1/datasets/set/photos/settings", "set", body)
	}

	// a decision by file is stored by content and survives a rename
	rv := photoReview(t, put(map[string]any{"decisions": map[string]any{"a.jpg": "keep"}, "reference": "b.jpg", "kind": "style"}))
	if it := reviewItem(rv, "a.jpg"); it == nil || it.Decision != photocheck.DecisionKeep || rv.Reference != "b.jpg" || rv.Kind != photocheck.KindStyle {
		t.Fatalf("settings not applied: %+v reference %q kind %s", it, rv.Reference, rv.Kind)
	}
	if err := os.Rename(filepath.Join(dir, "a.jpg"), filepath.Join(dir, "c.jpg")); err != nil {
		t.Fatal(err)
	}
	rv = settledReview(t, "set", "")
	if it := reviewItem(rv, "c.jpg"); it == nil || it.Decision != photocheck.DecisionKeep {
		t.Fatalf("decision lost on rename: %+v", it)
	}

	// null clears a decision and the reference; the kind stays
	rv = photoReview(t, put(map[string]any{"decisions": map[string]any{"c.jpg": nil}}))
	if it := reviewItem(rv, "c.jpg"); it.Decision != "" || rv.Reference != "b.jpg" {
		t.Fatalf("decision not cleared: %+v reference %q", it, rv.Reference)
	}
	rv = photoReview(t, put(map[string]any{"reference": nil}))
	if rv.Reference != "" || rv.Kind != photocheck.KindStyle {
		t.Fatalf("reference %q kind %s", rv.Reference, rv.Kind)
	}

	for _, body := range []any{
		[]byte("{"),
		map[string]any{"kind": "pet"},
		map[string]any{"reference": "missing.jpg"},
		map[string]any{"reference": "../escape.jpg"},
		map[string]any{"reference": 3},
		map[string]any{"decisions": map[string]any{"c.jpg": "maybe"}},
	} {
		want(t, put(body), 400, "invalid_request")
	}
	want(t, photoCall(t, putPhotoSettings, "PUT", "/v1/datasets/nope/photos/settings", "nope", map[string]any{}), 404, "not_found")

	// forget drops the settings with the analysis; the photos stay
	put(map[string]any{"decisions": map[string]any{"c.jpg": "intended"}, "reference": "b.jpg"})
	want(t, photoCall(t, forgetPhotos, "DELETE", "/v1/datasets/set/photos/analysis", "set", nil), 204, "")
	rv = settledReview(t, "set", "")
	if it := reviewItem(rv, "c.jpg"); it == nil || it.Decision != "" || rv.Reference != "" || rv.Kind != photocheck.KindPerson {
		t.Fatalf("after forget: %+v reference %q kind %s", it, rv.Reference, rv.Kind)
	}
	want(t, photoCall(t, forgetPhotos, "DELETE", "/v1/datasets/nope/photos/analysis", "nope", nil), 404, "not_found")
}
