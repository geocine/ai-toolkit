package v1

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/photocheck/pyjob"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// The face check route is registered by photos_routes.go (PH-U17), so this
// test calls the handler directly, like photos_test.go.

func TestPhotoFaces(t *testing.T) {
	e := setup(t)
	ctx := t.Context()
	dir := photoSet(t, e, "set", map[string][]byte{"a.jpg": photoScene(71), "sub/b.jpg": photoScene(72),
		".set_aside/c.jpg": photoScene(73)})
	faces := func(name string, body any) resp {
		t.Helper()
		return photoCall(t, checkFaces, "POST", "/v1/datasets/"+name+"/photos/faces", name, body)
	}

	// no body: a check without text
	r := faces("set", nil)
	want(t, r, 201, "")
	m := r.json(t)
	if m["type"] != photocheck.FaceJobType || m["gpu_ids"] != "cpu" || m["ref"] != dir || m["status"] != "queued" ||
		!strings.HasPrefix(m["name"].(string), "set_faces_") {
		t.Fatalf("job: %s", r.body)
	}
	id := m["id"].(string)
	p0 := m["config"].(map[string]any)["config"].(map[string]any)["process"].([]any)[0].(map[string]any)
	files := p0["files"].([]any)
	if p0["type"] != pyjob.ProcessType || p0["dataset_dir"] != dir || p0["text"] != false || len(files) != 2 ||
		files[0].(map[string]any)["file"] != "a.jpg" || files[1].(map[string]any)["file"] != "sub/b.jpg" {
		t.Fatalf("process: %v", p0)
	}
	if md := p0["models"].(map[string]any)["dir"]; md != filepath.Join(e.models, pyjob.ModelsSubdir) {
		t.Fatalf("models dir %v", md)
	}
	// the first check says it downloads the models
	if ws := m["warnings"].([]any); len(ws) != 1 || !strings.Contains(ws[0].(string), "downloads the face models (39 MB)") {
		t.Fatalf("warnings %v", ws)
	}

	// on the cpu queue, which exists and runs, with nothing to resume, sync
	// or upload
	q, err := store.GetQueueByGPU(ctx, "cpu")
	if err != nil || q == nil || !q.IsRunning {
		t.Fatalf("cpu queue %+v, %v", q, err)
	}
	pol, err := state.GetPolicy(ctx, id)
	if err != nil || pol.AutoResume == nil || *pol.AutoResume || pol.SyncRemote == nil || *pol.SyncRemote != "" ||
		pol.AutoUploadRemote == nil || *pol.AutoUploadRemote != "" {
		t.Fatalf("policy %+v, %v", pol, err)
	}
	// the dataset and its review name the job
	if pj, _ := e.call(t, "GET", "/v1/datasets/set", nil).json(t)["photo_job"].(map[string]any); pj == nil || pj["id"] != id {
		t.Fatalf("photo_job %v", pj)
	}
	if rv := settledReview(t, "set", ""); rv.Faces.State != "queued" || rv.Faces.JobID != id || !rv.Faces.Available {
		t.Fatalf("review faces %+v", rv.Faces)
	}

	// one check at a time per folder, while queued, running or stopping
	for _, status := range []string{"queued", "running", "stopping"} {
		if _, err := store.UpdateJob(ctx, id, store.Fields{"status": status}); err != nil {
			t.Fatal(err)
		}
		r = faces("set", map[string]any{"text": true})
		want(t, r, 409, "conflict")
		if !strings.Contains(string(r.body), m["name"].(string)) {
			t.Errorf("409 does not name the job: %s", r.body)
		}
	}
	// t.Context() is cancelled before cleanups run
	t.Cleanup(func() { store.UpdateJob(context.Background(), id, store.Fields{"status": "stopped"}) })

	// a done check does not block the next; text on, models in place, a
	// format the trainer skips
	if _, err := store.UpdateJob(ctx, id, store.Fields{"status": "completed"}); err != nil {
		t.Fatal(err)
	}
	zoo := filepath.Join(e.models, pyjob.ModelsSubdir)
	os.MkdirAll(zoo, 0o755)
	for _, key := range []string{"yunet4", "sface", "ppocr"} {
		pin := pyjob.Models[key]
		if err := os.WriteFile(filepath.Join(zoo, pin.File()), make([]byte, pin.Bytes), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "x.heic"), []byte("heic"), 0o644); err != nil {
		t.Fatal(err)
	}
	// the job names carry the second, so a second check in the same second
	// would clash; move the first job's name out of the way
	if _, err := store.UpdateJob(ctx, id, store.Fields{"name": "set_faces_old"}); err != nil {
		t.Fatal(err)
	}
	r = faces("set", map[string]any{"text": true})
	want(t, r, 201, "")
	m2 := r.json(t)
	t.Cleanup(func() { store.UpdateJob(context.Background(), m2["id"].(string), store.Fields{"status": "stopped"}) })
	p0 = m2["config"].(map[string]any)["config"].(map[string]any)["process"].([]any)[0].(map[string]any)
	if p0["text"] != true || len(p0["files"].([]any)) != 3 {
		t.Fatalf("second check: %v", p0)
	}
	if ws := m2["warnings"].([]any); len(ws) != 1 || ws[0] != "1 photo is in a format the face check cannot read (x.heic). Convert it first (Images step, or datasets.fixPhotos convert)." {
		t.Fatalf("warnings %v", ws)
	}

	// requests the handler refuses
	want(t, faces("set", []byte("{")), 400, "invalid_request")
	want(t, faces("missing", nil), 404, "not_found")
	photoSet(t, e, "empty", map[string][]byte{"notes.txt": []byte("x")})
	want(t, faces("empty", nil), 400, "invalid_request")
	big := map[string][]byte{}
	for i := range photocheck.MaxFaceJob + 1 {
		big[fmt.Sprintf("p%04d.jpg", i)] = nil
	}
	photoSet(t, e, "big", big)
	r = faces("big", nil)
	want(t, r, 400, "invalid_request")
	if !strings.Contains(string(r.body), "Split the dataset") {
		t.Errorf("400 over the limit: %s", r.body)
	}
	if n, _ := store.FindJobs(ctx, store.JobQuery{Where: `"job_type" = ?`, Args: []any{photocheck.FaceJobType}}); len(n) != 2 {
		t.Fatalf("%d face jobs, want 2", len(n))
	}
}

func TestPhotoFacesWarnings(t *testing.T) {
	if got := faceWarnings(nil, 0); got == nil || len(got) != 0 {
		t.Fatalf("no notes: %v", got)
	}
	// bmp, gif and tiff the face check reads, so they are not named
	got := faceWarnings([]string{"a.heic", "b.bmp", "c.HEIF", "d.gif", "e.avif", "sub/f.jxl", "g.tiff"}, 2_423_490)
	if len(got) != 2 || got[0] != "4 photos are in formats the face check cannot read (a.heic, c.HEIF, e.avif, …). Convert them first (Images step, or datasets.fixPhotos convert)." ||
		got[1] != "The first face check downloads the face models (2 MB) once." {
		t.Fatalf("notes %q", got)
	}
	if got := faceWarnings([]string{"a.bmp", "b.gif", "c.tif"}, 0); len(got) != 0 {
		t.Fatalf("readable formats: %q", got)
	}
	if got := faceWarnings(nil, 100); got[0] != "The first face check downloads the face models (1 MB) once." {
		t.Fatalf("small download %q", got)
	}
}
