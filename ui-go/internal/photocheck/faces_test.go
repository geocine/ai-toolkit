package photocheck

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/photocheck/pyjob"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func TestFacesAvailable(t *testing.T) {
	if !FacesAvailable {
		t.Fatal("faces.go did not turn the face check on")
	}
	s := setupService(t)
	s.write(t, map[string][]byte{"a.jpg": scene(61)})
	if r := s.settled(t, Options{}); !r.Faces.Available || r.Faces.State != "none" {
		t.Fatalf("faces %+v", r.Faces)
	}
}

func TestFaceJobName(t *testing.T) {
	at := time.Date(2026, 10, 24, 18, 15, 0, 0, time.FixedZone("PHT", 8*3600))
	if got := FaceJobName("jihan", at); got != "jihan_faces_20261024_101500" {
		t.Fatalf("name %s", got)
	}
}

// TestFaceFiles: a face check reads what a review lists, set-aside photos
// and _controls left out, each content once.
func TestFaceFiles(t *testing.T) {
	s := setupService(t)
	a := scene(62)
	s.write(t, map[string][]byte{
		"a.jpg": a, "a copy.jpg": a, "sub/b.png": testutil.PNG(testutil.Scene(63, 640, 480)), "x.heic": []byte("not really heic"),
		".set_aside/c.jpg": scene(64), "_controls/d.jpg": scene(65), "notes.txt": []byte("a caption"), ".thumbs/a.jpg": a,
	})
	files, convertible, err := FaceFiles(s.ctx, s.dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range files {
		names = append(names, f.File)
		if len(f.SHA256) != 64 || f.SHA256 != s.sha(t, f.File) {
			t.Errorf("%s: sha %q", f.File, f.SHA256)
		}
	}
	// "a copy.jpg" sorts first and stands for both copies
	if !slices.Equal(names, []string{"a copy.jpg", "sub/b.png", "x.heic"}) {
		t.Fatalf("files %v", names)
	}
	if !slices.Equal(convertible, []string{"x.heic"}) {
		t.Fatalf("convertible %v", convertible)
	}

	// a completed check of exactly those files is current: the review's set
	// signature matches and no photo counts as added
	shas := make([]string, len(files))
	items := map[string]FaceData{}
	for i, f := range files {
		shas[i] = f.SHA256
		items[f.SHA256] = oneFace(0.3, 300)
	}
	items[shas[2]] = FaceData{Algo: FaceAlgo, Faces: []Face{}, Error: "cannot read this file"}
	raw, err := BuildFaceJobConfig(s.dir, "set_faces_1", files, false, filepath.Join(s.root, "models", "opencv_zoo"))
	if err != nil {
		t.Fatal(err)
	}
	job := s.faceJob(t, "set_faces_1", "completed", nil, &FaceJobResult{Algo: FaceAlgo, Model: FaceModel, Items: items,
		Matrix: FaceMatrix{Faces: []string{shas[0] + ":0", shas[1] + ":0"}, Cos: encodeCos(2, func(i, j int) float64 { return 0.7 })}})
	if _, err := store.UpdateJob(s.ctx, job.ID, store.Fields{"job_config": string(raw)}); err != nil {
		t.Fatal(err)
	}
	r := s.settled(t, Options{})
	if r.Faces.State != "done" || r.Faces.JobID != job.ID || r.Faces.Stale || r.Faces.Added != 0 {
		t.Fatalf("faces after the check %+v", r.Faces)
	}

	// an empty dataset has nothing to check
	empty := filepath.Join(s.root, "datasets", "empty")
	testutil.WriteTree(empty, map[string][]byte{"only.txt": []byte("x"), ".set_aside/c.jpg": scene(66)})
	var ue *UserError
	if _, _, err := FaceFiles(s.ctx, empty); !errors.As(err, &ue) || ue.Conflict || !strings.Contains(ue.Message, "no photos") {
		t.Fatalf("empty: %v", err)
	}
	if _, _, err := FaceFiles(s.ctx, filepath.Join(s.root, "datasets", "missing")); err == nil || errors.As(err, &ue) {
		t.Fatalf("missing folder: %v", err)
	}
}

func TestFaceFilesLimit(t *testing.T) {
	s := setupService(t)
	tree := map[string][]byte{}
	for i := range MaxFaceJob + 1 {
		tree[fmt.Sprintf("p%04d.jpg", i)] = nil // never read: the count is checked before hashing
	}
	s.write(t, tree)
	var ue *UserError
	_, _, err := FaceFiles(s.ctx, s.dir)
	if !errors.As(err, &ue) || ue.Conflict || !strings.Contains(ue.Message, fmt.Sprintf("%d photos", MaxFaceJob+1)) ||
		!strings.Contains(ue.Message, "Split the dataset") {
		t.Fatalf("over the limit: %v", err)
	}
	os.Remove(filepath.Join(s.dir, "p0000.jpg"))
	if files, _, err := FaceFiles(s.ctx, s.dir); err != nil || len(files) != 1 { // empty files share one sha
		t.Fatalf("at the limit: %d files, %v", len(files), err)
	}
}

// TestBuildFaceJobConfig: the section 12.2 shape, which photo_faces.py and
// ingest both read.
func TestBuildFaceJobConfig(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "jihan")
	models := filepath.Join(t.TempDir(), "opencv_zoo")
	files := []FileRef{{File: "IMG_0001.jpg", SHA256: strings.Repeat("9f", 32)}, {File: "sub/IMG_0002.png", SHA256: strings.Repeat("a1", 32)}}
	raw, err := BuildFaceJobConfig(dir+string(filepath.Separator), "jihan_faces_20261024_101500", files, true, models)
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Job    string `json:"job"`
		Config struct {
			Name    string           `json:"name"`
			Process []map[string]any `json:"process"`
		} `json:"config"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Job != "extension" || cfg.Config.Name != "jihan_faces_20261024_101500" || len(cfg.Config.Process) != 1 {
		t.Fatalf("config %s", raw)
	}
	p := cfg.Config.Process[0]
	wantKeys := []string{"algo", "dataset_dir", "files", "models", "out", "text", "threads", "type"}
	var keys []string
	for k := range p {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	if !slices.Equal(keys, wantKeys) {
		t.Fatalf("process keys %v, want %v", keys, wantKeys)
	}
	if p["type"] != pyjob.ProcessType || p["algo"] != float64(FaceAlgo) || p["dataset_dir"] != dir || p["text"] != true ||
		p["threads"] != float64(FaceThreads) || p["out"] != FaceResultFile {
		t.Fatalf("process %v", p)
	}
	if fs := p["files"].([]any); len(fs) != 2 || fs[1].(map[string]any)["file"] != "sub/IMG_0002.png" || fs[1].(map[string]any)["sha256"] != files[1].SHA256 {
		t.Fatalf("files %v", fs)
	}
	m := p["models"].(map[string]any)
	if m["dir"] != models || len(m) != len(pyjob.Models)+1 {
		t.Fatalf("models %v", m)
	}
	for key, pin := range pyjob.Models {
		got, _ := m[key].(map[string]any)
		if got == nil || got["url"] != pin.URL || got["sha256"] != pin.SHA256 || got["bytes"] != float64(pin.Bytes) {
			t.Errorf("%s: %v, want %+v", key, got, pin)
		}
	}
	// ingest reads the photo list back for the set signature
	if got := jobShas(string(raw)); !slices.Equal(got, []string{files[0].SHA256, files[1].SHA256}) {
		t.Fatalf("jobShas %v", got)
	}

	for _, bad := range []struct {
		name  string
		files []FileRef
	}{{"", files}, {"x", nil}, {"x", make([]FileRef, MaxFaceJob+1)}} {
		if _, err := BuildFaceJobConfig(dir, bad.name, bad.files, false, models); err == nil {
			t.Errorf("name %q with %d files: no error", bad.name, len(bad.files))
		}
	}
}

func TestFaceDownload(t *testing.T) {
	dir := t.TempDir()
	base := pyjob.Models["yunet4"].Bytes + pyjob.Models["sface"].Bytes
	if got := FaceDownload(dir, false); got != base {
		t.Fatalf("empty folder: %d, want %d", got, base)
	}
	if got := FaceDownload(dir, true); got != base+pyjob.Models["ppocr"].Bytes {
		t.Fatalf("with text: %d", got)
	}
	if got := FaceDownload(filepath.Join(dir, "missing"), false); got != base {
		t.Fatalf("missing folder: %d", got)
	}
	y := pyjob.Models["yunet4"]
	os.WriteFile(filepath.Join(dir, y.File()), make([]byte, y.Bytes), 0o644)
	os.WriteFile(filepath.Join(dir, pyjob.Models["sface"].File()), []byte("truncated"), 0o644)
	if got := FaceDownload(dir, false); got != pyjob.Models["sface"].Bytes {
		t.Fatalf("yunet in place, sface truncated: %d", got)
	}
	if got := FaceModelsDir(filepath.Join("m", "models")); got != filepath.Join("m", "models", "opencv_zoo") {
		t.Fatalf("models dir %s", got)
	}
}
