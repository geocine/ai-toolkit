package photocheck

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
)

const upstreamDDL = `
CREATE TABLE "Settings" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "key" TEXT NOT NULL, "value" TEXT NOT NULL);
CREATE UNIQUE INDEX "Settings_key_key" ON "Settings"("key");
CREATE TABLE "Queue" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "gpu_ids" TEXT NOT NULL, "is_running" BOOLEAN NOT NULL DEFAULT false);
CREATE UNIQUE INDEX "Queue_gpu_ids_key" ON "Queue"("gpu_ids");
CREATE TABLE "Job" ("id" TEXT NOT NULL PRIMARY KEY, "name" TEXT NOT NULL, "gpu_ids" TEXT NOT NULL, "job_config" TEXT NOT NULL,
  "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "updated_at" DATETIME NOT NULL, "status" TEXT NOT NULL DEFAULT 'stopped',
  "stop" BOOLEAN NOT NULL DEFAULT false, "return_to_queue" BOOLEAN NOT NULL DEFAULT false, "step" INTEGER NOT NULL DEFAULT 0,
  "total_steps" INTEGER, "info" TEXT NOT NULL DEFAULT '', "speed_string" TEXT NOT NULL DEFAULT '', "queue_position" INTEGER NOT NULL DEFAULT 0,
  "pid" INTEGER, "job_type" TEXT NOT NULL DEFAULT 'train', "job_ref" TEXT, "save_now" BOOLEAN NOT NULL DEFAULT false,
  "sample_now" BOOLEAN NOT NULL DEFAULT false);
CREATE UNIQUE INDEX "Job_name_key" ON "Job"("name");
`

// svc is a test server: the upstream and fork databases, a training folder
// and an empty dataset folder.
type svc struct {
	ctx      context.Context
	root     string
	training string
	dir      string
}

func setupService(t *testing.T) *svc {
	t.Helper()
	root := testutil.TempDir(t)
	s := &svc{ctx: context.Background(), root: root, training: filepath.Join(root, "output"),
		dir: filepath.Join(root, "datasets", "set")}
	for _, d := range []string{s.training, s.dir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	dbPath := filepath.Join(root, "aitk_db.db")
	db, err := sql.Open("sqlite", store.FileURI(dbPath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(upstreamDDL); err != nil {
		t.Fatal(err)
	}
	db.Close()
	paths.ToolkitRoot, paths.DBPath = root, dbPath
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertSetting(s.ctx, "TRAINING_FOLDER", s.training); err != nil {
		t.Fatal(err)
	}
	settings.Flush()
	prev := ffmpegVersion
	ffmpegVersion = func() string { return "" }
	convertFormats = sync.OnceValue(detectFormats)
	t.Cleanup(func() {
		drain()
		ffmpegVersion = prev
		convertFormats = sync.OnceValue(detectFormats)
		store.DB.Close()
		state.DB.Close()
	})
	return s
}

// drain waits until the pool has nothing queued or running, so no
// background analysis outlives a test's databases.
func drain() {
	analysis.start()
	for {
		analysis.mu.Lock()
		n := len(analysis.byKey)
		analysis.mu.Unlock()
		if n == 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func (s *svc) write(t *testing.T, files map[string][]byte) {
	t.Helper()
	if err := testutil.WriteTree(s.dir, files); err != nil {
		t.Fatal(err)
	}
}

// settled is a review once every photo is analysed.
func (s *svc) settled(t *testing.T, opts Options) *Review {
	t.Helper()
	opts.Wait = 30 * time.Second
	r, err := ReviewDataset(s.ctx, s.dir, opts)
	if err != nil {
		t.Fatal(err)
	}
	if r.Pending != 0 {
		t.Fatalf("still %d pending", r.Pending)
	}
	return r
}

func itemFiles(r *Review) []string {
	var out []string
	for _, it := range r.Items {
		out = append(out, it.File)
	}
	return out
}

func itemByFile(r *Review, file string) *Item {
	for i := range r.Items {
		if r.Items[i].File == file {
			return &r.Items[i]
		}
	}
	return nil
}

func scene(seed int) []byte { return testutil.JPEG(testutil.Scene(seed, 1200, 900), 90) }

func TestReviewWalk(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{
		"b.jpg":               scene(1),
		"b.png":               testutil.PNG(testutil.Scene(2, 640, 640)),
		"c.heic":              []byte("\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"),
		"sub/d.jpg":           scene(3),
		"_controls/x.jpg":     scene(4),
		".set_aside/old.jpg":  scene(5),
		".thumbs/b.jpg.jpg":   scene(6),
		"notes.txt":           []byte("not a photo"),
		"b.txt":               []byte("a caption"),
		"clip.mp4":            []byte("not a photo either"),
		".hidden/secret.jpg":  scene(7),
		"sub/_controls/y.png": testutil.PNG(testutil.Scene(8, 64, 64)),
	})
	r, err := ReviewDataset(s.ctx, s.dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"b.jpg", "b.png", "c.heic", "sub/d.jpg"}; !slices.Equal(itemFiles(r), want) {
		t.Fatalf("items %v, want %v", itemFiles(r), want)
	}
	if r.Pending == 0 {
		t.Error("wait 0 should leave photos pending")
	}
	r = s.settled(t, Options{})
	if it := itemByFile(r, "c.heic"); it == nil || !slices.Contains(it.Flags, FlagUnsupported) || slices.Contains(it.Flags, FlagUnreadable) {
		t.Errorf("c.heic: %+v", it)
	}
	for _, f := range []string{"b.jpg", "b.png"} {
		if it := itemByFile(r, f); !slices.Contains(it.Flags, FlagSameStem) {
			t.Errorf("%s flags %v, want same_stem", f, it.Flags)
		}
	}
	if len(r.SetAside) != 1 || r.SetAside[0].File != ".set_aside/old.jpg" || r.SetAside[0].From != "old.jpg" || r.SetAside[0].Flags == nil {
		t.Errorf("set_aside %+v", r.SetAside)
	}
	if !slices.Equal(r.Convert, []string{"bmp", "gif", "tiff"}) {
		t.Errorf("convert %v", r.Convert)
	}
	if r.Faces.State != "none" || r.Faces.Available != FacesAvailable {
		t.Errorf("faces %+v", r.Faces)
	}
	// cached by content: a second review needs no analysis
	r, err = ReviewDataset(s.ctx, s.dir, Options{})
	if err != nil || r.Pending != 0 {
		t.Fatalf("cached review pending %d, %v", r.Pending, err)
	}
}

func TestSaveSettings(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{"a.jpg": scene(21), "b.jpg": scene(22)})
	s.settled(t, Options{})
	keep, style := DecisionKeep, KindStyle
	r, err := SaveSettings(s.ctx, s.dir, SettingsPatch{Kind: &style, Reference: json.RawMessage(`"b.jpg"`),
		Decisions: map[string]*Decision{"a.jpg": &keep}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Kind != KindStyle || r.Reference != "b.jpg" || itemByFile(r, "a.jpg").Decision != DecisionKeep {
		t.Fatalf("kind %s reference %q decision %q", r.Kind, r.Reference, itemByFile(r, "a.jpg").Decision)
	}
	// stored by content: it survives a rename
	if err := os.Rename(filepath.Join(s.dir, "a.jpg"), filepath.Join(s.dir, "c.jpg")); err != nil {
		t.Fatal(err)
	}
	r = s.settled(t, Options{})
	if itemByFile(r, "c.jpg").Decision != DecisionKeep {
		t.Fatalf("decision lost on rename: %+v", itemByFile(r, "c.jpg"))
	}
	r, err = SaveSettings(s.ctx, s.dir, SettingsPatch{Reference: json.RawMessage(`null`), Decisions: map[string]*Decision{"c.jpg": nil}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Reference != "" || itemByFile(r, "c.jpg").Decision != "" || r.Kind != KindStyle {
		t.Fatalf("clear: reference %q decision %q kind %s", r.Reference, itemByFile(r, "c.jpg").Decision, r.Kind)
	}
	var ue *UserError
	bad := Kind("pet")
	for _, p := range []SettingsPatch{
		{Kind: &bad},
		{Reference: json.RawMessage(`"missing.jpg"`)},
		{Reference: json.RawMessage(`"../escape.jpg"`)},
		{Decisions: map[string]*Decision{"c.jpg": new(Decision("maybe"))}},
	} {
		if _, err := SaveSettings(s.ctx, s.dir, p); !errors.As(err, &ue) || ue.Conflict {
			t.Errorf("%+v: err %v, want a 400", p, err)
		}
	}
}

func TestAssess(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{"stored.jpg": scene(31), "other.jpg": scene(32)})
	stored := s.settled(t, Options{})
	// the same photo measured "in the browser" is a duplicate of the stored one
	m := measureBytes(t, "new.jpg", scene(31))
	m.Source, m.Noise, m.Detail = "browser", nil, nil
	fresh := measureBytes(t, "fresh.jpg", scene(33))
	r, err := Assess(s.ctx, AssessRequest{Items: []Metrics{m, fresh}, Resolution: json.RawMessage(`[512, 768]`)}, s.dir)
	if err != nil {
		t.Fatal(err)
	}
	if r.Resolution != 768 || r.Total != 2 {
		t.Fatalf("resolution %d total %d", r.Resolution, r.Total)
	}
	it := itemByFile(r, "new.jpg")
	if it == nil || !slices.Contains(it.Flags, FlagDuplicate) || it.SameAs != "stored.jpg" {
		t.Fatalf("new.jpg %+v", it)
	}
	if itemByFile(r, "stored.jpg") != nil {
		t.Fatal("stored photos are not items")
	}
	if f := itemByFile(r, "fresh.jpg"); f == nil || len(f.Flags) != 0 {
		t.Fatalf("fresh.jpg %+v", f)
	}
	// metrics alone give the flags the stored check gives
	r, err = Assess(s.ctx, AssessRequest{Items: []Metrics{measureBytes(t, "other.jpg", scene(32))}}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := itemByFile(r, "other.jpg").Flags, itemByFile(stored, "other.jpg").Flags; !slices.Equal(got, want) {
		t.Errorf("assess flags %v, check flags %v", got, want)
	}
	r, err = Assess(s.ctx, AssessRequest{Items: []Metrics{}}, "")
	if err != nil || !slices.Equal(r.Convert, []string{"bmp", "gif", "tiff"}) || r.Items == nil {
		t.Fatalf("empty assess: %v %+v", err, r)
	}
	var ue *UserError
	for name, req := range map[string]AssessRequest{
		"too many":   {Items: make([]Metrics, MaxAssessItems+1)},
		"no file":    {Items: []Metrics{{Version: 1}}},
		"version":    {Items: []Metrics{{Version: 99, File: "a.jpg"}}},
		"twice":      {Items: []Metrics{{File: "a.jpg"}, {File: "./a.jpg"}}},
		"kind":       {Kind: "pet"},
		"resolution": {Resolution: json.RawMessage(`"huge"`)},
	} {
		if _, err := Assess(s.ctx, req, ""); !errors.As(err, &ue) {
			t.Errorf("%s: err %v, want a 400", name, err)
		}
	}
}

func TestCleanMetrics(t *testing.T) {
	nan := 0.0
	nan = nan / nan
	big := 1e9
	m, err := cleanMetrics(Metrics{File: `sub\..\..\a.jpg`, Sharp: -3, LumaMean: 900, ClipHi: nan, Noise: &big,
		Width: -5, Orientation: 12, DHash: "zz", Thumb32: "short", Layout: "AAAA", SHA256: "abc", Source: "server",
		Exif: &ExifInfo{Make: strings.Repeat("x", 500)}})
	if err != nil {
		t.Fatal(err)
	}
	if m.File != "a.jpg" || m.Sharp != 0 || m.LumaMean != 255 || m.ClipHi != 0 || *m.Noise != 1000 || m.Width != 0 ||
		m.Orientation != 1 || m.DHash != "" || m.Thumb32 != "" || m.Layout != "" || m.SHA256 != "" || m.Version != MetricsVersion ||
		len(m.Exif.Make) != 128 {
		t.Fatalf("%+v", m)
	}
}

func TestPoolDedupe(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{"a.jpg": scene(41)})
	p := filepath.Join(s.dir, "a.jpg")
	st, _ := os.Stat(p)
	a := analysis.analyse(p, "a.jpg", st, false)
	b := analysis.analyse(p, "a.jpg", st, true)
	if a != b {
		t.Fatal("the same file was queued twice")
	}
	<-a.done
	if a.err != nil || a.sha == "" || !a.m.Decoded {
		t.Fatalf("task %+v", a)
	}
	Warm(s.dir, []string{"a.jpg", "notes.txt", "missing.jpg"})
	s.settled(t, Options{})
}

func TestFFmpegVersion(t *testing.T) {
	for line, want := range map[string]bool{
		"ffmpeg version 7.1.1-essentials_build-www.gyan.dev Copyright (c) 2000-2025": true,
		"ffmpeg version n7.1 Copyright (c) 2000-2024":                                true,
		"ffmpeg version 8.0 Copyright":                                               true,
		"ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023":                      false,
		"ffmpeg version 7.0.2 Copyright":                                             false,
		"ffmpeg version N-118000-g1234abcd Copyright":                                true,
		"": false,
	} {
		if got := ffmpegReadsHEIF(line); got != want {
			t.Errorf("%q: %v, want %v", line, got, want)
		}
	}
	ffmpegVersion = func() string { return "ffmpeg version 7.1 Copyright" }
	convertFormats = sync.OnceValue(detectFormats)
	if got := ConvertFormats(); !slices.Equal(got, []string{"bmp", "gif", "tiff", "heic", "heif", "avif"}) {
		t.Errorf("formats %v", got)
	}
	ffmpegVersion = func() string { return "" }
	convertFormats = sync.OnceValue(detectFormats)
}

// faceJob adds a photo_faces job for the dataset with a faces.json.
func (s *svc) faceJob(t *testing.T, name, status string, shas []string, res *FaceJobResult) *store.Job {
	t.Helper()
	files := []map[string]string{}
	for _, sha := range shas {
		files = append(files, map[string]string{"file": sha[:8] + ".jpg", "sha256": sha})
	}
	cfg, _ := json.Marshal(map[string]any{"job": "extension", "config": map[string]any{"name": name,
		"process": []any{map[string]any{"type": "aitk_photo_faces", "files": files}}}})
	typ, ref := FaceJobType, s.dir
	j, err := store.CreateJob(s.ctx, store.NewJob{Name: name, GpuIDs: "cpu", JobConfig: string(cfg), JobType: &typ, JobRef: &ref})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.UpdateJob(s.ctx, j.ID, store.Fields{"status": status}); err != nil {
		t.Fatal(err)
	}
	if res != nil {
		b, _ := json.Marshal(res)
		if err := testutil.WriteTree(filepath.Join(s.training, name), map[string][]byte{FaceResultFile: b}); err != nil {
			t.Fatal(err)
		}
	}
	return j
}

func (s *svc) sha(t *testing.T, rel string) string {
	t.Helper()
	sum, err := state.FileSHA256(s.ctx, filepath.Join(s.dir, filepath.FromSlash(rel)), nil)
	if err != nil {
		t.Fatal(err)
	}
	return sum
}

// oneFace is face data with a single clear face of height fh.
func oneFace(fh float64, side int) FaceData {
	return FaceData{Algo: FaceAlgo, FrameSharp: 100, Faces: []Face{{
		Box:   [4]float64{0.45, 0.1, fh * 0.75, fh},
		Kps:   [5][2]float64{{0.47, 0.13}, {0.5, 0.13}, {0.485, 0.15}, {0.47, 0.17}, {0.5, 0.17}},
		Score: 0.95, Side: side, AlignSide: side, Sharp: 80, MeanY: 120, ClipHi: 0.01,
	}}}
}

func TestIngest(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{"a.jpg": scene(51), "b.jpg": scene(52), ".set_aside/c.jpg": scene(53)})
	s.settled(t, Options{})
	a, b := s.sha(t, "a.jpg"), s.sha(t, "b.jpg")
	res := &FaceJobResult{Algo: FaceAlgo, CV2: "4.10.0", Model: FaceModel,
		Items:  map[string]FaceData{a: oneFace(0.3, 300), b: oneFace(0.3, 300)},
		Matrix: FaceMatrix{Faces: []string{a + ":0", b + ":0"}, Cos: encodeCos(2, func(i, j int) float64 { return 0.7 })}}
	job := s.faceJob(t, "set_faces_1", "completed", []string{a, b}, res)

	r := s.settled(t, Options{})
	rows, err := state.GetPhotoFaces(s.ctx, []string{a, b}, FaceAlgo)
	if err != nil || len(rows) != 2 {
		t.Fatalf("face rows %d, %v", len(rows), err)
	}
	set, _ := state.GetPhotoSet(s.ctx, s.dir)
	id := storedIdentity(set)
	if id == nil || id.JobID != job.ID || id.SetSig != SetSig([]string{a, b}) || len(id.Faces) != 2 {
		t.Fatalf("identity %+v", id)
	}
	if _, err := os.Stat(filepath.Join(s.training, job.Name, FaceResultFile)); !os.IsNotExist(err) {
		t.Fatal("faces.json not deleted")
	}
	if r.Faces.State != "done" || r.Faces.JobID != job.ID || r.Faces.Checked != 2 || r.Faces.Stale || r.Faces.Added != 0 {
		t.Fatalf("faces %+v", r.Faces)
	}
	if r.Provisional {
		t.Error("checked faces: not provisional")
	}
	// a second review ingests nothing new
	updated := set.UpdatedAt
	s.settled(t, Options{})
	if again, _ := state.GetPhotoSet(s.ctx, s.dir); again.UpdatedAt != updated {
		t.Error("second review re-ingested")
	}
	// a photo added since the check makes it stale
	s.write(t, map[string][]byte{"d.jpg": scene(54)})
	r = s.settled(t, Options{})
	if !r.Faces.Stale || r.Faces.Added != 1 {
		t.Fatalf("after adding: %+v", r.Faces)
	}
	// a running job shows as running; the stored identity stays
	s.faceJob(t, "set_faces_2", "running", nil, nil)
	r = s.settled(t, Options{})
	if r.Faces.State != "running" || r.Faces.Checked != 2 {
		t.Fatalf("running: %+v", r.Faces)
	}

	// GC keeps what reviews still see (set-aside photos too) and drops the rest
	c := s.sha(t, ".set_aside/c.jpg")
	gone := strings.Repeat("0", 64)
	if err := state.PutPhotoMetrics(s.ctx, gone, MetricsCacheVersion, []byte(`{}`)); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-100 * 24 * time.Hour).UnixMilli()
	if _, err := state.DB.Exec(`UPDATE photo_metrics SET seen_at = ?`, old); err != nil {
		t.Fatal(err)
	}
	s.settled(t, Options{})
	if err := GC(s.ctx, time.Now()); err != nil {
		t.Fatal(err)
	}
	kept, _ := state.GetPhotoMetrics(s.ctx, []string{a, c, gone}, MetricsCacheVersion)
	if kept[a] == nil || kept[c] == nil || kept[gone] != nil {
		t.Fatalf("after GC: a %v, set-aside %v, unseen %v", kept[a] != nil, kept[c] != nil, kept[gone] != nil)
	}
	// and the state of a folder that is gone
	if err := state.PutPhotoSet(s.ctx, &state.PhotoSet{Dir: filepath.Join(s.root, "gone")}); err != nil {
		t.Fatal(err)
	}
	if err := GC(s.ctx, time.Now()); err != nil {
		t.Fatal(err)
	}
	if dirs, _ := state.PhotoSetDirs(s.ctx); !slices.Equal(dirs, []string{filepath.Clean(s.dir)}) {
		t.Fatalf("photo sets after GC %v", dirs)
	}
}

func TestForget(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{"a.jpg": scene(61), ".set_aside/b.jpg": scene(62)})
	s.settled(t, Options{})
	keep := DecisionKeep
	if _, err := SaveSettings(s.ctx, s.dir, SettingsPatch{Decisions: map[string]*Decision{"a.jpg": &keep}}); err != nil {
		t.Fatal(err)
	}
	a, b := s.sha(t, "a.jpg"), s.sha(t, ".set_aside/b.jpg")
	if err := Forget(s.ctx, s.dir); err != nil {
		t.Fatal(err)
	}
	if rows, _ := state.GetPhotoMetrics(s.ctx, []string{a, b}, MetricsCacheVersion); len(rows) != 0 {
		t.Fatalf("metrics left: %d", len(rows))
	}
	if set, _ := state.GetPhotoSet(s.ctx, s.dir); set != nil {
		t.Fatalf("photo set left: %+v", set)
	}
	if _, err := os.Stat(filepath.Join(s.dir, "a.jpg")); err != nil {
		t.Fatal("forget touched the photos")
	}
}

// Adding photos to a dataset: the readiness counts the photos already
// stored, while the items and counts stay the posted ones.
func TestAssessReadinessCountsStored(t *testing.T) {
	s := setupService(t)
	tree := map[string][]byte{}
	for i := range 15 {
		tree[fmt.Sprintf("st%02d.jpg", i)] = scene(400 + i)
	}
	s.write(t, tree)
	if r := s.settled(t, Options{}); r.Usable != 15 {
		t.Fatalf("stored usable %d", r.Usable)
	}
	posted := []Metrics{measureBytes(t, "again.jpg", scene(400)), measureBytes(t, "fresh.jpg", scene(420))}
	alone, err := Assess(s.ctx, AssessRequest{Items: posted}, "")
	if err != nil {
		t.Fatal(err)
	}
	if alone.Usable != 2 || alone.Grade != "F" || !slices.ContainsFunc(alone.Hints, func(h string) bool { return strings.HasPrefix(h, "Add ") }) {
		t.Fatalf("without the dataset: usable %d grade %s hints %v", alone.Usable, alone.Grade, alone.Hints)
	}
	r, err := Assess(s.ctx, AssessRequest{Items: posted}, s.dir)
	if err != nil {
		t.Fatal(err)
	}
	// again.jpg duplicates st00.jpg, so 15 stored + fresh.jpg
	if r.Total != 2 || len(r.Items) != 2 || r.Counts[FlagDuplicate] != 1 || r.Usable != 16 {
		t.Fatalf("with the dataset: total %d items %d counts %v usable %d", r.Total, len(r.Items), r.Counts, r.Usable)
	}
	if r.Grade == "F" || r.Grade == "D" || slices.ContainsFunc(r.Hints, func(h string) bool { return strings.HasPrefix(h, "Add ") && strings.Contains(h, "more") }) {
		t.Fatalf("with the dataset: grade %s (%s) hints %v", r.Grade, r.GradeLine, r.Hints)
	}
	if f := findingOf(r, "count_low"); f != nil {
		t.Errorf("count finding %+v", f)
	}

	// a pick too short for n: alone its note asks for more, with the
	// dataset it agrees with the readiness, which asks for none
	alone, err = Assess(s.ctx, AssessRequest{Items: posted, Pick: 10}, "")
	if err != nil {
		t.Fatal(err)
	}
	if alone.Pick == nil || !strings.Contains(alone.Pick.Note, alone.Hints[0]) {
		t.Fatalf("pick alone: %+v, hints %v", alone.Pick, alone.Hints)
	}
	r, err = Assess(s.ctx, AssessRequest{Items: posted, Pick: 10}, s.dir)
	if err != nil {
		t.Fatal(err)
	}
	if r.Pick == nil || !slices.Equal(r.Pick.Kept, []string{"fresh.jpg"}) || r.Usable != 16 {
		t.Fatalf("pick with the dataset: %+v, usable %d", r.Pick, r.Usable)
	}
	if want := "Only 1 photo passes, so all are kept."; r.Pick.Note != want {
		t.Errorf("pick note %q, want %q (hints %v)", r.Pick.Note, want, r.Hints)
	}

	// a dataset still short: the note gives the combined readiness's count
	small := setupService(t)
	small.write(t, map[string][]byte{"st00.jpg": scene(430), "st01.jpg": scene(431), "st02.jpg": scene(432)})
	small.settled(t, Options{})
	r, err = Assess(small.ctx, AssessRequest{Items: posted, Pick: 10}, small.dir)
	if err != nil {
		t.Fatal(err)
	}
	if r.Usable != 5 || len(r.Hints) == 0 || r.Hints[0] != "Add 10 more photos of just them." || r.Pick == nil ||
		r.Pick.Note != "Only 2 photos pass, so all are kept. Add 10 more photos of just them." {
		t.Fatalf("short dataset: usable %d hints %v pick %+v", r.Usable, r.Hints, r.Pick)
	}
}
