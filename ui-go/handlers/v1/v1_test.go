package v1

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"image"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	_ "modernc.org/sqlite"
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

type env struct {
	root, training, datasets, data, models string
	h                                      http.Handler
}

func setup(t *testing.T) *env {
	t.Helper()
	root := t.TempDir()
	e := &env{root: root, training: filepath.Join(root, "output"), datasets: filepath.Join(root, "datasets"),
		data: filepath.Join(root, "data"), models: filepath.Join(root, "models")}
	for _, d := range []string{e.training, e.datasets, e.data, e.models} {
		os.MkdirAll(d, 0o755)
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
	ctx := context.Background()
	for k, v := range map[string]string{"TRAINING_FOLDER": e.training, "DATASETS_FOLDER": e.datasets,
		"DATA_ROOT": e.data, "MODELS_PATH": e.models} {
		if err := store.UpsertSetting(ctx, k, v); err != nil {
			t.Fatal(err)
		}
	}
	settings.Flush()
	t.Setenv("MODELS_PATH", "")
	t.Setenv("AI_TOOLKIT_AUTH", "")
	t.Cleanup(func() {
		store.DB.Close()
		state.DB.Close()
		state.DB = nil
	})
	mux := http.NewServeMux()
	Register(mux)
	e.h = mux
	return e
}

type resp struct {
	code int
	body []byte
	hdr  http.Header
}

func (r resp) json(t *testing.T) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(r.body, &m); err != nil {
		t.Fatalf("not JSON (%d): %s", r.code, r.body)
	}
	return m
}

func (e *env) do(t *testing.T, method, target string, body io.Reader, hdr map[string]string) resp {
	t.Helper()
	req := httptest.NewRequest(method, target, body)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	e.h.ServeHTTP(rec, req)
	return resp{rec.Code, rec.Body.Bytes(), rec.Header()}
}

func (e *env) call(t *testing.T, method, target string, body any) resp {
	t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	return e.do(t, method, target, rd, map[string]string{"Content-Type": "application/json"})
}

func want(t *testing.T, r resp, code int, errCode string) {
	t.Helper()
	if r.code != code {
		t.Fatalf("status %d, want %d: %s", r.code, code, r.body)
	}
	if errCode != "" {
		got, _ := r.json(t)["error"].(map[string]any)
		if got["code"] != errCode {
			t.Fatalf("error code %v, want %s: %s", got["code"], errCode, r.body)
		}
	}
}

func config(name string) map[string]any {
	return map[string]any{"job": "extension", "config": map[string]any{"name": name,
		"process": []any{map[string]any{"type": "diffusion_trainer", "train": map[string]any{"steps": 10}}}}}
}

func TestJobLifecycle(t *testing.T) {
	e := setup(t)
	ctx := context.Background()

	want(t, e.call(t, "POST", "/v1/jobs", map[string]any{"name": "a", "config": map[string]any{"job": "x"}}), 422, "invalid_config")
	want(t, e.call(t, "POST", "/v1/jobs", map[string]any{"name": "../evil", "config": config("x")}), 400, "invalid_request")
	want(t, e.call(t, "POST", "/v1/jobs", map[string]any{"config": config("x")}), 400, "invalid_request")
	want(t, e.call(t, "POST", "/v1/jobs", map[string]any{"name": "a", "config": config("a"),
		"policy": map[string]any{"sync_remote": "nope"}}), 400, "invalid_request")

	r := e.call(t, "POST", "/v1/jobs", map[string]any{"name": "lora1", "gpu_ids": "1", "config": config("lora1"),
		"policy": map[string]any{"auto_resume": true}})
	want(t, r, 201, "")
	job := r.json(t)
	id := job["id"].(string)
	if job["state"] != "draft" || job["status"] != "stopped" || job["gpu_ids"] != "1" {
		t.Fatalf("new job: %s", r.body)
	}
	if p := job["policy"].(map[string]any); p["auto_resume"] != true || p["max_auto_resumes"] != float64(3) {
		t.Fatalf("policy: %v", p)
	}
	// key order of the stored config is the order the client sent
	raw := `{"name":"ordered","config":{"meta":{"v":1},"job":"extension","config":{"process":[{"type":"t","lr":1e-4}],"name":"ordered"}}}`
	r2 := e.do(t, "POST", "/v1/jobs", strings.NewReader(raw), map[string]string{"Content-Type": "application/json"})
	want(t, r2, 201, "")
	if !strings.Contains(string(r2.body), `"config":{"meta":{"v":1},"job":"extension","config":{"process":[{"type":"t","lr":1e-4}],"name":"ordered"}}`) {
		t.Fatalf("config key order or number text not kept: %s", r2.body)
	}
	if stored, _ := store.GetJob(ctx, r2.json(t)["id"].(string)); !strings.HasPrefix(stored.JobConfig, `{"meta":{"v":1},"job":`) {
		t.Fatalf("stored job_config: %s", stored.JobConfig)
	}
	want(t, e.call(t, "POST", "/v1/jobs", map[string]any{"name": "lora1", "config": config("lora1")}), 409, "conflict")

	r = e.call(t, "PATCH", "/v1/jobs/"+id, map[string]any{"gpu_ids": "0", "policy": map[string]any{"max_auto_resumes": 5}})
	want(t, r, 200, "")
	if j := r.json(t); j["gpu_ids"] != "0" || j["policy"].(map[string]any)["max_auto_resumes"] != float64(5) ||
		j["policy"].(map[string]any)["auto_resume"] != true {
		t.Fatalf("patch: %s", r.body)
	}

	// start queues it and runs its queue
	r = e.call(t, "POST", "/v1/jobs/"+id+"/start", nil)
	want(t, r, 200, "")
	if r.json(t)["state"] != "queued" {
		t.Fatalf("after start: %s", r.body)
	}
	q, _ := store.GetQueueByGPU(ctx, "0")
	if q == nil || !q.IsRunning {
		t.Fatalf("queue not running: %+v", q)
	}
	want(t, e.call(t, "PATCH", "/v1/jobs/"+id, map[string]any{"gpu_ids": "2"}), 409, "conflict")
	want(t, e.call(t, "DELETE", "/v1/jobs/"+id, nil), 409, "conflict")

	r = e.call(t, "GET", "/v1/queues", nil)
	if !strings.Contains(string(r.body), `"queued":1`) {
		t.Fatalf("queues: %s", r.body)
	}

	// stopping a queued job just takes it out of the queue
	r = e.call(t, "POST", "/v1/jobs/"+id+"/stop", nil)
	if r.json(t)["state"] != "stopped" {
		t.Fatalf("stop queued: %s", r.body)
	}

	// a running one becomes stopping until the trainer writes its ending
	store.UpdateJob(ctx, id, store.Fields{"status": "running", "step": 4, "info": "Training"})
	if r = e.call(t, "GET", "/v1/jobs/"+id, nil); r.json(t)["state"] != "running" {
		t.Fatalf("running: %s", r.body)
	}
	want(t, e.call(t, "POST", "/v1/jobs/"+id+"/start", nil), 409, "conflict")
	r = e.call(t, "POST", "/v1/jobs/"+id+"/stop", nil)
	if j := r.json(t); j["state"] != "stopping" || j["status"] != "stopping" {
		t.Fatalf("stop running: %s", r.body)
	}
	j, _ := store.GetJob(ctx, id)
	if !j.Stop {
		t.Fatal("stop flag not set")
	}
	r = e.call(t, "POST", "/v1/jobs/"+id+"/kill", nil)
	if r.json(t)["state"] != "stopped" {
		t.Fatalf("kill: %s", r.body)
	}
	if r = e.call(t, "POST", "/v1/jobs/"+id+"/save_now", nil); r.code != 200 {
		t.Fatalf("save_now: %s", r.body)
	}
	if j, _ := store.GetJob(ctx, id); !j.SaveNow {
		t.Fatal("save_now not set")
	}

	state.StartAttempt(ctx, id, 0, false)
	state.EndAttempt(ctx, id, "interrupted", "Interrupted: test", 4)
	if r = e.call(t, "GET", "/v1/jobs/"+id+"/attempts", nil); !strings.Contains(string(r.body), `"end_state":"interrupted"`) {
		t.Fatalf("attempts: %s", r.body)
	}

	// renaming a job with files would orphan them
	os.MkdirAll(filepath.Join(e.training, "lora1"), 0o755)
	want(t, e.call(t, "PATCH", "/v1/jobs/"+id, map[string]any{"name": "lora2"}), 409, "conflict")

	if r = e.call(t, "GET", "/v1/jobs?state=stopped", nil); len(r.json(t)["items"].([]any)) != 1 {
		t.Fatalf("filter: %s", r.body)
	}
	want(t, e.call(t, "DELETE", "/v1/jobs/"+id, nil), 204, "")
	if _, err := os.Stat(filepath.Join(e.training, "lora1")); !os.IsNotExist(err) {
		t.Fatal("job folder not deleted")
	}
	if a, _ := state.LastAttempt(ctx, id); a != nil {
		t.Fatal("attempts not deleted")
	}
	want(t, e.call(t, "GET", "/v1/jobs/"+id, nil), 404, "not_found")
}

func makeJob(t *testing.T, e *env, name string) string {
	t.Helper()
	r := e.call(t, "POST", "/v1/jobs", map[string]any{"name": name, "config": config(name)})
	want(t, r, 201, "")
	return r.json(t)["id"].(string)
}

func TestLogWindows(t *testing.T) {
	e := setup(t)
	id := makeJob(t, e, "logjob")
	folder := filepath.Join(e.training, "logjob")
	os.MkdirAll(folder, 0o755)
	text := "héllo wörld ✓ step 1\nüber 2 — done\n"
	os.WriteFile(filepath.Join(folder, "log.txt"), []byte(text), 0o644)

	// read it in tiny windows: chunks never split a character and join back exactly
	var got strings.Builder
	offset := 0.0
	for i := 0; i < 100; i++ {
		r := e.call(t, "GET", "/v1/jobs/"+id+"/log?limit=3&offset="+itoa(int(offset)), nil)
		m := r.json(t)
		got.WriteString(m["text"].(string))
		if m["next_offset"].(float64) <= offset && m["next_offset"].(float64) < m["size"].(float64) {
			t.Fatalf("no progress at %v: %s", offset, r.body)
		}
		offset = m["next_offset"].(float64)
		if offset >= m["size"].(float64) {
			break
		}
	}
	if got.String() != text {
		t.Fatalf("joined chunks %q != %q", got.String(), text)
	}
	r := e.call(t, "GET", "/v1/jobs/"+id+"/log?offset=-6", nil)
	if m := r.json(t); m["text"] != " done\n" && m["text"] != "done\n" {
		t.Fatalf("tail: %s", r.body)
	}
	want(t, e.call(t, "GET", "/v1/jobs/"+id+"/log?offset=x", nil), 400, "invalid_request")
	other := makeJob(t, e, "nolog")
	if r := e.call(t, "GET", "/v1/jobs/"+other+"/log", nil); r.json(t)["size"] != float64(0) {
		t.Fatalf("missing log: %s", r.body)
	}
}

func TestLossCheckpointsSamples(t *testing.T) {
	e := setup(t)
	id := makeJob(t, e, "art")
	folder := filepath.Join(e.training, "art")
	os.MkdirAll(filepath.Join(folder, "samples"), 0o755)
	db, _ := sql.Open("sqlite", store.FileURI(filepath.Join(folder, "loss_log.db")))
	for _, s := range []string{
		`CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL)`,
		`CREATE TABLE metric_keys (key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER)`,
		`CREATE TABLE metrics (step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL, value_text TEXT, PRIMARY KEY (step, key))`,
		`INSERT INTO metric_keys VALUES ('learning_rate', 1, 10), ('loss/loss', 1, 10)`,
	} {
		if _, err := db.Exec(s); err != nil {
			t.Fatal(err)
		}
	}
	for i := 1; i <= 10; i++ {
		db.Exec(`INSERT INTO steps VALUES (?, ?)`, i, 100.0+float64(i))
		db.Exec(`INSERT INTO metrics VALUES (?, 'loss/loss', ?, NULL), (?, 'learning_rate', 0.0001, NULL)`, i, 1.0/float64(i), i)
	}
	db.Close()

	m := e.call(t, "GET", "/v1/jobs/"+id+"/loss", nil).json(t)
	if m["key"] != "loss/loss" || len(m["points"].([]any)) != 10 {
		t.Fatalf("loss: %v", m)
	}
	p0 := m["points"].([]any)[0].(map[string]any)
	if p0["step"] != float64(1) || p0["wall_time"] != float64(101) {
		t.Fatalf("point: %v", p0)
	}
	m = e.call(t, "GET", "/v1/jobs/"+id+"/loss?key=learning_rate&after_step=4&stride=2&limit=2", nil).json(t)
	pts := m["points"].([]any)
	if len(pts) != 2 || pts[0].(map[string]any)["step"] != float64(6) {
		t.Fatalf("filtered: %v", m)
	}

	os.WriteFile(filepath.Join(folder, "art_000000020.safetensors"), []byte("twenty"), 0o644)
	os.WriteFile(filepath.Join(folder, "art_000000005.safetensors"), []byte("five"), 0o644)
	os.WriteFile(filepath.Join(folder, "art.safetensors"), []byte("final"), 0o644)
	os.WriteFile(filepath.Join(folder, "samples", "1700000000000__000000005_0.jpg"), []byte("x"), 0o644)
	r := e.call(t, "GET", "/v1/jobs/"+id+"/checkpoints?hash=true", nil)
	items := r.json(t)["items"].([]any)
	if len(items) != 3 || items[0].(map[string]any)["name"] != "art_000000005.safetensors" ||
		items[2].(map[string]any)["step"] != nil {
		t.Fatalf("checkpoints: %s", r.body)
	}
	five := sha256.Sum256([]byte("five"))
	if items[0].(map[string]any)["sha256"] != hex.EncodeToString(five[:]) {
		t.Fatalf("hash: %v", items[0])
	}
	var cached int
	state.DB.QueryRow(`SELECT COUNT(*) FROM file_hashes`).Scan(&cached)
	if cached != 3 {
		t.Fatalf("hash cache has %d rows", cached)
	}
	if r := e.call(t, "GET", "/v1/jobs/"+id, nil); r.json(t)["latest_checkpoint"].(map[string]any)["name"] != "art.safetensors" {
		t.Fatalf("latest checkpoint: %s", r.body)
	}
	r = e.call(t, "GET", "/v1/jobs/"+id+"/samples", nil)
	if s := r.json(t)["items"].([]any); len(s) != 1 || s[0].(map[string]any)["step"] != float64(5) {
		t.Fatalf("samples: %s", r.body)
	}
}

func pngBytes(w, h int) []byte {
	var b bytes.Buffer
	png.Encode(&b, image.NewRGBA(image.Rect(0, 0, w, h)))
	return b.Bytes()
}

func zipOf(t *testing.T, entries map[string][]byte, symlink string) []byte {
	t.Helper()
	var b bytes.Buffer
	zw := zip.NewWriter(&b)
	for name, data := range entries {
		f, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		f.Write(data)
	}
	if symlink != "" {
		h := &zip.FileHeader{Name: symlink}
		h.SetMode(os.ModeSymlink | 0o777)
		f, _ := zw.CreateHeader(h)
		f.Write([]byte("/etc/passwd"))
	}
	zw.Close()
	return b.Bytes()
}

func TestDatasets(t *testing.T) {
	e := setup(t)
	for _, bad := range []string{"", "../x", "a/b", ".hidden", " pad"} {
		want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": bad}), 400, "invalid_request")
	}
	want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": "set"}), 201, "")
	want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": "set"}), 409, "conflict")

	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	fw, _ := mw.CreateFormFile("files", "a.png")
	fw.Write(pngBytes(4, 3))
	fw, _ = mw.CreateFormFile("files", "a.txt")
	fw.Write([]byte("a caption"))
	mw.Close()
	r := e.do(t, "POST", "/v1/datasets/set/files", &b, map[string]string{"Content-Type": mw.FormDataContentType()})
	want(t, r, 200, "")

	r = e.call(t, "GET", "/v1/datasets/set", nil)
	m := r.json(t)
	if m["images"] != float64(1) || m["captions"] != float64(1) {
		t.Fatalf("dataset: %s", r.body)
	}
	item := m["items"].([]any)[0].(map[string]any)
	if item["file"] != "a.png" || item["caption"] != "a caption" || item["width"] != float64(4) || item["height"] != float64(3) {
		t.Fatalf("item: %v", item)
	}

	r = e.call(t, "PUT", "/v1/datasets/set/captions/a.png", map[string]any{"caption": "new caption"})
	if r.json(t)["caption"] != "new caption" {
		t.Fatalf("caption: %s", r.body)
	}
	want(t, e.call(t, "PUT", "/v1/datasets/set/captions/missing.png", map[string]any{"caption": "x"}), 404, "not_found")

	// a good archive: one top folder stripped, macOS junk skipped
	good := zipOf(t, map[string][]byte{"set/b.png": pngBytes(2, 2), "set/b.txt": []byte("b"),
		"set/sub/c.png": pngBytes(1, 1), "__MACOSX/set/._b.png": []byte("junk")}, "")
	r = e.do(t, "POST", "/v1/datasets/set/archive", bytes.NewReader(good), map[string]string{"Content-Type": "application/zip"})
	want(t, r, 200, "")
	if m := r.json(t); m["added"] != float64(3) || len(m["skipped"].([]any)) != 1 {
		t.Fatalf("archive: %s", r.body)
	}
	if _, err := os.Stat(filepath.Join(e.datasets, "set", "sub", "c.png")); err != nil {
		t.Fatal("nested file not extracted")
	}

	// malicious archives are rejected whole, nothing written
	for name, z := range map[string][]byte{
		"traversal": zipOf(t, map[string][]byte{"ok.png": pngBytes(1, 1), "../../escape.txt": []byte("x")}, ""),
		"absolute":  zipOf(t, map[string][]byte{"/abs.txt": []byte("x")}, ""),
		"symlink":   zipOf(t, map[string][]byte{"ok2.png": pngBytes(1, 1)}, "link"),
	} {
		r := e.do(t, "POST", "/v1/datasets/set/archive", bytes.NewReader(z), map[string]string{"Content-Type": "application/zip"})
		if r.code != 400 {
			t.Fatalf("%s archive: %d %s", name, r.code, r.body)
		}
	}
	for _, p := range []string{filepath.Join(e.root, "escape.txt"), filepath.Join(e.datasets, "set", "ok.png"),
		filepath.Join(e.datasets, "set", "ok2.png")} {
		if _, err := os.Stat(p); err == nil {
			t.Fatalf("%s was written by a rejected archive", p)
		}
	}

	want(t, e.call(t, "DELETE", "/v1/datasets/set/files/b.png", nil), 204, "")
	if _, err := os.Stat(filepath.Join(e.datasets, "set", "b.txt")); !os.IsNotExist(err) {
		t.Fatal("caption not deleted with its file")
	}
	// a traversal is cleaned to a path inside the dataset, never outside it
	want(t, e.call(t, "DELETE", "/v1/datasets/set/files/..%2F..%2Faitk_db.db", nil), 404, "not_found")
	if _, err := os.Stat(filepath.Join(e.root, "aitk_db.db")); err != nil {
		t.Fatal("traversal reached the database")
	}
	if r := e.call(t, "GET", "/v1/datasets", nil); len(r.json(t)["items"].([]any)) != 1 {
		t.Fatalf("list: %s", r.body)
	}
	var n int
	state.DB.QueryRow(`SELECT COUNT(*) FROM events WHERE type = 'dataset.changed'`).Scan(&n)
	if n < 5 {
		t.Fatalf("dataset.changed events: %d", n)
	}
	want(t, e.call(t, "DELETE", "/v1/datasets/set", nil), 204, "")
	want(t, e.call(t, "GET", "/v1/datasets/set", nil), 404, "not_found")
}

func TestFiles(t *testing.T) {
	e := setup(t)
	put := func(p, body, extra string) resp {
		return e.do(t, "PUT", "/v1/files/content?root=training&path="+p+extra, strings.NewReader(body), nil)
	}
	r := put("job/notes/a.txt", "hello world", "")
	want(t, r, 200, "")
	if m := r.json(t); m["size"] != float64(11) || m["sha256"] != "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9" {
		t.Fatalf("put: %s", r.body)
	}
	want(t, put("job/notes/a.txt", "x", ""), 409, "conflict")
	want(t, put("job/notes/a.txt", "hello again", "&overwrite=true"), 200, "")

	r = e.do(t, "GET", "/v1/files/content?root=training&path=job/notes/a.txt", nil, map[string]string{"Range": "bytes=0-4"})
	if r.code != 206 || string(r.body) != "hello" {
		t.Fatalf("range: %d %q", r.code, r.body)
	}
	etag := r.hdr.Get("ETag")
	if r = e.do(t, "GET", "/v1/files/content?root=training&path=job/notes/a.txt", nil, map[string]string{"If-None-Match": etag}); r.code != 304 {
		t.Fatalf("etag: %d", r.code)
	}

	r = e.call(t, "GET", "/v1/files?root=training&path=job", nil)
	items := r.json(t)["items"].([]any)
	if len(items) != 1 || items[0].(map[string]any)["type"] != "dir" {
		t.Fatalf("list: %s", r.body)
	}
	want(t, e.call(t, "GET", "/v1/files?root=nope", nil), 400, "invalid_request")

	// traversal stays inside the root
	want(t, put("../../escape.txt", "x", ""), 200, "")
	if _, err := os.Stat(filepath.Join(e.root, "escape.txt")); err == nil {
		t.Fatal("traversal escaped the root")
	}
	if _, err := os.Stat(filepath.Join(e.training, "escape.txt")); err != nil {
		t.Fatal("cleaned path not written inside the root")
	}

	want(t, e.call(t, "DELETE", "/v1/files/content?root=training&path=", nil), 400, "invalid_request")
	want(t, e.call(t, "DELETE", "/v1/files/content?root=training&path=job", nil), 400, "invalid_request")
	want(t, e.call(t, "DELETE", "/v1/files/content?root=training&path=job/notes/a.txt", nil), 204, "")
	want(t, e.call(t, "GET", "/v1/files/content?root=training&path=job/notes/a.txt", nil), 404, "not_found")
}

func TestKeysAndAuth(t *testing.T) {
	e := setup(t)
	authed := httpx.Auth(e.h)
	get := func(token string) int {
		req := httptest.NewRequest("GET", "/v1/jobs", nil)
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		rec := httptest.NewRecorder()
		authed.ServeHTTP(rec, req)
		return rec.Code
	}
	if get("") != 200 {
		t.Fatal("open server should not need a token")
	}
	r := e.call(t, "POST", "/v1/keys", map[string]any{"name": "desktop app"})
	want(t, r, 201, "")
	k := r.json(t)
	secret := k["key"].(string)
	if !strings.HasPrefix(secret, "aitk_") {
		t.Fatalf("key: %v", k)
	}
	if r = e.call(t, "GET", "/v1/keys", nil); strings.Contains(string(r.body), secret) || !strings.Contains(string(r.body), "desktop app") {
		t.Fatalf("list leaks or misses keys: %s", r.body)
	}
	if get("") != 401 || get("aitk_wrong") != 401 || get(secret) != 200 {
		t.Fatal("auth with keys")
	}
	other := e.call(t, "POST", "/v1/keys", map[string]any{"name": "agent"}).json(t)["key"].(string)
	want(t, e.call(t, "DELETE", "/v1/keys/"+k["id"].(string), nil), 204, "")
	want(t, e.call(t, "DELETE", "/v1/keys/"+k["id"].(string), nil), 404, "not_found")
	if get(secret) != 401 || get(other) != 200 {
		t.Fatal("revoked key still works, or the remaining one stopped working")
	}
	// health is public even with auth on
	t.Setenv("AI_TOOLKIT_AUTH", "admin")
	req := httptest.NewRequest("GET", "/v1/health", nil)
	rec := httptest.NewRecorder()
	authed.ServeHTTP(rec, req)
	if rec.Code != 200 || get("admin") != 200 {
		t.Fatalf("health %d", rec.Code)
	}
}

func TestSettings(t *testing.T) {
	e := setup(t)
	newTrain := filepath.Join(e.root, "elsewhere")
	r := e.call(t, "PATCH", "/v1/settings", map[string]any{"training_folder": newTrain, "hf_token": "hf_x",
		"auto_resume_default": true, "max_auto_resumes_default": 7})
	want(t, r, 200, "")
	m := r.json(t)
	if m["training_folder"] != newTrain || m["hf_token_set"] != true || m["auto_resume_default"] != true ||
		m["max_auto_resumes_default"] != float64(7) {
		t.Fatalf("settings: %s", r.body)
	}
	if strings.Contains(string(r.body), "hf_x") {
		t.Fatal("hf_token leaked")
	}
	if v, _, _ := store.GetSetting(context.Background(), "TRAINING_FOLDER"); v != newTrain {
		t.Fatalf("upstream row: %q", v)
	}
	want(t, e.call(t, "PATCH", "/v1/settings", map[string]any{"bogus": 1}), 400, "invalid_request")
	want(t, e.call(t, "PATCH", "/v1/settings", map[string]any{"max_auto_resumes_default": -1}), 400, "invalid_request")
	// a job without its own policy picks the defaults up
	id := makeJob(t, e, "pol")
	if p := e.call(t, "GET", "/v1/jobs/"+id, nil).json(t)["policy"].(map[string]any); p["auto_resume"] != true || p["max_auto_resumes"] != float64(7) {
		t.Fatalf("defaults not applied: %v", p)
	}
}

func TestSystemAndHealth(t *testing.T) {
	e := setup(t)
	os.WriteFile(filepath.Join(e.root, "version.py"), []byte(`VERSION = "9.9.9"`+"\n"), 0o644)
	m := e.call(t, "GET", "/v1/health", nil).json(t)
	if m["ok"] != true || m["upstream_version"] != "9.9.9" {
		t.Fatalf("health: %v", m)
	}
	m = e.call(t, "GET", "/v1/system", nil).json(t)
	if len(m["disks"].([]any)) == 0 || m["cpu"] == nil {
		t.Fatalf("system: %v", m)
	}
}

// every endpoint named in docs/headless.md is in the OpenAPI document
func TestOpenAPICoversDoc(t *testing.T) {
	e := setup(t)
	r := e.call(t, "GET", "/v1/openapi.json", nil)
	var doc struct {
		OpenAPI string                    `json:"openapi"`
		Paths   map[string]map[string]any `json:"paths"`
	}
	if err := json.Unmarshal(r.body, &doc); err != nil || doc.OpenAPI != "3.1.0" {
		t.Fatalf("openapi: %v", err)
	}
	md, err := os.ReadFile(filepath.Join("..", "..", "docs", "headless.md"))
	if err != nil {
		t.Fatal(err)
	}
	endpoint := regexp.MustCompile("`(GET|POST|PUT|PATCH|DELETE) (/v1/[^ `?]*)")
	found := 0
	for _, m := range endpoint.FindAllStringSubmatch(string(md), -1) {
		p := strings.ReplaceAll(m[2], "...}", "}")
		found++
		ops, ok := doc.Paths[p]
		if !ok {
			t.Errorf("openapi lacks path %s (from %s)", p, m[0])
			continue
		}
		if _, ok := ops[strings.ToLower(m[1])]; !ok {
			t.Errorf("openapi lacks %s %s", m[1], p)
		}
	}
	if found < 30 {
		t.Fatalf("only %d endpoints parsed from the doc", found)
	}
	// and every route this package serves is documented
	for _, o := range ops {
		if _, ok := doc.Paths[o.path][strings.ToLower(o.method)]; !ok {
			t.Errorf("table entry %s %s missing from the document", o.method, o.path)
		}
	}
}
