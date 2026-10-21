package jobfiles

import (
	"database/sql"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

// Upstream DDL (prisma migrate diff of ui/prisma/schema.prisma), Job + Settings.
const ddl = `
CREATE TABLE "Settings" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "key" TEXT NOT NULL, "value" TEXT NOT NULL);
CREATE UNIQUE INDEX "Settings_key_key" ON "Settings"("key");
CREATE TABLE "Job" (
    "id" TEXT NOT NULL PRIMARY KEY, "name" TEXT NOT NULL, "gpu_ids" TEXT NOT NULL, "job_config" TEXT NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "updated_at" DATETIME NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'stopped', "stop" BOOLEAN NOT NULL DEFAULT false,
    "return_to_queue" BOOLEAN NOT NULL DEFAULT false, "step" INTEGER NOT NULL DEFAULT 0, "total_steps" INTEGER,
    "info" TEXT NOT NULL DEFAULT '', "speed_string" TEXT NOT NULL DEFAULT '', "queue_position" INTEGER NOT NULL DEFAULT 0,
    "pid" INTEGER, "job_type" TEXT NOT NULL DEFAULT 'train', "job_ref" TEXT,
    "save_now" BOOLEAN NOT NULL DEFAULT false, "sample_now" BOOLEAN NOT NULL DEFAULT false);
CREATE UNIQUE INDEX "Job_name_key" ON "Job"("name");
`

// The trainer's loss log schema (toolkit/logging_aitk.py).
const lossDDL = `
CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL);
CREATE TABLE metric_keys (key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER);
CREATE TABLE metrics (step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL, value_text TEXT,
  PRIMARY KEY (step, key), FOREIGN KEY (step) REFERENCES steps(step) ON DELETE CASCADE);
`

type env struct {
	srv    *httptest.Server
	folder string // the job folder
	jobID  string
}

func setup(t *testing.T, jobName string) *env {
	t.Helper()
	dir := t.TempDir()
	db, err := sql.Open("sqlite", "file:"+filepath.ToSlash(filepath.Join(dir, "aitk_db.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if _, err := db.Exec(ddl); err != nil {
		t.Fatal(err)
	}
	store.DB = db
	training := filepath.Join(dir, "output")
	if _, err := db.Exec(`INSERT INTO "Settings" ("key","value") VALUES ('TRAINING_FOLDER', ?)`, training); err != nil {
		t.Fatal(err)
	}
	settings.Flush()
	job, err := store.CreateJob(t.Context(), store.NewJob{Name: jobName, GpuIDs: "0", JobConfig: "{}"})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	Register(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return &env{srv: srv, folder: filepath.Join(training, jobName), jobID: job.ID}
}

func (e *env) do(t *testing.T, method, path, body string) (int, http.Header, string) {
	t.Helper()
	req, _ := http.NewRequest(method, e.srv.URL+path, strings.NewReader(body))
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b := new(strings.Builder)
	_, _ = io_copy(b, res)
	return res.StatusCode, res.Header, b.String()
}

func io_copy(b *strings.Builder, res *http.Response) (int64, error) {
	buf := make([]byte, 32*1024)
	var n int64
	for {
		k, err := res.Body.Read(buf)
		b.Write(buf[:k])
		n += int64(k)
		if err != nil {
			return n, nil
		}
	}
}

func (e *env) write(t *testing.T, rel string, data []byte) string {
	t.Helper()
	p := filepath.Join(e.folder, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func decode[T any](t *testing.T, s string) T {
	t.Helper()
	var v T
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("bad json %q: %v", s, err)
	}
	return v
}

func TestJobNotFound(t *testing.T) {
	e := setup(t, "job")
	for _, c := range []struct{ method, path string }{
		{"GET", "files"}, {"GET", "log"}, {"GET", "loss"}, {"DELETE", "loss"},
		{"GET", "notes"}, {"POST", "notes"}, {"GET", "plugin"}, {"GET", "samples"},
	} {
		code, _, body := e.do(t, c.method, "/api/jobs/missing/"+c.path, "{}")
		if code != 404 || strings.TrimSpace(body) != `{"error":"Job not found"}` {
			t.Errorf("%s %s: %d %s", c.method, c.path, code, body)
		}
	}
}

func TestDecodeUTF8(t *testing.T) {
	cases := map[string]string{
		"ok é ✓":                       "ok é ✓",
		"\xe2\x96":                     "�",   // truncated 3-byte char: one replacement
		"\x96\x88a":                    "��a", // stray continuations: one each
		"\xed\xa0\x80":                 "���", // surrogate encoding
		"\xf0\x9f\x98":                 "�",
		"a\xe2\x96b":                   "a�b",
		"\xc0\xaf":                     "��",
		"\xf4\x90\x80\x80":             "����",
		"\xe2\x96\x88\xe2\x96\x88\xe2": "██�",
	}
	for in, want := range cases {
		if got := decodeUTF8([]byte(in)); got != want {
			t.Errorf("decodeUTF8(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestJSNumbers(t *testing.T) {
	num := map[string]float64{"": 0, " 12 ": 12, "1e3": 1000, "0x10": 16, "0b11": 3, ".5": 0.5, "5.": 5, "-Infinity": math.Inf(-1), "+3": 3}
	for in, want := range num {
		if got := jsNumber(in); got != want {
			t.Errorf("jsNumber(%q) = %v, want %v", in, got, want)
		}
	}
	for _, in := range []string{"abc", "1_000", "inf", "NaN", "0x", "1e", "--1", "12px"} {
		if got := jsNumber(in); !math.IsNaN(got) {
			t.Errorf("jsNumber(%q) = %v, want NaN", in, got)
		}
	}
	ints := map[string]float64{"12abc": 12, " -3": -3, "+7x": 7, "1e3": 1}
	for in, want := range ints {
		if got := jsParseInt(in); got != want {
			t.Errorf("jsParseInt(%q) = %v, want %v", in, got, want)
		}
	}
	for _, in := range []string{"", "abc", "-", " "} {
		if !math.IsNaN(jsParseInt(in)) {
			t.Errorf("jsParseInt(%q) want NaN", in)
		}
	}
	if toNumber([]any{"7"}) != 7 || toNumber([]any{}) != 0 || !math.IsNaN(toNumber([]any{1.0, 2.0})) ||
		toNumber(true) != 1 || toNumber(nil) != 0 || !math.IsNaN(toNumber(map[string]any{})) {
		t.Error("toNumber coercions differ from Number()")
	}
}

type logResp struct {
	Log    string `json:"log"`
	Offset int64  `json:"offset"`
	Reset  bool   `json:"reset"`
}

func TestLog(t *testing.T) {
	e := setup(t, "job")
	_, _, body := e.do(t, "GET", "/api/jobs/"+e.jobID+"/log", "")
	if got := decode[logResp](t, body); got != (logResp{"", 0, true}) {
		t.Fatalf("missing log: %+v", got)
	}
	e.write(t, "log.txt", []byte("line1\nline2\n"))
	base := "/api/jobs/" + e.jobID + "/log"
	check := func(query string, want logResp) {
		t.Helper()
		_, _, body := e.do(t, "GET", base+query, "")
		if got := decode[logResp](t, body); got != want {
			t.Errorf("%s: got %+v want %+v", query, got, want)
		}
	}
	check("", logResp{"line1\nline2\n", 12, true})
	check("?offset=6", logResp{"line2\n", 12, false})
	check("?offset=6abc", logResp{"line2\n", 12, false}) // parseInt
	check("?offset=12", logResp{"", 12, false})
	check("?offset=99", logResp{"line1\nline2\n", 12, true}) // truncated log -> fresh tail
	check("?offset=", logResp{"line1\nline2\n", 12, true})   // NaN
	check("?offset=-1", logResp{"line1\nline2\n\x00", 12, false})
	check("?offset=-2", logResp{"Error reading log file", 0, true})

	// tail of a large file drops the first (possibly partial) line, even when
	// the cut lands exactly on a line start, like upstream
	big := strings.Repeat("x", 100) + "\n" + strings.Repeat("y\n", maxTailBytes/2)
	e.write(t, "log.txt", []byte(big))
	_, _, body = e.do(t, "GET", base, "")
	got := decode[logResp](t, body)
	if got.Offset != int64(len(big)) || !got.Reset || !strings.HasPrefix(got.Log, "y\n") || len(got.Log) != maxTailBytes-2 {
		t.Errorf("tail: offset %d reset %v len %d prefix %q", got.Offset, got.Reset, len(got.Log), got.Log[:4])
	}
}

func TestFilesAndSamples(t *testing.T) {
	e := setup(t, "job")
	_, _, body := e.do(t, "GET", "/api/jobs/"+e.jobID+"/files", "")
	if strings.TrimSpace(body) != `{"files":[]}` {
		t.Fatalf("missing folder: %s", body)
	}
	_, _, body = e.do(t, "GET", "/api/jobs/"+e.jobID+"/samples", "")
	if strings.TrimSpace(body) != `{"samples":[]}` {
		t.Fatalf("missing samples: %s", body)
	}

	e.write(t, "b.safetensors", []byte("bb"))
	e.write(t, "a.safetensors", []byte("a"))
	e.write(t, "optimizer.pt", []byte("opt"))
	e.write(t, "notes.md", []byte("x"))
	_, _, body = e.do(t, "GET", "/api/jobs/"+e.jobID+"/files", "")
	type file struct {
		Path string `json:"path"`
		Size int64  `json:"size"`
	}
	files := decode[struct{ Files []file }](t, body).Files
	want := []file{
		{filepath.Join(e.folder, "a.safetensors"), 1},
		{filepath.Join(e.folder, "b.safetensors"), 2},
		{filepath.Join(e.folder, "optimizer.pt"), 3},
	}
	if len(files) != 3 || files[0] != want[0] || files[1] != want[1] || files[2] != want[2] {
		t.Errorf("files: %+v", files)
	}

	for _, n := range []string{"2.png", "1.jpg", "1.txt", "3.txt", "song_mp3", "x.webp", "x.txt", "song_.txt", "a.gif", ".thumbs/2.png.jpg"} {
		e.write(t, "samples/"+n, []byte("s"))
	}
	if err := os.MkdirAll(filepath.Join(e.folder, "samples", "dir.png"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, _, body = e.do(t, "GET", "/api/jobs/"+e.jobID+"/samples", "")
	got := decode[struct{ Samples []string }](t, body).Samples
	// 1.txt and x.txt are sidecars; "song_mp3" counts as media (no-dot quirk)
	// and its stem is "song" (slice to lastIndexOf('.') == -1), so song_.txt stays
	var names []string
	for _, p := range got {
		names = append(names, filepath.Base(p))
	}
	if strings.Join(names, ",") != "1.jpg,2.png,3.txt,song_.txt,song_mp3,x.webp" {
		t.Errorf("samples: %v", names)
	}
}

func TestNotesAndPlugin(t *testing.T) {
	e := setup(t, "job")
	base := "/api/jobs/" + e.jobID
	_, _, body := e.do(t, "GET", base+"/notes", "")
	if strings.TrimSpace(body) != `{"content":""}` {
		t.Fatalf("empty notes: %s", body)
	}
	code, _, body := e.do(t, "POST", base+"/notes", `{"content":"# hi <b>"}`)
	if code != 200 || strings.TrimSpace(body) != `{"success":true}` {
		t.Fatalf("save notes: %d %s", code, body)
	}
	_, _, body = e.do(t, "GET", base+"/notes", "")
	if strings.TrimSpace(body) != `{"content":"# hi <b>"}` {
		t.Errorf("notes roundtrip: %s", body)
	}
	e.do(t, "POST", base+"/notes", `{"content":5}`)
	if b, _ := os.ReadFile(filepath.Join(e.folder, "notes.md")); len(b) != 0 {
		t.Errorf("non-string content should save empty, got %q", b)
	}
	if code, _, _ := e.do(t, "POST", base+"/notes", `not json`); code != 500 {
		t.Errorf("bad json notes: %d", code)
	}

	_, _, body = e.do(t, "GET", base+"/plugin", "")
	if strings.TrimSpace(body) != `{"exists":false,"html":null}` {
		t.Errorf("no plugin: %s", body)
	}
	e.write(t, "plugin.html", []byte("<h1>x</h1>"))
	_, _, body = e.do(t, "GET", base+"/plugin?check=1", "")
	if strings.TrimSpace(body) != `{"exists":true,"html":null}` {
		t.Errorf("plugin check: %s", body)
	}
	code, hdr, body := e.do(t, "GET", base+"/plugin", "")
	if code != 200 || body != "<h1>x</h1>" || hdr.Get("Content-Type") != "text/html; charset=utf-8" || hdr.Get("Cache-Control") != "no-store" {
		t.Errorf("plugin html: %d %q %v", code, body, hdr)
	}
}

func makeLossDB(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite", "file:"+strings.NewReplacer("%", "%25", "?", "%3f", "#", "%23").Replace(filepath.ToSlash(path))+"?_pragma=journal_mode(WAL)")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(lossDDL); err != nil {
		t.Fatal(err)
	}
	for step := 0; step <= 10; step++ {
		db.Exec(`INSERT INTO steps VALUES (?, ?)`, step, 1000.5+float64(step))
		db.Exec(`INSERT INTO metrics VALUES (?, 'loss', ?, NULL)`, step, 1.0/float64(step+1))
	}
	db.Exec(`INSERT INTO metrics VALUES (3, 'lr', NULL, '0.001')`)
	db.Exec(`INSERT INTO metrics VALUES (4, 'lr', NULL, 'warmup')`)
	db.Exec(`INSERT INTO metric_keys VALUES ('loss', 0, 10), ('lr', 3, 4)`)
}

func TestLoss(t *testing.T) {
	// '%' and '#' in the job name exercise the sqlite URI escaping
	e := setup(t, "job 100% #1")
	base := "/api/jobs/" + e.jobID + "/loss"
	_, _, body := e.do(t, "GET", base, "")
	if strings.TrimSpace(body) != `{"keys":[],"key":"loss","points":[]}` {
		t.Fatalf("no db: %s", body)
	}
	makeLossDB(t, filepath.Join(e.folder, "loss_log.db"))

	type point struct {
		Step     int64    `json:"step"`
		WallTime float64  `json:"wall_time"`
		Value    *float64 `json:"value"`
	}
	type resp struct {
		Key    string   `json:"key"`
		Keys   []string `json:"keys"`
		Points []point  `json:"points"`
	}
	get := func(q string) (int, resp) {
		code, _, body := e.do(t, "GET", base+q, "")
		if code != 200 {
			return code, resp{}
		}
		return code, decode[resp](t, body)
	}
	_, r := get("")
	if r.Key != "loss" || strings.Join(r.Keys, ",") != "loss,lr" || len(r.Points) != 11 || r.Points[1].WallTime != 1001.5 || *r.Points[1].Value != 0.5 {
		t.Errorf("default: %+v", r)
	}
	_, r = get("?since_step=4&stride=2&limit=2")
	if len(r.Points) != 2 || r.Points[0].Step != 6 || r.Points[1].Step != 8 {
		t.Errorf("filters: %+v", r.Points)
	}
	_, r = get("?since_step=abc") // NaN binds NULL: no since filter
	if len(r.Points) != 11 {
		t.Errorf("NaN since_step: %d points", len(r.Points))
	}
	_, r = get("?stride=abc") // Math.max(1, NaN) = NaN: step % NULL matches nothing
	if len(r.Points) != 0 {
		t.Errorf("NaN stride: %d points", len(r.Points))
	}
	_, r = get("?key=lr")
	if len(r.Points) != 2 || *r.Points[0].Value != 0.001 || r.Points[1].Value != nil {
		t.Errorf("value_text fallback: %+v", r.Points)
	}
	if code, _ := get("?limit=abc"); code != 500 {
		t.Errorf("NaN limit should fail like upstream, got %d", code)
	}

	del := func(body string) (int, string) {
		code, _, b := e.do(t, "DELETE", base, body)
		return code, strings.TrimSpace(b)
	}
	const bad = `{"error":"min_step and max_step must be numbers with min_step <= max_step"}`
	for _, b := range []string{``, `{}`, `{"min_step":5,"max_step":2}`, `{"min_step":"x","max_step":2}`, `[1]`, `5`} {
		if code, out := del(b); code != 400 || out != bad {
			t.Errorf("DELETE %q: %d %s", b, code, out)
		}
	}
	if code, _ := del(`null`); code != 500 {
		t.Errorf("DELETE null: %d", code)
	}
	code, out := del(`{"min_step":"3","max_step":[4]}`)
	if code != 200 || out != `{"max_step":4,"min_step":3,"ok":true}` {
		t.Errorf("DELETE range: %d %s", code, out)
	}
	_, r = get("")
	if len(r.Points) != 9 || strings.Join(r.Keys, ",") != "loss" {
		t.Errorf("after delete: %d points, keys %v", len(r.Points), r.Keys)
	}

	os.Remove(filepath.Join(e.folder, "loss_log.db"))
	os.Remove(filepath.Join(e.folder, "loss_log.db-wal"))
	os.Remove(filepath.Join(e.folder, "loss_log.db-shm"))
	if code, out := del(`{"min_step":1,"max_step":2}`); code != 404 || out != `{"error":"No loss log for this job"}` {
		t.Errorf("DELETE without db: %d %s", code, out)
	}
}
