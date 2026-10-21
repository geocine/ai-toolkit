package inference

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/engines"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

func TestSearchParams(t *testing.T) {
	p := parseSearchParams("b=2&job=abc&a=x+y&c=%E2%9C%93&d=100%25&bad=%zz&&e")
	if v, _ := p.get("job"); v != "abc" {
		t.Fatalf("job = %q", v)
	}
	p.del("job")
	// order kept, URLSearchParams escaping (spaces as +, ~ escaped, * kept)
	want := "b=2&a=x+y&c=%E2%9C%93&d=100%25&bad=%25zz&e="
	if got := p.String(); got != want {
		t.Errorf("got %s\nwant %s", got, want)
	}
	if got := formEncode("a*b~c d/é"); got != "a*b%7Ec+d%2F%C3%A9" {
		t.Errorf("formEncode = %s", got)
	}
}

func TestEncodePath(t *testing.T) {
	if got := encodePath("outputs/req 1/img#2.png"); got != "outputs/req%201/img%232.png" {
		t.Errorf("encodePath = %s", got)
	}
}

func endpointFor(t *testing.T, srv *httptest.Server, token string) *engines.Endpoint {
	t.Helper()
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	return &engines.Endpoint{Host: u.Hostname(), Port: port, Token: token}
}

func TestForwardStreams(t *testing.T) {
	release := make(chan struct{})
	var gotReq *http.Request
	var gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotReq = r
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Engine", "yes")
		w.WriteHeader(201)
		io.WriteString(w, "event: progress\ndata: 1\n\n")
		w.(http.Flusher).Flush()
		<-release // hold the stream open: the first frame must already be through
		io.WriteString(w, "event: done\ndata: 2\n\n")
	}))
	defer upstream.Close()

	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		Forward(w, r, endpointFor(t, upstream, "tok123"), "generate", "a=1")
	}))
	defer front.Close()

	req, _ := http.NewRequest("POST", front.URL+"/x", strings.NewReader(`{"prompt":"hi"}`))
	req.Header.Set("Authorization", "Bearer secret")
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 201 {
		t.Errorf("status = %d", res.StatusCode)
	}
	if res.Header.Get("X-Engine") != "yes" || res.Header.Get("Cache-Control") != "no-store" || res.Header.Get("X-Accel-Buffering") != "no" {
		t.Errorf("headers = %v", res.Header)
	}

	rd := bufio.NewReader(res.Body)
	first := make(chan string, 1)
	go func() {
		line, _ := rd.ReadString('\n')
		first <- line
	}()
	select {
	case line := <-first:
		if line != "event: progress\n" {
			t.Errorf("first line = %q", line)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("first frame was buffered instead of streamed")
	}
	close(release)
	rest, _ := io.ReadAll(rd)
	if !strings.Contains(string(rest), "event: done") {
		t.Errorf("rest = %q", rest)
	}

	if gotReq.URL.Path != "/generate" || gotReq.URL.RawQuery != "a=1" {
		t.Errorf("upstream url = %s", gotReq.URL)
	}
	if gotReq.Header.Get("x-engine-token") != "tok123" || gotReq.Header.Get("Authorization") != "" {
		t.Errorf("upstream headers = %v", gotReq.Header)
	}
	if gotBody != `{"prompt":"hi"}` || gotReq.Header.Get("Content-Type") != "application/json" {
		t.Errorf("upstream body = %q", gotBody)
	}
}

func TestForwardUnreachable(t *testing.T) {
	rec := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/inference/x", nil)
	Forward(rec, r, &engines.Endpoint{Host: "127.0.0.1", Port: 1, Token: "t"}, "x", "")
	if rec.Code != 502 || !strings.Contains(rec.Body.String(), "engine unreachable: ") {
		t.Errorf("got %d %s", rec.Code, rec.Body.String())
	}
}

const ddl = `
CREATE TABLE "Settings" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "key" TEXT NOT NULL, "value" TEXT NOT NULL);
CREATE UNIQUE INDEX "Settings_key_key" ON "Settings"("key");
CREATE TABLE "Queue" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "gpu_ids" TEXT NOT NULL, "is_running" BOOLEAN NOT NULL DEFAULT false);
CREATE TABLE "Job" ("id" TEXT NOT NULL PRIMARY KEY, "name" TEXT NOT NULL, "gpu_ids" TEXT NOT NULL, "job_config" TEXT NOT NULL,
 "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "updated_at" DATETIME NOT NULL, "status" TEXT NOT NULL DEFAULT 'stopped',
 "stop" BOOLEAN NOT NULL DEFAULT false, "return_to_queue" BOOLEAN NOT NULL DEFAULT false, "step" INTEGER NOT NULL DEFAULT 0,
 "total_steps" INTEGER, "info" TEXT NOT NULL DEFAULT '', "speed_string" TEXT NOT NULL DEFAULT '', "queue_position" INTEGER NOT NULL DEFAULT 0,
 "pid" INTEGER, "job_type" TEXT NOT NULL DEFAULT 'train', "job_ref" TEXT, "save_now" BOOLEAN NOT NULL DEFAULT false,
 "sample_now" BOOLEAN NOT NULL DEFAULT false);
CREATE UNIQUE INDEX "Job_name_key" ON "Job"("name");`

// setupDB opens a fresh schema with TRAINING_FOLDER pointing at a temp dir.
func setupDB(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "aitk_db.db")
	raw, err := sql.Open("sqlite", store.FileURI(dbPath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(ddl); err != nil {
		t.Fatal(err)
	}
	raw.Close()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.DB.Close() })
	training := filepath.Join(dir, "output")
	if err := store.UpsertSetting(context.Background(), "TRAINING_FOLDER", training); err != nil {
		t.Fatal(err)
	}
	settings.Flush()
	return training
}

func serve(method, target, body string) *httptest.ResponseRecorder {
	mux := http.NewServeMux()
	Register(mux)
	rec := httptest.NewRecorder()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	mux.ServeHTTP(rec, httptest.NewRequest(method, target, rd))
	return rec
}

func TestStatusAndNoEngine(t *testing.T) {
	training := setupDB(t)
	ctx := context.Background()
	jt := "inference"
	job, err := store.CreateJob(ctx, store.NewJob{Name: "engine1", GpuIDs: "0", JobConfig: "{}", JobType: &jt})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.UpdateJob(ctx, job.ID, store.Fields{"status": "running"}); err != nil {
		t.Fatal(err)
	}

	rec := serve("GET", "/api/inference/status", "")
	want := `{"running":false,"engine":{"jobId":"` + job.ID + `","jobName":"engine1","gpuIds":"0","status":"running","ready":false},` +
		`"engines":[{"jobId":"` + job.ID + `","jobName":"engine1","gpuIds":"0","status":"running","ready":false}]}`
	if got := strings.TrimSpace(rec.Body.String()); got != want {
		t.Errorf("status\n got %s\nwant %s", got, want)
	}

	rec = serve("POST", "/api/inference/generate", `{}`)
	if rec.Code != 503 || strings.TrimSpace(rec.Body.String()) != `{"error":"no inference engine is running","engine":{"status":"running"}}` {
		t.Errorf("no engine: %d %s", rec.Code, rec.Body.String())
	}

	// a published endpoint with a live pid makes it ready and proxied to
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, r.URL.Path+"?"+r.URL.RawQuery+" token="+r.Header.Get("x-engine-token"))
	}))
	defer upstream.Close()
	ep := endpointFor(t, upstream, "tkn")
	folder := filepath.Join(training, "engine1")
	os.MkdirAll(folder, 0o755)
	raw, _ := json.Marshal(map[string]any{"host": ep.Host, "port": ep.Port, "token": "tkn", "pid": os.Getpid(),
		"job_folder": folder, "output_folder": folder, "started_at": 1, "extra": "kept"})
	os.WriteFile(filepath.Join(folder, "engine.json"), raw, 0o644)

	rec = serve("GET", "/api/inference/status?job="+job.ID, "")
	if !strings.Contains(rec.Body.String(), `"running":true`) || !strings.Contains(rec.Body.String(), `"ready":true`) {
		t.Errorf("status with endpoint: %s", rec.Body.String())
	}
	if _, err := engines.List(ctx, true); err != nil { // refresh the 2s cache
		t.Fatal(err)
	}
	rec = serve("GET", "/api/inference/models/list?z=1&job="+job.ID+"&a=2", "")
	if got := rec.Body.String(); got != "/models/list?z=1&a=2 token=tkn" {
		t.Errorf("proxied = %q", got)
	}
}

func TestOutputsDelete(t *testing.T) {
	training := setupDB(t)
	ctx := context.Background()
	jt := "inference"
	if _, err := store.CreateJob(ctx, store.NewJob{Name: "eng", GpuIDs: "0", JobConfig: "{}", JobType: &jt}); err != nil {
		t.Fatal(err)
	}
	outputs := filepath.Join(training, "eng", "outputs")
	req := filepath.Join(outputs, "req1")
	os.MkdirAll(req, 0o755)
	img := filepath.Join(req, "a.png")
	os.WriteFile(img, []byte("x"), 0o644)
	top := filepath.Join(outputs, "b.png")
	os.WriteFile(top, []byte("x"), 0o644)
	outside := filepath.Join(training, "secret.png")
	os.WriteFile(outside, []byte("x"), 0o644)

	body, _ := json.Marshal(map[string]any{"paths": []string{img, top, outside}})
	rec := serve("POST", "/api/inference/outputs/delete", string(body))
	var res struct{ Deleted, Refused []string }
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatal(rec.Body.String())
	}
	if len(res.Deleted) != 2 || len(res.Refused) != 1 || res.Refused[0] != outside {
		t.Errorf("result = %+v", res)
	}
	if _, err := os.Stat(req); !os.IsNotExist(err) {
		t.Error("empty request folder was not removed")
	}
	if _, err := os.Stat(outputs); err != nil {
		t.Error("outputs root itself was removed")
	}
	if _, err := os.Stat(outside); err != nil {
		t.Error("file outside outputs was deleted")
	}

	// a bad body is treated as {} -> nothing to do
	rec = serve("POST", "/api/inference/outputs/delete", "not json")
	if strings.TrimSpace(rec.Body.String()) != `{"deleted":[],"refused":[]}` {
		t.Errorf("bad body: %s", rec.Body.String())
	}
}

func TestEmptyPathIs404(t *testing.T) {
	if rec := serve("GET", "/api/inference/", ""); rec.Code != 404 {
		t.Errorf("empty path: %d", rec.Code)
	}
}
