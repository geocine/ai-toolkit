package v1unattended

import (
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/notify"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
)

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
CREATE UNIQUE INDEX "Job_name_key" ON "Job"("name");
`

func setup(t *testing.T) *httptest.Server {
	t.Helper()
	root := testutil.TempDir(t)
	dbPath := filepath.Join(root, "aitk_db.db")
	db, err := sql.Open("sqlite", store.FileURI(dbPath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(ddl); err != nil {
		t.Fatal(err)
	}
	db.Close()
	paths.ToolkitRoot, paths.DBPath = root, dbPath
	paths.DefaultTrainFolder = filepath.Join(root, "output")
	settings.Flush()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AITK_WHEN_DONE", "destroy")
	t.Setenv("AITK_WHEN_DONE_DELAY", "10m")
	t.Setenv("AITK_WHEN_DONE_DRY_RUN", "1")
	t.Setenv("AITK_NOTIFY", "")
	ctx, cancel := context.WithCancel(context.Background())
	go lifecycle.Start(ctx)
	go notify.Run(ctx)
	mux := http.NewServeMux()
	Register(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(func() { cancel(); srv.Close(); time.Sleep(50 * time.Millisecond); store.DB.Close(); state.DB.Close() })
	// wait for the manager
	for i := 0; i < 100; i++ {
		if _, err := lifecycle.Get(); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	return srv
}

func do(t *testing.T, srv *httptest.Server, method, path, body string) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest(method, srv.URL+path, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var m map[string]any
	json.Unmarshal(b, &m)
	return resp.StatusCode, m
}

func TestLifecycleAPI(t *testing.T) {
	srv := setup(t)
	code, s := do(t, srv, "GET", "/v1/lifecycle", "")
	if code != 200 || s["mode"] != "destroy" || s["dry_run"] != true || s["delay_s"] != 600.0 || s["state"] != "waiting" {
		t.Fatalf("%d %v", code, s)
	}
	// nothing to cancel or extend yet
	if code, _ := do(t, srv, "POST", "/v1/lifecycle/cancel", ""); code != 409 {
		t.Fatal(code)
	}
	if code, _ := do(t, srv, "POST", "/v1/lifecycle/extend", `{"minutes": 5}`); code != 409 {
		t.Fatal(code)
	}
	if code, _ := do(t, srv, "POST", "/v1/lifecycle/extend", `{}`); code != 400 {
		t.Fatal(code)
	}
	if code, _ := do(t, srv, "PATCH", "/v1/lifecycle", `{"mode": "explode"}`); code != 400 {
		t.Fatal(code)
	}
	code, s = do(t, srv, "PATCH", "/v1/lifecycle", `{"mode": "stop", "delay_s": 120, "on": "any"}`)
	if code != 200 || s["mode"] != "stop" || s["delay_s"] != 120.0 || s["on"] != "any" {
		t.Fatalf("%d %v", code, s)
	}
	// /now with no jobs: verify (nothing to keep) and act, dry run
	if code, s = do(t, srv, "POST", "/v1/lifecycle/now", ""); code != 200 {
		t.Fatalf("%d %v", code, s)
	}
	deadline := time.Now().Add(10 * time.Second)
	for s["state"] != "done" && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
		_, s = do(t, srv, "GET", "/v1/lifecycle", "")
	}
	if s["state"] != "done" || !strings.Contains(s["reason"].(string), "dry run") {
		t.Fatalf("%v", s)
	}
}

func TestNotifyAPI(t *testing.T) {
	srv := setup(t)
	var mu sync.Mutex
	var bodies []string
	hook := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		mu.Unlock()
	}))
	defer hook.Close()

	if code, _ := do(t, srv, "POST", "/v1/notify/test", ""); code != 409 {
		t.Fatal("test without sinks:", code)
	}
	if code, _ := do(t, srv, "POST", "/v1/notify/sinks", `{"url": "ftp://x"}`); code != 400 {
		t.Fatal(code)
	}
	secretURL := "webhook+" + hook.URL + "/hook?token=very-secret"
	code, sink := do(t, srv, "POST", "/v1/notify/sinks", `{"url": "`+secretURL+`"}`)
	if code != 201 || sink["kind"] != "webhook" || sink["source"] != "api" {
		t.Fatalf("%d %v", code, sink)
	}
	if code, _ := do(t, srv, "POST", "/v1/notify/sinks", `{"url": "`+secretURL+`"}`); code != 409 {
		t.Fatal("duplicate:", code)
	}
	code, v := do(t, srv, "GET", "/v1/notify", "")
	raw, _ := json.Marshal(v)
	if code != 200 || strings.Contains(string(raw), "very-secret") || len(v["sinks"].([]any)) != 1 || v["events"] != nil {
		t.Fatalf("%d %s", code, raw)
	}
	code, v = do(t, srv, "PATCH", "/v1/notify", `{"events": ["completed", "lifecycle"], "public_url": "https://pod.example", "sample_every_s": 60}`)
	if code != 200 || len(v["events"].([]any)) != 2 || v["public_url"] != "https://pod.example" || v["sample_every_s"] != 60.0 {
		t.Fatalf("%d %v", code, v)
	}
	if code, _ := do(t, srv, "PATCH", "/v1/notify", `{"events": ["nope"]}`); code != 400 {
		t.Fatal(code)
	}
	code, res := do(t, srv, "POST", "/v1/notify/test", `{"message": "hello phone"}`)
	items := res["items"].([]any)
	if code != 200 || len(items) != 1 || items[0].(map[string]any)["ok"] != true {
		t.Fatalf("%d %v", code, res)
	}
	mu.Lock()
	if len(bodies) != 1 || !strings.Contains(bodies[0], "hello phone") {
		t.Fatalf("%v", bodies)
	}
	mu.Unlock()
	if code, _ := do(t, srv, "DELETE", "/v1/notify/sinks/"+sink["id"].(string), ""); code != 204 {
		t.Fatal(code)
	}
	if code, _ := do(t, srv, "DELETE", "/v1/notify/sinks/"+sink["id"].(string), ""); code != 404 {
		t.Fatal(code)
	}
}
