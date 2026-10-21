package v1remotes

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/uploads"
)

const upstreamDDL = `
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

type harness struct {
	t    *testing.T
	srv  *httptest.Server
	root string
}

func setup(t *testing.T) *harness {
	t.Helper()
	root := t.TempDir()
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
	paths.DefaultTrainFolder = filepath.Join(root, "output")
	paths.DefaultDatasetsFolder = filepath.Join(root, "datasets")
	paths.DefaultDataRoot = filepath.Join(root, "data")
	paths.DefaultModelsFolder = filepath.Join(root, "models")
	settings.Flush()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	Register(mux)
	h := &harness{t: t, srv: httptest.NewServer(mux), root: root}
	t.Cleanup(func() { h.srv.Close(); store.DB.Close(); state.DB.Close() })
	return h
}

func (h *harness) do(method, path, body string) (int, map[string]any, string) {
	h.t.Helper()
	req, _ := http.NewRequest(method, h.srv.URL+path, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw := new(strings.Builder)
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		raw.Write(buf[:n])
		if err != nil {
			break
		}
	}
	var m map[string]any
	json.Unmarshal([]byte(raw.String()), &m)
	return resp.StatusCode, m, raw.String()
}

func errCode(m map[string]any) string {
	e, _ := m["error"].(map[string]any)
	c, _ := e["code"].(string)
	return c
}

func TestRemotesCRUD(t *testing.T) {
	h := setup(t)
	remoteDir := filepath.ToSlash(filepath.Join(t.TempDir(), "share"))

	for _, c := range []struct{ body, code string }{
		{`{"name": "", "kind": "fs"}`, "invalid_request"},
		{`{"name": "bad name!", "kind": "fs"}`, "invalid_request"},
		{`{"name": "x", "kind": "ftp"}`, "invalid_request"},
		{`{"name": "x", "kind": "fs", "config": {}}`, "invalid_request"},
		{`{"name": "x", "kind": "s3", "config": {"endpoint": "https://e", "bucket": "b"}}`, "invalid_request"},
		{`not json`, "invalid_request"},
	} {
		if status, m, raw := h.do("POST", "/v1/remotes", c.body); status != 400 || errCode(m) != c.code {
			t.Errorf("POST %s = %d %s", c.body, status, raw)
		}
	}

	status, m, raw := h.do("POST", "/v1/remotes", `{"name": "share", "kind": "fs", "config": {"path": "`+remoteDir+`"}}`)
	if status != 201 || m["kind"] != "fs" || !strings.HasPrefix(m["id"].(string), "rem_") {
		t.Fatalf("create fs = %d %s", status, raw)
	}
	fsID := m["id"].(string)
	if status, m, _ := h.do("POST", "/v1/remotes", `{"name": "share", "kind": "fs", "config": {"path": "/x"}}`); status != 409 || errCode(m) != "conflict" {
		t.Fatalf("duplicate = %d", status)
	}

	status, _, raw = h.do("POST", "/v1/remotes", `{"name": "r2", "kind": "s3",
		"config": {"endpoint": "https://acct.r2.cloudflarestorage.com", "bucket": "loras", "region": "auto"},
		"credentials": {"access_key_id": "AKID-VISIBLE", "secret_access_key": "TOP-SECRET-VALUE"}}`)
	if status != 201 || strings.Contains(raw, "TOP-SECRET-VALUE") || strings.Contains(raw, "AKID-VISIBLE") {
		t.Fatalf("create s3 = %d %s", status, raw)
	}

	status, m, raw = h.do("GET", "/v1/remotes", "")
	items, _ := m["items"].([]any)
	if status != 200 || len(items) != 2 || strings.Contains(raw, "TOP-SECRET-VALUE") || strings.Contains(raw, "credentials") {
		t.Fatalf("list = %d %s", status, raw)
	}

	if status, m, raw := h.do("POST", "/v1/remotes/"+fsID+"/test", ""); status != 200 || m["ok"] != true {
		t.Fatalf("test fs = %d %s", status, raw)
	}
	if status, _, _ := h.do("POST", "/v1/remotes/nope/test", ""); status != 404 {
		t.Fatalf("test missing = %d", status)
	}

	if status, _, _ := h.do("DELETE", "/v1/remotes/"+fsID, ""); status != 204 {
		t.Fatalf("delete = %d", status)
	}
	if status, m, _ := h.do("DELETE", "/v1/remotes/"+fsID, ""); status != 404 || errCode(m) != "not_found" {
		t.Fatalf("delete again = %d", status)
	}
}

func TestUploadsAPI(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	remoteDir := filepath.ToSlash(filepath.Join(t.TempDir(), "share"))
	if status, _, raw := h.do("POST", "/v1/remotes", `{"name": "share", "kind": "fs", "config": {"path": "`+remoteDir+`"}}`); status != 201 {
		t.Fatalf("create remote = %d %s", status, raw)
	}
	job, err := store.CreateJob(ctx, store.NewJob{Name: "my job", GpuIDs: "0", JobConfig: "{}"})
	if err != nil {
		t.Fatal(err)
	}
	os.MkdirAll(filepath.Join(h.root, "output", "my job"), 0o755)
	os.WriteFile(filepath.Join(h.root, "output", "my job", "my job_000000010.safetensors"), []byte("w10"), 0o644)
	os.MkdirAll(filepath.Join(h.root, "datasets", "set"), 0o755)
	os.WriteFile(filepath.Join(h.root, "datasets", "set", "a.png"), []byte("png"), 0o644)

	for _, c := range []struct {
		body   string
		status int
		code   string
	}{
		{`{"root": "datasets", "path": "set/a.png"}`, 400, "invalid_request"},
		{`{"remote": "share"}`, 400, "invalid_request"},
		{`{"remote": "share", "root": "etc", "path": "x"}`, 400, "invalid_request"},
		{`{"remote": "share", "root": "datasets", "path": "../aitk_db.db"}`, 400, "invalid_request"},
		{`{"remote": "share", "root": "datasets", "path": "set/missing.png"}`, 400, "invalid_request"},
		{`{"remote": "nope", "root": "datasets", "path": "set/a.png"}`, 404, "not_found"},
		{`{"remote": "share", "job_id": "` + job.ID + `"}`, 400, "invalid_request"},
		{`{"remote": "share", "job_id": "missing", "checkpoint": "x.safetensors"}`, 404, "not_found"},
		{`{"remote": "share", "job_id": "` + job.ID + `", "checkpoint": "../../aitk_db.db"}`, 400, "invalid_request"},
		{`{"remote": "share", "job_id": "` + job.ID + `", "checkpoint": "nope.safetensors"}`, 404, "not_found"},
	} {
		if status, m, raw := h.do("POST", "/v1/uploads", c.body); status != c.status || errCode(m) != c.code {
			t.Errorf("POST /v1/uploads %s = %d %s", c.body, status, raw)
		}
	}

	status, m, raw := h.do("POST", "/v1/uploads", `{"remote": "share", "job_id": "`+job.ID+`", "checkpoint": "my job_000000010.safetensors"}`)
	if status != 202 || m["state"] != "queued" || m["root"] != "training" || m["path"] != "my job/my job_000000010.safetensors" || m["job_id"] != job.ID {
		t.Fatalf("job upload = %d %s", status, raw)
	}
	jobUpload := m["id"].(string)
	status, m, raw = h.do("POST", "/v1/uploads", `{"remote": "share", "root": "datasets", "path": "set/a.png"}`)
	if status != 202 || m["job_id"] != nil {
		t.Fatalf("path upload = %d %s", status, raw)
	}

	if status, m, _ := h.do("POST", "/v1/uploads/"+jobUpload+"/retry", ""); status != 409 || errCode(m) != "conflict" {
		t.Fatalf("retry while queued = %d", status)
	}
	if status, _, _ := h.do("POST", "/v1/uploads/upl_nope/retry", ""); status != 404 {
		t.Fatalf("retry missing = %d", status)
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go uploads.Run(runCtx)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, m, _ := h.do("GET", "/v1/uploads/"+jobUpload, ""); m["state"] == "done" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	status, m, raw = h.do("GET", "/v1/uploads/"+jobUpload, "")
	if status != 200 || m["state"] != "done" || m["sha256"] == nil || m["bytes_done"].(float64) != 3 {
		t.Fatalf("job upload after run = %d %s", status, raw)
	}
	if b, _ := os.ReadFile(filepath.Join(remoteDir, "jobs", "my job", "my job_000000010.safetensors")); string(b) != "w10" {
		t.Fatalf("remote copy %q", b)
	}

	_, m, _ = h.do("GET", "/v1/uploads?job_id="+job.ID, "")
	if items := m["items"].([]any); len(items) != 1 {
		t.Fatalf("filter by job = %v", items)
	}
	_, m, _ = h.do("GET", "/v1/uploads", "")
	if items := m["items"].([]any); len(items) != 2 {
		t.Fatalf("all uploads = %v", items)
	}
	if status, _, _ := h.do("GET", "/v1/uploads?state=bogus", ""); status != 400 {
		t.Fatalf("bad state filter = %d", status)
	}
	if status, _, _ := h.do("GET", "/v1/uploads/upl_nope", ""); status != 404 {
		t.Fatalf("missing upload = %d", status)
	}
	if status, m, raw := h.do("POST", "/v1/uploads/"+jobUpload+"/retry", ""); status != 200 || m["state"] != "queued" {
		t.Fatalf("retry done = %d %s", status, raw)
	}
}
