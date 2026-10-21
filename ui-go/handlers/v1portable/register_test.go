package v1portable

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
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

func TestRoutes(t *testing.T) {
	root := t.TempDir()
	paths.ToolkitRoot, paths.DBPath = root, filepath.Join(root, "aitk_db.db")
	paths.DefaultTrainFolder, paths.DefaultDatasetsFolder = filepath.Join(root, "output"), filepath.Join(root, "datasets")
	settings.Flush()
	db, _ := sql.Open("sqlite", store.FileURI(paths.DBPath))
	if _, err := db.Exec(ddl); err != nil {
		t.Fatal(err)
	}
	db.Close()
	if err := store.Open(paths.DBPath); err != nil {
		t.Fatal(err)
	}
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.DB.Close(); state.DB.Close() })
	cfg, _ := json.Marshal(map[string]string{"path": t.TempDir()})
	if _, err := remote.Create(t.Context(), "shared", "fs", cfg, nil); err != nil {
		t.Fatal(err)
	}
	job, err := store.CreateJob(t.Context(), store.NewJob{Name: "j", GpuIDs: "0", JobConfig: `{"config":{"process":[{}]}}`})
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	Register(mux)
	do := func(method, path, body string) (int, string) {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		return rec.Code, rec.Body.String()
	}
	cases := []struct {
		method, path, body string
		status             int
		contains           string
	}{
		{"GET", "/v1/portable/jobs", "", 400, `"invalid_request"`},
		{"GET", "/v1/portable/jobs?remote=nope", "", 404, `"not_found"`},
		{"GET", "/v1/portable/jobs?remote=shared", "", 200, `{"items":[]}`},
		{"POST", "/v1/portable/import", `{"remote":"shared"}`, 400, "remote and name are required"},
		{"POST", "/v1/portable/import", `{"remote":"shared","name":"../x"}`, 400, "cannot be used"},
		{"POST", "/v1/portable/import", `{"remote":"shared","name":"absent"}`, 404, "no portable job"},
		{"POST", "/v1/portable/import", `not json`, 400, "invalid JSON"},
		{"GET", "/v1/jobs/missing/sync", "", 404, "job not found"},
		{"POST", "/v1/jobs/" + job.ID + "/sync", "", 409, "no sync_remote"},
		{"GET", "/v1/jobs/" + job.ID + "/sync", "", 200, `"remote":null`},
	}
	for _, c := range cases {
		status, body := do(c.method, c.path, c.body)
		if status != c.status || !strings.Contains(body, c.contains) {
			t.Errorf("%s %s: %d %s (want %d containing %q)", c.method, c.path, status, body, c.status, c.contains)
		}
	}
}
