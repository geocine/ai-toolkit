package v1templates

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
	_ "modernc.org/sqlite"
)

func setup(t *testing.T) (*http.ServeMux, string) {
	t.Helper()
	root, _ := filepath.Abs(filepath.Join("..", "..", ".."))
	if _, err := os.Stat(filepath.Join(root, "ui", "src", "app", "jobs", "new", "jobConfig.ts")); err != nil {
		t.Skipf("upstream ui not found: %v", err)
	}
	paths.ToolkitRoot = root
	dir := t.TempDir()
	db, err := sql.Open("sqlite", store.FileURI(filepath.Join(dir, "db.sqlite")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if _, err := db.Exec(`CREATE TABLE "Settings" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "key" TEXT NOT NULL UNIQUE, "value" TEXT NOT NULL)`); err != nil {
		t.Fatal(err)
	}
	datasets := filepath.Join(dir, "datasets")
	os.MkdirAll(filepath.Join(datasets, "faces"), 0o755)
	db.Exec(`INSERT INTO "Settings" ("key", "value") VALUES ('DATASETS_FOLDER', ?), ('TRAINING_FOLDER', ?)`, datasets, filepath.Join(dir, "out"))
	store.DB = db
	settings.Flush()
	mux := http.NewServeMux()
	Register(mux)
	return mux, datasets
}

func TestListAndRender(t *testing.T) {
	mux, datasets := setup(t)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/templates", nil))
	var cat struct {
		Archs []struct {
			Name           string `json:"name"`
			PythonAccepted bool   `json:"python_accepted"`
		} `json:"archs"`
		DefaultConfig map[string]any `json:"default_config"`
	}
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &cat) != nil || len(cat.Archs) == 0 || cat.DefaultConfig == nil {
		t.Fatalf("list %d: %s", rec.Code, rec.Body.String()[:min(300, rec.Body.Len())])
	}

	body := `{"arch":"sdxl","name":"face_lora","datasets":["faces"],"overrides":{"config.process[0].train.steps":1200}}`
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/templates/render", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("render %d: %s", rec.Code, rec.Body)
	}
	var res struct {
		Config struct {
			Config struct {
				Name    string `json:"name"`
				Process []struct {
					TrainingFolder string `json:"training_folder"`
					Train          struct {
						Steps int `json:"steps"`
					} `json:"train"`
					Model struct {
						Arch string `json:"arch"`
					} `json:"model"`
					Datasets []struct {
						FolderPath string `json:"folder_path"`
					} `json:"datasets"`
				} `json:"process"`
			} `json:"config"`
		} `json:"config"`
		Warnings []string `json:"warnings"`
	}
	json.Unmarshal(rec.Body.Bytes(), &res)
	p0 := res.Config.Config.Process[0]
	if res.Config.Config.Name != "face_lora" || p0.Model.Arch != "sdxl" || p0.Train.Steps != 1200 ||
		p0.Datasets[0].FolderPath != filepath.Join(datasets, "faces") || !strings.HasSuffix(p0.TrainingFolder, "out") ||
		len(res.Warnings) != 0 {
		t.Fatalf("render = %s", rec.Body)
	}

	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/templates/render", strings.NewReader(`{"arch":"nope"}`)))
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), `"invalid_request"`) {
		t.Fatalf("bad arch %d: %s", rec.Code, rec.Body)
	}
}
