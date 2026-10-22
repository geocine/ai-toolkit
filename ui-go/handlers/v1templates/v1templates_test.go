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

func TestFieldsCheckExamples(t *testing.T) {
	mux, _ := setup(t)
	get := func(path string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
		return rec
	}
	rec := get("/v1/templates/fields?section=train")
	var ref struct {
		Sections []struct {
			Path string `json:"path"`
		} `json:"sections"`
	}
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &ref) != nil || len(ref.Sections) < 2 || ref.Sections[0].Path != "train" {
		t.Fatalf("fields %d: %.300s", rec.Code, rec.Body)
	}
	for _, s := range ref.Sections {
		if s.Path != "train" && !strings.HasPrefix(s.Path, "train.") {
			t.Errorf("section=train returned %s", s.Path)
		}
	}
	if rec := get("/v1/templates/fields?section=nope"); rec.Code != 404 {
		t.Errorf("unknown section %d", rec.Code)
	}

	body := `{"config":{"config":{"process":[{"type":"sd_trainer","loss_type":"mse","train":{"steps":10}}]}}}`
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/templates/check", strings.NewReader(body)))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"fix":"it is read in train"`) {
		t.Fatalf("check %d: %s", rec.Code, rec.Body)
	}
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/templates/check", strings.NewReader(`{}`)))
	if rec.Code != 400 {
		t.Errorf("check without config %d", rec.Code)
	}

	if rec := get("/v1/templates/examples"); rec.Code != 200 || !strings.Contains(rec.Body.String(), `"train_lora_flux_24gb"`) {
		t.Fatalf("examples %d: %.300s", rec.Code, rec.Body)
	}
	if rec := get("/v1/templates/examples/train_lora_flux_24gb"); rec.Code != 200 || !strings.Contains(rec.Body.String(), `"yaml"`) {
		t.Fatalf("example %d", rec.Code)
	}
	if rec := get("/v1/templates/examples/nope"); rec.Code != 404 {
		t.Errorf("unknown example %d", rec.Code)
	}
}
