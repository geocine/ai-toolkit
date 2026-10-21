package worker

import (
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/proc"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

// On Windows jobs launch through a relay that is this executable re-run; in
// tests that is the test binary, so dispatch the relay subcommand here.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == RelayArg {
		os.Exit(RunRelay(os.Args[2:]))
	}
	os.Exit(m.Run())
}

// fakeRunPy stands in for the trainer: it reports into its Job row the way
// toolkit/ui_job_status.py does, honors the stop flag and SIGINT, and can
// crash on request (config.process[0].mode).
const fakeRunPy = `
import json, os, signal, sqlite3, sys, time
cfg = json.load(open(sys.argv[1]))
p = cfg["config"]["process"][0]
db = sqlite3.connect(p["sqlite_db_path"], timeout=30)
job = os.environ["AITK_JOB_ID"]
def upd(sql, *a):
    db.execute(sql, a + (job,)); db.commit()
env = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "IS_AI_TOOLKIT_UI", "HF_TOKEN", "PYTHONUNBUFFERED", "CUDA_DEVICE_ORDER")}
env["cwd"] = os.getcwd()
print("fake trainer started", flush=True)
if p.get("mode") == "crash":
    print("Traceback: boom", flush=True)
    sys.exit(3)
def stopped(*_):
    upd("UPDATE Job SET status = 'stopped', info = 'stopped by signal' WHERE id = ?")
    sys.exit(0)
signal.signal(signal.SIGINT, stopped)
upd("UPDATE Job SET info = ? WHERE id = ?", json.dumps(env))
for step in range(1, 600):
    upd("UPDATE Job SET step = ? WHERE id = ?", step)
    if db.execute("SELECT stop FROM Job WHERE id = ?", (job,)).fetchone()[0] == 1:
        upd("UPDATE Job SET status = 'stopped', info = 'stopped by flag' WHERE id = ?")
        sys.exit(0)
    time.sleep(0.1)
`

const ddl = `
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

func setupRoot(t *testing.T) string {
	t.Helper()
	py := paths.ResolvePython()
	if _, err := exec.LookPath(py); err != nil {
		t.Skipf("python not available (%s): %v", py, err)
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "run.py"), []byte(fakeRunPy), 0o644); err != nil {
		t.Fatal(err)
	}
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
	if err := store.UpsertSetting(context.Background(), "HF_TOKEN", "hf_fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.DB.Close() })
	return root
}

func createJob(t *testing.T, name, mode, gpu string) *store.Job {
	t.Helper()
	cfg, _ := json.Marshal(map[string]any{
		"job":    "extension",
		"config": map[string]any{"name": name, "process": []any{map[string]any{"type": "fake", "mode": mode}}},
	})
	j, err := store.CreateJob(context.Background(), store.NewJob{Name: name, GpuIDs: gpu, JobConfig: string(cfg), QueuePosition: 1000})
	if err != nil {
		t.Fatal(err)
	}
	return j
}

func waitJob(t *testing.T, id string, what string, cond func(*store.Job) bool) *store.Job {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	var j *store.Job
	for time.Now().Before(deadline) {
		var err error
		if j, err = store.GetJob(context.Background(), id); err != nil {
			t.Fatal(err)
		}
		if j != nil && cond(j) {
			return j
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s; last row: %+v", what, j)
	return nil
}

func TestJobCrashIsRecorded(t *testing.T) {
	root := setupRoot(t)
	ctx := context.Background()
	j := createJob(t, "crasher", "crash", "0")
	if err := StartJob(ctx, j.ID); err != nil {
		t.Fatal(err)
	}
	got := waitJob(t, j.ID, "error status", func(j *store.Job) bool { return j.Status == "error" })
	if runtime.GOOS == "windows" {
		if !strings.HasPrefix(got.Info, "Job process exited unexpectedly. Last log line: Traceback: boom") {
			t.Fatalf("info = %q", got.Info)
		}
	} else if got.Info != "Job process terminated with exit code 3." {
		t.Fatalf("info = %q", got.Info)
	}
	if got.Pid != nil {
		t.Fatalf("pid should be cleared, got %d", *got.Pid)
	}
	folder := filepath.Join(root, "output", "crasher")
	logText, _ := os.ReadFile(filepath.Join(folder, "log.txt"))
	for _, want := range []string{"fake trainer started", "Traceback: boom", got.Info} {
		if !strings.Contains(string(logText), want) {
			t.Fatalf("log.txt missing %q:\n%s", want, logText)
		}
	}
	cfg, _ := os.ReadFile(filepath.Join(folder, ".job_config.json"))
	if !strings.Contains(string(cfg), `"sqlite_db_path": `) || !strings.Contains(string(cfg), "\n  ") {
		t.Fatalf(".job_config.json not indented or missing sqlite_db_path:\n%s", cfg)
	}

	// a second launch rotates the previous log into logs/0_log.txt
	if err := StartJob(ctx, j.ID); err != nil {
		t.Fatal(err)
	}
	waitJob(t, j.ID, "second error", func(j *store.Job) bool { return j.Status == "error" && j.Pid == nil })
	if _, err := os.Stat(filepath.Join(folder, "logs", "0_log.txt")); err != nil {
		t.Fatalf("log not rotated: %v", err)
	}
}

func TestQueueRunsJobAndStopFlagStopsIt(t *testing.T) {
	root := setupRoot(t)
	ctx := context.Background()
	j := createJob(t, "runner", "run", "1")
	if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"status": "queued"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateQueue(ctx, "1", true); err != nil {
		t.Fatal(err)
	}

	// one worker pass picks the queued job up
	if err := processQueue(ctx); err != nil {
		t.Fatal(err)
	}
	running := waitJob(t, j.ID, "trainer steps", func(j *store.Job) bool { return j.Status == "running" && j.Step >= 3 && j.Pid != nil })

	var env map[string]string
	if err := json.Unmarshal([]byte(running.Info), &env); err != nil {
		t.Fatalf("info is not the env report: %q", running.Info)
	}
	want := map[string]string{"CUDA_VISIBLE_DEVICES": "1", "IS_AI_TOOLKIT_UI": "1", "HF_TOKEN": "hf_fake",
		"PYTHONUNBUFFERED": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
	for k, v := range want {
		if env[k] != v {
			t.Errorf("env %s = %q, want %q", k, env[k], v)
		}
	}
	if cwd, _ := filepath.EvalSymlinks(env["cwd"]); cwd != mustEval(t, root) {
		t.Errorf("job cwd = %q, want toolkit root %q", env["cwd"], root)
	}
	pidText, _ := os.ReadFile(filepath.Join(root, "output", "runner", "pid.txt"))
	if strings.TrimSpace(string(pidText)) == "" || !proc.Alive(*running.Pid) {
		t.Fatalf("pid.txt = %q, alive = %v", pidText, proc.Alive(*running.Pid))
	}

	// a second pass while it runs must not start anything else
	if err := processQueue(ctx); err != nil {
		t.Fatal(err)
	}

	// the stop route writes the stop flag; the trainer exits on its own
	if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"stop": true, "info": "Stopping job..."}); err != nil {
		t.Fatal(err)
	}
	stopped := waitJob(t, j.ID, "stopped status", func(j *store.Job) bool { return j.Status == "stopped" })
	if stopped.Info != "stopped by flag" {
		t.Fatalf("info = %q", stopped.Info)
	}
	pid := *running.Pid
	deadline := time.Now().Add(10 * time.Second)
	for proc.Alive(pid) && time.Now().Before(deadline) {
		time.Sleep(100 * time.Millisecond)
	}
	if proc.Alive(pid) {
		t.Fatalf("trainer pid %d still alive after stop", pid)
	}
	// a clean exit is not overwritten as an error
	time.Sleep(3 * time.Second)
	if final, _ := store.GetJob(ctx, j.ID); final.Status != "stopped" {
		t.Fatalf("status after exit = %q", final.Status)
	}

	// with nothing left queued, the next pass turns the queue off
	if err := processQueue(ctx); err != nil {
		t.Fatal(err)
	}
	q, _ := store.GetQueueByGPU(ctx, "1")
	if q.IsRunning {
		t.Fatal("queue should have been stopped")
	}
}

func TestSigintStopsJob(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows stops jobs through the stop flag, covered above")
	}
	setupRoot(t)
	ctx := context.Background()
	j := createJob(t, "sigint", "run", "0")
	if err := StartJob(ctx, j.ID); err != nil {
		t.Fatal(err)
	}
	running := waitJob(t, j.ID, "running", func(j *store.Job) bool { return j.Step >= 2 && j.Pid != nil })
	if err := proc.Interrupt(*running.Pid); err != nil {
		t.Fatal(err)
	}
	got := waitJob(t, j.ID, "stopped by signal", func(j *store.Job) bool { return j.Status == "stopped" })
	if got.Info != "stopped by signal" {
		t.Fatalf("info = %q", got.Info)
	}
}

func mustEval(t *testing.T, p string) string {
	t.Helper()
	r, err := filepath.EvalSymlinks(p)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
