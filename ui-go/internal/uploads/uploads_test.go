package uploads

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
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
	root, remoteDir string
}

func setup(t *testing.T) env {
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
	t.Cleanup(func() { store.DB.Close(); state.DB.Close() })
	e := env{root: root, remoteDir: filepath.Join(t.TempDir(), "remote")}
	cfg, _ := json.Marshal(map[string]string{"path": e.remoteDir})
	if _, err := remote.Create(context.Background(), "shared", "fs", cfg, nil); err != nil {
		t.Fatal(err)
	}
	return e
}

func write(t *testing.T, p string, data string) {
	t.Helper()
	os.MkdirAll(filepath.Dir(p), 0o755)
	if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func waitState(t *testing.T, id string, want ...string) *Upload {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	var u *Upload
	for time.Now().Before(deadline) {
		u, _ = Get(context.Background(), id)
		for _, w := range want {
			if u != nil && u.State == w {
				return u
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("upload %s never reached %v; last %+v", id, want, u)
	return nil
}

func TestQueueUploadsToFSRemote(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	write(t, filepath.Join(e.root, "output", "job one", "w_000000010.safetensors"), "weights")

	evs, stop := events.Subscribe(events.Filter{Types: map[string]bool{events.UploadState: true}})
	defer stop()

	u, err := Enqueue(ctx, "shared", nil, "training", "job one/w_000000010.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if u.State != Queued || u.Bytes != 7 {
		t.Fatalf("enqueued %+v", u)
	}
	// the same file while queued is not queued twice
	if dup, err := Enqueue(ctx, "shared", nil, "training", "job one/w_000000010.safetensors"); err != nil || dup.ID != u.ID {
		t.Fatalf("duplicate enqueue = %+v, %v", dup, err)
	}

	go Run(ctx)
	done := waitState(t, u.ID, Done)
	want, _ := artifacts.SHA256File(filepath.Join(e.root, "output", "job one", "w_000000010.safetensors"))
	if done.SHA256 == nil || *done.SHA256 != want || done.BytesDone != 7 || done.URL == nil || !strings.HasPrefix(*done.URL, "file://") {
		t.Fatalf("done upload %+v", done)
	}
	if b, err := os.ReadFile(filepath.Join(e.remoteDir, "training", "job one", "w_000000010.safetensors")); err != nil || string(b) != "weights" {
		t.Fatalf("remote copy %q, %v", b, err)
	}

	seen := map[string]bool{}
	timeout := time.After(2 * time.Second)
	for !seen[Done] {
		select {
		case ev := <-evs:
			var got Upload
			json.Unmarshal(ev.Data, &got)
			seen[got.State] = true
		case <-timeout:
			t.Fatalf("upload.state events seen: %v", seen)
		}
	}
	if !seen[Queued] || !seen[Running] {
		t.Fatalf("missing states in events: %v", seen)
	}

	for _, bad := range []struct{ root, path string }{
		{"training", "../outside.txt"}, {"training", "/etc/passwd"}, {"nope", "x"}, {"training", "job one/missing"}, {"training", "job one"},
	} {
		if _, err := Enqueue(ctx, "shared", nil, bad.root, bad.path); !errors.Is(err, ErrInvalid) {
			t.Errorf("Enqueue(%s, %s) = %v, want ErrInvalid", bad.root, bad.path, err)
		}
	}
	if _, err := Enqueue(ctx, "nope", nil, "training", "job one/w_000000010.safetensors"); !errors.Is(err, remote.ErrNotFound) {
		t.Errorf("unknown remote = %v", err)
	}
}

func TestFailureAndRetry(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	write(t, filepath.Join(e.root, "datasets", "set", "a.png"), "png")
	// the remote folder is a file, so copying into it fails
	write(t, e.remoteDir, "not a folder")

	go Run(ctx)
	u, err := Enqueue(ctx, "shared", nil, "datasets", "set/a.png")
	if err != nil {
		t.Fatal(err)
	}
	failed := waitState(t, u.ID, Failed)
	if failed.Error == nil || *failed.Error == "" {
		t.Fatalf("failed upload has no error: %+v", failed)
	}
	os.Remove(e.remoteDir)
	if _, err := Retry(ctx, u.ID); err != nil {
		t.Fatal(err)
	}
	ok := waitState(t, u.ID, Done)
	if ok.Error != nil {
		t.Fatalf("error not cleared: %+v", ok)
	}
	if b, _ := os.ReadFile(filepath.Join(e.remoteDir, "datasets", "set", "a.png")); string(b) != "png" {
		t.Fatalf("remote copy %q", b)
	}
}

func TestRetryWhileQueuedIsBusy(t *testing.T) {
	e := setup(t)
	write(t, filepath.Join(e.root, "data", "x.bin"), "x")
	u, err := Enqueue(context.Background(), "shared", nil, "data", "x.bin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Retry(context.Background(), u.ID); !errors.Is(err, ErrBusy) {
		t.Fatalf("retry of a queued upload = %v", err)
	}
	if got, err := Retry(context.Background(), "upl_missing"); got != nil || err != nil {
		t.Fatalf("retry of a missing upload = %+v, %v", got, err)
	}
}

func TestRunningUploadsAreRequeuedAtBoot(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	write(t, filepath.Join(e.root, "data", "y.bin"), "yy")
	u, err := Enqueue(ctx, "shared", nil, "data", "y.bin")
	if err != nil {
		t.Fatal(err)
	}
	// as if a previous process died mid-upload
	state.DB.Exec(`UPDATE uploads SET state = 'running', bytes_done = 1 WHERE id = ?`, u.ID)
	go Run(ctx)
	waitState(t, u.ID, Done)
}

func TestAutoUploadOnCheckpoint(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	withPolicy, err := store.CreateJob(ctx, store.NewJob{Name: "auto job", GpuIDs: "0", JobConfig: "{}"})
	if err != nil {
		t.Fatal(err)
	}
	without, err := store.CreateJob(ctx, store.NewJob{Name: "plain job", GpuIDs: "0", JobConfig: "{}"})
	if err != nil {
		t.Fatal(err)
	}
	ref := "shared"
	if err := state.SetPolicy(ctx, withPolicy.ID, state.Policy{AutoUploadRemote: &ref}); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(e.root, "output", "auto job", "auto job_000000020.safetensors"), "ckpt20")
	write(t, filepath.Join(e.root, "output", "plain job", "plain job_000000020.safetensors"), "ckpt20")

	go Run(ctx)
	go AutoUpload(ctx)
	time.Sleep(200 * time.Millisecond) // let AutoUpload subscribe

	step := 20
	events.Emit(ctx, events.JobCheckpoint, without.ID, artifacts.Checkpoint{Name: "plain job_000000020.safetensors", Step: &step})
	events.Emit(ctx, events.JobCheckpoint, withPolicy.ID, artifacts.Checkpoint{Name: "auto job_000000020.safetensors", Step: &step})

	deadline := time.Now().Add(10 * time.Second)
	var list []*Upload
	for time.Now().Before(deadline) {
		list, _ = List(ctx, withPolicy.ID, Done)
		if len(list) == 1 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if len(list) != 1 || list[0].Path != "auto job/auto job_000000020.safetensors" || list[0].Root != "training" {
		t.Fatalf("auto uploads = %+v", list)
	}
	if b, err := os.ReadFile(filepath.Join(e.remoteDir, "jobs", "auto job", "auto job_000000020.safetensors")); err != nil || string(b) != "ckpt20" {
		t.Fatalf("remote copy under jobs/: %q, %v", b, err)
	}
	if other, _ := List(ctx, without.ID, ""); len(other) != 0 {
		t.Fatalf("job without a policy was uploaded: %+v", other)
	}
}
