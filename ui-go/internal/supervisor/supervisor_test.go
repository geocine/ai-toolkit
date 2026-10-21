package supervisor

import (
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
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

func setup(t *testing.T) string {
	t.Helper()
	root := testutil.TempDir(t)
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
	settings.Flush()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.DB.Close(); state.DB.Close() })
	return root
}

func newSup() *Supervisor {
	return &Supervisor{prev: map[string]*snapshot{}, folders: map[string]*folderState{}, requeued: map[string]bool{}}
}

// deadPid returns the pid of a process that has already exited.
func deadPid(t *testing.T) int {
	t.Helper()
	name, args := "sh", []string{"-c", "exit 0"}
	if runtime.GOOS == "windows" {
		name, args = "cmd", []string{"/c", "exit 0"}
	}
	cmd := exec.Command(name, args...)
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
	return cmd.Process.Pid
}

func mkJob(t *testing.T, name string) *store.Job {
	t.Helper()
	cfg := `{"job":"extension","config":{"name":"` + name + `","process":[{"type":"x"}]}}`
	j, err := store.CreateJob(context.Background(), store.NewJob{Name: name, GpuIDs: "0", JobConfig: cfg})
	if err != nil {
		t.Fatal(err)
	}
	return j
}

func collect(f events.Filter) (func() []events.Event, func()) {
	ch, cancel := events.Subscribe(f)
	var got []events.Event
	drain := func() []events.Event {
		for {
			select {
			case e := <-ch:
				got = append(got, e)
			case <-time.After(50 * time.Millisecond):
				return got
			}
		}
	}
	return drain, cancel
}

func types(evs []events.Event) []string {
	out := make([]string, len(evs))
	for i, e := range evs {
		out[i] = e.Type
	}
	return out
}

func TestBootReconcileMarksDeadTrainerInterruptedAndResumes(t *testing.T) {
	setup(t)
	ctx := context.Background()
	j := mkJob(t, "orphan")
	pid := deadPid(t)
	if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"status": "running", "step": 120, "info": "Training", "pid": pid}); err != nil {
		t.Fatal(err)
	}
	on := true
	if err := state.SetPolicy(ctx, j.ID, state.Policy{AutoResume: &on}); err != nil {
		t.Fatal(err)
	}

	drain, cancel := collect(events.Filter{JobID: j.ID})
	defer cancel()
	s := newSup()
	// boot: opens the missing attempt, finds the trainer dead and marks it
	if err := s.pass(ctx, true); err != nil {
		t.Fatal(err)
	}
	got, _ := store.GetJob(ctx, j.ID)
	if got.Status != "error" || got.Pid != nil || got.Info[:12] != "Interrupted:" {
		t.Fatalf("after boot: status=%s pid=%v info=%q", got.Status, got.Pid, got.Info)
	}
	// next pass sees the transition: closes the attempt and auto-resumes
	if err := s.pass(ctx, false); err != nil {
		t.Fatal(err)
	}
	got, _ = store.GetJob(ctx, j.ID)
	if got.Status != "queued" || got.Info != "Auto-resuming after interruption" {
		t.Fatalf("after resume: status=%s info=%q", got.Status, got.Info)
	}
	q, _ := store.GetQueueByGPU(ctx, "0")
	if q == nil || !q.IsRunning {
		t.Fatalf("queue should be running: %+v", q)
	}
	atts, _ := state.Attempts(ctx, j.ID)
	if len(atts) != 1 || atts[0].EndState == nil || *atts[0].EndState != "interrupted" || atts[0].StartStep != 120 {
		t.Fatalf("attempts: %+v", atts)
	}

	// the worker would start it: the next attempt is marked auto
	store.UpdateJob(ctx, j.ID, store.Fields{"status": "running", "info": "Starting job..."})
	s.pass(ctx, false)
	atts, _ = state.Attempts(ctx, j.ID)
	if len(atts) != 2 || !atts[1].Auto || atts[1].EndedAt != nil {
		t.Fatalf("second attempt: %+v", atts[len(atts)-1])
	}
	evTypes := types(drain())
	want := map[string]bool{events.JobState: false, events.JobAttempt: false}
	for _, ty := range evTypes {
		if _, ok := want[ty]; ok {
			want[ty] = true
		}
	}
	for ty, seen := range want {
		if !seen {
			t.Errorf("missing %s event; got %v", ty, evTypes)
		}
	}
}

func TestAutoResumeStopsAfterMax(t *testing.T) {
	setup(t)
	ctx := context.Background()
	j := mkJob(t, "flaky")
	on, max := true, 1
	state.SetPolicy(ctx, j.ID, state.Policy{AutoResume: &on, MaxAutoResumes: &max})
	s := newSup()
	s.pass(ctx, true)
	for round := 1; round <= 2; round++ {
		store.UpdateJob(ctx, j.ID, store.Fields{"status": "running", "info": "Training", "pid": deadPid(t)})
		s.pass(ctx, false) // opens attempt
		s.lastAlive = time.Time{}
		s.pass(ctx, false) // liveness marks it gone
		s.pass(ctx, false) // transition: end attempt, maybe resume
		got, _ := store.GetJob(ctx, j.ID)
		if round == 1 && got.Status != "queued" {
			t.Fatalf("round 1 should resume, status %s", got.Status)
		}
		if round == 2 && got.Status != "error" {
			t.Fatalf("round 2 exceeds max_auto_resumes and must stay failed, status %s", got.Status)
		}
	}
}

func TestOwnErrorIsNotResumed(t *testing.T) {
	setup(t)
	ctx := context.Background()
	j := mkJob(t, "broken")
	on := true
	state.SetPolicy(ctx, j.ID, state.Policy{AutoResume: &on})
	s := newSup()
	s.pass(ctx, true)
	store.UpdateJob(ctx, j.ID, store.Fields{"status": "running", "info": "Training"})
	s.pass(ctx, false)
	// the trainer reports its own error (a traceback): failed, not interrupted
	store.UpdateJob(ctx, j.ID, store.Fields{"status": "error", "info": "Unknown model arch 'x'"})
	s.pass(ctx, false)
	got, _ := store.GetJob(ctx, j.ID)
	if got.Status != "error" {
		t.Fatalf("status %s", got.Status)
	}
	a, _ := state.LastAttempt(ctx, j.ID)
	if a.EndState == nil || *a.EndState != "failed" {
		t.Fatalf("attempt end %+v", a)
	}
}

func TestStoppingJobWithDeadTrainerBecomesStopped(t *testing.T) {
	setup(t)
	ctx := context.Background()
	j := mkJob(t, "stopper")
	s := newSup()
	s.pass(ctx, true)
	store.UpdateJob(ctx, j.ID, store.Fields{"status": "running", "stop": true, "info": "Stopping job...", "pid": deadPid(t)})
	s.pass(ctx, false)
	s.lastAlive = time.Time{}
	s.pass(ctx, false)
	got, _ := store.GetJob(ctx, j.ID)
	if got.Status != "stopped" {
		t.Fatalf("status %s info %q", got.Status, got.Info)
	}
}

func TestCheckpointAndSampleEvents(t *testing.T) {
	root := setup(t)
	ctx := context.Background()
	j := mkJob(t, "artist")
	folder := filepath.Join(root, "output", "artist")
	os.MkdirAll(filepath.Join(folder, "samples"), 0o755)
	// present at boot: never reported
	os.WriteFile(filepath.Join(folder, "artist_000000010.safetensors"), []byte("old"), 0o644)

	s := newSup()
	s.pass(ctx, true)
	store.UpdateJob(ctx, j.ID, store.Fields{"status": "running", "info": "Training", "pid": os.Getpid()})
	drain, cancel := collect(events.Filter{JobID: j.ID, Types: map[string]bool{events.JobCheckpoint: true, events.JobSample: true}})
	defer cancel()

	ck := filepath.Join(folder, "artist_000000020.safetensors")
	os.WriteFile(ck, []byte("partial"), 0o644)
	s.lastScan = time.Time{}
	s.pass(ctx, false) // first sight: not reported yet
	os.WriteFile(ck, []byte("partial-and-more"), 0o644)
	s.lastScan = time.Time{}
	s.pass(ctx, false) // size changed: still writing
	s.lastScan = time.Time{}
	s.pass(ctx, false) // size held: reported
	os.WriteFile(filepath.Join(folder, "samples", "1700000000000__000000020_0.jpg"), []byte("img"), 0o644)
	s.lastScan = time.Time{}
	s.pass(ctx, false)

	evs := drain()
	if len(evs) != 2 || evs[0].Type != events.JobCheckpoint || evs[1].Type != events.JobSample {
		t.Fatalf("events: %v", types(evs))
	}
	var c struct {
		Name string
		Step *int
		Size int64
	}
	json.Unmarshal(evs[0].Data, &c)
	if c.Name != "artist_000000020.safetensors" || c.Step == nil || *c.Step != 20 || c.Size != int64(len("partial-and-more")) {
		t.Fatalf("checkpoint event %s", evs[0].Data)
	}
	stored, _ := events.List(ctx, 0, events.Filter{JobID: j.ID, Types: map[string]bool{events.JobCheckpoint: true}}, 10)
	if len(stored) != 1 {
		t.Fatalf("stored checkpoint events: %d", len(stored))
	}
}
