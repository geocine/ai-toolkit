package lifecycle

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
	"github.com/geocine/aitoolkit/internal/uploads"
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

type env_ struct {
	root, shared string
}

func setup(t *testing.T) *env_ {
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
	paths.DefaultDatasetsFolder = filepath.Join(root, "datasets")
	paths.DefaultDataRoot = filepath.Join(root, "data")
	settings.Flush()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.DB.Close(); state.DB.Close() })
	shared := filepath.Join(root, "shared")
	os.MkdirAll(shared, 0o755)
	cfg, _ := json.Marshal(map[string]string{"path": shared})
	if _, err := remote.Create(context.Background(), "shared", "fs", cfg, nil); err != nil {
		t.Fatal(err)
	}
	pollEvery = 20 * time.Millisecond
	return &env_{root: root, shared: shared}
}

func write(t *testing.T, p, data string) {
	t.Helper()
	os.MkdirAll(filepath.Dir(p), 0o755)
	if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

// finishedJob creates a job that ran and ended with status (and, with
// checkpoint, a final model file in its folder).
func (e *env_) finishedJob(t *testing.T, name, status, info string, checkpoint bool) *store.Job {
	t.Helper()
	ctx := context.Background()
	cfg, _ := json.Marshal(map[string]any{"job": "extension", "config": map[string]any{"name": name,
		"process": []any{map[string]any{"type": "diffusion_trainer", "training_folder": filepath.Join(e.root, "output")}}}})
	j, err := store.CreateJob(ctx, store.NewJob{Name: name, GpuIDs: "0", JobConfig: string(cfg)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := state.StartAttempt(ctx, j.ID, 0, false); err != nil {
		t.Fatal(err)
	}
	if checkpoint {
		write(t, filepath.Join(e.root, "output", name, name+".safetensors"), "final weights of "+name)
	}
	if _, err := state.EndAttempt(ctx, j.ID, status, info, 20); err != nil {
		t.Fatal(err)
	}
	j, err = store.UpdateJob(ctx, j.ID, store.Fields{"status": status, "info": info, "step": 20, "total_steps": 20})
	if err != nil {
		t.Fatal(err)
	}
	return j
}

// fakeProvider records calls.
type fakeProvider struct {
	mu    sync.Mutex
	calls []string
	err   error
}

func (f *fakeProvider) Name() string { return "fake" }
func (f *fakeProvider) Stop(context.Context) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "stop")
	return "stopped", f.err
}
func (f *fakeProvider) Destroy(context.Context) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "destroy")
	return "destroyed", f.err
}
func (f *fakeProvider) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.calls...)
}

func startManager(t *testing.T, c Config, p Provider) *Manager {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	c, err := normalize(c)
	if err != nil {
		t.Fatal(err)
	}
	m := newManager(ctx, c, p)
	mgrMu.Lock()
	mgr = m
	mgrMu.Unlock()
	return m
}

func waitState(t *testing.T, m *Manager, want string) Status {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for {
		m.evaluate()
		s := m.Snapshot()
		if s.State == want {
			return s
		}
		if time.Now().After(deadline) {
			t.Fatalf("state %s (%s), want %s", s.State, s.Reason, want)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestDestroyAfterVerifiedSync(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	j := e.finishedJob(t, "lora_a", "completed", "Job completed", true)
	rn := "shared"
	state.SetPolicy(ctx, j.ID, state.Policy{SyncRemote: &rn})

	p := &fakeProvider{}
	m := startManager(t, Config{Mode: ModeDestroy, DelayS: 0, DryRun: true}, p)
	s := waitState(t, m, StateDone)
	if !strings.Contains(s.Reason, "dry run") || len(p.Calls()) != 0 {
		t.Fatalf("dry run called the provider or said %q: %v", s.Reason, p.Calls())
	}
	if len(s.Results) != 1 || !s.Results[0].Safe || s.Results[0].Via != "sync:shared" {
		t.Fatalf("results: %+v", s.Results)
	}
	// the saver pushed the final file itself (no sync had run)
	if _, err := os.Stat(filepath.Join(e.shared, "jobs", "lora_a", "manifest.json")); err != nil {
		t.Fatal(err)
	}
	// done stays done for that ending
	m.evaluate()
	if m.Snapshot().State != StateDone {
		t.Fatal("re-armed after done")
	}

	// for real (fake provider): a config change decides afresh
	f := false
	if _, err := Update(ctx, Patch{DryRun: &f}); err != nil {
		t.Fatal(err)
	}
	waitState(t, m, StateDone)
	if c := p.Calls(); len(c) != 1 || c[0] != "destroy" {
		t.Fatalf("provider calls %v", c)
	}
}

func TestHeldWhenResultsOnlyHere(t *testing.T) {
	e := setup(t)
	e.finishedJob(t, "lora_b", "completed", "Job completed", true)
	p := &fakeProvider{}
	m := startManager(t, Config{Mode: ModeStop}, p)
	s := waitState(t, m, StateHeld)
	if !strings.Contains(s.Reason, "results only on this server") || len(p.Calls()) != 0 {
		t.Fatalf("%q %v", s.Reason, p.Calls())
	}
	if len(s.Results) != 1 || s.Results[0].Safe {
		t.Fatalf("%+v", s.Results)
	}
}

func TestUploadSaverReplacesMismatch(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	j := e.finishedJob(t, "lora_c", "completed", "Job completed", true)
	// the upload remote holds an object of the wrong size: not proof
	rn := "shared"
	state.SetPolicy(ctx, j.ID, state.Policy{AutoUploadRemote: &rn})
	write(t, filepath.Join(e.shared, "jobs", "lora_c", "lora_c.safetensors"), "short")
	uctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go uploads.Run(uctx)
	res := verifyJob(ctx, j)
	// the upload saver re-uploads the newest file, so the second Stat matches
	if !res.Safe || res.Via != "upload:shared" {
		t.Fatalf("%+v", res)
	}
	o, _ := os.Stat(filepath.Join(e.shared, "jobs", "lora_c", "lora_c.safetensors"))
	if o.Size() != int64(len("final weights of lora_c")) {
		t.Fatalf("remote size %d", o.Size())
	}
}

func TestCountdownCancelExtendAndNewJob(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	e.finishedJob(t, "lora_d", "completed", "Job completed", false)
	p := &fakeProvider{}
	m := startManager(t, Config{Mode: ModeDestroy, DelayS: 600}, p)
	m.evaluate()
	s := m.Snapshot()
	if s.State != StateCountdown || s.Deadline == nil || s.JobName == nil || *s.JobName != "lora_d" {
		t.Fatalf("%+v", s)
	}
	first := *s.Deadline
	if s, err := Extend(5 * time.Minute); err != nil || !s.Deadline.Equal(first.Add(5*time.Minute)) {
		t.Fatalf("extend: %v %+v", err, s)
	}
	if _, err := Now(); err != nil {
		t.Fatal(err)
	}
	// /now acts at once; no checkpoints means nothing to keep
	waitState(t, m, StateDone)
	if c := p.Calls(); len(c) != 1 {
		t.Fatalf("calls %v", c)
	}

	// a new job ends: countdown again; cancel holds for that ending
	j2 := e.finishedJob(t, "lora_e", "completed", "Job completed", false)
	waitState(t, m, StateCountdown)
	if _, err := Cancel(); err != nil {
		t.Fatal(err)
	}
	m.evaluate()
	if s := m.Snapshot(); s.State != StateCancelled {
		t.Fatalf("%+v", s)
	}
	if _, err := Cancel(); !errors.Is(err, ErrConflict) {
		t.Fatalf("second cancel: %v", err)
	}
	// a restart keeps it cancelled
	m2 := startManager(t, Config{Mode: ModeDestroy, DelayS: 600}, p)
	m2.restoreHandled()
	m2.evaluate()
	if s := m2.Snapshot(); s.State != StateCancelled {
		t.Fatalf("after restart: %+v", s)
	}
	m = m2

	// work queued: back to waiting, and /now is refused while it runs
	store.UpdateJob(ctx, j2.ID, store.Fields{"status": "queued"})
	s = waitState(t, m, StateWaiting)
	if !strings.Contains(s.Reason, "lora_e") {
		t.Fatalf("%q", s.Reason)
	}
	if _, err := Now(); !errors.Is(err, ErrConflict) {
		t.Fatalf("now while queued: %v", err)
	}
	// it ends again: a new countdown, which a queued job cancels
	state.StartAttempt(ctx, j2.ID, 20, false)
	state.EndAttempt(ctx, j2.ID, "completed", "", 20)
	store.UpdateJob(ctx, j2.ID, store.Fields{"status": "completed"})
	waitState(t, m, StateCountdown)
	store.UpdateJob(ctx, j2.ID, store.Fields{"status": "running"})
	s = waitState(t, m, StateWaiting)
	if !strings.Contains(s.Reason, "countdown cancelled") {
		t.Fatalf("%q", s.Reason)
	}
}

func TestTriggerOn(t *testing.T) {
	e := setup(t)
	e.finishedJob(t, "lora_f", "error", "Traceback: boom", false)
	m := startManager(t, Config{Mode: ModeStop, DelayS: 600}, &fakeProvider{})
	m.evaluate()
	if s := m.Snapshot(); s.State != StateWaiting || !strings.Contains(s.Reason, "failed") {
		t.Fatalf("%+v", s)
	}
	on := OnAny
	if _, err := Update(context.Background(), Patch{On: &on}); err != nil {
		t.Fatal(err)
	}
	if s := m.Snapshot(); s.State != StateCountdown {
		t.Fatalf("%+v", s)
	}
	keep := ModeKeep
	if s, _ := Update(context.Background(), Patch{Mode: &keep}); s.State != StateOff {
		t.Fatalf("%+v", s)
	}
	bad := "explode"
	if _, err := Update(context.Background(), Patch{Mode: &bad}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("%v", err)
	}
}

func TestConfigFromEnv(t *testing.T) {
	t.Setenv("AITK_WHEN_DONE", "destroy")
	t.Setenv("AITK_WHEN_DONE_DELAY", "90s")
	t.Setenv("AITK_WHEN_DONE_ON", "any")
	t.Setenv("AITK_WHEN_DONE_DRY_RUN", "1")
	c, err := normalize(configFromEnv(Config{DelayS: 600}))
	if err != nil || c != (Config{Mode: ModeDestroy, On: OnAny, DelayS: 90, DryRun: true}) {
		t.Fatalf("%+v %v", c, err)
	}
	t.Setenv("AITK_WHEN_DONE_DELAY", "5")
	if c := configFromEnv(Config{}); c.DelayS != 300 {
		t.Fatalf("bare minutes: %+v", c)
	}
}

// provider HTTP: recorded against an httptest server, never the real APIs

type recorded struct {
	method, path, auth, body string
}

func fakeAPI(t *testing.T, answer func(r recorded) (int, string)) (*httptest.Server, *[]recorded) {
	t.Helper()
	var mu sync.Mutex
	var got []recorded
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		rec := recorded{r.Method, r.URL.Path, r.Header.Get("Authorization"), string(b)}
		mu.Lock()
		got = append(got, rec)
		mu.Unlock()
		status, body := answer(rec)
		w.WriteHeader(status)
		io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

func TestRunPodTerminate(t *testing.T) {
	srv, got := fakeAPI(t, func(r recorded) (int, string) { return http.StatusNoContent, "" })
	runpodAPI = srv.URL + "/v1"
	t.Cleanup(func() { runpodAPI = "https://rest.runpod.io/v1" })
	t.Setenv("RUNPOD_POD_ID", "abc123")
	t.Setenv("RUNPOD_API_KEY", "pod-key")
	t.Setenv("AITK_RUNPOD_API_KEY", "full-key")
	p := detectProvider()
	if p.Name() != "runpod" {
		t.Fatal(p.Name())
	}
	detail, err := p.Destroy(context.Background())
	if err != nil || !strings.Contains(detail, "terminated") {
		t.Fatalf("%q %v", detail, err)
	}
	if g := *got; len(g) != 1 || g[0].method != "DELETE" || g[0].path != "/v1/pods/abc123" || g[0].auth != "Bearer full-key" {
		t.Fatalf("%+v", g)
	}
}

func TestRunPodPodKeyFallsBackToStop(t *testing.T) {
	srv, got := fakeAPI(t, func(r recorded) (int, string) {
		if r.method == "DELETE" {
			return http.StatusForbidden, `{"error":"insufficient permissions"}`
		}
		return http.StatusOK, `{"id":"abc123","desiredStatus":"EXITED"}`
	})
	runpodAPI = srv.URL + "/v1"
	t.Cleanup(func() { runpodAPI = "https://rest.runpod.io/v1" })
	p := &runPod{id: "abc123", key: "pod-key"}
	detail, err := p.Destroy(context.Background())
	if err != nil || !strings.Contains(detail, "stopped") || !strings.Contains(detail, "AITK_RUNPOD_API_KEY") {
		t.Fatalf("%q %v", detail, err)
	}
	if g := *got; len(g) != 2 || g[1].method != "POST" || g[1].path != "/v1/pods/abc123/stop" || g[1].auth != "Bearer pod-key" {
		t.Fatalf("%+v", g)
	}
	// other errors are errors, not a silent stop
	srv2, _ := fakeAPI(t, func(recorded) (int, string) { return http.StatusInternalServerError, "boom" })
	runpodAPI = srv2.URL + "/v1"
	if _, err := p.Destroy(context.Background()); err == nil || !strings.Contains(err.Error(), "500") {
		t.Fatalf("%v", err)
	}
	if _, err := (&runPod{id: "x"}).Destroy(context.Background()); err == nil {
		t.Fatal("no key accepted")
	}
}

func TestVastDestroyAndStop(t *testing.T) {
	srv, got := fakeAPI(t, func(r recorded) (int, string) { return http.StatusOK, `{"success": true}` })
	vastAPI = srv.URL + "/api/v0"
	t.Cleanup(func() { vastAPI = "https://console.vast.ai/api/v0" })
	t.Setenv("RUNPOD_POD_ID", "")
	t.Setenv("CONTAINER_ID", "4242")
	t.Setenv("CONTAINER_API_KEY", "inst-key")
	p := detectProvider()
	if p.Name() != "vast" {
		t.Fatal(p.Name())
	}
	if _, err := p.Destroy(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	g := *got
	if len(g) != 2 || g[0].method != "DELETE" || g[0].path != "/api/v0/instances/4242/" || g[0].auth != "Bearer inst-key" {
		t.Fatalf("%+v", g)
	}
	if g[1].method != "PUT" || !strings.Contains(g[1].body, `"state":"stopped"`) {
		t.Fatalf("%+v", g[1])
	}
	// success: false is a failure
	srv2, _ := fakeAPI(t, func(recorded) (int, string) { return http.StatusOK, `{"success": false, "msg": "nope"}` })
	vastAPI = srv2.URL + "/api/v0"
	if _, err := p.Destroy(context.Background()); err == nil || !strings.Contains(err.Error(), "nope") {
		t.Fatalf("%v", err)
	}
}

func TestCommandAndExitProviders(t *testing.T) {
	cmd := "echo action=$AITK_WHEN_DONE_ACTION"
	if runtime.GOOS == "windows" {
		cmd = "echo action=%AITK_WHEN_DONE_ACTION%"
	}
	t.Setenv("AITK_WHEN_DONE_CMD", cmd)
	p := detectProvider()
	if p.Name() != "command" {
		t.Fatal(p.Name())
	}
	detail, err := p.Destroy(context.Background())
	if err != nil || !strings.Contains(detail, "action=destroy") {
		t.Fatalf("%q %v", detail, err)
	}
	t.Setenv("AITK_WHEN_DONE_CMD", "")
	t.Setenv("RUNPOD_POD_ID", "")
	t.Setenv("CONTAINER_ID", "")
	exited := make(chan struct{})
	SetExit(func() { close(exited) })
	exitDelay = 0
	p = detectProvider()
	if p.Name() != "exit" {
		t.Fatal(p.Name())
	}
	if _, err := p.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatal("exit not called")
	}
}

// Caption jobs are work while they run but never arm when done; an inference
// engine neither holds nor arms it.
func TestOnlyTrainingRunsArm(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	cap := e.finishedJob(t, "set_caption", "completed", "Captioning completed", false)
	store.UpdateJob(ctx, cap.ID, store.Fields{"job_type": "caption"})
	m := startManager(t, Config{Mode: ModeDestroy, DelayS: 600}, &fakeProvider{})
	m.evaluate()
	if s := m.Snapshot(); s.State != StateWaiting || !strings.Contains(s.Reason, "no job has finished") {
		t.Fatalf("finished caption job: %+v", s)
	}
	run := e.finishedJob(t, "lora_t", "completed", "Job completed", false)
	waitState(t, m, StateCountdown)

	// a running caption job holds the countdown like any work
	store.UpdateJob(ctx, cap.ID, store.Fields{"status": "running"})
	waitState(t, m, StateWaiting)
	store.UpdateJob(ctx, cap.ID, store.Fields{"status": "completed"})
	if s := waitState(t, m, StateCountdown); *s.JobName != "lora_t" {
		t.Fatalf("armed by %v", *s.JobName)
	}

	// a running engine does not hold it, and stopping it does not disarm it
	eng := e.finishedJob(t, "engine", "running", "Engine ready", false)
	store.UpdateJob(ctx, eng.ID, store.Fields{"job_type": "inference", "status": "running"})
	m.evaluate()
	if s := m.Snapshot(); s.State != StateCountdown {
		t.Fatalf("with an engine running: %+v", s)
	}
	store.UpdateJob(ctx, eng.ID, store.Fields{"status": "stopped"})
	m.evaluate()
	if s := m.Snapshot(); s.State != StateCountdown || *s.JobName != run.Name {
		t.Fatalf("after the engine stopped: %+v", s)
	}
}

// With on any, an interrupted run that was not resumed is an ending.
func TestInterruptedEndsAfterGrace(t *testing.T) {
	e := setup(t)
	e.finishedJob(t, "lora_i", "error", "Job process terminated with signal SIGKILL.", false)
	m := startManager(t, Config{Mode: ModeStop, On: OnAny, DelayS: 600}, &fakeProvider{})
	m.evaluate()
	if s := m.Snapshot(); s.State != StateWaiting {
		t.Fatalf("within the grace: %+v", s)
	}
	m.now = func() time.Time { return time.Now().Add(interruptGrace + time.Second) }
	m.evaluate()
	if s := m.Snapshot(); s.State != StateCountdown {
		t.Fatalf("after the grace: %+v", s)
	}
}

// An ending acted on is not acted on again after a restart, and transfers in
// flight hold the action at the deadline.
func TestDonePersistsAndTransfersHold(t *testing.T) {
	e := setup(t)
	e.finishedJob(t, "lora_p", "completed", "Job completed", false)
	p := &fakeProvider{}
	m := startManager(t, Config{Mode: ModeStop, DelayS: 0}, p)
	work := "1 upload(s)"
	m.transfers = func(context.Context) string { return work }
	m.evaluate()
	if s := m.Snapshot(); s.State != StateCountdown || !strings.Contains(s.Reason, "waiting for 1 upload(s)") {
		t.Fatalf("with an upload in flight: %+v", s)
	}
	if _, err := Now(); !errors.Is(err, ErrConflict) {
		t.Fatalf("now with an upload in flight: %v", err)
	}
	work = ""
	waitState(t, m, StateDone)

	m2 := startManager(t, Config{Mode: ModeStop, DelayS: 0}, p)
	m2.transfers = func(context.Context) string { return "" }
	m2.restoreHandled()
	m2.evaluate()
	time.Sleep(50 * time.Millisecond)
	if s := m2.Snapshot(); s.State != StateWaiting || !strings.Contains(s.Reason, "already handled") {
		t.Fatalf("after restart: %+v", s)
	}
	if c := p.Calls(); len(c) != 1 {
		t.Fatalf("calls %v", c)
	}
}
