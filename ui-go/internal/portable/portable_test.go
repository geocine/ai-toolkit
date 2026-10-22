package portable

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

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

// node is one simulated server: its own toolkit root, upstream DB and fork DB.
type node struct {
	root           string
	upstream, fork *sql.DB
}

func newNode(t *testing.T, shared string) *node {
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
	n := &node{root: root}
	n.setPaths()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	n.upstream = store.DB
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	n.fork = state.DB
	t.Cleanup(func() { n.upstream.Close(); n.fork.Close() })
	cfg, _ := json.Marshal(map[string]string{"path": shared})
	if _, err := remote.Create(context.Background(), "shared", "fs", cfg, nil); err != nil {
		t.Fatal(err)
	}
	return n
}

func (n *node) setPaths() {
	paths.ToolkitRoot = n.root
	paths.DBPath = filepath.Join(n.root, "aitk_db.db")
	paths.DefaultTrainFolder = filepath.Join(n.root, "output")
	paths.DefaultDatasetsFolder = filepath.Join(n.root, "datasets")
	paths.DefaultDataRoot = filepath.Join(n.root, "data")
	settings.Flush()
}

// use makes this node the one the package-level handles point at.
func (n *node) use() {
	n.setPaths()
	store.DB, state.DB = n.upstream, n.fork
}

func (n *node) ds(rel ...string) string {
	return filepath.Join(append([]string{n.root, "datasets"}, rel...)...)
}
func (n *node) tr(rel ...string) string {
	return filepath.Join(append([]string{n.root, "output"}, rel...)...)
}

func write(t *testing.T, p string, data string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func lossDB(t *testing.T, p string, steps int) {
	t.Helper()
	db, err := sql.Open("sqlite", store.FileURI(p))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	db.Exec(`CREATE TABLE IF NOT EXISTS metrics (step INTEGER, key TEXT, value_real REAL, value_text TEXT)`)
	for i := 1; i <= steps; i++ {
		db.Exec(`INSERT INTO metrics VALUES (?, 'loss', ?, NULL)`, i, 1.0/float64(i))
	}
}

const jobName = "my_lora"

// setupJob creates on n a synced job with a dataset, two checkpoints,
// optimizer state, loss log, log and a sample.
func setupJob(t *testing.T, n *node, remoteName string) *store.Job {
	t.Helper()
	ctx := context.Background()
	n.use()
	write(t, n.ds("set1", "a.png"), "png-a")
	write(t, n.ds("set1", "a.txt"), "caption a")
	write(t, n.ds("set1", "sub", "b.png"), "png-b")
	write(t, n.ds("set1", ".thumbs", "a.png.jpg"), "generated thumb")
	cfg, _ := json.Marshal(map[string]any{
		"job": "extension",
		"config": map[string]any{"name": jobName, "process": []any{map[string]any{
			"type": "diffusion_trainer", "training_folder": n.tr(), "sqlite_db_path": filepath.Join(n.root, "aitk_db.db"),
			"train":    map[string]any{"steps": 40, "lr": 1e-4},
			"datasets": []any{map[string]any{"folder_path": n.ds("set1"), "control_path": n.ds("set1", "sub")}},
		}}},
		"meta": map[string]any{"name": "[name]"},
	})
	j, err := store.CreateJob(ctx, store.NewJob{Name: jobName, GpuIDs: "0", JobConfig: string(cfg), QueuePosition: 1000})
	if err != nil {
		t.Fatal(err)
	}
	if j, err = store.UpdateJob(ctx, j.ID, store.Fields{"status": "running", "step": 20, "info": "Training"}); err != nil {
		t.Fatal(err)
	}
	write(t, n.tr(jobName, jobName+"_000000010.safetensors"), "weights-10")
	write(t, n.tr(jobName, jobName+"_000000020.safetensors"), "weights-20")
	write(t, n.tr(jobName, "optimizer.pt"), "optim-20")
	write(t, n.tr(jobName, "log.txt"), "step 1\nstep 20\n")
	write(t, n.tr(jobName, "samples", "1700000000000__000000020_0.jpg"), "sample-20")
	lossDB(t, n.tr(jobName, "loss_log.db"), 20)
	if _, err := state.StartAttempt(ctx, j.ID, 0, false); err != nil {
		t.Fatal(err)
	}
	rn := remoteName
	if err := state.SetPolicy(ctx, j.ID, state.Policy{SyncRemote: &rn}); err != nil {
		t.Fatal(err)
	}
	return j
}

func manifestOn(t *testing.T, shared string) *Manifest {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(shared, "jobs", jobName, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

func fileNames(fs []File) map[string]File {
	out := map[string]File{}
	for _, f := range fs {
		out[f.Name] = f
	}
	return out
}

func init() { stablePoll = 5 * time.Millisecond }

func code(err error) string {
	var p *v1api.Problem
	if errors.As(err, &p) {
		return p.Code
	}
	return ""
}

func TestExportImportResume(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a, b := newNode(t, shared), newNode(t, shared)
	j := setupJob(t, a, "shared")

	a.use()
	st, err := SyncJob(ctx, j.ID, ReasonStart)
	if err != nil {
		t.Fatal(err)
	}
	if st.LastError != nil || st.Step == nil || *st.Step != 20 || st.Lease == nil {
		t.Fatalf("status %+v", st)
	}
	m := manifestOn(t, shared)
	got := fileNames(m.Files)
	for _, want := range []string{jobName + "_000000020.safetensors", "optimizer.pt", "loss_log.db", "log.txt"} {
		if _, ok := got[want]; !ok {
			t.Errorf("manifest lacks %s: %v", want, got)
		}
	}
	if _, ok := got[jobName+"_000000010.safetensors"]; ok {
		t.Error("older checkpoint should not be part of the resumable state")
	}
	aNode, _ := state.NodeID(ctx)
	if m.Lease == nil || m.Lease.NodeID != aNode {
		t.Fatalf("lease %+v, want node %s", m.Lease, aNode)
	}
	// the root as it would appear inside JSON too (Windows backslashes escaped)
	escaped, _ := json.Marshal(a.root)
	if strings.Contains(string(m.Config), a.root) || strings.Contains(string(m.Config), filepath.ToSlash(a.root)) ||
		strings.Contains(string(m.Config), strings.Trim(string(escaped), `"`)) {
		t.Fatalf("config still has machine paths: %s", m.Config)
	}
	if !strings.Contains(string(m.Config), `"aitk://datasets/set1"`) || !strings.Contains(string(m.Config), `"aitk://datasets/set1/sub"`) ||
		!strings.Contains(string(m.Config), `"aitk://training"`) {
		t.Fatalf("config not portable: %s", m.Config)
	}
	if len(m.Datasets) != 1 || m.Datasets[0] != "set1" {
		t.Fatalf("datasets %v", m.Datasets)
	}
	var dm DatasetManifest
	raw, _ := os.ReadFile(filepath.Join(shared, "datasets", "set1", "manifest.json"))
	json.Unmarshal(raw, &dm)
	if len(dm.Files) != 3 { // a.png, a.txt, sub/b.png; .thumbs skipped
		t.Fatalf("dataset files %+v", dm.Files)
	}
	for _, f := range dm.Files {
		if _, err := os.Stat(filepath.Join(shared, "blobs", f.SHA256)); err != nil {
			t.Fatalf("blob for %s missing", f.Path)
		}
	}

	// the job ends cleanly on A: lease released
	if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"status": "stopped", "info": "Job stopped"}); err != nil {
		t.Fatal(err)
	}
	if _, err := SyncJob(ctx, j.ID, ReasonFinal); err != nil {
		t.Fatal(err)
	}
	if manifestOn(t, shared).Lease != nil {
		t.Fatal("lease should be released after a clean end")
	}

	// B lists and imports it
	b.use()
	list, err := ListRemoteJobs(ctx, "shared")
	if err != nil || len(list) != 1 || list[0].Name != jobName || list[0].Step != 20 {
		t.Fatalf("list %+v err %v", list, err)
	}
	gpu := "1"
	imported, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName, GpuIDs: &gpu})
	if err != nil {
		t.Fatal(err)
	}
	if imported.GpuIDs != "1" || imported.Step != 20 || !strings.Contains(imported.Info, "Imported from shared at step 20") {
		t.Fatalf("imported job %+v", imported)
	}
	for rel, want := range map[string]string{"a.png": "png-a", "a.txt": "caption a", "sub/b.png": "png-b"} {
		got, err := os.ReadFile(b.ds("set1", filepath.FromSlash(rel)))
		if err != nil || string(got) != want {
			t.Fatalf("dataset %s = %q, %v", rel, got, err)
		}
	}
	// what ai-toolkit needs to resume at step 20
	for name, want := range map[string]string{jobName + "_000000020.safetensors": "weights-20", "optimizer.pt": "optim-20"} {
		got, err := os.ReadFile(b.tr(jobName, name))
		if err != nil || string(got) != want {
			t.Fatalf("%s = %q, %v", name, got, err)
		}
	}
	if _, err := os.Stat(b.tr(jobName, "samples", "1700000000000__000000020_0.jpg")); err != nil {
		t.Errorf("sample not restored: %v", err)
	}
	lossCheck, err := sql.Open("sqlite", store.FileURI(b.tr(jobName, "loss_log.db"))+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	var n int
	if err := lossCheck.QueryRow(`SELECT COUNT(*) FROM metrics`).Scan(&n); err != nil || n != 20 {
		t.Fatalf("loss log rows %d err %v", n, err)
	}
	lossCheck.Close()
	jc, _ := os.ReadFile(b.tr(jobName, ".job_config.json"))
	if !strings.Contains(string(jc), jsonString(filepath.Join(b.root, "aitk_db.db"))) || !strings.Contains(string(jc), "\n  ") {
		t.Fatalf(".job_config.json not written for this node:\n%s", jc)
	}
	// the config now points at B's folders
	for _, want := range []string{jsonString(b.ds("set1")), jsonString(b.ds("set1", "sub")), jsonString(b.tr())} {
		if !strings.Contains(imported.JobConfig, want) {
			t.Errorf("config lacks %s: %s", want, imported.JobConfig)
		}
	}
	if strings.Contains(imported.JobConfig, "aitk://") {
		t.Errorf("config kept portable paths: %s", imported.JobConfig)
	}
	pol, _ := state.GetPolicy(ctx, imported.ID)
	if pol.SyncRemote == nil || *pol.SyncRemote != "shared" {
		t.Fatalf("policy %+v", pol)
	}
	attempts, _ := state.Attempts(ctx, imported.ID)
	if len(attempts) != 1 || attempts[0].EndedAt == nil {
		t.Fatalf("attempts not carried over / closed: %+v", attempts)
	}
	bNode, _ := state.NodeID(ctx)
	if l := manifestOn(t, shared).Lease; l == nil || l.NodeID != bNode {
		t.Fatalf("lease after import %+v, want %s", l, bNode)
	}
	// importing twice is a name conflict
	if _, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName}); code(err) != v1api.CodeConflict {
		t.Fatalf("second import: %v", err)
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestLeaseConflictForceAndLoss(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a, b := newNode(t, shared), newNode(t, shared)
	j := setupJob(t, a, "shared")
	a.use()
	if _, err := SyncJob(ctx, j.ID, ReasonStart); err != nil {
		t.Fatal(err)
	}

	b.use()
	_, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName})
	if code(err) != v1api.CodeConflict || !strings.Contains(err.Error(), "leased by") {
		t.Fatalf("import of a leased job: %v", err)
	}
	if _, err := os.Stat(b.tr(jobName)); err == nil {
		t.Fatal("refused import must not leave files")
	}
	if _, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName, Force: true}); err != nil {
		t.Fatalf("forced import: %v", err)
	}

	// A notices on its next renewal and stops its copy
	a.use()
	if err := RenewLease(ctx, j); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("renew after takeover: %v", err)
	}
	got, _ := store.GetJob(ctx, j.ID)
	if !got.Stop || !strings.Contains(got.Info, "lease taken over") {
		t.Fatalf("A's job not stopped: %+v", got)
	}
	// and A can no longer write the manifest
	if _, err := SyncJob(ctx, j.ID, ReasonCheckpoint); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("sync after takeover: %v", err)
	}
}

func TestLeaseRenewAndExpiry(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a := newNode(t, shared)
	j := setupJob(t, a, "shared")
	if _, err := SyncJob(ctx, j.ID, ReasonStart); err != nil {
		t.Fatal(err)
	}
	first := manifestOn(t, shared).Lease
	time.Sleep(20 * time.Millisecond)
	if err := RenewLease(ctx, j); err != nil {
		t.Fatal(err)
	}
	renewed := manifestOn(t, shared).Lease
	if !renewed.ExpiresAt.After(first.ExpiresAt) || renewed.ExpiresAt.Sub(renewed.HeartbeatAt) != LeaseTTL {
		t.Fatalf("lease not renewed: %+v -> %+v", first, renewed)
	}
	if renewed.Held(renewed.NodeID, time.Now()) {
		t.Fatal("own lease must not count as held")
	}
	if !renewed.Held("other", time.Now()) || renewed.Held("other", renewed.ExpiresAt.Add(time.Second)) {
		t.Fatal("Held must respect expiry")
	}
}

// A checkpoint event that arrives after the job finished (the final file is
// reported once its size settles) must not take the lease back.
func TestLateCheckpointAfterFinishKeepsLeaseReleased(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a := newNode(t, shared)
	j := setupJob(t, a, "shared")
	if _, err := SyncJob(ctx, j.ID, ReasonStart); err != nil {
		t.Fatal(err)
	}
	if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"status": "completed", "info": "Training completed"}); err != nil {
		t.Fatal(err)
	}
	if _, err := SyncJob(ctx, j.ID, ReasonFinal); err != nil {
		t.Fatal(err)
	}
	if _, err := SyncJob(ctx, j.ID, ReasonCheckpoint); err != nil {
		t.Fatal(err)
	}
	if l := manifestOn(t, shared).Lease; l != nil {
		t.Fatalf("finished job still leased: %+v", l)
	}
}

func TestPruneOldCheckpoints(t *testing.T) {
	ctx := context.Background()
	shared, keepDir := t.TempDir(), t.TempDir()
	a := newNode(t, shared)
	keepCfg, _ := json.Marshal(map[string]any{"path": keepDir, "keep_checkpoints": 1})
	if _, err := remote.Create(ctx, "keep1", "fs", keepCfg, nil); err != nil {
		t.Fatal(err)
	}
	j := setupJob(t, a, "shared")
	remoteCkpts := func(dir string) []string {
		var out []string
		filepath.WalkDir(filepath.Join(dir, "jobs", jobName, "files"), func(p string, d os.DirEntry, err error) error {
			if err == nil && !d.IsDir() && strings.HasSuffix(p, ".safetensors") {
				out = append(out, filepath.Base(p))
			}
			return nil
		})
		return out
	}
	if _, err := SyncJob(ctx, j.ID, ReasonStart); err != nil {
		t.Fatal(err)
	}
	write(t, a.tr(jobName, jobName+"_000000030.safetensors"), "weights-30")
	write(t, a.tr(jobName, "optimizer.pt"), "optim-30")
	if _, err := SyncJob(ctx, j.ID, ReasonCheckpoint); err != nil {
		t.Fatal(err)
	}
	if got := remoteCkpts(shared); len(got) != 1 || got[0] != jobName+"_000000030.safetensors" {
		t.Fatalf("default remote keeps only the newest: %v", got)
	}
	// only one optimizer.pt remains (superseded state is pruned)
	var optim int
	filepath.WalkDir(filepath.Join(shared, "jobs", jobName, "files"), func(p string, d os.DirEntry, err error) error {
		if err == nil && d.Name() == "optimizer.pt" {
			optim++
		}
		return nil
	})
	if optim != 1 {
		t.Fatalf("%d optimizer.pt copies remain", optim)
	}

	// keep_checkpoints: 1 keeps one older checkpoint besides the current
	k := "keep1"
	state.SetPolicy(ctx, j.ID, state.Policy{SyncRemote: &k})
	if _, err := SyncJob(ctx, j.ID, ReasonStart); err != nil {
		t.Fatal(err)
	}
	write(t, a.tr(jobName, jobName+"_000000040.safetensors"), "weights-40")
	if _, err := SyncJob(ctx, j.ID, ReasonCheckpoint); err != nil {
		t.Fatal(err)
	}
	write(t, a.tr(jobName, jobName+"_000000050.safetensors"), "weights-50")
	if _, err := SyncJob(ctx, j.ID, ReasonCheckpoint); err != nil {
		t.Fatal(err)
	}
	got := strings.Join(remoteCkpts(keepDir), ",")
	if !strings.Contains(got, "_000000050") || !strings.Contains(got, "_000000040") || strings.Contains(got, "_000000030") {
		t.Fatalf("keep_checkpoints 1: %s", got)
	}
}

func TestDatasetConflictAndReuse(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a, b := newNode(t, shared), newNode(t, shared)
	j := setupJob(t, a, "shared")
	a.use()
	if _, err := SyncJob(ctx, j.ID, ReasonFinal); err != nil {
		t.Fatal(err)
	}

	b.use()
	write(t, b.ds("set1", "a.png"), "something else")
	_, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName})
	if code(err) != v1api.CodeConflict || !strings.Contains(err.Error(), "different contents") {
		t.Fatalf("differing dataset: %v", err)
	}
	if l := manifestOn(t, shared).Lease; l != nil {
		t.Fatalf("failed import must give the lease back: %+v", l)
	}
	// an identical local copy is reused
	write(t, b.ds("set1", "a.png"), "png-a")
	write(t, b.ds("set1", "a.txt"), "caption a")
	write(t, b.ds("set1", "sub", "b.png"), "png-b")
	if _, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName}); err != nil {
		t.Fatalf("identical dataset should be reused: %v", err)
	}
}

func TestCorruptBlobRejected(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a, b := newNode(t, shared), newNode(t, shared)
	j := setupJob(t, a, "shared")
	a.use()
	if _, err := SyncJob(ctx, j.ID, ReasonFinal); err != nil {
		t.Fatal(err)
	}
	// corrupt the checkpoint on the remote
	m := manifestOn(t, shared)
	ck := fileNames(m.Files)[jobName+"_000000020.safetensors"]
	if err := os.WriteFile(filepath.Join(shared, filepath.FromSlash(ck.Key)), []byte("bit rot"), 0o644); err != nil {
		t.Fatal(err)
	}
	b.use()
	_, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName})
	if code(err) != v1api.CodeRemoteFailed || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("corrupt checkpoint: %v", err)
	}
	if j, _ := store.FindFirstJob(ctx, store.JobQuery{Where: `"name" = ?`, Args: []any{jobName}}); j != nil {
		t.Fatal("no job may be created from a corrupt bundle")
	}
	if _, err := os.Stat(b.tr(jobName)); err == nil {
		t.Fatal("no job folder may be left behind")
	}

	// a corrupt dataset blob is caught too
	os.RemoveAll(b.ds("set1"))
	sh := filepath.Join(shared, "blobs")
	entries, _ := os.ReadDir(sh)
	for _, e := range entries {
		os.WriteFile(filepath.Join(sh, e.Name()), []byte("x"), 0o644)
	}
	if _, err := Import(ctx, ImportRequest{Remote: "shared", Name: jobName}); code(err) != v1api.CodeRemoteFailed {
		t.Fatalf("corrupt blob: %v", err)
	}
	if _, err := os.Stat(b.ds("set1")); err == nil {
		t.Fatal("a corrupt dataset must not be put in place")
	}
}

func TestStartGuard(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a, b := newNode(t, shared), newNode(t, shared)
	j := setupJob(t, a, "shared")
	a.use()
	if _, err := SyncJob(ctx, j.ID, ReasonStart); err != nil {
		t.Fatal(err)
	}

	// B has its own synced copy queued
	b.use()
	bj := setupJob(t, b, "shared")
	ok, info := StartGuard(ctx, bj)
	if ok || !strings.HasPrefix(info, "Waiting: lease held by") {
		t.Fatalf("guard let a leased job start: %v %q", ok, info)
	}
	// cached while blocked
	if ok, _ := StartGuard(ctx, bj); ok {
		t.Fatal("blocked decision should be cached")
	}

	// once A's lease expires, B may start and takes the lease
	m := manifestOn(t, shared)
	m.Lease.ExpiresAt = time.Now().Add(-time.Minute)
	raw, _ := json.Marshal(m)
	os.WriteFile(filepath.Join(shared, "jobs", jobName, "manifest.json"), raw, 0o644)
	guardMu.Lock()
	delete(guardBlocked, bj.ID)
	guardMu.Unlock()
	if ok, info := StartGuard(ctx, bj); !ok {
		t.Fatalf("guard blocked after expiry: %q", info)
	}
	bNode, _ := state.NodeID(ctx)
	if l := manifestOn(t, shared).Lease; l == nil || l.NodeID != bNode {
		t.Fatalf("start must take the lease: %+v", l)
	}

	// jobs without sync_remote are never held back
	b.use()
	cfg := `{"config":{"process":[{}]}}`
	free, _ := store.CreateJob(ctx, store.NewJob{Name: "free", GpuIDs: "0", JobConfig: cfg})
	if ok, _ := StartGuard(ctx, free); !ok {
		t.Fatal("unsynced job blocked")
	}
}

func TestPortablePaths(t *testing.T) {
	ds, tr := filepath.Join(t.TempDir(), "datasets"), filepath.Join(t.TempDir(), "output")
	in, _ := json.Marshal(map[string]any{"a": filepath.Join(ds, "x", "y.png"), "b": tr, "c": filepath.Join(tr, "job", "f"),
		"d": "relative/path", "e": filepath.Join(ds+"-other", "z"), "n": 1e-4})
	out, names, err := ToPortable(string(in), ds, tr)
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	for _, want := range []string{`"aitk://datasets/x/y.png"`, `"aitk://training"`, `"aitk://training/job/f"`, `"relative/path"`, `0.0001`} {
		if !strings.Contains(s, want) && !(want == `0.0001` && strings.Contains(s, `1e-4`)) {
			t.Errorf("missing %s in %s", want, s)
		}
	}
	if strings.Contains(s, "aitk://datasets-other") || len(names) != 1 || names[0] != "x" {
		t.Fatalf("prefix sibling rewritten or names wrong: %s %v", s, names)
	}
	back, err := FromPortable(out, "/new/ds", "/new/out")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains([]byte(back), []byte(jsonString(filepath.Join("/new/ds", "x", "y.png")))) {
		t.Fatalf("not mapped back: %s", back)
	}
}

// A new job reusing a finished run's name may not write over it, and a node
// that is behind the remote (a paused pod that came back) may not roll it back.
func TestOtherRunAndStaleNodeNeverOverwrite(t *testing.T) {
	ctx := context.Background()
	shared := t.TempDir()
	a := newNode(t, shared)
	j := setupJob(t, a, "shared")
	if _, err := SyncJob(ctx, j.ID, ReasonStart); err != nil {
		t.Fatal(err)
	}
	first := manifestOn(t, shared)
	if first.RunID != j.ID {
		t.Fatalf("run id %q, want %q", first.RunID, j.ID)
	}

	// the run goes on elsewhere to step 30
	m := *first
	step := 30
	m.Files = append(append([]File{}, m.Files...), File{Name: jobName + "_000000030.safetensors", Key: "jobs/" + jobName + "/files/x/30", Step: &step})
	m.Step, m.Lease = 30, nil
	b, _ := json.Marshal(m)
	write(t, filepath.Join(shared, "jobs", jobName, "manifest.json"), string(b))
	if _, err := SyncJob(ctx, j.ID, ReasonCheckpoint); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("stale sync: %v", err)
	}
	if got := manifestOn(t, shared); got.Step != 30 || newestStep(got.Files) != 30 {
		t.Fatalf("remote rolled back to %d", got.Step)
	}
	if j2, _ := store.GetJob(ctx, j.ID); j2 == nil || !j2.Stop {
		t.Fatalf("stale job not stopped: %+v", j2)
	}

	// another server, a new job with the same name
	other := newNode(t, shared)
	j3 := setupJob(t, other, "shared")
	if _, err := SyncJob(ctx, j3.ID, ReasonStart); !errors.Is(err, ErrOtherRun) {
		t.Fatalf("same name, other run: %v", err)
	}
	if got := manifestOn(t, shared); got.RunID != j.ID || got.Step != 30 {
		t.Fatalf("other run overwrote the manifest: %+v", got)
	}
}
