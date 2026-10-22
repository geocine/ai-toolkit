package portable

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// JobSync is the event type for sync progress and errors:
// {remote, action, step, files, error}.
const JobSync = "job.sync"

// Sync reasons.
const (
	ReasonStart       = "start"       // job launched: datasets + manifest + lease
	ReasonCheckpoint  = "checkpoint"  // new save: files + manifest
	ReasonInterrupted = "interrupted" // trainer died: push what is there, keep the lease
	ReasonFinal       = "final"       // clean end: files + manifest, lease released
	ReasonManual      = "manual"      // POST /v1/jobs/{id}/sync
)

// ErrNotSynced: the job's policy has no sync_remote.
var ErrNotSynced = errors.New("job has no sync_remote")

// ErrLeaseLost: another node holds the job's lease; this node must not write.
var ErrLeaseLost = errors.New("lease held by another node")

// Status is the per-job sync record (GET /v1/jobs/{id}/sync).
type Status struct {
	Remote     string     `json:"remote"`
	LastSyncAt *time.Time `json:"last_sync_at"`
	LastError  *string    `json:"last_error"`
	Step       *int       `json:"step"`
	Lease      *Lease     `json:"lease"`
}

func statusKey(jobID string) string { return "portable." + jobID }

// GetStatus reads the stored sync record (lease not filled in).
func GetStatus(ctx context.Context, jobID string) (Status, error) {
	var s Status
	v, ok, err := state.Setting(ctx, statusKey(jobID))
	if err != nil || !ok {
		return s, err
	}
	err = json.Unmarshal([]byte(v), &s)
	return s, err
}

func saveStatus(ctx context.Context, jobID string, s Status) {
	s.Lease = nil
	b, _ := json.Marshal(s)
	if err := state.SetSetting(ctx, statusKey(jobID), string(b)); err != nil {
		log.Printf("portable: save status %s: %v", jobID, err)
	}
}

// per-job-name manifest locks: syncs and lease renewals read-modify-write it
var (
	manifestMu    sync.Mutex
	manifestLocks = map[string]*sync.Mutex{}
)

func lockManifest(name string) func() {
	manifestMu.Lock()
	l, ok := manifestLocks[name]
	if !ok {
		l = &sync.Mutex{}
		manifestLocks[name] = l
	}
	manifestMu.Unlock()
	l.Lock()
	return l.Unlock
}

// pushed remembers datasets already pushed to a remote by this process
var (
	pushedMu sync.Mutex
	pushed   = map[string]bool{}
)

func syncTarget(ctx context.Context, j *store.Job) (*remote.Remote, remote.Client, state.Policy, error) {
	pol, err := state.GetPolicy(ctx, j.ID)
	if err != nil {
		return nil, nil, pol, err
	}
	if pol.SyncRemote == nil || *pol.SyncRemote == "" {
		return nil, nil, pol, ErrNotSynced
	}
	r, c, err := remote.Open(ctx, *pol.SyncRemote)
	return r, c, pol, err
}

// SyncJob syncs the job's portable state now and returns its status.
func SyncJob(ctx context.Context, jobID, reason string) (Status, error) {
	j, err := store.GetJob(ctx, jobID)
	if err != nil {
		return Status{}, err
	}
	if j == nil {
		return Status{}, store.ErrNotFound
	}
	r, c, pol, err := syncTarget(ctx, j)
	if err != nil {
		return Status{}, err
	}
	st, _ := GetStatus(ctx, j.ID)
	st.Remote = r.Name
	files, lease, err := syncJob(ctx, j, r, c, pol, reason)
	now := time.Now().UTC().Truncate(time.Millisecond)
	data := map[string]any{"remote": r.Name, "action": reason, "step": j.Step, "files": files, "error": nil}
	if err != nil {
		msg := err.Error()
		st.LastError = &msg
		data["error"] = msg
		saveStatus(ctx, j.ID, st)
		events.Emit(ctx, JobSync, j.ID, data)
		if errors.Is(err, ErrLeaseLost) {
			stopForLostLease(ctx, j, lease)
		}
		return st, err
	}
	step := j.Step
	st.LastSyncAt, st.LastError, st.Step = &now, nil, &step
	saveStatus(ctx, j.ID, st)
	events.Emit(ctx, JobSync, j.ID, data)
	st.Lease = lease
	return st, nil
}

type localFile struct {
	path, name string
	step       *int
}

// syncJob uploads what the job needs to resume elsewhere and commits the
// manifest. Returns the uploaded file names and the lease written.
func syncJob(ctx context.Context, j *store.Job, r *remote.Remote, c remote.Client, pol state.Policy, reason string) ([]string, *Lease, error) {
	if err := ValidName(j.Name); err != nil {
		return nil, nil, err
	}
	dsRoot, err := settings.DatasetsRoot(ctx)
	if err != nil {
		return nil, nil, err
	}
	trRoot, err := settings.TrainingFolder(ctx)
	if err != nil {
		return nil, nil, err
	}
	cfg, names, err := ToPortable(j.JobConfig, dsRoot, trRoot)
	if err != nil {
		return nil, nil, fmt.Errorf("job config: %w", err)
	}
	for _, n := range names {
		key := r.ID + "\x00" + n
		pushedMu.Lock()
		done := pushed[key]
		pushedMu.Unlock()
		if done && reason != ReasonStart && reason != ReasonManual {
			continue
		}
		if err := pushDataset(ctx, c, dsRoot, n); err != nil {
			return nil, nil, err
		}
		pushedMu.Lock()
		pushed[key] = true
		pushedMu.Unlock()
	}

	folder := filepath.Join(trRoot, j.Name)
	tmp, err := os.MkdirTemp("", "aitk-sync-")
	if err != nil {
		return nil, nil, err
	}
	defer os.RemoveAll(tmp)
	locals, err := collectFiles(ctx, folder, tmp)
	if err != nil {
		return nil, nil, err
	}

	unlock := lockManifest(j.Name)
	defer unlock()
	prev, err := readManifest(ctx, c, j.Name)
	if err != nil {
		return nil, nil, err
	}
	me, err := state.NodeID(ctx)
	if err != nil {
		return nil, nil, err
	}
	if prev != nil && prev.Lease.Held(me, time.Now()) {
		return nil, prev.Lease, fmt.Errorf("%w: %s until %s", ErrLeaseLost, prev.Lease.NodeLabel, prev.Lease.ExpiresAt.Format(time.RFC3339))
	}
	runID := RunID(ctx, j.ID)
	if prev != nil && prev.RunID != "" && prev.RunID != runID {
		// a new job reusing a finished run's name: writing would replace that
		// run's manifest and prune its checkpoints
		return nil, nil, fmt.Errorf("%w: %q on %s was started by another job; rename this job or delete the remote run", ErrOtherRun, j.Name, r.Name)
	}
	var localNewest []File
	for _, lf := range locals {
		localNewest = append(localNewest, File{Step: lf.step})
	}
	if prev != nil {
		if remoteStep, localStep := newestStep(prev.Files), newestStep(localNewest); remoteStep > localStep {
			// another server carried the run further while this one was away
			// (a paused pod that came back): never roll the remote back
			return nil, prev.Lease, fmt.Errorf("%w: the remote run is at checkpoint step %d, this server only has %d", ErrLeaseLost, remoteStep, localStep)
		}
	}
	have := map[string]bool{}
	if prev != nil {
		for _, f := range prev.Files {
			have[f.Key] = true
		}
	}
	var uploaded []string
	files := make([]File, 0, len(locals))
	for _, lf := range locals {
		sha, size, err := hashFile(ctx, lf.path)
		if err != nil {
			return nil, nil, err
		}
		f := File{Name: lf.name, Key: fileKey(j.Name, sha, lf.name), Size: size, SHA256: sha, Step: lf.step}
		if !have[f.Key] {
			if _, err := c.Stat(ctx, f.Key); err != nil {
				if !errors.Is(err, remote.ErrNotFound) {
					return nil, nil, err
				}
				if _, err := c.Put(ctx, f.Key, lf.path, nil); err != nil {
					return nil, nil, fmt.Errorf("upload %s: %w", lf.name, err)
				}
				uploaded = append(uploaded, lf.name)
			}
		}
		files = append(files, f)
	}
	samples := syncSamples(ctx, c, j.Name, folder, prev)

	// remote ids are per server: carry the auto-upload remote by name
	if pol.AutoUploadRemote != nil && *pol.AutoUploadRemote != "" {
		if ar, err := remote.Get(ctx, *pol.AutoUploadRemote); err == nil && ar != nil {
			name := ar.Name
			pol.AutoUploadRemote = &name
		}
	}
	m := &Manifest{Name: j.Name, RunID: runID, Type: j.JobType, Ref: portableRef(j.JobRef, dsRoot, trRoot), GpuIDs: j.GpuIDs,
		Config: cfg, Policy: pol, Datasets: names, Step: j.Step, TotalSteps: j.TotalSteps,
		State: jobstate.FromStatus(j.Status, j.Step, j.Info), Files: files, Samples: samples}
	if m.Datasets == nil {
		m.Datasets = []string{}
	}
	if m.Attempts, err = state.Attempts(ctx, j.ID); err != nil {
		return nil, nil, err
	}
	// a job that is not running here holds no lease, whatever triggered the
	// sync (a late checkpoint event after the final sync used to re-take it)
	settled := false
	switch jobstate.FromStatus(j.Status, j.Step, j.Info) {
	case jobstate.Completed, jobstate.Stopped, jobstate.Failed, jobstate.Draft:
		settled = true
	}
	switch {
	case reason == ReasonFinal || settled:
		m.Lease = nil
	case reason == ReasonInterrupted && prev != nil:
		m.Lease = prev.Lease // left to expire unless the job resumes here
	default:
		if m.Lease, err = newLease(ctx); err != nil {
			return nil, nil, err
		}
	}
	if err := writeManifest(ctx, c, m); err != nil {
		return nil, nil, err
	}
	if err := prune(ctx, c, r, j.Name, files, pol); err != nil {
		log.Printf("portable: prune %s: %v", j.Name, err)
	}
	return uploaded, m.Lease, nil
}

func portableRef(ref *string, dsRoot, trRoot string) *string {
	if ref == nil {
		return nil
	}
	raw, _ := json.Marshal(*ref)
	out, _, err := ToPortable(string(raw), dsRoot, trRoot)
	if err != nil {
		return ref
	}
	var s string
	if json.Unmarshal(out, &s) != nil {
		return ref
	}
	return &s
}

// collectFiles picks what resuming needs: the newest step checkpoint, the
// final file when present, optimizer.pt, and snapshots of loss_log.db,
// log.txt and .job_config.json (copied into tmp so hash == upload).
func collectFiles(ctx context.Context, folder, tmp string) ([]localFile, error) {
	cps, err := artifacts.Checkpoints(folder)
	if err != nil {
		return nil, err
	}
	var out []localFile
	var newest *artifacts.Checkpoint
	for i := range cps {
		cp := &cps[i]
		if cp.Step == nil {
			if err := waitStable(ctx, cp.Path, 30*time.Second); err == nil {
				out = append(out, localFile{path: cp.Path, name: cp.Name})
			}
			continue
		}
		if newest == nil || *cp.Step > *newest.Step {
			newest = cp
		}
	}
	if newest != nil {
		if err := waitStable(ctx, newest.Path, 30*time.Second); err == nil {
			out = append(out, localFile{path: newest.Path, name: newest.Name, step: newest.Step})
		}
	}
	if p := filepath.Join(folder, "optimizer.pt"); exists(p) {
		if err := waitStable(ctx, p, 60*time.Second); err == nil {
			out = append(out, localFile{path: p, name: "optimizer.pt"})
		}
	}
	if p := filepath.Join(folder, "loss_log.db"); exists(p) {
		dst := filepath.Join(tmp, "loss_log.db")
		if err := snapshotSQLite(ctx, p, dst); err != nil {
			log.Printf("portable: %v", err)
		} else {
			out = append(out, localFile{path: dst, name: "loss_log.db"})
		}
	}
	for _, name := range []string{"log.txt", ".job_config.json"} {
		if p := filepath.Join(folder, name); exists(p) {
			dst := filepath.Join(tmp, name)
			if err := copyFile(p, dst); err == nil {
				out = append(out, localFile{path: dst, name: name})
			}
		}
	}
	return out, nil
}

func exists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// syncSamples uploads samples the remote lacks (best effort).
func syncSamples(ctx context.Context, c remote.Client, name, folder string, prev *Manifest) []File {
	known := map[string]File{}
	var out []File
	if prev != nil {
		for _, f := range prev.Samples {
			known[f.Name] = f
			out = append(out, f)
		}
	}
	smp, err := artifacts.Samples(folder)
	if err != nil {
		return out
	}
	for _, s := range smp {
		if _, ok := known[s.Name]; ok {
			continue
		}
		sha, size, err := hashFile(ctx, s.Path)
		if err != nil {
			continue
		}
		key := jobKey(name, "samples/"+s.Name)
		if _, err := c.Put(ctx, key, s.Path, nil); err != nil {
			log.Printf("portable: sample %s: %v", s.Name, err)
			continue
		}
		out = append(out, File{Name: s.Name, Key: key, Size: size, SHA256: sha, Step: s.Step})
	}
	if out == nil {
		out = []File{}
	}
	return out
}

var modelExts = []string{".safetensors", ".ckpt", ".pt", ".bin"}

// keepCheckpoints reads the remote's keep_checkpoints config: true keeps
// every old checkpoint, a number N keeps the N newest, unset keeps none.
func keepCheckpoints(r *remote.Remote) (all bool, n int) {
	var c struct {
		Keep json.RawMessage `json:"keep_checkpoints"`
	}
	if json.Unmarshal(r.Config, &c) != nil || len(c.Keep) == 0 {
		return false, 0
	}
	if string(c.Keep) == "true" {
		return true, 0
	}
	if v, err := strconv.Atoi(string(c.Keep)); err == nil && v > 0 {
		return false, v
	}
	return false, 0
}

// stepOf parses <name>_<step>.<ext>; -1 when there is no step.
func stepOf(file string) int {
	base := strings.TrimSuffix(file, filepath.Ext(file))
	i := strings.LastIndex(base, "_")
	if i < 0 {
		return -1
	}
	n, err := strconv.Atoi(base[i+1:])
	if err != nil {
		return -1
	}
	return n
}

func isCheckpoint(file string) bool {
	return file != "optimizer.pt" && slices.Contains(modelExts, strings.ToLower(filepath.Ext(file)))
}

// prune deletes files/ objects the new manifest no longer references, except
// old checkpoints the remote is configured to keep.
func prune(ctx context.Context, c remote.Client, r *remote.Remote, name string, current []File, pol state.Policy) error {
	objs, err := c.List(ctx, jobKey(name, "files/"))
	if err != nil {
		return err
	}
	ref := map[string]bool{}
	for _, f := range current {
		ref[f.Key] = true
	}
	keepAll, keepN := keepCheckpoints(r)
	if n, ok := state.KeepCheckpoints(ctx, pol); ok {
		keepAll, keepN = n < 0, max(n, 0)
	}
	type old struct {
		key  string
		step int
	}
	var oldCheckpoints []old
	for _, o := range objs {
		if ref[o.Key] {
			continue
		}
		file := o.Key[strings.LastIndex(o.Key, "/")+1:]
		if isCheckpoint(file) {
			if !keepAll {
				oldCheckpoints = append(oldCheckpoints, old{o.Key, stepOf(file)})
			}
			continue
		}
		// superseded optimizer state, loss log, log, config
		if err := c.Delete(ctx, o.Key); err != nil {
			return err
		}
	}
	sort.Slice(oldCheckpoints, func(i, k int) bool { return oldCheckpoints[i].step > oldCheckpoints[k].step })
	for i, o := range oldCheckpoints {
		if i < keepN {
			continue
		}
		if err := c.Delete(ctx, o.key); err != nil {
			return err
		}
	}
	return nil
}

// CurrentStatus is the stored sync record plus the lease as the remote has it.
func CurrentStatus(ctx context.Context, j *store.Job) (Status, *Lease, error) {
	st, err := GetStatus(ctx, j.ID)
	if err != nil {
		return st, nil, err
	}
	r, c, _, err := syncTarget(ctx, j)
	if err != nil {
		return st, nil, err
	}
	if st.Remote == "" {
		st.Remote = r.Name
	}
	m, err := readManifest(ctx, c, j.Name)
	if err != nil || m == nil {
		return st, nil, err
	}
	return st, m.Lease, nil
}

// stopForLostLease stops the local trainer when another node took the job.
func stopForLostLease(ctx context.Context, j *store.Job, holder *Lease) {
	if j.Status != "running" && j.Status != "stopping" && j.Status != "queued" {
		return
	}
	label := "another node"
	if holder != nil {
		label = holder.NodeLabel
	}
	info := "Stopping: lease taken over by " + label
	fields := store.Fields{"stop": true, "info": info}
	if j.Status == "queued" {
		fields = store.Fields{"status": "stopped", "info": info}
	}
	if _, err := store.UpdateJob(ctx, j.ID, fields); err != nil {
		log.Printf("portable: stop %s after lease loss: %v", j.Name, err)
		return
	}
	log.Printf("portable: job %s: %s", j.Name, info)
	events.Emit(ctx, JobSync, j.ID, map[string]any{"action": "lease_lost", "step": j.Step, "files": []string{},
		"error": info})
}

// RenewLease extends this node's lease on an active synced job. It reports
// lease loss (another live node holds it) as ErrLeaseLost and stops the job.
func RenewLease(ctx context.Context, j *store.Job) error {
	_, c, _, err := syncTarget(ctx, j)
	if err != nil {
		return err
	}
	holder, err := renewLease(ctx, c, j)
	if errors.Is(err, ErrLeaseLost) {
		stopForLostLease(ctx, j, holder)
	}
	return err
}

func renewLease(ctx context.Context, c remote.Client, j *store.Job) (*Lease, error) {
	unlock := lockManifest(j.Name)
	defer unlock()
	m, err := readManifest(ctx, c, j.Name)
	if err != nil || m == nil {
		return nil, err // no manifest yet: the start sync writes it
	}
	me, err := state.NodeID(ctx)
	if err != nil {
		return nil, err
	}
	if m.Lease.Held(me, time.Now()) {
		return m.Lease, ErrLeaseLost
	}
	if m.RunID != "" && m.RunID != RunID(ctx, j.ID) {
		return nil, fmt.Errorf("%w: %q", ErrOtherRun, j.Name)
	}
	if m.Lease, err = newLease(ctx); err != nil {
		return nil, err
	}
	m.Step, m.TotalSteps = j.Step, j.TotalSteps
	m.State = jobstate.FromStatus(j.Status, j.Step, j.Info)
	return m.Lease, writeManifest(ctx, c, m)
}
