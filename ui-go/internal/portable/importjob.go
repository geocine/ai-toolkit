package portable

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/supervisor"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// RemoteJob summarizes a portable job on a remote.
type RemoteJob struct {
	Name      string    `json:"name"`
	Step      int       `json:"step"`
	State     string    `json:"state"`
	Lease     *Lease    `json:"lease"`
	UpdatedAt time.Time `json:"updated_at"`
	Datasets  []string  `json:"datasets"`
}

// ListRemoteJobs reads every manifest under jobs/ on the remote.
func ListRemoteJobs(ctx context.Context, remoteName string) ([]RemoteJob, error) {
	_, c, err := remote.Open(ctx, remoteName)
	if err != nil {
		return nil, remoteErr(remoteName, err)
	}
	objs, err := c.List(ctx, "jobs/")
	if err != nil {
		return nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	out := []RemoteJob{}
	for _, o := range objs {
		rest, ok := strings.CutPrefix(o.Key, "jobs/")
		if !ok {
			continue
		}
		name, file, ok := strings.Cut(rest, "/")
		if !ok || file != "manifest.json" {
			continue
		}
		m, err := readManifest(ctx, c, name)
		if err != nil || m == nil {
			continue
		}
		out = append(out, RemoteJob{Name: m.Name, Step: m.Step, State: m.State, Lease: m.Lease,
			UpdatedAt: m.UpdatedAt, Datasets: m.Datasets})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

func remoteErr(name string, err error) error {
	if errors.Is(err, remote.ErrNotFound) {
		return v1api.Errorf(v1api.CodeNotFound, fmt.Sprintf("remote %q not found", name))
	}
	return err
}

// ImportRequest is POST /v1/portable/import.
type ImportRequest struct {
	Remote string  `json:"remote"`
	Name   string  `json:"name"`
	GpuIDs *string `json:"gpu_ids"`
	Start  bool    `json:"start"`
	Force  bool    `json:"force"`
}

// Import brings a portable job to this server so it can continue from its
// newest checkpoint. Errors are v1api problems (409 lease/name/dataset
// conflicts, 404 missing remote or job, 502 corrupt or unreachable remote).
func Import(ctx context.Context, req ImportRequest) (*store.Job, error) {
	if req.Remote == "" || req.Name == "" {
		return nil, v1api.Errorf(v1api.CodeInvalidRequest, "remote and name are required")
	}
	if err := ValidName(req.Name); err != nil {
		return nil, v1api.Errorf(v1api.CodeInvalidRequest, err.Error())
	}
	rem, c, err := remote.Open(ctx, req.Remote)
	if err != nil {
		return nil, remoteErr(req.Remote, err)
	}
	if existing, err := store.FindFirstJob(ctx, store.JobQuery{Where: `"name" = ?`, Args: []any{req.Name}}); err != nil {
		return nil, err
	} else if existing != nil {
		return nil, v1api.Errorf(v1api.CodeConflict, fmt.Sprintf("a job named %q already exists on this server", req.Name))
	}
	dsRoot, err := settings.DatasetsRoot(ctx)
	if err != nil {
		return nil, err
	}
	trRoot, err := settings.TrainingFolder(ctx)
	if err != nil {
		return nil, err
	}

	// take the lease first so two servers cannot import the same job at once
	m, prevLease, err := claim(ctx, c, req)
	if err != nil {
		return nil, err
	}
	release := func() {
		unlock := lockManifest(req.Name)
		defer unlock()
		if cur, err := readManifest(ctx, c, req.Name); err == nil && cur != nil {
			cur.Lease = prevLease
			writeManifest(ctx, c, cur)
		}
	}

	job, err := restore(ctx, c, rem, m, req, dsRoot, trRoot)
	if err != nil {
		release()
		return nil, err
	}
	if req.Start {
		if err := supervisor.Requeue(ctx, job, "Queued after import"); err != nil {
			return nil, err
		}
		if job, err = store.GetJob(ctx, job.ID); err != nil {
			return nil, err
		}
	}
	return job, nil
}

func claim(ctx context.Context, c remote.Client, req ImportRequest) (*Manifest, *Lease, error) {
	unlock := lockManifest(req.Name)
	defer unlock()
	m, err := readManifest(ctx, c, req.Name)
	if err != nil {
		return nil, nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	if m == nil {
		return nil, nil, v1api.Errorf(v1api.CodeNotFound, fmt.Sprintf("no portable job %q on remote %q", req.Name, req.Remote))
	}
	me, err := state.NodeID(ctx)
	if err != nil {
		return nil, nil, err
	}
	if m.Lease.Held(me, time.Now()) && !req.Force {
		return nil, nil, v1api.Errorf(v1api.CodeConflict, fmt.Sprintf("job %q is leased by %s until %s (use force to take it over)",
			req.Name, m.Lease.NodeLabel, m.Lease.ExpiresAt.Format(time.RFC3339)))
	}
	prev := m.Lease
	if m.Lease, err = newLease(ctx); err != nil {
		return nil, nil, err
	}
	if err := writeManifest(ctx, c, m); err != nil {
		return nil, nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	return m, prev, nil
}

// plainName is a single folder name: what a manifest from the bucket may
// name a job or dataset (it becomes a folder under the training or datasets root).
func plainName(s string) bool {
	return s != "" && !strings.HasPrefix(s, ".") && !strings.ContainsAny(s, `/\:`) && filepath.Base(s) == s
}

func restore(ctx context.Context, c remote.Client, rem *remote.Remote, m *Manifest, req ImportRequest, dsRoot, trRoot string) (*store.Job, error) {
	if !plainName(m.Name) {
		return nil, v1api.Errorf(v1api.CodeInvalidRequest, fmt.Sprintf("the manifest names the job %q, which is not a plain folder name", m.Name))
	}
	for _, name := range m.Datasets {
		if !plainName(name) {
			return nil, v1api.Errorf(v1api.CodeInvalidRequest, fmt.Sprintf("the manifest names a dataset %q, which is not a plain folder name", name))
		}
	}
	for _, name := range m.Datasets {
		if err := restoreDataset(ctx, c, dsRoot, name); err != nil {
			switch {
			case errors.Is(err, ErrDatasetConflict):
				return nil, v1api.Errorf(v1api.CodeConflict, err.Error())
			case errors.Is(err, ErrCorrupt):
				return nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
			}
			return nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
		}
	}

	folder := filepath.Join(trRoot, m.Name)
	if entries, err := os.ReadDir(folder); err == nil && len(entries) > 0 {
		if !req.Force {
			return nil, v1api.Errorf(v1api.CodeConflict, fmt.Sprintf("job folder %s already exists (use force to move it aside)", folder))
		}
		aside := fmt.Sprintf("%s.replaced-%d", folder, time.Now().Unix())
		if err := os.Rename(folder, aside); err != nil {
			return nil, err
		}
	} else if err == nil {
		os.Remove(folder)
	}
	if err := os.MkdirAll(trRoot, 0o755); err != nil {
		return nil, err
	}
	tmp, err := os.MkdirTemp(trRoot, ".import-"+m.Name+"-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	for _, f := range m.Files {
		if !safeRel(f.Name) || strings.Contains(f.Name, "/") {
			return nil, v1api.Errorf(v1api.CodeRemoteFailed, fmt.Sprintf("manifest lists an unsafe file name %q", f.Name))
		}
		if err := getVerified(ctx, c, f.Key, filepath.Join(tmp, f.Name), f.SHA256); err != nil {
			return nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
		}
	}
	if len(m.Samples) > 0 {
		sdir := filepath.Join(tmp, "samples")
		os.MkdirAll(sdir, 0o755)
		for _, s := range m.Samples {
			if safeRel(s.Name) && !strings.Contains(s.Name, "/") {
				getVerified(ctx, c, s.Key, filepath.Join(sdir, s.Name), s.SHA256) // best effort
			}
		}
	}

	config, err := FromPortable(m.Config, dsRoot, trRoot)
	if err != nil {
		return nil, v1api.Errorf(v1api.CodeInvalidConfig, "manifest config: "+err.Error())
	}
	// what the worker writes at launch, so the folder is complete on its own
	if err := writeJobConfig(filepath.Join(tmp, ".job_config.json"), config); err != nil {
		return nil, err
	}
	if err := os.Rename(tmp, folder); err != nil {
		return nil, err
	}

	gpu := m.GpuIDs
	if req.GpuIDs != nil && *req.GpuIDs != "" {
		gpu = *req.GpuIDs
	}
	var ref *string
	if m.Ref != nil {
		if p, ok := localPath(*m.Ref, dsRoot, trRoot); ok {
			ref = &p
		} else {
			ref = m.Ref
		}
	}
	jobType := m.Type
	maxPos, err := store.MaxQueuePosition(ctx)
	if err != nil {
		return nil, err
	}
	job, err := store.CreateJob(ctx, store.NewJob{Name: m.Name, GpuIDs: gpu, JobConfig: config,
		QueuePosition: maxPos + 1000, JobType: &jobType, JobRef: ref})
	if errors.Is(err, store.ErrUniqueViolation) {
		return nil, v1api.Errorf(v1api.CodeConflict, fmt.Sprintf("a job named %q already exists on this server", m.Name))
	}
	if err != nil {
		return nil, err
	}
	if job, err = store.UpdateJob(ctx, job.ID, store.Fields{
		"step": m.Step, "total_steps": m.TotalSteps,
		"info": fmt.Sprintf("Imported from %s at step %d", rem.Name, m.Step),
	}); err != nil {
		return nil, err
	}

	pol := m.Policy
	syncRemote := req.Remote
	pol.SyncRemote = &syncRemote
	if err := state.SetPolicy(ctx, job.ID, pol); err != nil {
		return nil, err
	}
	if err := copyAttempts(ctx, job.ID, m.Attempts); err != nil {
		return nil, err
	}
	now := time.Now().UTC().Truncate(time.Millisecond)
	step := m.Step
	saveStatus(ctx, job.ID, Status{Remote: rem.Name, LastSyncAt: &now, Step: &step})
	return job, nil
}

// writeJobConfig writes .job_config.json the way the worker does at launch.
func writeJobConfig(dst, config string) error {
	cfg, err := jsonx.ParseObject([]byte(config))
	if err != nil {
		return err
	}
	if c := cfg.Object("config"); c != nil {
		if procs := c.Array("process"); len(procs) > 0 {
			if p0, ok := procs[0].(*jsonx.Object); ok {
				p0.Set("sqlite_db_path", paths.DBPath)
			}
		}
	}
	out, err := jsonx.Indent(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, out, 0o644)
}

// copyAttempts keeps the job's history; an attempt left open by the lost
// server is closed as interrupted.
func copyAttempts(ctx context.Context, jobID string, attempts []*state.Attempt) error {
	for _, a := range attempts {
		ended, endState, reason := a.EndedAt, a.EndState, a.EndReason
		if ended == nil {
			t := time.Now().UTC()
			s, r := "interrupted", "the server running it was lost; imported elsewhere"
			ended, endState, reason = &t, &s, &r
		}
		var endMs *int64
		if ended != nil {
			v := ended.UnixMilli()
			endMs = &v
		}
		if _, err := state.DB.ExecContext(ctx, `INSERT OR IGNORE INTO job_attempts
			(job_id, n, started_at, ended_at, end_state, end_reason, start_step, end_step, auto, pid)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)`,
			jobID, a.N, a.StartedAt.UnixMilli(), endMs, endState, reason, a.StartStep, a.EndStep, a.Auto); err != nil {
			return err
		}
	}
	return nil
}
