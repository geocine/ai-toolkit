// Package jobview is the client-facing Job of /v1 (docs/headless.md, Jobs):
// upstream's row plus the fork's state. Every /v1 surface that returns a job
// uses it, so they all answer the same shape.
package jobview

import (
	"context"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Policy is the effective policy (defaults applied).
type Policy struct {
	AutoResume       bool    `json:"auto_resume"`
	MaxAutoResumes   int     `json:"max_auto_resumes"`
	AutoUploadRemote *string `json:"auto_upload_remote"`
	SyncRemote       *string `json:"sync_remote"`
}

// Job is the /v1 job.
type Job struct {
	ID               string                `json:"id"`
	Name             string                `json:"name"`
	Type             string                `json:"type"`
	Ref              *string               `json:"ref"`
	GpuIDs           string                `json:"gpu_ids"`
	State            string                `json:"state"`
	Status           string                `json:"status"`
	Info             string                `json:"info"`
	Step             int                   `json:"step"`
	TotalSteps       *int                  `json:"total_steps"`
	Speed            string                `json:"speed"`
	QueuePosition    int                   `json:"queue_position"`
	Pid              *int                  `json:"pid"`
	CreatedAt        string                `json:"created_at"`
	UpdatedAt        string                `json:"updated_at"`
	Config           any                   `json:"config"`
	Policy           Policy                `json:"policy"`
	Attempts         int                   `json:"attempts"`
	LastAttempt      *state.Attempt        `json:"last_attempt"`
	LatestCheckpoint *artifacts.Checkpoint `json:"latest_checkpoint"`
}

// Folder is the job's folder under the training root.
func Folder(ctx context.Context, name string) (string, error) {
	root, err := settings.TrainingFolder(ctx)
	if err != nil {
		return "", err
	}
	return filepath.Join(root, name), nil
}

// View builds the client-facing job.
func View(ctx context.Context, j *store.Job) (*Job, error) {
	v := &Job{
		ID: j.ID, Name: j.Name, Type: j.JobType, Ref: j.JobRef, GpuIDs: j.GpuIDs,
		State:  jobstate.FromStatus(j.Status, j.Step, j.Info),
		Status: j.Status, Info: j.Info, Step: j.Step, TotalSteps: j.TotalSteps, Speed: j.SpeedString,
		QueuePosition: j.QueuePosition, Pid: j.Pid,
		CreatedAt: v1api.Time(j.CreatedAt.Time), UpdatedAt: v1api.Time(j.UpdatedAt.Time),
	}
	if cfg, err := jsonx.Parse([]byte(j.JobConfig)); err == nil {
		v.Config = cfg
	} else {
		v.Config = j.JobConfig // not JSON: hand it back as stored
	}
	auto, max, upload, err := state.EffectivePolicy(ctx, j.ID)
	if err != nil {
		return nil, err
	}
	p, err := state.GetPolicy(ctx, j.ID)
	if err != nil {
		return nil, err
	}
	v.Policy = Policy{AutoResume: auto, MaxAutoResumes: max, SyncRemote: p.SyncRemote}
	if upload != "" {
		v.Policy.AutoUploadRemote = &upload
	}
	attempts, err := state.Attempts(ctx, j.ID)
	if err != nil {
		return nil, err
	}
	v.Attempts = len(attempts)
	if len(attempts) > 0 {
		v.LastAttempt = attempts[len(attempts)-1]
	}
	if folder, err := Folder(ctx, j.Name); err == nil {
		if cps, err := artifacts.Checkpoints(folder); err == nil && len(cps) > 0 {
			latest := cps[len(cps)-1]
			v.LatestCheckpoint = &latest
		}
	}
	return v, nil
}
