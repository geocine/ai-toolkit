package photocheck

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// FaceJobType is the job_type of a face check.
const FaceJobType = "photo_faces"

// FaceResultFile is what the face job writes in its job folder.
const FaceResultFile = "faces.json"

// LatestFaceJob is the newest photo_faces job for a dataset folder (its
// job_ref), or nil when there is none or the job database is not open.
func LatestFaceJob(ctx context.Context, dir string) (*store.Job, error) {
	if store.DB == nil {
		return nil, nil
	}
	return store.FindFirstJob(ctx, store.JobQuery{
		Where:   `"job_type" = ? AND "job_ref" = ?`,
		Args:    []any{FaceJobType, filepath.Clean(dir)},
		OrderBy: `"created_at" DESC`,
	})
}

// faceJobState maps a job's status onto FacesState.State.
func faceJobState(j *store.Job) string {
	if j == nil {
		return "none"
	}
	switch j.Status {
	case "queued":
		return "queued"
	case "running", "stopping":
		return "running"
	case "completed":
		return "done"
	case "error":
		return "failed"
	case "stopped":
		return "stopped"
	}
	return "none"
}

// ingestFaces reads a finished face job into the cache (section 12.4): when
// the newest photo_faces job for dir has completed and is not the one the
// stored identity came from, its faces.json rows go into photo_faces, its
// matrix and set signature into photo_sets.identity, and the file is
// deleted (job folders travel in bundles). A second call is a no-op, and a
// missing file after an ingest is fine. set is updated in place. It
// returns the job (nil when none).
func ingestFaces(ctx context.Context, dir string, set *state.PhotoSet) (*store.Job, error) {
	job, err := LatestFaceJob(ctx, dir)
	if err != nil || job == nil || job.Status != "completed" {
		return job, err
	}
	if id := storedIdentity(set); id != nil && id.JobID == job.ID {
		return job, nil
	}
	training, err := settings.TrainingFolder(ctx)
	if err != nil {
		return job, err
	}
	file := filepath.Join(training, job.Name, FaceResultFile)
	data, err := os.ReadFile(file)
	if errors.Is(err, fs.ErrNotExist) {
		return job, nil
	}
	if err != nil {
		return job, err
	}
	var res FaceJobResult
	if err := json.Unmarshal(data, &res); err != nil {
		return job, fmt.Errorf("read %s: %w", file, err)
	}
	algo := res.Algo
	if algo == 0 {
		algo = FaceAlgo
	}
	rows := make(map[string][]byte, len(res.Items))
	shas := make([]string, 0, len(res.Items))
	for sha, fd := range res.Items {
		if fd.Algo == 0 {
			fd.Algo = algo
		}
		b, err := json.Marshal(fd)
		if err != nil {
			return job, err
		}
		rows[sha] = b
		shas = append(shas, sha)
	}
	if listed := jobShas(job.JobConfig); len(listed) > 0 {
		shas = listed
	}
	if err := state.PutPhotoFaces(ctx, rows, algo); err != nil {
		return job, err
	}
	model := res.Model
	if model == "" {
		model = FaceModel
	}
	id := Identity{FaceMatrix: res.Matrix, SetSig: SetSig(shas), Model: model, JobID: job.ID, At: state.NowMs()}
	if id.Faces == nil {
		id.Faces = []string{}
	}
	b, err := json.Marshal(id)
	if err != nil {
		return job, err
	}
	set.Identity = b
	set.UpdatedAt = 0
	if err := state.PutPhotoSet(ctx, set); err != nil {
		return job, err
	}
	os.Remove(file)
	return job, nil
}

// jobShas are the photo shas a face job's config lists (section 12.2).
func jobShas(cfg string) []string {
	var c struct {
		Config struct {
			Process []struct {
				Files []struct {
					SHA256 string `json:"sha256"`
				} `json:"files"`
			} `json:"process"`
		} `json:"config"`
	}
	if json.Unmarshal([]byte(cfg), &c) != nil {
		return nil
	}
	var out []string
	for _, p := range c.Config.Process {
		for _, f := range p.Files {
			if f.SHA256 != "" {
				out = append(out, f.SHA256)
			}
		}
	}
	return out
}

// storedIdentity decodes the stored identity, nil when none or unreadable.
func storedIdentity(set *state.PhotoSet) *Identity {
	if set == nil || len(set.Identity) == 0 {
		return nil
	}
	var id Identity
	if json.Unmarshal(set.Identity, &id) != nil {
		return nil
	}
	return &id
}

// facesState is the face check's state for the set: the job's state, and
// whether the set changed since the stored check (stale: the current shas'
// signature differs; added: current photos with no face row).
func facesState(job *store.Job, id *Identity, shas []string, faced map[string]bool) FacesState {
	st := FacesState{Available: FacesAvailable, State: faceJobState(job)}
	if job != nil {
		st.JobID = job.ID
	}
	if id == nil {
		return st
	}
	st.Stale = id.SetSig != SetSig(shas)
	for _, s := range shas {
		if !faced[s] {
			st.Added++
		}
	}
	return st
}
