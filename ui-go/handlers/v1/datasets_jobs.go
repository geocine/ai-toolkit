package v1

import (
	"context"
	"encoding/json"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/store"
)

// datasetJobs returns the queued, running or stopping jobs that use dir:
// training runs through datasets[].folder_path, caption and photo_faces
// jobs through their job_ref. Callers filter by JobType: a photo fix is
// blocked by train and caption jobs, never by the dataset's own face check.
func datasetJobs(ctx context.Context, dir string) ([]*store.Job, error) {
	jobs, err := store.FindJobs(ctx, store.JobQuery{Where: `"status" IN ('queued', 'running', 'stopping')`})
	if err != nil {
		return nil, err
	}
	clean := filepath.Clean(dir)
	out := []*store.Job{}
	for _, j := range jobs {
		if j.JobRef != nil && filepath.Clean(*j.JobRef) == clean {
			out = append(out, j)
			continue
		}
		if readsFolder(j.JobConfig, clean) {
			out = append(out, j)
		}
	}
	return out, nil
}

// readsFolder reports whether a job config lists folder (clean) as one of
// its datasets' folder_path.
func readsFolder(jobConfig, folder string) bool {
	var cfg struct {
		Config struct {
			Process []struct {
				Datasets []struct {
					FolderPath string `json:"folder_path"`
				} `json:"datasets"`
			} `json:"process"`
		} `json:"config"`
	}
	if json.Unmarshal([]byte(jobConfig), &cfg) != nil {
		return false
	}
	for _, p := range cfg.Config.Process {
		for _, d := range p.Datasets {
			if d.FolderPath != "" && filepath.Clean(d.FolderPath) == folder {
				return true
			}
		}
	}
	return false
}
