package v1

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"runtime"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/jobview"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/supervisor"
	"github.com/geocine/aitoolkit/internal/templates"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// captionJobs are the dataset folder's caption jobs, newest first (upstream
// links them by job_ref = the folder, like its dataset page does).
func captionJobs(ctx context.Context, dir string) ([]*store.Job, error) {
	return store.FindJobs(ctx, store.JobQuery{Where: `"job_type" = 'caption' AND "job_ref" = ?`, Args: []any{dir},
		OrderBy: `"created_at" DESC`})
}

// latestCaptionJob is the dataset's newest caption job, or nil.
func latestCaptionJob(ctx context.Context, dir string) (*jobview.Job, error) {
	jobs, err := captionJobs(ctx, dir)
	if err != nil || len(jobs) == 0 {
		return nil, err
	}
	return jobview.View(ctx, jobs[0])
}

// captionDataset creates a caption job for the dataset with upstream's
// captioner defaults and queues it: POST /v1/datasets/{name}/caption.
func captionDataset(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	var b struct {
		templates.CaptionRequest
		GpuIDs *string `json:"gpu_ids"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	jobs, err := captionJobs(ctx, dir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	for _, j := range jobs {
		if busy(j.Status) {
			v1api.Fail(w, v1api.CodeConflict, "already captioning this dataset: job "+j.Name)
			return
		}
	}
	jobName := name + "_caption_" + time.Now().UTC().Format("20060102_150405")
	res, err := templates.RenderCaption(ctx, b.CaptionRequest, dir, jobName)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	raw, err := json.Marshal(res.Config)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	cfg, err := checkConfig(raw)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	gpu := "0"
	if b.GpuIDs != nil && strings.TrimSpace(*b.GpuIDs) != "" {
		gpu = strings.TrimSpace(*b.GpuIDs)
	}
	if runtime.GOOS == "darwin" {
		gpu = "mps"
	}
	max, err := store.MaxQueuePosition(ctx)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	typ := "caption"
	j, err := store.CreateJob(ctx, store.NewJob{Name: jobName, GpuIDs: gpu, JobConfig: cfg,
		QueuePosition: max + 1000, JobType: &typ, JobRef: &dir})
	if errors.Is(err, store.ErrUniqueViolation) {
		v1api.Fail(w, v1api.CodeConflict, "a job named "+jobName+" already exists")
		return
	}
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	// a caption job writes into the dataset: nothing to sync, upload or
	// resume elsewhere. Explicit "" and false, because nil fields take the
	// server's defaults (AITK_SYNC_REMOTE, ...) on the supervisor's next pass.
	none, no := "", false
	if err := state.SetPolicy(ctx, j.ID, state.Policy{AutoResume: &no, AutoUploadRemote: &none, SyncRemote: &none}); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if err := supervisor.Requeue(ctx, j, "Job queued"); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if j, err = store.GetJob(ctx, j.ID); err != nil || j == nil {
		v1api.Internal(w, r, errors.Join(err, errors.New("caption job vanished")))
		return
	}
	v, err := jobview.View(ctx, j)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.JSON(w, http.StatusCreated, struct {
		*jobview.Job
		Warnings []string `json:"warnings"`
	}{v, res.Warnings})
}
