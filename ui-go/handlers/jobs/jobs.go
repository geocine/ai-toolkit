// Package jobs ports the job lifecycle routes: ui/src/app/api/jobs/route.ts
// and jobs/[jobID]/{start,stop,delete,mark_stopped,kill,sample_now,save_now}.
// Job artifacts (files, log, loss, notes, plugin, samples) live in jobfiles.
package jobs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/geocine/aitoolkit/internal/apicache"
	"github.com/geocine/aitoolkit/internal/engines"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/proc"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /api/jobs", list)
	httpx.Handle(mux, "POST /api/jobs", save)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/start", start)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/stop", stop)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/delete", del)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/mark_stopped", markStopped)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/kill", kill)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/sample_now", flagNow("sample_now", "sample"))
	httpx.Handle(mux, "GET /api/jobs/{jobID}/save_now", flagNow("save_now", "save"))
}

func list(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	q := r.URL.Query()
	fail := func(err error) {
		log.Printf("GET /api/jobs: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to fetch training data")
	}

	if id := q.Get("id"); id != "" {
		job, err := store.GetJob(ctx, id)
		if err != nil {
			fail(err)
			return
		}
		httpx.OK(w, job)
		return
	}
	if ref := q.Get("job_ref"); ref != "" {
		job, err := store.FindFirstJob(ctx, store.JobQuery{
			Where: `"job_ref" = ?`, Args: []any{ref}, OrderBy: `"updated_at" DESC`,
		})
		if err != nil {
			fail(err)
			return
		}
		httpx.OK(w, job)
		return
	}

	query := store.JobQuery{OrderBy: `"created_at" DESC`}
	var jobType *string
	if t := q.Get("job_type"); t != "" {
		jobType = &t
		query.Where, query.Args = `"job_type" = ?`, []any{t}
	}
	if q.Get("only_active") == "true" {
		active, args := store.In("status", "running", "queued", "stopping")
		if query.Where != "" {
			query.Where += " AND "
		}
		query.Where += active
		query.Args = append(query.Args, args...)
		jobs, err := apicache.Cached("jobs-active", 5*time.Second, map[string]*string{"job_type": jobType},
			func() ([]*store.Job, error) { return store.FindJobs(ctx, query) })
		if err != nil {
			fail(err)
			return
		}
		httpx.OK(w, map[string]any{"jobs": jobs})
		return
	}
	jobs, err := store.FindJobs(ctx, query)
	if err != nil {
		fail(err)
		return
	}
	httpx.OK(w, map[string]any{"jobs": jobs})
}

// save creates or updates a job (POST /api/jobs). Fields absent from the body
// are left alone on update, like Prisma ignoring `undefined`.
func save(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	fail := func(err error) {
		if errors.Is(err, store.ErrUniqueViolation) {
			httpx.Error(w, http.StatusConflict, "Job name already exists")
			return
		}
		log.Printf("POST /api/jobs: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to save training data")
	}

	var body map[string]json.RawMessage
	if err := httpx.DecodeJSON(r, &body); err != nil {
		fail(err)
		return
	}
	str := func(key string) (string, bool, error) {
		raw, ok := body[key]
		if !ok {
			return "", false, nil
		}
		var s string
		err := json.Unmarshal(raw, &s)
		return s, true, err
	}

	f := store.Fields{}
	name, hasName, err := str("name")
	if err != nil {
		fail(err)
		return
	}
	if hasName {
		f["name"] = name
	}
	gpuIDs, hasGPU, err := str("gpu_ids")
	if err != nil {
		fail(err)
		return
	}
	if runtime.GOOS == "darwin" {
		gpuIDs, hasGPU = "mps", true
	}
	if hasGPU {
		f["gpu_ids"] = gpuIDs
	}
	// job_config arrives as an object and is stored as its JSON text
	var jobConfig string
	raw, hasConfig := body["job_config"]
	if hasConfig {
		var buf bytes.Buffer
		if err := json.Compact(&buf, raw); err != nil {
			fail(err)
			return
		}
		jobConfig = buf.String()
		f["job_config"] = jobConfig
	}
	var jobRef, jobType *string
	if raw, ok := body["job_ref"]; ok {
		if err := json.Unmarshal(raw, &jobRef); err != nil {
			fail(err)
			return
		}
		f["job_ref"] = jobRef
	}
	if raw, ok := body["job_type"]; ok {
		if err := json.Unmarshal(raw, &jobType); err != nil || jobType == nil {
			fail(errors.New("job_type must be a string"))
			return
		}
		f["job_type"] = *jobType
	}

	id, _, err := str("id")
	if err != nil {
		fail(err)
		return
	}
	if id != "" {
		job, err := store.UpdateJob(ctx, id, f)
		if err != nil {
			fail(err)
			return
		}
		httpx.OK(w, job)
		return
	}

	if !hasName || !hasGPU || !hasConfig {
		fail(errors.New("name, gpu_ids and job_config are required"))
		return
	}
	max, err := store.MaxQueuePosition(ctx)
	if err != nil {
		fail(err)
		return
	}
	job, err := store.CreateJob(ctx, store.NewJob{
		Name: name, GpuIDs: gpuIDs, JobConfig: jobConfig,
		QueuePosition: max + 1000, JobType: jobType, JobRef: jobRef,
	})
	if err != nil {
		fail(err)
		return
	}
	httpx.OK(w, job)
}

// jobOr404 loads the path's job, answering 404 when it does not exist.
func jobOr404(w http.ResponseWriter, r *http.Request) (*store.Job, bool) {
	job, err := store.GetJob(r.Context(), r.PathValue("jobID"))
	if err != nil {
		httpx.Internal(w, r, err)
		return nil, false
	}
	if job == nil {
		httpx.Error(w, http.StatusNotFound, "Job not found")
		return nil, false
	}
	return job, true
}

// start queues the job and returns it as it was before queueing.
func start(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	job, ok := jobOr404(w, r)
	if !ok {
		return
	}
	max, err := store.MaxQueuePosition(ctx)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if _, err := store.UpdateJob(ctx, job.ID, store.Fields{"queue_position": max + 1000}); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	queue, err := store.GetQueueByGPU(ctx, job.GpuIDs)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if queue == nil {
		if _, err := store.CreateQueue(ctx, job.GpuIDs, false); err != nil {
			httpx.Internal(w, r, err)
			return
		}
	}
	if _, err := store.UpdateJob(ctx, job.ID, store.Fields{
		"status": "queued", "stop": false, "return_to_queue": false, "info": "Job queued",
	}); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	httpx.OK(w, job)
}

// How long a stopping job gets for its graceful shutdown (KeyboardInterrupt ->
// on_error -> final DB write) before it is assumed hung and force-killed.
const (
	gracefulStopTimeout = 60 * time.Second
	backstopPoll        = 2 * time.Second
)

// scheduleForceKillBackstop (Windows) force-kills a job too hung to notice the
// stop flag. Polls so it stops watching the moment the pid dies and never
// touches a recycled pid.
func scheduleForceKillBackstop(pid int, jobID string) {
	go func() {
		started := time.Now()
		t := time.NewTicker(backstopPoll)
		defer t.Stop()
		for range t.C {
			if !proc.Alive(pid) {
				return
			}
			if time.Since(started) < gracefulStopTimeout {
				continue
			}
			log.Printf("Job %s (pid %d) still alive %ds after stop request, force killing", jobID, pid, int(gracefulStopTimeout.Seconds()))
			_ = proc.TreeKill(context.Background(), pid)
			return
		}
	}()
}

func stop(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	job, ok := jobOr404(w, r)
	if !ok {
		return
	}
	if _, err := store.UpdateJob(ctx, job.ID, store.Fields{"stop": true, "info": "Stopping job..."}); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if job.Pid != nil {
		log.Printf("Attempting to stop job %s with PID %d", job.ID, *job.Pid)
		var err error
		if runtime.GOOS == "windows" {
			// the stop flag written above is the signal; see proc.Interrupt
			scheduleForceKillBackstop(*job.Pid, job.ID)
		} else {
			err = proc.Interrupt(*job.Pid)
		}
		if err != nil {
			// process may have already exited, that's fine
			log.Printf("Error sending signal to process: %v", err)
		} else if _, err := store.UpdateJob(ctx, job.ID, store.Fields{"status": "stopped", "info": "Job stopped"}); err != nil {
			log.Printf("Error marking job stopped: %v", err)
		}
	} else {
		log.Printf("No PID found for job %s, cannot send stop signal", job.ID)
	}
	httpx.OK(w, job)
}

func del(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	job, ok := jobOr404(w, r)
	if !ok {
		return
	}
	root, err := settings.TrainingFolder(ctx)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if job.Status == "running" || job.Status == "stopping" {
		httpx.Error(w, http.StatusConflict, "Job is "+job.Status+"; stop it before deleting it")
		return
	}
	// a name like "", "." or ".." (or one with separators) must never resolve
	// to the training root or outside it
	folder := filepath.Clean(filepath.Join(root, job.Name))
	if job.Name != "" && filepath.Base(folder) == job.Name && filepath.Dir(folder) == filepath.Clean(root) {
		if err := os.RemoveAll(folder); err != nil {
			httpx.Internal(w, r, err)
			return
		}
	}
	if err := store.DeleteJob(ctx, job.ID); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	httpx.OK(w, job)
}

// markStopped returns the job as it was (null when missing; the update then
// fails like Prisma's and answers 500).
func markStopped(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("jobID")
	job, err := store.GetJob(ctx, id)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if _, err := store.UpdateJob(ctx, id, store.Fields{
		"stop": true, "status": "stopped", "info": "Job stopped", "pid": nil,
	}); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	log.Printf("Job %s marked as stopped", id)
	httpx.OK(w, job)
}

// kill force-stops a job that ignored the graceful stop (a hung model load).
func kill(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("jobID")
	job, err := store.GetJob(ctx, id)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	// the row's pid, else (inference engines) the one in its engine.json
	pid := 0
	if job != nil && job.Pid != nil {
		pid = *job.Pid
	}
	if pid == 0 {
		handles, err := engines.List(ctx, true)
		if err != nil {
			httpx.Internal(w, r, err)
			return
		}
		for _, h := range handles {
			if h.JobID == id && h.Endpoint != nil && h.Endpoint.Pid != 0 {
				pid = h.Endpoint.Pid
				break
			}
		}
	}
	if job == nil && pid == 0 {
		httpx.Error(w, http.StatusNotFound, "Job not found")
		return
	}
	killed := false
	if pid != 0 {
		if err := proc.TreeKill(ctx, pid); err != nil {
			log.Printf("kill %s pid %d: %v", id, pid, err)
		} else {
			killed = true
		}
	}
	if job != nil {
		if _, err := store.UpdateJob(ctx, id, store.Fields{
			"stop": true, "status": "stopped", "info": "Job force stopped", "pid": nil,
		}); err != nil {
			httpx.Internal(w, r, err)
			return
		}
	}
	httpx.OK(w, map[string]any{"ok": true, "killed": killed})
}

// flagNow sets save_now / sample_now so the trainer acts on its next step.
func flagNow(column, verb string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("jobID")
		job, err := store.UpdateJob(r.Context(), id, store.Fields{column: true})
		if err != nil {
			httpx.Internal(w, r, err)
			return
		}
		log.Printf("Job %s marked to %s on next step", id, verb)
		httpx.OK(w, job)
	}
}
