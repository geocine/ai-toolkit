package v1

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
	"strings"
	"time"
	"unicode"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/engines"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/proc"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/supervisor"
	"github.com/geocine/aitoolkit/internal/v1api"
)

type policyView struct {
	AutoResume       bool    `json:"auto_resume"`
	MaxAutoResumes   int     `json:"max_auto_resumes"`
	AutoUploadRemote *string `json:"auto_upload_remote"`
	SyncRemote       *string `json:"sync_remote"`
}

type jobView struct {
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
	Policy           policyView            `json:"policy"`
	Attempts         int                   `json:"attempts"`
	LastAttempt      *state.Attempt        `json:"last_attempt"`
	LatestCheckpoint *artifacts.Checkpoint `json:"latest_checkpoint"`
}

func jobFolder(ctx context.Context, name string) (string, error) {
	root, err := settings.TrainingFolder(ctx)
	if err != nil {
		return "", err
	}
	return filepath.Join(root, name), nil
}

// view builds the client-facing job.
func view(ctx context.Context, j *store.Job) (*jobView, error) {
	v := &jobView{
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
	v.Policy = policyView{AutoResume: auto, MaxAutoResumes: max, SyncRemote: p.SyncRemote}
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
	if folder, err := jobFolder(ctx, j.Name); err == nil {
		if cps, err := artifacts.Checkpoints(folder); err == nil && len(cps) > 0 {
			latest := cps[len(cps)-1]
			v.LatestCheckpoint = &latest
		}
	}
	return v, nil
}

// loadJob reads the path's job, answering 404 itself when missing.
func loadJob(w http.ResponseWriter, r *http.Request) (*store.Job, bool) {
	j, err := store.GetJob(r.Context(), r.PathValue("id"))
	if err != nil {
		v1api.Internal(w, r, err)
		return nil, false
	}
	if j == nil {
		v1api.Fail(w, v1api.CodeNotFound, "job not found")
		return nil, false
	}
	return j, true
}

// respondJob re-reads the job and writes its view with status.
func respondJob(w http.ResponseWriter, r *http.Request, status int, id string) {
	j, err := store.GetJob(r.Context(), id)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if j == nil {
		v1api.Fail(w, v1api.CodeNotFound, "job not found")
		return
	}
	v, err := view(r.Context(), j)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.JSON(w, status, v)
}

func busy(status string) bool {
	return status == "queued" || status == "running" || status == "stopping"
}

func listJobs(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	q := store.JobQuery{OrderBy: `"created_at" DESC`}
	if t := r.URL.Query().Get("type"); t != "" {
		q.Where, q.Args = `"job_type" = ?`, []any{t}
	}
	jobs, err := store.FindJobs(ctx, q)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	want := r.URL.Query().Get("state")
	items := []*jobView{}
	for _, j := range jobs {
		v, err := view(ctx, j)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if want == "" || v.State == want {
			items = append(items, v)
		}
	}
	v1api.Items(w, items)
}

// validJobName keeps names usable as a job folder.
func validJobName(name string) error {
	switch {
	case strings.TrimSpace(name) == "":
		return errors.New("name is required")
	case len(name) > 200:
		return errors.New("name is longer than 200 characters")
	case name == "." || name == ".." || strings.HasPrefix(name, "."):
		return errors.New("name may not start with a dot")
	case strings.ContainsAny(name, `/\:*?"<>|`):
		return errors.New(`name may not contain / \ : * ? " < > |`)
	}
	for _, r := range name {
		if unicode.IsControl(r) {
			return errors.New("name may not contain control characters")
		}
	}
	return nil
}

// checkConfig returns the compact JSON text of a job config, which must be an
// object with config.process[0] an object.
func checkConfig(raw json.RawMessage) (string, error) {
	cfg, err := jsonx.ParseObject(raw)
	if err != nil {
		return "", v1api.Errorf(v1api.CodeInvalidConfig, "config must be a JSON object")
	}
	inner := cfg.Object("config")
	if inner == nil {
		return "", v1api.Errorf(v1api.CodeInvalidConfig, "config.config must be an object")
	}
	process := inner.Array("process")
	if len(process) == 0 {
		return "", v1api.Errorf(v1api.CodeInvalidConfig, "config.config.process must be a non-empty array")
	}
	if _, ok := process[0].(*jsonx.Object); !ok {
		return "", v1api.Errorf(v1api.CodeInvalidConfig, "config.config.process[0] must be an object")
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return "", v1api.Errorf(v1api.CodeInvalidConfig, err.Error())
	}
	return buf.String(), nil
}

// policyPatch merges a client policy object into p; present keys set, null resets.
func policyPatch(ctx context.Context, raw json.RawMessage, p *state.Policy) error {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return v1api.Errorf(v1api.CodeInvalidRequest, "policy must be an object")
	}
	for k, v := range m {
		null := string(v) == "null"
		switch k {
		case "auto_resume":
			p.AutoResume = nil
			if !null {
				var b bool
				if err := json.Unmarshal(v, &b); err != nil {
					return v1api.Errorf(v1api.CodeInvalidRequest, "policy.auto_resume must be a boolean")
				}
				p.AutoResume = &b
			}
		case "max_auto_resumes":
			p.MaxAutoResumes = nil
			if !null {
				var n int
				if err := json.Unmarshal(v, &n); err != nil || n < 0 {
					return v1api.Errorf(v1api.CodeInvalidRequest, "policy.max_auto_resumes must be a non-negative integer")
				}
				p.MaxAutoResumes = &n
			}
		case "auto_upload_remote", "sync_remote":
			var target **string
			if k == "auto_upload_remote" {
				target = &p.AutoUploadRemote
			} else {
				target = &p.SyncRemote
			}
			*target = nil
			if !null {
				var s string
				if err := json.Unmarshal(v, &s); err != nil {
					return v1api.Errorf(v1api.CodeInvalidRequest, "policy."+k+" must be a string")
				}
				if s != "" {
					rem, err := remote.Get(ctx, s)
					if err != nil {
						return err
					}
					if rem == nil {
						return v1api.Errorf(v1api.CodeInvalidRequest, "policy."+k+": no remote named "+s)
					}
					*target = &rem.ID
				}
			}
		default:
			return v1api.Errorf(v1api.CodeInvalidRequest, "unknown policy field "+k)
		}
	}
	return nil
}

type jobBody struct {
	Name   *string         `json:"name"`
	GpuIDs *string         `json:"gpu_ids"`
	Config json.RawMessage `json:"config"`
	Type   *string         `json:"type"`
	Ref    *string         `json:"ref"`
	Policy json.RawMessage `json:"policy"`
}

func createJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var b jobBody
	if !v1api.Decode(w, r, &b) {
		return
	}
	if b.Name == nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "name is required")
		return
	}
	if err := validJobName(*b.Name); err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	gpu := "0"
	if b.GpuIDs != nil && strings.TrimSpace(*b.GpuIDs) != "" {
		gpu = strings.TrimSpace(*b.GpuIDs)
	}
	if runtime.GOOS == "darwin" {
		gpu = "mps" // like upstream's POST /api/jobs
	}
	if len(b.Config) == 0 {
		v1api.Fail(w, v1api.CodeInvalidConfig, "config is required")
		return
	}
	cfg, err := checkConfig(b.Config)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	var pol state.Policy
	if len(b.Policy) > 0 && string(b.Policy) != "null" {
		if err := policyPatch(ctx, b.Policy, &pol); err != nil {
			v1api.Respond(w, r, err)
			return
		}
	}
	if b.Type != nil && strings.TrimSpace(*b.Type) == "" {
		b.Type = nil
	}
	max, err := store.MaxQueuePosition(ctx)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	j, err := store.CreateJob(ctx, store.NewJob{Name: *b.Name, GpuIDs: gpu, JobConfig: cfg,
		QueuePosition: max + 1000, JobType: b.Type, JobRef: b.Ref})
	if errors.Is(err, store.ErrUniqueViolation) {
		v1api.Fail(w, v1api.CodeConflict, "a job named "+*b.Name+" already exists")
		return
	}
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if err := state.SetPolicy(ctx, j.ID, pol); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	respondJob(w, r, http.StatusCreated, j.ID)
}

func getJob(w http.ResponseWriter, r *http.Request) {
	if j, ok := loadJob(w, r); ok {
		respondJob(w, r, http.StatusOK, j.ID)
	}
}

func patchJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	var b jobBody
	if !v1api.Decode(w, r, &b) {
		return
	}
	f := store.Fields{}
	changesRun := b.Name != nil || b.GpuIDs != nil || len(b.Config) > 0 || b.Type != nil || b.Ref != nil
	if changesRun && busy(j.Status) {
		v1api.Fail(w, v1api.CodeConflict, "job is "+j.Status+"; stop it before changing it")
		return
	}
	if b.Name != nil && *b.Name != j.Name {
		if err := validJobName(*b.Name); err != nil {
			v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
			return
		}
		// renaming would orphan the folder ai-toolkit resumes from
		if folder, err := jobFolder(ctx, j.Name); err == nil {
			if _, err := os.Stat(folder); err == nil {
				v1api.Fail(w, v1api.CodeConflict, "job has files under its name; it cannot be renamed")
				return
			}
		}
		f["name"] = *b.Name
	}
	if b.GpuIDs != nil {
		if strings.TrimSpace(*b.GpuIDs) == "" {
			v1api.Fail(w, v1api.CodeInvalidRequest, "gpu_ids may not be empty")
			return
		}
		f["gpu_ids"] = strings.TrimSpace(*b.GpuIDs)
	}
	if len(b.Config) > 0 {
		cfg, err := checkConfig(b.Config)
		if err != nil {
			v1api.Respond(w, r, err)
			return
		}
		f["job_config"] = cfg
	}
	if b.Type != nil {
		f["job_type"] = *b.Type
	}
	if b.Ref != nil {
		f["job_ref"] = b.Ref
	}
	if len(b.Policy) > 0 && string(b.Policy) != "null" {
		pol, err := state.GetPolicy(ctx, j.ID)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if err := policyPatch(ctx, b.Policy, &pol); err != nil {
			v1api.Respond(w, r, err)
			return
		}
		if err := state.SetPolicy(ctx, j.ID, pol); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	if len(f) > 0 {
		if _, err := store.UpdateJob(ctx, j.ID, f); errors.Is(err, store.ErrUniqueViolation) {
			v1api.Fail(w, v1api.CodeConflict, "a job with that name already exists")
			return
		} else if err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	respondJob(w, r, http.StatusOK, j.ID)
}

func deleteJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	if busy(j.Status) {
		v1api.Fail(w, v1api.CodeConflict, "job is "+j.Status+"; stop it before deleting it")
		return
	}
	if r.URL.Query().Get("files") != "false" {
		folder, err := jobFolder(ctx, j.Name)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if err := os.RemoveAll(folder); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	if err := store.DeleteJob(ctx, j.ID); err != nil && !errors.Is(err, store.ErrNotFound) {
		v1api.Internal(w, r, err)
		return
	}
	if err := state.DeleteJobState(ctx, j.ID); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.NoContent(w)
}

// startJob queues the job and makes sure its GPU queue runs. A job folder
// with checkpoints resumes from the newest one (ai-toolkit does that itself).
func startJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	if j.Status == "running" || j.Status == "stopping" {
		v1api.Fail(w, v1api.CodeConflict, "job is already "+j.Status)
		return
	}
	if j.Status != "queued" {
		if err := supervisor.Requeue(ctx, j, "Job queued"); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	respondJob(w, r, http.StatusOK, j.ID)
}

// How long a stopping job gets for its graceful shutdown before it is assumed
// hung and force-killed (Windows; see handlers/jobs).
const (
	gracefulStopTimeout = 60 * time.Second
	backstopPoll        = 2 * time.Second
)

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

// stopJob asks the trainer to stop gracefully: it saves and writes "stopped"
// itself (the supervisor finishes the stop if the process is already gone).
// A queued job just leaves the queue.
func stopJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	switch j.Status {
	case "queued":
		if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"status": "stopped", "info": "Job stopped", "stop": false}); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	case "running", "stopping":
		if _, err := store.UpdateJob(ctx, j.ID, store.Fields{"stop": true, "status": "stopping", "info": "Stopping job..."}); err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if j.Pid != nil {
			if runtime.GOOS == "windows" {
				// the stop flag is the signal there; see proc.Interrupt
				scheduleForceKillBackstop(*j.Pid, j.ID)
			} else if err := proc.Interrupt(*j.Pid); err != nil {
				log.Printf("Error sending signal to job %s: %v", j.ID, err)
			}
		}
	}
	respondJob(w, r, http.StatusOK, j.ID)
}

// killJob force-stops the job (a hung model load ignoring the graceful stop).
func killJob(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	pid := 0
	if j.Pid != nil {
		pid = *j.Pid
	}
	if pid == 0 {
		if handles, err := engines.List(ctx, true); err == nil {
			for _, h := range handles {
				if h.JobID == j.ID && h.Endpoint != nil && h.Endpoint.Pid != 0 {
					pid = h.Endpoint.Pid
				}
			}
		}
	}
	if pid != 0 {
		if err := proc.TreeKill(ctx, pid); err != nil {
			log.Printf("kill %s pid %d: %v", j.ID, pid, err)
		}
	}
	if busy(j.Status) || pid != 0 {
		if _, err := store.UpdateJob(ctx, j.ID, store.Fields{
			"stop": true, "status": "stopped", "info": "Job force stopped", "pid": nil,
		}); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	respondJob(w, r, http.StatusOK, j.ID)
}

func flagJob(column string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		j, ok := loadJob(w, r)
		if !ok {
			return
		}
		if _, err := store.UpdateJob(r.Context(), j.ID, store.Fields{column: true}); err != nil {
			v1api.Internal(w, r, err)
			return
		}
		respondJob(w, r, http.StatusOK, j.ID)
	}
}

func jobAttempts(w http.ResponseWriter, r *http.Request) {
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	attempts, err := state.Attempts(r.Context(), j.ID)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Items(w, attempts)
}
