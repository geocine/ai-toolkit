// Package engines finds running inference-engine jobs and their endpoints.
// Mirrors ui/src/server/inferenceEngine.ts.
package engines

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/proc"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

// Endpoint is what an engine job writes to <job folder>/engine.json once its
// server is up.
type Endpoint struct {
	Host         string  `json:"host"`
	Port         int     `json:"port"`
	Token        string  `json:"token"`
	Pid          int     `json:"pid"`
	JobID        *string `json:"job_id,omitempty"`
	Device       string  `json:"device,omitempty"`
	JobFolder    string  `json:"job_folder"`
	OutputFolder string  `json:"output_folder"`
	StartedAt    float64 `json:"started_at"`

	raw json.RawMessage
}

// MarshalJSON emits engine.json as written, extra fields included, like upstream.
func (e Endpoint) MarshalJSON() ([]byte, error) {
	if len(e.raw) > 0 {
		return e.raw, nil
	}
	type plain Endpoint
	return json.Marshal(plain(e))
}

// Handle is one engine job, with its endpoint once published.
type Handle struct {
	JobID    string    `json:"jobId"`
	JobName  string    `json:"jobName"`
	GpuIDs   string    `json:"gpuIds"`
	Status   string    `json:"status"`
	Endpoint *Endpoint `json:"endpoint"`
}

func readEndpoint(jobFolder string) *Endpoint {
	raw, err := os.ReadFile(filepath.Join(jobFolder, "engine.json"))
	if err != nil {
		return nil
	}
	var ep Endpoint
	if err := json.Unmarshal(raw, &ep); err != nil {
		return nil
	}
	if ep.Host == "" || ep.Port == 0 {
		return nil
	}
	var compact bytes.Buffer
	if json.Compact(&compact, raw) == nil {
		ep.raw = compact.Bytes()
	}
	// a stale file from a crashed engine must not be proxied to
	if ep.Pid != 0 && !proc.Alive(ep.Pid) {
		return nil
	}
	return &ep
}

const cacheTTL = 2 * time.Second

var (
	mu       sync.Mutex
	cachedAt time.Time
	cached   []Handle
)

// List returns every inference-engine job that is running or starting, plus
// stopped rows whose process is still alive (reported as stopping).
func List(ctx context.Context, forceFresh bool) ([]Handle, error) {
	mu.Lock()
	if !forceFresh && cached != nil && time.Since(cachedAt) < cacheTTL {
		h := cached
		mu.Unlock()
		return h, nil
	}
	mu.Unlock()

	jobs, err := store.FindJobs(ctx, store.JobQuery{
		Where: `"job_type" = 'inference'`, OrderBy: `"updated_at" DESC`,
	})
	if err != nil {
		return nil, err
	}
	trainingFolder, err := settings.TrainingFolder(ctx)
	if err != nil {
		return nil, err
	}
	handles := []Handle{}
	for _, j := range jobs {
		active := j.Status == "running" || j.Status == "queued" || j.Status == "stopping"
		ep := readEndpoint(filepath.Join(trainingFolder, j.Name))
		if !active && ep == nil {
			continue
		}
		status := j.Status
		if !active && ep != nil {
			// process still alive after the row was stopped
			status = "stopping"
		}
		handles = append(handles, Handle{JobID: j.ID, JobName: j.Name, GpuIDs: j.GpuIDs, Status: status, Endpoint: ep})
	}
	mu.Lock()
	cached, cachedAt = handles, time.Now()
	mu.Unlock()
	return handles, nil
}

// Resolve returns the engine to talk to: the requested job, else the first
// running one with an endpoint, else the first one. nil when there is none.
func Resolve(ctx context.Context, jobID string) (*Handle, error) {
	handles, err := List(ctx, false)
	if err != nil {
		return nil, err
	}
	if jobID != "" {
		for i := range handles {
			if handles[i].JobID == jobID {
				return &handles[i], nil
			}
		}
		return nil, nil
	}
	for i := range handles {
		if handles[i].Endpoint != nil {
			return &handles[i], nil
		}
	}
	if len(handles) > 0 {
		return &handles[0], nil
	}
	return nil, nil
}
