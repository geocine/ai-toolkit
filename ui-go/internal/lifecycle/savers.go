package lifecycle

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/portable"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/uploads"
)

// Proof says where a job's results are off this server, verified.
type Proof struct {
	Via    string // e.g. "sync:s3", "upload:hf", "bundle:s3"
	Detail string
}

// Saver makes sure a job's results are on a remote before the server goes
// away (pushing them when they are not there yet) and proves it by checking
// the remote object exists with the local size. covered is false when the
// saver is not configured for the job (no sync remote, no upload remote...).
type Saver interface {
	Name() string
	Save(ctx context.Context, j *store.Job, newest artifacts.Checkpoint) (p Proof, covered bool, err error)
}

var (
	saversMu sync.Mutex
	savers   = []Saver{syncSaver{}, uploadSaver{}}
)

// RegisterSaver adds a saver, tried after the built-in ones.
func RegisterSaver(s Saver) {
	saversMu.Lock()
	savers = append(savers, s)
	saversMu.Unlock()
}

// RegisterSaverFirst adds a saver tried before the built-in ones (the run
// bundle saver in handlers/v1: a whole-run bundle is preferred when one is
// configured).
func RegisterSaverFirst(s Saver) {
	saversMu.Lock()
	savers = append([]Saver{s}, savers...)
	saversMu.Unlock()
}

func allSavers() []Saver {
	saversMu.Lock()
	defer saversMu.Unlock()
	return append([]Saver(nil), savers...)
}

// pollEvery is how often savers re-check a pending push; tests shrink it.
var pollEvery = 2 * time.Second

// verifyObject checks key exists on c with size bytes.
func verifyObject(ctx context.Context, c remote.Client, key string, size int64) error {
	o, err := c.Stat(ctx, key)
	if err != nil {
		if errors.Is(err, remote.ErrNotFound) {
			return fmt.Errorf("%s is not on the remote", key)
		}
		return err
	}
	if o.Size != size {
		return fmt.Errorf("%s is %d bytes on the remote, %d here", key, o.Size, size)
	}
	return nil
}

// syncSaver covers portable jobs (policy sync_remote): the manifest on the
// remote must list the newest checkpoint with its local size, and that
// object must exist with that size. When it does not, one final sync is run
// and checked again.
type syncSaver struct{}

func (syncSaver) Name() string { return "sync" }

func (syncSaver) Save(ctx context.Context, j *store.Job, newest artifacts.Checkpoint) (Proof, bool, error) {
	pol, err := state.GetPolicy(ctx, j.ID)
	if err != nil {
		return Proof{}, false, err
	}
	if pol.SyncRemote == nil || *pol.SyncRemote == "" {
		return Proof{}, false, nil
	}
	r, c, err := remote.Open(ctx, *pol.SyncRemote)
	if err != nil {
		return Proof{}, true, err
	}
	check := func() error {
		// jobs/<name>/manifest.json, see internal/portable (manifest.go)
		b, err := c.GetBytes(ctx, "jobs/"+j.Name+"/manifest.json")
		if err != nil {
			return fmt.Errorf("manifest: %w", err)
		}
		var m portable.Manifest
		if err := json.Unmarshal(b, &m); err != nil {
			return fmt.Errorf("manifest: %w", err)
		}
		for _, f := range m.Files {
			if f.Name == newest.Name {
				if f.Size != newest.Size {
					return fmt.Errorf("manifest has %s at %d bytes, %d here", f.Name, f.Size, newest.Size)
				}
				return verifyObject(ctx, c, f.Key, newest.Size)
			}
		}
		return fmt.Errorf("manifest does not list %s yet", newest.Name)
	}
	if err := check(); err != nil {
		if _, serr := portable.SyncJob(ctx, j.ID, portable.ReasonFinal); serr != nil {
			return Proof{}, true, fmt.Errorf("%v; final sync: %w", err, serr)
		}
		if err := check(); err != nil {
			return Proof{}, true, err
		}
	}
	return Proof{Via: "sync:" + r.Name, Detail: newest.Name + " synced to " + r.Name}, true, nil
}

// uploadSaver covers jobs with an auto_upload_remote: the newest checkpoint
// must be uploaded there (it is queued when no upload exists, and waited for
// while one is queued or running).
type uploadSaver struct{}

func (uploadSaver) Name() string { return "upload" }

func (uploadSaver) Save(ctx context.Context, j *store.Job, newest artifacts.Checkpoint) (Proof, bool, error) {
	_, _, ref, err := state.EffectivePolicy(ctx, j.ID)
	if err != nil {
		return Proof{}, false, err
	}
	if ref == "" {
		return Proof{}, false, nil
	}
	r, c, err := remote.Open(ctx, ref)
	if err != nil {
		return Proof{}, true, err
	}
	rel := j.Name + "/" + newest.Name
	// a job file upload lands at jobs/<job>/<file> (uploads.remoteKey)
	key := "jobs/" + rel
	if verifyObject(ctx, c, key, newest.Size) == nil {
		return Proof{Via: "upload:" + r.Name, Detail: newest.Name + " uploaded to " + r.Name}, true, nil
	}
	var pending *uploads.Upload
	if list, err := uploads.List(ctx, j.ID, ""); err == nil {
		for _, u := range list {
			if u.RemoteID == r.ID && filepath.ToSlash(u.Path) == rel && (u.State == uploads.Queued || u.State == uploads.Running) {
				pending = u
				break
			}
		}
	}
	if pending == nil {
		id := j.ID
		if pending, err = uploads.Enqueue(ctx, r.ID, &id, "training", rel); err != nil {
			return Proof{}, true, fmt.Errorf("queue upload of %s: %w", newest.Name, err)
		}
	}
	for {
		u, err := uploads.Get(ctx, pending.ID)
		if err != nil {
			return Proof{}, true, err
		}
		if u == nil {
			return Proof{}, true, errors.New("upload vanished")
		}
		if u.State == uploads.Failed {
			msg := "failed"
			if u.Error != nil {
				msg = *u.Error
			}
			return Proof{}, true, fmt.Errorf("upload of %s: %s", newest.Name, msg)
		}
		if u.State == uploads.Done {
			break
		}
		select {
		case <-ctx.Done():
			return Proof{}, true, fmt.Errorf("upload of %s still %s: %w", newest.Name, u.State, ctx.Err())
		case <-time.After(pollEvery):
		}
	}
	if err := verifyObject(ctx, c, key, newest.Size); err != nil {
		return Proof{}, true, err
	}
	return Proof{Via: "upload:" + r.Name, Detail: newest.Name + " uploaded to " + r.Name}, true, nil
}

// JobResult is the safety verdict for one finished job.
type JobResult struct {
	JobID   string `json:"job_id"`
	JobName string `json:"job_name"`
	Safe    bool   `json:"safe"`
	Via     string `json:"via"`
	Detail  string `json:"detail"`
}

// newestCheckpoint is what a job's result is: the final (step-less) file
// when present, else the highest step.
func newestCheckpoint(folder string) (*artifacts.Checkpoint, error) {
	cps, err := artifacts.Checkpoints(folder)
	if err != nil {
		return nil, err
	}
	var best *artifacts.Checkpoint
	for i := range cps {
		c := &cps[i]
		switch {
		case c.Step == nil:
			return c, nil
		case best == nil || *c.Step > *best.Step:
			best = c
		}
	}
	return best, nil
}

// verifyJob runs the savers for one job until one proves its newest
// checkpoint is off the server.
func verifyJob(ctx context.Context, j *store.Job) JobResult {
	res := JobResult{JobID: j.ID, JobName: j.Name}
	root, err := settings.TrainingFolder(ctx)
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	newest, err := newestCheckpoint(filepath.Join(root, j.Name))
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	if newest == nil {
		res.Safe, res.Via, res.Detail = true, "none", "no checkpoints to keep"
		return res
	}
	var problems []string
	for _, s := range allSavers() {
		p, covered, err := s.Save(ctx, j, *newest)
		if !covered {
			continue
		}
		if err == nil {
			res.Safe, res.Via, res.Detail = true, p.Via, p.Detail
			return res
		}
		problems = append(problems, s.Name()+": "+err.Error())
	}
	if len(problems) == 0 {
		res.Detail = newest.Name + " is only on this server (no sync or upload remote for this job)"
	} else {
		res.Detail = strings.Join(problems, "; ")
	}
	return res
}
