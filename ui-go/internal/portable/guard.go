package portable

import (
	"context"
	"errors"
	"log"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

const guardCache = 15 * time.Second

var (
	guardMu      sync.Mutex
	guardBlocked = map[string]guardEntry{}
)

type guardEntry struct {
	info  string
	until time.Time
}

// StartGuard is installed as worker.StartGuard: a synced job may start only
// when no other live node holds its lease, and starting takes the lease.
// When the remote cannot be reached the job starts anyway (the error is
// recorded): a storage outage must not stall training.
func StartGuard(ctx context.Context, j *store.Job) (bool, string) {
	guardMu.Lock()
	if e, ok := guardBlocked[j.ID]; ok && time.Now().Before(e.until) {
		guardMu.Unlock()
		return false, e.info
	}
	delete(guardBlocked, j.ID)
	guardMu.Unlock()

	_, c, _, err := syncTarget(ctx, j)
	if errors.Is(err, ErrNotSynced) {
		return true, ""
	}
	if err != nil {
		recordGuardError(ctx, j, err)
		return true, ""
	}
	held, err := takeLease(ctx, c, j)
	if err != nil {
		recordGuardError(ctx, j, err)
		return true, ""
	}
	if held != nil {
		info := "Waiting: lease held by " + held.NodeLabel + " until " + held.ExpiresAt.Format(time.RFC3339)
		guardMu.Lock()
		guardBlocked[j.ID] = guardEntry{info: info, until: time.Now().Add(guardCache)}
		guardMu.Unlock()
		return false, info
	}
	return true, ""
}

func recordGuardError(ctx context.Context, j *store.Job, err error) {
	log.Printf("portable: start guard %s: %v", j.Name, err)
	st, _ := GetStatus(ctx, j.ID)
	msg := err.Error()
	st.LastError = &msg
	saveStatus(ctx, j.ID, st)
}

// takeLease writes this node's lease into the manifest (creating a minimal
// one when the job was never synced). Returns the other node's lease when it
// is still held.
func takeLease(ctx context.Context, c remote.Client, j *store.Job) (*Lease, error) {
	unlock := lockManifest(j.Name)
	defer unlock()
	m, err := readManifest(ctx, c, j.Name)
	if err != nil {
		return nil, err
	}
	me, err := state.NodeID(ctx)
	if err != nil {
		return nil, err
	}
	if m != nil && m.Lease.Held(me, time.Now()) {
		return m.Lease, nil
	}
	if m == nil {
		if m, err = baseManifest(ctx, j); err != nil {
			return nil, err
		}
	}
	if m.Lease, err = newLease(ctx); err != nil {
		return nil, err
	}
	return nil, writeManifest(ctx, c, m)
}

// baseManifest describes a job that has nothing uploaded yet.
func baseManifest(ctx context.Context, j *store.Job) (*Manifest, error) {
	dsRoot, err := settings.DatasetsRoot(ctx)
	if err != nil {
		return nil, err
	}
	trRoot, err := settings.TrainingFolder(ctx)
	if err != nil {
		return nil, err
	}
	cfg, names, err := ToPortable(j.JobConfig, dsRoot, trRoot)
	if err != nil {
		return nil, err
	}
	pol, err := state.GetPolicy(ctx, j.ID)
	if err != nil {
		return nil, err
	}
	attempts, err := state.Attempts(ctx, j.ID)
	if err != nil {
		return nil, err
	}
	if names == nil {
		names = []string{}
	}
	return &Manifest{Name: j.Name, Type: j.JobType, Ref: portableRef(j.JobRef, dsRoot, trRoot), GpuIDs: j.GpuIDs,
		Config: cfg, Policy: pol, Datasets: names, Step: j.Step, TotalSteps: j.TotalSteps,
		State: jobstate.FromStatus(j.Status, j.Step, j.Info), Files: []File{}, Samples: []File{}, Attempts: attempts}, nil
}
