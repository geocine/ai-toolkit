package portable

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// Run reacts to job events with syncs and renews leases until ctx ends.
func Run(ctx context.Context) {
	ch, cancel := events.Subscribe(events.Filter{Types: map[string]bool{
		events.JobAttempt: true, events.JobCheckpoint: true, events.JobState: true,
	}})
	defer cancel()

	// jobs left active by a previous run of this server: renew and sync
	for _, j := range syncedJobs(ctx, true) {
		trigger(ctx, j.ID, ReasonCheckpoint)
	}

	t := time.NewTicker(LeaseRenew)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case e := <-ch:
			if e.JobID == nil {
				continue
			}
			if reason := reasonFor(e); reason != "" && isSynced(ctx, *e.JobID) {
				trigger(ctx, *e.JobID, reason)
			}
		case <-t.C:
			for _, j := range syncedJobs(ctx, true) {
				if err := RenewLease(ctx, j); err != nil && !errors.Is(err, ErrLeaseLost) {
					log.Printf("portable: renew lease %s: %v", j.Name, err)
				}
				// retry a sync that failed
				if st, _ := GetStatus(ctx, j.ID); st.LastError != nil {
					trigger(ctx, j.ID, ReasonCheckpoint)
				}
			}
		}
	}
}

// reasonFor maps an event to the sync it calls for ("" for none).
func reasonFor(e events.Event) string {
	switch e.Type {
	case events.JobCheckpoint:
		return ReasonCheckpoint
	case events.JobAttempt:
		var a state.Attempt
		if json.Unmarshal(e.Data, &a) == nil && a.EndedAt == nil {
			return ReasonStart
		}
	case events.JobState:
		var d struct {
			To string `json:"to"`
		}
		if json.Unmarshal(e.Data, &d) != nil {
			return ""
		}
		switch d.To {
		case jobstate.Completed, jobstate.Stopped, jobstate.Failed:
			return ReasonFinal
		case jobstate.Interrupted:
			return ReasonInterrupted
		}
	}
	return ""
}

func isSynced(ctx context.Context, jobID string) bool {
	p, err := state.GetPolicy(ctx, jobID)
	return err == nil && p.SyncRemote != nil && *p.SyncRemote != ""
}

// syncedJobs lists jobs with a sync_remote (only queued/running/stopping
// ones when activeOnly).
func syncedJobs(ctx context.Context, activeOnly bool) []*store.Job {
	if state.DB == nil || store.DB == nil {
		return nil
	}
	rows, err := state.DB.QueryContext(ctx, `SELECT job_id FROM job_policies WHERE sync_remote IS NOT NULL AND sync_remote != ''`)
	if err != nil {
		return nil
	}
	var ids []string
	for rows.Next() {
		var id string
		if rows.Scan(&id) == nil {
			ids = append(ids, id)
		}
	}
	rows.Close()
	var out []*store.Job
	for _, id := range ids {
		j, err := store.GetJob(ctx, id)
		if err != nil || j == nil {
			continue
		}
		if activeOnly && j.Status != "queued" && j.Status != "running" && j.Status != "stopping" {
			continue
		}
		out = append(out, j)
	}
	return out
}

// one sync at a time per job; a request arriving mid-sync runs after it
var (
	runMu   sync.Mutex
	running = map[string]bool{}
	pending = map[string]string{}
)

func trigger(ctx context.Context, jobID, reason string) {
	runMu.Lock()
	if running[jobID] {
		if p := pending[jobID]; !(p == ReasonStart && reason == ReasonCheckpoint) {
			pending[jobID] = reason
		}
		runMu.Unlock()
		return
	}
	running[jobID] = true
	runMu.Unlock()
	go func() {
		for {
			if _, err := SyncJob(ctx, jobID, reason); err != nil && !errors.Is(err, ErrNotSynced) {
				log.Printf("portable: sync %s (%s): %v", jobID, reason, err)
			}
			runMu.Lock()
			next, ok := pending[jobID]
			delete(pending, jobID)
			if !ok || ctx.Err() != nil {
				delete(running, jobID)
				runMu.Unlock()
				return
			}
			runMu.Unlock()
			reason = next
		}
	}()
}
