// Package v1portable serves portable jobs (docs/headless.md, "Portable
// jobs"): listing and importing jobs from a remote, and per-job sync.
package v1portable

import (
	"errors"
	"github.com/geocine/aitoolkit/internal/jobview"
	"net/http"
	"time"

	"github.com/geocine/aitoolkit/internal/bg"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/portable"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
	"github.com/geocine/aitoolkit/internal/worker"
)

// Register serves the routes and installs the sync loop and start guard.
func Register(mux *http.ServeMux) {
	worker.StartGuard = portable.StartGuard
	bg.Go("portable", portable.Run)

	httpx.Handle(mux, "GET /v1/portable/jobs", listJobs)
	httpx.Handle(mux, "POST /v1/portable/import", importJob)
	httpx.Handle(mux, "GET /v1/jobs/{id}/sync", getSync)
	httpx.Handle(mux, "POST /v1/jobs/{id}/sync", postSync)
}

func listJobs(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("remote")
	if name == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "remote is required")
		return
	}
	items, err := portable.ListRemoteJobs(r.Context(), name)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func importJob(w http.ResponseWriter, r *http.Request) {
	var req portable.ImportRequest
	if !v1api.Decode(w, r, &req) {
		return
	}
	job, err := portable.Import(r.Context(), req)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	view, err := jobview.View(r.Context(), job)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Created(w, view)
}

func jobOr404(w http.ResponseWriter, r *http.Request) *store.Job {
	j, err := store.GetJob(r.Context(), r.PathValue("id"))
	if err != nil {
		v1api.Internal(w, r, err)
		return nil
	}
	if j == nil {
		v1api.Fail(w, v1api.CodeNotFound, "job not found")
		return nil
	}
	return j
}

func syncError(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case errors.Is(err, portable.ErrNotSynced):
		v1api.Fail(w, v1api.CodeConflict, "job has no sync_remote in its policy")
	case errors.Is(err, portable.ErrLeaseLost):
		v1api.Fail(w, v1api.CodeConflict, err.Error())
	case errors.Is(err, remote.ErrNotFound):
		v1api.Fail(w, v1api.CodeNotFound, err.Error())
	default:
		v1api.Fail(w, v1api.CodeRemoteFailed, err.Error())
	}
}

func getSync(w http.ResponseWriter, r *http.Request) {
	j := jobOr404(w, r)
	if j == nil {
		return
	}
	st, lease, err := portable.CurrentStatus(r.Context(), j)
	if err != nil && !errors.Is(err, portable.ErrNotSynced) {
		syncError(w, r, err)
		return
	}
	st.Lease = lease
	v1api.OK(w, statusBody(st))
}

func postSync(w http.ResponseWriter, r *http.Request) {
	j := jobOr404(w, r)
	if j == nil {
		return
	}
	st, err := portable.SyncJob(r.Context(), j.ID, portable.ReasonManual)
	if err != nil {
		syncError(w, r, err)
		return
	}
	v1api.OK(w, statusBody(st))
}

func statusBody(st portable.Status) map[string]any {
	var at any
	if st.LastSyncAt != nil {
		at = v1api.Time(*st.LastSyncAt)
	}
	var remoteName any
	if st.Remote != "" {
		remoteName = st.Remote
	}
	return map[string]any{"remote": remoteName, "last_sync_at": at, "last_error": st.LastError, "step": st.Step,
		"lease": leaseBody(st.Lease)}
}

func leaseBody(l *portable.Lease) any {
	if l == nil {
		return nil
	}
	return map[string]any{"node_id": l.NodeID, "node_label": l.NodeLabel,
		"heartbeat_at": v1api.Time(l.HeartbeatAt), "expires_at": v1api.Time(l.ExpiresAt),
		"expired": time.Now().After(l.ExpiresAt)}
}
