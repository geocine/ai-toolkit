// Package jobfiles ports the job artifact routes under
// ui/src/app/api/jobs/[jobID]/: files, log, loss, notes, plugin, samples.
// The job lifecycle routes live in the jobs package.
package jobfiles

import (
	"net/http"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /api/jobs/{jobID}/files", files)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/log", logTail)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/loss", lossGet)
	httpx.Handle(mux, "DELETE /api/jobs/{jobID}/loss", lossDelete)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/notes", notesGet)
	httpx.Handle(mux, "POST /api/jobs/{jobID}/notes", notesPost)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/plugin", plugin)
	httpx.Handle(mux, "GET /api/jobs/{jobID}/samples", samples)
}

// jobFolder loads the path's job and returns <training folder>/<job name>,
// answering 404 when the job does not exist.
func jobFolder(w http.ResponseWriter, r *http.Request) (string, bool) {
	job, err := store.GetJob(r.Context(), r.PathValue("jobID"))
	if err != nil {
		httpx.Internal(w, r, err)
		return "", false
	}
	if job == nil {
		httpx.Error(w, http.StatusNotFound, "Job not found")
		return "", false
	}
	root, err := settings.TrainingFolder(r.Context())
	if err != nil {
		httpx.Internal(w, r, err)
		return "", false
	}
	return filepath.Join(root, job.Name), true
}
