// Package v1remotes serves /v1/remotes and /v1/uploads (docs/headless.md,
// "Remote uploads").
package v1remotes

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/uploads"
	"github.com/geocine/aitoolkit/internal/v1api"
)

func Register(mux *http.ServeMux) {
	uploads.Init()
	httpx.Handle(mux, "GET /v1/remotes", listRemotes)
	httpx.Handle(mux, "POST /v1/remotes", createRemote)
	httpx.Handle(mux, "DELETE /v1/remotes/{id}", deleteRemote)
	httpx.Handle(mux, "POST /v1/remotes/{id}/test", testRemote)
	httpx.Handle(mux, "POST /v1/uploads", createUpload)
	httpx.Handle(mux, "GET /v1/uploads", listUploads)
	httpx.Handle(mux, "GET /v1/uploads/{id}", getUpload)
	httpx.Handle(mux, "POST /v1/uploads/{id}/retry", retryUpload)
}

func listRemotes(w http.ResponseWriter, r *http.Request) {
	items, err := remote.List(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func validName(s string) bool {
	if s == "" || len(s) > 64 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

func createRemote(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name        string          `json:"name"`
		Kind        string          `json:"kind"`
		Config      json.RawMessage `json:"config"`
		Credentials json.RawMessage `json:"credentials"`
	}
	if !v1api.Decode(w, r, &body) {
		return
	}
	if !validName(body.Name) {
		v1api.Fail(w, v1api.CodeInvalidRequest, "name must be 1-64 letters, digits, '-', '_' or '.'")
		return
	}
	if !slices.Contains(remote.Kinds(), body.Kind) {
		v1api.Fail(w, v1api.CodeInvalidRequest, "kind must be one of "+strings.Join(remote.Kinds(), ", "))
		return
	}
	if len(body.Config) > 0 && body.Config[0] != '{' || len(body.Credentials) > 0 && body.Credentials[0] != '{' && string(body.Credentials) != "null" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "config and credentials must be objects")
		return
	}
	if string(body.Credentials) == "null" {
		body.Credentials = nil
	}
	if existing, err := remote.Get(r.Context(), body.Name); err != nil {
		v1api.Internal(w, r, err)
		return
	} else if existing != nil {
		v1api.Fail(w, v1api.CodeConflict, "a remote named "+body.Name+" already exists")
		return
	}
	rem, err := remote.Create(r.Context(), body.Name, body.Kind, body.Config, body.Credentials)
	if err != nil {
		if strings.Contains(err.Error(), "UNIQUE constraint failed") {
			v1api.Fail(w, v1api.CodeConflict, "a remote named "+body.Name+" already exists")
			return
		}
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	v1api.Created(w, rem)
}

func deleteRemote(w http.ResponseWriter, r *http.Request) {
	rem, err := remote.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if rem == nil {
		v1api.Fail(w, v1api.CodeNotFound, "remote not found")
		return
	}
	// a remote a running job syncs or uploads to, or with uploads in flight,
	// stays until they are done
	var users []string
	jobs, err := store.FindJobs(r.Context(), store.JobQuery{Where: `"status" IN ('queued', 'running', 'stopping')`})
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	for _, j := range jobs {
		p, err := state.GetPolicy(r.Context(), j.ID)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if (p.SyncRemote != nil && *p.SyncRemote == rem.ID) || (p.AutoUploadRemote != nil && *p.AutoUploadRemote == rem.ID) {
			users = append(users, j.Name)
		}
	}
	for _, st := range []string{uploads.Queued, uploads.Running} {
		list, err := uploads.List(r.Context(), "", st)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		for _, u := range list {
			if u.RemoteID == rem.ID {
				users = append(users, "an upload of "+filepath.Base(u.Path))
				break
			}
		}
	}
	if len(users) > 0 {
		v1api.Fail(w, v1api.CodeConflict, "remote "+rem.Name+" is in use by "+strings.Join(users, ", ")+"; stop them first")
		return
	}
	if _, err := remote.Delete(r.Context(), rem.ID); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.NoContent(w)
}

const testTimeout = 2 * time.Minute

func testRemote(w http.ResponseWriter, r *http.Request) {
	rem, err := remote.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if rem == nil {
		v1api.Fail(w, v1api.CodeNotFound, "remote not found")
		return
	}
	client, err := remote.NewClient(rem)
	if err != nil {
		v1api.OK(w, map[string]any{"ok": false, "detail": err.Error()})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), testTimeout)
	defer cancel()
	if err := client.Check(ctx); err != nil {
		v1api.OK(w, map[string]any{"ok": false, "detail": err.Error()})
		return
	}
	v1api.OK(w, map[string]any{"ok": true, "detail": "reachable and writable"})
}

func createUpload(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Remote     string `json:"remote"`
		Root       string `json:"root"`
		Path       string `json:"path"`
		JobID      string `json:"job_id"`
		Checkpoint string `json:"checkpoint"`
	}
	if !v1api.Decode(w, r, &body) {
		return
	}
	ctx := r.Context()
	if body.Remote == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "remote is required")
		return
	}
	var jobID *string
	root, rel := body.Root, body.Path
	switch {
	case body.JobID != "" || body.Checkpoint != "":
		if body.JobID == "" || body.Checkpoint == "" {
			v1api.Fail(w, v1api.CodeInvalidRequest, "job_id and checkpoint go together")
			return
		}
		if strings.ContainsAny(body.Checkpoint, `/\`) || body.Checkpoint == "." || body.Checkpoint == ".." {
			v1api.Fail(w, v1api.CodeInvalidRequest, "checkpoint must be a file name in the job folder")
			return
		}
		job, err := store.GetJob(ctx, body.JobID)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if job == nil {
			v1api.Fail(w, v1api.CodeNotFound, "job not found")
			return
		}
		trainRoot, err := settings.TrainingFolder(ctx)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if st, err := os.Stat(filepath.Join(trainRoot, job.Name, body.Checkpoint)); err != nil || !st.Mode().IsRegular() {
			v1api.Fail(w, v1api.CodeNotFound, "checkpoint "+body.Checkpoint+" not found in job "+job.Name)
			return
		}
		id := job.ID
		jobID, root, rel = &id, "training", job.Name+"/"+body.Checkpoint
	case body.Root == "" || body.Path == "":
		v1api.Fail(w, v1api.CodeInvalidRequest, "give either root and path, or job_id and checkpoint")
		return
	case !slices.Contains(uploads.Roots, body.Root):
		v1api.Fail(w, v1api.CodeInvalidRequest, "root must be one of "+strings.Join(uploads.Roots, ", "))
		return
	}
	u, err := uploads.Enqueue(ctx, body.Remote, jobID, root, rel)
	switch {
	case errors.Is(err, remote.ErrNotFound):
		v1api.Fail(w, v1api.CodeNotFound, "remote "+body.Remote+" not found")
	case errors.Is(err, uploads.ErrInvalid):
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
	case err != nil:
		v1api.Internal(w, r, err)
	default:
		v1api.JSON(w, http.StatusAccepted, u)
	}
}

func listUploads(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	st := q.Get("state")
	if st != "" && !slices.Contains([]string{uploads.Queued, uploads.Running, uploads.Done, uploads.Failed}, st) {
		v1api.Fail(w, v1api.CodeInvalidRequest, "state must be queued, running, done or failed")
		return
	}
	items, err := uploads.List(r.Context(), q.Get("job_id"), st)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func getUpload(w http.ResponseWriter, r *http.Request) {
	u, err := uploads.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if u == nil {
		v1api.Fail(w, v1api.CodeNotFound, "upload not found")
		return
	}
	v1api.OK(w, u)
}

func retryUpload(w http.ResponseWriter, r *http.Request) {
	u, err := uploads.Retry(r.Context(), r.PathValue("id"))
	switch {
	case errors.Is(err, uploads.ErrBusy):
		v1api.Fail(w, v1api.CodeConflict, "upload is still queued or running")
	case err != nil:
		v1api.Internal(w, r, err)
	case u == nil:
		v1api.Fail(w, v1api.CodeNotFound, "upload not found")
	default:
		v1api.OK(w, u)
	}
}
