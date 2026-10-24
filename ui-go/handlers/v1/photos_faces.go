package v1

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/jobview"
	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/supervisor"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// faceQueue is the queue face checks run on: no GPU, so a check starts at
// once beside a training run.
const faceQueue = "cpu"

// faceJobMu makes "is a face check busy for this folder" and creating one a
// single step, so two clicks never queue two checks. It covers only that
// step (queueFaceJob), never the hashing before it.
var faceJobMu sync.Mutex

// checkFaces is POST /v1/datasets/{name}/photos/faces, body {text?: bool}:
// it queues a photo_faces job for the dataset on the cpu queue (job_ref =
// the folder) and answers 201 with the job and warnings. 409 while a face
// check of the folder is queued or running; 400 over
// photocheck.MaxFaceJob photos. Like Auto caption it never auto-resumes,
// syncs or uploads. Reviews (photos/check) read the result in once it
// completes.
func checkFaces(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	var b struct {
		Text bool `json:"text"`
	}
	// the body is optional: no body checks faces without text
	body, err := io.ReadAll(io.LimitReader(r.Body, v1api.MaxJSONBody))
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "could not read the request body: "+err.Error())
		return
	}
	if len(bytes.TrimSpace(body)) > 0 {
		r.Body = io.NopCloser(bytes.NewReader(body))
		if !v1api.Decode(w, r, &b) {
			return
		}
	}

	// hashing a large set that was never reviewed can take a while, so it
	// runs before the lock: other face checks never wait on it
	files, convertible, err := photocheck.FaceFiles(ctx, dir)
	if err != nil {
		photoFail(w, r, err)
		return
	}
	modelsPath, err := settings.ModelsPath(ctx)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	modelsDir := photocheck.FaceModelsDir(modelsPath)
	jobName := photocheck.FaceJobName(name, time.Now())
	raw, err := photocheck.BuildFaceJobConfig(dir, jobName, files, b.Text, modelsDir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	cfg, err := checkConfig(raw)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	j, ok := queueFaceJob(w, r, dir, jobName, cfg)
	if !ok {
		return
	}
	if j, err = store.GetJob(ctx, j.ID); err != nil || j == nil {
		v1api.Internal(w, r, errors.Join(err, errors.New("face check job vanished")))
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
	}{v, faceWarnings(convertible, photocheck.FaceDownload(modelsDir, b.Text))})
}

// faceUnreadableExts are the formats the trainer skips that the face
// check's Pillow cannot open either; bmp, gif and tiff it reads.
var faceUnreadableExts = map[string]bool{".heic": true, ".heif": true, ".avif": true, ".jxl": true}

// faceWarnings are the notes a new face check answers with: photos in
// formats it cannot read (of convertible, the photos in formats the
// trainer skips), and the one-time model download.
func faceWarnings(convertible []string, download int64) []string {
	out := []string{}
	var unreadable []string
	for _, f := range convertible {
		if faceUnreadableExts[strings.ToLower(path.Ext(f))] {
			unreadable = append(unreadable, f)
		}
	}
	switch n := len(unreadable); {
	case n == 1:
		out = append(out, fmt.Sprintf("1 photo is in a format the face check cannot read (%s). Convert it first (Images step, or datasets.fixPhotos convert).",
			unreadable[0]))
	case n > 1:
		names := unreadable
		if n > 3 {
			names = append(names[:3:3], "…")
		}
		out = append(out, fmt.Sprintf("%d photos are in formats the face check cannot read (%s). Convert them first (Images step, or datasets.fixPhotos convert).",
			n, strings.Join(names, ", ")))
	}
	if download > 0 {
		mb := max(1, (download+500_000)/1_000_000)
		out = append(out, fmt.Sprintf("The first face check downloads the face models (%d MB) once.", mb))
	}
	return out
}

// queueFaceJob creates the face check job named jobName for the folder dir
// and queues it, under faceJobMu so the busy check and the create are one
// step. It answers the request itself when it returns false: 409 while a
// face check of dir is queued or running.
func queueFaceJob(w http.ResponseWriter, r *http.Request, dir, jobName, cfg string) (*store.Job, bool) {
	ctx := r.Context()
	faceJobMu.Lock()
	defer faceJobMu.Unlock()
	busyJobs, err := store.FindJobs(ctx, store.JobQuery{
		Where: `"job_type" = ? AND "job_ref" = ? AND "status" IN ('queued', 'running', 'stopping')`,
		Args:  []any{photocheck.FaceJobType, dir},
	})
	if err != nil {
		v1api.Internal(w, r, err)
		return nil, false
	}
	if len(busyJobs) > 0 {
		v1api.Fail(w, v1api.CodeConflict, "already checking faces in this dataset: job "+busyJobs[0].Name)
		return nil, false
	}
	maxPos, err := store.MaxQueuePosition(ctx)
	if err != nil {
		v1api.Internal(w, r, err)
		return nil, false
	}
	typ := photocheck.FaceJobType
	j, err := store.CreateJob(ctx, store.NewJob{Name: jobName, GpuIDs: faceQueue, JobConfig: cfg,
		QueuePosition: maxPos + 1000, JobType: &typ, JobRef: &dir})
	if errors.Is(err, store.ErrUniqueViolation) {
		v1api.Fail(w, v1api.CodeConflict, "a job named "+jobName+" already exists")
		return nil, false
	}
	if err != nil {
		v1api.Internal(w, r, err)
		return nil, false
	}
	// a face check only reads the dataset and leaves faces.json for the
	// next review: nothing to sync, upload or resume elsewhere. Explicit ""
	// and false, because nil fields take the server's defaults.
	none, no := "", false
	if err := state.SetPolicy(ctx, j.ID, state.Policy{AutoResume: &no, AutoUploadRemote: &none, SyncRemote: &none}); err != nil {
		v1api.Internal(w, r, err)
		return nil, false
	}
	if err := supervisor.Requeue(ctx, j, "Job queued"); err != nil {
		v1api.Internal(w, r, err)
		return nil, false
	}
	return j, true
}
