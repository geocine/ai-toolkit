package v1

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/jobview"
	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Photo checks (docs/headless.md, Datasets): the judgement of metrics
// measured elsewhere, the stored photos' review, the subject settings and
// forgetting the analysis. The fixes are in photos_fix.go. The work is in
// internal/photocheck; these handlers parse, guard and answer.

// GET photos/check waits this long for photos it has not analysed yet,
// unless wait_ms says otherwise (capped at maxPhotoWait).
const (
	defaultPhotoWait = 3 * time.Second
	maxPhotoWait     = 20 * time.Second
)

// photoFail answers a photocheck error: a *photocheck.UserError is a 400
// (a 409 when it is a conflict), a dataset that vanished mid-request is a
// 404, anything else a 500.
func photoFail(w http.ResponseWriter, r *http.Request, err error) {
	var ue *photocheck.UserError
	switch {
	case errors.As(err, &ue) && ue.Conflict:
		v1api.Fail(w, v1api.CodeConflict, ue.Message)
	case errors.As(err, &ue):
		v1api.Fail(w, v1api.CodeInvalidRequest, ue.Message)
	case errors.Is(err, fs.ErrNotExist):
		v1api.Fail(w, v1api.CodeNotFound, "dataset not found")
	default:
		v1api.Internal(w, r, err)
	}
}

// withFaceJob fills in review.faces.job, the client-facing view of the
// dataset's newest face check, when the review names one.
func withFaceJob(ctx context.Context, rv *photocheck.Review) error {
	if rv == nil || rv.Faces.JobID == "" {
		return nil
	}
	j, err := store.GetJob(ctx, rv.Faces.JobID)
	if err != nil || j == nil {
		return err
	}
	v, err := jobview.View(ctx, j)
	if err != nil {
		return err
	}
	rv.Faces.Job = v
	return nil
}

// answerReview writes a review with its face job filled in.
func answerReview(w http.ResponseWriter, r *http.Request, rv *photocheck.Review) {
	if err := withFaceJob(r.Context(), rv); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, rv)
}

// assessPhotos is POST /v1/photos/assess: photo metrics measured elsewhere
// (the studio measures in the browser before upload) judged with the same
// rules as photos/check. With dataset, that dataset's stored photos join
// the duplicate pass. items: [] is valid and returns the formats the
// server converts. At most photocheck.MaxAssessItems photos and
// photocheck.MaxAssessBody bytes.
func assessPhotos(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, photocheck.MaxAssessBody+1))
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "could not read the request body: "+err.Error())
		return
	}
	if len(body) > photocheck.MaxAssessBody {
		v1api.Fail(w, v1api.CodeTooLarge, fmt.Sprintf("the body is over %d MiB; send the photos in smaller batches", photocheck.MaxAssessBody>>20))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	var req photocheck.AssessRequest
	if !v1api.Decode(w, r, &req) {
		return
	}
	dir := ""
	if name := strings.TrimSpace(req.Dataset); name != "" {
		var ok bool
		if dir, ok = datasetDir(w, r, name, true); !ok {
			return
		}
	}
	rv, err := photocheck.Assess(r.Context(), req, dir)
	if err != nil {
		photoFail(w, r, err)
		return
	}
	v1api.OK(w, rv)
}

// checkPhotos is GET /v1/datasets/{name}/photos/check?kind=&resolution=&pick=&wait_ms=:
// every stored photo with its flags, the duplicate groups, findings, grade,
// coverage and hints. It waits up to wait_ms (default 3000, at most 20000)
// for photos not analysed yet; pending > 0 means the analysis carries on in
// the background and the client polls. resolution is 1024 or 512,768,1024.
func checkPhotos(w http.ResponseWriter, r *http.Request) {
	dir, ok := datasetDir(w, r, r.PathValue("name"), true)
	if !ok {
		return
	}
	q := r.URL.Query()
	opts := photocheck.Options{Kind: photocheck.Kind(strings.TrimSpace(q.Get("kind"))), Wait: defaultPhotoWait}
	var err error
	if opts.Resolutions, err = photocheck.ParseResolutionList(q.Get("resolution")); err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	if s := strings.TrimSpace(q.Get("pick")); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 || n > photocheck.MaxAssessItems {
			v1api.Fail(w, v1api.CodeInvalidRequest, fmt.Sprintf("pick must be a whole number between 0 and %d", photocheck.MaxAssessItems))
			return
		}
		opts.Pick = n
	}
	if s := strings.TrimSpace(q.Get("wait_ms")); s != "" {
		if opts.Wait, ok = photoWait(s); !ok {
			v1api.Fail(w, v1api.CodeInvalidRequest, "wait_ms must be a whole number of milliseconds")
			return
		}
	}
	rv, err := photocheck.ReviewDataset(r.Context(), dir, opts)
	if err != nil {
		photoFail(w, r, err)
		return
	}
	answerReview(w, r, rv)
}

// photoWait reads wait_ms, capped at maxPhotoWait. The cap applies before
// the conversion, since a huge ms overflows a Duration (and a negative wait
// means do not wait), and a number past the int range is past the cap too.
func photoWait(s string) (time.Duration, bool) {
	ms, err := strconv.Atoi(s)
	if errors.Is(err, strconv.ErrRange) && !strings.HasPrefix(s, "-") {
		ms, err = math.MaxInt, nil
	}
	if err != nil || ms < 0 {
		return 0, false
	}
	return time.Duration(min(ms, int(maxPhotoWait/time.Millisecond))) * time.Millisecond, true
}

// putPhotoSettings is PUT /v1/datasets/{name}/photos/settings: the subject
// kind, the "this is them" reference and per-photo decisions, by file and
// stored by content (null clears). It answers the new review.
func putPhotoSettings(w http.ResponseWriter, r *http.Request) {
	dir, ok := datasetDir(w, r, r.PathValue("name"), true)
	if !ok {
		return
	}
	var p photocheck.SettingsPatch
	if !v1api.Decode(w, r, &p) {
		return
	}
	rv, err := photocheck.SaveSettings(r.Context(), dir, p)
	if err != nil {
		photoFail(w, r, err)
		return
	}
	answerReview(w, r, rv)
}

// forgetPhotos is DELETE /v1/datasets/{name}/photos/analysis: the dataset's
// scores, face results, face similarities, reference and decisions go; the
// photos stay.
func forgetPhotos(w http.ResponseWriter, r *http.Request) {
	dir, ok := datasetDir(w, r, r.PathValue("name"), true)
	if !ok {
		return
	}
	if err := photocheck.Forget(r.Context(), dir); err != nil {
		photoFail(w, r, err)
		return
	}
	v1api.NoContent(w)
}
