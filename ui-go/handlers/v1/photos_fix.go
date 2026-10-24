package v1

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// fixPhotos is POST /v1/datasets/{name}/photos/fix: the photo fix ops run
// in order (photocheck.FixOps). Without apply it is a preview and nothing
// is written. With apply it answers 409 while a training run reads the
// dataset or a caption job writes it; the dataset's own face check never
// blocks. Every applied change emits dataset.changed once per request.
func fixPhotos(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	var req photocheck.FixRequest
	if !v1api.Decode(w, r, &req) {
		return
	}
	// an unknown op is a 400 even while a job holds the dataset
	for _, op := range req.Ops {
		if !slices.Contains(photocheck.FixOps, op.Op) {
			v1api.Fail(w, v1api.CodeInvalidRequest, fmt.Sprintf("unknown op %q; valid ops: %s", op.Op, strings.Join(photocheck.FixOps, ", ")))
			return
		}
	}
	if req.Apply {
		msg, err := photoFixBlocked(ctx, dir)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if msg != "" {
			v1api.Fail(w, v1api.CodeConflict, msg)
			return
		}
	}
	fix, err := photocheck.ApplyFix(ctx, dir, req)
	if err != nil {
		photoFail(w, r, err)
		return
	}
	if fix.Applied && len(fix.Changes) > 0 {
		changed(ctx, name)
	}
	if err := withFaceJob(ctx, fix.Review); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, fix)
}

// photoFixBlocked says why photo fixes cannot be applied to dir now ("" when
// they can): a queued, running or stopping training run reads the folder,
// or a caption job writes into it. photo_faces jobs only read the photos
// and write their results to their own folder, so they never block.
func photoFixBlocked(ctx context.Context, dir string) (string, error) {
	jobs, err := datasetJobs(ctx, dir)
	if err != nil {
		return "", err
	}
	for _, j := range jobs {
		switch j.JobType {
		case photocheck.FaceJobType:
		case "caption":
			return "Auto caption is running on this dataset. Wait for it or stop it", nil
		case "train", "":
			return j.Name + " is training on this dataset", nil
		default:
			return j.Name + " is using this dataset", nil
		}
	}
	return "", nil
}
