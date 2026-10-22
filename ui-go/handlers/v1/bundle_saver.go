package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/bundles"
	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/store"
)

// bundleSaver keeps a finished job as a run bundle in the library of the
// AITK_WHEN_DONE_BUNDLE remote before the server stops or is destroyed, so
// the run can be resumed on any server (docs/bundle-format.md). It runs
// before the sync and upload savers: a bundle is the whole run, not just its
// newest checkpoint.
type bundleSaver struct{}

func (bundleSaver) Name() string { return "bundle" }

func (bundleSaver) Save(ctx context.Context, j *store.Job, newest artifacts.Checkpoint) (lifecycle.Proof, bool, error) {
	ref := strings.TrimSpace(os.Getenv("AITK_WHEN_DONE_BUNDLE"))
	if ref == "" {
		return lifecycle.Proof{}, false, nil
	}
	rem, c, err := openRemote(ctx, ref)
	if err != nil {
		return lifecycle.Proof{}, true, err
	}
	opt := bundles.DefaultOptions()
	_, m, cleanup, err := buildBundle(ctx, j, opt)
	if err != nil {
		return lifecycle.Proof{}, true, err
	}
	cleanup()
	id := bundleFileName(m)
	zipKey, sideKey := libraryKeys(id)
	proof := lifecycle.Proof{Via: "bundle:" + rem.Name, Detail: zipKey + " in " + rem.Name}

	// verified holds when the library has this bundle with the newest
	// checkpoint in it and the zip at the size its summary records
	verified := func() error {
		b, err := c.GetBytes(ctx, sideKey)
		if err != nil {
			return fmt.Errorf("%s: %w", sideKey, err)
		}
		var s bundles.Summary
		if err := json.Unmarshal(b, &s); err != nil {
			return fmt.Errorf("%s: %w", sideKey, err)
		}
		if !slices.Contains(s.Checkpoints, newest.Name) {
			return fmt.Errorf("%s does not hold %s", zipKey, newest.Name)
		}
		o, err := c.Stat(ctx, zipKey)
		if err != nil {
			return fmt.Errorf("%s: %w", zipKey, err)
		}
		if o.Size != s.Size {
			return fmt.Errorf("%s is %d bytes on the remote, its summary says %d", zipKey, o.Size, s.Size)
		}
		return nil
	}
	if verified() == nil {
		return proof, true, nil
	}

	t := bundles.NewTask(ctx, bundles.KindPush)
	jobID, jobName, remoteName := j.ID, j.Name, rem.Name
	t.JobID, t.JobName, t.Remote, t.Source = &jobID, &jobName, &remoteName, "remote"
	t.Save(ctx, true)
	runPush(ctx, t, j.ID, c, id, true, opt)
	if t.State != bundles.TaskDone {
		msg := "bundle push did not finish"
		if t.Error != nil {
			msg = t.Error.Message
		}
		return lifecycle.Proof{}, true, fmt.Errorf("push %s: %s", zipKey, msg)
	}
	if err := verified(); err != nil {
		return lifecycle.Proof{}, true, err
	}
	return proof, true, nil
}
