package portable

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// blobGrace keeps blobs written this recently: a dataset push uploads its
// blobs before its manifest, so a fresh blob may belong to a push in flight
// (tests shorten it).
var blobGrace = time.Hour

// Deleted is what DeleteRemoteRun removed.
type Deleted struct {
	Name     string   `json:"name"`
	Objects  int      `json:"objects"`
	Bytes    int64    `json:"bytes"`
	Datasets []string `json:"datasets"` // datasets no other run on the remote used
}

// DeleteRemoteRun deletes a run's copy on a remote: everything under
// jobs/<name>/, its datasets when no other run there names them, and the
// blobs no dataset references any more. A run whose lease is live (it is
// training somewhere) is refused unless force. Nothing on any server is
// touched: a job here keeps its local files.
func DeleteRemoteRun(ctx context.Context, remoteName, name string, force bool) (Deleted, error) {
	out := Deleted{Name: name, Datasets: []string{}}
	if err := ValidName(name); err != nil {
		return out, v1api.Errorf(v1api.CodeInvalidRequest, err.Error())
	}
	r, c, err := remote.Open(ctx, remoteName)
	if err != nil {
		return out, remoteErr(remoteName, err)
	}
	unlock := lockManifest(name)
	defer unlock()
	m, err := readManifest(ctx, c, name)
	if err != nil {
		return out, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	objs, err := c.List(ctx, jobKey(name, ""))
	if err != nil {
		return out, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	if m == nil && len(objs) == 0 {
		return out, v1api.Errorf(v1api.CodeNotFound, fmt.Sprintf("no run %q on %s", name, r.Name))
	}
	if m != nil && m.Lease != nil && time.Now().Before(m.Lease.ExpiresAt) && !force {
		return out, v1api.Errorf(v1api.CodeConflict, fmt.Sprintf("%s is training on %s (lease until %s): stop it there first, or delete with force",
			name, m.Lease.NodeLabel, m.Lease.ExpiresAt.Format(time.RFC3339)))
	}
	// the manifest last, so an interrupted delete still lists the run
	manifest := manifestKey(name)
	for _, o := range objs {
		if o.Key == manifest {
			continue
		}
		if err := c.Delete(ctx, o.Key); err != nil {
			return out, v1api.Errorf(v1api.CodeRemoteFailed, fmt.Sprintf("delete %s: %v", o.Key, err))
		}
		out.Objects++
		out.Bytes += o.Size
	}
	for _, o := range objs {
		if o.Key == manifest {
			if err := c.Delete(ctx, o.Key); err != nil {
				return out, v1api.Errorf(v1api.CodeRemoteFailed, fmt.Sprintf("delete %s: %v", o.Key, err))
			}
			out.Objects++
			out.Bytes += o.Size
		}
	}
	if m == nil || len(m.Datasets) == 0 {
		return out, nil
	}

	// datasets no other run names
	used, err := datasetsInUse(ctx, c, name)
	if err != nil {
		return out, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	for _, ds := range m.Datasets {
		if used[ds] || ValidName(ds) != nil {
			continue
		}
		st, err := c.Stat(ctx, datasetKey(ds))
		if err != nil {
			continue // already gone
		}
		if err := c.Delete(ctx, datasetKey(ds)); err != nil {
			return out, v1api.Errorf(v1api.CodeRemoteFailed, fmt.Sprintf("delete dataset %s: %v", ds, err))
		}
		out.Objects++
		out.Bytes += st.Size
		out.Datasets = append(out.Datasets, ds)
		pushedMu.Lock()
		delete(pushed, r.ID+"\x00"+ds) // a job here pushes it again on its next sync
		pushedMu.Unlock()
	}
	if len(out.Datasets) == 0 {
		return out, nil
	}
	n, size, err := sweepBlobs(ctx, c)
	out.Objects += n
	out.Bytes += size
	if err != nil {
		return out, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	return out, nil
}

// datasetsInUse is every dataset a run's manifest names, except run skip.
func datasetsInUse(ctx context.Context, c remote.Client, skip string) (map[string]bool, error) {
	objs, err := c.List(ctx, "jobs/")
	if err != nil {
		return nil, err
	}
	used := map[string]bool{}
	for _, o := range objs {
		run, file, ok := strings.Cut(strings.TrimPrefix(o.Key, "jobs/"), "/")
		if !ok || file != "manifest.json" || run == skip {
			continue
		}
		m, err := readManifest(ctx, c, run)
		if err != nil {
			return nil, err
		}
		if m != nil {
			for _, ds := range m.Datasets {
				used[ds] = true
			}
		}
	}
	return used, nil
}

// sweepBlobs deletes blobs no dataset manifest references, except recent
// ones (blobGrace).
func sweepBlobs(ctx context.Context, c remote.Client) (int, int64, error) {
	manifests, err := c.List(ctx, "datasets/")
	if err != nil {
		return 0, 0, err
	}
	ref := map[string]bool{}
	for _, o := range manifests {
		if !strings.HasSuffix(o.Key, "/manifest.json") {
			continue
		}
		b, err := c.GetBytes(ctx, o.Key)
		if err != nil {
			return 0, 0, err
		}
		var dm DatasetManifest
		if err := json.Unmarshal(b, &dm); err != nil {
			// an unreadable manifest might reference anything: keep every blob
			return 0, 0, fmt.Errorf("dataset manifest %s is unreadable, so no blobs were removed: %v", o.Key, err)
		}
		for _, f := range dm.Files {
			ref[f.SHA256] = true
		}
	}
	blobs, err := c.List(ctx, "blobs/")
	if err != nil {
		return 0, 0, err
	}
	var n int
	var size int64
	cutoff := time.Now().Add(-blobGrace)
	for _, o := range blobs {
		if ref[strings.TrimPrefix(o.Key, "blobs/")] || o.Modified.After(cutoff) {
			continue
		}
		if err := c.Delete(ctx, o.Key); err != nil {
			return n, size, fmt.Errorf("delete %s: %w", o.Key, err)
		}
		n++
		size += o.Size
	}
	return n, size, nil
}
