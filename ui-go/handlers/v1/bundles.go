package v1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/bundles"
	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/portable"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Run bundles (docs/headless.md, "Run bundles"): a job's dataset(s) and
// folder in one zip, downloaded, pushed to a remote's bundles/ library, or
// imported here to continue training. The format is internal/bundles.

const libraryPrefix = "bundles/"

// at most this many bundle tasks (imports, pushes) run at once
var (
	taskSlots = make(chan struct{}, 2)
	tasksWG   sync.WaitGroup // tests wait for task goroutines to end
)

func runTask(fn func()) {
	tasksWG.Add(1)
	go func() {
		defer tasksWG.Done()
		taskSlots <- struct{}{}
		defer func() { <-taskSlots }()
		fn()
	}()
}

func boolParam(s string, def bool) (bool, error) {
	switch strings.ToLower(s) {
	case "":
		return def, nil
	case "1", "true", "yes":
		return true, nil
	case "0", "false", "no":
		return false, nil
	}
	return false, fmt.Errorf("%q is not a boolean", s)
}

func checkpointsParam(s string) (string, error) {
	switch s {
	case "":
		return bundles.CheckpointsLatest, nil
	case bundles.CheckpointsAll, bundles.CheckpointsLatest, bundles.CheckpointsNone:
		return s, nil
	}
	return "", errors.New("checkpoints must be all, latest or none")
}

// bundleOptionsQuery reads ?checkpoints=&optimizer=&samples=&caches=
// (latents= is an alias of caches).
func bundleOptionsQuery(q url.Values) (bundles.Options, error) {
	opt := bundles.DefaultOptions()
	var err error
	if opt.Checkpoints, err = checkpointsParam(q.Get("checkpoints")); err != nil {
		return opt, v1api.Errorf(v1api.CodeInvalidRequest, err.Error())
	}
	if !q.Has("caches") && q.Has("latents") {
		q.Set("caches", q.Get("latents"))
	}
	for name, dst := range map[string]*bool{"optimizer": &opt.Optimizer, "samples": &opt.Samples, "caches": &opt.Caches} {
		if *dst, err = boolParam(q.Get(name), *dst); err != nil {
			return opt, v1api.Errorf(v1api.CodeInvalidRequest, name+": "+err.Error())
		}
	}
	return opt, nil
}

// bundleDatasets orders the datasets a portable config references: the
// first process's folder_paths first (data/, data_1/, ...), then any other.
func bundleDatasets(cfg json.RawMessage, names []string, dsRoot string) ([]bundles.SourceDataset, []string) {
	var order, warnings []string
	seen := map[string]bool{}
	if o, err := jsonx.ParseObject(cfg); err == nil {
		if inner := o.Object("config"); inner != nil && len(inner.Array("process")) > 0 {
			if p0, ok := inner.Array("process")[0].(*jsonx.Object); ok {
				for _, d := range p0.Array("datasets") {
					do, ok := d.(*jsonx.Object)
					if !ok {
						continue
					}
					v, _ := do.Get("folder_path")
					fp, _ := v.(string)
					rest, ok := strings.CutPrefix(fp, "aitk://datasets/")
					if !ok {
						if fp != "" {
							warnings = append(warnings, "dataset folder "+fp+" is outside the datasets folder and is not in the bundle")
						}
						continue
					}
					n, _, _ := strings.Cut(rest, "/")
					if !seen[n] {
						seen[n] = true
						order = append(order, n)
					}
				}
			}
		}
	}
	for _, n := range names {
		if !seen[n] {
			seen[n] = true
			order = append(order, n)
		}
	}
	var out []bundles.SourceDataset
	for _, n := range order {
		dir := filepath.Join(dsRoot, n)
		if st, err := os.Stat(dir); err != nil || !st.IsDir() {
			warnings = append(warnings, "dataset "+n+" is missing on this server and is not in the bundle")
			continue
		}
		out = append(out, bundles.SourceDataset{Name: n, Dir: dir})
	}
	return out, warnings
}

// buildBundle collects a job's bundle. cleanup removes its snapshots.
func buildBundle(ctx context.Context, j *store.Job, opt bundles.Options) ([]bundles.Entry, *bundles.Manifest, func(), error) {
	nothing := func() {}
	dsRoot, err := settings.DatasetsRoot(ctx)
	if err != nil {
		return nil, nil, nothing, err
	}
	trRoot, err := settings.TrainingFolder(ctx)
	if err != nil {
		return nil, nil, nothing, err
	}
	cfg, names, err := portable.ToPortable(j.JobConfig, dsRoot, trRoot)
	if err != nil {
		return nil, nil, nothing, v1api.Errorf(v1api.CodeInvalidConfig, "job config: "+err.Error())
	}
	pol, err := state.GetPolicy(ctx, j.ID)
	if err != nil {
		return nil, nil, nothing, err
	}
	staging, err := bundles.StagingDir(ctx)
	if err != nil {
		return nil, nil, nothing, err
	}
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return nil, nil, nothing, err
	}
	tmp, err := os.MkdirTemp(staging, "export-")
	if err != nil {
		return nil, nil, nothing, err
	}
	cleanup := func() { os.RemoveAll(tmp) }
	datasets, warnings := bundleDatasets(cfg, names, dsRoot)
	m := bundles.Manifest{Name: j.Name, ServerVersion: version(), UpstreamVersion: upstreamVersion(), Type: j.JobType,
		TotalSteps: j.TotalSteps, Config: cfg, Policy: bundles.Policy{AutoResume: pol.AutoResume, MaxAutoResumes: pol.MaxAutoResumes},
		Warnings: warnings}
	m.BaseModel, m.Arch, _ = bundles.Describe(cfg)
	if m.TotalSteps == nil {
		_, _, m.TotalSteps = bundles.Describe(cfg)
	}
	entries, man, err := bundles.Collect(ctx, bundles.Source{JobFolder: filepath.Join(trRoot, j.Name), Datasets: datasets,
		Manifest: m, Running: busy(j.Status), TempDir: tmp}, opt)
	if err != nil {
		cleanup()
		return nil, nil, nothing, err
	}
	return entries, man, cleanup, nil
}

func bundleFileName(m *bundles.Manifest) string { return fmt.Sprintf("%s_step%d", m.Name, m.Step) }

// downloadBundle streams the job's bundle (GET also answers HEAD). Stored
// entries make its size known up front, so Content-Length is exact.
func downloadBundle(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	opt, err := bundleOptionsQuery(r.URL.Query())
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	entries, m, cleanup, err := buildBundle(ctx, j, opt)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	defer cleanup()
	size, err := bundles.Size(entries)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	h := w.Header()
	h.Set("Content-Type", "application/zip")
	h.Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": bundleFileName(m) + ".zip"}))
	h.Set("Content-Length", strconv.FormatInt(size, 10))
	h.Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodHead {
		return
	}
	if err := bundles.Write(ctx, w, entries, nil); err != nil && ctx.Err() == nil {
		log.Printf("bundle %s: %v", j.Name, err)
	}
}

// validBundleID keeps library ids to one object name.
func validBundleID(id string) error {
	if err := validJobName(id); err != nil {
		return errors.New(strings.Replace(err.Error(), "name", "bundle id", 1))
	}
	return nil
}

func libraryKeys(id string) (zipKey, sidecar string) {
	return libraryPrefix + id + ".zip", libraryPrefix + id + ".json"
}

func openRemote(ctx context.Context, name string) (*remote.Remote, remote.Client, error) {
	if name == "" {
		return nil, nil, v1api.Errorf(v1api.CodeInvalidRequest, "remote is required")
	}
	rem, c, err := remote.Open(ctx, name)
	if errors.Is(err, remote.ErrNotFound) {
		return nil, nil, v1api.Errorf(v1api.CodeNotFound, "remote "+name+" not found")
	}
	if err != nil {
		return nil, nil, v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	return rem, c, nil
}

func pushBundle(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	var b struct {
		Remote      string `json:"remote"`
		Bundle      string `json:"bundle"`
		Overwrite   bool   `json:"overwrite"`
		Checkpoints string `json:"checkpoints"`
		Optimizer   *bool  `json:"optimizer"`
		Samples     *bool  `json:"samples"`
		Caches      *bool  `json:"caches"`
		Latents     *bool  `json:"latents"` // alias of caches
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	opt := bundles.DefaultOptions()
	var err error
	if opt.Checkpoints, err = checkpointsParam(b.Checkpoints); err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	if b.Caches == nil {
		b.Caches = b.Latents
	}
	for src, dst := range map[*bool]*bool{b.Optimizer: &opt.Optimizer, b.Samples: &opt.Samples, b.Caches: &opt.Caches} {
		if src != nil {
			*dst = *src
		}
	}
	if b.Bundle != "" {
		if err := validBundleID(b.Bundle); err != nil {
			v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
			return
		}
	}
	rem, c, err := openRemote(ctx, b.Remote)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	if b.Bundle != "" && !b.Overwrite {
		zipKey, _ := libraryKeys(b.Bundle)
		if _, err := c.Stat(ctx, zipKey); err == nil {
			v1api.Fail(w, v1api.CodeConflict, "remote "+rem.Name+" already has bundle "+b.Bundle+" (pass overwrite)")
			return
		}
	}
	t := bundles.NewTask(ctx, bundles.KindPush)
	jobID, jobName, remoteName := j.ID, j.Name, rem.Name
	t.JobID, t.JobName, t.Remote, t.Source = &jobID, &jobName, &remoteName, "remote"
	t.Save(ctx, true)
	reply := *t // the task goroutine changes t from here on
	runTask(func() { runPush(context.Background(), t, j.ID, c, b.Bundle, b.Overwrite, opt) })
	v1api.JSON(w, http.StatusAccepted, &reply)
}

func failTask(ctx context.Context, t *bundles.Task, err error) {
	var p *v1api.Problem
	if errors.As(err, &p) {
		t.Fail(ctx, p.Code, p.Message)
		return
	}
	log.Printf("bundle %s %s: %v", t.Kind, t.ID, err)
	t.Fail(ctx, v1api.CodeInternal, err.Error())
}

// runPush builds the bundle into the staging folder, uploads it to
// bundles/<id>.zip and writes the sidecar summary next to it.
func runPush(ctx context.Context, t *bundles.Task, jobID string, c remote.Client, id string, overwrite bool, opt bundles.Options) {
	t.State, t.Phase = bundles.TaskRunning, "building"
	t.Save(ctx, true)
	j, err := store.GetJob(ctx, jobID)
	if err != nil || j == nil {
		failTask(ctx, t, v1api.Errorf(v1api.CodeNotFound, "job not found"))
		return
	}
	entries, m, cleanup, err := buildBundle(ctx, j, opt)
	if err != nil {
		failTask(ctx, t, err)
		return
	}
	defer cleanup()
	if id == "" {
		id = bundleFileName(m)
		if !overwrite {
			zipKey, _ := libraryKeys(id)
			if _, err := c.Stat(ctx, zipKey); err == nil {
				failTask(ctx, t, v1api.Errorf(v1api.CodeConflict, "the remote already has bundle "+id+" (pass overwrite)"))
				return
			}
		}
	}
	zipKey, sideKey := libraryKeys(id)
	t.Key, t.Warnings = &zipKey, m.Warnings
	if t.Bytes, err = bundles.Size(entries); err != nil {
		failTask(ctx, t, err)
		return
	}
	staging, _ := bundles.StagingDir(ctx)
	f, err := os.CreateTemp(staging, "push-*.zip")
	if err != nil {
		failTask(ctx, t, err)
		return
	}
	defer os.Remove(f.Name())
	h := sha256.New()
	err = bundles.Write(ctx, io.MultiWriter(f, h), entries, func(n int64) {
		t.BytesDone = n
		t.Save(ctx, false)
	})
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		failTask(ctx, t, err)
		return
	}
	sum := hex.EncodeToString(h.Sum(nil))

	t.Phase, t.BytesDone = "uploading", 0
	t.Save(ctx, true)
	u, err := c.Put(ctx, zipKey, f.Name(), func(done int64) {
		t.BytesDone = min(done, t.Bytes)
		t.Save(ctx, false)
	})
	if err != nil {
		failTask(ctx, t, v1api.Errorf(v1api.CodeRemoteFailed, err.Error()))
		return
	}
	side, _ := json.MarshalIndent(bundles.SummaryOf(m, t.Bytes, sum), "", "  ")
	if err := c.PutBytes(ctx, sideKey, side); err != nil {
		failTask(ctx, t, v1api.Errorf(v1api.CodeRemoteFailed, "bundle uploaded but its summary failed: "+err.Error()))
		return
	}
	t.URL = &u
	t.State, t.Phase, t.BytesDone = bundles.TaskDone, "", t.Bytes
	t.Save(ctx, true)
}

// libraryItem is a bundle in a remote's library.
type libraryItem struct {
	ID       string           `json:"id"`
	Key      string           `json:"key"`
	Size     int64            `json:"size"`
	Modified string           `json:"modified"`
	Summary  *bundles.Summary `json:"summary"` // null when the zip has no sidecar
}

func listLibrary(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	_, c, err := openRemote(ctx, r.URL.Query().Get("remote"))
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	objs, err := c.List(ctx, libraryPrefix)
	if err != nil {
		v1api.Fail(w, v1api.CodeRemoteFailed, err.Error())
		return
	}
	sidecars := map[string]bool{}
	for _, o := range objs {
		if id, ok := strings.CutSuffix(strings.TrimPrefix(o.Key, libraryPrefix), ".json"); ok {
			sidecars[id] = true
		}
	}
	items := []libraryItem{}
	for _, o := range objs {
		id, ok := strings.CutSuffix(strings.TrimPrefix(o.Key, libraryPrefix), ".zip")
		if !ok || strings.Contains(id, "/") {
			continue
		}
		it := libraryItem{ID: id, Key: o.Key, Size: o.Size, Modified: v1api.Time(o.Modified)}
		if sidecars[id] {
			_, sideKey := libraryKeys(id)
			if b, err := c.GetBytes(ctx, sideKey); err == nil {
				var s bundles.Summary
				if json.Unmarshal(b, &s) == nil {
					it.Summary = &s
				}
			}
		}
		items = append(items, it)
	}
	sort.SliceStable(items, func(i, j int) bool { return items[i].Modified > items[j].Modified })
	v1api.Items(w, items)
}

func deleteLibraryBundle(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("id")
	if err := validBundleID(id); err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	_, c, err := openRemote(ctx, r.URL.Query().Get("remote"))
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	zipKey, sideKey := libraryKeys(id)
	if _, err := c.Stat(ctx, zipKey); errors.Is(err, remote.ErrNotFound) {
		v1api.Fail(w, v1api.CodeNotFound, "no bundle "+id+" in the library")
		return
	} else if err != nil {
		v1api.Fail(w, v1api.CodeRemoteFailed, err.Error())
		return
	}
	for _, k := range []string{zipKey, sideKey} {
		if err := c.Delete(ctx, k); err != nil && !errors.Is(err, remote.ErrNotFound) {
			v1api.Fail(w, v1api.CodeRemoteFailed, err.Error())
			return
		}
	}
	v1api.NoContent(w)
}

func taskByID(kind string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		t, err := bundles.GetTask(r.Context(), kind, r.PathValue("id"))
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if t == nil {
			v1api.Fail(w, v1api.CodeNotFound, kind+" not found")
			return
		}
		v1api.OK(w, t)
	}
}

func listTasks(kind string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		items, err := bundles.ListTasks(r.Context(), kind)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if job := r.URL.Query().Get("job_id"); job != "" {
			kept := items[:0]
			for _, t := range items {
				if t.JobID != nil && *t.JobID == job {
					kept = append(kept, t)
				}
			}
			items = kept
		}
		v1api.Items(w, items)
	}
}

// bundleLoop fails tasks a previous process left running, then expires
// stale upload sessions hourly.
func bundleLoop(ctx context.Context) {
	if state.DB != nil {
		bundles.RecoverTasks(ctx, startedAt)
	}
	t := time.NewTicker(time.Hour)
	defer t.Stop()
	for {
		if store.DB != nil {
			bundles.ExpireUploads(ctx)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}
