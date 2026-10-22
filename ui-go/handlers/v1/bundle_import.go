package v1

import (
	"archive/zip"
	"context"
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
	"runtime"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/bundles"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/portable"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/supervisor"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// importOpts are what an import does with the bundle once it is here.
type importOpts struct {
	name, dataset, gpu string
	policy             *state.Policy
	start              bool
}

// importBody is POST /v1/bundles/imports: one source (url, remote + key,
// or upload_id) plus the options.
type importBody struct {
	URL      string          `json:"url"`
	Remote   string          `json:"remote"`
	Key      string          `json:"key"`
	UploadID string          `json:"upload_id"`
	Name     string          `json:"name"`
	Dataset  string          `json:"dataset"`
	GpuIDs   string          `json:"gpu_ids"`
	Policy   json.RawMessage `json:"policy"`
	Start    bool            `json:"start"`
}

func (b importBody) opts(ctx context.Context) (importOpts, error) {
	o := importOpts{name: strings.TrimSpace(b.Name), dataset: strings.TrimSpace(b.Dataset), gpu: strings.TrimSpace(b.GpuIDs), start: b.Start}
	if o.name != "" {
		if err := validJobName(o.name); err != nil {
			return o, v1api.Errorf(v1api.CodeInvalidRequest, err.Error())
		}
	}
	if o.dataset != "" {
		if err := validDatasetName(o.dataset); err != nil {
			return o, v1api.Errorf(v1api.CodeInvalidRequest, err.Error())
		}
	}
	if len(b.Policy) > 0 && string(b.Policy) != "null" {
		var p state.Policy
		if err := policyPatch(ctx, b.Policy, &p); err != nil {
			return o, err
		}
		o.policy = &p
	}
	return o, nil
}

// createImport starts an import. JSON names the source; any other body is
// the bundle itself (raw zip with options in the query, or multipart with
// a "file" part and option fields), staged as an upload first.
func createImport(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if mt != "" && mt != "application/json" {
		importBodyUpload(w, r, mt)
		return
	}
	var b importBody
	if !v1api.Decode(w, r, &b) {
		return
	}
	o, err := b.opts(ctx)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	n := 0
	for _, set := range []bool{b.URL != "", b.Remote != "" || b.Key != "", b.UploadID != ""} {
		if set {
			n++
		}
	}
	if n != 1 {
		v1api.Fail(w, v1api.CodeInvalidRequest, "give exactly one source: url, remote and key, or upload_id")
		return
	}
	switch {
	case b.UploadID != "":
		startUploadImport(w, r, b.UploadID, o, "upload")
	case b.URL != "":
		u, err := url.Parse(b.URL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			v1api.Fail(w, v1api.CodeInvalidRequest, "url must be an http(s) URL")
			return
		}
		t := bundles.NewTask(ctx, bundles.KindImport)
		shown := (&url.URL{Scheme: u.Scheme, Host: u.Host, Path: u.Path}).String()
		t.Source, t.URL = "url", &shown
		t.Save(ctx, true)
		reply := *t // the task goroutine changes t from here on
		runTask(func() {
			bg := context.Background()
			id, err := stageURL(bg, t, b.URL)
			if err != nil {
				failTask(bg, t, err)
				return
			}
			prepareAndRestore(bg, t, id, o)
		})
		v1api.JSON(w, http.StatusAccepted, &reply)
	default:
		if b.Remote == "" || b.Key == "" {
			v1api.Fail(w, v1api.CodeInvalidRequest, "remote and key go together")
			return
		}
		rem, c, err := openRemote(ctx, b.Remote)
		if err != nil {
			v1api.Respond(w, r, err)
			return
		}
		obj, err := c.Stat(ctx, b.Key)
		if errors.Is(err, remote.ErrNotFound) {
			v1api.Fail(w, v1api.CodeNotFound, "remote "+rem.Name+" has no "+b.Key)
			return
		} else if err != nil {
			v1api.Fail(w, v1api.CodeRemoteFailed, err.Error())
			return
		}
		t := bundles.NewTask(ctx, bundles.KindImport)
		name, key := rem.Name, b.Key
		t.Source, t.Remote, t.Key, t.Bytes = "remote", &name, &key, obj.Size
		t.Save(ctx, true)
		reply := *t // the task goroutine changes t from here on
		runTask(func() {
			bg := context.Background()
			id, err := stageRemote(bg, t, c, b.Key, obj.Size)
			if err != nil {
				failTask(bg, t, err)
				return
			}
			prepareAndRestore(bg, t, id, o)
		})
		v1api.JSON(w, http.StatusAccepted, &reply)
	}
}

// startUploadImport plans the import of a complete upload now, so name and
// dataset conflicts answer 409 right away, and restores in the background.
func startUploadImport(w http.ResponseWriter, r *http.Request, uploadID string, o importOpts, source string) {
	ctx := r.Context()
	u, err := bundles.GetUpload(ctx, uploadID)
	if errors.Is(err, bundles.ErrNoUpload) {
		v1api.Fail(w, v1api.CodeNotFound, "upload "+uploadID+" not found (uploads expire a day after their last chunk)")
		return
	} else if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if !u.Complete {
		v1api.Fail(w, v1api.CodeConflict, fmt.Sprintf("upload %s has %d of %d bytes", u.ID, u.Received, u.Size))
		return
	}
	prep, err := prepareImport(ctx, uploadID, o)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	t := bundles.NewTask(ctx, bundles.KindImport)
	id := uploadID
	t.Source, t.UploadID = source, &id
	t.Save(ctx, true)
	reply := *t // the task goroutine changes t from here on
	runTask(func() {
		bg := context.Background()
		defer prep.close()
		if err := restoreImport(bg, t, prep, o); err != nil {
			failTask(bg, t, err)
		}
	})
	v1api.JSON(w, http.StatusAccepted, &reply)
}

func prepareAndRestore(ctx context.Context, t *bundles.Task, uploadID string, o importOpts) {
	t.UploadID = &uploadID
	t.Phase, t.Bytes, t.BytesDone = "reading", 0, 0
	t.Save(ctx, true)
	prep, err := prepareImport(ctx, uploadID, o)
	if err != nil {
		failTask(ctx, t, err)
		return
	}
	defer prep.close()
	if err := restoreImport(ctx, t, prep, o); err != nil {
		failTask(ctx, t, err)
	}
}

// stageBody copies a bundle into a new upload session. size -1 = unknown.
func stageBody(ctx context.Context, src io.Reader, size int64, filename string, progress func(int64)) (*bundles.Upload, error) {
	if size > bundles.MaxBytes {
		return nil, v1api.Errorf(v1api.CodeTooLarge, "bundle is larger than 1 TiB")
	}
	u, err := bundles.CreateUpload(ctx, size, filename)
	if err != nil {
		return nil, err
	}
	p, err := bundles.UploadPath(ctx, u.ID)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*bundles.Upload, error) {
		bundles.DeleteUpload(ctx, u.ID)
		return nil, err
	}
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fail(err)
	}
	var r io.Reader = io.LimitReader(src, bundles.MaxBytes+1)
	if progress != nil {
		r = &progressReader{r: r, fn: progress}
	}
	n, err := io.Copy(f, r)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	switch {
	case err != nil:
		return fail(err)
	case n > bundles.MaxBytes:
		return fail(v1api.Errorf(v1api.CodeTooLarge, "bundle is larger than 1 TiB"))
	case size >= 0 && n != size:
		return fail(v1api.Errorf(v1api.CodeInvalidRequest, fmt.Sprintf("received %d of %d bytes", n, size)))
	}
	return bundles.FinishUpload(ctx, u.ID)
}

type progressReader struct {
	r  io.Reader
	n  int64
	fn func(int64)
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.n += int64(n)
	p.fn(p.n)
	return n, err
}

// stageURL downloads the bundle at u into an upload session.
func stageURL(ctx context.Context, t *bundles.Task, u string) (string, error) {
	t.State, t.Phase = bundles.TaskRunning, "downloading"
	t.Save(ctx, true)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", v1api.Errorf(v1api.CodeInvalidRequest, err.Error())
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", v1api.Errorf(v1api.CodeRemoteFailed, "download: "+err.Error())
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", v1api.Errorf(v1api.CodeRemoteFailed, "download: "+resp.Status)
	}
	t.Bytes = max(resp.ContentLength, 0)
	name := filepath.Base(req.URL.Path)
	up, err := stageBody(ctx, resp.Body, resp.ContentLength, name, func(n int64) {
		t.BytesDone = n
		t.Save(ctx, false)
	})
	if err != nil {
		return "", err
	}
	return up.ID, nil
}

// stageRemote copies the remote's object into an upload session.
func stageRemote(ctx context.Context, t *bundles.Task, c remote.Client, key string, size int64) (string, error) {
	t.State, t.Phase = bundles.TaskRunning, "downloading"
	t.Save(ctx, true)
	up, err := bundles.CreateUpload(ctx, size, filepath.Base(key))
	if err != nil {
		return "", err
	}
	p, err := bundles.UploadPath(ctx, up.ID)
	if err != nil {
		return "", err
	}
	if err := c.Get(ctx, key, p, func(n int64) {
		t.BytesDone = n
		t.Save(ctx, false)
	}); err != nil {
		bundles.DeleteUpload(ctx, up.ID)
		return "", v1api.Errorf(v1api.CodeRemoteFailed, err.Error())
	}
	if _, err := bundles.FinishUpload(ctx, up.ID); err != nil {
		return "", err
	}
	return up.ID, nil
}

// importBodyUpload is the single-request import for curl users: the body is
// the bundle (raw, options in the query) or multipart (fields name,
// dataset, gpu_ids, start, policy and a "file" part).
func importBodyUpload(w http.ResponseWriter, r *http.Request, mt string) {
	ctx := r.Context()
	q := r.URL.Query()
	fields := map[string]string{}
	for k := range q {
		fields[k] = q.Get(k)
	}
	var up *bundles.Upload
	var err error
	if mt == "multipart/form-data" {
		mr, merr := r.MultipartReader()
		if merr != nil {
			v1api.Fail(w, v1api.CodeInvalidRequest, "bad multipart body: "+merr.Error())
			return
		}
		for {
			part, perr := mr.NextPart()
			if perr == io.EOF {
				break
			}
			if perr != nil {
				err = v1api.Errorf(v1api.CodeInvalidRequest, "bad multipart body: "+perr.Error())
				break
			}
			if part.FormName() == "file" {
				if up != nil {
					part.Close()
					err = v1api.Errorf(v1api.CodeInvalidRequest, `only one "file" part`)
					break
				}
				up, err = stageBody(ctx, part, -1, part.FileName(), nil)
				part.Close()
				if err != nil {
					break
				}
				continue
			}
			b, _ := io.ReadAll(io.LimitReader(part, 1<<20))
			part.Close()
			fields[part.FormName()] = string(b)
		}
		if err == nil && up == nil {
			err = v1api.Errorf(v1api.CodeInvalidRequest, `multipart body has no "file" part`)
		}
	} else {
		name := q.Get("filename")
		if _, params, perr := mime.ParseMediaType(r.Header.Get("Content-Disposition")); perr == nil && params["filename"] != "" {
			name = params["filename"]
		}
		up, err = stageBody(ctx, r.Body, r.ContentLength, name, nil)
	}
	if err != nil {
		if up != nil {
			bundles.DeleteUpload(ctx, up.ID)
		}
		v1api.Respond(w, r, err)
		return
	}
	b := importBody{Name: fields["name"], Dataset: fields["dataset"], GpuIDs: fields["gpu_ids"]}
	if s := fields["policy"]; s != "" {
		b.Policy = json.RawMessage(s)
	}
	if b.Start, err = boolParam(fields["start"], false); err != nil {
		bundles.DeleteUpload(ctx, up.ID)
		v1api.Fail(w, v1api.CodeInvalidRequest, "start: "+err.Error())
		return
	}
	o, err := b.opts(ctx)
	if err != nil {
		bundles.DeleteUpload(ctx, up.ID)
		v1api.Respond(w, r, err)
		return
	}
	startUploadImport(w, r, up.ID, o, "upload")
}

type importDataset struct {
	plan  *bundles.PlanDataset
	name  string
	dir   string
	reuse bool
}

// preparedImport is a bundle checked against this server: names resolved,
// conflicts ruled out, nothing written yet.
type preparedImport struct {
	zr             *zip.ReadCloser
	plan           *bundles.Plan
	uploadID       string
	jobName        string
	datasets       []importDataset
	dsRoot, trRoot string
	closed         bool
}

// close releases the zip (before its upload is deleted: Windows cannot
// remove an open file). Safe to call twice.
func (p *preparedImport) close() {
	if !p.closed {
		p.closed = true
		p.zr.Close()
	}
}

func datasetExists(name string) error {
	return v1api.Errorf(v1api.CodeDatasetExists, fmt.Sprintf(
		"dataset %q already exists with different files; pass dataset to restore it under another name", name))
}

func prepareImport(ctx context.Context, uploadID string, o importOpts) (*preparedImport, error) {
	p, err := bundles.UploadPath(ctx, uploadID)
	if errors.Is(err, bundles.ErrNoUpload) {
		return nil, v1api.Errorf(v1api.CodeNotFound, "upload "+uploadID+" not found")
	} else if err != nil {
		return nil, err
	}
	return prepareZip(ctx, p, uploadID, o)
}

// prepareZip plans the import of the bundle at zipPath (uploadID is its
// session, removed after a successful import; "" for a local file).
func prepareZip(ctx context.Context, zipPath, uploadID string, o importOpts) (prep *preparedImport, err error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, v1api.Errorf(v1api.CodeInvalidRequest, "not a zip archive: "+err.Error())
	}
	defer func() {
		if err != nil {
			zr.Close()
		}
	}()
	plan, err := bundles.Read(&zr.Reader)
	if err != nil {
		return nil, err
	}
	prep = &preparedImport{zr: zr, plan: plan, uploadID: uploadID}
	if prep.dsRoot, err = settings.DatasetsRoot(ctx); err != nil {
		return nil, err
	}
	if prep.trRoot, err = settings.TrainingFolder(ctx); err != nil {
		return nil, err
	}
	prep.jobName = o.name
	if prep.jobName == "" {
		prep.jobName = plan.Manifest.Name
		if err := validJobName(prep.jobName); err != nil {
			return nil, v1api.Errorf(v1api.CodeInvalidRequest, fmt.Sprintf("the bundle's job name %q is not usable here (%v); pass name", prep.jobName, err))
		}
	}
	if j, err := store.FindFirstJob(ctx, store.JobQuery{Where: `"name" = ?`, Args: []any{prep.jobName}}); err != nil {
		return nil, err
	} else if j != nil {
		return nil, v1api.Errorf(v1api.CodeConflict, fmt.Sprintf("a job named %q already exists; pass name to import under another", prep.jobName))
	}
	if entries, err := os.ReadDir(filepath.Join(prep.trRoot, prep.jobName)); err == nil && len(entries) > 0 {
		return nil, v1api.Errorf(v1api.CodeConflict, fmt.Sprintf("the training folder already has a %s folder; pass name to import under another", prep.jobName))
	}
	seen := map[string]bool{}
	var dsBytes int64
	for i := range plan.Datasets {
		pd := &plan.Datasets[i]
		name := pd.Ref.Name
		if i == 0 && o.dataset != "" {
			name = o.dataset
		}
		if name == "" {
			name = prep.jobName
			if i > 0 {
				name = prep.jobName + "_" + strconv.Itoa(i)
			}
		}
		if err := validDatasetName(name); err != nil {
			return nil, v1api.Errorf(v1api.CodeInvalidRequest, fmt.Sprintf("dataset name %q: %v; pass dataset", name, err))
		}
		if seen[name] {
			return nil, v1api.Errorf(v1api.CodeInvalidRequest, "two of the bundle's datasets would both be named "+name)
		}
		seen[name] = true
		d := importDataset{plan: pd, name: name, dir: filepath.Join(prep.dsRoot, name)}
		if st, err := os.Stat(d.dir); err == nil && st.IsDir() {
			same, err := bundles.SameFiles(d.dir, pd.Files)
			if err != nil {
				return nil, err
			}
			if !same {
				return nil, datasetExists(name)
			}
			d.reuse = true
		} else {
			dsBytes += pd.Bytes
		}
		prep.datasets = append(prep.datasets, d)
	}
	for _, need := range []struct {
		dir   string
		bytes int64
	}{{prep.dsRoot, dsBytes}, {prep.trRoot, plan.JobBytes}} {
		if free, ok := freeAt(need.dir); ok && uint64(need.bytes) > free {
			return nil, v1api.Errorf(v1api.CodeTooLarge, fmt.Sprintf("restoring needs %d bytes in %s but only %d are free", need.bytes, need.dir, free))
		}
	}
	return prep, nil
}

// freeAt is the free space of the disk holding dir (or its nearest parent).
func freeAt(dir string) (uint64, bool) {
	probe := dir
	for {
		if _, err := os.Stat(probe); err == nil || filepath.Dir(probe) == probe {
			break
		}
		probe = filepath.Dir(probe)
	}
	free, _, err := diskSpace(probe)
	return free, err == nil
}

func dirInUse(dir string) bool {
	entries, err := os.ReadDir(dir)
	return err == nil && len(entries) > 0
}

// restoreImport writes the datasets and the job folder (each through a temp
// folder renamed into place) and creates the job. On failure everything it
// created is removed again.
func restoreImport(ctx context.Context, t *bundles.Task, prep *preparedImport, o importOpts) (err error) {
	plan, m := prep.plan, &prep.plan.Manifest
	t.State, t.Phase, t.BytesDone, t.Bytes = bundles.TaskRunning, "restoring", 0, plan.JobBytes
	for _, d := range prep.datasets {
		if !d.reuse {
			t.Bytes += d.plan.Bytes
		}
	}
	jobName := prep.jobName
	t.JobName = &jobName
	t.Skipped = plan.Skipped
	if len(t.Skipped) > 200 {
		t.Skipped = append(t.Skipped[:200:200], fmt.Sprintf("... and %d more", len(plan.Skipped)-200))
	}
	t.Warnings = append([]string{}, m.Warnings...)
	t.Save(ctx, true)
	progress := func(n int64) {
		t.BytesDone += n
		t.Save(ctx, false)
	}

	var created []string
	defer func() {
		if err != nil {
			for _, p := range created {
				os.RemoveAll(p)
			}
		}
	}()
	extractInto := func(root, dir, name string, files []bundles.File) error {
		if err := os.MkdirAll(root, 0o755); err != nil {
			return err
		}
		tmp, err := os.MkdirTemp(root, ".import-"+name+"-")
		if err != nil {
			return err
		}
		defer os.RemoveAll(tmp)
		if err := bundles.Extract(ctx, files, tmp, progress); err != nil {
			return err
		}
		if dir == filepath.Join(prep.trRoot, jobName) {
			config, err := localConfig(prep)
			if err != nil {
				return err
			}
			if err := portable.WriteJobConfig(filepath.Join(tmp, ".job_config.json"), config); err != nil {
				return err
			}
		}
		if dirInUse(dir) {
			if root == prep.dsRoot {
				return datasetExists(name)
			}
			return v1api.Errorf(v1api.CodeConflict, "the training folder already has a "+name+" folder")
		}
		os.Remove(dir) // an empty leftover folder
		if err := os.Rename(tmp, dir); err != nil {
			return err
		}
		created = append(created, dir)
		return nil
	}

	var newDatasets []string
	for _, d := range prep.datasets {
		if d.reuse {
			continue
		}
		if err := extractInto(prep.dsRoot, d.dir, d.name, d.plan.Files); err != nil {
			return err
		}
		newDatasets = append(newDatasets, d.name)
	}
	if len(prep.datasets) > 0 {
		first := prep.datasets[0].name
		t.Dataset = &first
	}

	config, err := localConfig(prep)
	if err != nil {
		return err
	}
	files := make([]bundles.File, len(plan.JobFiles))
	for i, f := range plan.JobFiles {
		files[i] = bundles.File{Zip: f.Zip, Rel: bundles.RenameCheckpoint(f.Rel, m.Name, jobName)}
	}
	if err := extractInto(prep.trRoot, filepath.Join(prep.trRoot, jobName), jobName, files); err != nil {
		return err
	}

	gpu := o.gpu
	if gpu == "" {
		gpu = "0"
	}
	if runtime.GOOS == "darwin" {
		gpu = "mps" // like POST /v1/jobs
	}
	jobType := m.Type
	if jobType == "" {
		jobType = "train"
	}
	maxPos, err := store.MaxQueuePosition(ctx)
	if err != nil {
		return err
	}
	job, err := store.CreateJob(ctx, store.NewJob{Name: jobName, GpuIDs: gpu, JobConfig: config,
		QueuePosition: maxPos + 1000, JobType: &jobType})
	if errors.Is(err, store.ErrUniqueViolation) {
		return v1api.Errorf(v1api.CodeConflict, "a job named "+jobName+" already exists")
	}
	if err != nil {
		return err
	}
	jobID := job.ID
	t.JobID = &jobID
	total := m.TotalSteps
	if total == nil {
		_, _, total = bundles.Describe(json.RawMessage(config))
	}
	if _, err := store.UpdateJob(ctx, job.ID, store.Fields{"step": m.Step, "total_steps": total,
		"info": fmt.Sprintf("Imported from a bundle at step %d", m.Step)}); err != nil {
		return err
	}
	pol := state.Policy{AutoResume: m.Policy.AutoResume, MaxAutoResumes: m.Policy.MaxAutoResumes}
	if o.policy != nil {
		pol = *o.policy
	}
	if err := state.SetPolicy(ctx, job.ID, pol); err != nil {
		return err
	}
	state.ApplyDefaultPolicy(ctx, job.ID)
	for _, n := range newDatasets {
		events.Emit(ctx, events.DatasetChange, "", map[string]string{"name": n})
	}
	if o.start {
		startImported(t, job)
	}
	prep.close()
	if prep.uploadID != "" {
		if err := bundles.DeleteUpload(ctx, prep.uploadID); err == nil {
			t.UploadID = nil
		}
	}
	t.State, t.Phase, t.BytesDone = bundles.TaskDone, "", t.Bytes
	t.Save(ctx, true)
	return nil
}

// startImported queues an imported job (ai-toolkit resumes from its newest
// save); a failure leaves it imported, with a warning on the task.
func startImported(t *bundles.Task, job *store.Job) {
	if err := supervisor.Requeue(context.Background(), job, "Queued after import"); err != nil {
		log.Printf("bundle import %s: start: %v", job.Name, err)
		t.Warnings = append(t.Warnings, "imported but not started: "+err.Error())
	}
}

// localConfig is the bundle's job config for this server.
func localConfig(prep *preparedImport) (string, error) {
	dirs := map[string]string{}
	for _, d := range prep.datasets {
		if d.plan.Source != "" {
			dirs[d.plan.Source] = d.dir
		}
	}
	config, err := bundles.Localize(prep.plan.Manifest.Config, dirs, prep.dsRoot, prep.trRoot, prep.jobName)
	if err != nil {
		return "", v1api.Errorf(v1api.CodeInvalidConfig, "bundle config: "+err.Error())
	}
	return checkConfig(json.RawMessage(config))
}

// ---- upload sessions ----

func uploadError(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case errors.Is(err, bundles.ErrNoUpload):
		v1api.Fail(w, v1api.CodeNotFound, "upload not found (uploads expire a day after their last chunk)")
	default:
		v1api.Respond(w, r, err)
	}
}

func createBundleUpload(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Size     int64  `json:"size"`
		Filename string `json:"filename"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	if b.Size <= 0 {
		v1api.Fail(w, v1api.CodeInvalidRequest, "size (bytes, > 0) is required")
		return
	}
	if b.Size > bundles.MaxBytes {
		v1api.Fail(w, v1api.CodeTooLarge, "bundle is larger than 1 TiB")
		return
	}
	trRoot, err := settings.TrainingFolder(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if free, ok := freeAt(trRoot); ok && uint64(b.Size) > free {
		v1api.Fail(w, v1api.CodeTooLarge, fmt.Sprintf("the upload needs %d bytes but only %d are free", b.Size, free))
		return
	}
	u, err := bundles.CreateUpload(r.Context(), b.Size, b.Filename)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Created(w, u)
}

func listBundleUploads(w http.ResponseWriter, r *http.Request) {
	items, err := bundles.ListUploads(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func getBundleUpload(w http.ResponseWriter, r *http.Request) {
	u, err := bundles.GetUpload(r.Context(), r.PathValue("id"))
	if err != nil {
		uploadError(w, r, err)
		return
	}
	v1api.OK(w, u)
}

// putBundleChunk appends a chunk. Its offset is the Content-Range start (or
// ?offset=); re-sending a chunk at or before the bytes received is
// harmless, a gap answers 409 with the bytes received in the message.
func putBundleChunk(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("id")
	// read what an error left unread before answering: closing with unread
	// bytes resets the connection and the client never sees the 409 that
	// says where to resume
	defer io.Copy(io.Discard, io.LimitReader(r.Body, bundles.MaxChunk))
	u, err := bundles.GetUpload(ctx, id)
	if err != nil {
		uploadError(w, r, err)
		return
	}
	var offset int64
	if h := r.Header.Get("Content-Range"); h != "" {
		start, end, total, err := bundles.ParseContentRange(h)
		if err != nil {
			v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
			return
		}
		if total >= 0 && total != u.Size {
			v1api.Fail(w, v1api.CodeInvalidRequest, fmt.Sprintf("Content-Range total %d is not the upload's size %d", total, u.Size))
			return
		}
		if end >= u.Size {
			v1api.Fail(w, v1api.CodeInvalidRequest, "Content-Range runs past the upload's size")
			return
		}
		if end-start+1 > bundles.MaxChunk {
			v1api.Fail(w, v1api.CodeTooLarge, fmt.Sprintf("a chunk may carry at most %d bytes", bundles.MaxChunk))
			return
		}
		offset = start
	} else if s := r.URL.Query().Get("offset"); s != "" {
		if offset, err = strconv.ParseInt(s, 10, 64); err != nil || offset < 0 {
			v1api.Fail(w, v1api.CodeInvalidRequest, "offset must be a non-negative integer")
			return
		}
	} else {
		v1api.Fail(w, v1api.CodeInvalidRequest, "send Content-Range: bytes <start>-<end>/<size> (or ?offset=)")
		return
	}
	u, err = bundles.WriteChunk(ctx, id, offset, r.Body)
	switch {
	case errors.Is(err, bundles.ErrOffset):
		v1api.Fail(w, v1api.CodeConflict, fmt.Sprintf("chunk starts at %d but %d bytes are received; resume from %d", offset, u.Received, u.Received))
	case errors.Is(err, bundles.ErrOverflow):
		v1api.Fail(w, v1api.CodeTooLarge, "chunk runs past the upload's size or the chunk limit")
	case err != nil:
		uploadError(w, r, err)
	default:
		v1api.OK(w, u)
	}
}

func deleteBundleUpload(w http.ResponseWriter, r *http.Request) {
	if err := bundles.DeleteUpload(r.Context(), r.PathValue("id")); err != nil {
		uploadError(w, r, err)
		return
	}
	v1api.NoContent(w)
}
