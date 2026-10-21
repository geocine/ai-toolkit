// Package uploads is the queue that copies files to remotes (the `uploads`
// table of the fork database): explicit uploads from /v1/uploads and
// automatic ones for every new checkpoint of a job whose policy names an
// auto_upload_remote. See docs/headless.md, "Remote uploads".
package uploads

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/bg"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/safepath"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// Upload states.
const (
	Queued  = "queued"
	Running = "running"
	Done    = "done"
	Failed  = "failed"
)

// Upload is one file copied to a remote.
type Upload struct {
	ID        string    `json:"id"`
	RemoteID  string    `json:"remote_id"`
	JobID     *string   `json:"job_id"`
	Root      string    `json:"root"`
	Path      string    `json:"path"`
	URL       *string   `json:"url"`
	State     string    `json:"state"`
	Bytes     int64     `json:"bytes"`
	BytesDone int64     `json:"bytes_done"`
	SHA256    *string   `json:"sha256"`
	Error     *string   `json:"error"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Roots are the folders uploads may read from.
var Roots = []string{"datasets", "training", "data", "models"}

// RootDir resolves a root name to its folder.
func RootDir(ctx context.Context, root string) (string, error) {
	switch root {
	case "datasets":
		return settings.DatasetsRoot(ctx)
	case "training":
		return settings.TrainingFolder(ctx)
	case "data":
		return settings.DataRoot(ctx)
	case "models":
		p, err := settings.ModelsPath(ctx)
		if err != nil {
			return "", err
		}
		return filepath.Abs(p)
	}
	return "", fmt.Errorf("unknown root %q (one of %s)", root, strings.Join(Roots, ", "))
}

// ErrInvalid marks a request problem (bad root, path outside it, not a file).
var ErrInvalid = errors.New("invalid upload")

// LocalPath is the file an upload reads, checked to stay inside its root.
func LocalPath(ctx context.Context, root, rel string) (string, error) {
	dir, err := RootDir(ctx, root)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	if strings.TrimSpace(rel) == "" {
		return "", fmt.Errorf("%w: path is required", ErrInvalid)
	}
	p := safepath.Resolve(filepath.Join(dir, filepath.FromSlash(rel)), []string{dir})
	if p == "" || p == filepath.Clean(dir) {
		return "", fmt.Errorf("%w: path %q is outside the %s root", ErrInvalid, rel, root)
	}
	return p, nil
}

// remoteKey is where an upload lands: job files under jobs/<job name>/,
// anything else under <root>/<path>.
func remoteKey(u *Upload) string {
	rel := strings.TrimLeft(filepath.ToSlash(u.Path), "/")
	if u.JobID != nil && u.Root == "training" {
		return "jobs/" + rel
	}
	return u.Root + "/" + rel
}

const cols = `id, remote_id, job_id, root, path, url, state, bytes, bytes_done, sha256, error, created_at, updated_at`

func scan(r interface{ Scan(...any) error }) (*Upload, error) {
	var u Upload
	var jobID, url, sha, errText sql.NullString
	var created, updated int64
	if err := r.Scan(&u.ID, &u.RemoteID, &jobID, &u.Root, &u.Path, &url, &u.State, &u.Bytes, &u.BytesDone,
		&sha, &errText, &created, &updated); err != nil {
		return nil, err
	}
	str := func(n sql.NullString) *string {
		if !n.Valid {
			return nil
		}
		v := n.String
		return &v
	}
	u.JobID, u.URL, u.SHA256, u.Error = str(jobID), str(url), str(sha), str(errText)
	u.CreatedAt, u.UpdatedAt = state.Time(created), state.Time(updated)
	return &u, nil
}

// Get returns an upload, or nil.
func Get(ctx context.Context, id string) (*Upload, error) {
	u, err := scan(state.DB.QueryRowContext(ctx, `SELECT `+cols+` FROM uploads WHERE id = ?`, id))
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return u, err
}

// List returns uploads newest first, optionally for one job and/or state.
func List(ctx context.Context, jobID, st string) ([]*Upload, error) {
	q, args := `SELECT `+cols+` FROM uploads WHERE 1 = 1`, []any{}
	if jobID != "" {
		q += ` AND job_id = ?`
		args = append(args, jobID)
	}
	if st != "" {
		q += ` AND state = ?`
		args = append(args, st)
	}
	rows, err := state.DB.QueryContext(ctx, q+` ORDER BY created_at DESC, id DESC LIMIT 1000`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Upload{}
	for rows.Next() {
		u, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, u)
	}
	return out, rows.Err()
}

func emit(ctx context.Context, u *Upload) {
	jobID := ""
	if u.JobID != nil {
		jobID = *u.JobID
	}
	events.Emit(ctx, events.UploadState, jobID, u)
}

// Enqueue queues root/rel for upload to the remote (id or name). An upload of
// the same file to the same remote that is already queued or running is
// returned instead of a duplicate.
func Enqueue(ctx context.Context, remoteRef string, jobID *string, root, rel string) (*Upload, error) {
	r, err := remote.Get(ctx, remoteRef)
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, fmt.Errorf("remote %q: %w", remoteRef, remote.ErrNotFound)
	}
	local, err := LocalPath(ctx, root, rel)
	if err != nil {
		return nil, err
	}
	st, err := os.Stat(local)
	if err != nil || !st.Mode().IsRegular() {
		return nil, fmt.Errorf("%w: %s/%s is not a file", ErrInvalid, root, rel)
	}
	rel = filepath.ToSlash(rel)

	var existing string
	err = state.DB.QueryRowContext(ctx, `SELECT id FROM uploads WHERE remote_id = ? AND root = ? AND path = ?
		AND state IN ('queued', 'running') LIMIT 1`, r.ID, root, rel).Scan(&existing)
	if err == nil {
		return Get(ctx, existing)
	} else if err != sql.ErrNoRows {
		return nil, err
	}

	now := state.NowMs()
	u := &Upload{ID: state.NewID("upl"), RemoteID: r.ID, JobID: jobID, Root: root, Path: rel, State: Queued,
		Bytes: st.Size(), CreatedAt: state.Time(now), UpdatedAt: state.Time(now)}
	if _, err := state.DB.ExecContext(ctx, `INSERT INTO uploads (id, remote_id, job_id, root, path, state, bytes,
		bytes_done, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`,
		u.ID, u.RemoteID, u.JobID, u.Root, u.Path, u.State, u.Bytes, now, now); err != nil {
		return nil, err
	}
	emit(ctx, u)
	wake()
	return u, nil
}

// ErrBusy is returned when retrying an upload that is still queued or running.
var ErrBusy = errors.New("upload is still queued or running")

// Retry puts a finished upload back in the queue.
func Retry(ctx context.Context, id string) (*Upload, error) {
	res, err := state.DB.ExecContext(ctx, `UPDATE uploads SET state = 'queued', error = NULL, bytes_done = 0,
		updated_at = ? WHERE id = ? AND state IN ('failed', 'done')`, state.NowMs(), id)
	if err != nil {
		return nil, err
	}
	u, err := Get(ctx, id)
	if err != nil || u == nil {
		return u, err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return u, ErrBusy
	}
	emit(ctx, u)
	wake()
	return u, nil
}

// Concurrency is how many uploads run at once.
const Concurrency = 2

var (
	wakeCh   = make(chan struct{}, 1)
	initOnce sync.Once
	// progressEvery throttles bytes_done writes and upload.state events
	progressEvery = time.Second
)

func wake() {
	select {
	case wakeCh <- struct{}{}:
	default:
	}
}

// Init registers the upload worker and the auto-upload listener (idempotent).
func Init() {
	initOnce.Do(func() {
		bg.Go("uploads", Run)
		bg.Go("uploads-auto", AutoUpload)
	})
}

// Run is the worker: uploads left running by a previous process are queued
// again, then queued uploads are processed Concurrency at a time.
func Run(ctx context.Context) {
	if _, err := state.DB.ExecContext(ctx, `UPDATE uploads SET state = 'queued', bytes_done = 0, updated_at = ?
		WHERE state = 'running'`, state.NowMs()); err != nil {
		log.Printf("uploads: requeue: %v", err)
	}
	sem := make(chan struct{}, Concurrency)
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		for {
			// take a slot first, so nothing is marked running while it waits
			acquired := false
			select {
			case sem <- struct{}{}:
				acquired = true
			default:
			}
			if !acquired {
				break // no free slot; a finishing upload wakes the loop
			}
			u, err := claim(ctx)
			if err != nil || u == nil {
				<-sem
				if err != nil && ctx.Err() == nil {
					log.Printf("uploads: claim: %v", err)
				}
				break
			}
			go func() {
				defer func() { <-sem; wake() }()
				process(ctx, u)
			}()
		}
		select {
		case <-ctx.Done():
			return
		case <-wakeCh:
		case <-t.C:
		}
	}
}

var claimMu sync.Mutex

// claim marks the oldest queued upload running, when a worker slot is free
// (the caller blocks on the semaphore before processing).
func claim(ctx context.Context) (*Upload, error) {
	claimMu.Lock()
	defer claimMu.Unlock()
	var id string
	err := state.DB.QueryRowContext(ctx, `SELECT id FROM uploads WHERE state = 'queued' ORDER BY created_at, id LIMIT 1`).Scan(&id)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if _, err := state.DB.ExecContext(ctx, `UPDATE uploads SET state = 'running', updated_at = ? WHERE id = ? AND state = 'queued'`,
		state.NowMs(), id); err != nil {
		return nil, err
	}
	return Get(ctx, id)
}

func process(ctx context.Context, u *Upload) {
	fail := func(err error) {
		msg := err.Error()
		u.State, u.Error, u.UpdatedAt = Failed, &msg, time.Now().UTC()
		if _, dbErr := state.DB.ExecContext(context.WithoutCancel(ctx), `UPDATE uploads SET state = 'failed', error = ?, updated_at = ?
			WHERE id = ?`, msg, state.NowMs(), u.ID); dbErr != nil {
			log.Printf("uploads: %s: %v", u.ID, dbErr)
		}
		log.Printf("uploads: %s %s/%s failed: %s", u.ID, u.Root, u.Path, msg)
		emit(ctx, u)
	}
	emit(ctx, u)

	local, err := LocalPath(ctx, u.Root, u.Path)
	if err != nil {
		fail(err)
		return
	}
	st, err := os.Stat(local)
	if err != nil {
		fail(fmt.Errorf("local file: %w", err))
		return
	}
	sum, err := artifacts.SHA256File(local)
	if err != nil {
		fail(fmt.Errorf("hash: %w", err))
		return
	}
	u.Bytes, u.SHA256 = st.Size(), &sum
	state.DB.ExecContext(ctx, `UPDATE uploads SET bytes = ?, sha256 = ?, updated_at = ? WHERE id = ?`,
		u.Bytes, sum, state.NowMs(), u.ID)

	_, client, err := remote.Open(ctx, u.RemoteID)
	if err != nil {
		fail(err)
		return
	}
	var last time.Time
	var mu sync.Mutex
	progress := func(done int64) {
		mu.Lock()
		defer mu.Unlock()
		if time.Since(last) < progressEvery {
			return
		}
		last = time.Now()
		u.BytesDone, u.UpdatedAt = min(done, u.Bytes), last.UTC()
		state.DB.ExecContext(ctx, `UPDATE uploads SET bytes_done = ?, updated_at = ? WHERE id = ?`, u.BytesDone, state.NowMs(), u.ID)
		emit(ctx, u)
	}
	url, err := client.Put(ctx, remoteKey(u), local, progress)
	if err != nil {
		fail(err)
		return
	}
	mu.Lock()
	defer mu.Unlock()
	u.State, u.URL, u.BytesDone, u.Error, u.UpdatedAt = Done, &url, u.Bytes, nil, time.Now().UTC()
	if _, err := state.DB.ExecContext(ctx, `UPDATE uploads SET state = 'done', url = ?, bytes_done = bytes, error = NULL,
		updated_at = ? WHERE id = ?`, url, state.NowMs(), u.ID); err != nil {
		log.Printf("uploads: %s: %v", u.ID, err)
	}
	emit(ctx, u)
}

// AutoUpload enqueues each new checkpoint of a job whose effective policy has
// an auto_upload_remote.
func AutoUpload(ctx context.Context) {
	ch, cancel := events.Subscribe(events.Filter{Types: map[string]bool{events.JobCheckpoint: true}})
	defer cancel()
	for {
		select {
		case <-ctx.Done():
			return
		case e := <-ch:
			if e.JobID != nil {
				autoUploadOne(ctx, *e.JobID, e.Data)
			}
		}
	}
}

func autoUploadOne(ctx context.Context, jobID string, data json.RawMessage) {
	_, _, remoteRef, err := state.EffectivePolicy(ctx, jobID)
	if err != nil || remoteRef == "" {
		return
	}
	var cp artifacts.Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil || cp.Name == "" {
		return
	}
	job, err := store.GetJob(ctx, jobID)
	if err != nil || job == nil {
		return
	}
	id := jobID
	if _, err := Enqueue(ctx, remoteRef, &id, "training", job.Name+"/"+cp.Name); err != nil {
		log.Printf("uploads: auto-upload %s of %s to %s: %v", cp.Name, job.Name, remoteRef, err)
	}
}
