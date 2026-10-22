package bundles

import (
	"context"
	"encoding/json"
	"log"
	"sort"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/state"
)

// Task kinds, states and the events they emit (data: Task).
const (
	KindImport = "import"
	KindPush   = "push"

	TaskQueued  = "queued"
	TaskRunning = "running"
	TaskDone    = "done"
	TaskFailed  = "failed"

	EventImport = "bundle.import"
	EventPush   = "bundle.push"
)

// TaskError is a failed task's v1 error ({code, message}).
type TaskError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Task is a bundle import or push running in the background. Phases:
// import: downloading -> reading -> restoring; push: building -> uploading.
// bytes/bytes_done are the current phase's (bytes 0 = unknown).
type Task struct {
	ID        string     `json:"id"`
	Kind      string     `json:"kind"`
	State     string     `json:"state"`
	Phase     string     `json:"phase"`
	Bytes     int64      `json:"bytes"`
	BytesDone int64      `json:"bytes_done"`
	JobID     *string    `json:"job_id"`
	JobName   *string    `json:"job_name"`
	Dataset   *string    `json:"dataset"` // import: first dataset restored or reused
	Source    string     `json:"source"`  // import: url | remote | upload | local; push: remote
	Ref       *string    `json:"ref"`     // import from AITK_BOOTSTRAP: the ref it came from
	Remote    *string    `json:"remote"`
	Key       *string    `json:"key"`       // remote object (import source or push target)
	URL       *string    `json:"url"`       // import: the source URL (no query); push: where the remote stored it
	UploadID  *string    `json:"upload_id"` // import: the staged file, kept on failure so a retry needs no new transfer
	Skipped   []string   `json:"skipped"`
	Warnings  []string   `json:"warnings"`
	Error     *TaskError `json:"error"`
	CreatedAt time.Time  `json:"created_at"`
	UpdatedAt time.Time  `json:"updated_at"`
}

const taskPrefix = "bundle.task."

var (
	tasksMu  sync.Mutex
	lastSave = map[string]time.Time{}
)

// NewTask creates and stores a queued task.
func NewTask(ctx context.Context, kind string) *Task {
	prefix := "bimp"
	if kind == KindPush {
		prefix = "bpsh"
	}
	now := time.Now().UTC().Truncate(time.Millisecond)
	t := &Task{ID: state.NewID(prefix), Kind: kind, State: TaskQueued, Skipped: []string{}, Warnings: []string{},
		CreatedAt: now, UpdatedAt: now}
	t.Save(ctx, true)
	return t
}

// Save stores the task and emits its event; progress-only saves (force
// false) are throttled to one a second.
func (t *Task) Save(ctx context.Context, force bool) {
	tasksMu.Lock()
	if !force && time.Since(lastSave[t.ID]) < time.Second {
		tasksMu.Unlock()
		return
	}
	lastSave[t.ID] = time.Now()
	if t.State == TaskDone || t.State == TaskFailed {
		delete(lastSave, t.ID)
	}
	t.UpdatedAt = time.Now().UTC().Truncate(time.Millisecond)
	b, _ := json.Marshal(t)
	tasksMu.Unlock()
	ctx = context.WithoutCancel(ctx)
	if err := state.SetSetting(ctx, taskPrefix+t.ID, string(b)); err != nil {
		log.Printf("bundles: save task %s: %v", t.ID, err)
	}
	typ := EventImport
	if t.Kind == KindPush {
		typ = EventPush
	}
	jobID := ""
	if t.JobID != nil {
		jobID = *t.JobID
	}
	events.Emit(ctx, typ, jobID, t)
}

// Fail ends the task with a v1 error.
func (t *Task) Fail(ctx context.Context, code, msg string) {
	t.State, t.Error = TaskFailed, &TaskError{Code: code, Message: msg}
	t.Save(ctx, true)
}

// GetTask reads a task of kind ("" for any); nil when there is none.
func GetTask(ctx context.Context, kind, id string) (*Task, error) {
	v, ok, err := state.Setting(ctx, taskPrefix+id)
	if err != nil || !ok {
		return nil, err
	}
	var t Task
	if err := json.Unmarshal([]byte(v), &t); err != nil {
		return nil, err
	}
	if kind != "" && t.Kind != kind {
		return nil, nil
	}
	return &t, nil
}

// ListTasks returns tasks of kind, newest first.
func ListTasks(ctx context.Context, kind string) ([]*Task, error) {
	rows, err := state.DB.QueryContext(ctx, `SELECT value FROM settings WHERE key LIKE ? ORDER BY key`, taskPrefix+"%")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Task{}
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return nil, err
		}
		var t Task
		if json.Unmarshal([]byte(v), &t) == nil && (kind == "" || t.Kind == kind) {
			out = append(out, &t)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.After(out[j].CreatedAt) })
	return out, rows.Err()
}

// TaskRetention is how long finished tasks are listed.
const TaskRetention = 7 * 24 * time.Hour

// RecoverTasks fails tasks created before this process started (before)
// that never finished, and drops old finished ones (run once at start).
func RecoverTasks(ctx context.Context, before time.Time) {
	tasks, err := ListTasks(ctx, "")
	if err != nil {
		log.Printf("bundles: recover tasks: %v", err)
		return
	}
	for _, t := range tasks {
		switch {
		case (t.State == TaskQueued || t.State == TaskRunning) && t.CreatedAt.Before(before):
			t.Fail(ctx, "internal", "the server restarted while this ran; start it again")
		case time.Since(t.UpdatedAt) > TaskRetention:
			state.DB.ExecContext(ctx, `DELETE FROM settings WHERE key = ?`, taskPrefix+t.ID)
		}
	}
}
