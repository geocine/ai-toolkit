// Package events is the append-only event log clients follow with a cursor
// (docs/headless.md, Events): stored in the fork database, fanned out live to
// subscribers, served as JSON pages and as a Server-Sent Events stream.
package events

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
)

// Event types.
const (
	JobCreated    = "job.created"
	JobDeleted    = "job.deleted"
	JobState      = "job.state"
	JobProgress   = "job.progress"
	JobLoss       = "job.loss"
	JobCheckpoint = "job.checkpoint"
	JobSample     = "job.sample"
	JobLog        = "job.log" // streamed only, never stored
	JobAttempt    = "job.attempt"
	UploadState   = "upload.state"
	DatasetChange = "dataset.changed"
)

// Event is one entry of the log.
type Event struct {
	ID    int64           `json:"id"`
	Type  string          `json:"type"`
	JobID *string         `json:"job_id"`
	At    time.Time       `json:"at"`
	Data  json.RawMessage `json:"data"`
}

type subscriber struct {
	ch     chan Event
	filter Filter
}

var (
	subMu sync.Mutex
	subs  = map[*subscriber]struct{}{}
)

// Filter selects events; zero values match everything.
type Filter struct {
	JobID string
	Types map[string]bool
}

func (f Filter) match(e Event) bool {
	if f.JobID != "" && (e.JobID == nil || *e.JobID != f.JobID) {
		return false
	}
	return len(f.Types) == 0 || f.Types[e.Type]
}

// ParseTypes turns "a,b" into a set (nil for "").
func ParseTypes(s string) map[string]bool {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	m := map[string]bool{}
	for _, t := range strings.Split(s, ",") {
		if t = strings.TrimSpace(t); t != "" {
			m[t] = true
		}
	}
	return m
}

// Emit stores an event and delivers it to live subscribers. jobID may be "".
// Failures are logged, never returned: events must not break the action
// that produced them.
func Emit(ctx context.Context, typ, jobID string, data any) {
	raw, err := json.Marshal(data)
	if err != nil {
		log.Printf("events: marshal %s: %v", typ, err)
		return
	}
	e := Event{Type: typ, At: time.Now().UTC().Truncate(time.Millisecond), Data: raw}
	if jobID != "" {
		e.JobID = &jobID
	}
	if typ != JobLog && state.DB != nil {
		res, err := state.DB.ExecContext(ctx, `INSERT INTO events (type, job_id, at, data) VALUES (?, ?, ?, ?)`,
			typ, e.JobID, e.At.UnixMilli(), string(raw))
		if err != nil {
			log.Printf("events: store %s: %v", typ, err)
			return
		}
		e.ID, _ = res.LastInsertId()
	}
	publish(e)
}

func publish(e Event) {
	subMu.Lock()
	defer subMu.Unlock()
	for s := range subs {
		if !s.filter.match(e) {
			continue
		}
		select {
		case s.ch <- e:
		default:
			// a stalled client drops live events; it catches up from the
			// store by cursor when it reconnects
		}
	}
}

// Subscribe returns a channel of live events matching f; call cancel to stop.
func Subscribe(f Filter) (<-chan Event, func()) {
	s := &subscriber{ch: make(chan Event, 256), filter: f}
	subMu.Lock()
	subs[s] = struct{}{}
	subMu.Unlock()
	return s.ch, func() {
		subMu.Lock()
		delete(subs, s)
		subMu.Unlock()
	}
}

// List returns stored events with id > after, oldest first.
func List(ctx context.Context, after int64, f Filter, limit int) ([]Event, error) {
	q := `SELECT id, type, job_id, at, data FROM events WHERE id > ?`
	args := []any{after}
	if f.JobID != "" {
		q += ` AND job_id = ?`
		args = append(args, f.JobID)
	}
	if len(f.Types) > 0 {
		ph := make([]string, 0, len(f.Types))
		for t := range f.Types {
			ph = append(ph, "?")
			args = append(args, t)
		}
		q += ` AND type IN (` + strings.Join(ph, ", ") + `)`
	}
	q += ` ORDER BY id LIMIT ?`
	args = append(args, limit)
	rows, err := state.DB.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Event{}
	for rows.Next() {
		var e Event
		var jobID sql.NullString
		var at int64
		var data string
		if err := rows.Scan(&e.ID, &e.Type, &jobID, &at, &data); err != nil {
			return nil, err
		}
		if jobID.Valid {
			e.JobID = &jobID.String
		}
		e.At, e.Data = state.Time(at), json.RawMessage(data)
		out = append(out, e)
	}
	return out, rows.Err()
}

// Retention is how long stored events are kept (AI_TOOLKIT_EVENT_RETENTION,
// a Go duration, default 14 days).
func Retention() time.Duration {
	if d, err := time.ParseDuration(os.Getenv("AI_TOOLKIT_EVENT_RETENTION")); err == nil && d > 0 {
		return d
	}
	return 14 * 24 * time.Hour
}

// Prune deletes events older than the retention, hourly, until ctx ends.
func Prune(ctx context.Context) {
	t := time.NewTicker(time.Hour)
	defer t.Stop()
	for {
		cutoff := time.Now().Add(-Retention()).UnixMilli()
		if _, err := state.DB.ExecContext(ctx, `DELETE FROM events WHERE at < ?`, cutoff); err != nil && ctx.Err() == nil {
			log.Printf("events: prune: %v", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}
