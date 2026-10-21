package events

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
	"unicode/utf8"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Register serves GET /v1/events and GET /v1/events/stream.
func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /v1/events", list)
	httpx.Handle(mux, "GET /v1/events/stream", stream)
}

func parseQuery(r *http.Request) (after int64, f Filter, err error) {
	q := r.URL.Query()
	if s := q.Get("after"); s != "" {
		if after, err = strconv.ParseInt(s, 10, 64); err != nil || after < 0 {
			return 0, f, fmt.Errorf("after must be a non-negative integer")
		}
	}
	return after, Filter{JobID: q.Get("job_id"), Types: ParseTypes(q.Get("types"))}, nil
}

func list(w http.ResponseWriter, r *http.Request) {
	after, f, err := parseQuery(r)
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	limit := 500
	if s := r.URL.Query().Get("limit"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			limit = min(n, 5000)
		}
	}
	items, err := List(r.Context(), after, f, limit)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	next := ""
	if len(items) > 0 {
		next = strconv.FormatInt(items[len(items)-1].ID, 10)
	}
	v1api.ItemsPage(w, items, next)
}

const pingEvery = 15 * time.Second

// stream is Server-Sent Events: stored events after the cursor first, then
// live ones. Last-Event-ID (sent by EventSource on reconnect) wins over ?after.
func stream(w http.ResponseWriter, r *http.Request) {
	after, f, err := parseQuery(r)
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	if s := r.Header.Get("Last-Event-ID"); s != "" {
		if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			after = n
		}
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		v1api.Fail(w, v1api.CodeInternal, "streaming unsupported")
		return
	}
	ctx := r.Context()

	// subscribe before replaying so nothing falls between the two
	live, cancel := Subscribe(f)
	defer cancel()

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // nginx and some proxies buffer otherwise
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, ": connected\n\n")
	flusher.Flush()

	last := after
	for {
		batch, err := List(ctx, last, f, 1000)
		if err != nil {
			return
		}
		for _, e := range batch {
			if writeEvent(w, e) != nil {
				return
			}
			last = e.ID
		}
		flusher.Flush()
		if len(batch) < 1000 {
			break
		}
	}

	if f.JobID != "" && (len(f.Types) == 0 || f.Types[JobLog]) {
		go tailLog(ctx, f.JobID)
	}

	ping := time.NewTicker(pingEvery)
	defer ping.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case e := <-live:
			if e.ID != 0 && e.ID <= last {
				continue // already replayed
			}
			if writeEvent(w, e) != nil {
				return
			}
			if e.ID != 0 {
				last = e.ID
			}
			flusher.Flush()
		case <-ping.C:
			if _, err := fmt.Fprint(w, ": ping\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func writeEvent(w io.Writer, e Event) error {
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	if e.ID != 0 {
		if _, err := fmt.Fprintf(w, "id: %d\n", e.ID); err != nil {
			return err
		}
	}
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e.Type, b)
	return err
}

const (
	logPoll     = 500 * time.Millisecond
	logMaxChunk = 64 << 10
)

// tailLog publishes job.log events for new bytes in the job's log.txt while
// the stream is open (from the current end; history is GET /v1/jobs/{id}/log).
// Every stream of the job receives them, which is harmless: they are live-only.
func tailLog(ctx context.Context, jobID string) {
	job, err := store.GetJob(ctx, jobID)
	if err != nil || job == nil {
		return
	}
	root, err := settings.TrainingFolder(ctx)
	if err != nil {
		return
	}
	path := filepath.Join(root, job.Name, "log.txt")
	var offset int64 = -1
	t := time.NewTicker(logPoll)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		st, err := os.Stat(path)
		if err != nil {
			continue
		}
		size := st.Size()
		if offset < 0 || size < offset { // first look, or rotated/truncated
			offset = size
			continue
		}
		if size == offset {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		buf := make([]byte, min(size-offset, logMaxChunk))
		n, _ := f.ReadAt(buf, offset)
		f.Close()
		buf = buf[:n]
		// never split a UTF-8 character across events
		for len(buf) > 0 && !utf8.Valid(buf) && len(buf) > n-utf8.UTFMax {
			buf = buf[:len(buf)-1]
		}
		if len(buf) == 0 {
			continue
		}
		Emit(ctx, JobLog, jobID, map[string]any{"offset": offset, "text": string(buf)})
		offset += int64(len(buf))
	}
}
