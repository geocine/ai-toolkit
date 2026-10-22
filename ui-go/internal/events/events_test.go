package events

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func setup(t *testing.T) *httptest.Server {
	t.Helper()
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close() })
	mux := http.NewServeMux()
	Register(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestListWithCursorAndFilters(t *testing.T) {
	srv := setup(t)
	ctx := context.Background()
	Emit(ctx, JobState, "j1", map[string]string{"to": "running"})
	Emit(ctx, JobProgress, "j1", map[string]int{"step": 1})
	Emit(ctx, JobState, "j2", map[string]string{"to": "queued"})
	Emit(ctx, JobLog, "j1", map[string]string{"text": "not stored"})

	get := func(q string) (items []Event, next *string) {
		resp, err := http.Get(srv.URL + "/v1/events" + q)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var body struct {
			Items      []Event `json:"items"`
			NextCursor *string `json:"next_cursor"`
		}
		json.NewDecoder(resp.Body).Decode(&body)
		return body.Items, body.NextCursor
	}
	all, next := get("")
	if len(all) != 3 || next == nil || *next != "3" {
		t.Fatalf("all: %d next %v", len(all), next)
	}
	after, _ := get("?after=1")
	if len(after) != 2 || after[0].ID != 2 {
		t.Fatalf("after=1: %+v", after)
	}
	j1, _ := get("?job_id=j1&types=job.state")
	if len(j1) != 1 || j1[0].Type != JobState || *j1[0].JobID != "j1" {
		t.Fatalf("filtered: %+v", j1)
	}
	if resp, _ := http.Get(srv.URL + "/v1/events?after=x"); resp.StatusCode != 400 {
		t.Fatalf("bad cursor status %d", resp.StatusCode)
	}
	if now, _ := get("?after=now"); len(now) != 0 {
		t.Fatalf("after=now must skip stored events, got %d", len(now))
	}
}

func TestStreamReplaysThenFollows(t *testing.T) {
	srv := setup(t)
	ctx := context.Background()
	Emit(ctx, JobState, "j1", map[string]string{"to": "a"}) // id 1
	Emit(ctx, JobState, "j1", map[string]string{"to": "b"}) // id 2
	Emit(ctx, JobState, "j2", map[string]string{"to": "c"}) // id 3, other job

	req, _ := http.NewRequest("GET", srv.URL+"/v1/events/stream?job_id=j1&types=job.state&after=0", nil)
	req.Header.Set("Last-Event-ID", "1") // wins over ?after
	rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	resp, err := http.DefaultClient.Do(req.WithContext(rctx))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type %q", ct)
	}

	ids := make(chan string, 10)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			if line := sc.Text(); strings.HasPrefix(line, "id: ") {
				ids <- strings.TrimPrefix(line, "id: ")
			}
		}
	}()
	expect := func(want string) {
		select {
		case got := <-ids:
			if got != want {
				t.Fatalf("event id %s, want %s", got, want)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("timed out waiting for event %s", want)
		}
	}
	expect("2") // replayed after Last-Event-ID 1; id 3 is another job
	time.Sleep(100 * time.Millisecond)
	Emit(ctx, JobProgress, "j1", map[string]int{"step": 5}) // id 4, filtered out by type
	Emit(ctx, JobState, "j1", map[string]string{"to": "d"}) // id 5, live
	expect("5")
}

func TestLongPoll(t *testing.T) {
	srv := setup(t)
	ctx := context.Background()
	Emit(ctx, JobState, "j1", map[string]string{"to": "running"})
	get := func(q string) ([]Event, time.Duration) {
		start := time.Now()
		resp, err := http.Get(srv.URL + "/v1/events" + q)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var body struct {
			Items []Event `json:"items"`
		}
		json.NewDecoder(resp.Body).Decode(&body)
		return body.Items, time.Since(start)
	}
	// history answers at once
	if items, took := get("?after=0&wait_ms=5000"); len(items) != 1 || took > 2*time.Second {
		t.Fatalf("history: %d in %s", len(items), took)
	}
	// nothing newer: wait for the next stored event (live-only ones do not count)
	go func() {
		time.Sleep(200 * time.Millisecond)
		Emit(ctx, JobLog, "j1", map[string]string{"text": "x"})
		time.Sleep(200 * time.Millisecond)
		Emit(ctx, JobProgress, "j1", map[string]int{"step": 2})
	}()
	items, took := get("?after=1&wait_ms=5000")
	if len(items) != 1 || items[0].Type != JobProgress || took < 300*time.Millisecond || took > 4*time.Second {
		t.Fatalf("long poll: %+v in %s", items, took)
	}
	// and gives up after wait_ms
	if items, took := get("?after=2&wait_ms=300"); len(items) != 0 || took < 250*time.Millisecond {
		t.Fatalf("timeout: %d in %s", len(items), took)
	}
}

// Two open streams of one job each get every log chunk exactly once: each
// stream's tailer used to publish to all of them.
func TestStreamLogOncePerStream(t *testing.T) {
	srv := setup(t)
	dir := testutil.TempDir(t)
	db, err := sql.Open("sqlite", "file:"+filepath.ToSlash(filepath.Join(dir, "aitk_db.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if _, err := db.Exec(`
CREATE TABLE "Settings" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "key" TEXT NOT NULL, "value" TEXT NOT NULL);
CREATE TABLE "Job" (
    "id" TEXT NOT NULL PRIMARY KEY, "name" TEXT NOT NULL, "gpu_ids" TEXT NOT NULL, "job_config" TEXT NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "updated_at" DATETIME NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'stopped', "stop" BOOLEAN NOT NULL DEFAULT false,
    "return_to_queue" BOOLEAN NOT NULL DEFAULT false, "step" INTEGER NOT NULL DEFAULT 0, "total_steps" INTEGER,
    "info" TEXT NOT NULL DEFAULT '', "speed_string" TEXT NOT NULL DEFAULT '', "queue_position" INTEGER NOT NULL DEFAULT 0,
    "pid" INTEGER, "job_type" TEXT NOT NULL DEFAULT 'train', "job_ref" TEXT,
    "save_now" BOOLEAN NOT NULL DEFAULT false, "sample_now" BOOLEAN NOT NULL DEFAULT false);`); err != nil {
		t.Fatal(err)
	}
	prev := store.DB
	store.DB = db
	t.Cleanup(func() { store.DB = prev; settings.Flush() })
	training := filepath.Join(dir, "output")
	if _, err := db.Exec(`INSERT INTO "Settings" ("key","value") VALUES ('TRAINING_FOLDER', ?)`, training); err != nil {
		t.Fatal(err)
	}
	settings.Flush()
	job, err := store.CreateJob(t.Context(), store.NewJob{Name: "tail", GpuIDs: "0", JobConfig: "{}"})
	if err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(training, "tail", "log.txt")
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(logPath, []byte("start\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	rctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	open := func() chan string {
		req, _ := http.NewRequestWithContext(rctx, "GET", srv.URL+"/v1/events/stream?job_id="+job.ID+"&types=job.log", nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { resp.Body.Close() })
		texts := make(chan string, 10)
		go func() {
			sc := bufio.NewScanner(resp.Body)
			for sc.Scan() {
				if line, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
					var e struct {
						Data struct{ Text string } `json:"data"`
					}
					json.Unmarshal([]byte(line), &e)
					texts <- e.Data.Text
				}
			}
		}()
		return texts
	}
	a, b := open(), open()
	time.Sleep(3 * logPoll) // both tailers have taken their first look
	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("step 1\n")
	f.Close()

	for name, texts := range map[string]chan string{"a": a, "b": b} {
		var got []string
		deadline := time.After(4 * logPoll)
	collect:
		for {
			select {
			case s := <-texts:
				got = append(got, s)
			case <-deadline:
				break collect
			}
		}
		if len(got) != 1 || got[0] != "step 1\n" {
			t.Fatalf("stream %s got %q, want one \"step 1\n\"", name, got)
		}
	}
}
