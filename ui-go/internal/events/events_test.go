package events

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
)

func setup(t *testing.T) *httptest.Server {
	t.Helper()
	if err := state.Open(filepath.Join(t.TempDir(), "aitk_go.db")); err != nil {
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
