package notify

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// PublicURL derives the server's address from each provider's env, in order.
func TestPublicURLFromProvider(t *testing.T) {
	for _, k := range []string{"PORT", "RUNPOD_POD_ID", "PUBLIC_IPADDR", "VAST_TCP_PORT_8675", "QUICKPOD_PORT_8675", "QUICKPOD_PORT_9000"} {
		t.Setenv(k, "")
	}
	if got := PublicURL(Config{}); got != "" {
		t.Fatalf("no provider env: %q", got)
	}
	if got := PublicURL(Config{PublicURL: "https://srv.example/"}); got != "https://srv.example" {
		t.Fatalf("configured: %q", got)
	}

	// QuickPod: the host IP and the external port mapped to 8675
	t.Setenv("PUBLIC_IPADDR", "203.0.113.7")
	if got := PublicURL(Config{}); got != "" {
		t.Fatalf("an IP alone: %q", got)
	}
	t.Setenv("QUICKPOD_PORT_8675", "40123")
	if got := PublicURL(Config{}); got != "http://203.0.113.7:40123" {
		t.Fatalf("quickpod: %q", got)
	}
	// it follows PORT like Vast.ai does
	t.Setenv("PORT", "9000")
	if got := PublicURL(Config{}); got != "" {
		t.Fatalf("quickpod with PORT=9000 and no mapping: %q", got)
	}
	t.Setenv("QUICKPOD_PORT_9000", "40200")
	if got := PublicURL(Config{}); got != "http://203.0.113.7:40200" {
		t.Fatalf("quickpod with PORT=9000: %q", got)
	}
	t.Setenv("PORT", "")

	// Vast.ai comes first when both map the port, RunPod before either
	t.Setenv("VAST_TCP_PORT_8675", "31000")
	if got := PublicURL(Config{}); got != "http://203.0.113.7:31000" {
		t.Fatalf("vast: %q", got)
	}
	t.Setenv("RUNPOD_POD_ID", "abc123")
	if got := PublicURL(Config{}); got != "https://abc123-8675.proxy.runpod.net" {
		t.Fatalf("runpod: %q", got)
	}
	if got := PublicURL(Config{PublicURL: "https://srv.example"}); got != "https://srv.example" {
		t.Fatalf("configured wins: %q", got)
	}
}

// A face check job sends no started, failed or completed message: its log
// tail names photos and it never ran on a GPU.
func TestComposeSkipsFaceJobs(t *testing.T) {
	setupDB(t)
	ctx := context.Background()
	srv, all := receiver(t, nil)
	cfgMu.Lock()
	cfg, loaded = Config{Sinks: []Sink{{ID: "w", URL: "webhook+" + srv.URL + "/w"}}, PublicURL: "https://srv.example"}, true
	cfgMu.Unlock()
	runCtx = ctx
	typ := photocheck.FaceJobType
	j, err := store.CreateJob(ctx, store.NewJob{Name: "faces", GpuIDs: "cpu", JobConfig: "{}", JobType: &typ})
	if err != nil {
		t.Fatal(err)
	}
	state.StartAttempt(ctx, j.ID, 0, false)
	c := newComposer()
	ev := func(typ string, data any) events.Event {
		b, _ := json.Marshal(data)
		id := j.ID
		return events.Event{Type: typ, JobID: &id, Data: b, At: time.Now()}
	}
	c.handle(ctx, ev(events.JobAttempt, state.Attempt{N: 1}))
	c.handle(ctx, ev(events.JobState, map[string]any{"to": jobstate.Failed, "info": "Error: me.jpg"}))
	c.handle(ctx, ev(events.JobState, map[string]any{"to": jobstate.Completed}))
	fctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	Flush(fctx)
	if n := len(all()); n != 0 {
		t.Fatalf("%d messages for a face check job", n)
	}
}
