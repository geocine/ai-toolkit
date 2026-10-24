package provision

import (
	"context"
	"testing"
	"time"
)

func TestWatchOneListPerProviderPerTick(t *testing.T) {
	e := newEnv(t)
	a := e.launch(RunPod, nil)
	b := e.launch(RunPod, nil)
	v := e.launch(Vast, nil)
	e.clouds[RunPod].running("pod1")
	e.clouds[RunPod].running("pod2")
	e.clouds[Vast].running("pod1")

	e.s.Tick(e.ctx)
	for k, want := range map[Kind]int{RunPod: 1, Vast: 1, QuickPod: 0} {
		if n := e.clouds[k].n("List"); n != want {
			t.Errorf("%s: %d List calls, want %d", k, n, want)
		}
		if n := e.clouds[k].n("Get"); n != 0 {
			t.Errorf("%s: %d Get calls", k, n)
		}
	}
	for _, id := range []string{a.ID, b.ID, v.ID} {
		if r := e.row(id); r.Status != StatusReady {
			t.Fatalf("%s: %s", id, r.Status)
		}
	}
	e.s.Tick(e.ctx) // nothing is due yet
	e.clock.Advance(11 * time.Second)
	e.s.Tick(e.ctx) // ready rows wait a minute
	if n := e.clouds[RunPod].n("List"); n != 1 {
		t.Fatalf("ready rows refreshed early: %d", n)
	}
	e.clock.Advance(50 * time.Second)
	e.s.Tick(e.ctx)
	if n := e.clouds[RunPod].n("List"); n != 2 {
		t.Fatalf("after a minute: %d", n)
	}
}

func TestWatchRetryAfterPause(t *testing.T) {
	e := newEnv(t)
	e.launch(Vast, nil)
	c := e.clouds[Vast]
	c.listErr = &Error{Provider: Vast, Class: ClassRateLimited, Status: 429, Message: "slow down", RetryAfter: 30 * time.Second}
	e.s.Tick(e.ctx)
	e.clock.Advance(11 * time.Second)
	e.s.Tick(e.ctx)
	e.clock.Advance(11 * time.Second)
	e.s.Tick(e.ctx)
	if n := c.n("List"); n != 1 {
		t.Fatalf("paused provider listed %d times", n)
	}
	c.listErr = nil
	e.clock.Advance(9 * time.Second)
	e.s.Tick(e.ctx)
	if n := c.n("List"); n != 2 {
		t.Fatalf("after Retry-After: %d", n)
	}

	// a rate limit on any call pauses the provider, not only on List
	c.offersErr = &Error{Provider: Vast, Class: ClassRateLimited, Status: 429, Message: "slow down", RetryAfter: 30 * time.Second}
	if _, err := e.s.Offers(e.ctx, Vast, OfferQuery{}); ClassOf(err) != ClassRateLimited {
		t.Fatalf("offers: %v", err)
	}
	e.clock.Advance(11 * time.Second)
	e.s.Tick(e.ctx)
	if n := c.n("List"); n != 2 {
		t.Fatalf("listed after an Offers rate limit: %d", n)
	}
	e.clock.Advance(20 * time.Second)
	e.s.Tick(e.ctx)
	if n := c.n("List"); n != 3 {
		t.Fatalf("after the Offers Retry-After: %d", n)
	}
}

func TestWatchConfirmsGoneWithOneGet(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[RunPod]
	m := e.launch(RunPod, nil)
	fresh := e.launch(RunPod, nil)
	c.running("pod1")
	c.remove("pod2") // not listed yet, never seen: no Get
	e.s.Tick(e.ctx)
	if n := c.n("Get"); n != 0 {
		t.Fatalf("Get for an unseen row: %d", n)
	}
	if r := e.row(fresh.ID); r.Status != StatusCreating {
		t.Fatalf("unseen row: %s", r.Status)
	}
	c.remove("pod1")
	e.clock.Advance(61 * time.Second)
	e.s.Tick(e.ctx)
	if n := c.n("Get"); n != 1 {
		t.Fatalf("confirming gets: %d", n)
	}
	if r := e.row(m.ID); r.Status != StatusTerminated || r.EndedAt == nil {
		t.Fatalf("gone row: %+v", r)
	}
	e.clock.Advance(10 * time.Minute)
	e.s.Tick(e.ctx)
	if n := c.n("Get"); n != 1 {
		t.Fatal("an ended row was refreshed")
	}
}

func TestWatchSkipsKindWithoutKey(t *testing.T) {
	e := newEnv(t)
	e.launch(Vast, nil)
	if err := e.s.RemoveKey(e.ctx, Vast); err != nil {
		t.Fatal(err)
	}
	e.s.Tick(e.ctx)
	if n := e.clouds[Vast].n("List"); n != 0 {
		t.Fatalf("listed without a key: %d", n)
	}
}

func TestWatchDue(t *testing.T) {
	s := &Service{}
	now := t0
	ended := now.UnixMilli()
	for _, c := range []struct {
		r     row
		after time.Duration
		due   bool
	}{
		{row{ID: "a", Status: StatusStarting}, 9 * time.Second, false},
		{row{ID: "b", Status: StatusStarting}, 10 * time.Second, true},
		{row{ID: "c", Status: StatusReady}, 30 * time.Second, false},
		{row{ID: "d", Status: StatusReady, Problem: ProblemOpenServer}, 10 * time.Second, true},
		{row{ID: "e", Status: StatusReady}, time.Minute, true},
		{row{ID: "f", Status: StatusStopped}, 4 * time.Minute, false},
		{row{ID: "g", Status: StatusStopped}, 5 * time.Minute, true},
		{row{ID: "h", Status: StatusFailed}, 10 * time.Second, true},
		{row{ID: "i", Status: StatusTerminated, EndedAt: &ended}, time.Hour, false},
	} {
		s.markRefreshed(c.r.ID, now)
		if got := s.due(&c.r, now.Add(c.after)); got != c.due {
			t.Errorf("%s %s after %s: due %v", c.r.ID, c.r.Status, c.after, got)
		}
	}
	if !s.due(&row{ID: "never", Status: StatusReady}, now) {
		t.Error("a row never refreshed is due")
	}
}

func TestWatchRunStops(t *testing.T) {
	e := newEnv(t)
	ctx, cancel := context.WithCancel(e.ctx)
	done := make(chan struct{})
	go func() { e.s.Run(ctx); close(done) }()
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Run did not stop")
	}
}
