package vast

import (
	"context"
	"sync"
	"testing"
	"time"
)

// sleepClock is a fake Clock whose Sleep advances time at once.
type sleepClock struct {
	mu    sync.Mutex
	t     time.Time
	slept []time.Duration
}

func newSleepClock() *sleepClock { return &sleepClock{t: time.Unix(1_700_000_000, 0)} }

func (c *sleepClock) Now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }

func (c *sleepClock) Sleep(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.slept = append(c.slept, d)
	c.t = c.t.Add(d)
	return nil
}

func (c *sleepClock) advance(d time.Duration) { c.mu.Lock(); c.t = c.t.Add(d); c.mu.Unlock() }

func (c *sleepClock) sleeps() []time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]time.Duration(nil), c.slept...)
}

func TestLimiterSpacesShowCalls(t *testing.T) {
	c := newSleepClock()
	l := NewLimiter(c)
	ctx := context.Background()
	for range 2 {
		if err := l.Wait(ctx, EpShow); err != nil {
			t.Fatal(err)
		}
	}
	if s := c.sleeps(); len(s) != 1 || s[0] != 2*time.Second {
		t.Fatalf("second show should wait 2s, slept %v", s)
	}
	// another endpoint has its own interval
	if err := l.Wait(ctx, EpManage); err != nil || len(c.sleeps()) != 1 {
		t.Fatalf("manage after show must not wait: %v %v", err, c.sleeps())
	}
	// once the interval has passed, no wait
	c.advance(3 * time.Second)
	if err := l.Wait(ctx, EpShow); err != nil || len(c.sleeps()) != 1 {
		t.Fatalf("show after 3s must not wait: %v", c.sleeps())
	}
	// part of the interval passed: wait the rest
	c.advance(500 * time.Millisecond)
	_ = l.Wait(ctx, EpShow)
	if s := c.sleeps(); s[len(s)-1] != 1500*time.Millisecond {
		t.Fatalf("rest of the interval: %v", s)
	}
}

func TestLimiterIntervals(t *testing.T) {
	want := map[Endpoint]time.Duration{EpCreate: 4500 * time.Millisecond, EpDestroy: 3 * time.Second, EpShow: 2 * time.Second,
		EpList: time.Second, EpManage: time.Second, EpTemplates: 3 * time.Second}
	for ep, d := range want {
		c := newSleepClock()
		l := NewLimiter(c)
		_ = l.Wait(context.Background(), ep)
		_ = l.Wait(context.Background(), ep)
		if s := c.sleeps(); len(s) != 1 || s[0] != d {
			t.Errorf("%s: slept %v, want %v", ep, s, d)
		}
	}
	// the account call has no documented limit
	c := newSleepClock()
	l := NewLimiter(c)
	for range 5 {
		_ = l.Wait(context.Background(), EpUser)
	}
	if len(c.sleeps()) != 0 {
		t.Errorf("user calls waited: %v", c.sleeps())
	}
}

func TestLimiterSearchBucket(t *testing.T) {
	c := newSleepClock()
	l := NewLimiter(c)
	ctx := context.Background()
	for i := range 10 {
		c.advance(time.Second) // searches 1s apart: t+1 .. t+10
		if err := l.Wait(ctx, EpSearch); err != nil || len(c.sleeps()) != 0 {
			t.Fatalf("search %d waited: %v %v", i+1, err, c.sleeps())
		}
	}
	// the 11th within the minute waits until the first leaves the window
	if err := l.Wait(ctx, EpSearch); err != nil {
		t.Fatal(err)
	}
	if s := c.sleeps(); len(s) != 1 || s[0] != 51*time.Second {
		t.Fatalf("11th search slept %v, want 51s", s)
	}
}

func TestLimiterHonoursContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	l := NewLimiter(nil) // the wall clock: a cancelled wait returns at once
	if err := l.Wait(ctx, EpCreate); err != nil {
		t.Fatal(err)
	}
	cancel()
	start := time.Now()
	if err := l.Wait(ctx, EpCreate); err == nil {
		t.Fatal("a cancelled wait must fail")
	}
	if time.Since(start) > time.Second {
		t.Fatal("the cancelled wait blocked")
	}
}

// blockClock is a fake Clock whose Sleep blocks until the test releases it
// (or ctx ends), so a test can hold one waiter mid-sleep.
type blockClock struct {
	t       time.Time
	sleep   chan time.Duration // each Sleep reports its duration here first
	release chan struct{}
}

func newBlockClock() *blockClock {
	return &blockClock{t: time.Unix(1_700_000_000, 0), sleep: make(chan time.Duration, 16), release: make(chan struct{})}
}

func (c *blockClock) Now() time.Time { return c.t }

func (c *blockClock) Sleep(ctx context.Context, d time.Duration) error {
	c.sleep <- d
	select {
	case <-c.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// sleeping waits for the next waiter to enter Sleep and returns its duration.
func (c *blockClock) sleeping(t *testing.T) time.Duration {
	t.Helper()
	select {
	case d := <-c.sleep:
		return d
	case <-time.After(5 * time.Second):
		t.Fatal("no waiter reached Sleep")
		return 0
	}
}

// within runs fn and fails the test when it has not returned after a second.
func within(t *testing.T, what string, fn func() error) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		return err
	case <-time.After(time.Second):
		t.Fatalf("%s blocked behind another wait", what)
		return nil
	}
}

func TestLimiterSearchWaitDoesNotBlockOtherEndpoints(t *testing.T) {
	c := newBlockClock()
	l := NewLimiter(c)
	ctx := context.Background()
	for range 10 {
		if err := l.Wait(ctx, EpSearch); err != nil {
			t.Fatal(err)
		}
	}
	searched := make(chan error, 2)
	go func() { searched <- l.Wait(ctx, EpSearch) }()
	if d := c.sleeping(t); d != searchWindow {
		t.Fatalf("11th search sleeps %v, want %v", d, searchWindow)
	}
	// a second queued search books the slot after it, not the same one
	go func() { searched <- l.Wait(ctx, EpSearch) }()
	if d := c.sleeping(t); d != searchWindow {
		t.Fatalf("12th search sleeps %v, want %v (all ten earlier searches share one instant)", d, searchWindow)
	}
	if n := len(l.search); n != 12 {
		t.Fatalf("booked searches %d, want 12", n)
	}
	// while both searches sleep, a show call goes straight out
	if err := within(t, "show", func() error { return l.Wait(ctx, EpShow) }); err != nil {
		t.Fatal(err)
	}
	// and a show with a short deadline honours it rather than queueing
	short, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
	defer cancel()
	if err := within(t, "show with a deadline", func() error { return l.Wait(short, EpShow) }); err == nil {
		t.Fatal("a second show within 2s must wait, and its deadline must cut the wait")
	}
	close(c.release)
	for range 2 {
		if err := <-searched; err != nil {
			t.Fatal(err)
		}
	}
}

func TestLimiterQueuedSearchesBookSuccessiveSlots(t *testing.T) {
	c := newBlockClock()
	l := NewLimiter(c)
	ctx := context.Background()
	for i := range 10 {
		c.t = c.t.Add(time.Second) // t+1 .. t+10
		if err := l.Wait(ctx, EpSearch); err != nil {
			t.Fatalf("search %d: %v", i+1, err)
		}
	}
	done := make(chan error, 2)
	go func() { done <- l.Wait(ctx, EpSearch) }()
	first := c.sleeping(t)
	go func() { done <- l.Wait(ctx, EpSearch) }()
	second := c.sleeping(t)
	if first != 51*time.Second || second != 52*time.Second {
		t.Fatalf("queued searches sleep %v and %v, want 51s and 52s", first, second)
	}
	close(c.release)
	for range 2 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
}

func TestLimiterCancelledWaiterDoesNotQueue(t *testing.T) {
	c := newBlockClock()
	l := NewLimiter(c)
	ctx := context.Background()
	if err := l.Wait(ctx, EpCreate); err != nil {
		t.Fatal(err)
	}
	first := make(chan error, 1)
	go func() { first <- l.Wait(ctx, EpCreate) }()
	if d := c.sleeping(t); d != 4500*time.Millisecond {
		t.Fatalf("second create sleeps %v, want 4.5s", d)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := within(t, "cancelled create", func() error { return l.Wait(cancelled, EpCreate) }); err != context.Canceled {
		t.Fatalf("cancelled create: %v, want context.Canceled", err)
	}
	if got, want := l.last[EpCreate], c.t.Add(4500*time.Millisecond); !got.Equal(want) {
		t.Fatalf("the cancelled waiter booked a slot: last %v, want %v", got, want)
	}
	close(c.release)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
}

func TestLimiterCancelledSleepHandsSlotBack(t *testing.T) {
	c := newBlockClock()
	l := NewLimiter(c)
	ctx := context.Background()
	if err := l.Wait(ctx, EpShow); err != nil {
		t.Fatal(err)
	}
	booked := l.last[EpShow]
	waiting, cancel := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- l.Wait(waiting, EpShow) }()
	_ = c.sleeping(t)
	cancel()
	if err := <-done; err != context.Canceled {
		t.Fatalf("cancelled show: %v", err)
	}
	if got := l.last[EpShow]; !got.Equal(booked) {
		t.Fatalf("slot not handed back: last %v, want %v", got, booked)
	}
}
