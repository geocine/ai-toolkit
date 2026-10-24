package vast

import (
	"context"
	"sync"
	"time"
)

// Endpoint names a Vast rate-limit class (vast.md §1.5: a minimum interval
// between calls per endpoint and identity; offer search is "10 per minute").
type Endpoint string

const (
	EpCreate    Endpoint = "create"    // PUT /asks/{id}/
	EpDestroy   Endpoint = "destroy"   // DELETE /instances/{id}/
	EpShow      Endpoint = "show"      // GET /instances/{id}/
	EpList      Endpoint = "list"      // GET /api/v1/instances/
	EpManage    Endpoint = "manage"    // PUT /instances/{id}/ (stop, start) and request_logs
	EpTemplates Endpoint = "templates" // GET /template/
	EpSearch    Endpoint = "search"    // POST /bundles/ (token bucket)
	EpUser      Endpoint = "user"      // GET /users/current/ (no documented limit)
)

// minInterval is the x-rate-limit.threshold of each endpoint (vast.md §1.5).
var minInterval = map[Endpoint]time.Duration{
	EpCreate:    4500 * time.Millisecond,
	EpDestroy:   3 * time.Second,
	EpShow:      2 * time.Second,
	EpList:      time.Second,
	EpManage:    time.Second,
	EpTemplates: 3 * time.Second,
}

const (
	searchBurst  = 10 // offer searches per window
	searchWindow = time.Minute
)

// Clock is the limiter's time source. Tests pass one whose Sleep advances a
// fake time, so nothing waits for real.
type Clock interface {
	Now() time.Time
	Sleep(ctx context.Context, d time.Duration) error
}

type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

func (realClock) Sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Limiter spaces calls so this server stays under Vast's per-endpoint
// thresholds instead of collecting 429s. A caller books its slot under the
// mutex and sleeps outside it, so a wait on one endpoint never holds up
// another, and every waiter honours its own ctx.
type Limiter struct {
	mu     sync.Mutex
	clock  Clock
	last   map[Endpoint]time.Time // the latest booked slot per endpoint
	search []time.Time            // booked search slots in the window, sorted; may lie ahead of now
}

// NewLimiter returns a limiter on c (nil = the wall clock).
func NewLimiter(c Clock) *Limiter {
	if c == nil {
		c = realClock{}
	}
	return &Limiter{clock: c, last: map[Endpoint]time.Time{}}
}

// Wait books the next slot for ep and blocks until it comes. A wait cut
// short by ctx hands its slot back when nobody booked after it.
func (l *Limiter) Wait(ctx context.Context, ep Endpoint) error {
	_, err := l.wait(ctx, ep, -1)
	return err
}

// WaitAtMost is Wait for a caller that will not queue longer than limit:
// when ep's next slot is more than limit away it books nothing and returns
// that wait as busy at once, so the caller can answer "busy, try again in
// busy" instead of outwaiting its own deadline.
func (l *Limiter) WaitAtMost(ctx context.Context, ep Endpoint, limit time.Duration) (busy time.Duration, err error) {
	return l.wait(ctx, ep, max(limit, 0))
}

// wait books and sleeps; limit < 0 waits as long as it takes.
func (l *Limiter) wait(ctx context.Context, ep Endpoint, limit time.Duration) (busy time.Duration, err error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	b, ok, busy := l.book(ep, limit)
	if busy > 0 {
		return busy, nil
	}
	if !ok || !b.at.After(b.now) {
		return 0, nil
	}
	if err := l.clock.Sleep(ctx, b.at.Sub(b.now)); err != nil {
		l.release(ep, b)
		return 0, err
	}
	return 0, nil
}

// booking is one reserved slot: when it was booked, when it is, and the
// endpoint's previous slot (zero = none) for handing it back.
type booking struct {
	now, at, prev time.Time
}

// book reserves the next slot for ep. It reports false for an endpoint
// without a documented limit. With limit >= 0 a slot more than limit away
// is not booked: busy is how far away it is.
func (l *Limiter) book(ep Endpoint, limit time.Duration) (b booking, ok bool, busy time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.clock.Now()
	b = booking{now: now, at: now}
	tooFar := func() bool { return limit >= 0 && b.at.Sub(now) > limit }
	if ep == EpSearch {
		l.search = dropBefore(l.search, now.Add(-searchWindow))
		if n := len(l.search); n >= searchBurst {
			// the slot opens when the search searchBurst places back leaves the window
			if t := l.search[n-searchBurst].Add(searchWindow); t.After(now) {
				b.at = t
			}
		}
		if tooFar() {
			return b, true, b.at.Sub(now)
		}
		l.search = append(l.search, b.at)
		return b, true, 0
	}
	gap, ok := minInterval[ep]
	if !ok {
		return b, false, 0
	}
	if last, seen := l.last[ep]; seen {
		b.prev = last
		if t := last.Add(gap); t.After(now) {
			b.at = t
		}
	}
	if tooFar() {
		return b, true, b.at.Sub(now)
	}
	l.last[ep] = b.at
	return b, true, 0
}

// release hands a cancelled slot back if it is still the latest one booked.
// A slot with later bookings behind it stays booked, which only spaces those
// calls more than needed.
func (l *Limiter) release(ep Endpoint, b booking) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if ep == EpSearch {
		if n := len(l.search); n > 0 && l.search[n-1].Equal(b.at) {
			l.search = l.search[:n-1]
		}
		return
	}
	if !l.last[ep].Equal(b.at) {
		return
	}
	if b.prev.IsZero() {
		delete(l.last, ep)
	} else {
		l.last[ep] = b.prev
	}
}

// dropBefore keeps the times strictly after cutoff (ts is sorted).
func dropBefore(ts []time.Time, cutoff time.Time) []time.Time {
	i := 0
	for i < len(ts) && !ts[i].After(cutoff) {
		i++
	}
	return ts[i:]
}
