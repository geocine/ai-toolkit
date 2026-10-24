package provision

import (
	"context"
	"slices"
	"sync"
	"time"
)

// cacheTTL is how long an offers list or a quote is reused. Offers and the
// launch re-quote go through it, and so do the recommender's tier queries
// and images.check {offer}, so a burst of calls costs one provider call.
const cacheTTL = 30 * time.Second

// sharedCallTimeout bounds a shared offers or quote call. The call runs
// detached from its first caller, so one caller going away never fails the
// others waiting on the same key.
const sharedCallTimeout = 60 * time.Second

// offersKey is the user-facing part of an OfferQuery plus the kind and the
// key it ran with. Filter and ImageInfo are derived from Image by the
// service and never part of the key.
type offersKey struct {
	kind      Kind
	key       string // keyFingerprint
	gpu       string
	minVRAM   int
	maxVRAM   int
	maxPrice  float64
	gpuCount  int
	cloud     string
	diskGB    int
	limit     int
	image     string
	modelArch string
	showAll   bool
	fast      bool
}

func offersKeyFor(k Kind, fp string, q OfferQuery) offersKey {
	ok := offersKey{kind: k, key: fp, gpu: q.GPU, minVRAM: q.MinVRAMGB, maxVRAM: q.MaxVRAMGB, maxPrice: q.MaxPrice,
		gpuCount: q.GPUCount, cloud: q.Cloud, diskGB: q.DiskGB, limit: q.Limit, image: q.Image, showAll: q.ShowAll, fast: q.Fast}
	if q.Model != nil {
		ok.modelArch = q.Model.Arch
	}
	return ok
}

// quoteKey: one offer at one disk size and GPU count (PLAN §5 Launch step 6).
type quoteKey struct {
	kind     Kind
	key      string
	offerID  string
	diskGB   int
	gpuCount int
	cloud    string
}

type cached[T any] struct {
	val T
	at  time.Time
}

// flight shares one in-progress call per key among concurrent callers.
type flight[T any] struct {
	done chan struct{}
	val  T
	err  error
}

// caches holds the 30 s offers and quote caches with their in-flight calls.
type caches struct {
	mu        sync.Mutex
	offers    map[offersKey]cached[[]Offer]
	offersFly map[offersKey]*flight[[]Offer]
	quotes    map[quoteKey]cached[Offer]
	quotesFly map[quoteKey]*flight[Offer]
}

// load returns m[k] when fresh, else runs fn once for every concurrent caller
// of the same key and stores a successful result. Errors are never cached.
// fn gets a context detached from ctx (bounded by sharedCallTimeout); a
// caller whose ctx ends stops waiting without failing the others. Expired
// entries are swept on each insert, so the maps hold only the last 30 s.
func load[K comparable, T any](ctx context.Context, mu *sync.Mutex, m *map[K]cached[T], fly *map[K]*flight[T], k K, now func() time.Time, fn func(context.Context) (T, error)) (T, error) {
	mu.Lock()
	if *m == nil {
		*m, *fly = map[K]cached[T]{}, map[K]*flight[T]{}
	}
	if c, ok := (*m)[k]; ok && now().Sub(c.at) < cacheTTL {
		mu.Unlock()
		return c.val, nil
	}
	if f, ok := (*fly)[k]; ok {
		mu.Unlock()
		select {
		case <-f.done:
			return f.val, f.err
		case <-ctx.Done():
			var zero T
			return zero, ctx.Err()
		}
	}
	f := &flight[T]{done: make(chan struct{})}
	(*fly)[k] = f
	mu.Unlock()

	fctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), sharedCallTimeout)
	f.val, f.err = fn(fctx)
	cancel()
	mu.Lock()
	delete(*fly, k)
	if f.err == nil {
		t := now()
		for key, c := range *m {
			if t.Sub(c.at) >= cacheTTL {
				delete(*m, key)
			}
		}
		(*m)[k] = cached[T]{val: f.val, at: t}
	}
	mu.Unlock()
	close(f.done)
	return f.val, f.err
}

// dropOffers forgets every cached offers list and quote of k (its key changed).
func (s *Service) dropOffers(k Kind) {
	s.cache.mu.Lock()
	defer s.cache.mu.Unlock()
	for key := range s.cache.offers {
		if key.kind == k {
			delete(s.cache.offers, key)
		}
	}
	for key := range s.cache.quotes {
		if key.kind == k {
			delete(s.cache.quotes, key)
		}
	}
}

// Offers lists k's offers, reusing an identical query's answer for 30 s.
func (s *Service) Offers(ctx context.Context, k Kind, q OfferQuery) ([]Offer, error) {
	c, key, err := requireKey(ctx, k)
	if err != nil {
		return nil, err
	}
	ck := offersKeyFor(k, keyFingerprint(WireKey(key)), q)
	offers, err := load(ctx, &s.cache.mu, &s.cache.offers, &s.cache.offersFly, ck, s.now, func(ctx context.Context) ([]Offer, error) {
		o, err := c.Offers(ctx, WireKey(key), q)
		s.noteRetry(k, err)
		if o == nil && err == nil {
			o = []Offer{}
		}
		return o, err
	})
	if err != nil {
		return nil, scrub(err, key)
	}
	return slices.Clone(offers), nil
}

// Quote re-reads one offer through the 30 s quote cache.
func (s *Service) Quote(ctx context.Context, k Kind, offerID string, q OfferQuery) (Offer, error) {
	c, key, err := requireKey(ctx, k)
	if err != nil {
		return Offer{}, err
	}
	return s.quote(ctx, c, key, offerID, q)
}

func (s *Service) quote(ctx context.Context, c Cloud, key, offerID string, q OfferQuery) (Offer, error) {
	k := c.Kind()
	qk := quoteKey{kind: k, key: keyFingerprint(WireKey(key)), offerID: offerID, diskGB: q.DiskGB, gpuCount: q.GPUCount, cloud: q.Cloud}
	o, err := load(ctx, &s.cache.mu, &s.cache.quotes, &s.cache.quotesFly, qk, s.now, func(ctx context.Context) (Offer, error) {
		o, err := c.Quote(ctx, WireKey(key), offerID, q)
		s.noteRetry(k, err)
		return o, err
	})
	return o, scrub(err, key)
}

// priceRose is the 409 a launch gets when the offer now costs more than the
// caller agreed to.
func priceRose(k Kind, price, limit float64) error {
	return Errorf(k, ClassConflict, 0,
		"The price is now $%.2f/h, above your limit of $%.2f/h. Check the offer and try again.", price, limit)
}
