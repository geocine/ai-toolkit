package provision

import (
	"context"
	"slices"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/v1api"
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
// The host is what is cached; Service.Quote judges each caller's image on it.
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
// Each fresh call reads the image once (cached by the image check) and
// hands the adapter its host filter, so every offer carries host and fit.
func (s *Service) Offers(ctx context.Context, k Kind, q OfferQuery) ([]Offer, error) {
	c, key, err := requireKey(ctx, k)
	if err != nil {
		return nil, err
	}
	ck := offersKeyFor(k, keyFingerprint(WireKey(key)), q)
	offers, err := load(ctx, &s.cache.mu, &s.cache.offers, &s.cache.offersFly, ck, s.now, func(ctx context.Context) ([]Offer, error) {
		o, err := c.Offers(ctx, WireKey(key), withImageFilter(ctx, q))
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

// Quote re-reads one offer through the 30 s quote cache, with its host
// and fit for q.Image.
func (s *Service) Quote(ctx context.Context, k Kind, offerID string, q OfferQuery) (Offer, error) {
	c, key, err := requireKey(ctx, k)
	if err != nil {
		return Offer{}, err
	}
	if q.Filter == nil {
		q = withImageFilter(ctx, q)
	}
	o, err := s.quote(ctx, c, key, offerID, q)
	if err == nil && o.Host != nil {
		// the cache is keyed on the host only: judge this caller's image and model
		DecorateOffer(&o, *o.Host, q)
	}
	return o, err
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
// caller agreed to; details.price_per_hour is the new price.
func priceRose(k Kind, price, limit float64) error {
	return &priceRoseError{err: Errorf(k, ClassConflict, 0,
		"The price is now $%.2f/h, above your limit of $%.2f/h. Check the offer and try again.", price, limit), price: price}
}

// priceRoseError is the adapter-shaped conflict plus the new price.
type priceRoseError struct {
	err   *Error
	price float64
}

func (e *priceRoseError) Error() string { return e.err.Error() }
func (e *priceRoseError) Unwrap() error { return e.err }

// V1Details is the 409 body with the new price (v1machines.DetailedError).
func (e *priceRoseError) V1Details() (code, message string, details any) {
	return v1api.CodeConflict, e.err.Message, map[string]any{"price_per_hour": e.price}
}
