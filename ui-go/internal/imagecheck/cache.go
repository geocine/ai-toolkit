package imagecheck

import (
	"container/list"
	"context"
	"maps"
	"slices"
	"sync"
	"time"
)

// Cache sizes and lifetimes (IMAGE-PLAN §2.3). Everything is in memory: a
// machine keeps its own snapshot of the Image it launched with.
const (
	lruSize     = 256              // images by digest|platform: content-addressed, so they never go stale
	refTTL      = 10 * time.Minute // ref -> digest; refresh bypasses it
	negativeTTL = 60 * time.Second // not_found and denied, so a debounced typo cannot hammer a registry
	tokenTTL    = 60 * time.Second // a token without expires_in (GHCR)
)

// lru is a fixed-size map of images by imageKey, evicting the least recently used.
type lru struct {
	size  int
	order *list.List // front = most recent; values are *lruEntry
	items map[string]*list.Element
}

type lruEntry struct {
	key string
	img *Image
}

func newLRU(size int) *lru {
	return &lru{size: size, order: list.New(), items: map[string]*list.Element{}}
}

func (l *lru) get(key string) (*Image, bool) {
	e, ok := l.items[key]
	if !ok {
		return nil, false
	}
	l.order.MoveToFront(e)
	return e.Value.(*lruEntry).img, true
}

func (l *lru) put(key string, img *Image) {
	if e, ok := l.items[key]; ok {
		e.Value.(*lruEntry).img = img
		l.order.MoveToFront(e)
		return
	}
	l.items[key] = l.order.PushFront(&lruEntry{key, img})
	for l.order.Len() > l.size {
		last := l.order.Back()
		l.order.Remove(last)
		delete(l.items, last.Value.(*lruEntry).key)
	}
}

type expiring[T any] struct {
	val T
	exp time.Time
}

// cache holds every per-client cache behind one lock.
type cache struct {
	mu     sync.Mutex
	images *lru
	refs   map[string]expiring[string] // normalized ref|platform -> imageKey
	neg    map[string]expiring[*Error] // normalized ref|platform -> not_found / denied
	tokens map[string]expiring[string] // api|repo -> bearer token
	flight map[string]*call
}

func newCache() *cache {
	return &cache{
		images: newLRU(lruSize),
		refs:   map[string]expiring[string]{},
		neg:    map[string]expiring[*Error]{},
		tokens: map[string]expiring[string]{},
		flight: map[string]*call{},
	}
}

// imageKey is the LRU key for an image read for platform. An index digest
// names every platform it lists, so the digest alone would hand an arm64
// read back to an amd64 caller.
func imageKey(digest, platform string) string { return digest + "|" + platform }

// image is the image digest names when it was read for platform.
func (c *cache) image(digest, platform string) (*Image, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.images.get(imageKey(digest, platform))
}

// storeImage keeps img under both its digests (for the platform it was read
// for) and remembers which image key resolved to.
func (c *cache) storeImage(key string, img *Image, now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.images.put(imageKey(img.PlatformDigest, img.Platform), img)
	c.images.put(imageKey(img.Digest, img.Platform), img)
	c.refs[key] = expiring[string]{imageKey(img.Digest, img.Platform), now.Add(refTTL)}
}

// fresh is the cached image key resolved to within refTTL, or the cached
// negative answer within negativeTTL.
func (c *cache) fresh(key string, now time.Time) (*Image, *Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if n, ok := c.neg[key]; ok {
		if now.Before(n.exp) {
			return nil, n.val
		}
		delete(c.neg, key)
	}
	if r, ok := c.refs[key]; ok {
		if now.Before(r.exp) {
			if img, ok := c.images.get(r.val); ok {
				return img, nil
			}
		}
		delete(c.refs, key)
	}
	return nil, nil
}

func (c *cache) remember(key string, err *Error, now time.Time) {
	if err.Kind != KindNotFound && err.Kind != KindDenied {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.neg[key] = expiring[*Error]{err, now.Add(negativeTTL)}
}

func (c *cache) forget(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.neg, key)
	delete(c.refs, key)
}

func (c *cache) token(key string, now time.Time) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t, ok := c.tokens[key]
	if ok && now.Before(t.exp) {
		return t.val, true
	}
	return "", false
}

func (c *cache) storeToken(key, token string, ttl time.Duration, now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tokens[key] = expiring[string]{token, now.Add(ttl)}
}

// call is one in-flight fetch that concurrent callers for the same key share.
type call struct {
	done chan struct{}
	img  *Image
	err  error
}

// do runs fn once per key at a time, on its own goroutine; callers that
// arrive while it runs get its result. Each caller waits only as long as its
// own ctx: leaving early never cancels the shared fetch, so fn must not use
// any one caller's ctx for cancellation.
func (c *cache) do(ctx context.Context, key string, fn func() (*Image, error)) (*Image, error) {
	c.mu.Lock()
	f, ok := c.flight[key]
	if !ok {
		f = &call{done: make(chan struct{})}
		c.flight[key] = f
		go func() {
			f.img, f.err = fn()
			c.mu.Lock()
			delete(c.flight, key)
			c.mu.Unlock()
			close(f.done)
		}()
	}
	c.mu.Unlock()
	select {
	case <-f.done:
		return f.img, f.err
	case <-ctx.Done():
		return nil, &Error{Kind: KindUnreachable, Message: "the registry did not answer in time", cause: ctx.Err()}
	}
}

// clone is a deep copy, so a caller can never change a cached image.
func (img *Image) clone() *Image {
	c := *img
	c.Platforms = slices.Clone(img.Platforms)
	c.LayerDigests = slices.Clone(img.LayerDigests)
	c.Labels = maps.Clone(img.Labels)
	c.Env = maps.Clone(img.Env)
	c.ExposedPorts = slices.Clone(img.ExposedPorts)
	c.Requires.CompatBranches = slices.Clone(img.Requires.CompatBranches)
	c.Requires.SM = slices.Clone(img.Requires.SM)
	c.Requires.PTX = slices.Clone(img.Requires.PTX)
	c.Requires.FlashAttnSM = slices.Clone(img.Requires.FlashAttnSM)
	c.Requires.Sources = maps.Clone(img.Requires.Sources)
	if img.RateLimit != nil {
		rl := *img.RateLimit
		c.RateLimit = &rl
	}
	return &c
}
