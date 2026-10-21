// Package apicache caches slow API work. Mirrors ui/src/server/apiCache.ts:
// values are reused until stale, concurrent callers share one in-flight
// fetch, and failed fetches are not cached.
package apicache

import (
	"encoding/json"
	"strings"
	"sync"
	"time"
)

// DefaultStale matches upstream's DEFAULT_STALE_TIME_MS.
const DefaultStale = 5 * time.Second

type entry struct {
	done  chan struct{}
	value any
	err   error
	at    time.Time
}

var (
	mu    sync.Mutex
	cache = map[string]*entry{}
)

func cacheKey(key string, params any) string {
	if params == nil {
		return key
	}
	b, _ := json.Marshal(params)
	return key + ":" + string(b)
}

// Cached returns the fresh cached value for key+params, or runs fetch.
func Cached[T any](key string, stale time.Duration, params any, fetch func() (T, error)) (T, error) {
	k := cacheKey(key, params)
	mu.Lock()
	if e, ok := cache[k]; ok && time.Since(e.at) < stale {
		mu.Unlock()
		<-e.done
		if e.err != nil {
			var zero T
			return zero, e.err
		}
		return e.value.(T), nil
	}
	e := &entry{done: make(chan struct{}), at: time.Now()}
	cache[k] = e
	mu.Unlock()

	v, err := fetch()
	e.value, e.err = v, err
	close(e.done)
	if err != nil {
		mu.Lock()
		if cache[k] == e {
			delete(cache, k)
		}
		mu.Unlock()
	}
	return v, err
}

// Invalidate removes key (every params variant when params is nil).
func Invalidate(key string, params any) {
	mu.Lock()
	defer mu.Unlock()
	if params != nil {
		delete(cache, cacheKey(key, params))
		return
	}
	delete(cache, key)
	for k := range cache {
		if strings.HasPrefix(k, key+":") {
			delete(cache, k)
		}
	}
}
