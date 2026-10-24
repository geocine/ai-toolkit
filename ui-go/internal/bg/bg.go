// Package bg runs background loops that handler packages need (upload queue,
// lease heartbeats, ...) without each of them touching main: a package calls
// bg.Go from its Register, and main starts them all with Start.
package bg

import (
	"context"
	"log"
	"sort"
	"sync"
)

var (
	mu      sync.Mutex
	loops   = map[string]func(context.Context){}
	skip    = map[string]bool{}
	started context.Context
)

// Go registers a named loop; it runs once Start is called (or immediately
// when Start already ran), unless Skip named it. A name registers at most once.
func Go(name string, fn func(context.Context)) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := loops[name]; dup {
		return
	}
	loops[name] = fn
	if started != nil && !skip[name] {
		go run(started, name, fn)
	}
}

// Skip marks loops that never start (controller mode). Call before Start.
func Skip(names ...string) {
	mu.Lock()
	defer mu.Unlock()
	for _, n := range names {
		skip[n] = true
	}
}

// Names lists every registered loop name, skipped or not (for the controller
// allow-list test).
func Names() []string {
	mu.Lock()
	defer mu.Unlock()
	out := make([]string, 0, len(loops))
	for name := range loops {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Start runs every registered loop that is not skipped until ctx is done.
func Start(ctx context.Context) {
	mu.Lock()
	defer mu.Unlock()
	started = ctx
	for name, fn := range loops {
		if !skip[name] {
			go run(ctx, name, fn)
		}
	}
}

func run(ctx context.Context, name string, fn func(context.Context)) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("bg %s: panic: %v", name, r)
		}
	}()
	fn(ctx)
}
