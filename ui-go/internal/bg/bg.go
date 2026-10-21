// Package bg runs background loops that handler packages need (upload queue,
// lease heartbeats, ...) without each of them touching main: a package calls
// bg.Go from its Register, and main starts them all with Start.
package bg

import (
	"context"
	"log"
	"sync"
)

var (
	mu      sync.Mutex
	loops   = map[string]func(context.Context){}
	started context.Context
)

// Go registers a named loop; it runs once Start is called (or immediately
// when Start already ran). A name registers at most once.
func Go(name string, fn func(context.Context)) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := loops[name]; dup {
		return
	}
	loops[name] = fn
	if started != nil {
		go run(started, name, fn)
	}
}

// Start runs every registered loop until ctx is done.
func Start(ctx context.Context) {
	mu.Lock()
	defer mu.Unlock()
	started = ctx
	for name, fn := range loops {
		go run(ctx, name, fn)
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
