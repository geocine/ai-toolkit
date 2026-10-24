package bg

import (
	"context"
	"slices"
	"testing"
	"time"
)

// reset gives a test a fresh registry and puts the old one back after it.
func reset(t *testing.T) {
	mu.Lock()
	oldLoops, oldSkip, oldStarted := loops, skip, started
	loops, skip, started = map[string]func(context.Context){}, map[string]bool{}, nil
	mu.Unlock()
	t.Cleanup(func() {
		mu.Lock()
		loops, skip, started = oldLoops, oldSkip, oldStarted
		mu.Unlock()
	})
}

func TestSkip(t *testing.T) {
	reset(t)
	ran := make(chan string, 8)
	loop := func(name string) func(context.Context) {
		return func(context.Context) { ran <- name }
	}
	Go("keep", loop("keep"))
	Go("drop", loop("drop"))
	Skip("drop", "late-drop")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	Start(ctx)
	// registered after Start: runs at once unless skipped
	Go("late", loop("late"))
	Go("late-drop", loop("late-drop"))

	got := map[string]bool{}
	timeout := time.After(2 * time.Second)
	for len(got) < 2 {
		select {
		case n := <-ran:
			got[n] = true
		case <-timeout:
			t.Fatalf("loops that ran: %v", got)
		}
	}
	select {
	case n := <-ran:
		got[n] = true
	case <-time.After(100 * time.Millisecond):
	}
	if !got["keep"] || !got["late"] || got["drop"] || got["late-drop"] {
		t.Fatalf("loops that ran: %v", got)
	}
	if names := Names(); !slices.Equal(names, []string{"drop", "keep", "late", "late-drop"}) {
		t.Fatalf("Names = %v", names)
	}
}
