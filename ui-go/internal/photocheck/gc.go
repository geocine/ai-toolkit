package photocheck

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/geocine/aitoolkit/internal/bg"
	"github.com/geocine/aitoolkit/internal/state"
)

const (
	// gcKeep is how long a photo's analysis outlives the last review that
	// read it. Reviews walk .set_aside/ too, so left-out photos stay seen.
	gcKeep = 90 * 24 * time.Hour
	// gcEvery is the GC period; the first run waits gcFirst after start.
	gcEvery = 24 * time.Hour
	gcFirst = 10 * time.Minute
)

// StartGC registers the daily photo-cache GC (bg loop "photocheck-gc"). The
// photo routes call it from their registration, so it starts with the other
// loops; the studio API skips it.
func StartGC() {
	bg.Go("photocheck-gc", func(ctx context.Context) {
		t := time.NewTimer(gcFirst)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
			if err := GC(ctx, time.Now()); err != nil {
				log.Printf("photocheck-gc: %v", err)
			}
			t.Reset(gcEvery)
		}
	})
}

// GC deletes metrics and face rows not seen by a review for 90 days, and
// the photo state of dataset folders that no longer exist.
func GC(ctx context.Context, now time.Time) error {
	if _, err := state.GCPhotos(ctx, now.Add(-gcKeep)); err != nil {
		return err
	}
	dirs, err := state.PhotoSetDirs(ctx)
	if err != nil {
		return err
	}
	for _, d := range dirs {
		if _, err := os.Stat(d); isNotExist(err) {
			if err := state.DeletePhotoSet(ctx, d); err != nil {
				return err
			}
		}
	}
	return nil
}
