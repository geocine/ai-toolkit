package provision

import (
	"context"
	"errors"
	"log"
	"math/rand/v2"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
)

// watchEvery is the watcher tick; each tick waits ±1 s more or less.
const watchEvery = 5 * time.Second

// Run is the watcher loop (bg.Go "machines"): one Tick every 5 s ± 1 s until
// ctx ends.
func (s *Service) Run(ctx context.Context) {
	for {
		if state.DB != nil {
			s.Tick(ctx)
		}
		jitter := time.Duration(rand.Int64N(int64(2*time.Second))) - time.Second
		select {
		case <-ctx.Done():
			return
		case <-time.After(watchEvery + jitter):
		}
	}
}

// due reports whether r should be refreshed now: transitional rows and rows
// with a problem every 10 s, ready ones every minute, stopped ones every
// 5 min, ended ones never.
func (s *Service) due(r *row, now time.Time) bool {
	if r.EndedAt != nil {
		return false
	}
	last := s.lastRefresh(r.ID)
	if last.IsZero() {
		return true
	}
	every := refreshReady
	switch {
	case r.Status.Transitional() || r.Problem != "" || r.Status == StatusFailed || r.Status == StatusUnknown:
		every = refreshBusy
	case r.Status == StatusStopped:
		every = refreshStopped
	}
	return now.Sub(last) >= every
}

// Tick refreshes every due machine: per provider with due rows, one List
// (matched by provider id, or by label for QuickPod rows not resolved yet)
// and one Get only to confirm a machine that left the list. A provider that
// answered with a Retry-After is skipped until it passes.
func (s *Service) Tick(ctx context.Context) {
	rows, err := listRows(ctx, true)
	if err != nil {
		log.Printf("provision: watcher: %v", err)
		return
	}
	now := s.now()
	byKind := map[Kind][]string{}
	for _, r := range rows {
		if !s.due(r, now) {
			continue
		}
		if r.Simulated != Simulated.Load() {
			s.withRow(ctx, r.ID, func(r *row) error {
				s.markRefreshed(r.ID, now)
				_, err := s.splitRow(ctx, r)
				return err
			})
			continue
		}
		byKind[r.Provider] = append(byKind[r.Provider], r.ID)
	}
	for _, k := range allKinds {
		ids := byKind[k]
		if len(ids) == 0 || s.paused(k, now) {
			continue
		}
		c, ok := Lookup(k)
		if !ok {
			continue
		}
		key, source, err := storedKey(ctx, k)
		if err != nil || source == "" {
			continue
		}
		unlock := s.lockKind(k)
		insts, err := c.List(ctx, WireKey(key), LabelPrefix)
		unlock()
		if err != nil {
			s.noteRetry(k, err)
			log.Printf("provision: watcher: %v", scrub(err, key))
			continue
		}
		byID, byLabel := map[string]Instance{}, map[string]Instance{}
		for _, in := range insts {
			if in.ProviderID != "" {
				byID[in.ProviderID] = in
			}
			if in.Label != "" {
				byLabel[in.Label] = in
			}
		}
		for _, id := range ids {
			s.withRow(ctx, id, func(r *row) error {
				s.markRefreshed(r.ID, s.now())
				return s.refreshListed(ctx, c, key, r, byID, byLabel)
			})
		}
	}
}

// withRow runs fn on the current row under its lock; ended rows are skipped.
func (s *Service) withRow(ctx context.Context, id string, fn func(*row) error) {
	unlock := s.lockMachine(id)
	defer unlock()
	r, err := getRow(ctx, id)
	if errors.Is(err, errNoRow) {
		return // forgotten meanwhile
	}
	if err == nil && r.EndedAt == nil {
		err = fn(r)
	}
	if err != nil {
		log.Printf("provision: watcher %s: %v", id, err)
	}
}

func (s *Service) refreshListed(ctx context.Context, c Cloud, key string, r *row, byID, byLabel map[string]Instance) error {
	if r.ProviderID != "" {
		if in, ok := byID[r.ProviderID]; ok {
			return s.apply(ctx, r, &in, false)
		}
	} else if in, ok := byLabel[r.Label]; ok {
		return s.apply(ctx, r, &in, false)
	}
	if r.ProviderID == "" || !s.wasSeen(r) {
		// not created at the provider yet, or not listed yet: only the
		// timeouts (never_listed for QuickPod) apply
		return s.apply(ctx, r, nil, r.ProviderID == "")
	}
	// it left the list: one Get tells gone from a list that lags
	unlock := s.lockKind(r.Provider)
	inst, err := c.Get(ctx, WireKey(key), Ref{ID: r.ProviderID, OfferID: r.Spec.Offer.ID, Label: r.Label})
	unlock()
	switch {
	case err == nil:
		return s.apply(ctx, r, &inst, false)
	case ClassOf(err) == ClassNotFound:
		return s.apply(ctx, r, nil, true)
	default:
		s.noteRetry(r.Provider, err)
		return s.apply(ctx, r, nil, false)
	}
}
