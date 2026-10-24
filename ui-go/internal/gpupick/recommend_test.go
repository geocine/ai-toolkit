package gpupick

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/imagetest"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// fakeManager answers Providers and Offers; the other Manager methods are
// never called by gpupick (the nil embedded interface panics if they are).
type fakeManager struct {
	provision.Manager
	views  []provision.ProviderView
	offers map[provision.Kind][]provision.Offer
	errs   map[provision.Kind]error
	// errAt fails one query of a provider (nil: answer it).
	errAt func(provision.Kind, provision.OfferQuery) error
	// block holds a provider's Offers until the channel closes, ignoring
	// ctx like the offers cache's detached shared flight.
	block map[provision.Kind]chan struct{}
	// fast makes Offers honour Fast like vastmock: a missing or slower
	// inet_down / disk_bw fails the gte filter.
	fast bool
	// image shapes the registry mock useFake starts.
	image imagetest.Options

	mu      sync.Mutex
	queries []fakeQuery
}

type fakeQuery struct {
	kind provision.Kind
	q    provision.OfferQuery
}

func (f *fakeManager) Providers(context.Context) ([]provision.ProviderView, error) {
	return f.views, nil
}

// Offers filters like the adapters (VRAM floor and exclusive cap, price,
// GPU count), sorts by price and applies Limit (default 50, max 200).
func (f *fakeManager) Offers(_ context.Context, k provision.Kind, q provision.OfferQuery) ([]provision.Offer, error) {
	f.mu.Lock()
	f.queries = append(f.queries, fakeQuery{k, q})
	f.mu.Unlock()
	if ch := f.block[k]; ch != nil {
		<-ch
	}
	if err := f.errs[k]; err != nil {
		return nil, err
	}
	if f.errAt != nil {
		if err := f.errAt(k, q); err != nil {
			return nil, err
		}
	}
	var out []provision.Offer
	for _, o := range f.offers[k] {
		v := offerVRAM(o)
		switch {
		case q.MinVRAMGB > 0 && v < float64(q.MinVRAMGB),
			q.MaxVRAMGB > 0 && v >= float64(q.MaxVRAMGB),
			q.MaxPrice > 0 && o.PricePerHour > q.MaxPrice,
			q.GPUCount > 0 && o.GPUCount != q.GPUCount:
			continue
		}
		if f.fast && q.Fast && (o.Host == nil || o.Host.InetDownMbps < hostfit.FastInetDownMbps || o.Host.DiskBWMBps < hostfit.FastDiskBWMBps) {
			continue
		}
		out = append(out, o)
	}
	slices.SortStableFunc(out, func(a, b provision.Offer) int { return cmp.Compare(a.PricePerHour, b.PricePerHour) })
	limit := q.Limit
	if limit <= 0 {
		limit = 50
	}
	if len(out) > min(limit, 200) {
		out = out[:min(limit, 200)]
	}
	return out, nil
}

func (f *fakeManager) queriesFor(k provision.Kind) []provision.OfferQuery {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []provision.OfferQuery
	for _, q := range f.queries {
		if q.kind == k {
			out = append(out, q.q)
		}
	}
	return out
}

// useFake makes f provision.Default (configured kinds have a key), points
// the image check at registrymock and restores both after the test.
func useFake(t *testing.T, f *fakeManager, configured ...provision.Kind) *fakeManager {
	t.Helper()
	for _, k := range []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod} {
		f.views = append(f.views, provision.ProviderView{
			KindInfo:   provision.KindInfo{Kind: k, Label: provision.Label(k), Caps: capsOf(k)},
			Configured: slices.Contains(configured, k),
		})
	}
	if f.offers == nil {
		f.offers = map[provision.Kind][]provision.Offer{}
	}
	old := provision.Default
	provision.Default = f
	t.Cleanup(func() { provision.Default = old })
	imagetest.Use(t, f.image)
	return f
}

func recommend(t *testing.T, q Query) Recommendation {
	t.Helper()
	rec, err := Recommend(context.Background(), q)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	return rec
}

func slow(o provision.Offer) provision.Offer {
	return with(o, func(_ *provision.Offer, h *hostfit.Host) { h.InetDownMbps = 73 })
}

func hasNote(rec Recommendation, k provision.Kind, code string) *Note {
	for i, n := range rec.Notes {
		if n.Provider == k && n.Code == code {
			return &rec.Notes[i]
		}
	}
	return nil
}

func TestRecommendStepsUpWhen24IsSlow(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.Vast)
	for i := range 5 {
		f.offers[provision.Vast] = append(f.offers[provision.Vast], slow(mk(provision.Vast, fmt.Sprint("slow", i), "RTX 4090", 23.99, 0.30)))
	}
	f.offers[provision.Vast] = append(f.offers[provision.Vast], mk(provision.Vast, "5090", "RTX 5090", 31.84, 0.90))
	rec := recommend(t, Query{Providers: []provision.Kind{provision.Vast}, Model: &hostfit.ModelRef{Arch: "flux"}})
	if rec.TierGB != 32 || rec.Value == nil || rec.Value.Offer.ID != "5090" || rec.Default != "value" {
		t.Fatalf("tier %d value %+v", rec.TierGB, rec.Value)
	}
	if len(rec.Tiers) != 2 || rec.Tiers[0].TierGB != 24 || rec.Tiers[0].Rejected[RejectInet] != 5 || rec.Tiers[0].Qualified != 0 {
		t.Errorf("tiers %+v", rec.Tiers)
	}
	if rec.StartTierGB != 24 || rec.Message != "" || rec.Value.Reason != "RTX 5090 · 32 GB · $0.90/h · 900 Mbps · CUDA 13.2 OK" {
		t.Errorf("start %d message %q reason %q", rec.StartTierGB, rec.Message, rec.Value.Reason)
	}
}

func TestRecommendSlowHostsCannotCrowdThePage(t *testing.T) {
	f := useFake(t, &fakeManager{fast: true}, provision.Vast)
	var offers []provision.Offer
	for i := range 64 {
		offers = append(offers, slow(mk(provision.Vast, fmt.Sprintf("slow%02d", i), "RTX 3090", 24, 0.10+float64(i)*0.001)))
	}
	offers = append(offers, mk(provision.Vast, "fast-4090", "RTX 4090", 23.99, 0.50))
	for i := range 10 { // qualifying offers past row 50 by price
		offers = append(offers, mk(provision.Vast, fmt.Sprint("fast-3090-", i), "RTX 3090", 24, 0.60+float64(i)*0.01))
		offers = append(offers, mk(provision.Vast, fmt.Sprint("fast-5090-", i), "RTX 5090", 31.84, 0.70+float64(i)*0.01))
	}
	f.offers[provision.Vast] = offers
	rec := recommend(t, Query{Providers: []provision.Kind{provision.Vast}})
	if rec.TierGB != 24 || rec.Value == nil || rec.Value.Offer.ID != "fast-4090" || rec.Speed.Offer.ID != "fast-4090" {
		t.Fatalf("tier %d value %+v speed %+v", rec.TierGB, rec.Value, rec.Speed)
	}
	qs := f.queriesFor(provision.Vast)
	if len(qs) != 1 {
		t.Fatalf("%d queries, want 1 (the walk stops at 24)", len(qs))
	}
	q := qs[0]
	if !q.Fast || q.Limit != 200 || q.MinVRAMGB != 21 || q.MaxVRAMGB != 29 || q.GPUCount != 1 || q.DiskGB != rec.Need.DiskGB {
		t.Errorf("tier query %+v", q)
	}
}

func TestRecommendTierQueries(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.RunPod)
	f.offers[provision.RunPod] = []provision.Offer{mk(provision.RunPod, "h100", "H100 SXM", 80, 2.99)}
	rec := recommend(t, Query{Providers: []provision.Kind{provision.RunPod}, Image: provision.DefaultImage, Cloud: "community", MaxPrice: 3, DiskGB: 300})
	if rec.TierGB != 80 || rec.Value == nil {
		t.Fatalf("tier %d value %+v", rec.TierGB, rec.Value)
	}
	want := [][2]int{{21, 29}, {28, 44}, {43, 72}, {72, 87}}
	qs := f.queriesFor(provision.RunPod)
	if len(qs) != len(want) {
		t.Fatalf("%d queries, want %d", len(qs), len(want))
	}
	for i, q := range qs {
		if q.MinVRAMGB != want[i][0] || q.MaxVRAMGB != want[i][1] || q.Cloud != "community" || q.MaxPrice != 3 || q.DiskGB != 300 ||
			q.Image != provision.DefaultImage || !q.Fast || q.Limit != TierLimit {
			t.Errorf("query %d: %+v, want vram %v", i, q, want[i])
		}
	}
	if rec.Value.Reason != "H100 SXM · 80 GB · $2.99/h · net not reported · CUDA 13.0+ by RunPod" {
		t.Errorf("reason %q", rec.Value.Reason)
	}

	// the last tier has no cap
	f2 := useFake(t, &fakeManager{}, provision.Vast)
	rec = recommend(t, Query{Providers: []provision.Kind{provision.Vast}})
	qs = f2.queriesFor(provision.Vast)
	if rec.Value != nil || len(qs) != len(Ladder) || qs[len(qs)-1].MinVRAMGB != 162 || qs[len(qs)-1].MaxVRAMGB != 0 {
		t.Errorf("walk of an empty provider: %d queries, last %+v", len(qs), qs[len(qs)-1])
	}
}

func TestRecommendCrossProvider(t *testing.T) {
	f := useFake(t, &fakeManager{errs: map[provision.Kind]error{
		provision.Vast: &provision.Error{Provider: provision.Vast, Class: provision.ClassRateLimited, Status: 429, Message: "slow down", RetryAfter: 30 * time.Second},
	}}, provision.RunPod, provision.Vast)
	f.offers[provision.RunPod] = []provision.Offer{mk(provision.RunPod, "rp", "RTX 4090", 24, 0.69)}
	rec := recommend(t, Query{})
	if !slices.Equal(rec.Providers, []provision.Kind{provision.RunPod, provision.Vast}) {
		t.Errorf("providers %v", rec.Providers)
	}
	if rec.Value == nil || rec.Value.Offer.Provider != provision.RunPod {
		t.Fatalf("value %+v", rec.Value)
	}
	if !strings.HasPrefix(rec.Value.Reason, "RTX 4090 on RunPod · ") {
		t.Errorf("cross-provider reason %q", rec.Value.Reason)
	}
	if n := hasNote(rec, provision.Vast, v1api.CodeRateLimited); n == nil || !strings.Contains(n.Message, "Vast.ai") {
		t.Errorf("notes %+v, want a Vast rate_limited note", rec.Notes)
	}
	if qs := f.queriesFor(provision.Vast); len(qs) != 1 {
		t.Errorf("Vast asked %d times after failing, want 1", len(qs))
	}

	// One provider that fails is the error, mappable to the 429.
	_, err := Recommend(context.Background(), Query{Providers: []provision.Kind{provision.Vast}})
	if code, _, ra, ok := provision.V1(err); !ok || code != v1api.CodeRateLimited || ra != 30*time.Second {
		t.Errorf("single provider error %v", err)
	}
	// Across providers, it fails only when every provider failed; the notes come with it.
	f.errs[provision.RunPod] = provision.Errorf(provision.RunPod, provision.ClassUnavailable, 502, "bad gateway")
	rec, err = Recommend(context.Background(), Query{})
	if provision.ClassOf(err) != provision.ClassUnavailable || hasNote(rec, provision.Vast, v1api.CodeRateLimited) == nil ||
		hasNote(rec, provision.RunPod, v1api.CodeRemoteFailed) == nil {
		t.Errorf("all failed: err %v notes %+v", err, rec.Notes)
	}
}

func TestRecommendSlowProviderTimesOut(t *testing.T) {
	old := providerTimeout
	providerTimeout = 50 * time.Millisecond
	t.Cleanup(func() { providerTimeout = old })
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	f := useFake(t, &fakeManager{block: map[provision.Kind]chan struct{}{provision.Vast: release}}, provision.RunPod, provision.Vast)
	f.offers[provision.RunPod] = []provision.Offer{mk(provision.RunPod, "rp", "RTX 4090", 24, 0.69)}

	start := time.Now()
	rec := recommend(t, Query{})
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("Recommend took %v; Vast should be cut off after %v", d, providerTimeout)
	}
	if rec.Value == nil || rec.Value.Offer.ID != "rp" {
		t.Errorf("value %+v, want RunPod's offer", rec.Value)
	}
	if n := hasNote(rec, provision.Vast, v1api.CodeRemoteFailed); n == nil || !strings.HasPrefix(n.Message, "Vast.ai did not answer") {
		t.Errorf("notes %+v, want a Vast remote_failed note", rec.Notes)
	}

	// One provider: the timeout is a *provision.Error the API maps to 502.
	start = time.Now()
	_, err := Recommend(context.Background(), Query{Providers: []provision.Kind{provision.Vast}})
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("single provider took %v", d)
	}
	var pe *provision.Error
	if code, _, _, ok := provision.V1(err); !ok || code != v1api.CodeRemoteFailed || !errors.As(err, &pe) {
		t.Errorf("single provider timeout: %v", err)
	}
}

func TestRecommendStopsWhenTheProviderFailsPartway(t *testing.T) {
	limited := &provision.Error{Provider: provision.Vast, Class: provision.ClassRateLimited, Status: 429, Message: "slow down", RetryAfter: 30 * time.Second}
	f := useFake(t, &fakeManager{errAt: func(k provision.Kind, q provision.OfferQuery) error {
		if k == provision.Vast && q.MinVRAMGB >= 28 {
			return limited
		}
		return nil
	}}, provision.RunPod, provision.Vast)
	f.offers[provision.Vast] = []provision.Offer{slow(mk(provision.Vast, "a", "RTX 4090", 24, 0.3))}

	// One provider: the 429 itself, not "nothing qualifies".
	rec, err := Recommend(context.Background(), Query{Providers: []provision.Kind{provision.Vast}})
	if code, _, ra, ok := provision.V1(err); !ok || code != v1api.CodeRateLimited || ra != 30*time.Second {
		t.Fatalf("single provider: err %v", err)
	}
	if len(rec.Tiers) != 1 || rec.Tiers[0].TierGB != 24 {
		t.Errorf("tiers %+v, want only the answered 24", rec.Tiers)
	}

	// Across providers: a 200 whose message says where the walk stopped.
	f.offers[provision.RunPod] = nil
	f.errAt = func(_ provision.Kind, q provision.OfferQuery) error {
		if q.MinVRAMGB >= 28 {
			return limited
		}
		return nil
	}
	rec = recommend(t, Query{Model: &hostfit.ModelRef{Arch: "flux"}})
	want := fmt.Sprintf("No offer from 24 to 24 GB has fast internet, enough disk (%d GB) and a driver for CUDA 13.0 right now. "+
		"Most were left out for slow internet (1). Pick one yourself from the list. The search stopped at 32 GB because RunPod and Vast.ai failed.", rec.Need.DiskGB)
	if rec.Value != nil || rec.Message != want {
		t.Errorf("message\n got %q\nwant %q", rec.Message, want)
	}
	if len(rec.Tiers) != 1 {
		t.Errorf("tiers %+v, want only the answered 24", rec.Tiers)
	}
}

func TestRecommendNeedsAKey(t *testing.T) {
	useFake(t, &fakeManager{})
	_, err := Recommend(context.Background(), Query{})
	var p *v1api.Problem
	if !errors.As(err, &p) || p.Code != v1api.CodeInvalidRequest || p.Message != "Add an API key for RunPod, Vast.ai or QuickPod first" {
		t.Errorf("no key at all: %v", err)
	}
	_, err = Recommend(context.Background(), Query{Providers: []provision.Kind{provision.RunPod}})
	if !errors.As(err, &p) || p.Code != v1api.CodeInvalidRequest || p.Message != "Add a RunPod API key first" {
		t.Errorf("RunPod without a key: %v", err)
	}
	rec, err := Recommend(context.Background(), Query{Providers: []provision.Kind{provision.RunPod, provision.Vast}})
	if !errors.As(err, &p) || p.Message != "Add an API key for RunPod or Vast.ai first" {
		t.Errorf("listed providers without keys: %v", err)
	}
	b, _ := json.Marshal(rec)
	for _, k := range []string{`"providers":[]`, `"tiers":[]`, `"ladder":[24,`, `"value":null`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("error-path JSON lacks %s: %s", k, b)
		}
	}
	if _, err := Recommend(context.Background(), Query{Providers: []provision.Kind{"nope"}}); !errors.As(err, &p) || p.Code != v1api.CodeNotFound {
		t.Errorf("unknown provider: %v", err)
	}
}

func TestRecommendSkipsProvidersWithoutKeys(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.Vast)
	f.offers[provision.Vast] = []provision.Offer{mk(provision.Vast, "v", "RTX 4090", 24, 0.4)}
	rec := recommend(t, Query{Providers: []provision.Kind{provision.QuickPod, provision.Vast}})
	if rec.Value == nil || !slices.Equal(rec.Providers, []provision.Kind{provision.Vast}) || hasNote(rec, provision.QuickPod, NoteNotConfigured) == nil {
		t.Errorf("providers %v notes %+v value %+v", rec.Providers, rec.Notes, rec.Value)
	}
	if len(f.queriesFor(provision.QuickPod)) != 0 {
		t.Error("QuickPod was asked without a key")
	}
}

func TestRecommendNothingQualifies(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.Vast)
	f.offers[provision.Vast] = []provision.Offer{
		slow(mk(provision.Vast, "a", "RTX 4090", 24, 0.3)),
		slow(mk(provision.Vast, "b", "RTX 3090", 24, 0.2)),
		slow(mk(provision.Vast, "c", "RTX 3090", 24, 0.25)),
		with(mk(provision.Vast, "d", "RTX 3090", 24, 0.25), func(_ *provision.Offer, h *hostfit.Host) { h.DiskBWMBps = 50 }),
	}
	rec := recommend(t, Query{Providers: []provision.Kind{provision.Vast}, Model: &hostfit.ModelRef{Arch: "flux"}})
	if rec.Value != nil || rec.Speed != nil || rec.Default != "" || rec.TierGB != 0 {
		t.Fatalf("picks %+v %+v default %q", rec.Value, rec.Speed, rec.Default)
	}
	want := fmt.Sprintf("No offer from 24 to 180 GB has fast internet, enough disk (%d GB) and a driver for CUDA 13.0 right now. "+
		"Most were left out for slow internet (3). Pick one yourself from the list.", rec.Need.DiskGB)
	if rec.Message != want {
		t.Errorf("message\n got %q\nwant %q", rec.Message, want)
	}
	if len(rec.Tiers) != len(Ladder) {
		t.Errorf("walked %d tiers, want all %d", len(rec.Tiers), len(Ladder))
	}

	// a 48 GB start
	f.offers[provision.Vast] = []provision.Offer{
		with(mk(provision.Vast, "a6000", "RTX A6000", 47.99, 0.45), codes("block", "host_cuda")),
		with(mk(provision.Vast, "a6000b", "RTX A6000", 47.99, 0.47), codes("block", "host_cuda")),
		with(mk(provision.Vast, "l40s", "L40S", 44.99, 0.80), func(o *provision.Offer, _ *hostfit.Host) { r := 0.5; o.Reliability = &r }),
		mk(provision.Vast, "4090", "RTX 4090", 24, 0.3), // below the start tier: ignored
	}
	rec = recommend(t, Query{Providers: []provision.Kind{provision.Vast}, Model: &hostfit.ModelRef{Arch: "hidream"}})
	want = fmt.Sprintf("No offer from 48 to 180 GB has fast internet, enough disk (%d GB) and a driver for CUDA 13.0 right now. "+
		"Most were left out for an old driver (2). Pick one yourself from the list.", rec.Need.DiskGB)
	if rec.StartTierGB != 48 || rec.Message != want {
		t.Errorf("start %d message\n got %q\nwant %q", rec.StartTierGB, rec.Message, want)
	}
	if rec.Tiers[0].TierGB != 48 || rec.Tiers[0].Offers != 3 || rec.Tiers[0].Rejected[RejectReliability] != 1 {
		t.Errorf("tiers[0] %+v", rec.Tiers[0])
	}

	// zero offers
	f.offers[provision.Vast] = nil
	rec = recommend(t, Query{Providers: []provision.Kind{provision.Vast}, DiskGB: 2000})
	if want := "No offers match these filters (disk 2000 GB). Change the filters or pick a provider with more stock."; rec.Message != want {
		t.Errorf("zero offers: %q", rec.Message)
	}
	rec = recommend(t, Query{Providers: []provision.Kind{provision.Vast}, DiskGB: 2000, MaxPrice: 0.5})
	if want := "No offers match these filters (disk 2000 GB, max $0.50/h). Change the filters or pick a provider with more stock."; rec.Message != want {
		t.Errorf("zero offers with a price cap: %q", rec.Message)
	}
}

func TestRecommendImageNotPublished(t *testing.T) {
	f := useFake(t, &fakeManager{image: imagetest.Options{Unpublished: true}}, provision.Vast)
	f.offers[provision.Vast] = []provision.Offer{mk(provision.Vast, "v", "RTX 4090", 24, 0.4)}

	rec := recommend(t, Query{})
	if rec.Value == nil || rec.Speed == nil {
		t.Fatalf("picks %+v %+v, want picks from the fallback requirements", rec.Value, rec.Speed)
	}
	n := hasNote(rec, "", "image_denied")
	if n == nil || !strings.Contains(n.Message, "isn't published yet") {
		t.Fatalf("notes %+v, want image_denied", rec.Notes)
	}
	if rec.Message != n.Message {
		t.Errorf("message %q, want the image sentence %q", rec.Message, n.Message)
	}
	if rec.Need.ImagePeakGB != hostfit.DefaultImagePeakGB || rec.Need.DiskGB < hostfit.MinDiskGBUnknownModel {
		t.Errorf("need %+v, want the fallback image peak", rec.Need)
	}
	if !strings.HasSuffix(rec.Value.Reason, "CUDA 13.2 OK") {
		t.Errorf("reason %q", rec.Value.Reason)
	}

	// nothing qualifies: the image sentence comes first
	f.offers[provision.Vast] = nil
	rec = recommend(t, Query{})
	if !strings.HasPrefix(rec.Message, n.Message+" No offers match these filters") {
		t.Errorf("message %q", rec.Message)
	}
}

func TestRecommendWithoutModelNeedsADefaultDisk(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.Vast)
	small := 50
	f.offers[provision.Vast] = []provision.Offer{with(mk(provision.Vast, "v", "RTX 4090", 24, 0.4), func(o *provision.Offer, _ *hostfit.Host) { o.MaxDiskGB = &small })}
	rec := recommend(t, Query{Providers: []provision.Kind{provision.Vast}})
	if rec.Need.DiskGB < 100 || rec.Need.ModelSource != "default" {
		t.Errorf("need %+v", rec.Need)
	}
	if rec.Value != nil || rec.Tiers[0].Rejected[RejectDisk] != 1 {
		t.Errorf("value %+v tiers[0] %+v", rec.Value, rec.Tiers[0])
	}
	if rec.Model != nil || rec.StartTierGB != 24 {
		t.Errorf("model %+v start %d", rec.Model, rec.StartTierGB)
	}
}

func TestRecommendUnknownModel(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.Vast)
	f.offers[provision.Vast] = []provision.Offer{mk(provision.Vast, "v", "RTX 4090", 24, 0.4)}
	rec := recommend(t, Query{Providers: []provision.Kind{provision.Vast}, Model: &hostfit.ModelRef{Arch: "brand_new"}})
	if rec.StartTierGB != 24 || rec.Value == nil || hasNote(rec, "", NoteModelUnknown) == nil {
		t.Errorf("start %d value %+v notes %+v", rec.StartTierGB, rec.Value, rec.Notes)
	}
}

func TestRecommendUndecoratedOffers(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.Vast)
	f.offers[provision.Vast] = []provision.Offer{with(mk(provision.Vast, "v", "RTX 4090", 24, 0.4), func(o *provision.Offer, _ *hostfit.Host) { o.Fit = nil })}
	rec := recommend(t, Query{Providers: []provision.Kind{provision.Vast}})
	if rec.Value != nil || rec.Tiers[0].Rejected[RejectFitBlock] != 1 {
		t.Errorf("value %+v tiers %+v", rec.Value, rec.Tiers[0])
	}
	if n := hasNote(rec, provision.Vast, NoteNotDecorated); n == nil || n.Offers != 1 {
		t.Errorf("notes %+v", rec.Notes)
	}
	if !strings.Contains(rec.Message, "not runnable (1)") {
		t.Errorf("message %q", rec.Message)
	}
}

func TestRecommendJSONHasNoNulls(t *testing.T) {
	useFake(t, &fakeManager{}, provision.QuickPod)
	rec := recommend(t, Query{})
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	for _, k := range []string{"providers", "tiers", "notes"} {
		if _, ok := m[k].([]any); !ok {
			t.Errorf("%s = %v, want an array", k, m[k])
		}
	}
	if _, ok := m["thresholds"].(map[string]any)["ladder"].([]any); !ok {
		t.Errorf("ladder = %v", m["thresholds"])
	}
	for _, tr := range m["tiers"].([]any) {
		if _, ok := tr.(map[string]any)["rejected"].(map[string]any); !ok {
			t.Errorf("tier %v: rejected is not an object", tr)
		}
	}
	for _, k := range []string{"value", "speed", "model"} {
		if v, ok := m[k]; !ok || v != nil {
			t.Errorf("%s = %v, want null", k, v)
		}
	}
	if m["default"] != "" || m["tier_gb"] != 0.0 {
		t.Errorf("default %v tier_gb %v", m["default"], m["tier_gb"])
	}
}

func TestRecommendIsDeterministic(t *testing.T) {
	f := useFake(t, &fakeManager{}, provision.RunPod, provision.Vast, provision.QuickPod)
	f.offers[provision.Vast] = []provision.Offer{mk(provision.Vast, "v1", "RTX 4090", 24, 0.40), mk(provision.Vast, "v2", "RTX 3090", 24, 0.22)}
	f.offers[provision.QuickPod] = []provision.Offer{mk(provision.QuickPod, "q1", "RTX 4090", 24, 0.40), mk(provision.QuickPod, "q2", "RTX 3090", 24, 0.22)}
	f.offers[provision.RunPod] = []provision.Offer{mk(provision.RunPod, "r1", "RTX 4090", 24, 0.40)}
	first := recommend(t, Query{})
	if first.Value.Offer.ID != "v2" || first.Speed.Offer.ID != "v1" {
		t.Fatalf("value %s speed %s, want the Vast offers (provider order before QuickPod; verified before RunPod)", first.Value.Offer.ID, first.Speed.Offer.ID)
	}
	for range 10 {
		for _, k := range []provision.Kind{provision.Vast, provision.QuickPod} {
			slices.Reverse(f.offers[k])
		}
		rec := recommend(t, Query{})
		if rec.Value.Offer.ID != first.Value.Offer.ID || rec.Speed.Offer.ID != first.Speed.Offer.ID {
			t.Fatalf("value %s speed %s", rec.Value.Offer.ID, rec.Speed.Offer.ID)
		}
	}
}
