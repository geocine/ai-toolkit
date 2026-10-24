package hostfit

import (
	"context"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit/hfmock"
	"github.com/geocine/aitoolkit/internal/netguard"
)

type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *fakeClock) add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// newHF starts an hfmock and a sizer wired to it on a fake clock.
func newHF(t *testing.T, o hfmock.Options) (*hfmock.Server, *ModelSizer, *fakeClock) {
	t.Helper()
	srv := hfmock.New(o)
	srv.Start()
	t.Cleanup(srv.Close)
	clk := &fakeClock{t: time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC)}
	s := NewSizer(SizerOptions{Base: srv.URL(), HTTP: netguard.NewClient(netguard.Options{AllowLoopback: true}), Now: clk.now})
	return srv, s, clk
}

func TestModelSizer(t *testing.T) {
	_, s, _ := newHF(t, hfmock.Options{})
	ctx := t.Context()
	cases := []struct {
		m    ModelRef
		want float64
	}{
		// The diffusers subfolders only: the root flux1-dev.safetensors is not fetched.
		{ModelRef{Arch: "flux"}, 33.7},
		// Root transformer + ae, plus the Mistral shards (not consolidated.safetensors).
		{ModelRef{Arch: "flux2"}, 112.8},
		{ModelRef{Arch: "qwen_image"}, 57.7},
		{ModelRef{Arch: "wan22_14b"}, 126.2},
		{ModelRef{Arch: "zimage"}, 32.8},
		// The job's own repo wins over the default.
		{ModelRef{Arch: "qwen_image_edit", NameOrPath: "Qwen/Qwen-Image"}, 57.7},
	}
	for _, c := range cases {
		gb, src := s.Size(ctx, c.m)
		if gb != c.want || src != "hf" {
			t.Errorf("%+v: %v GB from %s, want %v from hf", c.m, gb, src, c.want)
		}
	}
	// A local path cannot be listed: the table.
	if gb, src := s.Size(ctx, ModelRef{Arch: "flux", NameOrPath: "/models/flux1"}); gb != 33.7 || src != "table" {
		t.Errorf("local path: %v %s", gb, src)
	}
	// An unknown arch sums its own repo's root safetensors, else none.
	if gb, src := s.Size(ctx, ModelRef{Arch: "mystery", NameOrPath: "black-forest-labs/FLUX.1-dev"}); gb != 24.1 || src != "hf" {
		t.Errorf("unknown arch on an HF repo: %v %s", gb, src)
	}
	if gb, src := s.Size(ctx, ModelRef{Arch: "mystery"}); gb != 0 || src != "none" {
		t.Errorf("unknown arch: %v %s", gb, src)
	}
	// A repo the API does not know falls back to the table.
	if gb, src := s.Size(ctx, ModelRef{Arch: "wan21"}); gb != 80.4 || src != "table" {
		t.Errorf("unlisted repo: %v %s", gb, src)
	}
	// A nil sizer is the table.
	if gb, src := (*ModelSizer)(nil).Size(ctx, ModelRef{Arch: "flux2"}); gb != 112.8 || src != "table" {
		t.Errorf("nil sizer: %v %s", gb, src)
	}
}

func TestModelSizerDownAndCache(t *testing.T) {
	srv, s, clk := newHF(t, hfmock.Options{Faults: hfmock.Faults{Status: map[string]int{"black-forest-labs/FLUX.1-dev": 429}}})
	ctx := t.Context()
	if gb, src := s.Size(ctx, ModelRef{Arch: "flux"}); gb != 33.7 || src != "table" {
		t.Errorf("HF 429: %v %s", gb, src)
	}
	// A failure is retried only after its short TTL.
	s.Size(ctx, ModelRef{Arch: "flux"})
	if n := countPath(srv.Requests(), "FLUX.1-dev"); n != 1 {
		t.Errorf("failed tree retried at once: %d calls", n)
	}
	clk.add(6 * time.Minute)
	s.Size(ctx, ModelRef{Arch: "flux"})
	if n := countPath(srv.Requests(), "FLUX.1-dev"); n != 2 {
		t.Errorf("failed tree not retried after its TTL: %d calls", n)
	}
	for range 3 {
		s.Size(ctx, ModelRef{Arch: "qwen_image"})
	}
	if n := countPath(srv.Requests(), "Qwen/Qwen-Image"); n != 1 {
		t.Errorf("%d tree calls in a day, want 1", n)
	}
	clk.add(23 * time.Hour)
	s.Size(ctx, ModelRef{Arch: "qwen_image"})
	if n := countPath(srv.Requests(), "Qwen/Qwen-Image"); n != 1 {
		t.Errorf("cache expired early: %d calls", n)
	}
	clk.add(2 * time.Hour)
	s.Size(ctx, ModelRef{Arch: "qwen_image"})
	if n := countPath(srv.Requests(), "Qwen/Qwen-Image"); n != 2 {
		t.Errorf("after 25 h: %d calls, want 2", n)
	}
}

// A blackholed Hugging Face runs into the caller's deadline: that counts
// as the repo failing (cached for the short TTL), so the next NeedFor does
// not stall on it again. A cancelled caller is not cached.
func TestModelSizerDeadlineCached(t *testing.T) {
	srv, s, _ := newHF(t, hfmock.Options{Faults: hfmock.Faults{SlowMs: 5000}})
	size := func() (float64, string) {
		ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
		defer cancel()
		return s.Size(ctx, ModelRef{Arch: "flux"})
	}
	if gb, src := size(); gb != 33.7 || src != "table" {
		t.Errorf("timed out: %v %s", gb, src)
	}
	start := time.Now()
	if gb, src := size(); gb != 33.7 || src != "table" || time.Since(start) > 50*time.Millisecond {
		t.Errorf("second call: %v %s after %v", gb, src, time.Since(start))
	}
	if n := countPath(srv.Requests(), "FLUX.1-dev"); n != 1 {
		t.Errorf("timed-out tree asked %d times, want 1", n)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	s.Size(ctx, ModelRef{Arch: "qwen_image"})
	if _, cached := s.trees["Qwen/Qwen-Image"]; cached {
		t.Error("a cancelled caller was cached as a repo failure")
	}
}

func TestModelSizerPages(t *testing.T) {
	srv, s, _ := newHF(t, hfmock.Options{Faults: hfmock.Faults{PageSize: 5}})
	if gb, src := s.Size(t.Context(), ModelRef{Arch: "flux"}); gb != 33.7 || src != "hf" {
		t.Errorf("paged listing: %v %s", gb, src)
	}
	if n := countPath(srv.Requests(), "FLUX.1-dev"); n < 5 {
		t.Errorf("only %d pages read", n)
	}
	// A next page on another host is not followed.
	base, _ := url.Parse(srv.URL())
	if got := nextPage(`<https://evil.example/api/models/x/tree/main?cursor=5>; rel="next"`, base); got != "" {
		t.Errorf("followed %q", got)
	}
}

func TestInitSizer(t *testing.T) {
	t.Cleanup(func() { InitSizer(false); DefaultSizer = nil })
	InitSizer(true)
	first := DefaultSizer
	if gb, src := DefaultSizer.Size(t.Context(), ModelRef{Arch: "flux"}); gb != 33.7 || src != "hf" {
		t.Errorf("simulated sizer: %v %s", gb, src)
	}
	InitSizer(true)
	if DefaultSizer == first {
		t.Error("InitSizer did not replace the sizer")
	}
	if gb, src := DefaultSizer.Size(t.Context(), ModelRef{Arch: "flux2"}); gb != 112.8 || src != "hf" {
		t.Errorf("second simulated sizer: %v %s", gb, src)
	}
	InitSizer(false)
	if DefaultSizer == nil || DefaultSizer.base != hfBase {
		t.Errorf("real sizer base %q", DefaultSizer.base)
	}
}

func countPath(paths []string, part string) int {
	n := 0
	for _, p := range paths {
		if strings.Contains(p, part) {
			n++
		}
	}
	return n
}
