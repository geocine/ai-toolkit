package mockkit

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestPhase(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	steps := []time.Duration{3 * time.Second, 6 * time.Second, 4 * time.Second}
	for _, c := range []struct {
		after time.Duration
		want  int
	}{{0, 0}, {2999 * time.Millisecond, 0}, {3 * time.Second, 1}, {8 * time.Second, 1}, {9 * time.Second, 2}, {13 * time.Second, 3}, {time.Hour, 3}} {
		if got := Phase(t0.Add(c.after), t0, steps...); got != c.want {
			t.Errorf("after %v: phase %d, want %d", c.after, got, c.want)
		}
	}
	if Phase(t0, t0) != 0 {
		t.Error("no steps is phase 0")
	}
	// zero delays (FakeClock tests) pass every step at once
	if Phase(t0, t0, 0, 0) != 2 {
		t.Error("zero steps elapse immediately")
	}
}

func TestFakeClock(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	c := NewFakeClock(t0)
	c.Advance(90 * time.Second)
	if !c.Now().Equal(t0.Add(90 * time.Second)) {
		t.Fatalf("now %v", c.Now())
	}
	var o Options
	o.Defaults()
	if _, ok := o.Clock.(RealClock); !ok || o.Credit != 25 {
		t.Fatalf("defaults %+v", o)
	}
}

func TestAuthorize(t *testing.T) {
	sim := Options{SimSecret: "S"}
	if f, ok := sim.Authorize("S"); !ok || f != (Faults{}) {
		t.Errorf("plain secret: %+v %v", f, ok)
	}
	if f, ok := sim.Authorize("S.sim_full"); !ok || !f.NoCapacity {
		t.Errorf("fault suffix: %+v %v", f, ok)
	}
	if _, ok := sim.Authorize("S.sim_bad"); ok {
		t.Error("sim_bad must be refused")
	}
	if f, ok := sim.Authorize("S.whatever"); !ok || f != (Faults{}) {
		t.Errorf("an unknown suffix is a plain key: %+v %v", f, ok)
	}
	for _, k := range []string{"", "T", "T.sim_full", "sim_full", "SX"} {
		if _, ok := sim.Authorize(k); ok {
			t.Errorf("simulate must refuse %q", k)
		}
	}

	keys := Options{Keys: []string{"good"}}
	if _, ok := keys.Authorize("good"); !ok {
		t.Error("listed key")
	}
	if _, ok := keys.Authorize("other"); ok {
		t.Error("unlisted key")
	}
	anyKey := Options{Faults: Faults{Slow: true}}
	if f, ok := anyKey.Authorize("x"); !ok || !f.Slow {
		t.Errorf("nil Keys accepts any key and keeps test faults: %+v %v", f, ok)
	}
	if _, ok := anyKey.Authorize(""); ok {
		t.Error("the empty key is never accepted")
	}
	if _, ok := (Options{Faults: Faults{Unauthorized: true}}).Authorize("x"); ok {
		t.Error("Faults.Unauthorized refuses every key")
	}
}

// Every fault the later units read has a key a user can type in simulate.
func TestFaultNames(t *testing.T) {
	sim := Options{SimSecret: "S"}
	check := func(key string, want func(Faults) bool) {
		t.Helper()
		f, _ := sim.Authorize("S." + key)
		if !want(f) {
			t.Errorf("%s: %+v", key, f)
		}
	}
	check("sim_broke", func(f Faults) bool { return f.NoCredit })
	check("sim_429", func(f Faults) bool { return f.RateLimitEvery == 3 })
	check("sim_crash", func(f Faults) bool { return f.CrashAfterRunning })
	check("sim_noport", func(f Faults) bool { return f.NoPortMapping })
	check("sim_open", func(f Faults) bool { return f.OpenNode })
	check("sim_slow", func(f Faults) bool { return f.Slow && f.Scale(time.Second) == 10*time.Second })
	check("sim_replace", func(f Faults) bool { return f.PodEnv == PodEnvReplace })
	check("sim_ignore", func(f Faults) bool { return f.PodEnv == PodEnvIgnore })
	check("sim_gone", func(f Faults) bool { return f.GoneAsStatus })
	check("sim_nokernel", func(f Faults) bool { return f.CUDA == CUDANoKernel })
	check("sim_olddriver", func(f Faults) bool { return f.CUDA == CUDAOldDriver })
	check("sim_stale", func(f Faults) bool { return f.StaleImage })

	names := FaultNames()
	if !slices.IsSorted(names) || len(names) != 14 || !slices.Contains(names, "sim_stale") {
		t.Errorf("FaultNames: %v", names)
	}
	if (Faults{}).Scale(time.Second) != time.Second {
		t.Error("Scale without Slow")
	}
}

// Every node-side fault reaches the simulated node through NodeSpec, and the
// launch ref travels with it (sim/node.go sees nothing else).
func TestNodeSpecWithFaults(t *testing.T) {
	sim := Options{SimSecret: "S"}
	base := NodeSpec{Kind: "vast", InstanceID: "1", GPU: "RTX 4090", GPUCount: 1, VRAMGB: 24, Image: "ghcr.io/geocine/aitoolkit-go:f53a407"}
	for _, c := range []struct {
		key  string
		want NodeSpec
	}{
		{"S", base},
		{"S.sim_open", func() NodeSpec { n := base; n.OpenNode = true; return n }()},
		{"S.sim_nokernel", func() NodeSpec { n := base; n.CUDA = CUDANoKernel; return n }()},
		{"S.sim_olddriver", func() NodeSpec { n := base; n.CUDA = CUDAOldDriver; return n }()},
		{"S.sim_stale", func() NodeSpec { n := base; n.StaleImage = true; return n }()},
	} {
		f, ok := sim.Authorize(c.key)
		if !ok {
			t.Fatalf("%s refused", c.key)
		}
		got := base.WithFaults(f)
		if got.OpenNode != c.want.OpenNode || got.CUDA != c.want.CUDA || got.StaleImage != c.want.StaleImage ||
			got.Image != base.Image || got.Kind != base.Kind || got.VRAMGB != base.VRAMGB {
			t.Errorf("%s: %+v, want %+v", c.key, got, c.want)
		}
	}
	// a clean key clears faults a reused spec carried
	dirty := NodeSpec{OpenNode: true, CUDA: CUDANoKernel, StaleImage: true}
	if got := dirty.WithFaults(Faults{}); got.OpenNode || got.CUDA != CUDAOK || got.StaleImage {
		t.Errorf("clean faults: %+v", got)
	}
}

func TestRecorderAndWriteJSON(t *testing.T) {
	var rec Recorder
	h := rec.Wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 16)
		n, _ := r.Body.Read(buf)
		WriteJSON(w, http.StatusCreated, map[string]string{"echo": string(buf[:n])}, "")
	}))
	req := httptest.NewRequest("POST", "/v2/pods?x=1", strings.NewReader("hello"))
	req.Header.Set("Authorization", "Bearer k")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 201 || w.Header().Get("Content-Type") != "application/json" || strings.TrimSpace(w.Body.String()) != `{"echo":"hello"}` {
		t.Fatalf("response %d %v %s", w.Code, w.Header(), w.Body)
	}
	got := rec.Requests()
	if len(got) != 1 || got[0].Method != "POST" || got[0].Path != "/v2/pods" || got[0].RawQuery != "x=1" ||
		string(got[0].Body) != "hello" || got[0].Header.Get("Authorization") != "Bearer k" {
		t.Fatalf("recorded %+v", got)
	}

	w = httptest.NewRecorder()
	WriteJSON(w, 200, []int{1}, "text/html") // a mock may imitate a provider's content type
	if w.Header().Get("Content-Type") != "text/html" {
		t.Fatalf("content type %q", w.Header().Get("Content-Type"))
	}
}
