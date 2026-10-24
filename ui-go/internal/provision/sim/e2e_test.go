package sim

import (
	"context"
	"errors"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/imagetest"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/quickpod/quickpodmock"
	"github.com/geocine/aitoolkit/internal/provision/vast"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
	"github.com/geocine/aitoolkit/internal/v1api"
)

var kinds = []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod}

// simEnv is simulate mode on a FakeClock with no delays, through
// provision.Default and the real adapters, mocks, nodes and prober.
type simEnv struct {
	t     *testing.T
	ctx   context.Context
	clock *mockkit.FakeClock
	svc   *provision.Service
}

func newSim(t *testing.T) *simEnv {
	t.Helper()
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close() })
	for _, k := range []string{"AITK_PROVISION_PROVIDERS", "AITK_PROVISION_MAX_RUNNING", "AITK_PROVISION_READY_TIMEOUT",
		"RUNPOD_POD_ID", "CONTAINER_ID", "CONTAINER_LABEL", "AI_TOOLKIT_AUTH"} {
		t.Setenv(k, "")
	}
	for _, k := range kinds {
		t.Setenv(provision.EnvKeyName(k), "")
	}
	clock := mockkit.NewFakeClock(time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC))
	stop, err := start(context.Background(), stubInner(), clock, mockkit.Delays{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(stop)
	oldPace := vast.Pace
	vast.Pace = vast.NewLimiter(&paceClock{t: clock.Now()})
	t.Cleanup(func() { vast.Pace = oldPace })
	svc, ok := provision.Default.(*provision.Service)
	if !ok {
		t.Fatalf("provision.Default is %T", provision.Default)
	}
	oldNow := svc.Now
	svc.Now = clock.Now
	t.Cleanup(func() { svc.Now = oldNow })
	t.Cleanup(svc.WaitChecks) // before the database closes
	return &simEnv{t: t, ctx: context.Background(), clock: clock, svc: svc}
}

// paceClock lets the Vast limiter space its calls without waiting for real.
type paceClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *paceClock) Now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *paceClock) Sleep(_ context.Context, d time.Duration) error {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
	return nil
}

func (e *simEnv) setKey(k provision.Kind, key string) error {
	_, err := e.svc.SetKey(e.ctx, k, provision.ProviderUpdate{APIKey: &key})
	return err
}

// launch sets key (a fault name or "") and launches the first offer.
func (e *simEnv) launch(k provision.Kind, key string) (provision.Machine, error) {
	e.t.Helper()
	if err := e.setKey(k, key); err != nil {
		e.t.Fatalf("%s SetKey(%q): %v", k, key, err)
	}
	offers, err := e.svc.Offers(e.ctx, k, provision.OfferQuery{})
	if err != nil || len(offers) == 0 {
		e.t.Fatalf("%s offers: %d %v", k, len(offers), err)
	}
	o := offers[0]
	req := provision.LaunchRequest{Provider: k, OfferID: o.ID, MaxPricePerHour: o.PricePerHour + 1}
	if k == provision.QuickPod {
		ts, err := e.svc.Templates(e.ctx, k)
		if err != nil {
			e.t.Fatal(err)
		}
		i := slices.IndexFunc(ts, func(t provision.Template) bool { return t.Ours })
		if i < 0 {
			e.t.Fatalf("no aitk template among %+v", ts)
		}
		req.TemplateID = ts[i].ID
	}
	m, _, err := e.svc.Launch(e.ctx, req, "")
	return m, err
}

func (e *simEnv) mustLaunch(k provision.Kind, key string) provision.Machine {
	e.t.Helper()
	m, err := e.launch(k, key)
	if err != nil {
		e.t.Fatalf("%s launch: %v", k, err)
	}
	if !m.Simulated {
		e.t.Fatalf("%s machine is not flagged simulated", k)
	}
	return m
}

// until ticks the watcher 10 s of FakeClock at a time until ok holds.
func (e *simEnv) until(id string, steps int, what string, ok func(provision.Machine) bool) provision.Machine {
	e.t.Helper()
	var m provision.Machine
	for range steps {
		e.clock.Advance(10 * time.Second)
		e.svc.Tick(e.ctx)
		e.svc.WaitChecks() // a machine's CUDA check runs in the background
		var err error
		if m, err = e.svc.Get(e.ctx, id, false); err != nil {
			e.t.Fatal(err)
		}
		if ok(m) {
			return m
		}
	}
	e.t.Fatalf("%s %s: not %s after %d ticks: status %s problem %v detail %q", m.Provider, id, what, steps, m.Status, deref(m.Problem), m.Detail)
	return m
}

func deref(p *provision.Problem) string {
	if p == nil {
		return ""
	}
	return string(*p)
}

func deref2(s *string) string {
	if s == nil {
		return "<nil>"
	}
	return *s
}

func ready(m provision.Machine) bool { return m.Status == provision.StatusReady && m.Problem == nil }

func status(s provision.Status) func(provision.Machine) bool {
	return func(m provision.Machine) bool { return m.Status == s }
}

func problem(p provision.Problem) func(provision.Machine) bool {
	return func(m provision.Machine) bool { return m.Problem != nil && *m.Problem == p }
}

func code(err error) string {
	if c, _, _, ok := provision.V1(err); ok {
		return c
	}
	return ""
}

// systemStatus is GET <base>/v1/system with token.
func systemStatus(t *testing.T, base, token string) int {
	t.Helper()
	res, _ := get(t, base+"/v1/system", map[string]string{"Authorization": "Bearer " + token})
	return res.StatusCode
}

func TestSimLifecycle(t *testing.T) {
	for _, k := range kinds {
		t.Run(string(k), func(t *testing.T) {
			e := newSim(t)
			m := e.mustLaunch(k, "")
			m = e.until(m.ID, 6, "ready", ready)
			if m.BaseURL == nil || !strings.HasPrefix(*m.BaseURL, "http://127.0.0.1:") {
				t.Fatalf("base_url %v", m.BaseURL)
			}

			link, err := e.svc.Link(e.ctx, m.ID)
			if err != nil {
				t.Fatal(err)
			}
			if got := systemStatus(t, link.BaseURL, link.Token); got != 200 {
				t.Fatalf("link token on the node: %d", got)
			}
			if got := systemStatus(t, link.BaseURL, "aitk_wrong"); got != 401 {
				t.Fatalf("wrong token on the node: %d", got)
			}
			if _, err := e.svc.Logs(e.ctx, m.ID, 50); err != nil {
				t.Fatalf("logs: %v", err)
			}

			// stop closes the node's listener; start opens a new one
			oldBase := link.BaseURL
			if _, err := e.svc.Stop(e.ctx, m.ID); err != nil {
				t.Fatal(err)
			}
			e.until(m.ID, 12, "stopped", status(provision.StatusStopped))
			if _, err := http.Get(oldBase + "/v1/health"); err == nil {
				t.Fatal("the stopped machine's node still answers")
			}
			if _, err := e.svc.Start(e.ctx, m.ID); err != nil {
				t.Fatal(err)
			}
			m = e.until(m.ID, 12, "ready again", ready)
			if m.BaseURL == nil || *m.BaseURL == oldBase {
				t.Fatalf("restarted base_url %v, was %s", m.BaseURL, oldBase)
			}
			link, _ = e.svc.Link(e.ctx, m.ID)
			if got := systemStatus(t, link.BaseURL, link.Token); got != 200 {
				t.Fatalf("restarted node: %d", got)
			}

			if _, err := e.svc.Terminate(e.ctx, m.ID, false); err != nil {
				t.Fatal(err)
			}
			e.until(m.ID, 12, "terminated", status(provision.StatusTerminated))
			if _, err := http.Get(link.BaseURL + "/v1/health"); err == nil {
				t.Fatal("the terminated machine's node still answers")
			}
		})
	}
}

func TestSimKeyFaults(t *testing.T) {
	for _, k := range kinds {
		t.Run(string(k), func(t *testing.T) {
			e := newSim(t)
			err := e.setKey(k, "sim_bad")
			if code(err) != v1api.CodeProviderAuth || v1api.StatusOf(code(err)) != 400 {
				t.Fatalf("sim_bad: %v (code %q)", err, code(err))
			}
			if _, err := e.launch(k, "sim_full"); code(err) != v1api.CodeNoCapacity || v1api.StatusOf(code(err)) != 409 {
				t.Fatalf("sim_full: %v (code %q)", err, code(err))
			}
			if _, err := e.launch(k, "sim_broke"); code(err) != v1api.CodeInsufficientCredit || v1api.StatusOf(code(err)) != 402 {
				t.Fatalf("sim_broke: %v (code %q)", err, code(err))
			}
		})
	}
}

func TestSimMachineFaults(t *testing.T) {
	t.Run("vast noport", func(t *testing.T) {
		e := newSim(t)
		m := e.mustLaunch(provision.Vast, "sim_noport")
		m = e.until(m.ID, 25, "no_port", problem(provision.ProblemNoPort))
		if e.clock.Now().Sub(m.CreatedAt) < 3*time.Minute {
			t.Fatalf("no_port before the 3 min grace: %s", e.clock.Now().Sub(m.CreatedAt))
		}
	})
	t.Run("quickpod ignore", func(t *testing.T) {
		e := newSim(t)
		m := e.mustLaunch(provision.QuickPod, "sim_ignore")
		e.until(m.ID, 6, "token_mismatch", problem(provision.ProblemTokenMismatch))
		tok := quickpodmock.TemplateToken
		if _, err := e.svc.Update(e.ctx, m.ID, provision.MachinePatch{Token: &tok}); err != nil {
			t.Fatal(err)
		}
		m, _ = e.svc.Get(e.ctx, m.ID, false)
		if !ready(m) {
			m = e.until(m.ID, 3, "ready with the template token", ready)
		}
	})
	t.Run("quickpod replace", func(t *testing.T) {
		e := newSim(t)
		m := e.mustLaunch(provision.QuickPod, "sim_replace")
		e.until(m.ID, 25, "no_port", problem(provision.ProblemNoPort))
	})
	for _, k := range kinds {
		t.Run(string(k)+" open", func(t *testing.T) {
			e := newSim(t)
			m := e.mustLaunch(k, "sim_open")
			m = e.until(m.ID, 6, "open_server", problem(provision.ProblemOpenServer))
			if m.Status != provision.StatusReady {
				t.Fatalf("open server status %s", m.Status)
			}
		})
		t.Run(string(k)+" crash", func(t *testing.T) {
			e := newSim(t)
			m := e.mustLaunch(k, "sim_crash")
			e.until(m.ID, 12, "failed", status(provision.StatusFailed))
		})
	}
}

// Simulate never reaches a provider: the real hosts are refused, and each
// mock answers 401 to a key without this process's secret.
func TestSimRefusesRealProviders(t *testing.T) {
	newSim(t)
	if !provision.Simulated.Load() {
		t.Fatal("simulate did not set provision.Simulated")
	}
	_, err := provision.Do(context.Background(), provision.Call{Kind: provision.RunPod, Method: http.MethodGet, URL: "https://api.runpod.io/v2/pods"})
	if err == nil || !strings.Contains(err.Error(), "refusing to call api.runpod.io") {
		t.Fatalf("real RunPod host: %v", err)
	}
	for _, k := range kinds {
		c, ok := provision.Lookup(k)
		if !ok {
			t.Fatalf("%s is not registered", k)
		}
		for _, key := range []string{"not-the-secret", "not-the-secret.sim_full"} {
			if _, err := c.Verify(context.Background(), key); provision.ClassOf(err) != provision.ClassAuth {
				t.Fatalf("%s key %q: %v", k, key, err)
			}
		}
		if _, err := c.Verify(context.Background(), provision.WireKey("")); err != nil {
			t.Fatalf("%s with the sim secret: %v", k, err)
		}
	}
}

// stop puts the adapters back and leaves simulate mode.
func TestSimStopRestores(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stop, err := start(ctx, stubInner(), mockkit.RealClock{}, mockkit.Delays{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := start(ctx, stubInner(), mockkit.RealClock{}, mockkit.Delays{}); err == nil {
		t.Fatal("a second simulate session started")
	}
	cancel() // ctx ending stops it too
	stop()
	if provision.Simulated.Load() {
		t.Fatal("still simulated after stop")
	}
	stop, err = start(context.Background(), stubInner(), mockkit.RealClock{}, mockkit.Delays{})
	if err != nil {
		t.Fatalf("restart after stop: %v", err)
	}
	stop()
}

func TestDelays(t *testing.T) {
	for _, c := range []struct {
		env  string
		want mockkit.Delays
	}{
		{"", mockkit.Delays{Provision: 5 * time.Second, Start: 10 * time.Second, App: 4 * time.Second}},
		{"30s", mockkit.Delays{Provision: 10 * time.Second, Start: 20 * time.Second, App: 4 * time.Second}},
		{"9", mockkit.Delays{Provision: 3 * time.Second, Start: 6 * time.Second, App: 4 * time.Second}},
		{"0", mockkit.Delays{}},
		{"soon", mockkit.Delays{Provision: 5 * time.Second, Start: 10 * time.Second, App: 4 * time.Second}},
	} {
		t.Setenv("AITK_PROVISION_SIMULATE_DELAY", c.env)
		if got := Delays(); got != c.want {
			t.Errorf("%q: %+v, want %+v", c.env, got, c.want)
		}
	}
	for v, want := range map[string]bool{"1": true, "true": true, "TRUE": true, "0": false, "": false, "yes": false} {
		t.Setenv("AITK_PROVISION_SIMULATE", v)
		if Enabled() != want {
			t.Errorf("AITK_PROVISION_SIMULATE=%q: %v", v, !want)
		}
	}
}

// tripwire fails the test on any request through http.DefaultTransport to a
// host that is not loopback: simulate checks images against registrymock.
type tripwire struct {
	t    *testing.T
	next http.RoundTripper
}

func (w tripwire) RoundTrip(r *http.Request) (*http.Response, error) {
	if h := r.URL.Hostname(); h != "127.0.0.1" && h != "::1" && h != "localhost" {
		w.t.Errorf("simulate reached %s", r.URL.Host)
		return nil, errors.New("tripwire: " + r.URL.Host)
	}
	return w.next.RoundTrip(r)
}

func useTripwire(t *testing.T) {
	old := http.DefaultTransport
	http.DefaultTransport = tripwire{t, old}
	t.Cleanup(func() { http.DefaultTransport = old })
}

// Simulate reads the default image from the in-process registry: the image
// check passes, offers carry host and fit, and a launch pins the build and
// goes ready, with nothing leaving the machine.
func TestSimImageCheckOffline(t *testing.T) {
	useTripwire(t)
	e := newSim(t)
	if imagecheck.Default == nil {
		t.Fatal("simulate did not set the image check")
	}
	img, err := provision.ImageChecker().Inspect(e.ctx, provision.DefaultImage, "", false)
	if err != nil || img.Revision != registrymock.Revision || !img.GoServer {
		t.Fatalf("inspect %v: %+v", err, img)
	}
	if hostfit.DefaultSizer == nil {
		t.Fatal("simulate did not set the model sizer")
	}
	for _, k := range kinds {
		t.Run(string(k), func(t *testing.T) {
			if err := e.setKey(k, ""); err != nil {
				t.Fatal(err)
			}
			offers, err := e.svc.Offers(e.ctx, k, provision.OfferQuery{})
			if err != nil || len(offers) == 0 {
				t.Fatalf("offers %d %v", len(offers), err)
			}
			for _, o := range offers {
				if o.Host == nil || o.Fit == nil || o.Fit.Level == imagecheck.LevelBlock {
					t.Fatalf("offer %s host %+v fit %+v", o.ID, o.Host, o.Fit)
				}
			}
			m := e.mustLaunch(k, "")
			want := imagecheck.DefaultRepo + ":" + registrymock.Revision
			if k == provision.QuickPod {
				want = quickpodmock.AitkTemplateImage // the template's image, as it is
			}
			if m.ImageLaunchRef == nil || *m.ImageLaunchRef != want {
				t.Fatalf("launch ref %v, want %s", deref2(m.ImageLaunchRef), want)
			}
			if m.ImageDigest == nil {
				t.Fatal("no image digest")
			}
			m = e.until(m.ID, 6, "ready", ready)
			if len(m.Warnings) != 0 {
				t.Fatalf("warnings %v", m.Warnings)
			}
		})
	}
}

// Simulate lists the live-shaped hosts the image can't run, so Show all
// has refused rows to show, and QuickPod has a template pinned to a build.
func TestSimShowAllBlocks(t *testing.T) {
	useTripwire(t)
	e := newSim(t)
	for _, k := range []provision.Kind{provision.Vast, provision.QuickPod} {
		t.Run(string(k), func(t *testing.T) {
			if err := e.setKey(k, ""); err != nil {
				t.Fatal(err)
			}
			blocks := func(q provision.OfferQuery) int {
				offers, err := e.svc.Offers(e.ctx, k, q)
				if err != nil || len(offers) == 0 {
					t.Fatalf("offers %d %v", len(offers), err)
				}
				n := 0
				for _, o := range offers {
					if o.Fit != nil && o.Fit.Level == imagecheck.LevelBlock {
						n++
					}
				}
				return n
			}
			if n := blocks(provision.OfferQuery{}); n != 0 {
				t.Fatalf("%d blocked offers without show_all", n)
			}
			if n := blocks(provision.OfferQuery{ShowAll: true, Limit: 100}); n == 0 {
				t.Fatal("show_all lists no offer the image blocks")
			}
		})
	}
	ts, err := e.svc.Templates(e.ctx, provision.QuickPod)
	if err != nil {
		t.Fatal(err)
	}
	pinned := imagecheck.DefaultRepo + ":" + registrymock.Revision
	if !slices.ContainsFunc(ts, func(t provision.Template) bool { return t.ID == quickpodmock.PinnedTemplateUUID && t.Image == pinned }) {
		t.Fatalf("no template pinned to %s among %+v", pinned, ts)
	}
}

func TestSimImageFaults(t *testing.T) {
	for _, k := range kinds {
		t.Run(string(k)+" nokernel", func(t *testing.T) {
			e := newSim(t)
			m := e.mustLaunch(k, "sim_nokernel")
			m = e.until(m.ID, 6, "cuda_failed", problem(provision.ProblemCUDAFailed))
			if m.Status != provision.StatusReady || !strings.Contains(m.Detail, "PyTorch build") {
				t.Fatalf("%s %q", m.Status, m.Detail)
			}
		})
	}
	t.Run("runpod olddriver", func(t *testing.T) {
		e := newSim(t)
		m := e.mustLaunch(provision.RunPod, "sim_olddriver")
		m = e.until(m.ID, 6, "cuda_failed", problem(provision.ProblemCUDAFailed))
		if !strings.Contains(m.Detail, "driver is too old") {
			t.Fatalf("detail %q", m.Detail)
		}
	})
	t.Run("runpod stale", func(t *testing.T) {
		e := newSim(t)
		m := e.mustLaunch(provision.RunPod, "sim_stale")
		m = e.until(m.ID, 6, "ready", ready)
		if !slices.ContainsFunc(m.Warnings, func(w string) bool {
			return strings.Contains(w, "running build "+StaleVersion) && strings.Contains(w, registrymock.Revision)
		}) {
			t.Fatalf("warnings %v", m.Warnings)
		}
	})
	t.Run("runpod pull denied", func(t *testing.T) {
		e := newSim(t)
		active.rp.PullFailText = "denied: requested access to the resource is denied"
		m := e.mustLaunch(provision.RunPod, "")
		m = e.until(m.ID, 6, "image_pull_failed", problem(provision.ProblemImagePullFailed))
		if m.Status != provision.StatusFailed || !strings.Contains(m.Detail, "isn't published yet") {
			t.Fatalf("%s %q", m.Status, m.Detail)
		}
	})
	t.Run("runpod manifest unknown", func(t *testing.T) {
		e := newSim(t)
		active.rp.PullFailText = "manifest unknown"
		m := e.mustLaunch(provision.RunPod, "")
		m = e.until(m.ID, 6, "image_pull_failed", problem(provision.ProblemImagePullFailed))
		if !strings.Contains(m.Detail, "couldn't find the image tag") {
			t.Fatalf("detail %q", m.Detail)
		}
	})
}

// Before the default image is published (D-1, D-17), offers are still
// judged from its built-in requirements and launches are refused.
func TestSimImageUnpublished(t *testing.T) {
	e := newSim(t)
	imagetest.Use(t, imagetest.Options{Unpublished: true})
	if err := e.setKey(provision.Vast, ""); err != nil {
		t.Fatal(err)
	}
	offers, err := e.svc.Offers(e.ctx, provision.Vast, provision.OfferQuery{})
	if err != nil || len(offers) == 0 {
		t.Fatalf("offers %d %v", len(offers), err)
	}
	for _, o := range offers {
		if o.Host == nil || o.Fit == nil || slices.Contains(o.Fit.Codes, "image_denied") {
			t.Fatalf("offer %s fit %+v", o.ID, o.Fit)
		}
	}
	_, err = e.launch(provision.Vast, "")
	var b *provision.ImageBlockedError
	if !errors.As(err, &b) || !strings.Contains(b.Message, "isn't published yet") {
		t.Fatalf("launch: %v", err)
	}
	if n := len(active.vm.Instances()); n != 0 {
		t.Fatalf("%d instances created", n)
	}
}
