package provision

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/notify"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// ---- a scriptable adapter and prober ----

// stubCloud is an in-memory provider: Launch creates an instance the test
// then moves through its states with set.
type stubCloud struct {
	mu        sync.Mutex
	kind      Kind
	caps      Caps
	verifyErr error
	offers    []Offer
	offersErr error
	quote     Offer
	quoteErr  error
	templates []Template
	launchErr error
	listErr   error
	getErr    error
	actErr    error
	logs      string
	noID      bool // QuickPod: create returns no id; the pod is found by label
	insts     map[string]*Instance
	launched  []LaunchSpec
	keys      []string
	calls     map[string]int
	next      int
}

func newStub(k Kind) *stubCloud {
	c := &stubCloud{kind: k, insts: map[string]*Instance{}, calls: map[string]int{}}
	switch k {
	case RunPod:
		c.caps = Caps{CloudChoice: true, CountChoice: true, StopStart: true, HTTPS: "yes", SelfDestroy: "key", EnvCountMax: 50, EnvTotalMax: 100000}
	case Vast:
		c.caps = Caps{StopStart: true, StopKeepsDisk: true, HTTPS: "no", SelfDestroy: "builtin", EnvValueMax: 256, EnvTotalMax: 32768}
	case QuickPod:
		c.caps = Caps{TemplateRequired: true, StopStart: true, StopKeepsDisk: true, HTTPS: "unconfirmed", SelfDestroy: "key", EnvFlagString: true}
		c.templates = []Template{{ID: "tpl-aitk", Name: "aitk", Ours: true, PublishesPort: true}}
	}
	c.quote = Offer{ID: "offer-1", Provider: k, GPU: "RTX 4090", GPUCount: 1, VRAMGB: 24, PricePerHour: 0.45, Location: "EU"}
	c.offers = []Offer{c.quote}
	return c
}

func (c *stubCloud) count(name, key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls[name]++
	c.keys = append(c.keys, key)
}

func (c *stubCloud) n(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls[name]
}

func (c *stubCloud) Kind() Kind { return c.kind }
func (c *stubCloud) Info() KindInfo {
	return KindInfo{Kind: c.kind, Label: Label(c.kind), Stage: "stable", Caps: c.caps}
}
func (c *stubCloud) Verify(_ context.Context, key string) (Account, error) {
	c.count("Verify", key)
	credit := 25.0
	return Account{Credit: &credit, Detail: "Credit $25.00"}, c.verifyErr
}
func (c *stubCloud) Offers(_ context.Context, key string, q OfferQuery) ([]Offer, error) {
	c.count("Offers", key)
	return slices.Clone(c.offers), c.offersErr
}
func (c *stubCloud) Quote(_ context.Context, key, offerID string, q OfferQuery) (Offer, error) {
	c.count("Quote", key)
	c.mu.Lock()
	defer c.mu.Unlock()
	o := c.quote
	o.ID = offerID
	return o, c.quoteErr
}
func (c *stubCloud) Templates(_ context.Context, key string) ([]Template, error) {
	c.count("Templates", key)
	return c.templates, nil
}
func (c *stubCloud) Launch(_ context.Context, key string, s LaunchSpec) (Ref, error) {
	c.count("Launch", key)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.launched = append(c.launched, s)
	if c.launchErr != nil {
		return Ref{}, c.launchErr
	}
	c.next++
	id := fmt.Sprintf("pod%d", c.next)
	c.insts[id] = &Instance{ProviderID: id, Label: s.Label, Status: StatusCreating, ProviderStatus: "CREATED"}
	if c.noID {
		return Ref{OfferID: s.OfferID, Label: s.Label}, nil
	}
	return Ref{ID: id, OfferID: s.OfferID, Label: s.Label}, nil
}
func (c *stubCloud) find(ref Ref) (*Instance, bool) {
	if in, ok := c.insts[ref.ID]; ok && ref.ID != "" {
		return in, true
	}
	if ref.ID == "" {
		for _, in := range c.insts {
			if in.Label == ref.Label {
				return in, true
			}
		}
	}
	return nil, false
}
func (c *stubCloud) Get(_ context.Context, key string, ref Ref) (Instance, error) {
	c.count("Get", key)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.getErr != nil {
		return Instance{}, c.getErr
	}
	in, ok := c.find(ref)
	if !ok {
		return Instance{}, Errorf(c.kind, ClassNotFound, 404, "no pod %s", ref.ID)
	}
	return *in, nil
}
func (c *stubCloud) List(_ context.Context, key, prefix string) ([]Instance, error) {
	c.count("List", key)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.listErr != nil {
		return nil, c.listErr
	}
	var out []Instance
	for _, in := range c.insts {
		if strings.HasPrefix(in.Label, prefix) {
			out = append(out, *in)
		}
	}
	return out, nil
}
func (c *stubCloud) Stop(_ context.Context, key, id string) error {
	c.count("Stop", key)
	return c.actErr
}
func (c *stubCloud) Start(_ context.Context, key, id string) error {
	c.count("Start", key)
	return c.actErr
}
func (c *stubCloud) Terminate(_ context.Context, key, id string) error {
	c.count("Terminate", key)
	return c.actErr
}
func (c *stubCloud) Logs(_ context.Context, key, id string, tail int) (string, error) {
	c.count("Logs", key)
	return c.logs, c.actErr
}

// set changes the provider's view of instance id.
func (c *stubCloud) set(id string, fn func(*Instance)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	fn(c.insts[id])
}

// running puts id in the running state with one endpoint of the kind its
// provider hands out (FilterEndpoints drops any other): the https proxy on
// RunPod and QuickPod, a direct public IP on Vast.
func (c *stubCloud) running(id string) {
	ep := Endpoint{URL: "https://" + id + "-8675.proxy.runpod.net", Secure: true, Via: "proxy"}
	switch c.kind {
	case QuickPod:
		ep.URL = "https://" + id + "-8675.proxy.quickpod.io"
	case Vast:
		ep = Endpoint{URL: "http://203.0.113.10:41000", Via: "direct"}
	}
	c.set(id, func(in *Instance) {
		in.Status, in.ProviderStatus = StatusRunning, "RUNNING"
		in.Endpoints = []Endpoint{ep}
	})
}

func (c *stubCloud) remove(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.insts, id)
}

type fakeProber struct {
	mu     sync.Mutex
	down   bool
	open   bool
	accept string // "" accepts every token
	calls  int
}

func (p *fakeProber) Probe(_ context.Context, eps []Endpoint, token string, _ bool) ProbeResult {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	if p.down || len(eps) == 0 {
		return ProbeResult{Err: "connection refused"}
	}
	if p.accept != "" && token != p.accept {
		return ProbeResult{BaseURL: eps[0].URL, Health: true, TokenRejected: true}
	}
	enforced := !p.open
	return ProbeResult{BaseURL: eps[0].URL, Health: true, TokenOK: true, AuthEnforced: &enforced}
}

func (p *fakeProber) setDown(v bool) { p.mu.Lock(); p.down = v; p.mu.Unlock() }

// ---- the test environment ----

type testEnv struct {
	t      *testing.T
	ctx    context.Context
	s      *Service
	clock  *mockkit.FakeClock
	clouds map[Kind]*stubCloud
	prober *fakeProber
	hf     string
	sinks  notify.Config
}

var t0 = time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC)

// newEnv opens a fresh fork database and installs stub adapters for all
// three kinds, a fake prober and a fake clock.
func newEnv(t *testing.T) *testEnv {
	t.Helper()
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close() })
	for _, k := range []string{"AITK_PROVISION_PROVIDERS", "AITK_PROVISION_MAX_RUNNING", "AITK_PROVISION_READY_TIMEOUT",
		"RUNPOD_POD_ID", "CONTAINER_ID", "CONTAINER_LABEL", EnvKeyName(RunPod), EnvKeyName(Vast), EnvKeyName(QuickPod)} {
		t.Setenv(k, "")
	}
	wasSim := Simulated.Load()
	Simulated.Store(false)
	t.Cleanup(func() { Simulated.Store(wasSim) })

	e := &testEnv{t: t, ctx: context.Background(), clock: mockkit.NewFakeClock(t0), clouds: map[Kind]*stubCloud{}, prober: &fakeProber{}}
	for _, k := range allKinds {
		c := newStub(k)
		e.clouds[k] = c
		t.Cleanup(SetForTest(c))
	}
	oldProber, oldHF, oldNotify := DefaultProber, hfToken, notifyConfig
	DefaultProber = e.prober
	hfToken = func(context.Context) (string, error) { return e.hf, nil }
	notifyConfig = func() notify.Config { return e.sinks }
	t.Cleanup(func() { DefaultProber, hfToken, notifyConfig = oldProber, oldHF, oldNotify })
	e.s = &Service{Now: e.clock.Now}
	return e
}

func (e *testEnv) setKey(k Kind, key string) {
	e.t.Helper()
	if _, err := e.s.SetKey(e.ctx, k, ProviderUpdate{APIKey: &key}); err != nil {
		e.t.Fatal(err)
	}
}

func (e *testEnv) req(k Kind) LaunchRequest {
	r := LaunchRequest{Provider: k, OfferID: "offer-1", MaxPricePerHour: 0.5}
	if k == QuickPod {
		r.TemplateID = "tpl-aitk"
	}
	return r
}

// launch starts a machine on k (key set first) and fails the test on error.
func (e *testEnv) launch(k Kind, mod func(*LaunchRequest)) Machine {
	e.t.Helper()
	if _, src, _ := storedKey(e.ctx, k); src == "" {
		e.setKey(k, "key-"+string(k)+"-0001")
	}
	req := e.req(k)
	if mod != nil {
		mod(&req)
	}
	m, created, err := e.s.Launch(e.ctx, req, "")
	if err != nil {
		e.t.Fatal(err)
	}
	if !created {
		e.t.Fatal("not created")
	}
	return m
}

// refresh advances past the 5 s Get cache and re-reads id from its provider.
func (e *testEnv) refresh(id string) Machine {
	e.t.Helper()
	e.clock.Advance(6 * time.Second)
	m, err := e.s.Get(e.ctx, id, true)
	if err != nil {
		e.t.Fatal(err)
	}
	return m
}

func (e *testEnv) row(id string) *row {
	e.t.Helper()
	r, err := getRow(e.ctx, id)
	if err != nil {
		e.t.Fatal(err)
	}
	return r
}

// codeOf is the /v1 code an error answers with.
func codeOf(err error) string {
	var p *v1api.Problem
	if errors.As(err, &p) {
		return p.Code
	}
	if code, _, _, ok := V1(err); ok {
		return code
	}
	if err != nil {
		return "internal"
	}
	return ""
}

func problemOf(m Machine) Problem {
	if m.Problem == nil {
		return ""
	}
	return *m.Problem
}

// ---- keys ----

func TestKeyStoreAndHint(t *testing.T) {
	e := newEnv(t)
	v, err := e.s.SetKey(e.ctx, RunPod, ProviderUpdate{APIKey: ptr("rpa_secret_abcd")})
	if err != nil {
		t.Fatal(err)
	}
	if !v.Configured || v.KeySource != "api" || v.KeyHint == nil || *v.KeyHint != "…abcd" {
		t.Fatalf("view %+v", v)
	}
	if got := e.clouds[RunPod].keys; len(got) != 1 || got[0] != "rpa_secret_abcd" {
		t.Fatalf("verify saw %v", got)
	}
	ps, err := e.s.Providers(e.ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 3 || !ps[0].Configured || ps[1].Configured || ps[0].MaxRunning != DefaultMaxRunning {
		t.Fatalf("providers %+v", ps)
	}
	if err := e.s.RemoveKey(e.ctx, RunPod); err != nil {
		t.Fatal(err)
	}
	if v, _ := e.s.view(e.ctx, e.clouds[RunPod]); v.Configured {
		t.Fatal("key still configured after RemoveKey")
	}
}

func TestKeyEnvFallback(t *testing.T) {
	e := newEnv(t)
	t.Setenv(EnvKeyName(Vast), "vast_env_key_9876")
	t.Setenv("AITK_VAST_API_KEY", "self_terminate_key") // never a launcher key
	key, src, err := storedKey(e.ctx, Vast)
	if err != nil || key != "vast_env_key_9876" || src != "env" {
		t.Fatalf("got %q %q %v", key, src, err)
	}
	e.setKey(Vast, "vast_api_key_1111")
	if key, src, _ := storedKey(e.ctx, Vast); key != "vast_api_key_1111" || src != "api" {
		t.Fatalf("setting must win: %q %q", key, src)
	}
}

func TestKeySimSource(t *testing.T) {
	e := newEnv(t)
	Simulated.Store(true)
	SetSimSecret("simsecret0000")
	t.Cleanup(func() { SetSimSecret("") })
	v, err := e.s.SetKey(e.ctx, QuickPod, ProviderUpdate{APIKey: ptr("")})
	if err != nil {
		t.Fatal(err)
	}
	if !v.Configured || v.KeySource != "sim" || v.KeyHint != nil || !v.Simulated {
		t.Fatalf("view %+v", v)
	}
	if got := e.clouds[QuickPod].keys; got[0] != "simsecret0000" {
		t.Fatalf("verify got %q, want the sim secret", got[0])
	}
	e.setKey(Vast, "sim_full")
	if got := e.clouds[Vast].keys; got[0] != "simsecret0000.sim_full" {
		t.Fatalf("verify got %q", got[0])
	}
}

func TestKeySimKeepsRealKey(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "rpa_real_key_0001")
	Simulated.Store(true)
	SetSimSecret("simsecret0000")
	t.Cleanup(func() { SetSimSecret("") })
	e.setKey(RunPod, "sim_full")
	if key, src, _ := storedKey(e.ctx, RunPod); key != "sim_full" || src != "sim" {
		t.Fatalf("simulate key: %q %q", key, src)
	}
	if err := e.s.RemoveKey(e.ctx, RunPod); err != nil {
		t.Fatal(err)
	}
	Simulated.Store(false)
	if key, src, _ := storedKey(e.ctx, RunPod); key != "rpa_real_key_0001" || src != "api" {
		t.Fatalf("the real key after simulate: %q %q", key, src)
	}
}

func TestKeyVerifyBeforeStore(t *testing.T) {
	e := newEnv(t)
	e.clouds[RunPod].verifyErr = Errorf(RunPod, ClassAuth, 401, "bad key")
	_, err := e.s.SetKey(e.ctx, RunPod, ProviderUpdate{APIKey: ptr("rpa_wrong_key_0000")})
	if codeOf(err) != v1api.CodeProviderAuth {
		t.Fatalf("got %v", err)
	}
	if _, src, _ := storedKey(e.ctx, RunPod); src != "" {
		t.Fatal("a rejected key was stored")
	}
	if _, err := e.s.SetKey(e.ctx, RunPod, ProviderUpdate{APIKey: ptr("")}); codeOf(err) != v1api.CodeInvalidRequest {
		t.Fatalf("empty key outside simulate: %v", err)
	}
	pt, err := e.s.Test(e.ctx, RunPod)
	if err != nil || pt.OK || !strings.Contains(pt.Detail, "No RunPod API key") {
		t.Fatalf("test without a key: %+v %v", pt, err)
	}
}

func TestKeyRedaction(t *testing.T) {
	e := newEnv(t)
	const key = "rpa_planted_SECRET_1234"
	c := e.clouds[RunPod]
	leak := Errorf(RunPod, ClassUnavailable, 500, "upstream said: key "+key+" failed")
	c.verifyErr = leak
	if _, err := e.s.SetKey(e.ctx, RunPod, ProviderUpdate{APIKey: ptr(key)}); err == nil || strings.Contains(err.Error(), key) {
		t.Fatalf("SetKey: %v", err)
	}
	c.verifyErr = nil
	e.setKey(RunPod, key)
	c.verifyErr = leak
	if pt, _ := e.s.Test(e.ctx, RunPod); pt.OK || strings.Contains(pt.Detail, key) {
		t.Fatalf("Test: %+v", pt)
	}
	c.verifyErr = nil
	c.offersErr = fmt.Errorf("wrapped: %w", leak)
	if _, err := e.s.Offers(e.ctx, RunPod, OfferQuery{}); err == nil || strings.Contains(err.Error(), key) {
		t.Fatalf("Offers: %v", err)
	} else if _, msg, _, _ := V1(err); strings.Contains(msg, key) {
		t.Fatalf("Offers V1: %s", msg)
	}
	c.quoteErr = leak
	if _, _, err := e.s.Launch(e.ctx, e.req(RunPod), ""); err == nil || strings.Contains(err.Error(), key) {
		t.Fatalf("Launch quote: %v", err)
	}
	c.quoteErr = nil
	c.launchErr = leak
	if _, _, err := e.s.Launch(e.ctx, e.req(RunPod), ""); err == nil || strings.Contains(err.Error(), key) {
		t.Fatalf("Launch: %v", err)
	}
	ms, _ := e.s.List(e.ctx, false)
	if len(ms) != 1 || strings.Contains(ms[0].Detail, key) {
		t.Fatalf("failed machine detail %+v", ms)
	}
	c.launchErr = nil
	m := e.launch(RunPod, nil)
	c.running("pod1")
	e.refresh(m.ID)
	c.actErr = leak
	if _, err := e.s.Stop(e.ctx, m.ID); err == nil || strings.Contains(err.Error(), key) {
		t.Fatalf("Stop: %v", err)
	}
	if _, err := e.s.Logs(e.ctx, m.ID, 10); err == nil || strings.Contains(err.Error(), key) {
		t.Fatalf("Logs: %v", err)
	}
}

func TestKeyMaxRunning(t *testing.T) {
	e := newEnv(t)
	for _, n := range []int{0, 21} {
		if _, err := e.s.SetKey(e.ctx, Vast, ProviderUpdate{MaxRunning: &n}); codeOf(err) != v1api.CodeInvalidRequest {
			t.Fatalf("max_running %d: %v", n, err)
		}
	}
	t.Setenv("AITK_PROVISION_MAX_RUNNING", "4")
	if n, _ := maxRunning(e.ctx, Vast); n != 4 {
		t.Fatalf("env default: %d", n)
	}
	five := 5
	v, err := e.s.SetKey(e.ctx, Vast, ProviderUpdate{MaxRunning: &five})
	if err != nil || v.MaxRunning != 5 {
		t.Fatalf("%+v %v", v, err)
	}
	if e.clouds[Vast].n("Verify") != 0 {
		t.Fatal("max_running alone must not verify")
	}
}

func TestServiceProvidersMakeNoCalls(t *testing.T) {
	e := newEnv(t)
	t.Setenv(EnvKeyName(RunPod), "env_key_abcd")
	if _, err := e.s.Providers(e.ctx); err != nil {
		t.Fatal(err)
	}
	for k, c := range e.clouds {
		if len(c.calls) != 0 {
			t.Fatalf("%s: calls %v", k, c.calls)
		}
	}
	t.Setenv("AITK_PROVISION_PROVIDERS", "runpod,vast")
	if ps, _ := e.s.Providers(e.ctx); len(ps) != 2 {
		t.Fatalf("kill switch: %d providers", len(ps))
	}
}

// ---- launch ----

func TestLaunchGuards(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "rpa_key_0001")
	e.setKey(QuickPod, "qpk_key_0001")
	cases := []struct {
		name string
		k    Kind
		mod  func(*LaunchRequest)
		code string
	}{
		{"no key", Vast, nil, v1api.CodeInvalidRequest},
		{"bad provider", "aws", nil, v1api.CodeInvalidRequest},
		{"no max price", RunPod, func(r *LaunchRequest) { r.MaxPricePerHour = 0 }, v1api.CodeInvalidRequest},
		{"no offer", RunPod, func(r *LaunchRequest) { r.OfferID = " " }, v1api.CodeInvalidRequest},
		{"long name", RunPod, func(r *LaunchRequest) { r.Name = strings.Repeat("n", 65) }, v1api.CodeInvalidRequest},
		{"control char", RunPod, func(r *LaunchRequest) { r.Name = "a\x07b" }, v1api.CodeInvalidRequest},
		{"image space", RunPod, func(r *LaunchRequest) { r.Image = "ghcr.io/x/y:latest extra" }, v1api.CodeInvalidRequest},
		{"image long", RunPod, func(r *LaunchRequest) { r.Image = strings.Repeat("a", 1025) }, v1api.CodeInvalidRequest},
		{"disk small", RunPod, func(r *LaunchRequest) { r.DiskGB = 19 }, v1api.CodeInvalidRequest},
		{"disk big", RunPod, func(r *LaunchRequest) { r.DiskGB = 2001 }, v1api.CodeInvalidRequest},
		{"cloud", RunPod, func(r *LaunchRequest) { r.Cloud = "spot" }, v1api.CodeInvalidRequest},
		{"no template", QuickPod, func(r *LaunchRequest) { r.TemplateID = "" }, v1api.CodeInvalidRequest},
		{"unknown template", QuickPod, func(r *LaunchRequest) { r.TemplateID = "tpl-other" }, v1api.CodeInvalidRequest},
		{"cors path", RunPod, func(r *LaunchRequest) { r.CORSOrigins = []string{"https://a.example/x"} }, v1api.CodeInvalidRequest},
		{"cors scheme", RunPod, func(r *LaunchRequest) { r.CORSOrigins = []string{"ftp://a.example"} }, v1api.CodeInvalidRequest},
		{"studio url", RunPod, func(r *LaunchRequest) { r.StudioURL = "javascript:alert(1)" }, v1api.CodeInvalidRequest},
		{"request id", RunPod, func(r *LaunchRequest) { r.RequestID = "short" }, v1api.CodeInvalidRequest},
		{"when done", RunPod, func(r *LaunchRequest) { r.WhenDone = "later" }, v1api.CodeInvalidRequest},
		{"stop without remote", RunPod, func(r *LaunchRequest) { r.WhenDone = WhenStop }, v1api.CodeInvalidConfig},
		{"reserved env", RunPod, func(r *LaunchRequest) { r.Env = map[string]string{"AI_TOOLKIT_AUTH": "x"} }, v1api.CodeInvalidConfig},
		{"copy unknown remote", RunPod, func(r *LaunchRequest) { r.CopyRemote = "nope" }, v1api.CodeInvalidRequest},
	}
	for _, c := range cases {
		req := e.req(c.k)
		if c.mod != nil {
			c.mod(&req)
		}
		_, _, err := e.s.Launch(e.ctx, req, "")
		if got := codeOf(err); got != c.code {
			t.Errorf("%s: code %q (%v), want %q", c.name, got, err, c.code)
		}
	}
	if n := e.clouds[RunPod].n("Launch") + e.clouds[QuickPod].n("Launch"); n != 0 {
		t.Fatalf("a refused launch reached the provider %d times", n)
	}
	if ms, _ := e.s.List(e.ctx, false); len(ms) != 0 {
		t.Fatalf("refused launches left rows: %+v", ms)
	}
}

func TestLaunchCreates(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "rpa_key_0001")
	ch, cancel := events.Subscribe(events.Filter{Types: map[string]bool{events.MachineState: true}})
	defer cancel()
	m, created, err := e.s.Launch(e.ctx, LaunchRequest{Provider: RunPod, OfferID: "offer-1", MaxPricePerHour: 0.5,
		StudioURL: "https://studio.example.com/"}, "https://studio.example.com")
	if err != nil || !created {
		t.Fatal(created, err)
	}
	if m.Status != StatusCreating || m.ProviderID != "pod1" || m.Name != "RTX 4090 on RunPod" || m.GPU != "RTX 4090" ||
		m.Label != LabelFor(m.ID) || !strings.HasPrefix(m.ID, "mch_") || m.DiskGB != DefaultDiskGB || m.Image != DefaultImage ||
		m.WhenDone != WhenKeep || m.PricePerHour == nil || *m.PricePerHour != 0.45 || m.RelayPath != "/v1/machines/"+m.ID+"/relay" ||
		m.ImageDigest != nil || m.ImageLaunchRef != nil || m.ImageWarnings == nil || m.Warnings == nil || m.Simulated {
		t.Fatalf("machine %+v", m)
	}
	spec := e.clouds[RunPod].launched[0]
	tok := e.row(m.ID).Token
	if spec.Label != m.Label || spec.Image != DefaultImage || spec.DiskGB != DefaultDiskGB || spec.OfferID != "offer-1" ||
		spec.Env["AI_TOOLKIT_AUTH"] != tok || !strings.HasPrefix(tok, "aitk_") || spec.Env["AITK_MACHINE_ID"] != m.ID ||
		spec.Env["AITK_CORS_ORIGINS"] != "https://studio.example.com" {
		t.Fatalf("spec %+v", spec)
	}
	r := e.row(m.ID)
	if !slices.Equal(r.Spec.EnvKeys, []string{"AITK_CORS_ORIGINS", "AITK_MACHINE_ID", "AITK_STUDIO_URL", "AI_TOOLKIT_AUTH"}) {
		t.Fatalf("env keys %v", r.Spec.EnvKeys)
	}
	raw, _ := json.Marshal(r.Spec)
	if strings.Contains(string(raw), tok) {
		t.Fatal("the spec holds an env value")
	}
	select {
	case ev := <-ch:
		if strings.Contains(string(ev.Data), tok) {
			t.Fatal("event carries the token")
		}
	default:
		t.Fatal("no machine.state on launch")
	}
	if ps, _ := e.s.Providers(e.ctx); ps[0].Running != 1 {
		t.Fatalf("running %d", ps[0].Running)
	}
}

func TestLaunchIdempotentRequestID(t *testing.T) {
	e := newEnv(t)
	m := e.launch(Vast, func(r *LaunchRequest) { r.RequestID = "req_00000001" })
	again, created, err := e.s.Launch(e.ctx, LaunchRequest{Provider: Vast, OfferID: "offer-1", MaxPricePerHour: 0.5, RequestID: "req_00000001"}, "")
	if err != nil || created || again.ID != m.ID {
		t.Fatalf("repeat: %v %v %v", again.ID, created, err)
	}
	if n := e.clouds[Vast].n("Launch"); n != 1 {
		t.Fatalf("launch calls %d", n)
	}
}

func TestLaunchMaxRunning(t *testing.T) {
	e := newEnv(t)
	e.launch(RunPod, nil)
	e.launch(RunPod, nil)
	_, _, err := e.s.Launch(e.ctx, e.req(RunPod), "")
	if codeOf(err) != v1api.CodeConflict || !strings.Contains(err.Error(), "2 RunPod servers are already running") {
		t.Fatalf("got %v", err)
	}
	e.launch(Vast, nil) // the cap is per provider
	three := 3
	if _, err := e.s.SetKey(e.ctx, RunPod, ProviderUpdate{MaxRunning: &three}); err != nil {
		t.Fatal(err)
	}
	m := e.launch(RunPod, nil)
	if m.Name != "RTX 4090 on RunPod 3" {
		t.Fatalf("name %q", m.Name)
	}
}

func TestLaunchPriceRose(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "rpa_key_0001")
	e.clouds[RunPod].quote.PricePerHour = 0.52
	req := e.req(RunPod)
	req.MaxPricePerHour = 0.47
	_, _, err := e.s.Launch(e.ctx, req, "")
	_, msg, _, _ := V1(err)
	if codeOf(err) != v1api.CodeConflict || msg != "The price is now $0.52/h, above your limit of $0.47/h. Check the offer and try again." {
		t.Fatalf("got %q %v", msg, err)
	}
	if e.clouds[RunPod].n("Launch") != 0 {
		t.Fatal("launched above the limit")
	}
}

func TestLaunchQuoteCache(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[Vast]
	e.launch(Vast, nil)
	e.launch(Vast, func(r *LaunchRequest) { r.Name = "second" })
	if n := c.n("Quote"); n != 1 {
		t.Fatalf("quotes within 30 s: %d", n)
	}
	if _, err := e.s.Quote(e.ctx, Vast, "offer-1", OfferQuery{DiskGB: DefaultDiskGB, GPUCount: 1}); err != nil || c.n("Quote") != 1 {
		t.Fatalf("Manager.Quote shares the cache: %d %v", c.n("Quote"), err)
	}
	if _, err := e.s.Quote(e.ctx, Vast, "offer-1", OfferQuery{DiskGB: 200, GPUCount: 1}); err != nil || c.n("Quote") != 2 {
		t.Fatalf("another disk size is another quote: %d", c.n("Quote"))
	}
	e.clock.Advance(31 * time.Second)
	if _, err := e.s.Quote(e.ctx, Vast, "offer-1", OfferQuery{DiskGB: DefaultDiskGB, GPUCount: 1}); err != nil || c.n("Quote") != 3 {
		t.Fatalf("after 30 s: %d", c.n("Quote"))
	}
}

func TestLaunchFailureRecorded(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "vast_key_0001")
	e.clouds[Vast].launchErr = Errorf(Vast, ClassCredit, 402, "balance $0.10")
	_, _, err := e.s.Launch(e.ctx, e.req(Vast), "")
	if codeOf(err) != v1api.CodeInsufficientCredit {
		t.Fatalf("got %v", err)
	}
	ms, _ := e.s.List(e.ctx, false)
	if len(ms) != 1 || ms[0].Status != StatusFailed || ms[0].EndedAt == nil || !strings.Contains(ms[0].Detail, "balance") ||
		!slices.Contains(ms[0].Actions, "forget") || slices.Contains(ms[0].Actions, "terminate") {
		t.Fatalf("machines %+v", ms)
	}
	if active, _ := e.s.List(e.ctx, true); len(active) != 0 {
		t.Fatal("a failed launch is still active")
	}
	e.clouds[Vast].launchErr = nil
	if m := e.launch(Vast, nil); m.Name != "RTX 4090 on Vast.ai" {
		t.Fatalf("the failed row kept its name: %q", m.Name)
	}
}

func TestLaunchSurvivesCallerCancel(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "rpa_key_0001")
	entered, gate := make(chan struct{}), make(chan struct{})
	t.Cleanup(SetForTest(&slowLaunch{stubCloud: e.clouds[RunPod], entered: entered, gate: gate}))
	ctx, cancel := context.WithCancel(e.ctx)
	type result struct {
		m   Machine
		err error
	}
	done := make(chan result)
	go func() {
		m, _, err := e.s.Launch(ctx, e.req(RunPod), "")
		done <- result{m, err}
	}()
	<-entered
	cancel() // the client goes away mid-create
	close(gate)
	res := <-done
	if res.err != nil {
		t.Fatalf("a cancelled caller aborted the create: %v", res.err)
	}
	if r := e.row(res.m.ID); r.ProviderID != "pod1" || r.Status != StatusCreating || r.EndedAt != nil {
		t.Fatalf("the outcome was not recorded: %+v", r)
	}
}

// slowLaunch holds a create until gate opens and fails it the way an HTTP
// call does when its context ends meanwhile.
type slowLaunch struct {
	*stubCloud
	entered, gate chan struct{}
}

func (c *slowLaunch) Launch(ctx context.Context, key string, s LaunchSpec) (Ref, error) {
	close(c.entered)
	<-c.gate
	if err := ctx.Err(); err != nil {
		return Ref{}, err
	}
	return c.stubCloud.Launch(ctx, key, s)
}

func TestLaunchNames(t *testing.T) {
	e := newEnv(t)
	e.clouds[RunPod].quote.GPUCount = 2
	a := e.launch(RunPod, nil)
	b := e.launch(RunPod, nil)
	if a.Name != "2× RTX 4090 on RunPod" || b.Name != "2× RTX 4090 on RunPod 2" || a.GPUCount != 2 {
		t.Fatalf("%q %q", a.Name, b.Name)
	}
	_, _, err := e.s.Launch(e.ctx, LaunchRequest{Provider: Vast, OfferID: "o", MaxPricePerHour: 1, Name: a.Name}, "")
	if codeOf(err) != v1api.CodeInvalidRequest { // no Vast key yet
		t.Fatalf("got %v", err)
	}
	e.setKey(Vast, "vast_key_0001")
	if _, _, err := e.s.Launch(e.ctx, LaunchRequest{Provider: Vast, OfferID: "o", MaxPricePerHour: 1, Name: a.Name}, ""); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("a live name is unique: %v", err)
	}
	if _, err := e.s.Update(e.ctx, b.ID, MachinePatch{Name: ptr(a.Name)}); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("rename onto a live name: %v", err)
	}
	if m, err := e.s.Update(e.ctx, b.ID, MachinePatch{Name: ptr("trainer")}); err != nil || m.Name != "trainer" {
		t.Fatalf("rename: %v %v", m.Name, err)
	}
	if m, err := e.s.Get(e.ctx, "trainer", false); err != nil || m.ID != b.ID {
		t.Fatalf("get by live name: %v %v", m.ID, err)
	}
}

func TestWhenDoneDefault(t *testing.T) {
	for _, k := range allKinds {
		for _, remote := range []bool{false, true} {
			for _, key := range []bool{false, true} {
				req := LaunchRequest{Provider: k, SelfDestroyKey: key}
				if remote {
					req.CopyRemote = "r2"
				}
				got, warn, err := whenDone(k, req)
				want, wantWarn := WhenKeep, ""
				if remote && (k != QuickPod || key) {
					want = WhenDestroy
				}
				if remote && k == QuickPod && !key {
					wantWarn = warnQuickPodKeep
				}
				if err != nil || got != want || warn != wantWarn {
					t.Errorf("%s remote=%v key=%v: %q %q %v, want %q %q", k, remote, key, got, warn, err, want, wantWarn)
				}
			}
		}
	}
	for _, c := range []struct {
		k    Kind
		req  LaunchRequest
		code string
	}{
		{RunPod, LaunchRequest{WhenDone: WhenDestroy}, v1api.CodeInvalidConfig},
		{Vast, LaunchRequest{WhenDone: WhenStop}, v1api.CodeInvalidConfig},
		{QuickPod, LaunchRequest{WhenDone: WhenDestroy, CopyRemote: "r2"}, v1api.CodeInvalidConfig},
		{QuickPod, LaunchRequest{WhenDone: WhenStop, CopyRemote: "r2", SelfDestroyKey: true}, ""},
		{RunPod, LaunchRequest{WhenDone: WhenKeep, CopyRemote: "r2"}, ""},
	} {
		got, _, err := whenDone(c.k, c.req)
		if codeOf(err) != c.code || (c.code == "" && got != c.req.WhenDone) {
			t.Errorf("%s %+v: %q %v", c.k, c.req, got, err)
		}
	}
}

func TestWhenDoneLaunchWarning(t *testing.T) {
	e := newEnv(t)
	addS3Remote(t, "r2")
	m := e.launch(QuickPod, func(r *LaunchRequest) { r.CopyRemote = "r2" })
	if m.WhenDone != WhenKeep || len(m.Warnings) == 0 || m.Warnings[0] != warnQuickPodKeep {
		t.Fatalf("%q %v", m.WhenDone, m.Warnings)
	}
	if _, ok := e.clouds[QuickPod].launched[0].Env["AITK_WHEN_DONE"]; ok {
		t.Fatal("keep sends no AITK_WHEN_DONE")
	}
	// the warnings live in the spec, so a restarted server shows them too
	if got := (&Service{}).machine(e.row(m.ID)); !slices.Equal(got.Warnings, m.Warnings) {
		t.Fatalf("after restart %v", got.Warnings)
	}
	m = e.launch(Vast, func(r *LaunchRequest) { r.CopyRemote = "r2" })
	if m.WhenDone != WhenDestroy || e.clouds[Vast].launched[0].Env["AITK_WHEN_DONE"] != "destroy" ||
		e.clouds[Vast].launched[0].Env["AITK_WHEN_DONE_PROVIDER"] != "vast" {
		t.Fatalf("vast %q", m.WhenDone)
	}
	m = e.launch(QuickPod, func(r *LaunchRequest) { r.CopyRemote = "r2"; r.SelfDestroyKey = true })
	if m.WhenDone != WhenDestroy || len(m.Warnings) != 0 || e.clouds[QuickPod].launched[1].Env["AITK_QUICKPOD_API_KEY"] == "" {
		t.Fatalf("quickpod with the key: %q %v", m.WhenDone, m.Warnings)
	}
	_, _, err := e.s.Launch(e.ctx, LaunchRequest{Provider: QuickPod, OfferID: "o", MaxPricePerHour: 1, TemplateID: "tpl-aitk",
		CopyRemote: "r2", WhenDone: WhenDestroy}, "")
	if codeOf(err) != v1api.CodeInvalidConfig {
		t.Fatalf("QuickPod destroy without the key: %v", err)
	}
}

// ---- store ----

func TestStoreRoundTrip(t *testing.T) {
	e := newEnv(t)
	stopped, maxDisk, rel := 0.02, 500, 0.99
	price := 0.45
	rid := "req_roundtrip"
	r := &row{
		ID: "mch_round", Provider: Vast, ProviderID: "123", Label: "aitk-round", Name: "round", Token: "aitk_tok",
		Status: StatusRunning, Problem: ProblemPullSlow, Detail: "slow", ProviderStatus: "loading/running", Pending: pendingStop,
		Spec: machineSpec{
			Offer: Offer{ID: "o1", Provider: Vast, GPU: "RTX 4090", GPUCount: 1, VRAMGB: 24, PricePerHour: 0.45,
				StoppedPricePerHour: &stopped, Location: "SE", Reliability: &rel, Availability: "available", MaxDiskGB: &maxDisk},
			GPUCount: 1, Cloud: "", DiskGB: 120, Image: DefaultImage, TemplateID: "t", WhenDone: WhenDestroy, SelfDestroyKey: true,
			CopiedRemote: "r2", EnvKeys: []string{"A", "B"}, Warnings: []string{"w1"},
			ImageCheck: json.RawMessage(`{"ref":"ghcr.io/geocine/aitoolkit-go:latest","ok":true}`), ImageWarnings: []string{"iw"}, PullS: 212.5,
		},
		BaseURL: "http://203.0.113.10:41000", Endpoints: []Endpoint{{URL: "http://203.0.113.10:41000", Via: "direct"}},
		ImageLaunchRef: "ghcr.io/geocine/aitoolkit-go:abc1234", ImageDigest: "sha256:" + strings.Repeat("a", 64),
		Price: &price, StoppedPrice: &stopped, RequestID: &rid, Simulated: true,
		CreatedAt: 1000, UpdatedAt: 2000, RunningAt: ptr(int64(3000)), ReadyAt: ptr(int64(4000)), LastOKAt: ptr(int64(5000)),
	}
	if err := insertRow(e.ctx, r); err != nil {
		t.Fatal(err)
	}
	got := e.row(r.ID)
	if !reflect.DeepEqual(got, r) {
		t.Fatalf("round trip\n got %+v\nwant %+v", got, r)
	}
	r.EndedAt, r.ImageDigest, r.Spec.PullS = ptr(int64(6000)), "sha256:"+strings.Repeat("b", 64), 99
	if err := updateRow(e.ctx, r); err != nil {
		t.Fatal(err)
	}
	if got := e.row(r.ID); !reflect.DeepEqual(got, r) {
		t.Fatalf("update round trip\n got %+v\nwant %+v", got, r)
	}
	if got, _ := rowByRequestID(e.ctx, rid); got == nil || got.ID != r.ID {
		t.Fatal("by request id")
	}
	m := e.s.machine(r)
	if m.ImageDigest == nil || *m.ImageDigest != r.ImageDigest || m.ImageLaunchRef == nil || *m.ImageLaunchRef != r.ImageLaunchRef ||
		!slices.Equal(m.ImageWarnings, []string{"iw"}) || !slices.Equal(m.Warnings, []string{"w1"}) || m.Secure ||
		m.Location != "SE" || m.DiskGB != 120 || problemOf(m) != ProblemPullSlow {
		t.Fatalf("machine %+v", m)
	}
	raw, _ := json.Marshal(m)
	if strings.Contains(string(raw), "aitk_tok") {
		t.Fatal("the Machine JSON carries the token")
	}
}

// ---- offers cache ----

func TestOffersCache(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "vast_key_0001")
	c := e.clouds[Vast]
	q := OfferQuery{GPU: "4090", MinVRAMGB: 24, Image: DefaultImage}
	if _, err := e.s.Offers(e.ctx, Vast, q); err != nil {
		t.Fatal(err)
	}
	e.clock.Advance(10 * time.Second)
	o, err := e.s.Offers(e.ctx, Vast, q)
	if err != nil || len(o) != 1 || c.n("Offers") != 1 {
		t.Fatalf("same query within 30 s: %d calls %v", c.n("Offers"), err)
	}
	o[0].GPU = "mutated" // callers get their own slice
	if again, _ := e.s.Offers(e.ctx, Vast, q); again[0].GPU != "RTX 4090" {
		t.Fatal("the cached slice was shared")
	}
	q2 := q
	q2.Model = nil
	q2.Fast = true
	if _, err := e.s.Offers(e.ctx, Vast, q2); err != nil || c.n("Offers") != 2 {
		t.Fatalf("different query: %d", c.n("Offers"))
	}
	q3 := q
	q3.Filter, q3.ImageInfo = &hostfit.Filter{}, &imagecheck.Image{}
	if _, err := e.s.Offers(e.ctx, Vast, q3); c.n("Offers") != 2 {
		t.Fatalf("Filter/ImageInfo are not part of the key: %d %v", c.n("Offers"), err)
	}
	e.clock.Advance(21 * time.Second)
	if _, err := e.s.Offers(e.ctx, Vast, q); err != nil || c.n("Offers") != 3 {
		t.Fatalf("after 30 s: %d", c.n("Offers"))
	}
	e.setKey(Vast, "vast_key_0002") // a key change clears the kind's entries
	if _, err := e.s.Offers(e.ctx, Vast, q); err != nil || c.n("Offers") != 4 {
		t.Fatalf("after a key change: %d", c.n("Offers"))
	}
	c.offersErr = Errorf(Vast, ClassUnavailable, 502, "down")
	e.clock.Advance(31 * time.Second)
	if _, err := e.s.Offers(e.ctx, Vast, q); err == nil {
		t.Fatal("error swallowed")
	}
	c.offersErr = nil
	if _, err := e.s.Offers(e.ctx, Vast, q); err != nil || c.n("Offers") != 6 {
		t.Fatalf("errors are not cached: %d", c.n("Offers"))
	}
}

func TestOffersCacheSweep(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "vast_key_0001")
	for _, gpu := range []string{"4", "40", "409", "4090"} {
		if _, err := e.s.Offers(e.ctx, Vast, OfferQuery{GPU: gpu}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := e.s.Quote(e.ctx, Vast, "offer-1", OfferQuery{DiskGB: DefaultDiskGB}); err != nil {
		t.Fatal(err)
	}
	e.clock.Advance(31 * time.Second)
	if _, err := e.s.Offers(e.ctx, Vast, OfferQuery{GPU: "A100"}); err != nil {
		t.Fatal(err)
	}
	if _, err := e.s.Quote(e.ctx, Vast, "offer-2", OfferQuery{DiskGB: DefaultDiskGB}); err != nil {
		t.Fatal(err)
	}
	e.s.cache.mu.Lock()
	n, nq := len(e.s.cache.offers), len(e.s.cache.quotes)
	e.s.cache.mu.Unlock()
	if n != 1 || nq != 1 {
		t.Fatalf("expired entries kept: %d offers, %d quotes", n, nq)
	}
}

func TestOffersCacheSingleflightCallerCancel(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "rpa_key_0001")
	gate := make(chan struct{})
	t.Cleanup(SetForTest(&gatedCloud{stubCloud: e.clouds[RunPod], gate: gate}))
	q := OfferQuery{GPU: "A100"}
	first, cancel := context.WithCancel(e.ctx)
	firstDone := make(chan error)
	go func() {
		_, err := e.s.Offers(first, RunPod, q)
		firstDone <- err
	}()
	time.Sleep(50 * time.Millisecond) // the first caller runs the shared call
	second := make(chan error)
	go func() {
		o, err := e.s.Offers(e.ctx, RunPod, q)
		if err == nil && len(o) != 1 {
			err = fmt.Errorf("got %d offers", len(o))
		}
		second <- err
	}()
	gone, cancelGone := context.WithCancel(e.ctx)
	cancelGone()
	if _, err := e.s.Offers(gone, RunPod, q); !errors.Is(err, context.Canceled) {
		t.Fatalf("a waiter whose ctx ended: %v", err)
	}
	time.Sleep(50 * time.Millisecond)
	cancel() // the first caller goes away
	close(gate)
	if err := <-second; err != nil {
		t.Fatalf("the first caller's cancel failed another caller: %v", err)
	}
	<-firstDone
	if n := e.clouds[RunPod].n("Offers"); n != 1 {
		t.Fatalf("%d calls", n)
	}
}

func TestOffersCacheSingleflight(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "rpa_key_0001")
	gate := make(chan struct{})
	slow := &gatedCloud{stubCloud: e.clouds[RunPod], gate: gate}
	t.Cleanup(SetForTest(slow))
	var wg sync.WaitGroup
	for range 5 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := e.s.Offers(e.ctx, RunPod, OfferQuery{GPU: "A100"}); err != nil {
				t.Error(err)
			}
		}()
	}
	time.Sleep(50 * time.Millisecond)
	close(gate)
	wg.Wait()
	if n := e.clouds[RunPod].n("Offers"); n != 1 {
		t.Fatalf("concurrent identical queries made %d calls", n)
	}
}

type gatedCloud struct {
	*stubCloud
	gate chan struct{}
}

func (g *gatedCloud) Offers(ctx context.Context, key string, q OfferQuery) ([]Offer, error) {
	select {
	case <-g.gate:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return g.stubCloud.Offers(ctx, key, q)
}

// ---- status overlays ----

func TestServiceReadyFlow(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[RunPod]
	m := e.launch(RunPod, nil)
	c.set("pod1", func(in *Instance) { in.Status = StatusStarting })
	if m = e.refresh(m.ID); m.Status != StatusStarting {
		t.Fatalf("status %s", m.Status)
	}
	c.running("pod1")
	c.set("pod1", func(in *Instance) { in.Location = "US-TX-3" })
	m = e.refresh(m.ID)
	if m.Status != StatusReady || m.BaseURL == nil || *m.BaseURL != "https://pod1-8675.proxy.runpod.net" || !m.Secure ||
		m.ReadyAt == nil || problemOf(m) != "" || m.Location != "US-TX-3" || !slices.Equal(m.Actions, []string{"stop", "terminate"}) {
		t.Fatalf("machine %+v", m)
	}
	link, err := e.s.Link(e.ctx, m.ID)
	if err != nil || link.Token != e.row(m.ID).Token || link.BaseURL != *m.BaseURL || !link.Secure {
		t.Fatalf("link %+v %v", link, err)
	}
	rt, err := e.s.RelayTarget(e.ctx, m.ID)
	if err != nil || rt.BaseURL != *m.BaseURL || rt.Token != link.Token {
		t.Fatalf("relay %+v %v", rt, err)
	}
}

// The relay only ever reaches a ready machine at an address its provider's
// rules allow; a stale base_url on a machine that is not ready is refused.
func TestServiceRelayTargetNotReady(t *testing.T) {
	e := newEnv(t)
	m := e.launch(RunPod, nil)
	e.clouds[RunPod].running("pod1")
	if m = e.refresh(m.ID); m.Status != StatusReady {
		t.Fatalf("status %s", m.Status)
	}
	for _, c := range []struct {
		status Status
		base   string
	}{
		{StatusStarting, *m.BaseURL},
		{StatusRunning, *m.BaseURL},
		{StatusStopped, *m.BaseURL},
		{StatusReady, "http://203.0.113.10:41000"}, // RunPod is reached through its proxy only
	} {
		r := e.row(m.ID)
		r.Status, r.BaseURL = c.status, c.base
		if err := updateRow(e.ctx, r); err != nil {
			t.Fatal(err)
		}
		if rt, err := e.s.RelayTarget(e.ctx, m.ID); codeOf(err) != v1api.CodeConflict {
			t.Fatalf("%s %s: %+v %v", c.status, c.base, rt, err)
		}
	}
}

// Only the endpoints the provider's rules allow are probed with the token:
// a RunPod record carrying a direct IP never becomes ready through it.
func TestServiceProbesAllowedEndpoints(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[RunPod]
	m := e.launch(RunPod, nil)
	direct := Endpoint{URL: "http://203.0.113.10:41000", Via: "direct"}
	c.set("pod1", func(in *Instance) {
		in.Status, in.ProviderStatus, in.Endpoints = StatusRunning, "RUNNING", []Endpoint{direct}
	})
	if m = e.refresh(m.ID); m.Status != StatusRunning || m.BaseURL != nil {
		t.Fatalf("direct only: %+v", m)
	}
	proxy := Endpoint{URL: "https://pod1-8675.proxy.runpod.net", Secure: true, Via: "proxy"}
	c.set("pod1", func(in *Instance) { in.Endpoints = []Endpoint{direct, proxy} })
	if m = e.refresh(m.ID); m.Status != StatusReady || m.BaseURL == nil || *m.BaseURL != proxy.URL {
		t.Fatalf("direct then proxy: %+v", m)
	}
}

func TestServiceNotReadyTimeout(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[Vast]
	m := e.launch(Vast, nil)
	c.set("pod1", func(in *Instance) { in.Status = StatusStarting })
	e.clock.Advance(19 * time.Minute)
	if m = e.refresh(m.ID); m.Status != StatusStarting {
		t.Fatalf("before 20 min: %s", m.Status)
	}
	e.clock.Advance(2 * time.Minute)
	m = e.refresh(m.ID)
	if m.Status != StatusFailed || problemOf(m) != ProblemNotReady || m.Detail != detailNotReady || m.EndedAt != nil ||
		!slices.Contains(m.Actions, "terminate") {
		t.Fatalf("machine %+v", m)
	}
	if c.n("Terminate") != 0 {
		t.Fatal("the service terminated on its own")
	}
	c.running("pod1") // it answers late: it recovers
	if m = e.refresh(m.ID); m.Status != StatusReady || problemOf(m) != "" {
		t.Fatalf("late ready: %+v", m)
	}
}

func TestServiceNoPortGrace(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[Vast]
	m := e.launch(Vast, nil)
	c.set("pod1", func(in *Instance) { in.Status, in.PortMissing = StatusRunning, true })
	if m = e.refresh(m.ID); m.Status != StatusRunning || problemOf(m) != "" {
		t.Fatalf("at once: %+v", m)
	}
	e.clock.Advance(2 * time.Minute)
	if m = e.refresh(m.ID); problemOf(m) != "" {
		t.Fatalf("within 3 min: %+v", m)
	}
	e.clock.Advance(time.Minute)
	if m = e.refresh(m.ID); problemOf(m) != ProblemNoPort || m.Detail != detailNoPort || m.Status != StatusRunning {
		t.Fatalf("after 3 min: %+v", m)
	}
}

func TestServiceTokenMismatchThenPatch(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[QuickPod]
	e.prober.accept = "template-token-0000"
	m := e.launch(QuickPod, nil)
	c.running("pod1")
	if m = e.refresh(m.ID); m.Status != StatusRunning || problemOf(m) != ProblemTokenMismatch || m.Detail != detailMismatch {
		t.Fatalf("mismatch: %+v", m)
	}
	if _, err := e.s.Link(e.ctx, m.ID); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("link before ready: %v", err)
	}
	if _, err := e.s.Update(e.ctx, m.ID, MachinePatch{Token: ptr("bad token")}); codeOf(err) != v1api.CodeInvalidRequest {
		t.Fatalf("token with a space: %v", err)
	}
	calls := c.n("Get")
	m, err := e.s.Update(e.ctx, m.ID, MachinePatch{Token: ptr("template-token-0000")})
	if err != nil || m.Status != StatusReady || problemOf(m) != "" {
		t.Fatalf("after PATCH token: %+v %v", m, err)
	}
	if c.n("Get") != calls {
		t.Fatal("re-probing needs no provider call")
	}
	if link, _ := e.s.Link(e.ctx, m.ID); link.Token != "template-token-0000" {
		t.Fatalf("link token %q", link.Token)
	}
}

func TestServiceOpenServer(t *testing.T) {
	e := newEnv(t)
	e.prober.open = true
	m := e.launch(RunPod, nil)
	e.clouds[RunPod].running("pod1")
	m = e.refresh(m.ID)
	if m.Status != StatusReady || problemOf(m) != ProblemOpenServer || m.Detail != detailOpen {
		t.Fatalf("machine %+v", m)
	}
	if _, err := e.s.Link(e.ctx, m.ID); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("an open server is never linked: %v", err)
	}
}

func TestServiceUnreachableAfterReady(t *testing.T) {
	e := newEnv(t)
	m := e.launch(RunPod, nil)
	e.clouds[RunPod].running("pod1")
	if m = e.refresh(m.ID); m.Status != StatusReady {
		t.Fatal(m.Status)
	}
	e.prober.setDown(true)
	e.clock.Advance(4 * time.Minute)
	if m = e.refresh(m.ID); m.Status != StatusReady || problemOf(m) != "" {
		t.Fatalf("within 5 min: %+v", m)
	}
	e.clock.Advance(2 * time.Minute)
	if m = e.refresh(m.ID); m.Status != StatusRunning || problemOf(m) != ProblemUnreachable || !strings.Contains(m.Detail, "stopped answering") {
		t.Fatalf("after 5 min: %+v", m)
	}
	e.prober.setDown(false)
	if m = e.refresh(m.ID); m.Status != StatusReady || problemOf(m) != "" {
		t.Fatalf("back: %+v", m)
	}
}

func TestServiceNeverListed(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[QuickPod]
	c.noID = true
	m := e.launch(QuickPod, nil)
	if m.ProviderID != "" || m.Status != StatusCreating {
		t.Fatalf("%+v", m)
	}
	label := m.Label
	c.mu.Lock()
	c.insts["pod1"].Label = "someone-else" // not listed under our label
	c.mu.Unlock()
	e.clock.Advance(4 * time.Minute)
	if m = e.refresh(m.ID); m.Status != StatusCreating {
		t.Fatalf("within 5 min: %+v", m)
	}
	e.clock.Advance(2 * time.Minute)
	m = e.refresh(m.ID)
	if m.Status != StatusFailed || problemOf(m) != ProblemNeverListed || m.EndedAt != nil || m.Detail != detailNeverListed ||
		!slices.Equal(m.Actions, []string{"forget"}) {
		t.Fatalf("after 5 min: %+v", m)
	}
	c.mu.Lock()
	c.insts["pod1"].Label = label // it appears after all
	c.mu.Unlock()
	c.running("pod1")
	if m = e.refresh(m.ID); m.ProviderID != "pod1" || m.Status != StatusReady {
		t.Fatalf("resolved: %+v", m)
	}
}

func TestServiceNeverListedNamesProvider(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[RunPod]
	c.noID = true // a create whose outcome was never recorded
	m := e.launch(RunPod, nil)
	c.mu.Lock()
	c.insts["pod1"].Label = "someone-else"
	c.mu.Unlock()
	e.clock.Advance(6 * time.Minute)
	m = e.refresh(m.ID)
	if problemOf(m) != ProblemNeverListed || !strings.Contains(m.Detail, "RunPod") || strings.Contains(m.Detail, "QuickPod") {
		t.Fatalf("never_listed on RunPod: %+v", m)
	}
}

func TestServiceQuickPodResolvedByLabel(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[QuickPod]
	c.noID = true
	m := e.launch(QuickPod, nil)
	if _, err := e.s.Terminate(e.ctx, m.ID, false); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("terminate before it is listed: %v", err)
	}
	c.set("pod1", func(in *Instance) { in.Status = StatusStarting })
	e.clock.Advance(11 * time.Second)
	e.s.Tick(e.ctx)
	if r := e.row(m.ID); r.ProviderID != "pod1" || r.Status != StatusStarting {
		t.Fatalf("row %+v", r)
	}
}

func TestServiceSimulateSplit(t *testing.T) {
	e := newEnv(t)
	real := e.launch(Vast, nil)
	Simulated.Store(true)
	SetSimSecret("simsecret0000")
	t.Cleanup(func() { SetSimSecret("") })
	m := e.refresh(real.ID)
	if problemOf(m) != ProblemNotRefreshed || len(m.Actions) != 0 {
		t.Fatalf("real row in simulate: %+v", m)
	}
	gets := e.clouds[Vast].n("Get")
	if _, err := e.s.Stop(e.ctx, real.ID); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("act on a real row in simulate: %v", err)
	}
	if e.clouds[Vast].n("Get") != gets || e.clouds[Vast].n("Stop") != 0 {
		t.Fatal("a real row was sent to the provider while simulating")
	}
	sim := e.launch(RunPod, nil)
	if !sim.Simulated {
		t.Fatal("not flagged simulated")
	}
	Simulated.Store(false)
	m = e.refresh(sim.ID)
	if m.Status != StatusTerminated || m.EndedAt == nil || m.Detail != detailSimEnded {
		t.Fatalf("simulated row after restart: %+v", m)
	}
	if e.clouds[RunPod].n("Get") != 0 {
		t.Fatal("a simulated row reached the provider")
	}
	if m = e.refresh(real.ID); problemOf(m) == ProblemNotRefreshed {
		t.Fatalf("the real row stays not_refreshed: %+v", m)
	}
}

func TestServiceSelfConfirm(t *testing.T) {
	e := newEnv(t)
	m := e.launch(RunPod, nil)
	t.Setenv("RUNPOD_POD_ID", "pod1")
	if got, _ := e.s.Get(e.ctx, m.ID, false); !got.Self {
		t.Fatal("not self")
	}
	if _, err := e.s.Terminate(e.ctx, m.ID, false); codeOf(err) != v1api.CodeInvalidRequest || !strings.Contains(err.Error(), "confirm_self") {
		t.Fatalf("got %v", err)
	}
	m, err := e.s.Terminate(e.ctx, m.ID, true)
	if err != nil || m.Status != StatusTerminating || e.clouds[RunPod].n("Terminate") != 1 {
		t.Fatalf("%+v %v", m, err)
	}
	e.clouds[RunPod].remove("pod1")
	if m = e.refresh(m.ID); m.Status != StatusTerminated || m.EndedAt == nil || m.Detail != "" {
		t.Fatalf("after terminate: %+v", m)
	}
}

func TestServiceForgetRules(t *testing.T) {
	e := newEnv(t)
	m := e.launch(Vast, nil)
	if err := e.s.Forget(e.ctx, m.ID, false); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("forget a live machine: %v", err)
	}
	if err := e.s.Forget(e.ctx, m.ID, true); err != nil {
		t.Fatalf("force: %v", err)
	}
	if _, err := e.s.Get(e.ctx, m.ID, false); codeOf(err) != v1api.CodeNotFound {
		t.Fatalf("still there: %v", err)
	}
	m = e.launch(Vast, nil)
	e.clouds[Vast].remove("pod2") // deleted in the provider console
	if got := e.refresh(m.ID); got.Status != StatusTerminated {
		t.Fatalf("gone outside: %+v", got)
	} else if !strings.Contains(got.Detail, "deleted outside") || !slices.Equal(got.Actions, []string{"forget"}) {
		t.Fatalf("gone detail/actions: %+v", got)
	}
	ch, cancel := events.Subscribe(events.Filter{Types: map[string]bool{events.MachineDeleted: true}})
	defer cancel()
	if err := e.s.Forget(e.ctx, m.ID, false); err != nil {
		t.Fatalf("forget an ended machine: %v", err)
	}
	select {
	case ev := <-ch:
		if string(ev.Data) != `{"id":"`+m.ID+`"}` {
			t.Fatalf("data %s", ev.Data)
		}
	default:
		t.Fatal("no machine.deleted")
	}
}

func TestServiceStopStart(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[RunPod]
	m := e.launch(RunPod, nil)
	if _, err := e.s.Start(e.ctx, m.ID); codeOf(err) != v1api.CodeConflict {
		t.Fatalf("start while creating: %v", err)
	}
	c.running("pod1")
	e.refresh(m.ID)
	m, err := e.s.Stop(e.ctx, m.ID)
	if err != nil || m.Status != StatusStopping || slices.Contains(m.Actions, "stop") {
		t.Fatalf("stop: %+v %v", m, err)
	}
	if m = e.refresh(m.ID); m.Status != StatusStopping {
		t.Fatalf("provider still running: %s", m.Status)
	}
	c.set("pod1", func(in *Instance) { in.Status, in.Endpoints = StatusStopped, nil })
	if m = e.refresh(m.ID); m.Status != StatusStopped || m.Detail != "" || m.BaseURL != nil || !slices.Equal(m.Actions, []string{"start", "terminate"}) {
		t.Fatalf("stopped: %+v", m)
	}
	if m, err = e.s.Start(e.ctx, m.ID); err != nil || m.Status != StatusStarting {
		t.Fatalf("start: %+v %v", m, err)
	}
	if m = e.refresh(m.ID); m.Status != StatusStarting {
		t.Fatalf("provider not moved yet: %s", m.Status)
	}
	c.running("pod1")
	if m = e.refresh(m.ID); m.Status != StatusReady {
		t.Fatalf("restarted: %+v", m)
	}
	// stopped without our asking (when done, or the container exited)
	c.set("pod1", func(in *Instance) { in.Status, in.Endpoints = StatusStopped, nil })
	if m = e.refresh(m.ID); m.Status != StatusStopped || m.Detail != detailSelfStopped {
		t.Fatalf("self stopped: %+v", m)
	}
}

func TestServiceProviderError(t *testing.T) {
	e := newEnv(t)
	m := e.launch(Vast, nil)
	e.clouds[Vast].set("pod1", func(in *Instance) {
		in.Status, in.ProviderStatus, in.Detail = StatusFailed, "exited", "container crashed"
	})
	if m = e.refresh(m.ID); m.Status != StatusFailed || problemOf(m) != ProblemProvider || m.Detail != "container crashed" || m.EndedAt != nil {
		t.Fatalf("%+v", m)
	}
	// a problem the image checks set survives a refresh that finds nothing new
	r := e.row(m.ID)
	r.Status, r.Problem, r.Detail = StatusStarting, ProblemCUDAFailed, "cuda"
	if err := updateRow(e.ctx, r); err != nil {
		t.Fatal(err)
	}
	e.clouds[Vast].set("pod1", func(in *Instance) { in.Status, in.Detail = StatusStarting, "" })
	if m = e.refresh(m.ID); problemOf(m) != ProblemCUDAFailed {
		t.Fatalf("foreign problem cleared: %+v", m)
	}
}

func TestServiceGetRefreshCache(t *testing.T) {
	e := newEnv(t)
	m := e.launch(RunPod, nil)
	c := e.clouds[RunPod]
	e.clock.Advance(6 * time.Second)
	for range 3 {
		if _, err := e.s.Get(e.ctx, m.ID, true); err != nil {
			t.Fatal(err)
		}
	}
	if n := c.n("Get"); n != 1 {
		t.Fatalf("Get calls within 5 s: %d", n)
	}
	e.clock.Advance(5 * time.Second)
	e.s.Get(e.ctx, m.ID, true)
	if n := c.n("Get"); n != 2 {
		t.Fatalf("after 5 s: %d", n)
	}
	e.s.Get(e.ctx, m.ID, false)
	if n := c.n("Get"); n != 2 {
		t.Fatal("refresh=false called the provider")
	}
	c.getErr = Errorf(RunPod, ClassUnavailable, 503, "down")
	e.clock.Advance(6 * time.Second)
	if got, err := e.s.Get(e.ctx, m.ID, true); err != nil || got.Status != StatusCreating {
		t.Fatalf("an outage keeps the row: %+v %v", got, err)
	}
}

func TestServiceLogs(t *testing.T) {
	e := newEnv(t)
	m := e.launch(Vast, nil)
	c := e.clouds[Vast]
	c.logs = strings.Repeat("x", logsMax) + "tail"
	l, err := e.s.Logs(e.ctx, m.ID, 99999)
	if err != nil || !l.Truncated || len(l.Text) != logsMax || !strings.HasSuffix(l.Text, "tail") || l.Source != "vast" {
		t.Fatalf("logs %d %v %v", len(l.Text), l.Truncated, err)
	}
	c.logs = "short"
	if l, _ := e.s.Logs(e.ctx, m.ID, 0); l.Text != "short" || l.Truncated {
		t.Fatalf("%+v", l)
	}
}

// ---- events ----

func TestServiceEventsOnePerTransition(t *testing.T) {
	e := newEnv(t)
	c := e.clouds[RunPod]
	ch, cancel := events.Subscribe(events.Filter{Types: map[string]bool{events.MachineState: true}})
	defer cancel()
	m := e.launch(RunPod, nil)
	token := e.row(m.ID).Token
	e.refresh(m.ID) // creating -> creating: nothing
	c.set("pod1", func(in *Instance) { in.Status = StatusStarting })
	e.refresh(m.ID)
	e.refresh(m.ID) // no change
	c.running("pod1")
	e.refresh(m.ID) // starting -> ready
	e.refresh(m.ID)
	e.s.Terminate(e.ctx, m.ID, false)
	c.remove("pod1")
	e.refresh(m.ID)

	var statuses []Status
	for len(ch) > 0 {
		ev := <-ch
		if strings.Contains(string(ev.Data), token) {
			t.Fatalf("event carries the token: %s", ev.Data)
		}
		var got Machine
		if err := json.Unmarshal(ev.Data, &got); err != nil {
			t.Fatal(err)
		}
		statuses = append(statuses, got.Status)
	}
	want := []Status{StatusCreating, StatusStarting, StatusReady, StatusTerminating, StatusTerminated}
	if !slices.Equal(statuses, want) {
		t.Fatalf("events %v, want %v", statuses, want)
	}
}

func ptr[T any](v T) *T { return &v }
