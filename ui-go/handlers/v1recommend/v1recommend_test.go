package v1recommend

import (
	"cmp"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/handlers/v1machines"
	"github.com/geocine/aitoolkit/internal/codemode"
	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/imagetest"
	"github.com/geocine/aitoolkit/internal/provision/sim"
	"github.com/geocine/aitoolkit/internal/provision/vast"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

// fakeManager answers Providers and Offers, the only calls the recommender
// makes; every other call panics (the embedded nil Manager).
type fakeManager struct {
	provision.Manager
	views  []provision.ProviderView
	offers map[provision.Kind][]provision.Offer
	errs   map[provision.Kind]error

	mu      sync.Mutex
	queries []provision.OfferQuery
}

func (f *fakeManager) Providers(context.Context) ([]provision.ProviderView, error) {
	return f.views, nil
}

// Offers filters by the VRAM band and price like the adapters and sorts by
// price.
func (f *fakeManager) Offers(_ context.Context, k provision.Kind, q provision.OfferQuery) ([]provision.Offer, error) {
	f.mu.Lock()
	f.queries = append(f.queries, q)
	f.mu.Unlock()
	if err := f.errs[k]; err != nil {
		return nil, err
	}
	var out []provision.Offer
	for _, o := range f.offers[k] {
		v := o.Host.VRAMGB
		if (q.MinVRAMGB > 0 && v < float64(q.MinVRAMGB)) || (q.MaxVRAMGB > 0 && v >= float64(q.MaxVRAMGB)) ||
			(q.MaxPrice > 0 && o.PricePerHour > q.MaxPrice) {
			continue
		}
		out = append(out, o)
	}
	slices.SortStableFunc(out, func(a, b provision.Offer) int { return cmp.Compare(a.PricePerHour, b.PricePerHour) })
	return out, nil
}

func (f *fakeManager) lastQuery(t *testing.T) provision.OfferQuery {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.queries) == 0 {
		t.Fatal("no offers query was made")
	}
	return f.queries[len(f.queries)-1]
}

// newEnv serves the routes against a fake provision.Manager (configured
// kinds have a key) and the registry and Hugging Face mocks.
func newEnv(t *testing.T, configured ...provision.Kind) (*fakeManager, http.Handler) {
	t.Helper()
	t.Setenv("AITK_PROVISION_PROVIDERS", "")
	imagetest.Use(t, imagetest.Options{})
	f := &fakeManager{offers: map[provision.Kind][]provision.Offer{}, errs: map[provision.Kind]error{}}
	for _, k := range []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod} {
		reports := k != provision.RunPod
		f.views = append(f.views, provision.ProviderView{
			KindInfo:   provision.KindInfo{Kind: k, Label: provision.Label(k), Caps: provision.Caps{ReportsInet: reports, ReportsDiskBW: reports}},
			Configured: slices.Contains(configured, k),
		})
	}
	old := provision.Default
	provision.Default = f
	t.Cleanup(func() { provision.Default = old })
	mux := http.NewServeMux()
	Register(mux)
	return f, mux
}

// offer is a decorated single-GPU offer that qualifies: fast internet and
// disk, a CUDA 13.2 driver. RunPod reports no bandwidth and relies on its
// own CUDA placement, like the adapter.
func offer(k provision.Kind, id, gpu string, vram, price float64) provision.Offer {
	h := hostfit.Host{Provider: string(k), GPUName: gpu, GPUCount: 1, VRAMGB: vram, GPUVendor: "nvidia", CPUArch: "amd64",
		CUDAMax: "13.2", DiskGB: 500, InetDownMbps: 900, DiskBWMBps: 2000}
	rel := 0.99
	o := provision.Offer{ID: id, Provider: k, GPU: gpu, GPUCount: 1, VRAMGB: int(math.Round(vram)), PricePerHour: price,
		Reliability: &rel, Host: &h, Fit: &hostfit.OfferFit{Level: "ok", Codes: []string{}, VRAM: "unknown"}}
	if k == provision.RunPod {
		h.InetDownMbps, h.DiskBWMBps, h.DiskGB, h.CUDAMax, h.PlacementFloor = 0, 0, 0, "", true
		o.Reliability = nil
		o.Fit.Codes, o.Fit.Level = []string{"host_cuda_placement"}, "info"
	}
	return o
}

type response struct {
	code   int
	header http.Header
	body   []byte
}

func get(t *testing.T, h http.Handler, path string) response {
	t.Helper()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
	return response{w.Code, w.Header(), w.Body.Bytes()}
}

func (r response) errCode(t *testing.T) (code, message string) {
	t.Helper()
	var body struct {
		Error struct{ Code, Message string }
	}
	if err := json.Unmarshal(r.body, &body); err != nil {
		t.Fatalf("decode %s: %v", r.body, err)
	}
	return body.Error.Code, body.Error.Message
}

type pickView struct {
	Kind   string
	Reason string
	TierGB int `json:"tier_gb"`
	Offer  struct {
		ID       string
		Provider string
	}
}

type recView struct {
	Providers []string
	Default   string
	TierGB    int `json:"tier_gb"`
	Value     *pickView
	Speed     *pickView
	Notes     []struct{ Provider, Code, Message string }
	Message   string
}

func (r response) rec(t *testing.T) recView {
	t.Helper()
	if r.code != http.StatusOK {
		t.Fatalf("status %d: %s", r.code, r.body)
	}
	var v recView
	if err := json.Unmarshal(r.body, &v); err != nil {
		t.Fatalf("decode %s: %v", r.body, err)
	}
	return v
}

func (v recView) note(provider, code string) bool {
	return slices.ContainsFunc(v.Notes, func(n struct{ Provider, Code, Message string }) bool {
		return n.Provider == provider && n.Code == code
	})
}

// noNulls fails when an array or record the TS declares is null.
func noNulls(t *testing.T, body []byte) {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"providers", "tiers", "notes", "thresholds", "need"} {
		if m[k] == nil {
			t.Errorf("%s is null: %s", k, body)
		}
	}
	if th, _ := m["thresholds"].(map[string]any); th == nil || th["ladder"] == nil {
		t.Errorf("thresholds.ladder is null: %s", body)
	}
	tiers, _ := m["tiers"].([]any)
	for i, tr := range tiers {
		if tr.(map[string]any)["rejected"] == nil {
			t.Errorf("tiers[%d].rejected is null: %s", i, body)
		}
	}
}

func TestProviderRecommend(t *testing.T) {
	f, h := newEnv(t, provision.Vast)
	f.offers[provision.Vast] = []provision.Offer{
		offer(provision.Vast, "3090", "RTX 3090", 24, 0.20),
		offer(provision.Vast, "4090", "RTX 4090", 23.99, 0.40),
		offer(provision.Vast, "5090", "RTX 5090", 31.84, 0.90),
	}
	const image = "ghcr.io/geocine/aitoolkit-go:cu130"
	r := get(t, h, "/v1/providers/vast/recommend?model=flux&image="+image+"&disk_gb=150&max_price=2")
	noNulls(t, r.body)
	rec := r.rec(t)
	if rec.Default != "value" || rec.TierGB != 24 || rec.Value == nil || rec.Speed == nil ||
		rec.Value.Offer.ID != "3090" || rec.Speed.Offer.ID != "4090" || !slices.Equal(rec.Providers, []string{"vast"}) {
		t.Fatalf("recommendation %s", r.body)
	}
	if want := "RTX 3090 · 24 GB · $0.20/h · 900 Mbps · CUDA 13.2 OK"; rec.Value.Reason != want {
		t.Errorf("reason %q, want %q", rec.Value.Reason, want)
	}
	q := f.lastQuery(t)
	if q.Image != image || q.Model == nil || q.Model.Arch != "flux" || q.DiskGB < 150 || q.MaxPrice != 2 || !q.Fast || q.GPUCount != 1 {
		t.Errorf("offers query %+v", q)
	}
}

func TestProviderRecommendNothingQualifies(t *testing.T) {
	_, h := newEnv(t, provision.Vast)
	r := get(t, h, "/v1/providers/vast/recommend?disk_gb=2000")
	noNulls(t, r.body)
	rec := r.rec(t)
	if rec.Value != nil || rec.Speed != nil || rec.Default != "" || rec.TierGB != 0 ||
		rec.Message != "No offers match these filters (disk 2000 GB). Change the filters or pick a provider with more stock." {
		t.Fatalf("recommendation %s", r.body)
	}
}

func TestProviderRecommendErrors(t *testing.T) {
	f, h := newEnv(t, provision.Vast)
	cases := []struct {
		path, code, message string
	}{
		{"/v1/providers/foo/recommend", "not_found", "unknown provider"},
		{"/v1/providers/runpod/recommend", "invalid_request", "Add a RunPod API key first"},
		{"/v1/providers/vast/recommend?disk_gb=5", "invalid_request", "disk_gb must be a whole number from 20 to 2000"},
		{"/v1/providers/vast/recommend?disk_gb=lots", "invalid_request", "disk_gb must be a whole number from 20 to 2000"},
		{"/v1/providers/vast/recommend?max_price=0", "invalid_request", "max_price must be a price above 0 ($/h)"},
		{"/v1/providers/vast/recommend?cloud=private", "invalid_request", "cloud must be secure or community"},
		{"/v1/providers/vast/recommend?image=a%20b", "invalid_request", "image must be an image reference without spaces"},
	}
	for _, c := range cases {
		r := get(t, h, c.path)
		if code, msg := r.errCode(t); code != c.code || msg != c.message {
			t.Errorf("%s: %d %s %q, want %s %q", c.path, r.code, code, msg, c.code, c.message)
		}
	}
	if len(f.queries) != 0 {
		t.Errorf("a refused request searched offers: %+v", f.queries)
	}

	// a disabled provider is unknown, like on every /v1/providers/{kind} route
	t.Setenv("AITK_PROVISION_PROVIDERS", "runpod")
	if r := get(t, h, "/v1/providers/vast/recommend"); r.code != http.StatusNotFound {
		t.Errorf("disabled provider: %d %s", r.code, r.body)
	}
}

// A provider's 429 is the per-provider answer, with Retry-After; across
// providers it is a note and the others still give picks.
func TestRecommendRateLimited(t *testing.T) {
	f, h := newEnv(t, provision.RunPod, provision.Vast)
	e := provision.Errorf(provision.Vast, provision.ClassRateLimited, 429, "slow down")
	e.RetryAfter = 30 * time.Second
	f.errs[provision.Vast] = e
	f.offers[provision.RunPod] = []provision.Offer{offer(provision.RunPod, "NVIDIA GeForce RTX 4090|SECURE|COMMUNITY|1", "RTX 4090", 24, 0.69)}

	r := get(t, h, "/v1/providers/vast/recommend?model=flux")
	if code, _ := r.errCode(t); r.code != http.StatusTooManyRequests || code != "rate_limited" || r.header.Get("Retry-After") != "30" {
		t.Fatalf("per provider: %d %s Retry-After %q", r.code, r.body, r.header.Get("Retry-After"))
	}

	r = get(t, h, "/v1/hosts/recommend?model=flux")
	noNulls(t, r.body)
	rec := r.rec(t)
	if r.header.Get("Retry-After") != "" || !rec.note("vast", "rate_limited") || rec.Value == nil || rec.Value.Offer.Provider != "runpod" {
		t.Fatalf("across providers: %s", r.body)
	}
	if want := "RTX 4090 on RunPod · 24 GB · $0.69/h · net not reported · CUDA 13.0+ by RunPod"; rec.Value.Reason != want {
		t.Errorf("reason %q, want %q", rec.Value.Reason, want)
	}
}

func TestHostsRecommend(t *testing.T) {
	f, h := newEnv(t, provision.RunPod, provision.Vast)
	f.offers[provision.RunPod] = []provision.Offer{offer(provision.RunPod, "rp-4090", "RTX 4090", 24, 0.69)}
	f.offers[provision.Vast] = []provision.Offer{offer(provision.Vast, "v-4090", "RTX 4090", 23.99, 0.34)}

	rec := get(t, h, "/v1/hosts/recommend").rec(t)
	if !slices.Equal(rec.Providers, []string{"runpod", "vast"}) || rec.Value == nil || rec.Value.Offer.ID != "v-4090" {
		t.Fatalf("all providers: %+v", rec)
	}
	if want := "RTX 4090 on Vast.ai · 24 GB · $0.34/h · 900 Mbps · CUDA 13.2 OK"; rec.Value.Reason != want {
		t.Errorf("reason %q, want %q", rec.Value.Reason, want)
	}

	// the comma list narrows the search; QuickPod has no key and is a note
	rec = get(t, h, "/v1/hosts/recommend?providers=runpod,%20quickpod&cloud=community").rec(t)
	if !slices.Equal(rec.Providers, []string{"runpod"}) || rec.Value == nil || rec.Value.Offer.ID != "rp-4090" || !rec.note("quickpod", "not_configured") {
		t.Fatalf("providers=runpod,quickpod: %+v", rec)
	}
	if q := f.lastQuery(t); q.Cloud != "community" {
		t.Errorf("cloud not passed: %+v", q)
	}

	for path, want := range map[string]string{
		"/v1/hosts/recommend?providers=foo":               `providers: "foo" is not a provider (runpod, vast or quickpod)`,
		"/v1/hosts/recommend?providers=vast,foo":          `providers: "foo" is not a provider (runpod, vast or quickpod)`,
		"/v1/hosts/recommend?providers=quickpod":          "Add a QuickPod API key first",
		"/v1/hosts/recommend?max_price=-1":                "max_price must be a price above 0 ($/h)",
		"/v1/hosts/recommend?providers=vast&disk_gb=3000": "disk_gb must be a whole number from 20 to 2000",
	} {
		r := get(t, h, path)
		if code, msg := r.errCode(t); r.code != http.StatusBadRequest || code != "invalid_request" || msg != want {
			t.Errorf("%s: %d %s %q, want %q", path, r.code, code, msg, want)
		}
	}

	// a provider turned off on this server is a note beside others, and a
	// 400 (not the per-provider 404) when it is all that was listed
	t.Setenv("AITK_PROVISION_PROVIDERS", "runpod,vast")
	rec = get(t, h, "/v1/hosts/recommend?providers=runpod,quickpod").rec(t)
	if !slices.Equal(rec.Providers, []string{"runpod"}) || rec.Value == nil || !rec.note("quickpod", "not_configured") {
		t.Fatalf("providers=runpod,quickpod with quickpod off: %+v", rec)
	}
	for path, want := range map[string]string{
		"/v1/hosts/recommend?providers=quickpod":          `providers: "quickpod" isn't enabled on this server`,
		"/v1/hosts/recommend?providers=quickpod,quickpod": `providers: "quickpod" isn't enabled on this server`,
	} {
		r := get(t, h, path)
		if code, msg := r.errCode(t); r.code != http.StatusBadRequest || code != "invalid_request" || msg != want {
			t.Errorf("%s: %d %s %q, want %q", path, r.code, code, msg, want)
		}
	}
	t.Setenv("AITK_PROVISION_PROVIDERS", "runpod")
	if r := get(t, h, "/v1/hosts/recommend?providers=vast,quickpod"); r.code != http.StatusBadRequest ||
		!strings.Contains(string(r.body), `providers: \"vast\", \"quickpod\" aren't enabled on this server`) {
		t.Errorf("two providers off: %d %s", r.code, r.body)
	}
}

func TestHostsRecommendNoKey(t *testing.T) {
	_, h := newEnv(t)
	r := get(t, h, "/v1/hosts/recommend?model=flux")
	if code, msg := r.errCode(t); r.code != http.StatusBadRequest || code != "invalid_request" ||
		msg != "Add an API key for RunPod, Vast.ai or QuickPod first" {
		t.Fatalf("%d %s", r.code, r.body)
	}
}

// When every provider searched fails, the answer is still 200: the notes
// and the message say why.
func TestHostsRecommendAllFailed(t *testing.T) {
	f, h := newEnv(t, provision.Vast)
	f.errs[provision.Vast] = provision.Errorf(provision.Vast, provision.ClassUnavailable, 503, "bad gateway")
	r := get(t, h, "/v1/hosts/recommend")
	noNulls(t, r.body)
	rec := r.rec(t)
	if rec.Value != nil || !rec.note("vast", "remote_failed") ||
		rec.Message != "No provider could be searched right now: Vast.ai did not answer: bad gateway." {
		t.Fatalf("%s", r.body)
	}
}

// An agent in simulate mode: providers.recommend and hosts.recommend (every
// provider has a simulated key) through Auth, the real provisioning
// service, the Vast mock (decorated offers) and the simulated registry.
func TestCodeModeRecommend(t *testing.T) {
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close() })
	const admin = "admin-token-for-recommend"
	for _, k := range []string{"AITK_PROVISION_PROVIDERS", "AITK_PROVISION_SIMULATE_REGISTRY"} {
		t.Setenv(k, "")
	}
	t.Setenv("AI_TOOLKIT_AUTH", admin)
	for _, k := range []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod} {
		t.Setenv(provision.EnvKeyName(k), "")
	}
	oldPace := vast.Pace
	vast.Pace = vast.NewLimiter(&paceClock{t: time.Now()})
	t.Cleanup(func() { vast.Pace = oldPace })

	ctx, cancel := context.WithCancel(context.Background())
	stop, err := sim.Start(ctx, http.NewServeMux())
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() { stop(); cancel() })

	mux := http.NewServeMux()
	v1machines.Register(mux)
	Register(mux)
	res := codemode.Run(context.Background(), `
await aitk.providers.setKey("vast", { api_key: "sim-key" });
const rec = await aitk.providers.recommend("vast", { model: "flux" });
const all = await aitk.hosts.recommend({ model: "flux" });
return { value: rec.value, default: rec.default, tier: rec.tier_gb, providers: all.providers, all_value: all.value && all.value.offer.id };`,
		codemode.Options{Call: codemode.HandlerCaller(httpx.Auth(mux), "Bearer "+admin, ""), Timeout: 60 * time.Second})
	if res.Error != nil {
		t.Fatalf("%+v\nlogs: %s", res.Error, res.Logs)
	}
	var out struct {
		Value *struct {
			Reason string
			Offer  struct{ ID, Provider string }
		}
		Default   string
		Tier      int
		Providers []string
		AllValue  string `json:"all_value"`
	}
	if err := json.Unmarshal(res.Result, &out); err != nil {
		t.Fatal(err)
	}
	if out.Value == nil || out.Value.Offer.Provider != "vast" || out.Value.Reason == "" || out.Default != "value" || out.Tier < 24 ||
		!slices.Contains(out.Providers, "vast") || out.AllValue == "" {
		t.Fatalf("flow result %s", res.Result)
	}
	if strings.Contains(string(res.Result), "sim-key") {
		t.Fatalf("the key leaked: %s", res.Result)
	}
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
