package v1machines

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/geocine/aitoolkit/internal/codemode"
	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/mcpserver"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/imagetest"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/sim"
	"github.com/geocine/aitoolkit/internal/provision/vast"
	"github.com/geocine/aitoolkit/internal/provision/vast/vastmock"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// fakeManager records what the handlers hand the service and answers with
// fixed shapes, or with err when set.
type fakeManager struct {
	mu          sync.Mutex
	calls       []string
	err         error
	created     bool
	query       provision.OfferQuery
	launch      provision.LaunchRequest
	origin      string
	update      provision.MachinePatch
	refresh     bool
	active      bool
	confirmSelf bool
	force       bool
	tail        int
	kind        provision.Kind
	id          string
}

func (f *fakeManager) note(call string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, call)
	return f.err
}

func (f *fakeManager) last() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.calls) == 0 {
		return ""
	}
	return f.calls[len(f.calls)-1]
}

func machine(id string) provision.Machine {
	return provision.Machine{ID: id, Name: "RTX 4090 on RunPod", Provider: provision.RunPod, Status: provision.StatusReady,
		Actions: []string{"stop", "terminate"}, Warnings: []string{}, ImageWarnings: []string{}}
}

func (f *fakeManager) Providers(context.Context) ([]provision.ProviderView, error) {
	return []provision.ProviderView{{KindInfo: provision.KindInfo{Kind: provision.RunPod, Label: "RunPod"}}}, f.note("Providers")
}
func (f *fakeManager) SetKey(_ context.Context, k provision.Kind, in provision.ProviderUpdate) (provision.ProviderView, error) {
	f.kind = k
	return provision.ProviderView{KindInfo: provision.KindInfo{Kind: k}, Configured: true}, f.note("SetKey")
}
func (f *fakeManager) RemoveKey(_ context.Context, k provision.Kind) error {
	f.kind = k
	return f.note("RemoveKey")
}
func (f *fakeManager) Test(_ context.Context, k provision.Kind) (provision.ProviderTest, error) {
	f.kind = k
	return provision.ProviderTest{OK: false, Detail: "key rejected"}, f.note("Test")
}
func (f *fakeManager) Offers(_ context.Context, k provision.Kind, q provision.OfferQuery) ([]provision.Offer, error) {
	f.kind, f.query = k, q
	return []provision.Offer{{ID: "o1", Provider: k, GPU: "RTX 4090", GPUCount: 1, VRAMGB: 24, PricePerHour: 0.4}}, f.note("Offers")
}
func (f *fakeManager) Quote(_ context.Context, k provision.Kind, id string, q provision.OfferQuery) (provision.Offer, error) {
	return provision.Offer{ID: id, Provider: k}, f.note("Quote")
}
func (f *fakeManager) Templates(_ context.Context, k provision.Kind) ([]provision.Template, error) {
	f.kind = k
	return nil, f.note("Templates")
}
func (f *fakeManager) Launch(_ context.Context, req provision.LaunchRequest, origin string) (provision.Machine, bool, error) {
	f.launch, f.origin = req, origin
	return machine("mch_1"), f.created, f.note("Launch")
}
func (f *fakeManager) List(_ context.Context, active bool) ([]provision.Machine, error) {
	f.active = active
	return []provision.Machine{machine("mch_1")}, f.note("List")
}
func (f *fakeManager) Get(_ context.Context, id string, refresh bool) (provision.Machine, error) {
	f.id, f.refresh = id, refresh
	return machine(id), f.note("Get")
}
func (f *fakeManager) Update(_ context.Context, id string, p provision.MachinePatch) (provision.Machine, error) {
	f.id, f.update = id, p
	return machine(id), f.note("Update")
}
func (f *fakeManager) Stop(_ context.Context, id string) (provision.Machine, error) {
	f.id = id
	return machine(id), f.note("Stop")
}
func (f *fakeManager) Start(_ context.Context, id string) (provision.Machine, error) {
	f.id = id
	return machine(id), f.note("Start")
}
func (f *fakeManager) Terminate(_ context.Context, id string, confirmSelf bool) (provision.Machine, error) {
	f.id, f.confirmSelf = id, confirmSelf
	return machine(id), f.note("Terminate")
}
func (f *fakeManager) Forget(_ context.Context, id string, force bool) error {
	f.id, f.force = id, force
	return f.note("Forget")
}
func (f *fakeManager) Link(_ context.Context, id string) (provision.MachineLink, error) {
	f.id = id
	return provision.MachineLink{Name: "x", BaseURL: "https://x", Token: "aitk_link", RelayPath: "/v1/machines/" + id + "/relay"}, f.note("Link")
}
func (f *fakeManager) Logs(_ context.Context, id string, tail int) (provision.MachineLogs, error) {
	f.id, f.tail = id, tail
	return provision.MachineLogs{Text: "hello", Source: "runpod"}, f.note("Logs")
}
func (f *fakeManager) RelayTarget(_ context.Context, id string) (provision.RelayTarget, error) {
	f.id = id
	return provision.RelayTarget{MachineID: id, BaseURL: "http://127.0.0.1:1"}, f.note("RelayTarget")
}
func (f *fakeManager) Run(context.Context) {}

// openState opens a fresh fork database (machines, settings, keys) and
// clears the env the guard and the providers read.
func openState(t *testing.T) {
	t.Helper()
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close() })
	for _, k := range []string{"AI_TOOLKIT_AUTH", "AITK_PROVISION_PROVIDERS", "AITK_PROVISION_MAX_RUNNING", "RUNPOD_POD_ID", "CONTAINER_ID", "CONTAINER_LABEL"} {
		t.Setenv(k, "")
	}
	for _, k := range []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod} {
		t.Setenv(provision.EnvKeyName(k), "")
	}
}

func newEnv(t *testing.T) (*fakeManager, http.Handler) {
	t.Helper()
	openState(t)
	f := &fakeManager{created: true}
	old := provision.Default
	provision.Default = f
	t.Cleanup(func() { provision.Default = old })
	mux := http.NewServeMux()
	Register(mux)
	return f, mux
}

// call serves one request from a loopback peer unless an option changes it.
func call(t *testing.T, h http.Handler, method, path, body string, opts ...func(*http.Request)) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	r.RemoteAddr = "127.0.0.1:40000"
	for _, o := range opts {
		o(r)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func fromRemote(r *http.Request) { r.RemoteAddr = "203.0.113.9:5555" }

func errCode(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var body struct {
		Error struct{ Code, Message string }
	}
	json.Unmarshal(w.Body.Bytes(), &body)
	return body.Error.Code
}

func TestRoutes(t *testing.T) {
	f, h := newEnv(t)
	for _, c := range []struct {
		method, path, body string
		status             int
		call, shape        string // the Manager method; a JSON key the body has ("" = no body)
	}{
		{"GET", "/v1/providers", "", 200, "Providers", "items"},
		{"PUT", "/v1/providers/runpod", `{"api_key":"k_123456"}`, 200, "SetKey", "configured"},
		{"DELETE", "/v1/providers/vast", "", 204, "RemoveKey", ""},
		{"POST", "/v1/providers/quickpod/test", "", 200, "Test", "detail"},
		{"GET", "/v1/providers/runpod/offers", "", 200, "Offers", "items"},
		{"GET", "/v1/providers/quickpod/templates", "", 200, "Templates", "items"},
		{"GET", "/v1/machines?active=1", "", 200, "List", "items"},
		{"POST", "/v1/machines", `{"provider":"runpod","offer_id":"o1","max_price_per_hour":0.5}`, 201, "Launch", "id"},
		{"GET", "/v1/machines/mch_1?refresh=1", "", 200, "Get", "status"},
		{"PATCH", "/v1/machines/mch_1", `{"name":"big one"}`, 200, "Update", "id"},
		{"POST", "/v1/machines/mch_1/stop", "", 200, "Stop", "actions"},
		{"POST", "/v1/machines/mch_1/start", "", 200, "Start", "actions"},
		{"POST", "/v1/machines/mch_1/terminate", "", 200, "Terminate", "status"},
		{"DELETE", "/v1/machines/mch_1?force=1", "", 204, "Forget", ""},
		{"POST", "/v1/machines/mch_1/link", "", 200, "Link", "token"},
		{"GET", "/v1/machines/mch_1/logs?tail=50", "", 200, "Logs", "text"},
	} {
		w := call(t, h, c.method, c.path, c.body)
		if w.Code != c.status || f.last() != c.call {
			t.Errorf("%s %s: %d via %s, want %d via %s (%s)", c.method, c.path, w.Code, f.last(), c.status, c.call, w.Body)
			continue
		}
		if c.shape == "" {
			if w.Body.Len() != 0 {
				t.Errorf("%s %s: body %q, want none", c.method, c.path, w.Body)
			}
			continue
		}
		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Errorf("%s %s: %v", c.method, c.path, err)
		} else if _, ok := body[c.shape]; !ok {
			t.Errorf("%s %s: no %q in %s", c.method, c.path, c.shape, w.Body)
		}
	}
	// lists are {"items": [...]}, never null
	if w := call(t, h, "GET", "/v1/providers/quickpod/templates", ""); strings.TrimSpace(w.Body.String()) != `{"items":[]}` {
		t.Errorf("empty list = %s", w.Body)
	}
	// the values reach the service
	if !f.active || f.id != "mch_1" || !f.force || f.tail != 50 || f.update.Name == nil || *f.update.Name != "big one" {
		t.Errorf("params lost: %+v", f)
	}
	call(t, h, "GET", "/v1/machines/mch_2", "")
	call(t, h, "GET", "/v1/machines", "")
	if f.refresh || f.active || f.id != "mch_2" {
		t.Errorf("refresh/active default to false: %+v", f)
	}
}

func TestLaunchCreatedOrRepeated(t *testing.T) {
	f, h := newEnv(t)
	body := `{"provider":"vast","offer_id":"o9","max_price_per_hour":0.75,"request_id":"req_abcdefgh","skip_image_check":true,"when_done":"destroy","copy_remote":"r2"}`
	w := call(t, h, "POST", "/v1/machines", body, func(r *http.Request) { r.Header.Set("Origin", "https://studio.example") })
	if w.Code != 201 {
		t.Fatalf("first launch %d %s", w.Code, w.Body)
	}
	want := provision.LaunchRequest{Provider: provision.Vast, OfferID: "o9", MaxPricePerHour: 0.75, RequestID: "req_abcdefgh",
		SkipImageCheck: true, WhenDone: provision.WhenDestroy, CopyRemote: "r2"}
	if !reflect.DeepEqual(f.launch, want) || f.origin != "https://studio.example" {
		t.Fatalf("launch got %+v origin %q", f.launch, f.origin)
	}
	f.created = false
	if w := call(t, h, "POST", "/v1/machines", body); w.Code != 200 {
		t.Fatalf("repeated request_id %d %s", w.Code, w.Body)
	}
	if w := call(t, h, "POST", "/v1/machines", `{"provider":`); w.Code != 400 || errCode(t, w) != v1api.CodeInvalidRequest {
		t.Fatalf("bad JSON %d %s", w.Code, w.Body)
	}
}

func TestUnknownProvider(t *testing.T) {
	f, h := newEnv(t)
	for _, p := range []string{"/v1/providers/nope/offers", "/v1/providers/RunPod/templates", "/v1/providers/x/setup"} {
		if w := call(t, h, "GET", p, ""); w.Code != 404 || errCode(t, w) != v1api.CodeNotFound || !strings.Contains(w.Body.String(), "unknown provider") {
			t.Errorf("%s: %d %s", p, w.Code, w.Body)
		}
	}
	// the kill switch disables a kind like an unknown one
	t.Setenv("AITK_PROVISION_PROVIDERS", "runpod,vast")
	if w := call(t, h, "PUT", "/v1/providers/quickpod", `{"api_key":"abcdefgh"}`); w.Code != 404 {
		t.Errorf("disabled quickpod: %d %s", w.Code, w.Body)
	}
	if w := call(t, h, "GET", "/v1/providers/vast/offers", ""); w.Code != 200 || f.kind != provision.Vast {
		t.Errorf("enabled vast: %d %s", w.Code, w.Body)
	}
}

// detailed is an error that carries details (like a blocked image check).
type detailed struct{ checks []string }

func (d *detailed) Error() string { return "blocked" }
func (d *detailed) V1Details() (string, string, any) {
	return v1api.CodeImageBlocked, "This image can't run on that host.", map[string]any{"checks": d.checks}
}

func TestErrorMapping(t *testing.T) {
	f, h := newEnv(t)
	for _, c := range []struct {
		err    error
		status int
		code   string
	}{
		{provision.Errorf(provision.RunPod, provision.ClassAuth, 401, "bad key"), 400, v1api.CodeProviderAuth},
		{provision.Errorf(provision.Vast, provision.ClassCredit, 402, "balance"), 402, v1api.CodeInsufficientCredit},
		{provision.Errorf(provision.Vast, provision.ClassCapacity, 409, "taken"), 409, v1api.CodeNoCapacity},
		{provision.Errorf(provision.Vast, provision.ClassConflict, 0, "The price is now $0.52/h"), 409, v1api.CodeConflict},
		{provision.Errorf(provision.QuickPod, provision.ClassInvalid, 400, "disk"), 400, v1api.CodeInvalidRequest},
		{provision.Errorf(provision.QuickPod, provision.ClassNotFound, 404, "gone"), 404, v1api.CodeNotFound},
		{&provision.Error{Provider: provision.Vast, Class: provision.ClassRateLimited, Status: 429, RetryAfter: 7 * time.Second}, 429, v1api.CodeRateLimited},
		{provision.Errorf(provision.RunPod, provision.ClassUnavailable, 503, "down"), 502, v1api.CodeRemoteFailed},
		{v1api.Errorf(v1api.CodeInvalidConfig, "When done needs a remote so results are kept"), 422, v1api.CodeInvalidConfig},
		{errors.New("disk on fire"), 500, v1api.CodeInternal},
	} {
		f.err = c.err
		w := call(t, h, "POST", "/v1/machines", `{"provider":"vast","offer_id":"o1","max_price_per_hour":1}`)
		if w.Code != c.status || errCode(t, w) != c.code {
			t.Errorf("%v: %d %s, want %d %s", c.err, w.Code, w.Body, c.status, c.code)
		}
		if c.code == v1api.CodeRateLimited && w.Header().Get("Retry-After") != "7" {
			t.Errorf("Retry-After %q, want 7", w.Header().Get("Retry-After"))
		}
	}
	// an error with details answers with them (422 image_blocked)
	f.err = &detailed{checks: []string{"host_cuda"}}
	w := call(t, h, "POST", "/v1/machines", `{"provider":"vast","offer_id":"o1","max_price_per_hour":1}`)
	var body struct {
		Error struct {
			Code    string
			Details struct{ Checks []string }
		}
	}
	json.Unmarshal(w.Body.Bytes(), &body)
	if w.Code != 422 || body.Error.Code != v1api.CodeImageBlocked || len(body.Error.Details.Checks) != 1 {
		t.Errorf("detailed: %d %s", w.Code, w.Body)
	}
}

func TestOfferQueryParams(t *testing.T) {
	f, h := newEnv(t)
	w := call(t, h, "GET", "/v1/providers/vast/offers?gpu=4090&min_vram_gb=24&max_price=0.8&gpu_count=2&cloud=community&disk_gb=150"+
		"&limit=200&image=ghcr.io/geocine/aitoolkit-go:abc1234&model=flux&show_all=1&fast=true&max_vram_gb=48", "")
	if w.Code != 200 {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	want := provision.OfferQuery{GPU: "4090", MinVRAMGB: 24, MaxPrice: 0.8, GPUCount: 2, Cloud: "community", DiskGB: 150, Limit: 200,
		Image: "ghcr.io/geocine/aitoolkit-go:abc1234", Model: &hostfit.ModelRef{Arch: "flux"}, ShowAll: true, Fast: true, MaxVRAMGB: 48}
	if !reflect.DeepEqual(f.query, want) {
		t.Fatalf("query\n got %+v\nwant %+v", f.query, want)
	}
	call(t, h, "GET", "/v1/providers/vast/offers?show_all=0&fast=no", "")
	if !reflect.DeepEqual(f.query, provision.OfferQuery{}) {
		t.Errorf("defaults: %+v", f.query)
	}
	for _, q := range []string{"min_vram_gb=abc", "max_price=0", "max_price=-1", "limit=500", "limit=0", "gpu_count=0", "disk_gb=5", "cloud=moon", "max_vram_gb=-1", "image=a%20b"} {
		if w := call(t, h, "GET", "/v1/providers/vast/offers?"+q, ""); w.Code != 400 || errCode(t, w) != v1api.CodeInvalidRequest {
			t.Errorf("%s: %d %s", q, w.Code, w.Body)
		}
	}
}

func TestSetupGuide(t *testing.T) {
	_, h := newEnv(t)
	var g provision.SetupGuide
	w := call(t, h, "GET", "/v1/providers/vast/setup?disk_gb=150&cors_origin=https://studio.example&generate_token=1", "")
	if err := json.Unmarshal(w.Body.Bytes(), &g); err != nil || w.Code != 200 {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	if g.Kind != provision.Vast || g.DiskGB != 150 || !strings.HasPrefix(g.Token, "aitk_") || !strings.Contains(g.Env[0].Value, g.Token) {
		t.Errorf("guide %+v", g)
	}
	// it never takes a token from the caller
	w = call(t, h, "GET", "/v1/providers/runpod/setup?token=aitk_mine_1234567", "")
	if w.Code != 200 || strings.Contains(w.Body.String(), "aitk_mine_1234567") || strings.Contains(w.Body.String(), `"token"`) {
		t.Errorf("token param: %d %s", w.Code, w.Body)
	}
	for _, q := range []string{"disk_gb=5", "cors_origin=nope", "image=a%3Bb"} {
		if w := call(t, h, "GET", "/v1/providers/runpod/setup?"+q, ""); w.Code != 400 {
			t.Errorf("%s: %d %s", q, w.Code, w.Body)
		}
	}
}

func TestTerminateBody(t *testing.T) {
	f, h := newEnv(t)
	if w := call(t, h, "POST", "/v1/machines/mch_1/terminate", `{"confirm_self":true}`); w.Code != 200 || !f.confirmSelf {
		t.Fatalf("confirm_self: %d %v", w.Code, f.confirmSelf)
	}
	if w := call(t, h, "POST", "/v1/machines/mch_1/terminate", ""); w.Code != 200 || f.confirmSelf {
		t.Fatalf("empty body: %d %v", w.Code, f.confirmSelf)
	}
	if w := call(t, h, "POST", "/v1/machines/mch_1/terminate", `{"confirm_self":`); w.Code != 400 {
		t.Fatalf("bad JSON: %d", w.Code)
	}
}

// The open-server guard: without AI_TOOLKIT_AUTH or keys, only this machine
// may rent, start, link or relay.
func TestOpenServerGuard(t *testing.T) {
	f, h := newEnv(t)
	guarded := []struct{ method, path, body string }{
		{"PUT", "/v1/providers/runpod", `{"api_key":"abcdefgh"}`},
		{"POST", "/v1/machines", `{"provider":"runpod","offer_id":"o1","max_price_per_hour":1}`},
		{"POST", "/v1/machines/mch_1/start", ""},
		{"POST", "/v1/machines/mch_1/link", ""},
		{"GET", "/v1/machines/mch_1/relay/v1/system", ""},
		{"POST", "/v1/machines/mch_1/relay/v1/jobs", "{}"},
	}
	if !httpx.IsOpen() {
		t.Fatal("the test server is not open")
	}
	viaProxy := func(r *http.Request) { r.Header.Set("X-Forwarded-For", "198.51.100.4") }
	for _, c := range guarded {
		for name, opt := range map[string]func(*http.Request){"remote": fromRemote, "proxied": viaProxy} {
			f.calls = nil
			w := call(t, h, c.method, c.path, c.body, opt)
			if w.Code != 403 || errCode(t, w) != v1api.CodeForbidden || !strings.Contains(w.Body.String(), "Set AI_TOOLKIT_AUTH") || len(f.calls) != 0 {
				t.Errorf("%s %s %s: %d %s (calls %v)", name, c.method, c.path, w.Code, w.Body, f.calls)
			}
		}
		if w := call(t, h, c.method, c.path, c.body); w.Code == 403 {
			t.Errorf("loopback %s %s refused", c.method, c.path)
		}
	}
	// reads and stops are not guarded
	if w := call(t, h, "GET", "/v1/machines", "", fromRemote); w.Code != 200 {
		t.Errorf("remote list: %d", w.Code)
	}
	if w := call(t, h, "POST", "/v1/machines/mch_1/stop", "", fromRemote); w.Code != 200 {
		t.Errorf("remote stop: %d", w.Code)
	}
	// with a token configured the server is not open (Auth checks the token)
	t.Setenv("AI_TOOLKIT_AUTH", "admin-token-123")
	for _, c := range guarded[:4] {
		if w := call(t, h, c.method, c.path, c.body, fromRemote); w.Code == 403 {
			t.Errorf("auth on: %s %s refused", c.method, c.path)
		}
	}
}

func TestAuthenticated(t *testing.T) {
	openState(t)
	req := func(opts ...func(*http.Request)) *http.Request {
		r := httptest.NewRequest("GET", "/v1/machines/m/relay/api/img/a.png", nil)
		r.RemoteAddr = "127.0.0.1:1"
		for _, o := range opts {
			o(r)
		}
		return r
	}
	bearer := func(tok string) func(*http.Request) {
		return func(r *http.Request) { r.Header.Set("Authorization", "Bearer "+tok) }
	}
	if !authenticated(req()) || authenticated(req(fromRemote)) {
		t.Error("open server: only loopback is authenticated")
	}
	t.Setenv("AI_TOOLKIT_AUTH", "admin-token-123")
	if !authenticated(req(bearer("admin-token-123"), fromRemote)) {
		t.Error("the admin token is authenticated")
	}
	if authenticated(req()) || authenticated(req(bearer("admin-token-12"))) || authenticated(req(bearer(""))) {
		t.Error("no or wrong token is not authenticated, even from loopback")
	}
	_, key, err := state.CreateKey(context.Background(), "agent")
	if err != nil {
		t.Fatal(err)
	}
	if !authenticated(req(bearer(key), fromRemote)) {
		t.Error("an API key is authenticated")
	}
}

func TestRelayNotReady(t *testing.T) {
	f, h := newEnv(t)
	f.err = v1api.Errorf(v1api.CodeConflict, "x isn't ready yet")
	if w := call(t, h, "GET", "/v1/machines/mch_1/relay/v1/system", ""); w.Code != 409 || f.id != "mch_1" {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
}

type headerTransport struct{ header http.Header }

func (h headerTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	for k, v := range h.header {
		r.Header[k] = v
	}
	return http.DefaultTransport.RoundTrip(r)
}

// An agent on this machine rents through /mcp on an open server like the
// studio does; one from elsewhere, or through a proxy, is refused.
func TestMCPOpenServerGuard(t *testing.T) {
	f, h := newEnv(t)
	if !httpx.IsOpen() {
		t.Fatal("the test server is not open")
	}
	mcpserver.SetAPI(httpx.Auth(h))
	t.Cleanup(func() { mcpserver.SetAPI(nil) })
	mux := http.NewServeMux()
	mcpserver.Register(mux)
	remote := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if remote != "" {
			r.RemoteAddr = remote
		}
		mux.ServeHTTP(w, r)
	}))
	defer srv.Close()

	const code = `await aitk.providers.setKey("runpod", { api_key: "abcdefgh" });
return (await aitk.machines.launch({ provider: "runpod", offer_id: "o1", max_price_per_hour: 1 })).id;`
	run := func(header http.Header) (string, bool) {
		t.Helper()
		ctx := context.Background()
		client := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "1"}, nil)
		session, err := client.Connect(ctx, &mcp.StreamableClientTransport{
			Endpoint: srv.URL + "/mcp", HTTPClient: &http.Client{Transport: headerTransport{header}}, MaxRetries: -1,
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer session.Close()
		res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "aitk_run", Arguments: map[string]any{"code": code}})
		if err != nil {
			t.Fatal(err)
		}
		var b strings.Builder
		for _, c := range res.Content {
			if tc, ok := c.(*mcp.TextContent); ok {
				b.WriteString(tc.Text)
			}
		}
		return b.String(), res.IsError
	}

	if text, isErr := run(nil); isErr || !strings.Contains(text, `"mch_1"`) || strings.Join(f.calls, ",") != "SetKey,Launch" {
		t.Fatalf("loopback agent refused: %s (calls %v)", text, f.calls)
	}
	for name, c := range map[string]struct {
		remote string
		header http.Header
	}{
		"remote":      {remote: "203.0.113.9:5555"},
		"proxied":     {header: http.Header{"X-Forwarded-For": {"198.51.100.4"}}},
		"forged peer": {remote: "203.0.113.9:5555", header: http.Header{"X-Aitk-Peer": {"127.0.0.1:1"}}},
	} {
		f.calls, remote = nil, c.remote
		text, isErr := run(c.header)
		if !isErr || !strings.Contains(text, "Set AI_TOOLKIT_AUTH") || len(f.calls) != 0 {
			t.Errorf("%s agent: %s (calls %v)", name, text, f.calls)
		}
	}
}

// End to end in simulate mode, as an agent: offers -> launch -> wait -> link,
// through Auth, the real service, the RunPod adapter and its mock. The
// provider key and the machine token never leak into other responses.
func TestSimulatedAgentFlow(t *testing.T) {
	openState(t)
	const admin = "admin-token-for-e2e"
	t.Setenv("AI_TOOLKIT_AUTH", admin)
	t.Setenv("AITK_PROVISION_SIMULATE_DELAY", "0")

	nodeInner := http.NewServeMux()
	nodeInner.HandleFunc("GET /v1/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ok":true,"version":"test"}`))
	})
	nodeInner.HandleFunc("GET /v1/system", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"gpus":[]}`)) })
	ctx, cancel := context.WithCancel(context.Background())
	stop, err := sim.Start(ctx, nodeInner)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() { stop(); cancel() })

	mux := http.NewServeMux()
	Register(mux)
	type seen struct{ path, body string }
	var mu sync.Mutex
	var bodies []seen
	spy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := httptest.NewRecorder()
		httpx.Auth(mux).ServeHTTP(rec, r)
		mu.Lock()
		bodies = append(bodies, seen{r.Method + " " + r.URL.Path, rec.Body.String()})
		mu.Unlock()
		for k, v := range rec.Header() {
			w.Header()[k] = v
		}
		w.WriteHeader(rec.Code)
		w.Write(rec.Body.Bytes())
	})

	res := codemode.Run(context.Background(), `
await aitk.providers.setKey("runpod", { api_key: "TOP-SECRET-KEY" });
const providers = (await aitk.providers.list()).items;
const offers = (await aitk.providers.offers("runpod", { limit: 5 })).items;
const o = offers[0];
const req = { provider: "runpod", offer_id: o.id, max_price_per_hour: o.price_per_hour + 1, request_id: "e2e_req_0001" };
const m = await aitk.machines.launch(req);
const again = await aitk.machines.launch(req);
const ready = await aitk.machines.wait(m.id, { poll_ms: 250, timeout_ms: 30000 });
const link = await aitk.machines.link(m.id);
const logs = await aitk.machines.logs(m.id, { tail: 20 });
const list = await aitk.machines.list({ active: true });
const gone = await aitk.machines.terminate(m.id);
return { configured: providers.find(p => p.kind === "runpod").configured, n: offers.length, id: m.id, again: again.id,
  status: ready.status, problem: ready.problem, base: link.base_url, token: link.token, listed: list.items.length,
  logs: typeof logs.text, gone: gone.status };`,
		codemode.Options{Call: codemode.HandlerCaller(spy, "Bearer "+admin, ""), Timeout: 60 * time.Second})
	if res.Error != nil {
		t.Fatalf("%+v\nlogs: %s", res.Error, res.Logs)
	}
	var out struct {
		Configured        bool
		N                 int
		ID, Again, Status string
		Problem           *string
		Base, Token       string
		Listed            int
		Logs, Gone        string
	}
	if err := json.Unmarshal(res.Result, &out); err != nil {
		t.Fatal(err)
	}
	if !out.Configured || out.N == 0 || out.ID == "" || out.Again != out.ID || out.Status != "ready" || out.Problem != nil ||
		!strings.HasPrefix(out.Base, "http://127.0.0.1:") || !strings.HasPrefix(out.Token, "aitk_") || out.Listed != 1 || out.Logs != "string" ||
		out.Gone != string(provision.StatusTerminating) && out.Gone != string(provision.StatusTerminated) {
		t.Fatalf("flow result %s", res.Result)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, b := range bodies {
		if strings.Contains(b.body, "TOP-SECRET-KEY") {
			t.Errorf("%s leaked the provider key: %s", b.path, b.body)
		}
		if strings.Contains(b.body, out.Token) && !strings.HasSuffix(b.path, "/link") {
			t.Errorf("%s leaked the machine token: %s", b.path, b.body)
		}
	}
}

// nowClock paces the Vast limiter without waiting.
type nowClock struct{}

func (nowClock) Now() time.Time                             { return time.Now() }
func (nowClock) Sleep(context.Context, time.Duration) error { return nil }

// The offers route with image= runs the image check (registrymock through
// imagetest) and sends the image's host filter to Vast, server-side; every
// offer comes back with host and fit.
func TestOffersImageFilterReachesVast(t *testing.T) {
	openState(t)
	imagetest.Use(t, imagetest.Options{})
	m := vastmock.New(mockkit.Options{})
	t.Cleanup(m.Close)
	oldAPI, oldPace := vast.API, vast.Pace
	vast.API, vast.Pace = m.Start(), vast.NewLimiter(nowClock{})
	t.Cleanup(func() { vast.API, vast.Pace = oldAPI, oldPace })
	t.Setenv(provision.EnvKeyName(provision.Vast), "vast-key-for-the-offers-test")
	old := provision.Default
	provision.Default = &provision.Service{}
	t.Cleanup(func() { provision.Default = old })
	mux := http.NewServeMux()
	Register(mux)

	w := call(t, mux, "GET", "/v1/providers/vast/offers?image=ghcr.io/geocine/aitoolkit-go:latest&model=flux&fast=1", "")
	if w.Code != 200 {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	var body map[string]any
	for _, r := range m.Rec.Requests() {
		if r.Method == "POST" && strings.HasPrefix(r.Path, "/api/v0/bundles") {
			if err := json.Unmarshal(r.Body, &body); err != nil {
				t.Fatal(err)
			}
		}
	}
	got, _ := json.Marshal(map[string]any{"cuda": body["cuda_max_good"], "cc": body["compute_cap"], "arch": body["cpu_arch"],
		"disk": body["disk_space"], "inet": body["inet_down"]})
	want := `{"arch":{"eq":"amd64"},"cc":{"in":[800,860,890,900,1000,1030,1200]},"cuda":{"gte":13},"disk":{"gte":80},"inet":{"gte":500}}`
	if string(got) != want {
		t.Fatalf("Vast filter %s\nwant %s", got, want)
	}
	if _, has := body["driver_vers"]; has {
		t.Fatal("driver_vers sent")
	}
	var out struct {
		Items []provision.Offer `json:"items"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil || len(out.Items) == 0 {
		t.Fatalf("offers %s (%v)", w.Body, err)
	}
	for _, o := range out.Items {
		if o.Host == nil || o.Fit == nil || o.Fit.VRAM == "" {
			t.Fatalf("offer %s host %+v fit %+v", o.ID, o.Host, o.Fit)
		}
	}
}
