package runpod

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/runpod/runpodmock"
)

var ctx = context.Background()

const key = "rpa_TESTKEY_0123456789"

// setup starts a mock on a FakeClock (unless o has a clock) and points the
// package bases at it.
func setup(t *testing.T, o mockkit.Options) (*Adapter, *runpodmock.Mock, *mockkit.FakeClock) {
	t.Helper()
	clock := mockkit.NewFakeClock(time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC))
	if o.Clock == nil {
		o.Clock = clock
	}
	m := runpodmock.New(o)
	base := m.Start()
	t.Cleanup(m.Close)
	pointAt(t, base)
	return New(), m, clock
}

func pointAt(t *testing.T, base string) {
	oldAPI, oldGQL, oldProxy, oldWait := API, GraphQL, ProxyURL, LogsWait
	API, GraphQL = base, base+"/graphql"
	t.Cleanup(func() { API, GraphQL, ProxyURL, LogsWait = oldAPI, oldGQL, oldProxy, oldWait })
}

func lastReq(t *testing.T, m *runpodmock.Mock, method, pathPrefix string) mockkit.Recorded {
	t.Helper()
	reqs := m.Rec.Requests()
	for i := len(reqs) - 1; i >= 0; i-- {
		if reqs[i].Method == method && strings.HasPrefix(reqs[i].Path, pathPrefix) {
			return reqs[i]
		}
	}
	t.Fatalf("no %s %s recorded", method, pathPrefix)
	return mockkit.Recorded{}
}

func launch(t *testing.T, a *Adapter, label string) provision.Ref {
	t.Helper()
	ref, err := a.Launch(ctx, key, provision.LaunchSpec{Label: label, OfferID: OfferID("NVIDIA GeForce RTX 4090", "SECURE", 1),
		Image: "ghcr.io/geocine/aitoolkit-go:abc1234", DiskGB: 100, Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_token_value_123"}})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}
	return ref
}

func TestRegisteredWithInfo(t *testing.T) {
	c, ok := provision.Lookup(provision.RunPod)
	if !ok {
		t.Fatal("runpod is not registered")
	}
	info := c.Info()
	want := provision.Caps{CloudChoice: true, CountChoice: true, StopStart: true, HTTPS: "yes", SelfDestroy: "key", Credit: true,
		EnvCountMax: 50, EnvTotalMax: 100000, ImageDigestRef: "unconfirmed"}
	if info.Caps != want || info.Stage != "stable" || info.Label != "RunPod" || info.Kind != provision.RunPod {
		t.Fatalf("info %+v", info)
	}
	if !slices.Equal(info.KeyScopes, []string{"pods:read", "pods:write", "catalog:read", "templates:read"}) || info.KeysURL != "https://console.runpod.io/user/settings" || len(info.Unconfirmed) != 3 {
		t.Fatalf("scopes/urls %+v", info)
	}
}

func TestVerifyCallsAndCredit(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{Credit: 12.5})
	acct, err := a.Verify(ctx, key)
	if err != nil || acct.Credit == nil || *acct.Credit != 12.5 {
		t.Fatalf("verify: %+v %v", acct, err)
	}
	reqs := m.Rec.Requests()
	if len(reqs) != 3 {
		t.Fatalf("calls %d", len(reqs))
	}
	if r := reqs[0]; r.Method != "GET" || r.Path != "/v2/catalog/gpus" || r.RawQuery != "" {
		t.Errorf("catalog check %s %s?%s (the catalog has no limit param)", r.Method, r.Path, r.RawQuery)
	}
	if r := reqs[1]; r.Path != "/v2/pods" || r.RawQuery != "limit=1" {
		t.Errorf("pods check %s?%s", r.Path, r.RawQuery)
	}
	if r := reqs[2]; r.Method != "POST" || r.Path != "/graphql" || !strings.Contains(string(r.Body), "clientBalance") {
		t.Errorf("graphql %s %s %s", r.Method, r.Path, r.Body)
	}
	for _, r := range reqs {
		if r.Header.Get("Authorization") != "Bearer "+key {
			t.Errorf("%s: auth header %q", r.Path, r.Header.Get("Authorization"))
		}
	}
}

func TestVerifyGraphQLFailureLeavesCreditNil(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	m.FailNext(runpodmock.Injected{Path: "/graphql", Status: 500, Detail: "boom"})
	if acct, err := a.Verify(ctx, key); err != nil || acct.Credit != nil {
		t.Fatalf("graphql 500: %+v %v", acct, err)
	}
	GraphQL = API + "/not-graphql" // 404 problem body: no data field
	if acct, err := a.Verify(ctx, key); err != nil || acct.Credit != nil {
		t.Fatalf("graphql 404: %+v %v", acct, err)
	}
}

func TestVerifyAuthAndScopes(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{Keys: []string{key}})
	if _, err := a.Verify(ctx, "rpa_wrong_key_000"); provision.ClassOf(err) != provision.ClassAuth {
		t.Fatalf("bad key: %v", err)
	}
	m.FailNext(runpodmock.Injected{Method: "GET", Path: "/v2/catalog/gpus", Status: 403, Detail: "access denied"})
	if _, err := a.Verify(ctx, key); provision.ClassOf(err) != provision.ClassAuth || !strings.Contains(err.Error(), "catalog:read") {
		t.Fatalf("catalog 403: %v", err)
	}
	m.FailNext(runpodmock.Injected{Method: "GET", Path: "/v2/pods", Status: 403, Detail: "access denied"})
	if _, err := a.Verify(ctx, key); provision.ClassOf(err) != provision.ClassAuth || !strings.Contains(err.Error(), "pods:read") {
		t.Fatalf("pods 403: %v", err)
	}
}

func TestOffersQueryPriceAndMapping(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	offers, err := a.Offers(ctx, key, provision.OfferQuery{DiskGB: 100})
	if err != nil {
		t.Fatal(err)
	}
	if r := lastReq(t, m, "GET", "/v2/catalog/gpus"); r.RawQuery != "cloud=SECURE&count=1&include=AVAILABILITY&product=POD" {
		t.Errorf("catalog query %q", r.RawQuery)
	}
	var names []string
	for _, o := range offers {
		names = append(names, o.GPU)
	}
	// H100 has availability NONE; the rest are cheapest first.
	if !slices.Equal(names, []string{"RTX 3090", "RTX 4090", "RTX A6000", "A100 PCIe"}) {
		t.Fatalf("offers %v", names)
	}
	o := offers[1]
	disk := 100 * 0.10 / 730
	if o.ID != "NVIDIA GeForce RTX 4090|SECURE|1" || o.Provider != provision.RunPod || o.VRAMGB != 24 || o.GPUCount != 1 ||
		o.PricePerHour != round4(0.44+disk) || o.StoppedPricePerHour == nil || *o.StoppedPricePerHour != 0 ||
		o.Availability != "high" || o.Location != "" || o.Reliability != nil || o.MaxDiskGB != nil || !o.HTTPS || o.Cloud != "secure" ||
		o.Host != nil || o.Fit != nil {
		t.Fatalf("4090 offer %+v", o)
	}

	// community x2: price x count; the A100's community maxCount is 2 and the H100 has no community cloud.
	offers, err = a.Offers(ctx, key, provision.OfferQuery{Cloud: "community", GPUCount: 2, DiskGB: 50})
	if err != nil {
		t.Fatal(err)
	}
	if r := lastReq(t, m, "GET", "/v2/catalog/gpus"); r.RawQuery != "cloud=COMMUNITY&count=2&include=AVAILABILITY&product=POD" {
		t.Errorf("community query %q", r.RawQuery)
	}
	if len(offers) != 4 || offers[0].GPU != "RTX 3090" || offers[0].PricePerHour != round4(2*0.22+50*0.10/730) || offers[0].ID != "NVIDIA GeForce RTX 3090|COMMUNITY|2" {
		t.Fatalf("community offers %+v", offers)
	}
	// 3 GPUs is over the A100's community maxCount
	offers, _ = a.Offers(ctx, key, provision.OfferQuery{Cloud: "community", GPUCount: 3})
	if len(offers) != 3 || slices.ContainsFunc(offers, func(o provision.Offer) bool { return o.GPU == "A100 PCIe" }) {
		t.Fatalf("maxCount: %+v", offers)
	}
	if _, err := a.Offers(ctx, key, provision.OfferQuery{Cloud: "spot"}); provision.ClassOf(err) != provision.ClassInvalid {
		t.Fatalf("bad cloud: %v", err)
	}
}

func TestOffersFilters(t *testing.T) {
	a, _, _ := setup(t, mockkit.Options{})
	gpus := func(q provision.OfferQuery) []string {
		t.Helper()
		offers, err := a.Offers(ctx, key, q)
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, o := range offers {
			out = append(out, o.GPU)
		}
		return out
	}
	for _, c := range []struct {
		name string
		q    provision.OfferQuery
		want []string
	}{
		{"min vram", provision.OfferQuery{MinVRAMGB: 48}, []string{"RTX A6000", "A100 PCIe"}},
		{"max vram is exclusive", provision.OfferQuery{MaxVRAMGB: 48}, []string{"RTX 3090", "RTX 4090"}},
		{"min and max", provision.OfferQuery{MinVRAMGB: 40, MaxVRAMGB: 80}, []string{"RTX A6000"}},
		{"gpu substring on the name", provision.OfferQuery{GPU: "a100"}, []string{"A100 PCIe"}},
		{"gpu substring on the id", provision.OfferQuery{GPU: "GeForce"}, []string{"RTX 3090", "RTX 4090"}},
		{"max price incl. disk", provision.OfferQuery{MaxPrice: 0.45}, []string{"RTX 3090"}},
		{"limit", provision.OfferQuery{Limit: 2}, []string{"RTX 3090", "RTX 4090"}},
		{"fast is a no-op (no bandwidth on RunPod)", provision.OfferQuery{Fast: true}, []string{"RTX 3090", "RTX 4090", "RTX A6000", "A100 PCIe"}},
	} {
		if got := gpus(c.q); !slices.Equal(got, c.want) {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

func TestQuoteRoundTripAndCapacity(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	offers, err := a.Offers(ctx, key, provision.OfferQuery{GPU: "4090", DiskGB: 100})
	if err != nil || len(offers) != 1 {
		t.Fatalf("offers %v %v", offers, err)
	}
	got, err := a.Quote(ctx, key, offers[0].ID, provision.OfferQuery{DiskGB: 100})
	if err != nil || got.ID != offers[0].ID || got.PricePerHour != offers[0].PricePerHour {
		t.Fatalf("quote %+v %v", got, err)
	}
	r := lastReq(t, m, "GET", "/v2/catalog/gpus/")
	if r.Path != "/v2/catalog/gpus/NVIDIA GeForce RTX 4090" || r.RawQuery != "cloud=SECURE&count=1&include=AVAILABILITY&product=POD" {
		t.Errorf("quote request %s?%s", r.Path, r.RawQuery)
	}
	// a bigger disk re-prices
	if got, _ := a.Quote(ctx, key, offers[0].ID, provision.OfferQuery{DiskGB: 200}); got.PricePerHour != round4(0.44+200*0.10/730) {
		t.Errorf("disk re-price %v", got.PricePerHour)
	}
	m.SetAvailability("NVIDIA GeForce RTX 4090", "NONE")
	if _, err := a.Quote(ctx, key, offers[0].ID, provision.OfferQuery{}); provision.ClassOf(err) != provision.ClassCapacity {
		t.Fatalf("sold out: %v", err)
	}
	if _, err := a.Quote(ctx, key, OfferID("NVIDIA Imaginary 9000", "SECURE", 1), provision.OfferQuery{}); provision.ClassOf(err) != provision.ClassCapacity {
		t.Fatalf("gone type: %v", err)
	}
	for _, bad := range []string{"", "x", "a|b|c", "NVIDIA A40|SECURE|0", "NVIDIA A40|ALL|1", "|SECURE|1"} {
		if _, err := a.Quote(ctx, key, bad, provision.OfferQuery{}); provision.ClassOf(err) != provision.ClassInvalid {
			t.Errorf("offer id %q: %v", bad, err)
		}
	}
}

func TestLaunchBodyPassesTheContract(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	ref, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-abc", OfferID: OfferID("NVIDIA RTX A6000", "COMMUNITY", 2),
		Image: "ghcr.io/geocine/aitoolkit-go:abc1234", DiskGB: 120, Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_x", "AITK_MACHINE_ID": "mch_abc"}})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}
	if ref.ID == "" || ref.Label != "aitk-abc" || ref.OfferID != "NVIDIA RTX A6000|COMMUNITY|2" {
		t.Fatalf("ref %+v", ref)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(lastReq(t, m, "POST", "/v2/pods").Body, &body); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(body))
	for k := range body {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	if !slices.Equal(keys, []string{"cloud", "disk", "env", "gpu", "image", "name", "ports"}) {
		t.Errorf("body fields %v (no startSsh, mounts or dataCenterIds)", keys)
	}
	if string(body["ports"]) != `["8675/http"]` || string(body["gpu"]) != `{"id":"NVIDIA RTX A6000","count":2}` || string(body["cloud"]) != `"COMMUNITY"` || string(body["disk"]) != "120" {
		t.Errorf("body %s", lastReq(t, m, "POST", "/v2/pods").Body)
	}
	pods := m.Instances()
	if len(pods) != 1 || pods[0].ID != ref.ID || pods[0].Env["AITK_MACHINE_ID"] != "mch_abc" || pods[0].Image != "ghcr.io/geocine/aitoolkit-go:abc1234" {
		t.Fatalf("mock pods %+v", pods)
	}
}

func TestLifecycleUnderFakeClock(t *testing.T) {
	a, _, clock := setup(t, mockkit.Options{Delays: mockkit.Delays{Provision: 3 * time.Second, Start: 6 * time.Second}})
	if _, err := a.Offers(ctx, key, provision.OfferQuery{}); err != nil { // fills the display-name cache
		t.Fatal(err)
	}
	ref := launch(t, a, "aitk-life")
	get := func() provision.Instance {
		t.Helper()
		inst, err := a.Get(ctx, key, ref)
		if err != nil {
			t.Fatalf("get: %v", err)
		}
		return inst
	}
	inst := get()
	if inst.Status != provision.StatusCreating || inst.ProviderStatus != "PROVISIONING" || inst.Label != "aitk-life" || inst.ProviderID != ref.ID {
		t.Fatalf("creating %+v", inst)
	}
	if !slices.Equal(inst.Actions, []string{"stop", "terminate"}) {
		t.Errorf("provisioning actions %v", inst.Actions)
	}
	wantEP := provision.Endpoint{URL: "https://" + ref.ID + "-8675.proxy.runpod.net", Secure: true, Via: "proxy"}
	if len(inst.Endpoints) != 1 || inst.Endpoints[0] != wantEP {
		t.Errorf("endpoints %+v", inst.Endpoints)
	}
	clock.Advance(3 * time.Second)
	if st := get().Status; st != provision.StatusStarting {
		t.Fatalf("after 3s: %s", st)
	}
	clock.Advance(6 * time.Second)
	inst = get()
	if inst.Status != provision.StatusRunning || inst.GPU != "RTX 4090" || inst.VRAMGB != 24 || inst.GPUCount != 1 || inst.Location != "US-KS-2" ||
		inst.PricePerHour == nil || *inst.PricePerHour != round4(0.44+100*0.10/730) || inst.CreatedAt.IsZero() {
		t.Fatalf("running %+v", inst)
	}
	if !slices.Equal(inst.Actions, []string{"stop", "terminate"}) { // restart is not ours
		t.Errorf("running actions %v", inst.Actions)
	}

	if err := a.Stop(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	clock.Advance(3 * time.Second)
	inst = get()
	if inst.Status != provision.StatusStopped || *inst.PricePerHour != 0 || !slices.Equal(inst.Actions, []string{"start", "terminate"}) {
		t.Fatalf("stopped %+v", inst)
	}
	if err := a.Stop(ctx, key, ref.ID); provision.ClassOf(err) != provision.ClassConflict {
		t.Fatalf("stop while stopped: %v", err)
	}
	if err := a.Start(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	if st := get().Status; st != provision.StatusStarting {
		t.Fatalf("after start: %s", st)
	}
	clock.Advance(6 * time.Second)
	if st := get().Status; st != provision.StatusRunning {
		t.Fatalf("restarted: %s", st)
	}

	if err := a.Terminate(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Get(ctx, key, ref); provision.ClassOf(err) != provision.ClassNotFound {
		t.Fatalf("get after terminate: %v", err)
	}
	if err := a.Terminate(ctx, key, ref.ID); err != nil {
		t.Fatalf("terminate twice: %v", err)
	}
}

func TestGetOnAFreshAdapterReadsTheCatalog(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	ref := launch(t, New(), "aitk-fresh") // another adapter launched it: nothing cached here
	inst, err := a.Get(ctx, key, ref)
	if err != nil || inst.GPU != "RTX 4090" || inst.VRAMGB != 24 {
		t.Fatalf("fresh get %+v %v", inst, err)
	}
	if r := lastReq(t, m, "GET", "/v2/catalog/gpus"); r.Path != "/v2/catalog/gpus" || r.RawQuery != "" {
		t.Errorf("catalog fill %s?%s", r.Path, r.RawQuery)
	}
	catalogCalls := func() (n int) {
		for _, r := range m.Rec.Requests() {
			if r.Path == "/v2/catalog/gpus" {
				n++
			}
		}
		return n
	}
	if _, err := a.Get(ctx, key, ref); err != nil || catalogCalls() != 1 {
		t.Fatalf("a cached type reads the catalog again: %d calls, %v", catalogCalls(), err)
	}
	if got, err := New().List(ctx, key, "aitk-"); err != nil || len(got) != 1 || got[0].GPU != "RTX 4090" || got[0].VRAMGB != 24 || catalogCalls() != 2 {
		t.Fatalf("fresh list %+v %v (%d catalog calls)", got, err, catalogCalls())
	}
	// a failed catalog read keeps the id and 0
	b := New()
	m.FailNext(runpodmock.Injected{Method: "GET", Path: "/v2/catalog/gpus", Status: 503, Detail: "upstream"})
	if inst, err := b.Get(ctx, key, ref); err != nil || inst.GPU != "NVIDIA GeForce RTX 4090" || inst.VRAMGB != 0 {
		t.Fatalf("catalog down %+v %v", inst, err)
	}
}

func TestStartWithoutCreditIsCredit(t *testing.T) {
	a, _, _ := setup(t, mockkit.Options{SimSecret: key})
	ref := launch(t, a, "aitk-broke")
	if err := a.Stop(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	if err := a.Start(ctx, key+".sim_broke", ref.ID); provision.ClassOf(err) != provision.ClassCredit {
		t.Fatalf("start without credit: %v", err)
	}
}

func TestOffersLimitIsCappedAt200(t *testing.T) {
	gpus := make([]string, 0, 260)
	for i := range 260 {
		gpus = append(gpus, `{"id":"NVIDIA T`+strconv.Itoa(i)+`","name":"T`+strconv.Itoa(i)+`","memory":24,"secure":true,"availability":"HIGH","price":{"secure":0.5},"maxCount":{"secure":8}}`)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"gpus":[` + strings.Join(gpus, ",") + `]}`))
	}))
	defer srv.Close()
	pointAt(t, srv.URL)
	for limit, want := range map[int]int{0: 50, 120: 120, 500: 200} {
		if got, err := New().Offers(ctx, key, provision.OfferQuery{Limit: limit}); err != nil || len(got) != want {
			t.Errorf("limit %d: %d offers, want %d (%v)", limit, len(got), want, err)
		}
	}
}

func TestGoneAsTerminatedStatus(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{Faults: mockkit.Faults{GoneAsStatus: true}})
	ref := launch(t, a, "aitk-gone")
	if err := a.Terminate(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Get(ctx, key, ref); provision.ClassOf(err) != provision.ClassNotFound {
		t.Fatalf("TERMINATED: %v", err)
	}
	if r := lastReq(t, m, "GET", "/v2/pods/"); !strings.HasSuffix(r.Path, ref.ID) {
		t.Fatalf("last get %s", r.Path)
	}
	if got, _ := a.List(ctx, key, "aitk-"); len(got) != 0 {
		t.Fatalf("a terminated pod is not listed: %+v", got)
	}
}

func TestCrashIsFailed(t *testing.T) {
	a, _, _ := setup(t, mockkit.Options{Faults: mockkit.Faults{CrashAfterRunning: true}})
	ref := launch(t, a, "aitk-crash")
	inst, err := a.Get(ctx, key, ref)
	if err != nil || inst.Status != provision.StatusFailed || inst.ProviderStatus != "ERROR" || inst.Detail == "" {
		t.Fatalf("crash %+v %v", inst, err)
	}
}

func TestListFiltersByLabel(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	launch(t, a, "aitk-one")
	launch(t, a, "aitk-two")
	launch(t, a, "someone-elses")
	got, err := a.List(ctx, key, "aitk-")
	if err != nil || len(got) != 2 {
		t.Fatalf("list %+v %v", got, err)
	}
	if r := lastReq(t, m, "GET", "/v2/pods"); r.RawQuery != "limit=1000" {
		t.Errorf("list query %q", r.RawQuery)
	}
	for _, inst := range got {
		if !strings.HasPrefix(inst.Label, "aitk-") || inst.Status != provision.StatusRunning || len(inst.Endpoints) != 1 {
			t.Errorf("listed %+v", inst)
		}
	}
}

func TestListFollowsNextCursor(t *testing.T) {
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		queries = append(queries, r.URL.RawQuery)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("cursor") == "" {
			_, _ = w.Write([]byte(`{"pods":[{"id":"p1","name":"aitk-1","status":"RUNNING","actions":[]}],"pagination":{"nextCursor":"c2","hasNextPage":true}}`))
			return
		}
		_, _ = w.Write([]byte(`{"pods":[{"id":"p2","name":"aitk-2","status":"EXITED","actions":[]}],"pagination":{"nextCursor":null,"hasNextPage":false}}`))
	}))
	defer srv.Close()
	pointAt(t, srv.URL)
	got, err := New().List(ctx, key, "aitk-")
	if err != nil || len(got) != 2 || got[1].Status != provision.StatusStopped {
		t.Fatalf("list %+v %v", got, err)
	}
	if !slices.Equal(queries, []string{"limit=1000", "cursor=c2&limit=1000"}) {
		t.Fatalf("queries %v", queries)
	}
}

func TestTemplates(t *testing.T) {
	a, _, _ := setup(t, mockkit.Options{})
	got, err := a.Templates(ctx, key)
	if err != nil || len(got) != 2 {
		t.Fatalf("templates %+v %v", got, err)
	}
	ours := provision.Template{ID: "tpl_aitk0001", Name: "aitk-go headless", Image: "ghcr.io/geocine/aitoolkit-go:latest", Ours: true, PublishesPort: true, HasToken: true}
	if got[0] != ours {
		t.Errorf("aitk template %+v", got[0])
	}
	if got[1].Ours || got[1].PublishesPort || got[1].HasToken {
		t.Errorf("other template %+v", got[1])
	}
}

func TestTemplatesFollowNextCursor(t *testing.T) {
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		queries = append(queries, r.URL.RawQuery)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("cursor") == "" {
			_, _ = w.Write([]byte(`{"templates":[{"id":"t1","name":"aitk","image":"ghcr.io/geocine/aitoolkit-go:latest","ports":["8675/http"],"env":{"AI_TOOLKIT_AUTH":"x"}}],"pagination":{"nextCursor":"c2","hasNextPage":true}}`))
			return
		}
		_, _ = w.Write([]byte(`{"templates":[{"id":"t2","name":"other","image":"nginx","ports":[]}],"pagination":{"nextCursor":null,"hasNextPage":false}}`))
	}))
	defer srv.Close()
	pointAt(t, srv.URL)
	got, err := New().Templates(ctx, key)
	if err != nil || len(got) != 2 || got[0].ID != "t1" || !got[0].Ours || !got[0].HasToken || got[1].ID != "t2" || got[1].Ours {
		t.Fatalf("templates %+v %v", got, err)
	}
	if !slices.Equal(queries, []string{"", "cursor=c2"}) {
		t.Fatalf("queries %v", queries)
	}
}

func TestTemplates403NamesTheScope(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	m.FailNext(runpodmock.Injected{Method: "GET", Path: "/v2/templates", Status: 403, Detail: "access denied"})
	if _, err := a.Templates(ctx, key); provision.ClassOf(err) != provision.ClassAuth || !strings.Contains(err.Error(), "templates:read") {
		t.Fatalf("templates 403: %v", err)
	}
}

func TestStatusMap(t *testing.T) {
	for raw, want := range map[string]provision.Status{
		"PROVISIONING": provision.StatusCreating, "STARTING": provision.StatusStarting, "RUNNING": provision.StatusRunning,
		"EXITED": provision.StatusStopped, "ERROR": provision.StatusFailed, "TERMINATED": provision.StatusTerminated, "PAUSED": provision.StatusUnknown,
	} {
		if got, _ := mapStatus(raw); got != want {
			t.Errorf("%s -> %s, want %s", raw, got, want)
		}
	}
}

func TestErrorClasses(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	ref := launch(t, a, "aitk-err")
	offer := OfferID("NVIDIA GeForce RTX 4090", "SECURE", 1)
	doLaunch := func() error {
		_, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-x", OfferID: offer, Image: "img", DiskGB: 50})
		return err
	}
	doGet := func() error { _, err := a.Get(ctx, key, ref); return err }
	for _, c := range []struct {
		name   string
		inj    runpodmock.Injected
		call   func() error
		class  provision.Class
		inText string
	}{
		{"create 400 without a rule word is capacity", runpodmock.Injected{Method: "POST", Path: "/v2/pods", Status: 400, Detail: "no available machines for the requested GPU type"}, doLaunch, provision.ClassCapacity, "no available machines"},
		{"create 400 naming a rule is invalid", runpodmock.Injected{Method: "POST", Path: "/v2/pods", Status: 400, Detail: "gpu.allowedCudaVersions and gpu.minCudaVersion are mutually exclusive"}, doLaunch, provision.ClassInvalid, "mutually exclusive"},
		{"create 400 about a network volume is invalid", runpodmock.Injected{Method: "POST", Path: "/v2/pods", Status: 400, Detail: "network volume is in another data center"}, doLaunch, provision.ClassInvalid, "network volume"},
		{"create 403 is the pool", runpodmock.Injected{Method: "POST", Path: "/v2/pods", Status: 403, Detail: "access denied"}, doLaunch, provision.ClassCapacity, "can't use this GPU pool"},
		{"get 403 is a scope", runpodmock.Injected{Method: "GET", Path: "/v2/pods/" + ref.ID, Status: 403, Detail: "access denied"}, doGet, provision.ClassAuth, "pods:read"},
		{"401", runpodmock.Injected{Path: "/v2/pods/" + ref.ID, Status: 401, Detail: "invalid api key"}, doGet, provision.ClassAuth, "invalid api key"},
		{"402", runpodmock.Injected{Path: "/v2/pods", Method: "POST", Status: 402, Detail: "insufficient balance"}, doLaunch, provision.ClassCredit, "insufficient"},
		{"422 joins errors", runpodmock.Injected{Path: "/v2/pods", Method: "POST", Status: 422, Detail: "Request validation failed.", Errors: []string{"$.name: required", "$.gpu.id: required"}}, doLaunch, provision.ClassInvalid, "$.name: required; $.gpu.id: required"},
		{"413", runpodmock.Injected{Path: "/v2/pods", Method: "POST", Status: 413, Detail: "Request body exceeds the 102400 byte limit."}, doLaunch, provision.ClassInvalid, "102400"},
		{"404", runpodmock.Injected{Path: "/v2/pods/" + ref.ID, Status: 404, Detail: "pod not found"}, doGet, provision.ClassNotFound, "pod not found"},
		{"409", runpodmock.Injected{Path: "/v2/pods/" + ref.ID + "/action", Status: 409, Detail: "invalid action"}, func() error { return a.Stop(ctx, key, ref.ID) }, provision.ClassConflict, "invalid action"},
		{"5xx", runpodmock.Injected{Path: "/v2/pods/" + ref.ID, Status: 503, Detail: "upstream"}, doGet, provision.ClassUnavailable, "upstream"},
	} {
		m.FailNext(c.inj)
		err := c.call()
		if provision.ClassOf(err) != c.class || !strings.Contains(err.Error(), c.inText) {
			t.Errorf("%s: %v (want %s containing %q)", c.name, err, c.class, c.inText)
		}
	}
	if code, msg, _, ok := provision.V1(func() error {
		m.FailNext(runpodmock.Injected{Method: "POST", Path: "/v2/pods", Status: 403, Detail: "access denied"})
		return doLaunch()
	}()); !ok || code != provision.CodeNoCapacity || !strings.Contains(msg, "RunPod") {
		t.Errorf("v1 mapping %s %q", code, msg)
	}
}

func TestRateLimitRetryAfter(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	m.FailNext(runpodmock.Injected{Path: "/v2/catalog/gpus", Status: 429, Detail: "rate limit exceeded for the minute window", Header: http.Header{"Retry-After": {"7"}}})
	_, err := a.Offers(ctx, key, provision.OfferQuery{})
	var e *provision.Error
	if provision.ClassOf(err) != provision.ClassRateLimited || !asErr(err, &e) || e.RetryAfter != 7*time.Second {
		t.Fatalf("Retry-After: %v", err)
	}
	// RateLimit alone: the longest exhausted window
	m.FailNext(runpodmock.Injected{Path: "/v2/catalog/gpus", Status: 429, Detail: "rate limit exceeded",
		Header: http.Header{"Ratelimit": {`"minute";r=0;t=12, "hour";r=2800;t=1812`}}})
	_, err = a.Offers(ctx, key, provision.OfferQuery{})
	if !asErr(err, &e) || e.Class != provision.ClassRateLimited || e.RetryAfter != 12*time.Second {
		t.Fatalf("RateLimit: %v", err)
	}
	// the mock's own fault: every 2nd call
	a2, _, _ := setup(t, mockkit.Options{Faults: mockkit.Faults{RateLimitEvery: 2}})
	if _, err := a2.Offers(ctx, key, provision.OfferQuery{}); err != nil {
		t.Fatal(err)
	}
	_, err = a2.Offers(ctx, key, provision.OfferQuery{})
	if !asErr(err, &e) || e.Class != provision.ClassRateLimited || e.RetryAfter != 2*time.Second {
		t.Fatalf("fault 429: %v", err)
	}
}

func asErr(err error, e **provision.Error) bool {
	pe, ok := err.(*provision.Error)
	*e = pe
	return ok
}

func TestMockFaultsMapToClasses(t *testing.T) {
	for _, c := range []struct {
		f     mockkit.Faults
		class provision.Class
	}{
		{mockkit.Faults{NoCapacity: true}, provision.ClassCapacity},
		{mockkit.Faults{NoCredit: true}, provision.ClassCredit},
		{mockkit.Faults{Unauthorized: true}, provision.ClassAuth},
	} {
		a, _, _ := setup(t, mockkit.Options{Faults: c.f})
		_, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-f", OfferID: OfferID("NVIDIA GeForce RTX 4090", "SECURE", 1), Image: "img", DiskGB: 50})
		if provision.ClassOf(err) != c.class {
			t.Errorf("%+v: %v", c.f, err)
		}
	}
	// a NONE type cannot be placed
	a, _, _ := setup(t, mockkit.Options{})
	_, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-h", OfferID: OfferID("NVIDIA H100 80GB HBM3", "SECURE", 1), Image: "img", DiskGB: 50})
	if provision.ClassOf(err) != provision.ClassCapacity {
		t.Errorf("NONE: %v", err)
	}
}

func TestPlantedSecretsNeverInErrors(t *testing.T) {
	const planted = "rpa_PLANTED_SECRET_9f8e7d"
	const token = "aitk_PLANTED_TOKEN_abcdef"
	a, m, _ := setup(t, mockkit.Options{})
	m.FailNext(runpodmock.Injected{Path: "/v2/catalog/gpus", Status: 401, Detail: "key " + planted + " is not valid"})
	_, err := a.Offers(ctx, planted, provision.OfferQuery{})
	if err == nil || strings.Contains(err.Error(), planted) || !strings.Contains(err.Error(), "[redacted]") {
		t.Fatalf("key leaked: %v", err)
	}
	m.FailNext(runpodmock.Injected{Method: "POST", Path: "/v2/pods", Status: 422, Detail: "Request validation failed.", Errors: []string{"$.env.AI_TOOLKIT_AUTH: " + token + " is odd"}})
	_, err = a.Launch(ctx, planted, provision.LaunchSpec{Label: "aitk-s", OfferID: OfferID("NVIDIA GeForce RTX 4090", "SECURE", 1), Image: "img", DiskGB: 50,
		Env: map[string]string{"AI_TOOLKIT_AUTH": token}})
	if err == nil || strings.Contains(err.Error(), token) {
		t.Fatalf("env value leaked: %v", err)
	}
	// network errors too
	pointAt(t, "http://127.0.0.1:1")
	_, err = a.Offers(ctx, planted, provision.OfferQuery{})
	if provision.ClassOf(err) != provision.ClassUnavailable || strings.Contains(err.Error(), planted) {
		t.Fatalf("network error: %v", err)
	}
}

func TestLogsFromSSE(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	ref := launch(t, a, "aitk-logs")
	text, err := a.Logs(ctx, key, ref.ID, 10)
	if err != nil || !strings.HasPrefix(text, "Pulling image ghcr.io/geocine/aitoolkit-go:abc1234\n") || !strings.HasSuffix(text, "aitk server listening on :8675") {
		t.Fatalf("logs %q %v", text, err)
	}
	if r := lastReq(t, m, "GET", "/v2/pods/"+ref.ID+"/logs"); r.RawQuery != "tail=10" || r.Header.Get("Accept") != "text/event-stream" {
		t.Errorf("logs request %q %v", r.RawQuery, r.Header)
	}
	if text, _ := a.Logs(ctx, key, ref.ID, 1); text != "aitk server listening on :8675" {
		t.Errorf("tail 1: %q", text)
	}
	// the live stream stays open: Logs returns what it read when LogsWait runs out
	m.HoldLogs(true)
	LogsWait = 200 * time.Millisecond
	start := time.Now()
	text, err = a.Logs(ctx, key, ref.ID, 100)
	if err != nil || !strings.Contains(text, "Container started") || time.Since(start) > 5*time.Second {
		t.Fatalf("held stream: %q %v after %v", text, err, time.Since(start))
	}
	if _, err := a.Logs(ctx, key, "nopod", 10); provision.ClassOf(err) != provision.ClassNotFound {
		t.Fatalf("missing pod logs: %v", err)
	}
}

func TestProxyURLAndSimulatedEndpoint(t *testing.T) {
	if got := ProxyURL("abc123xyz"); got != "https://abc123xyz-8675.proxy.runpod.net" {
		t.Fatalf("proxy url %q", got)
	}
	var started []mockkit.NodeSpec
	a, m, _ := setup(t, mockkit.Options{Node: func(n mockkit.NodeSpec) (string, func()) {
		started = append(started, n)
		return "http://127.0.0.1:4" + string(rune('0'+len(started))) + "000", func() {}
	}})
	ProxyURL = m.NodeURL
	ref := launch(t, a, "aitk-sim")
	inst, err := a.Get(ctx, key, ref)
	if err != nil || len(inst.Endpoints) != 1 || inst.Endpoints[0] != (provision.Endpoint{URL: "http://127.0.0.1:41000", Secure: false, Via: "proxy"}) {
		t.Fatalf("sim endpoint %+v %v", inst.Endpoints, err)
	}
	if len(started) != 1 || started[0].Env["AI_TOOLKIT_AUTH"] != "aitk_token_value_123" || started[0].Kind != "runpod" || started[0].Image != "ghcr.io/geocine/aitoolkit-go:abc1234" {
		t.Fatalf("node spec %+v", started)
	}
}
