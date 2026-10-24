package vast

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/vast/vastmock"
)

const testKey = "vast-test-key-0123456789"

var t0 = time.Unix(1_790_000_000, 0)

type env struct {
	m     *vastmock.Mock
	clock *mockkit.FakeClock
	pace  *sleepClock
	c     provision.Cloud
}

// setup starts a vastmock and points the adapter at it, with the limiter on
// a fake clock so nothing waits for real.
func setup(t *testing.T, o mockkit.Options) env {
	t.Helper()
	clock := mockkit.NewFakeClock(t0)
	o.Clock = clock
	m := vastmock.New(o)
	base := m.Start()
	t.Cleanup(m.Close)
	u, _ := url.Parse(base)
	oldAPI, oldPace, oldLogs := API, Pace, LogsHostAllowed
	pace := newSleepClock()
	API, Pace = base, NewLimiter(pace)
	LogsHostAllowed = func(host string) bool { return host == u.Host }
	t.Cleanup(func() { API, Pace, LogsHostAllowed = oldAPI, oldPace, oldLogs })
	c, ok := provision.Lookup(provision.Vast)
	if !ok {
		t.Fatal("vast adapter not registered")
	}
	return env{m: m, clock: clock, pace: pace, c: c}
}

func (e env) last(t *testing.T, method, pathPrefix string) mockkit.Recorded {
	t.Helper()
	reqs := e.m.Rec.Requests()
	for i := len(reqs) - 1; i >= 0; i-- {
		if reqs[i].Method == method && strings.HasPrefix(reqs[i].Path, pathPrefix) {
			return reqs[i]
		}
	}
	t.Fatalf("no %s %s recorded", method, pathPrefix)
	return mockkit.Recorded{}
}

func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("body %s: %v", b, err)
	}
	return v
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func ids(offers []provision.Offer) []string {
	out := make([]string, len(offers))
	for i, o := range offers {
		out[i] = o.ID
	}
	return out
}

func TestUnits(t *testing.T) {
	if vramFilterMB(24) != 24000 {
		t.Errorf("vramFilterMB(24) = %d", vramFilterMB(24))
	}
	// the fixture RTX 4090 reports 24564 (MiB labelled MB): passes a 24 GB filter, shows 24
	if 24564 < vramFilterMB(24) || vramGB(24564) != 24 || vramGB(49140) != 48 {
		t.Errorf("vramGB(24564) = %d, vramGB(49140) = %d", vramGB(24564), vramGB(49140))
	}
}

func TestInfo(t *testing.T) {
	info := adapter{}.Info()
	c := info.Caps
	if info.Kind != provision.Vast || info.Stage != "stable" || !c.StopStart || !c.StopKeepsDisk || c.HTTPS != "no" ||
		c.SelfDestroy != "builtin" || !c.Credit || c.EnvValueMax != 256 || c.EnvTotalMax != 32768 ||
		c.ImageDigestRef != "unconfirmed" || !c.ReportsInet || !c.ReportsDiskBW || c.CloudChoice || c.CountChoice || c.TemplateRequired {
		t.Fatalf("info %+v", info)
	}
	if info.KeysURL != "https://cloud.vast.ai/manage-keys/" || !slices.Equal(info.KeyScopes, []string{"user_read", "instance_read", "instance_write"}) {
		t.Fatalf("keys %q %v", info.KeysURL, info.KeyScopes)
	}
}

func TestOffersBodyAndMapping(t *testing.T) {
	e := setup(t, mockkit.Options{})
	offers, err := e.c.Offers(context.Background(), testKey, provision.OfferQuery{MinVRAMGB: 24, DiskGB: 100})
	if err != nil {
		t.Fatal(err)
	}
	rec := e.last(t, "POST", "/api/v0/bundles/")
	if rec.Header.Get("Authorization") != "Bearer "+testKey {
		t.Fatalf("auth header %q", rec.Header.Get("Authorization"))
	}
	body := decode(t, rec.Body)
	keys := slices.Sorted(func(yield func(string) bool) {
		for k := range body {
			if !yield(k) {
				return
			}
		}
	})
	want := []string{"allocated_storage", "direct_port_count", "external", "gpu_ram", "limit", "num_gpus", "order", "rentable", "rented", "type", "verified"}
	if !slices.Equal(keys, want) {
		t.Fatalf("body keys %v, want %v", keys, want)
	}
	exact := `{"allocated_storage":100,"direct_port_count":{"gte":1},"external":{"eq":false},"gpu_ram":{"gte":24000},"limit":50,` +
		`"num_gpus":{"eq":1},"order":[["dph_total","asc"]],"rentable":{"eq":true},"rented":{"eq":false},"type":"on-demand","verified":{"eq":true}}`
	if got, _ := json.Marshal(body); string(got) != exact {
		t.Fatalf("body\n%s\nwant\n%s", got, exact)
	}
	// 2x (num_gpus), no direct ports and the rented row are left out; cheapest first
	if got := ids(offers); !slices.Equal(got, []string{"90000001", "90000006", "57278070", "90000002", "55830991"}) {
		t.Fatalf("offers %v", got)
	}
	var o provision.Offer
	for _, x := range offers {
		if x.ID == "57278070" {
			o = x
		}
	}
	// dph_total = dph_base + storage_cost * 100 / 720 (the mock recomputes it for allocated_storage)
	if o.GPU != "RTX 4090" || o.GPUCount != 1 || o.VRAMGB != 24 || !near(o.PricePerHour, 0.44+0.2*100/720) ||
		o.StoppedPricePerHour == nil || !near(*o.StoppedPricePerHour, 0.2*100/720) || o.Location != "Türkiye, TR" ||
		o.Reliability == nil || !near(*o.Reliability, 0.8812925) || o.MaxDiskGB == nil || *o.MaxDiskGB != 638 ||
		o.Availability != "available" || o.HTTPS || o.Provider != provision.Vast || o.Host == nil || o.Fit != nil {
		t.Fatalf("offer %+v", o)
	}
	for i := 1; i < len(offers); i++ {
		if offers[i].PricePerHour < offers[i-1].PricePerHour {
			t.Fatalf("not sorted by price: %v", ids(offers))
		}
	}
}

func searchLimit(t *testing.T, e env) float64 {
	t.Helper()
	return decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)["limit"].(float64)
}

func TestOffersLimit(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ctx := context.Background()
	for _, c := range []struct {
		limit int
		want  float64
	}{{200, 200}, {0, 50}, {-3, 50}, {500, 200}, {7, 7}} {
		if _, err := e.c.Offers(ctx, testKey, provision.OfferQuery{Limit: c.limit}); err != nil {
			t.Fatal(err)
		}
		if got := searchLimit(t, e); got != c.want {
			t.Errorf("Limit %d sent limit %v, want %v", c.limit, got, c.want)
		}
	}
	offers, err := e.c.Offers(ctx, testKey, provision.OfferQuery{Limit: 2})
	if err != nil || len(offers) != 2 {
		t.Fatalf("Limit 2: %v %v", ids(offers), err)
	}
}

func TestOffersGPUTextWidensAndFiltersClientSide(t *testing.T) {
	e := setup(t, mockkit.Options{})
	offers, err := e.c.Offers(context.Background(), testKey, provision.OfferQuery{GPU: "a6000", Limit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if got := searchLimit(t, e); got != 200 {
		t.Fatalf("a GPU text search sends limit %v, want 200", got)
	}
	if got := ids(offers); !slices.Equal(got, []string{"90000002"}) {
		t.Fatalf("a6000 offers %v", got)
	}
	// the text never reaches Vast (no substring operator on gpu_name)
	if strings.Contains(string(e.last(t, "POST", "/api/v0/bundles/").Body), "a6000") {
		t.Fatal("GPU text sent to Vast")
	}
	// the user's limit applies after the client-side match
	offers, err = e.c.Offers(context.Background(), testKey, provision.OfferQuery{GPU: "RTX 4090", Limit: 2})
	if err != nil || len(offers) != 2 || offers[0].GPU != "RTX 4090" {
		t.Fatalf("4090 limit 2: %v %v", ids(offers), err)
	}
}

func TestOffersMaxVRAMPriceAndFast(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ctx := context.Background()
	offers, err := e.c.Offers(ctx, testKey, provision.OfferQuery{MinVRAMGB: 24, MaxVRAMGB: 40})
	if err != nil {
		t.Fatal(err)
	}
	ram := decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)["gpu_ram"]
	if got, _ := json.Marshal(ram); string(got) != `{"gte":24000,"lt":40960}` {
		t.Fatalf("gpu_ram %s", got)
	}
	if slices.Contains(ids(offers), "90000002") || len(offers) == 0 {
		t.Fatalf("the 48 GB A6000 must be capped out: %v", ids(offers))
	}

	offers, err = e.c.Offers(ctx, testKey, provision.OfferQuery{MaxPrice: 0.4})
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := json.Marshal(decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)["dph_total"]); string(got) != `{"lte":0.4}` {
		t.Fatalf("dph_total %s", got)
	}
	for _, o := range offers {
		if o.PricePerHour > 0.4 {
			t.Fatalf("offer above MaxPrice: %+v", o)
		}
	}

	offers, err = e.c.Offers(ctx, testKey, provision.OfferQuery{Fast: true})
	if err != nil {
		t.Fatal(err)
	}
	body := decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)
	in, _ := json.Marshal(body["inet_down"])
	disk, _ := json.Marshal(body["disk_bw"])
	if string(in) != `{"gte":500}` || string(disk) != `{"gte":300}` {
		t.Fatalf("fast filters %s %s", in, disk)
	}
	// 265 Mbps and 73 Mbps hosts are left out server-side
	if got := ids(offers); !slices.Equal(got, []string{"90000001", "90000002", "55830991"}) {
		t.Fatalf("fast offers %v", got)
	}
}

func TestQuote(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ctx := context.Background()
	o, err := e.c.Quote(ctx, testKey, "57278070", provision.OfferQuery{DiskGB: 50})
	if err != nil {
		t.Fatal(err)
	}
	if o.ID != "57278070" || !near(o.PricePerHour, 0.44+0.2*50/720) {
		t.Fatalf("quote %+v", o)
	}
	body := decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)
	if got, _ := json.Marshal(body["id"]); string(got) != `{"eq":57278070}` || body["limit"] != 1.0 || body["allocated_storage"] != 50.0 {
		t.Fatalf("quote body %v", body)
	}
	if _, has := body["num_gpus"]; has {
		t.Fatal("a quote must not filter on the GPU count the offer already fixes")
	}
	// a 2x offer quotes as itself
	if o, err := e.c.Quote(ctx, testKey, "90000003", provision.OfferQuery{}); err != nil || o.GPUCount != 2 {
		t.Fatalf("2x quote %+v %v", o, err)
	}
	if _, err := e.c.Quote(ctx, testKey, "90000005", provision.OfferQuery{}); provision.ClassOf(err) != provision.ClassCapacity {
		t.Fatalf("rented offer: %v", err)
	}
	if _, err := e.c.Quote(ctx, testKey, "nope", provision.OfferQuery{}); provision.ClassOf(err) != provision.ClassInvalid {
		t.Fatalf("bad id: %v", err)
	}
}

func TestVerifyReadsCreditAndDropsAPIKey(t *testing.T) {
	e := setup(t, mockkit.Options{Credit: 12.5})
	acc, err := e.c.Verify(context.Background(), testKey)
	if err != nil {
		t.Fatal(err)
	}
	if acc.Credit == nil || *acc.Credit != 12.5 {
		t.Fatalf("credit %+v", acc)
	}
	// the raw answer echoes api_key; nothing of it survives
	raw := e.last(t, "GET", "/api/v0/users/current/")
	if raw.Path != "/api/v0/users/current/" {
		t.Fatalf("path %q", raw.Path)
	}
	if strings.Contains(fmt.Sprintf("%+v %s", acc, acc.Detail), testKey) {
		t.Fatalf("api_key surfaced: %+v", acc)
	}
	if _, err := e.c.Verify(context.Background(), ""); provision.ClassOf(err) != provision.ClassAuth {
		t.Fatalf("no key: %v", err)
	}
}

func launch(t *testing.T, e env, offer, label string) provision.Ref {
	t.Helper()
	ref, err := e.c.Launch(context.Background(), testKey, provision.LaunchSpec{
		Label: label, OfferID: offer, Image: "ghcr.io/geocine/aitoolkit-go:abc1234", DiskGB: 100,
		Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_tok", "OPEN_BUTTON_PORT": "8675"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return ref
}

func TestLaunchBody(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ref := launch(t, e, "57278070", "aitk-abc")
	if ref.ID != "30000001" || ref.OfferID != "57278070" || ref.Label != "aitk-abc" {
		t.Fatalf("ref %+v", ref)
	}
	rec := e.last(t, "PUT", "/api/v0/asks/")
	if rec.Path != "/api/v0/asks/57278070/" {
		t.Fatalf("path %q", rec.Path)
	}
	exact := `{"args_str":"/start.sh","cancel_unavail":true,"client_id":"me","disk":100,` +
		`"env":{"-p 8675:8675":"1","AI_TOOLKIT_AUTH":"aitk_tok","OPEN_BUTTON_PORT":"8675"},` +
		`"image":"ghcr.io/geocine/aitoolkit-go:abc1234","label":"aitk-abc","runtype":"args"}`
	if got, _ := json.Marshal(decode(t, rec.Body)); string(got) != exact {
		t.Fatalf("create body\n%s\nwant\n%s", got, exact)
	}
	if !e.m.Rented(57278070) {
		t.Fatal("the offer should be rented after a create")
	}
	// the same offer again: 410 with cancel_unavail -> capacity
	_, err := e.c.Launch(context.Background(), testKey, provision.LaunchSpec{Label: "aitk-x", OfferID: "57278070", Image: "img", DiskGB: 100})
	if provision.ClassOf(err) != provision.ClassCapacity {
		t.Fatalf("taken offer: %v", err)
	}
	// an unknown offer: 404 no_such_ask -> capacity
	_, err = e.c.Launch(context.Background(), testKey, provision.LaunchSpec{Label: "aitk-x", OfferID: "11", Image: "img", DiskGB: 100})
	if provision.ClassOf(err) != provision.ClassCapacity {
		t.Fatalf("unknown offer: %v", err)
	}
	// no image: 400 invalid_args -> invalid
	_, err = e.c.Launch(context.Background(), testKey, provision.LaunchSpec{Label: "aitk-x", OfferID: "55830991", DiskGB: 100})
	if provision.ClassOf(err) != provision.ClassInvalid {
		t.Fatalf("no image: %v", err)
	}
}

func get(t *testing.T, e env, id string) provision.Instance {
	t.Helper()
	in, err := e.c.Get(context.Background(), testKey, provision.Ref{ID: id})
	if err != nil {
		t.Fatalf("get %s: %v", id, err)
	}
	return in
}

func TestLifecycle(t *testing.T) {
	e := setup(t, mockkit.Options{Delays: mockkit.Delays{Provision: 3 * time.Second, Start: 6 * time.Second}})
	ctx := context.Background()
	ref := launch(t, e, "57278070", "aitk-life")

	in := get(t, e, ref.ID)
	if in.Status != provision.StatusCreating || in.ProviderStatus != "null/running" || len(in.Endpoints) != 0 || in.PortMissing {
		t.Fatalf("provisioning: %+v", in)
	}
	if in.Label != "aitk-life" || in.GPU != "RTX 4090" || in.GPUCount != 1 || in.VRAMGB != 24 || in.CreatedAt.Unix() != t0.Unix() ||
		in.PricePerHour == nil || !near(*in.PricePerHour, 0.44+0.2*100/720) || in.StoppedPricePerHour == nil ||
		!slices.Equal(in.Actions, []string{"stop", "terminate"}) {
		t.Fatalf("instance facts: %+v", in)
	}
	if in.Env["AI_TOOLKIT_AUTH"] != "aitk_tok" || in.Env["-p 8675:8675"] != "" || len(in.Env) != 2 {
		t.Fatalf("env from extra_env: %v", in.Env)
	}
	rec := e.last(t, "GET", "/api/v0/instances/")
	if rec.Path != "/api/v0/instances/"+ref.ID+"/" || rec.RawQuery != "owner=me" {
		t.Fatalf("show request %s?%s", rec.Path, rec.RawQuery)
	}

	e.clock.Advance(3 * time.Second)
	if in = get(t, e, ref.ID); in.Status != provision.StatusStarting || in.ProviderStatus != "loading/running" {
		t.Fatalf("loading: %+v", in)
	}
	e.clock.Advance(6 * time.Second)
	in = get(t, e, ref.ID)
	if in.Status != provision.StatusRunning || len(in.Endpoints) != 1 || in.PortMissing {
		t.Fatalf("running: %+v", in)
	}
	ep := in.Endpoints[0]
	port := e.m.Instances()[0].HostPort
	if ep.URL != "http://203.0.113.11:"+port || ep.Secure || ep.Via != "direct" {
		t.Fatalf("endpoint %+v (host port %q)", ep, port)
	}

	if err := e.c.Stop(ctx, testKey, ref.ID); err != nil {
		t.Fatal(err)
	}
	stop := e.last(t, "PUT", "/api/v0/instances/")
	if string(stop.Body) != `{"state":"stopped"}` {
		t.Fatalf("stop body %s", stop.Body)
	}
	in = get(t, e, ref.ID)
	if in.Status != provision.StatusStopped || len(in.Endpoints) != 0 || !slices.Equal(in.Actions, []string{"start", "terminate"}) {
		t.Fatalf("stopped: %+v", in)
	}

	if err := e.c.Start(ctx, testKey, ref.ID); err != nil {
		t.Fatal(err)
	}
	if string(e.last(t, "PUT", "/api/v0/instances/").Body) != `{"state":"running"}` {
		t.Fatal("start body")
	}
	if in = get(t, e, ref.ID); in.Status != provision.StatusStarting {
		t.Fatalf("restart goes back through loading: %+v", in)
	}
	e.clock.Advance(6 * time.Second)
	in = get(t, e, ref.ID)
	if in.Status != provision.StatusRunning || len(in.Endpoints) != 1 || in.Endpoints[0].URL == ep.URL {
		t.Fatalf("a restart must map a new HostPort: %+v (was %s)", in.Endpoints, ep.URL)
	}

	if err := e.c.Terminate(ctx, testKey, ref.ID); err != nil {
		t.Fatal(err)
	}
	del := e.last(t, "DELETE", "/api/v0/instances/")
	if string(del.Body) != `{}` || del.Path != "/api/v0/instances/"+ref.ID+"/" {
		t.Fatalf("destroy %s %s", del.Path, del.Body)
	}
	if _, err := e.c.Get(ctx, testKey, ref); provision.ClassOf(err) != provision.ClassNotFound {
		t.Fatalf("instances:null must be not_found: %v", err)
	}
	if err := e.c.Terminate(ctx, testKey, ref.ID); err != nil {
		t.Fatalf("terminating a gone instance is nil: %v", err)
	}
	if err := e.c.Stop(ctx, testKey, ref.ID); provision.ClassOf(err) != provision.ClassNotFound {
		t.Fatalf("stop a gone instance: %v", err)
	}
}

func TestNoPortMappingAndCrash(t *testing.T) {
	e := setup(t, mockkit.Options{Faults: mockkit.Faults{NoPortMapping: true}})
	ref := launch(t, e, "57278070", "aitk-np")
	if in := get(t, e, ref.ID); in.Status != provision.StatusRunning || !in.PortMissing || len(in.Endpoints) != 0 {
		t.Fatalf("running without 8675: %+v", in)
	}

	e = setup(t, mockkit.Options{Faults: mockkit.Faults{CrashAfterRunning: true}, Delays: mockkit.Delays{Start: 2 * time.Second}})
	ref = launch(t, e, "57278070", "aitk-crash")
	e.clock.Advance(2 * time.Second)
	if in := get(t, e, ref.ID); in.Status != provision.StatusRunning {
		t.Fatalf("before the crash: %+v", in)
	}
	e.clock.Advance(2 * time.Second)
	in := get(t, e, ref.ID)
	if in.Status != provision.StatusFailed || in.ProviderStatus != "exited/running" || in.Detail == "" || !slices.Equal(in.Actions, []string{"terminate"}) {
		t.Fatalf("crashed: %+v", in)
	}
}

func TestNodeFactoryGetsTheContainer(t *testing.T) {
	var got mockkit.NodeSpec
	node := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(node.Close)
	stopped := 0
	e := setup(t, mockkit.Options{Faults: mockkit.Faults{OpenNode: true}, Node: func(n mockkit.NodeSpec) (string, func()) {
		got = n
		return node.URL, func() { stopped++ }
	}})
	ref := launch(t, e, "55830991", "aitk-node")
	in := get(t, e, ref.ID)
	if len(in.Endpoints) != 1 || in.Endpoints[0].URL != node.URL {
		t.Fatalf("endpoint %+v, want the node %s", in.Endpoints, node.URL)
	}
	if got.Kind != "vast" || got.InstanceID != ref.ID || got.Env["AI_TOOLKIT_AUTH"] != "aitk_tok" || got.Env["CONTAINER_ID"] != ref.ID ||
		got.Env["CONTAINER_API_KEY"] == "" || got.Image != "ghcr.io/geocine/aitoolkit-go:abc1234" || !got.OpenNode || got.GPU != "RTX 4090" || got.VRAMGB != 24 {
		t.Fatalf("node spec %+v", got)
	}
	if _, has := got.Env["-p 8675:8675"]; has {
		t.Fatal("docker flags are not container env")
	}
	if err := e.c.Stop(context.Background(), testKey, ref.ID); err != nil || stopped != 1 {
		t.Fatalf("stop must stop the node: %v %d", err, stopped)
	}
}

func TestListPagesAndFiltersByLabel(t *testing.T) {
	e := setup(t, mockkit.Options{})
	e.m.PageSize = 2
	launch(t, e, "57278070", "aitk-a")
	launch(t, e, "55830991", "other-b")
	launch(t, e, "90000001", "aitk-c")
	got, err := e.c.List(context.Background(), testKey, "aitk-")
	if err != nil {
		t.Fatal(err)
	}
	var labels []string
	for _, in := range got {
		labels = append(labels, in.Label)
	}
	if !slices.Equal(labels, []string{"aitk-a", "aitk-c"}) {
		t.Fatalf("labels %v", labels)
	}
	var pages []mockkit.Recorded
	for _, r := range e.m.Rec.Requests() {
		if r.Path == "/api/v1/instances/" {
			pages = append(pages, r)
		}
	}
	if len(pages) != 2 || pages[0].RawQuery != "limit=25" || !strings.Contains(pages[1].RawQuery, "after_token=") {
		t.Fatalf("pages %+v", pages)
	}
}

func TestTemplates(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ts, err := e.c.Templates(context.Background(), testKey)
	if err != nil {
		t.Fatal(err)
	}
	rec := e.last(t, "GET", "/api/v0/template/")
	q, _ := url.ParseQuery(rec.RawQuery)
	if q.Get("select_filters") != `{"creator_id":{"eq":123456}}` {
		t.Fatalf("select_filters %q", q.Get("select_filters"))
	}
	if len(ts) != 1 {
		t.Fatalf("own templates only: %+v", ts)
	}
	tp := ts[0]
	if tp.ID != "5915f1dc1ce881defb572015eb9d8178" || tp.Name != "aitk headless" || tp.Image != "ghcr.io/geocine/aitoolkit-go:latest" ||
		!tp.Ours || !tp.PublishesPort || !tp.HasToken {
		t.Fatalf("template %+v", tp)
	}
}

func TestLogs(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ref := launch(t, e, "57278070", "aitk-logs")
	text, err := e.c.Logs(context.Background(), testKey, ref.ID, 2)
	if err != nil {
		t.Fatal(err)
	}
	if text != "Starting /start.sh\naitk server listening on :8675\n" {
		t.Fatalf("logs %q", text)
	}
	if string(e.last(t, "PUT", "/api/v0/instances/request_logs/").Body) != `{"tail":"2"}` {
		t.Fatal("request_logs body")
	}
	// the first GET of the link is 404 (async upload): one 0.5 s retry
	if s := e.pace.sleeps(); !slices.Contains(s, 500*time.Millisecond) {
		t.Fatalf("no retry wait: %v", s)
	}
	// the link is fetched without the key
	for _, r := range e.m.Rec.Requests() {
		if strings.HasPrefix(r.Path, "/logs/") && r.Header.Get("Authorization") != "" {
			t.Fatal("the logs link was fetched with the key")
		}
	}
	// a link outside the allow-list is refused
	LogsHostAllowed = func(host string) bool { return host == "s3.amazonaws.com" }
	if _, err := e.c.Logs(context.Background(), testKey, ref.ID, 10); err == nil {
		t.Fatal("a result_url off the allow-list must be refused")
	}
}

func TestLogsURLAllowList(t *testing.T) {
	old := LogsHostAllowed
	t.Cleanup(func() { LogsHostAllowed = old })
	for raw, want := range map[string]bool{
		"https://s3.amazonaws.com/vast.ai/instance_logs/x.log": true,
		"https://vast-logs.s3.us-west-2.amazonaws.com/x.log":   true,
		"http://s3.amazonaws.com/vast.ai/instance_logs/x.log":  false, // not https
		"https://evil.example.com/x.log":                       false,
		"https://amazonaws.com.evil.example/x.log":             false,
		"https://user:pw@s3.amazonaws.com/x.log":               false,
		"http://127.0.0.1:9/logs/x.log":                        false, // loopback, but not admitted
		"":                                                     false,
	} {
		if got := logsURLAllowed(raw); got != want {
			t.Errorf("%q: %v, want %v", raw, got, want)
		}
	}
	LogsHostAllowed = func(host string) bool { return host == "127.0.0.1:9" }
	if !logsURLAllowed("http://127.0.0.1:9/logs/x.log") || logsURLAllowed("http://127.0.0.2:9/logs/x.log") {
		t.Error("an admitted loopback host may be plain http (the in-process mock)")
	}
}

func TestFaultsThroughTheMock(t *testing.T) {
	ctx := context.Background()
	q := provision.OfferQuery{}

	e := setup(t, mockkit.Options{Faults: mockkit.Faults{Unauthorized: true}})
	if _, err := e.c.Verify(ctx, testKey); provision.ClassOf(err) != provision.ClassAuth || strings.Contains(err.Error(), testKey) {
		t.Fatalf("unauthorized: %v", err)
	}

	e = setup(t, mockkit.Options{Faults: mockkit.Faults{NoCapacity: true}})
	if _, err := e.c.Launch(ctx, testKey, provision.LaunchSpec{Label: "aitk-x", OfferID: "57278070", Image: "i", DiskGB: 100}); provision.ClassOf(err) != provision.ClassCapacity {
		t.Fatalf("no capacity: %v", err)
	}

	e = setup(t, mockkit.Options{Faults: mockkit.Faults{NoCredit: true}})
	if _, err := e.c.Launch(ctx, testKey, provision.LaunchSpec{Label: "aitk-x", OfferID: "57278070", Image: "i", DiskGB: 100}); provision.ClassOf(err) != provision.ClassCredit {
		t.Fatalf("no credit: %v", err)
	}

	e = setup(t, mockkit.Options{})
	e.m.SetCredit(0)
	if _, err := e.c.Launch(ctx, testKey, provision.LaunchSpec{Label: "aitk-x", OfferID: "57278070", Image: "i", DiskGB: 100}); provision.ClassOf(err) != provision.ClassCredit {
		t.Fatalf("zero credit: %v", err)
	}

	e = setup(t, mockkit.Options{Faults: mockkit.Faults{RateLimitEvery: 1}})
	_, err := e.c.Offers(ctx, testKey, q)
	if provision.ClassOf(err) != provision.ClassRateLimited || !strings.Contains(err.Error(), "API requests too frequent") {
		t.Fatalf("plain-text 429: %v", err)
	}
}

// A full search window answers rate_limited at once (SearchOnly, with the
// wait as RetryAfter) instead of queueing past the recommender's budget.
func TestSearchWindowFullFailsFast(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ctx := context.Background()
	for i := range 10 {
		if _, err := e.c.Offers(ctx, testKey, provision.OfferQuery{MinVRAMGB: 24, DiskGB: 100}); err != nil {
			t.Fatalf("search %d: %v", i+1, err)
		}
	}
	before := len(e.m.Rec.Requests())
	_, err := e.c.Offers(ctx, testKey, provision.OfferQuery{MinVRAMGB: 24, DiskGB: 100})
	var pe *provision.Error
	if !asError(err, &pe) || pe.Class != provision.ClassRateLimited || !pe.SearchOnly || pe.RetryAfter < 50*time.Second || pe.RetryAfter > 62*time.Second {
		t.Fatalf("11th search: %+v", pe)
	}
	if n := len(e.m.Rec.Requests()); n != before {
		t.Fatalf("the refused search was sent (%d requests, was %d)", n, before)
	}
	if s := e.pace.sleeps(); len(s) != 0 {
		t.Fatalf("slept %v", s)
	}
	// a slot within searchMaxWait is still waited for
	e.pace.advance(50 * time.Second)
	if _, err := e.c.Offers(ctx, testKey, provision.OfferQuery{MinVRAMGB: 24, DiskGB: 100}); err != nil {
		t.Fatalf("after 50 s: %v", err)
	}
	if s := e.pace.sleeps(); len(s) != 1 || s[0] > searchMaxWait {
		t.Fatalf("sleeps %v", s)
	}
	// the instance endpoints are untouched by the search window
	if _, err := e.c.List(ctx, testKey, provision.LabelPrefix); err != nil {
		t.Fatalf("list: %v", err)
	}
}

func TestPlantedKeyNeverInErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprintf(w, `{"success":false,"error":"invalid_args","msg":"bad key %s in request"}`, testKey)
	}))
	t.Cleanup(srv.Close)
	old, oldPace := API, Pace
	API, Pace = srv.URL, NewLimiter(newSleepClock())
	t.Cleanup(func() { API, Pace = old, oldPace })
	_, err := adapter{}.Offers(context.Background(), testKey, provision.OfferQuery{})
	if err == nil || strings.Contains(err.Error(), testKey) || !strings.Contains(err.Error(), "[redacted]") {
		t.Fatalf("error %v", err)
	}
}

func TestSuccessFalseOn200IsAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"success":false,"error":"no_such_instance","msg":"Instance 5 not found"}`))
	}))
	t.Cleanup(srv.Close)
	old, oldPace := API, Pace
	API, Pace = srv.URL, NewLimiter(newSleepClock())
	t.Cleanup(func() { API, Pace = old, oldPace })
	err := adapter{}.Stop(context.Background(), testKey, "5")
	var e *provision.Error
	if provision.ClassOf(err) != provision.ClassNotFound || !strings.Contains(err.Error(), "Instance 5 not found") {
		t.Fatalf("200 success:false: %v", err)
	}
	if !asError(err, &e) || e.Status != 200 {
		t.Fatalf("status %+v", e)
	}
}

func asError(err error, out **provision.Error) bool { return errors.As(err, out) }

func TestClassify(t *testing.T) {
	reset := strconv.FormatInt(time.Now().Add(20*time.Second).Unix(), 10)
	for _, c := range []struct {
		name   string
		op     string
		status int
		body   string
		header http.Header
		want   provision.Class
	}{
		{"401", opAny, 401, `{"success":false,"error":"auth_error","msg":"Invalid user key"}`, nil, provision.ClassAuth},
		{"404 on search", opSearch, 404, `{"error":"unauthorized","msg":"Invalid user key"}`, nil, provision.ClassAuth},
		{"404 invalid user key", opAny, 404, `{"error":"not_found","msg":"invalid user key"}`, nil, provision.ClassAuth},
		{"auth_error on 200", opAny, 200, `{"success":false,"error":"auth_error","msg":"Invalid user key"}`, nil, provision.ClassAuth},
		{"403 on asks", opAsks, 403, `{"success":false,"error":"invalid_args","msg":"error 403/3586: Offer 1 is not your own."}`, nil, provision.ClassInvalid},
		{"403 elsewhere", opAny, 403, `{"error":"not_authorized","msg":"Not authorized"}`, nil, provision.ClassAuth},
		{"404 on asks", opAsks, 404, `{"success":false,"error":"invalid_args","msg":"error 404/3603: no_such_ask Instance type by id 1 is not available."}`, nil, provision.ClassCapacity},
		{"410 on asks", opAsks, 410, `{"success":false,"error":"no_such_ask","msg":"error 410/3907: no_such_ask Instance type 1 is no longer available."}`, nil, provision.ClassCapacity},
		{"missing_billing", opAsks, 400, `{"success":false,"error":"missing_billing","msg":"no billing"}`, nil, provision.ClassCredit},
		{"invalid_args", opAsks, 400, `{"success":false,"error":"invalid_args","msg":"image is required"}`, nil, provision.ClassInvalid},
		{"invalid_price", opAsks, 400, `{"success":false,"error":"invalid_price","msg":"price"}`, nil, provision.ClassInvalid},
		{"invalid_request", opAny, 400, `{"error":"invalid_request","msg":"Invalid json body"}`, nil, provision.ClassInvalid},
		{"no_such_instance", opAny, 404, `{"success":false,"error":"no_such_instance","msg":"gone"}`, nil, provision.ClassNotFound},
		{"not_found", opAny, 404, `{"success":false,"error":"not_found","msg":"Instance not found"}`, nil, provision.ClassNotFound},
		{"plain-text 429", opAny, 429, `API requests too frequent endpoint threshold=1.0`, http.Header{"X-Ratelimit-Reset": {reset}}, provision.ClassRateLimited},
		{"json 429", opAny, 429, `{"detail":"API requests too frequent endpoint threshold=1.0"}`, nil, provision.ClassRateLimited},
		{"rate_limit_exceeded", opAny, 400, `{"success":false,"error":"rate_limit_exceeded","msg":"API requests too frequent endpoint threshold=3.0"}`, nil, provision.ClassRateLimited},
		{"search quota", opAny, 429, `{"error":"search_quota_exceeded","msg":"quota"}`, nil, provision.ClassRateLimited},
		{"500", opAny, 502, `<html>bad gateway</html>`, nil, provision.ClassUnavailable},
		{"200 success false", opAny, 200, `{"success":false,"msg":"something"}`, nil, provision.ClassInvalid},
	} {
		h := c.header
		if h == nil {
			h = http.Header{}
		}
		err := classify(c.op, provision.Response{Status: c.status, Header: h, Body: []byte(c.body)}, testKey)
		var e *provision.Error
		if !asError(err, &e) || e.Class != c.want || e.Provider != provision.Vast || e.Status != c.status || e.Message == "" {
			t.Errorf("%s: %+v, want %s", c.name, e, c.want)
		}
		switch c.name {
		case "plain-text 429":
			if e.Message != "API requests too frequent endpoint threshold=1.0" || e.RetryAfter <= 0 || e.RetryAfter > 21*time.Second {
				t.Errorf("plain 429: %q %v", e.Message, e.RetryAfter)
			}
		case "json 429":
			if !strings.Contains(e.Message, "too frequent") {
				t.Errorf("json 429 detail: %q", e.Message)
			}
		case "search quota":
			if e.Message != "Vast's daily search quota is used up until UTC midnight" || e.RetryAfter <= 0 || e.RetryAfter > 24*time.Hour {
				t.Errorf("quota: %q %v", e.Message, e.RetryAfter)
			}
			if !e.SearchOnly {
				t.Errorf("quota: not SearchOnly, so it would pause the watcher")
			}
		default:
			if e.SearchOnly {
				t.Errorf("%s: SearchOnly", c.name)
			}
		}
	}
	if d := untilUTCMidnight(time.Date(2026, 10, 24, 23, 0, 0, 0, time.UTC)); d != time.Hour {
		t.Errorf("until midnight: %v", d)
	}
}

func TestStatusMapping(t *testing.T) {
	s := func(v string) *string { return &v }
	for _, c := range []struct {
		actual   *string
		intended string
		want     provision.Status
	}{
		{nil, "running", provision.StatusCreating},
		{s("created"), "running", provision.StatusStarting},
		{s("loading"), "running", provision.StatusStarting},
		{s("scheduling"), "running", provision.StatusStarting},
		{s("running"), "running", provision.StatusRunning},
		{s("running"), "stopped", provision.StatusStopping},
		{s("exited"), "stopped", provision.StatusStopped},
		{nil, "stopped", provision.StatusStopped},
		{s("exited"), "running", provision.StatusFailed},
		{s("unknown"), "running", provision.StatusFailed},
		{s("offline"), "running", provision.StatusFailed},
		{s("destroyed"), "running", provision.StatusUnknown},
	} {
		if got := mapStatus(c.actual, c.intended); got != c.want {
			t.Errorf("%v/%s: %s, want %s", c.actual, c.intended, got, c.want)
		}
	}
	// HostPort is a string in the live API; a number is tolerated; junk is no mapping
	for raw, want := range map[string]string{
		`{"8675/tcp":[{"HostIp":"0.0.0.0","HostPort":"41234"}]}`: "41234",
		`{"8675/tcp":[{"HostIp":"0.0.0.0","HostPort":41234}]}`:   "41234",
		`{"22/tcp":[{"HostPort":"41235"}]}`:                      "",
		`{}`:                                                     "",
		`null`:                                                   "",
		`[22, 8675]`:                                             "",
		`{"8675/tcp":[{"HostPort":"x"}]}`:                        "",
	} {
		if got := hostPort(json.RawMessage(raw)); got != want {
			t.Errorf("hostPort(%s) = %q, want %q", raw, got, want)
		}
	}
}

func TestTrailingSlashesAlwaysSent(t *testing.T) {
	e := setup(t, mockkit.Options{})
	ctx := context.Background()
	ref := launch(t, e, "57278070", "aitk-ts")
	_, _ = e.c.Verify(ctx, testKey)
	_, _ = e.c.Offers(ctx, testKey, provision.OfferQuery{})
	_ = get(t, e, ref.ID)
	_, _ = e.c.List(ctx, testKey, "aitk-")
	_ = e.c.Stop(ctx, testKey, ref.ID)
	_ = e.c.Terminate(ctx, testKey, ref.ID)
	for _, r := range e.m.Rec.Requests() {
		if !strings.HasSuffix(r.Path, "/") {
			t.Errorf("%s %s has no trailing slash", r.Method, r.Path)
		}
	}
}
