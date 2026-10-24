package quickpod

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/quickpod/quickpodmock"
)

var t0 = time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC)

const key = "qpk_test1234.secretValue"

var a = adapter{}

// setup starts a mock with a FakeClock and points the adapter at it.
func setup(t *testing.T, o mockkit.Options) (*quickpodmock.Mock, *mockkit.FakeClock) {
	t.Helper()
	clock := mockkit.NewFakeClock(t0)
	o.Clock = clock
	m := quickpodmock.New(o)
	u := m.Start()
	t.Cleanup(m.Close)
	oldAPI, oldNow, oldProxy, oldEnv, oldPort := API, now, UseProxy, PodEnv, PodPortFlag
	API, now = u, clock.Now
	t.Cleanup(func() { API, now, UseProxy, PodEnv, PodPortFlag = oldAPI, oldNow, oldProxy, oldEnv, oldPort })
	return m, clock
}

// stub points the adapter at a hand-written server (shapes the mock never sends).
func stub(t *testing.T, h http.HandlerFunc) {
	t.Helper()
	s := httptest.NewServer(h)
	t.Cleanup(s.Close)
	old := API
	API = s.URL
	t.Cleanup(func() { API = old })
}

func class(t *testing.T, err error, want provision.Class) {
	t.Helper()
	if got := provision.ClassOf(err); got != want {
		t.Fatalf("class %q (%v), want %q", got, err, want)
	}
}

func lastReq(m *quickpodmock.Mock, path string) mockkit.Recorded {
	var out mockkit.Recorded
	for _, r := range m.Rec.Requests() {
		if r.Path == path {
			out = r
		}
	}
	return out
}

func TestRegisteredAndInfo(t *testing.T) {
	c, ok := provision.Lookup(provision.QuickPod)
	if !ok || c.Kind() != provision.QuickPod {
		t.Fatal("quickpod is not registered")
	}
	in := a.Info()
	if in.Stage != "beta" || in.Label != "QuickPod" || !in.Caps.TemplateRequired || !in.Caps.EnvFlagString ||
		in.Caps.HTTPS != "unconfirmed" || in.Caps.SelfDestroy != "key" || !in.Caps.StopKeepsDisk {
		t.Fatalf("info %+v", in)
	}
	if in.Caps.ImageDigestRef != "unconfirmed" || !in.Caps.ReportsInet || !in.Caps.ReportsDiskBW {
		t.Fatalf("image/bandwidth caps %+v", in.Caps)
	}
	if len(in.Unconfirmed) != 4 || !slices.Contains(in.KeyScopes, "templates:read") || in.KeysURL != "https://console.quickpod.io/settings" {
		t.Fatalf("unconfirmed/scopes %+v", in)
	}
}

func TestOffersMapping(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	got, err := a.Offers(context.Background(), key, provision.OfferQuery{})
	if err != nil {
		t.Fatal(err)
	}
	// 1 GPU by default; the offline 4090 and the 4-GPU row are dropped.
	var ids []string
	for _, o := range got {
		ids = append(ids, o.ID)
	}
	if want := []string{"6289141", "6302336", "7000001"}; !slices.Equal(ids, want) {
		t.Fatalf("ids %v, want %v (sorted by price)", ids, want)
	}
	o := got[0]
	if o.Provider != provision.QuickPod || o.GPU != "RTX 5060 Ti" || o.GPUCount != 1 || o.VRAMGB != 16 {
		t.Fatalf("offer %+v", o)
	}
	// 0.14 compute + 0.1 $/TB-h x 100 GB / 1000
	if o.PricePerHour != 0.15 || o.StoppedPricePerHour == nil || *o.StoppedPricePerHour != 0.01 {
		t.Fatalf("price %v stopped %v", o.PricePerHour, o.StoppedPricePerHour)
	}
	if o.Reliability == nil || *o.Reliability != 0.982958 || o.Location != "US, Indiana, Carmel" ||
		o.MaxDiskGB == nil || *o.MaxDiskGB != 444 || !o.HTTPS || o.Availability != "available" || o.Host == nil || o.Fit != nil {
		t.Fatalf("offer fields %+v", o)
	}
	if got[1].VRAMGB != 24 || got[1].PricePerHour != 0.195 || got[2].GPU != "RTX A6000" || got[2].VRAMGB != 48 {
		t.Fatalf("others %+v %+v", got[1], got[2])
	}
	// /rentable is public: no credentials go with it.
	r := lastReq(m, "/rentable")
	if r.Header.Get("X-API-Key") != "" || r.Header.Get("Authorization") != "" || r.RawQuery != "" {
		t.Fatalf("rentable request %+v", r)
	}
}

func TestOffersFilters(t *testing.T) {
	setup(t, mockkit.Options{})
	for _, c := range []struct {
		name string
		q    provision.OfferQuery
		want []string
	}{
		{"min vram", provision.OfferQuery{MinVRAMGB: 20}, []string{"6302336", "7000001"}},
		{"max vram exclusive", provision.OfferQuery{MaxVRAMGB: 24}, []string{"6289141"}},
		{"max vram above", provision.OfferQuery{MaxVRAMGB: 25}, []string{"6289141", "6302336"}},
		{"tier 24", provision.OfferQuery{MinVRAMGB: 21, MaxVRAMGB: 29}, []string{"6302336"}},
		{"gpu substring", provision.OfferQuery{GPU: "3090"}, []string{"6302336"}},
		{"gpu case", provision.OfferQuery{GPU: "rtx a6000"}, []string{"7000001"}},
		{"count", provision.OfferQuery{GPUCount: 4}, []string{"72868451"}},
		{"disk", provision.OfferQuery{DiskGB: 500}, []string{"6302336", "7000001"}},
		{"price", provision.OfferQuery{MaxPrice: 0.2}, []string{"6289141", "6302336"}},
		{"fast", provision.OfferQuery{Fast: true}, []string{"6289141", "7000001"}},
		{"limit", provision.OfferQuery{Limit: 1}, []string{"6289141"}},
	} {
		got, err := a.Offers(context.Background(), key, c.q)
		if err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, o := range got {
			ids = append(ids, o.ID)
		}
		if !slices.Equal(ids, c.want) {
			t.Errorf("%s: %v, want %v", c.name, ids, c.want)
		}
	}
	// The disk is priced: 500 GB on the 3090 host at 0.15 $/TB-h.
	got, _ := a.Offers(context.Background(), key, provision.OfferQuery{GPU: "3090", DiskGB: 500})
	if got[0].PricePerHour != 0.255 || *got[0].StoppedPricePerHour != 0.075 {
		t.Fatalf("disk price %+v", got[0])
	}
}

// A 0 or missing bandwidth passes Fast (D-15); only a reported value below
// the threshold drops the host.
func TestFastUnreported(t *testing.T) {
	var r rawOffer
	r.NumGPUs, r.GPUVRAM, r.Machine.Online = 1, 24576, true
	q := provision.OfferQuery{Fast: true}
	if !keep(r, q, 100, 1) {
		t.Fatal("no bandwidth reported must pass")
	}
	r.Machine.InetDown = 499
	if keep(r, q, 100, 1) {
		t.Fatal("499 Mbps must not pass")
	}
	r.Machine.InetDown, r.Machine.DiskBW = 500, 299
	if keep(r, q, 100, 1) {
		t.Fatal("299 MB/s disk must not pass")
	}
	r.Machine.DiskBW = 300
	if !keep(r, q, 100, 1) {
		t.Fatal("500 Mbps and 300 MB/s pass")
	}
}

func TestOffersLimitCap(t *testing.T) {
	rows := make([]map[string]any, 250)
	for i := range rows {
		rows[i] = map[string]any{"id": i + 1, "gpu_type": "NVIDIA GeForce RTX 4090", "num_gpus": 1, "gpu_vram": 24564,
			"hourly_cost": 0.3, "max_disk_size": 500, "_machines": map[string]any{"online": true}}
	}
	stub(t, func(w http.ResponseWriter, r *http.Request) { json.NewEncoder(w).Encode(rows) })
	for _, c := range []struct{ limit, want int }{{0, 50}, {200, 200}, {500, 200}} {
		got, err := a.Offers(context.Background(), key, provision.OfferQuery{Limit: c.limit})
		if err != nil || len(got) != c.want {
			t.Errorf("limit %d: %d offers (%v), want %d", c.limit, len(got), err, c.want)
		}
	}
}

func TestQuote(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	ctx := context.Background()
	o, err := a.Quote(ctx, key, "6302336", provision.OfferQuery{DiskGB: 200})
	if err != nil || o.PricePerHour != 0.21 || o.ID != "6302336" {
		t.Fatalf("quote %+v %v", o, err)
	}
	if r := lastReq(m, "/update/offer_is_busy"); r.RawQuery != "offer_id=6302336" {
		t.Fatalf("busy check %q", r.RawQuery)
	}
	_, err = a.Quote(ctx, key, "6302336", provision.OfferQuery{DiskGB: 800})
	class(t, err, provision.ClassInvalid)
	_, err = a.Quote(ctx, key, "999", provision.OfferQuery{})
	class(t, err, provision.ClassCapacity)
	m.Occupy(6302336)
	_, err = a.Quote(ctx, key, "6302336", provision.OfferQuery{})
	class(t, err, provision.ClassCapacity)
}

func TestQuoteBusy(t *testing.T) {
	stub(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update/offer_is_busy" {
			w.Write([]byte("true"))
			return
		}
		w.Write([]byte(`[{"id":7,"gpu_type":"NVIDIA GeForce RTX 3090","num_gpus":1,"gpu_vram":24576,"hourly_cost":0.2,"_machines":{"online":true}}]`))
	})
	_, err := a.Quote(context.Background(), key, "7", provision.OfferQuery{})
	class(t, err, provision.ClassCapacity)
}

func TestTemplates(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	m.AddTemplate(quickpodmock.Template{UUID: "pinned", Name: "aitk pinned", ImagePath: "ghcr.io/geocine/aitoolkit-go", VersionTag: "f53a407",
		DockerOptions: "--shm-size=16g -p   8675:8675"})
	got, err := a.Templates(context.Background(), key)
	if err != nil {
		t.Fatal(err)
	}
	want := []provision.Template{
		{ID: quickpodmock.AitkTemplateUUID, Name: "aitk headless", Image: "ghcr.io/geocine/aitoolkit-go:latest", Ours: true, PublishesPort: true, HasToken: true},
		{ID: quickpodmock.OtherTemplateUUID, Name: "FluxGym", Image: "quickpod/fluxgym:latest"},
		{ID: "pinned", Name: "aitk pinned", Image: "ghcr.io/geocine/aitoolkit-go:f53a407", Ours: true, PublishesPort: true},
	}
	if !slices.Equal(got, want) {
		t.Fatalf("templates\n%+v\nwant\n%+v", got, want)
	}
	b, _ := json.Marshal(got)
	if strings.Contains(string(b), quickpodmock.TemplateToken) || strings.Contains(string(b), "docker_options") {
		t.Fatalf("docker_options leaked: %s", b)
	}
	r := lastReq(m, "/templates")
	if r.Header.Get("X-API-Key") != key || r.Header.Get("Authorization") != "ApiKey "+key {
		t.Fatalf("auth headers %v", r.Header)
	}
}

func TestLaunchBody(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	ctx := context.Background()
	spec := provision.LaunchSpec{Label: "aitk-abc", OfferID: "6302336", Image: "ghcr.io/geocine/aitoolkit-go:f53a407", DiskGB: 150,
		TemplateID: quickpodmock.AitkTemplateUUID, Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_tok", "AITK_STUDIO_URL": "https://studio.example.com/x"}}
	ref, err := a.Launch(ctx, key, spec)
	if err != nil {
		t.Fatal(err)
	}
	if ref != (provision.Ref{OfferID: "6302336", Label: "aitk-abc"}) {
		t.Fatalf("ref %+v (QuickPod returns no id)", ref)
	}
	r := lastReq(m, "/update/api/createpod")
	if r.Header.Get("X-API-Key") != key || r.Header.Get("Authorization") != "ApiKey "+key {
		t.Fatalf("auth headers %v", r.Header)
	}
	var body map[string]any
	if err := json.Unmarshal(r.Body, &body); err != nil {
		t.Fatal(err)
	}
	// offers_id an int, disk_size a string, docker_options only -e pairs in key order.
	want := map[string]any{"offers_id": 6302336.0, "template_uuid": quickpodmock.AitkTemplateUUID, "disk_size": "150", "altname": "aitk-abc",
		"docker_options": "-e AITK_STUDIO_URL=https://studio.example.com/x -e AI_TOOLKIT_AUTH=aitk_tok"}
	if len(body) != len(want) {
		t.Fatalf("body %v, want %v", body, want)
	}
	for k, v := range want {
		if body[k] != v {
			t.Errorf("%s = %#v, want %#v", k, body[k], v)
		}
	}
	if strings.Contains(string(r.Body), "aitoolkit-go") {
		t.Error("the image comes from the template and must not be sent")
	}

	// Another create on the same offer: busy.
	_, err = a.Launch(ctx, key, spec)
	class(t, err, provision.ClassCapacity)

	// PodPortFlag adds the port mapping first; PodEnv=false drops the pairs.
	PodPortFlag = true
	spec.OfferID = "7000001"
	if _, err := a.Launch(ctx, key, spec); err != nil {
		t.Fatal(err)
	}
	json.Unmarshal(lastReq(m, "/update/api/createpod").Body, &body)
	if s, _ := body["docker_options"].(string); !strings.HasPrefix(s, "-p 8675:8675 -e ") {
		t.Fatalf("docker_options %q", s)
	}
	PodPortFlag, PodEnv = false, false
	spec.OfferID = "6289141"
	if _, err := a.Launch(ctx, key, spec); err != nil {
		t.Fatal(err)
	}
	body = nil
	json.Unmarshal(lastReq(m, "/update/api/createpod").Body, &body)
	if _, has := body["docker_options"]; has {
		t.Fatalf("PodEnv off still sent %v", body["docker_options"])
	}
}

func TestLaunchRefusesUnsafeEnv(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	for _, env := range []map[string]string{
		{"A": "a b"}, {"A": "1\v-v\v/:/host"}, {"A": "1\f-v"}, {"A": "1\u0085-v"}, {"A": "1\u00a0-v"}, {"A": "1\x00"},
		{"A": "x;rm"}, {"A": "$(id)"}, {"a": "1"}, {"A-B": "1"},
	} {
		_, err := a.Launch(context.Background(), key, provision.LaunchSpec{Label: "aitk-x", OfferID: "6302336", TemplateID: quickpodmock.AitkTemplateUUID, Env: env})
		class(t, err, provision.ClassInvalid)
	}
	_, err := a.Launch(context.Background(), key, provision.LaunchSpec{Label: "aitk-x", OfferID: "6302336"})
	class(t, err, provision.ClassInvalid)
	_, err = a.Launch(context.Background(), key, provision.LaunchSpec{Label: "aitk-x", OfferID: "abc", TemplateID: "t"})
	class(t, err, provision.ClassInvalid)
	if n := len(m.Rec.Requests()); n != 0 {
		t.Fatalf("%d requests sent for refused launches", n)
	}
}

func TestLifecycle(t *testing.T) {
	m, clock := setup(t, mockkit.Options{Delays: mockkit.Delays{Provision: 3 * time.Second, Start: 6 * time.Second}})
	ctx := context.Background()
	ref, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-life", OfferID: "6289141", DiskGB: 100,
		TemplateID: quickpodmock.AitkTemplateUUID, Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_tok"}})
	if err != nil {
		t.Fatal(err)
	}
	expect := func(want provision.Status) provision.Instance {
		t.Helper()
		in, err := a.Get(ctx, key, ref)
		if err != nil {
			t.Fatal(err)
		}
		if in.Status != want {
			t.Fatalf("status %s (%s), want %s", in.Status, in.ProviderStatus, want)
		}
		return in
	}
	// Resolved by offers_id + altname while the id is unknown.
	in := expect(provision.StatusCreating)
	if in.ProviderID == "" || in.Label != "aitk-life" || in.ProviderStatus != "-/running" || len(in.Endpoints) != 1 {
		t.Fatalf("creating %+v", in)
	}
	ref.ID = in.ProviderID
	clock.Advance(9 * time.Second)
	in = expect(provision.StatusRunning)
	seg, _, _ := strings.Cut(in.ProviderID, "-")
	want := []provision.Endpoint{{URL: "https://" + seg + "-8675.proxy.quickpod.io", Secure: true, Via: "proxy"}, {URL: "http://203.0.113.11:21829", Via: "direct"}}
	if !slices.Equal(in.Endpoints, want) || in.PortMissing {
		t.Fatalf("endpoints %+v", in.Endpoints)
	}
	if in.GPU != "RTX 5060 Ti" || in.GPUCount != 1 || in.VRAMGB != 16 || in.Location != "US, Indiana, Carmel" ||
		*in.PricePerHour != 0.15 || *in.StoppedPricePerHour != 0.01 || !in.CreatedAt.Equal(t0) ||
		!slices.Equal(in.Actions, []string{"stop", "terminate"}) {
		t.Fatalf("running %+v", in)
	}
	list, err := a.List(ctx, key, provision.LabelPrefix)
	if err != nil || len(list) != 1 || list[0].ProviderID != ref.ID {
		t.Fatalf("list %+v %v", list, err)
	}
	if list, _ := a.List(ctx, key, "other-"); len(list) != 0 {
		t.Fatalf("prefix filter %+v", list)
	}

	if err := a.Stop(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	if r := lastReq(m, "/update/api/stoppod"); r.Method != http.MethodGet || r.RawQuery != "pod_uuid="+ref.ID {
		t.Fatalf("stop request %+v", r)
	}
	expect(provision.StatusStopping)
	clock.Advance(3 * time.Second)
	in = expect(provision.StatusStopped)
	if !slices.Equal(in.Actions, []string{"start", "terminate"}) {
		t.Fatalf("stopped actions %v", in.Actions)
	}
	if err := a.Start(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	expect(provision.StatusStarting)
	clock.Advance(6 * time.Second)
	expect(provision.StatusRunning)

	logs, err := a.Logs(ctx, key, ref.ID, 1)
	if err != nil || logs != "aitk: listening on :8675" {
		t.Fatalf("logs %q %v", logs, err)
	}

	if err := a.Terminate(ctx, key, ref.ID); err != nil {
		t.Fatal(err)
	}
	in = expect(provision.StatusTerminated)
	if len(in.Endpoints) != 0 || in.Actions != nil || in.ProviderStatus != "destroyed" {
		t.Fatalf("terminated %+v", in)
	}
	if err := a.Terminate(ctx, key, ref.ID); err != nil {
		t.Fatalf("terminate again: %v", err)
	}
	_, err = a.Get(ctx, key, provision.Ref{ID: "nope"})
	class(t, err, provision.ClassNotFound)
	err = a.Stop(ctx, key, "")
	class(t, err, provision.ClassInvalid)
}

// The 60 s creating window is measured on the adapter's clock.
func TestCreatingWindow(t *testing.T) {
	p := rawPod{Intended: "running", CreatedAt: t0.UnixMilli()}
	if s, _ := statusOf(p, t0.Add(59*time.Second)); s != provision.StatusCreating {
		t.Fatalf("59 s: %s", s)
	}
	if s, _ := statusOf(p, t0.Add(61*time.Second)); s != provision.StatusStarting {
		t.Fatalf("61 s: %s", s)
	}
}

func TestStatusTable(t *testing.T) {
	late := t0.Add(time.Hour)
	for _, c := range []struct {
		state, intended string
		destroyed       bool
		want            provision.Status
	}{
		{"running", "running", true, provision.StatusTerminated},
		{"exited", "exited", true, provision.StatusTerminated},
		{"running", "exited", false, provision.StatusStopping},
		{"running", "running", false, provision.StatusRunning},
		{"running", "", false, provision.StatusRunning},
		{"exited", "running", false, provision.StatusStarting},
		{"", "running", false, provision.StatusStarting},
		{"created", "running", false, provision.StatusStarting},
		{"exited", "exited", false, provision.StatusStopped},
		{"exited", "", false, provision.StatusStopped},
		{"dead", "running", false, provision.StatusFailed},
		{"paused", "exited", false, provision.StatusUnknown},
		{"", "", false, provision.StatusUnknown},
	} {
		p := rawPod{State: c.state, Intended: c.intended, Destroyed: c.destroyed, CreatedAt: t0.UnixMilli()}
		got, detail := statusOf(p, late)
		if got != c.want {
			t.Errorf("%q/%q destroyed=%v: %s, want %s", c.state, c.intended, c.destroyed, got, c.want)
		}
		if got == provision.StatusUnknown && !strings.Contains(detail, `"`+c.state+`"`) {
			t.Errorf("unknown detail %q lacks the raw state", detail)
		}
	}
}

func TestParse8675(t *testing.T) {
	for _, c := range []struct {
		in   string
		ip   string
		port int
		ok   bool
	}{
		{"8675 -> 1.2.3.4:40601", "1.2.3.4", 40601, true},
		{"22 -> 1.2.3.4:40600 | 8675 -> 1.2.3.4:40601", "1.2.3.4", 40601, true},
		{"8675 -> 96.12.34.56:40601|22 -> 96.12.34.56:40600", "96.12.34.56", 40601, true},
		{"22 -> 1.2.3.4:40600, 8675/tcp -> 1.2.3.4:40602", "1.2.3.4", 40602, true},
		{"   8675/tcp   ->   5.6.7.8:1234   ", "5.6.7.8", 1234, true},
		{"18675 -> 1.2.3.4:40601", "", 0, false},
		{"22 -> 1.2.3.4:40600", "", 0, false},
		{"8675 -> host.example:40601", "", 0, false},
		{"8675 -> 1.2.3.4:99999", "", 0, false},
		{"8675 -> 1.2.3.4", "", 0, false},
		{"", "", 0, false},
	} {
		ip, port, ok := Parse8675(c.in)
		if ip != c.ip || port != c.port || ok != c.ok {
			t.Errorf("Parse8675(%q) = %q %d %v, want %q %d %v", c.in, ip, port, ok, c.ip, c.port, c.ok)
		}
	}
}

func TestProxyURL(t *testing.T) {
	if got := ProxyURL("a1b2c3d4-e5f6-4a7b-8c9d-0123456789ab"); got != "https://a1b2c3d4-8675.proxy.quickpod.io" {
		t.Fatalf("proxy %q", got)
	}
	UseProxy = false
	defer func() { UseProxy = true }()
	in := instanceOf(rawPod{Names: "a1b2-c3", State: "running", Intended: "running", PortMappings: "8675 -> 1.2.3.4:5"}, t0)
	if len(in.Endpoints) != 1 || in.Endpoints[0].Via != "direct" {
		t.Fatalf("UseProxy=false endpoints %+v", in.Endpoints)
	}
}

func TestCreditErrors(t *testing.T) {
	for _, status := range []int{http.StatusPaymentRequired, http.StatusBadRequest} {
		m, _ := setup(t, mockkit.Options{})
		m.CreditStatus = status
		m.SetCredit(0.1)
		_, err := a.Launch(context.Background(), key, provision.LaunchSpec{Label: "aitk-x", OfferID: "6302336", TemplateID: quickpodmock.AitkTemplateUUID})
		class(t, err, provision.ClassCredit)
		var e *provision.Error
		if !errors.As(err, &e) || e.Status != status || !strings.Contains(e.Message, "$0.10 left") {
			t.Fatalf("credit error at %d: %+v", status, e)
		}
	}
	setup(t, mockkit.Options{Faults: mockkit.Faults{NoCredit: true}})
	_, err := a.Launch(context.Background(), key, provision.LaunchSpec{Label: "aitk-x", OfferID: "6302336", TemplateID: quickpodmock.AitkTemplateUUID})
	class(t, err, provision.ClassCredit)
}

func TestCapacityAuthConflict(t *testing.T) {
	ctx := context.Background()
	setup(t, mockkit.Options{Faults: mockkit.Faults{NoCapacity: true}})
	_, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-x", OfferID: "6302336", TemplateID: quickpodmock.AitkTemplateUUID})
	class(t, err, provision.ClassCapacity)
	if e := err.(*provision.Error); e.Status != http.StatusNotAcceptable {
		t.Fatalf("status %d", e.Status)
	}

	setup(t, mockkit.Options{Faults: mockkit.Faults{Unauthorized: true}})
	_, err = a.Verify(ctx, key)
	class(t, err, provision.ClassAuth)

	m, _ := setup(t, mockkit.Options{})
	if _, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-x", OfferID: "6302336", TemplateID: quickpodmock.AitkTemplateUUID}); err != nil {
		t.Fatal(err)
	}
	pods := m.Instances()
	id := pods[0].UUID
	// Another account's key: 403 -> auth.
	err = a.Stop(ctx, "qpk_other.key", id)
	class(t, err, provision.ClassAuth)
	// Stopped, then someone else rents the GPU: start is 409 -> conflict.
	if err := a.Stop(ctx, key, id); err != nil {
		t.Fatal(err)
	}
	m.Occupy(6302336)
	err = a.Start(ctx, key, id)
	class(t, err, provision.ClassConflict)
	if e := err.(*provision.Error); e.Message != "This machine is in use at the moment" {
		t.Fatalf("conflict message %q", e.Message)
	}
	// Other provider text is kept in brackets.
	if e := apiError("start", provision.Response{Status: http.StatusConflict, Body: []byte(`{"message":"busy"}`)}, key); e.Message != "This machine is in use at the moment (busy)" {
		t.Fatalf("conflict message %q", e.Message)
	}
}

func TestRateLimited(t *testing.T) {
	setup(t, mockkit.Options{Faults: mockkit.Faults{RateLimitEvery: 2}})
	ctx := context.Background()
	if _, err := a.Verify(ctx, key); err != nil {
		t.Fatal(err)
	}
	_, err := a.Verify(ctx, key)
	class(t, err, provision.ClassRateLimited)
	if e := err.(*provision.Error); e.RetryAfter != 2*time.Second {
		t.Fatalf("retry after %v", e.RetryAfter)
	}
}

func TestVerify(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	m.SetCredit(12.53)
	acc, err := a.Verify(context.Background(), key)
	if err != nil || acc.Credit == nil || *acc.Credit != 12.53 || acc.Detail != "$12.53 credit" {
		t.Fatalf("verify %+v %v", acc, err)
	}
	if r := lastReq(m, "/update/api/me"); r.Header.Get("X-API-Key") != key || r.Header.Get("Authorization") != "ApiKey "+key {
		t.Fatalf("auth headers %v", r.Header)
	}
}

func TestErrorClassesAndRedaction(t *testing.T) {
	for _, c := range []struct {
		op     string
		status int
		body   string
		want   provision.Class
	}{
		{"create", 400, `{"error":"insufficient_credit","current_credit":0.42}`, provision.ClassCredit},
		{"start", 409, `{"error":"insufficient_credit","action":"start"}`, provision.ClassCredit},
		{"create", 406, `{"message":"Duplicate or Invalid ID"}`, provision.ClassCapacity},
		{"create", 409, `{"message":"offer is busy"}`, provision.ClassCapacity},
		{"create", 400, `{"message":"The Offer Is Busy"}`, provision.ClassCapacity},
		{"start", 409, `{"message":"busy"}`, provision.ClassConflict},
		{"stop", 401, `{"error":"unauthorized"}`, provision.ClassAuth},
		{"stop", 403, `{"error":"forbidden"}`, provision.ClassAuth},
		{"stop", 400, `{"details":"pod_uuid is required"}`, provision.ClassInvalid},
		{"stop", 500, `oops`, provision.ClassUnavailable},
		{"stop", 502, ``, provision.ClassUnavailable},
		{"stop", 402, `{}`, provision.ClassCredit},
	} {
		e := apiError(c.op, provision.Response{Status: c.status, Body: []byte(c.body), Header: http.Header{}}, key)
		if e.Class != c.want || e.Message == "" {
			t.Errorf("%s %d %s: %s %q, want %s", c.op, c.status, c.body, e.Class, e.Message, c.want)
		}
	}
	if e := apiError("x", provision.Response{Status: 400, Body: []byte(`{"error":"","details":"d","message":"m"}`)}, key); e.Message != "d" {
		t.Errorf("first non-empty of error, details, message: %q", e.Message)
	}

	// A planted key never reaches an error.
	stub(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"bad key ` + r.Header.Get("X-API-Key") + `"}`))
	})
	_, err := a.Verify(context.Background(), key)
	if err == nil || strings.Contains(err.Error(), key) || !strings.Contains(err.Error(), "[redacted]") {
		t.Fatalf("key leaked: %v", err)
	}
}

// An error that echoes the createpod body carries no env value: the machine
// token and the S3 secret are redacted like the key, on an error status and
// on a 200 without success.
func TestLaunchRedactsEnv(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusOK} {
		stub(t, func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			w.WriteHeader(status)
			json.NewEncoder(w).Encode(map[string]string{"error": "bad docker_options in " + string(b) + " for " + r.Header.Get("X-API-Key")})
		})
		_, err := a.Launch(context.Background(), key, provision.LaunchSpec{Label: "aitk-x", OfferID: "6302336", TemplateID: "t",
			Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_SECRETTOKEN123", "AITK_S3_SECRET_ACCESS_KEY": "S3SECRETVALUE", "AITK_X": "1"}})
		class(t, err, provision.ClassInvalid)
		msg := err.Error()
		for _, sec := range []string{"aitk_SECRETTOKEN123", "S3SECRETVALUE", key} {
			if strings.Contains(msg, sec) {
				t.Fatalf("%d: %q leaked: %s", status, sec, msg)
			}
		}
		if !strings.Contains(msg, "AI_TOOLKIT_AUTH=[redacted]") || !strings.Contains(msg, "AITK_X=1") {
			t.Fatalf("%d: message %s", status, msg)
		}
	}
}

// sim_crash: the pod reads running once (the node starts), then dead, which
// is failed, and the node stops.
func TestCrashAfterRunning(t *testing.T) {
	var mu sync.Mutex
	starts, stops := 0, 0
	factory := func(mockkit.NodeSpec) (string, func()) {
		mu.Lock()
		starts++
		mu.Unlock()
		return "http://127.0.0.1:45001", func() { mu.Lock(); stops++; mu.Unlock() }
	}
	setup(t, mockkit.Options{Faults: mockkit.Faults{CrashAfterRunning: true}, Node: factory})
	ctx := context.Background()
	ref, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-crash", OfferID: "6302336", TemplateID: quickpodmock.AitkTemplateUUID,
		Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_tok"}})
	if err != nil {
		t.Fatal(err)
	}
	in, err := a.Get(ctx, key, ref)
	if err != nil || in.Status != provision.StatusRunning {
		t.Fatalf("first read %+v %v", in, err)
	}
	mu.Lock()
	if starts != 1 || stops != 0 {
		t.Fatalf("after running: %d starts, %d stops", starts, stops)
	}
	mu.Unlock()
	in, err = a.Get(ctx, key, provision.Ref{ID: in.ProviderID})
	if err != nil || in.Status != provision.StatusFailed || in.Detail != "QuickPod reports the container as dead." || in.ProviderStatus != "dead/running" {
		t.Fatalf("second read %+v %v", in, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if starts != 1 || stops != 1 {
		t.Fatalf("after dead: %d starts, %d stops", starts, stops)
	}
}

// PodEnv append/replace/ignore, asserted at the mock level: the node env has
// our token; no 8675 mapping; the template's token.
func TestPodEnvModes(t *testing.T) {
	for _, c := range []struct {
		mode      mockkit.PodEnvMode
		token     string // "" = no node starts
		portThere bool
	}{
		{mockkit.PodEnvAppend, "aitk_ours", true},
		{mockkit.PodEnvReplace, "", false},
		{mockkit.PodEnvIgnore, quickpodmock.TemplateToken, true},
	} {
		var mu sync.Mutex
		var nodes []mockkit.NodeSpec
		factory := func(n mockkit.NodeSpec) (string, func()) {
			mu.Lock()
			nodes = append(nodes, n)
			mu.Unlock()
			return "http://127.0.0.1:45001", func() {}
		}
		setup(t, mockkit.Options{Faults: mockkit.Faults{PodEnv: c.mode}, Node: factory})
		ctx := context.Background()
		ref, err := a.Launch(ctx, key, provision.LaunchSpec{Label: "aitk-env", OfferID: "6302336", TemplateID: quickpodmock.AitkTemplateUUID,
			Env: map[string]string{"AI_TOOLKIT_AUTH": "aitk_ours", "AITK_X": "1"}})
		if err != nil {
			t.Fatal(err)
		}
		in, err := a.Get(ctx, key, ref)
		if err != nil || in.Status != provision.StatusRunning {
			t.Fatalf("%q: %+v %v", c.mode, in, err)
		}
		if in.PortMissing == c.portThere {
			t.Errorf("%q: PortMissing %v", c.mode, in.PortMissing)
		}
		mu.Lock()
		if c.token == "" {
			if len(nodes) != 0 {
				t.Errorf("%q: a node started without a port mapping", c.mode)
			}
		} else if len(nodes) != 1 || nodes[0].Env["AI_TOOLKIT_AUTH"] != c.token || nodes[0].Kind != "quickpod" ||
			nodes[0].Image != "ghcr.io/geocine/aitoolkit-go:latest" || nodes[0].VRAMGB != 24 {
			t.Errorf("%q: nodes %+v", c.mode, nodes)
		}
		mu.Unlock()
		if c.portThere {
			if len(in.Endpoints) != 2 || in.Endpoints[1].URL != "http://127.0.0.1:45001" {
				t.Errorf("%q: endpoints %+v", c.mode, in.Endpoints)
			}
		}
	}
}
