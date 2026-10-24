package v1images

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/handlers/v1machines"
	"github.com/geocine/aitoolkit/internal/codemode"
	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/imagetest"
	"github.com/geocine/aitoolkit/internal/provision/sim"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

const defaultImage = "ghcr.io/geocine/aitoolkit-go:latest"

// fakeManager answers the offer and template lookups; every other call
// panics (the embedded nil Manager), since these routes make none.
type fakeManager struct {
	provision.Manager
	mu        sync.Mutex
	host      *hostfit.Host
	err       error
	kind      provision.Kind
	offerID   string
	query     provision.OfferQuery
	templates []provision.Template
}

func (f *fakeManager) Quote(_ context.Context, k provision.Kind, id string, q provision.OfferQuery) (provision.Offer, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.kind, f.offerID, f.query = k, id, q
	return provision.Offer{ID: id, Provider: k, Host: f.host}, f.err
}

func (f *fakeManager) Templates(context.Context, provision.Kind) ([]provision.Template, error) {
	return f.templates, f.err
}

// newEnv serves the routes against registrymock and hfmock (imagetest) and
// a fake provision.Manager. The GPU verdicts are judged afresh.
func newEnv(t *testing.T, o imagetest.Options) (*fakeManager, http.Handler) {
	t.Helper()
	t.Setenv("AITK_PROVISION_PROVIDERS", "")
	imagetest.Use(t, o)
	f := &fakeManager{}
	old, oldBuild := provision.Default, ServerBuild
	provision.Default = f
	t.Cleanup(func() { provision.Default, ServerBuild = old, oldBuild })
	resetGPUs(t)
	mux := http.NewServeMux()
	Register(mux)
	return f, mux
}

func resetGPUs(t *testing.T) {
	gpusCache.mu.Lock()
	gpusCache.items = nil
	gpusCache.mu.Unlock()
	t.Cleanup(func() {
		gpusCache.mu.Lock()
		gpusCache.items = nil
		gpusCache.mu.Unlock()
	})
}

type response struct {
	code int
	body []byte
}

// call serves one request; no answer may carry a secret.
func call(t *testing.T, h http.Handler, method, path, body string) response {
	t.Helper()
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	noSecrets(t, method+" "+path, w.Body.Bytes())
	return response{w.Code, w.Body.Bytes()}
}

// noSecrets fails when body carries what registrymock plants in the image
// config (HF_TOKEN, AWS_SECRET_ACCESS_KEY and their values) or its bearer
// token.
func noSecrets(t *testing.T, what string, body []byte) {
	t.Helper()
	for _, secret := range []string{"hf_planted_secret", "planted-secret", "HF_TOKEN", "AWS_SECRET", "mock-token"} {
		if strings.Contains(string(body), secret) {
			t.Fatalf("%s leaks %q: %s", what, secret, body)
		}
	}
}

func (r response) into(t *testing.T, v any) {
	t.Helper()
	if err := json.Unmarshal(r.body, v); err != nil {
		t.Fatalf("decode %s: %v", r.body, err)
	}
}

func (r response) errCode(t *testing.T) string {
	t.Helper()
	var body struct {
		Error struct{ Code, Message string }
	}
	r.into(t, &body)
	return body.Error.Code
}

// checkView is an ImageCheck as a client reads it.
type checkView struct {
	OK        bool    `json:"ok"`
	Level     string  `json:"level"`
	LaunchRef *string `json:"launch_ref"`
	Pinned    *string `json:"pinned"`
	Image     *struct {
		Revision string            `json:"revision"`
		Env      map[string]string `json:"env"`
	} `json:"image"`
	Host   *hostfit.Host    `json:"host"`
	Need   *hostfit.Need    `json:"need"`
	Pull   *hostfit.Pull    `json:"pull"`
	VRAM   *hostfit.VRAMFit `json:"vram"`
	Checks []checkItem      `json:"checks"`
}

type checkItem struct {
	Level, Code, Message, Fix string
	Action                    *struct {
		Kind  string
		Value any
	}
}

func (c checkView) find(code string) (checkItem, bool) {
	for _, x := range c.Checks {
		if x.Code == code {
			return x, true
		}
	}
	return checkItem{}, false
}

func postCheck(t *testing.T, h http.Handler, body string) checkView {
	t.Helper()
	r := call(t, h, "POST", "/v1/images/check", body)
	if r.code != 200 {
		t.Fatalf("check %s: %d %s", body, r.code, r.body)
	}
	var v checkView
	r.into(t, &v)
	if v.Checks == nil {
		t.Fatalf("checks is null: %s", r.body)
	}
	return v
}

func TestCheckDefaultImage(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{})
	v := postCheck(t, h, `{"image":"`+defaultImage+`"}`)
	if !v.OK || v.Level == "block" || v.Image == nil || v.Image.Revision != registrymock.Revision {
		t.Fatalf("default image: %+v", v)
	}
	if v.LaunchRef == nil || *v.LaunchRef != "ghcr.io/geocine/aitoolkit-go:"+registrymock.Revision {
		t.Fatalf("launch_ref %v", v.LaunchRef)
	}
	if v.Pinned == nil || !strings.HasPrefix(*v.Pinned, "ghcr.io/geocine/aitoolkit-go@sha256:") {
		t.Fatalf("pinned %v", v.Pinned)
	}
	if v.Host != nil || v.Pull != nil || v.VRAM != nil || v.Need == nil || v.Need.DiskGB < hostfit.MinDiskGBUnknownModel {
		t.Fatalf("no host: host %v pull %v vram %v need %+v", v.Host, v.Pull, v.VRAM, v.Need)
	}
	if c, ok := v.find("image_unpinned"); !ok || c.Level != "info" || !strings.Contains(c.Message, registrymock.Revision) {
		t.Fatalf("image_unpinned: %+v", v.Checks)
	}
	if len(v.Image.Env) == 0 {
		t.Fatal("env is empty: the allowed CUDA entries should be kept")
	}
}

func TestCheckRequests(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{})
	for _, c := range []struct{ body, code string }{
		{`{}`, "invalid_request"},
		{`{"image":"  "}`, "invalid_request"},
		{``, "invalid_request"},
		{`{"image":`, "invalid_request"},
		{`{"image":"alpine","disk_gb":-1}`, "invalid_request"},
		{`{"image":"alpine","disk_gb":5}`, "invalid_request"}, // below what a launch accepts
		{`{"image":"alpine","disk_gb":2001}`, "invalid_request"},
		{`{"image":"alpine","offer":{"provider":"nope","offer_id":"x"}}`, "invalid_request"},
		{`{"image":"alpine","offer":{"provider":"runpod"}}`, "invalid_request"},
	} {
		if r := call(t, h, "POST", "/v1/images/check", c.body); r.code != 400 || r.errCode(t) != c.code {
			t.Errorf("%s: %d %s", c.body, r.code, r.body)
		}
	}
	// a reference that does not parse is a block, still 200
	v := postCheck(t, h, `{"image":"Not An/Image"}`)
	if c, ok := v.find("image_ref"); v.OK || v.Level != "block" || !ok || c.Level != "block" || v.LaunchRef != nil || v.Image != nil {
		t.Fatalf("bad ref: %+v", v)
	}
}

func TestCheckHost(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{})

	// gpu_name alone fills the architecture from the table
	v := postCheck(t, h, `{"image":"`+defaultImage+`","host":{"gpu_name":"Tesla V100","cuda_max":"13.0"}}`)
	if c, ok := v.find("gpu_sm"); v.OK || !ok || c.Level != "block" || v.LaunchRef != nil || v.Host == nil || v.Host.SM != 70 {
		t.Fatalf("V100: %+v", v)
	}

	v = postCheck(t, h, `{"image":"`+defaultImage+`","host":{"gpu_name":"RTX 4090","cuda_max":"12.8"}}`)
	if c, ok := v.find("host_cuda"); v.OK || !ok || c.Action == nil || c.Action.Kind != "filter_hosts" {
		t.Fatalf("old driver: %+v", v)
	}

	v = postCheck(t, h, `{"image":"`+defaultImage+`","model":{"arch":"flux"},"disk_gb":200,
		"host":{"provider":"vast","gpu_name":"RTX 4090","vram_gb":24,"cuda_max":"13.3","disk_gb":500,"inet_down_mbps":1376,"disk_bw_mbps":2842}}`)
	if !v.OK || v.LaunchRef == nil || v.Pull == nil || v.Pull.ImageMin <= 0 || v.Pull.ModelMin == nil || v.VRAM == nil || v.VRAM.Verdict != hostfit.VRAMFits {
		t.Fatalf("fitting host: %+v", v)
	}
	if v.Need == nil || v.Need.DiskGB < 80 || v.Need.ModelSource == "default" {
		t.Fatalf("need with flux: %+v", v.Need)
	}
	if _, ok := v.find("pull_eta"); !ok {
		t.Fatalf("no pull_eta: %+v", v.Checks)
	}

	// a disk below the image itself blocks
	v = postCheck(t, h, `{"image":"`+defaultImage+`","disk_gb":20,"host":{"gpu_name":"RTX 4090","cuda_max":"13.3"}}`)
	if c, ok := v.find("disk_image"); v.OK || !ok || c.Level != "block" {
		t.Fatalf("small disk: %+v", v)
	}
}

func TestCheckImageNotPublished(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{Unpublished: true})
	v := postCheck(t, h, `{"image":"`+defaultImage+`"}`)
	c, ok := v.find("image_denied")
	if v.OK || !ok || c.Level != "block" || !strings.Contains(c.Message, "isn't published yet") || v.Image != nil || v.LaunchRef != nil {
		t.Fatalf("unpublished: %+v", v)
	}
	if v.Need == nil || v.Need.ImagePeakGB != hostfit.DefaultImagePeakGB {
		t.Fatalf("the fallback should size the disk: %+v", v.Need)
	}
	// the built-in requirements still judge the host
	v = postCheck(t, h, `{"image":"`+defaultImage+`","host":{"gpu_name":"Tesla V100","cuda_max":"13.0"}}`)
	if _, ok := v.find("gpu_sm"); !ok {
		t.Fatalf("fallback did not judge the host: %+v", v.Checks)
	}
	v = postCheck(t, h, `{"image":"`+defaultImage+`","host":{"gpu_name":"RTX 4090","cuda_max":"13.3"}}`)
	for _, c := range v.Checks {
		if c.Level == "block" && c.Code != "image_denied" {
			t.Fatalf("a fitting host got %+v", c)
		}
	}
}

func TestCheckRegistryTrouble(t *testing.T) {
	// a registry 500 is a warn: ok stays true, nothing is pinned
	_, h := newEnv(t, imagetest.Options{Faults: registrymock.Faults{Status: map[string]int{
		"/v2/geocine/aitoolkit-go/manifests/latest": 500}}})
	v := postCheck(t, h, `{"image":"`+defaultImage+`"}`)
	if c, ok := v.find("image_unreachable"); !v.OK || v.Level != "warn" || !ok || c.Level != "warn" || v.Image != nil || v.LaunchRef != nil {
		t.Fatalf("registry 500: %+v", v)
	}

	// a registry outside the allowlist is never contacted: a warn
	v = postCheck(t, h, `{"image":"quay.io/someone/thing:1"}`)
	if c, ok := v.find("image_unchecked"); !v.OK || !ok || !strings.Contains(c.Message, "quay.io") {
		t.Fatalf("quay.io: %+v", v)
	}
}

func TestCheckMissingTag(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{})
	v := postCheck(t, h, `{"image":"ghcr.io/geocine/aitoolkit-go:0.14.0"}`)
	c, ok := v.find("image_not_found")
	if v.OK || !ok || !strings.Contains(c.Fix, registrymock.Revision) || c.Action == nil || c.Action.Kind != "use_image" {
		t.Fatalf("missing tag: %+v", v)
	}
}

func TestCheckOffer(t *testing.T) {
	f, h := newEnv(t, imagetest.Options{})
	f.host = &hostfit.Host{Provider: "vast", GPUName: "Tesla V100-SXM2-16GB", SM: 70, SMSource: "table", GPUVendor: "nvidia",
		CPUArch: "amd64", CUDAMax: "13.0", DiskGB: 200, VRAMGB: 16}
	v := postCheck(t, h, `{"image":"`+defaultImage+`","offer":{"provider":"vast","offer_id":"123"},"model":{"arch":"flux"},"disk_gb":120,
		"host":{"gpu_name":"RTX 4090"}}`)
	if _, ok := v.find("gpu_sm"); v.OK || !ok || v.Host == nil || v.Host.SM != 70 {
		t.Fatalf("the offer's host should win: %+v", v)
	}
	if f.kind != provision.Vast || f.offerID != "123" || f.query.Image != defaultImage || f.query.DiskGB != 120 ||
		f.query.Model == nil || f.query.Model.Arch != "flux" {
		t.Fatalf("quote got %s %s %+v", f.kind, f.offerID, f.query)
	}

	// an offer whose host is unknown runs the image-side checks only
	f.host = nil
	if v = postCheck(t, h, `{"image":"`+defaultImage+`","offer":{"provider":"runpod","offer_id":"x"}}`); !v.OK || v.Host != nil {
		t.Fatalf("hostless offer: %+v", v)
	}

	// a provider error is answered like the offers route
	f.err = provision.Errorf(provision.Vast, provision.ClassCapacity, 0, "That offer was just taken.")
	r := call(t, h, "POST", "/v1/images/check", `{"image":"`+defaultImage+`","offer":{"provider":"vast","offer_id":"123"}}`)
	if r.code != 409 || r.errCode(t) != "no_capacity" {
		t.Fatalf("taken offer: %d %s", r.code, r.body)
	}

	// a disabled provider is refused before any lookup
	f.err = nil
	t.Setenv("AITK_PROVISION_PROVIDERS", "runpod")
	if r := call(t, h, "POST", "/v1/images/check", `{"image":"alpine","offer":{"provider":"vast","offer_id":"1"}}`); r.code != 400 {
		t.Fatalf("disabled provider: %d %s", r.code, r.body)
	}
}

func TestTags(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{})
	r := call(t, h, "GET", "/v1/images/tags?image=ghcr.io/geocine/aitoolkit-go&limit=8", "")
	var out struct {
		Items []struct {
			Tag, Digest, Revision string
			Moving                bool
		}
	}
	if r.code != 200 {
		t.Fatalf("tags: %d %s", r.code, r.body)
	}
	r.into(t, &out)
	tags := map[string]bool{}
	for _, it := range out.Items {
		tags[it.Tag] = it.Moving
	}
	if moving, ok := tags["latest"]; !ok || !moving {
		t.Fatalf("latest missing or not moving: %s", r.body)
	}
	if moving, ok := tags[registrymock.Revision]; !ok || moving {
		t.Fatalf("build tag missing or moving: %s", r.body)
	}
	// the default repository when image is left out
	if r := call(t, h, "GET", "/v1/images/tags", ""); r.code != 200 || !strings.Contains(string(r.body), registrymock.Revision) {
		t.Fatalf("default repo: %d %s", r.code, r.body)
	}
	for path, want := range map[string]int{
		"/v1/images/tags?limit=0":                      400,
		"/v1/images/tags?limit=21":                     400,
		"/v1/images/tags?limit=x":                      400,
		"/v1/images/tags?image=Bad%20Ref":              400,
		"/v1/images/tags?image=quay.io/a/b":            400,
		"/v1/images/tags?image=ghcr.io/nobody/nothing": 404,
	} {
		if r := call(t, h, "GET", path, ""); r.code != want {
			t.Errorf("%s: %d %s, want %d", path, r.code, r.body, want)
		}
	}
}

func TestTagsUnpublished(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{Unpublished: true})
	r := call(t, h, "GET", "/v1/images/tags", "")
	if r.code != 404 || r.errCode(t) != "not_found" || !strings.Contains(string(r.body), "isn't published yet") {
		t.Fatalf("unpublished: %d %s", r.code, r.body)
	}
}

type recView struct {
	Choice *struct {
		Channel   string   `json:"channel"`
		Image     string   `json:"image"`
		LaunchRef *string  `json:"launch_ref"`
		Why       []string `json:"why"`
	} `json:"choice"`
	Alternatives []json.RawMessage `json:"alternatives"`
	Checks       []checkItem       `json:"checks"`
	Action       *struct{ Kind string }
}

func recommend(t *testing.T, h http.Handler, body string) recView {
	t.Helper()
	r := call(t, h, "POST", "/v1/images/recommend", body)
	if r.code != 200 {
		t.Fatalf("recommend %s: %d %s", body, r.code, r.body)
	}
	var v recView
	r.into(t, &v)
	if v.Alternatives == nil || v.Checks == nil {
		t.Fatalf("null lists: %s", r.body)
	}
	return v
}

func TestRecommend(t *testing.T) {
	f, h := newEnv(t, imagetest.Options{})

	// this server's build is published: it is the choice
	ServerBuild = func() (string, string) { return registrymock.RevisionFull[:12], registrymock.Version }
	v := recommend(t, h, ``)
	if v.Choice == nil || v.Choice.Channel != "same" || v.Choice.LaunchRef == nil || *v.Choice.LaunchRef != "ghcr.io/geocine/aitoolkit-go:"+registrymock.Revision {
		t.Fatalf("same build: %+v", v.Choice)
	}

	// a dev build has no published image: the newest one
	ServerBuild = func() (string, string) { return "dev", "" }
	if v = recommend(t, h, `{}`); v.Choice == nil || v.Choice.Channel != "latest" || v.Choice.Image != defaultImage {
		t.Fatalf("dev build: %+v", v.Choice)
	}

	// nothing runs on a V100: no choice, filter hosts
	v = recommend(t, h, `{"host":{"gpu_name":"Tesla V100","cuda_max":"13.0"}}`)
	if v.Choice != nil || v.Action == nil || v.Action.Kind != "filter_hosts" || len(v.Checks) == 0 {
		t.Fatalf("V100: %+v", v)
	}

	// through an offer
	f.host = &hostfit.Host{Provider: "runpod", GPUName: "NVIDIA GeForce RTX 4090", SM: 89, GPUVendor: "nvidia", CPUArch: "amd64",
		VRAMGB: 24, PlacementFloor: true, CUDASource: "runpod.placement"}
	if v = recommend(t, h, `{"offer":{"provider":"runpod","offer_id":"NVIDIA GeForce RTX 4090|SECURE|1"},"model":{"arch":"flux"}}`); v.Choice == nil {
		t.Fatalf("offer: %+v", v)
	}
	if f.offerID != "NVIDIA GeForce RTX 4090|SECURE|1" || f.query.Model == nil {
		t.Fatalf("quote got %q %+v", f.offerID, f.query)
	}

	// a custom image is the only candidate
	v = recommend(t, h, `{"image":"ghcr.io/geocine/aitoolkit-go:oldsha1"}`)
	if v.Choice == nil || v.Choice.Channel != "custom" || len(v.Alternatives) != 0 {
		t.Fatalf("custom: %+v", v)
	}

	// a QuickPod template's image
	f.templates = []provision.Template{{ID: "tpl_1", Name: "aitk", Image: defaultImage, Ours: true}}
	if v = recommend(t, h, `{"template_id":"tpl_1"}`); v.Choice == nil || v.Choice.Channel != "template" {
		t.Fatalf("template: %+v", v)
	}
	if r := call(t, h, "POST", "/v1/images/recommend", `{"template_id":"tpl_9"}`); r.code != 404 {
		t.Fatalf("unknown template: %d %s", r.code, r.body)
	}
	for _, body := range []string{`{"prefer":"oldest"}`, `{"disk_gb":5000}`, `{"disk_gb":5}`, `{"offer":{"provider":"x","offer_id":"1"}}`, `[`} {
		if r := call(t, h, "POST", "/v1/images/recommend", body); r.code != 400 {
			t.Errorf("%s: %d %s", body, r.code, r.body)
		}
	}
}

func TestRecommendImageNotPublished(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{Unpublished: true})
	v := recommend(t, h, `{"host":{"gpu_name":"RTX 4090","cuda_max":"13.3"}}`)
	if v.Choice != nil || len(v.Checks) == 0 || v.Checks[0].Code != "image_denied" || !strings.Contains(v.Checks[0].Message, "isn't published yet") {
		t.Fatalf("unpublished: %+v", v)
	}
	if v.Action != nil {
		t.Fatalf("an image-side block has no host action: %+v", v.Action)
	}
}

type gpuView struct {
	Name    string  `json:"name"`
	SM      int     `json:"sm"`
	Speed   float64 `json:"speed"`
	Verdict string  `json:"verdict"`
	Note    string  `json:"note"`
	Aliases []string
}

func gpus(t *testing.T, h http.Handler) map[string]gpuView {
	t.Helper()
	r := call(t, h, "GET", "/v1/hosts/gpus", "")
	if r.code != 200 {
		t.Fatalf("gpus: %d %s", r.code, r.body)
	}
	var out struct{ Items []gpuView }
	r.into(t, &out)
	m := map[string]gpuView{}
	for _, g := range out.Items {
		if g.Aliases == nil {
			t.Fatalf("%s: aliases null", g.Name)
		}
		m[g.Name] = g
	}
	return m
}

func TestHostGPUs(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{})
	m := gpus(t, h)
	if len(m) < 40 {
		t.Fatalf("only %d GPUs", len(m))
	}
	for name, want := range map[string]string{
		"RTX 4090": "ok", "RTX 5090": "ok", "H100 SXM": "ok", "Tesla V100": "block", "MI300X": "block", "GH200": "block", "Tesla T4": "warn",
	} {
		g, ok := m[name]
		if !ok {
			t.Errorf("%s missing", name)
			continue
		}
		if g.Verdict != want {
			t.Errorf("%s: verdict %s (%s), want %s", name, g.Verdict, g.Note, want)
		}
		if want != "ok" && g.Note == "" {
			t.Errorf("%s: no note", name)
		}
	}
	if g := m["RTX 4090"]; g.Note != "Runs via the sm_86 build" || g.Speed != 1 {
		t.Errorf("RTX 4090: %+v", g)
	}
	if g := m["Tesla V100"]; !strings.Contains(g.Note, "sm_70") {
		t.Errorf("V100 note %q", g.Note)
	}
}

// While the default image is not published, its built-in requirements give
// the same verdicts.
func TestHostGPUsFallback(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{Unpublished: true})
	m := gpus(t, h)
	if m["RTX 4090"].Verdict != "ok" || m["Tesla V100"].Verdict != "block" || m["Tesla T4"].Verdict != "warn" {
		t.Fatalf("fallback verdicts: %+v %+v %+v", m["RTX 4090"], m["Tesla V100"], m["Tesla T4"])
	}
}

// A caller who leaves does not decide the verdicts: the image is still read,
// and a fallback is never kept on a caller's cancellation.
func TestHostGPUsCallerGone(t *testing.T) {
	gone, cancel := context.WithCancel(context.Background())
	cancel()

	newEnv(t, imagetest.Options{})
	if items := GPUs(gone); len(items) < 40 {
		t.Fatalf("only %d GPUs", len(items))
	}
	gpusCache.mu.Lock()
	kept := gpusCache.items != nil
	gpusCache.mu.Unlock()
	if !kept {
		t.Fatal("the image was read: its verdicts should be kept")
	}

	newEnv(t, imagetest.Options{Unpublished: true})
	GPUs(gone)
	gpusCache.mu.Lock()
	kept = gpusCache.items != nil
	gpusCache.mu.Unlock()
	if kept {
		t.Fatal("a fallback answered to a caller who left was kept")
	}
}

func TestHostFit(t *testing.T) {
	_, h := newEnv(t, imagetest.Options{})
	var out struct {
		Model hostfit.ModelRef
		Tiers []struct {
			VRAMGB  float64 `json:"vram_gb"`
			Verdict string  `json:"verdict"`
			Note    string  `json:"note"`
			Recipe  string  `json:"recipe"`
		}
		Need hostfit.Need
	}
	r := call(t, h, "POST", "/v1/hosts/fit", `{"model":{"arch":"flux"}}`)
	if r.code != 200 {
		t.Fatalf("fit: %d %s", r.code, r.body)
	}
	r.into(t, &out)
	if out.Model.Arch != "flux" || len(out.Tiers) != 5 || out.Tiers[0].VRAMGB != 16 || out.Tiers[1].Verdict != hostfit.VRAMFits || out.Need.DiskGB < 80 {
		t.Fatalf("five tiers: %s", r.body)
	}

	r = call(t, h, "POST", "/v1/hosts/fit", `{"model":{"arch":"hidream"},"gpus":[{"name":"RTX 4090","count":2},{"vram_gb":48}],"host":{"gpu_name":"H100 SXM"}}`)
	if r.code != 200 {
		t.Fatalf("gpus: %d %s", r.code, r.body)
	}
	r.into(t, &out)
	if len(out.Tiers) != 3 || out.Tiers[0].VRAMGB != 24 || out.Tiers[0].Verdict != hostfit.VRAMNo || !strings.Contains(out.Tiers[0].Note, "Each job uses one GPU") ||
		out.Tiers[1].Verdict != hostfit.VRAMFits || out.Tiers[2].VRAMGB != 80 {
		t.Fatalf("per GPU: %s", r.body)
	}

	for _, body := range []string{`{}`, `{"model":{}}`, `{"model":{"arch":"flux"},"gpus":[{"name":"mystery"}]}`, `{"model":{"arch":"flux"},"gpus":[{"vram_gb":-1}]}`, ``} {
		if r := call(t, h, "POST", "/v1/hosts/fit", body); r.code != 400 {
			t.Errorf("%s: %d %s", body, r.code, r.body)
		}
	}
}

// An agent in simulate mode: offers -> images.recommend -> images.check ->
// hosts, through Auth, the real provisioning service, the RunPod mock and
// the simulated registry.
func TestCodeModeRecommendThenCheck(t *testing.T) {
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close() })
	const admin = "admin-token-for-images"
	for _, k := range []string{"AITK_PROVISION_PROVIDERS", "AITK_PROVISION_SIMULATE_REGISTRY"} {
		t.Setenv(k, "")
	}
	t.Setenv("AI_TOOLKIT_AUTH", admin)
	t.Setenv(provision.EnvKeyName(provision.RunPod), "")
	resetGPUs(t)
	oldBuild := ServerBuild
	ServerBuild = func() (string, string) { return registrymock.Revision, registrymock.Version }
	t.Cleanup(func() { ServerBuild = oldBuild })

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
await aitk.providers.setKey("runpod", { api_key: "sim-key" });
const offers = (await aitk.providers.offers("runpod", { model: "flux", limit: 5 })).items;
const offer = offers.find((o) => o.fit && o.fit.level !== "block");
const ref = { provider: "runpod", offer_id: offer.id };
const rec = await aitk.images.recommend({ offer: ref, model: { arch: "flux" } });
const check = await aitk.images.check({ image: rec.choice.image, offer: ref, model: { arch: "flux" } });
const gpus = (await aitk.hosts.gpus()).items;
const fit = await aitk.hosts.fit({ model: { arch: "flux" } });
const tags = (await aitk.images.tags({ limit: 4 })).items;
return { channel: rec.choice.channel, launch: rec.choice.launch_ref, ok: check.ok, check_launch: check.launch_ref,
  host: check.host && check.host.gpu_name, disk: check.need.disk_gb, gpus: gpus.length, tiers: fit.tiers.length, tags: tags.length };`,
		codemode.Options{Call: codemode.HandlerCaller(httpx.Auth(mux), "Bearer "+admin, ""), Timeout: 60 * time.Second})
	if res.Error != nil {
		t.Fatalf("%+v\nlogs: %s", res.Error, res.Logs)
	}
	noSecrets(t, "flow result", res.Result)
	var out struct {
		Channel, Launch, Host   string
		CheckLaunch             string `json:"check_launch"`
		OK                      bool
		Disk, GPUs, Tiers, Tags int
	}
	if err := json.Unmarshal(res.Result, &out); err != nil {
		t.Fatal(err)
	}
	want := "ghcr.io/geocine/aitoolkit-go:" + registrymock.Revision
	if out.Channel != "same" || out.Launch != want || !out.OK || out.CheckLaunch != want || out.Host == "" ||
		out.Disk < 80 || out.GPUs < 40 || out.Tiers != 5 || out.Tags == 0 {
		t.Fatalf("flow result %s", res.Result)
	}
}
