package provision

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/provision/imagetest"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Hosts the stub quotes carry.
var (
	hostV100 = hostfit.Host{Provider: "vast", GPUName: "Tesla V100", GPUCount: 1, VRAMGB: 16, SM: 70, GPUVendor: "nvidia",
		CPUArch: "amd64", CUDAMax: "13.0", DiskGB: 500, InetDownMbps: 900, DiskBWMBps: 2000}
	host4090 = hostfit.Host{Provider: "vast", GPUName: "RTX 4090", GPUCount: 1, VRAMGB: 24, SM: 89, GPUVendor: "nvidia",
		CPUArch: "amd64", CUDAMax: "13.2", DiskGB: 500, InetDownMbps: 900, DiskBWMBps: 2000}
)

// withHost makes k's stub quote (and offer) carry h.
func (e *testEnv) withHost(k Kind, h hostfit.Host, maxDisk *int) {
	c := e.clouds[k]
	c.mu.Lock()
	defer c.mu.Unlock()
	hc := h
	c.quote.Host, c.quote.MaxDiskGB = &hc, maxDisk
	c.offers = []Offer{c.quote}
}

// blocked asserts err is the 422 image_blocked with code among its checks.
func blocked(t *testing.T, err error, code string) *ImageBlockedError {
	t.Helper()
	var b *ImageBlockedError
	if !errors.As(err, &b) {
		t.Fatalf("err %v (%T), want ImageBlockedError", err, err)
	}
	c, msg, details := b.V1Details()
	checks, _ := details.(map[string]any)["checks"].([]imagecheck.Check)
	if c != v1api.CodeImageBlocked || msg == "" || !slices.ContainsFunc(checks, func(x imagecheck.Check) bool {
		return x.Code == code && x.Level == imagecheck.LevelBlock
	}) {
		t.Fatalf("details %s %q %+v, want a %s block", c, msg, checks, code)
	}
	return b
}

func TestLaunchImageBlocksHostWithoutCreate(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "vast-key-0001")
	e.withHost(Vast, hostV100, nil)
	_, _, err := e.s.Launch(e.ctx, e.req(Vast), "")
	blocked(t, err, "gpu_sm")
	if n := e.clouds[Vast].n("Launch"); n != 0 {
		t.Fatalf("%d create calls on a blocked launch", n)
	}
	if ms, _ := e.s.List(e.ctx, false); len(ms) != 0 {
		t.Fatalf("a blocked launch left %d rows", len(ms))
	}
	// skip_image_check never bypasses a host-side block
	req := e.req(Vast)
	req.SkipImageCheck = true
	_, _, err = e.s.Launch(e.ctx, req, "")
	if b := blocked(t, err, "gpu_sm"); !strings.Contains(b.Message, "skip_image_check") {
		t.Fatalf("message %q", b.Message)
	}
	if n := e.clouds[Vast].n("Launch"); n != 0 {
		t.Fatalf("%d create calls", n)
	}
}

func TestLaunchImagePinsBuild(t *testing.T) {
	e := newEnv(t)
	e.withHost(Vast, host4090, nil)
	m := e.launch(Vast, nil)
	spec := e.clouds[Vast].launched[0]
	if spec.Image != pinnedImage {
		t.Fatalf("adapter got image %q, want %q", spec.Image, pinnedImage)
	}
	digest := e.images.Registry.Digest(registrymock.GHCR, registrymock.DefaultRepo, "latest")
	if m.Image != DefaultImage || m.ImageLaunchRef == nil || *m.ImageLaunchRef != pinnedImage ||
		m.ImageDigest == nil || *m.ImageDigest != digest || len(m.ImageWarnings) != 0 {
		t.Fatalf("machine image %q ref %v digest %v (want %s) warnings %v", m.Image, m.ImageLaunchRef, m.ImageDigest, digest, m.ImageWarnings)
	}
	r := e.row(m.ID)
	snap := snapshotOf(r)
	if snap.LaunchRef != pinnedImage || snap.Digest != digest || snap.Revision != registrymock.Revision || snap.Level == "" ||
		!slices.ContainsFunc(snap.Checks, func(c imagecheck.Check) bool { return c.Code == "image_unpinned" }) {
		t.Fatalf("snapshot %s", r.Spec.ImageCheck)
	}
	// 6.57 GB at 900 Mbps (eta 0.5) plus unpacking at 150 MB/s: about 3.5 min
	if r.Spec.PullS < 150 || r.Spec.PullS > 300 {
		t.Fatalf("pull_s %v", r.Spec.PullS)
	}
	if r.Spec.DiskGB != hostfit.MinDiskGBUnknownModel {
		t.Fatalf("disk %d, want the need without a model", r.Spec.DiskGB)
	}
}

func TestLaunchImageCUDAFloorReachesAdapter(t *testing.T) {
	e := newEnv(t)
	c := &ctxCloud{stubCloud: newStub(RunPod)}
	t.Cleanup(SetForTest(c))
	e.launch(RunPod, nil)
	if c.cudaMin != "13.0" {
		t.Fatalf("LaunchCUDAMin %q, want 13.0", c.cudaMin)
	}
}

type ctxCloud struct {
	*stubCloud
	cudaMin string
}

func (c *ctxCloud) Launch(ctx context.Context, key string, s LaunchSpec) (Ref, error) {
	c.cudaMin = LaunchCUDAMin(ctx)
	return c.stubCloud.Launch(ctx, key, s)
}

func TestLaunchImageSkip(t *testing.T) {
	e := newEnv(t)
	e.withHost(Vast, host4090, nil)
	// a custom image that isn't the Go server: blocked, unless skipped
	req := e.req(Vast)
	req.Image = "ostris/aitoolkit:latest"
	e.setKey(Vast, "vast-key-0001")
	_, _, err := e.s.Launch(e.ctx, req, "")
	blocked(t, err, "image_not_go")
	m := e.launch(Vast, func(r *LaunchRequest) { r.Image, r.SkipImageCheck = "ostris/aitoolkit:latest", true })
	if !slices.Contains(m.ImageWarnings, warnImageSkipped) || !snapshotOf(e.row(m.ID)).Skipped {
		t.Fatalf("warnings %v", m.ImageWarnings)
	}
	// a registry that can't be read is a warn: the launch goes ahead, and a skip is recorded
	m = e.launch(Vast, func(r *LaunchRequest) { r.Image, r.SkipImageCheck, r.Name = "quay.io/acme/trainer:1", true, "quay" })
	if !slices.Contains(m.ImageWarnings, warnImageSkipped) || !slices.Contains(m.ImageWarnings, "Images on quay.io can't be checked.") {
		t.Fatalf("warnings %v", m.ImageWarnings)
	}
	if got := e.clouds[Vast].launched[len(e.clouds[Vast].launched)-1].Image; got != "quay.io/acme/trainer:1" {
		t.Fatalf("adapter got %q", got)
	}
}

func TestLaunchImageDiskDefault(t *testing.T) {
	e := newEnv(t)
	e.withHost(Vast, host4090, ptr(60))
	m := e.launch(Vast, nil)
	// the need (100 GB) does not fit the 60 GB host: ask for what it has, with a warning
	if m.DiskGB != 60 || e.clouds[Vast].launched[0].DiskGB != 60 {
		t.Fatalf("disk %d / %d", m.DiskGB, e.clouds[Vast].launched[0].DiskGB)
	}
	if !slices.ContainsFunc(m.ImageWarnings, func(w string) bool { return strings.Contains(w, "60 GB fits the image") }) {
		t.Fatalf("warnings %v", m.ImageWarnings)
	}
	// an explicit disk above the host's is still refused
	e.setKey(Vast, "vast-key-0001")
	req := e.req(Vast)
	req.DiskGB = 80
	if _, _, err := e.s.Launch(e.ctx, req, ""); codeOf(err) != v1api.CodeInvalidRequest {
		t.Fatalf("err %v", err)
	}
	// a host too small for the image blocks
	e.withHost(Vast, host4090, ptr(20))
	req.DiskGB = 20
	_, _, err := e.s.Launch(e.ctx, req, "")
	blocked(t, err, "disk_image")
}

// diskCapCloud refuses a disk above the offer's maximum, as QuickPod's
// Quote does (ClassInvalid), instead of quoting it.
type diskCapCloud struct{ *stubCloud }

func (c *diskCapCloud) Quote(ctx context.Context, key, offerID string, q OfferQuery) (Offer, error) {
	o, err := c.stubCloud.Quote(ctx, key, offerID, q)
	if err == nil && o.MaxDiskGB != nil && q.DiskGB > *o.MaxDiskGB {
		return Offer{}, Errorf(c.kind, ClassInvalid, 0, "%d GB of disk is more than this offer's %d GB", q.DiskGB, *o.MaxDiskGB)
	}
	return o, err
}

// On a provider that refuses a disk above the host's, an omitted disk_gb is
// still capped to the host with the disk_model warning (IMAGE-PLAN §4.5 step 7).
func TestLaunchImageDiskDefaultRefusingQuote(t *testing.T) {
	e := newEnv(t)
	c := &diskCapCloud{e.clouds[QuickPod]}
	t.Cleanup(SetForTest(c))
	pinned := imagecheck.DefaultRepo + ":" + registrymock.Revision
	c.templates = []Template{{ID: "tpl-aitk", Name: "aitk", Image: pinned, Ours: true, PublishesPort: true}}
	h := host4090
	h.Provider = "quickpod"
	e.withHost(QuickPod, h, ptr(80))
	m := e.launch(QuickPod, nil)
	if m.DiskGB != 80 || c.launched[0].DiskGB != 80 {
		t.Fatalf("disk %d / %d, want the host's 80 GB", m.DiskGB, c.launched[0].DiskGB)
	}
	if !slices.ContainsFunc(m.ImageWarnings, func(w string) bool { return strings.Contains(w, "80 GB fits the image") }) {
		t.Fatalf("warnings %v", m.ImageWarnings)
	}
	// an explicit disk above the host's is the provider's refusal, as it was
	req := e.req(QuickPod)
	req.DiskGB, req.Name = 90, "explicit"
	if _, _, err := e.s.Launch(e.ctx, req, ""); ClassOf(err) != ClassInvalid || !strings.Contains(err.Error(), "90 GB") {
		t.Fatalf("err %v", err)
	}
	// a host too small for the image blocks (past the quote cache)
	e.withHost(QuickPod, h, ptr(20))
	e.clock.Advance(cacheTTL)
	req.DiskGB = 0
	_, _, err := e.s.Launch(e.ctx, req, "")
	blocked(t, err, "disk_image")
}

// A registry that can't be read lets the launch go ahead, with a warning on
// the machine as well as among its image warnings (IMAGE-PLAN §4.5 step 4).
func TestLaunchImageRegistryUnreachable(t *testing.T) {
	e := newEnv(t)
	imagetest.Use(t, imagetest.Options{Faults: registrymock.Faults{Status: map[string]int{
		"/v2/" + registrymock.DefaultRepo + "/manifests/latest": http.StatusBadGateway}}})
	e.withHost(Vast, host4090, nil)
	m := e.launch(Vast, nil)
	if !slices.Contains(m.ImageWarnings, "Couldn't check the image right now.") {
		t.Fatalf("image warnings %v", m.ImageWarnings)
	}
	if !slices.ContainsFunc(m.Warnings, func(w string) bool { return strings.Contains(w, "Couldn't check the image before launch") }) {
		t.Fatalf("machine warnings %v", m.Warnings)
	}
	if snap := snapshotOf(e.row(m.ID)); !snap.Fallback || m.ImageDigest != nil {
		t.Fatalf("snapshot %+v digest %v", snap, m.ImageDigest)
	}
}

// The quote cache holds the host; each caller's image and model are judged
// on it, so a second caller never reads the first one's fit.
func TestQuoteFitPerCaller(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "vast-key-0001")
	e.withHost(Vast, host4090, nil)
	first, err := e.s.Quote(e.ctx, Vast, "offer-1", OfferQuery{})
	if err != nil || first.Fit == nil || first.Fit.VRAM != hostfit.VRAMUnknown {
		t.Fatalf("default quote %v fit %+v", err, first.Fit)
	}
	second, err := e.s.Quote(e.ctx, Vast, "offer-1", OfferQuery{Model: &hostfit.ModelRef{Arch: "flux2"}})
	if err != nil || second.Fit == nil || second.Fit.VRAM == hostfit.VRAMUnknown {
		t.Fatalf("model quote %v fit %+v", err, second.Fit)
	}
	// a custom image that can't be read is not judged: no fit, the host kept
	custom, err := e.s.Quote(e.ctx, Vast, "offer-1", OfferQuery{Image: "ghcr.io/acme/nothere:1"})
	if err != nil || custom.Fit != nil || custom.Host == nil {
		t.Fatalf("custom image quote %v fit %+v", err, custom.Fit)
	}
	if first.Fit.VRAM != hostfit.VRAMUnknown {
		t.Fatalf("the first caller's fit changed: %+v", first.Fit)
	}
	if n := e.clouds[Vast].n("Quote"); n != 1 {
		t.Fatalf("%d provider quotes, want 1 (cached)", n)
	}
}

func TestLaunchImageUnpublished(t *testing.T) {
	e := newEnv(t)
	imagetest.Use(t, imagetest.Options{Unpublished: true})
	e.withHost(Vast, host4090, nil)
	e.setKey(Vast, "vast-key-0001")
	_, _, err := e.s.Launch(e.ctx, e.req(Vast), "")
	b := blocked(t, err, "image_denied")
	if !strings.Contains(b.Message, "isn't published yet") {
		t.Fatalf("message %q", b.Message)
	}
	if n := e.clouds[Vast].n("Launch"); n != 0 {
		t.Fatalf("%d create calls", n)
	}
}

func TestOffersGetImageFilter(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "vast-key-0001")
	if _, err := e.s.Offers(e.ctx, Vast, OfferQuery{Model: &hostfit.ModelRef{Arch: "flux"}, Fast: true}); err != nil {
		t.Fatal(err)
	}
	q := e.clouds[Vast].queries[0]
	if q.Filter == nil || q.ImageInfo == nil || q.ImageInfo.Requires.Source == "fallback" ||
		q.Filter.CUDAMin != (hostfit.Version{Major: 13}) || q.Filter.MinDiskGB != 80 || q.Filter.MinVRAMGB == 0 ||
		q.Filter.MinInetDownMbps != hostfit.FastInetDownMbps || slices.Contains(q.Filter.SMAllowed, 75) {
		t.Fatalf("filter %+v image %v", q.Filter, q.ImageInfo)
	}
}

func TestOffersImageFallback(t *testing.T) {
	e := newEnv(t)
	imagetest.Use(t, imagetest.Options{Unpublished: true})
	e.setKey(Vast, "vast-key-0001")
	if _, err := e.s.Offers(e.ctx, Vast, OfferQuery{}); err != nil {
		t.Fatal(err)
	}
	// the default image still filters hosts, from its built-in requirements
	q := e.clouds[Vast].queries[0]
	if q.Filter == nil || q.ImageInfo == nil || q.ImageInfo.Requires.Source != "fallback" || q.Filter.CUDAMin.Major != 13 {
		t.Fatalf("filter %+v image %+v", q.Filter, q.ImageInfo)
	}
	// another image that can't be read is not judged
	if _, err := e.s.Offers(e.ctx, Vast, OfferQuery{Image: "ghcr.io/acme/nothere:1"}); err != nil {
		t.Fatal(err)
	}
	if q := e.clouds[Vast].queries[1]; q.Filter != nil || q.ImageInfo != nil {
		t.Fatalf("filter %+v for an unreadable custom image", q.Filter)
	}
}

func TestDecorateOffer(t *testing.T) {
	q := withImageFilter(context.Background(), OfferQuery{})
	var o Offer
	DecorateOffer(&o, hostfit.Host{GPUName: "RTX 4090"}, OfferQuery{})
	if o.Host == nil || o.Host.GPUName != "RTX 4090" || o.Fit != nil {
		t.Fatalf("without a filter: host %+v fit %+v", o.Host, o.Fit)
	}
	DecorateOffer(&o, hostV100, q)
	if o.Fit == nil || o.Fit.Level != imagecheck.LevelBlock || !slices.Contains(o.Fit.Codes, "gpu_sm") || o.Fit.VRAM != hostfit.VRAMUnknown {
		t.Fatalf("V100 fit %+v", o.Fit)
	}
	if KeepOffer(hostV100, q) || !KeepOffer(host4090, q) || !KeepOffer(hostV100, OfferQuery{}) {
		t.Fatal("KeepOffer")
	}
	q.Filter.ShowAll = true
	if !KeepOffer(hostV100, q) {
		t.Fatal("show_all keeps the V100, badged")
	}
}

func TestPriceRoseDetails(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "vast-key-0001")
	e.clouds[Vast].quote.PricePerHour = 0.62
	_, _, err := e.s.Launch(e.ctx, e.req(Vast), "")
	var d interface {
		V1Details() (string, string, any)
	}
	if !errors.As(err, &d) || ClassOf(err) != ClassConflict {
		t.Fatalf("err %v (%T)", err, err)
	}
	code, msg, details := d.V1Details()
	if code != v1api.CodeConflict || !strings.Contains(msg, "$0.62/h") || details.(map[string]any)["price_per_hour"] != 0.62 {
		t.Fatalf("%s %q %v", code, msg, details)
	}
}

// ---- after launch ----

// The machine sends no headers until its CUDA check is done (up to 90 s), so
// the client's header timeout must be the call's, not netguard's 8 s.
func TestCUDAClientWaitsForTheCheck(t *testing.T) {
	for _, sim := range []bool{false, true} {
		tr, ok := cudaClient(sim).Transport.(*http.Transport)
		if !ok || tr.ResponseHeaderTimeout < cudaCallTimeout {
			t.Fatalf("simulated=%v: header timeout %v, want at least %v", sim, tr.ResponseHeaderTimeout, cudaCallTimeout)
		}
	}
	// the real call path, against a machine that answers once its check ends
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/system/cuda" || r.Header.Get("Authorization") != "Bearer tok-1" {
			http.NotFound(w, r)
			return
		}
		time.Sleep(300 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":false,"error":"CUDA error: no kernel image is available for execution on the device","failure":"no_kernel_image"}`))
	}))
	defer srv.Close()
	res, err := cudaCheck(context.Background(), srv.URL, "tok-1", true)
	if err != nil || !res.Found || res.OK || res.Failure == nil || *res.Failure != "no_kernel_image" {
		t.Fatalf("cuda check %+v %v", res, err)
	}
	// the strict client refuses loopback: a real machine is never this server's neighbour
	if _, err := cudaCheck(context.Background(), srv.URL, "tok-1", false); err == nil {
		t.Fatal("the strict client reached loopback")
	}
}

// readyMachine launches on k and refreshes it until ready.
func (e *testEnv) readyMachine(k Kind) Machine {
	e.t.Helper()
	m := e.launch(k, nil)
	e.clouds[k].running(m.ProviderID)
	if m = e.refresh(m.ID); m.Status != StatusReady {
		e.t.Fatalf("status %s problem %v", m.Status, problemOf(m))
	}
	return m
}

func TestAfterReadyCUDAFailed(t *testing.T) {
	e := newEnv(t)
	msg, failure := "AcceleratorError: CUDA error: no kernel image is available for execution on the device", "no_kernel_image"
	e.cuda.answer(cudaResult{Found: true, OK: false, Error: &msg, Failure: &failure}, nil)
	m := e.readyMachine(RunPod)
	if problemOf(m) != ProblemCUDAFailed || !strings.Contains(m.Detail, "isn't supported by the image's PyTorch build") {
		t.Fatalf("problem %v detail %q", problemOf(m), m.Detail)
	}
	// it stays across refreshes, and the check is not asked again
	m = e.refresh(m.ID)
	if problemOf(m) != ProblemCUDAFailed || m.Status != StatusReady || e.cuda.calls != 1 {
		t.Fatalf("after refresh: %s %v, %d checks", m.Status, problemOf(m), e.cuda.calls)
	}
}

func TestAfterReadyCUDAOKAndOldImage(t *testing.T) {
	e := newEnv(t)
	e.cuda.answer(cudaResult{}, nil) // an image without the route: skipped silently
	m := e.readyMachine(RunPod)
	if m.Problem != nil || e.cuda.calls != 1 {
		t.Fatalf("problem %v, %d checks", problemOf(m), e.cuda.calls)
	}
	e.refresh(m.ID)
	if e.cuda.calls != 1 {
		t.Fatalf("%d checks after the route was missing", e.cuda.calls)
	}
}

func TestAfterReadyCUDARetriesUnanswered(t *testing.T) {
	e := newEnv(t)
	e.cuda.answer(cudaResult{}, errors.New("timeout"))
	m := e.readyMachine(RunPod)
	for range cudaTries + 2 {
		m = e.refresh(m.ID)
	}
	if m.Problem != nil || e.cuda.calls != cudaTries {
		t.Fatalf("problem %v, %d checks (want %d)", problemOf(m), e.cuda.calls, cudaTries)
	}
}

func TestAfterReadyBuildMismatch(t *testing.T) {
	e := newEnv(t)
	e.prober.version = "0a1b2c3"
	m := e.readyMachine(RunPod)
	want := buildWarning("0a1b2c3", registrymock.Revision)
	if !slices.Contains(m.Warnings, want) {
		t.Fatalf("warnings %v", m.Warnings)
	}
	n := 0
	for _, w := range e.refresh(m.ID).Warnings {
		if w == want {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("warning repeated: %v", m.Warnings)
	}
	// the same build (full sha) says nothing
	e2 := newEnv(t)
	e2.prober.version = registrymock.RevisionFull
	if m := e2.readyMachine(RunPod); len(m.Warnings) != 0 {
		t.Fatalf("warnings %v", m.Warnings)
	}
}

func TestPullWatchdog(t *testing.T) {
	e := newEnv(t)
	e.withHost(Vast, hostfit.Host{Provider: "vast", GPUName: "RTX 4090", SM: 89, GPUVendor: "nvidia", CPUArch: "amd64",
		CUDAMax: "13.2", DiskGB: 500, InetDownMbps: 40}, nil)
	m := e.launch(Vast, nil)
	pullS := e.row(m.ID).Spec.PullS
	after := pullSlowAfter(pullS)
	if after <= pullSlowMin || after != time.Duration(pullS*3*float64(time.Second)) {
		t.Fatalf("pull %v s, watchdog after %v", pullS, after)
	}
	e.clouds[Vast].set(m.ProviderID, func(in *Instance) { in.Status = StatusStarting })
	e.clock.Advance(after - time.Minute)
	if m = e.refresh(m.ID); m.Problem != nil {
		t.Fatalf("too early: %v", problemOf(m))
	}
	e.clock.Advance(2 * time.Minute)
	m = e.refresh(m.ID)
	if problemOf(m) != ProblemPullSlow || !strings.Contains(m.Detail, "doesn't bill while it's loading") || m.Status != StatusStarting {
		t.Fatalf("%s %v %q", m.Status, problemOf(m), m.Detail)
	}
	// it stays while pulling (not_ready_timeout does not replace it), then clears once running
	e.clock.Advance(30 * time.Minute)
	if m = e.refresh(m.ID); problemOf(m) != ProblemPullSlow {
		t.Fatalf("while pulling: %v", problemOf(m))
	}
	e.clouds[Vast].running(m.ProviderID)
	if m = e.refresh(m.ID); m.Status != StatusReady || m.Problem != nil {
		t.Fatalf("after the pull: %s %v", m.Status, problemOf(m))
	}
}

func TestPullWatchdogFloor(t *testing.T) {
	if got := pullSlowAfter(60); got != pullSlowMin {
		t.Fatalf("a 1 min pull is watched after %v", got)
	}
	if !strings.Contains(pullSlowDetail(Vast), "Vast.ai") || strings.Contains(pullSlowDetail(RunPod), "bill") {
		t.Fatal("billing copy is Vast's only")
	}
}

func TestImagePullFailedFromLogs(t *testing.T) {
	e := newEnv(t)
	m := e.launch(RunPod, nil)
	c := e.clouds[RunPod]
	c.logs = "Pulling image " + pinnedImage + "\nError response from daemon: pull access denied"
	c.set(m.ProviderID, func(in *Instance) {
		in.Status, in.ProviderStatus, in.Detail = StatusFailed, "ERROR", "RunPod reports the container in an unrecoverable error state."
	})
	m = e.refresh(m.ID)
	if problemOf(m) != ProblemImagePullFailed || !strings.Contains(m.Detail, "isn't published yet") || !strings.Contains(m.Detail, "build_and_push.sh") {
		t.Fatalf("%v %q", problemOf(m), m.Detail)
	}
	logsRead := c.n("Logs")
	if m = e.refresh(m.ID); problemOf(m) != ProblemImagePullFailed || c.n("Logs") != logsRead {
		t.Fatalf("after refresh %v, logs read %d more times", problemOf(m), c.n("Logs")-logsRead)
	}
}

func TestFailureDetail(t *testing.T) {
	for _, tc := range []struct {
		text, ref string
		problem   Problem
		has       string
	}{
		{"manifest unknown", pinnedImage, ProblemImagePullFailed, "couldn't find the image tag"},
		{"pull access denied", "ghcr.io/acme/x:1", ProblemImagePullFailed, "private or missing"},
		{"pull access denied", pinnedImage, ProblemImagePullFailed, "isn't published yet"},
		{"write /var/lib: no space left on device", pinnedImage, ProblemImagePullFailed, "Use at least 150 GB"},
		{"nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0", pinnedImage, ProblemCUDAFailed, "driver is too old"},
	} {
		p, d, ok := failureDetail(tc.text, tc.ref, 100)
		if !ok || p != tc.problem || !strings.Contains(d, tc.has) {
			t.Errorf("%q: %v %v %q", tc.text, ok, p, d)
		}
	}
	if _, _, ok := failureDetail("Container exited with an error", pinnedImage, 100); ok {
		t.Error("an unknown failure is classified")
	}
}

func TestSnapshotRoundTrip(t *testing.T) {
	e := newEnv(t)
	m := e.launch(RunPod, nil)
	var raw map[string]any
	if err := json.Unmarshal(e.row(m.ID).Spec.ImageCheck, &raw); err != nil || raw["launch_ref"] != pinnedImage || raw["image"] != DefaultImage {
		t.Fatalf("snapshot %v %v", raw, err)
	}
}

// QuickPod runs the template's image: it is what gets checked, stored and
// left as it is (createpod has no image field).
func TestLaunchImageQuickPodTemplate(t *testing.T) {
	e := newEnv(t)
	pinned := imagecheck.DefaultRepo + ":" + registrymock.Revision
	// a template whose image can't run here is refused before the create
	e.clouds[QuickPod].templates = []Template{{ID: "tpl-aitk", Name: "aitk", Image: "ostris/aitoolkit:latest"}}
	e.setKey(QuickPod, "qp-key-0001")
	_, _, err := e.s.Launch(e.ctx, e.req(QuickPod), "")
	blocked(t, err, "image_not_go")

	e.clouds[QuickPod].templates = []Template{{ID: "tpl-aitk", Name: "aitk", Image: pinned, Ours: true, PublishesPort: true}}
	m := e.launch(QuickPod, nil)
	if m.Image != pinned || m.ImageLaunchRef == nil || *m.ImageLaunchRef != pinned || m.ImageDigest == nil {
		t.Fatalf("image %q ref %v digest %v", m.Image, m.ImageLaunchRef, m.ImageDigest)
	}
	// a template on a moving tag stays on it
	e.clouds[QuickPod].templates[0].Image = DefaultImage
	m = e.launch(QuickPod, func(r *LaunchRequest) { r.Name = "second" })
	if *m.ImageLaunchRef != DefaultImage || snapshotOf(e.row(m.ID)).Revision != registrymock.Revision {
		t.Fatalf("ref %v", *m.ImageLaunchRef)
	}
}
