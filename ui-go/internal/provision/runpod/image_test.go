package runpod

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
)

// imageQuery is q with the filter the service derives for today's default
// image (its built-in requirements: CUDA 13.0, sm 75-120).
func imageQuery(q provision.OfferQuery) provision.OfferQuery {
	img, _ := hostfit.FallbackImage(imagecheck.DefaultRepo + ":latest")
	f := hostfit.FilterFor(img.Requires, hostfit.NeedFor(img, q.Model, nil), q.Model, hostfit.Policy{Fast: q.Fast, ShowAll: q.ShowAll})
	q.Filter, q.ImageInfo = &f, img
	return q
}

func TestOffersImageHostAndFit(t *testing.T) {
	a, _, _ := setup(t, mockkit.Options{})
	offers, err := a.Offers(ctx, key, imageQuery(provision.OfferQuery{}))
	if err != nil {
		t.Fatal(err)
	}
	byGPU := map[string]provision.Offer{}
	for _, o := range offers {
		if o.Host == nil || o.Fit == nil || !o.Host.PlacementFloor || o.Host.SM == 0 || o.Host.InetDownMbps != 0 {
			t.Fatalf("offer %s host %+v fit %+v", o.GPU, o.Host, o.Fit)
		}
		byGPU[o.GPU] = o
	}
	// 4090: 13.2 free, placed at the floor (an info); A6000: 13.0 listed but none free (a warn)
	if o := byGPU["RTX 4090"]; o.Host.CUDAMax != "13.2" || o.Fit.Level == "block" || o.Fit.Level == "warn" {
		t.Fatalf("4090 host %+v fit %+v", o.Host, o.Fit)
	}
	if o := byGPU["RTX A6000"]; o.Fit.Level != "warn" || !slices.Contains(o.Fit.Codes, "runpod_no_cuda_capacity") {
		t.Fatalf("A6000 fit %+v", o.Fit)
	}
	// no bandwidth on RunPod v2: no pull time, and Fast drops nothing (D-15)
	if byGPU["RTX 4090"].Fit.PullMin != nil {
		t.Fatal("pull time without a reported bandwidth")
	}
	fast, err := a.Offers(ctx, key, imageQuery(provision.OfferQuery{Fast: true}))
	if err != nil || len(fast) != len(offers) {
		t.Fatalf("fast: %d offers, want %d (%v)", len(fast), len(offers), err)
	}
}

func TestOffersImageVRAMFloor(t *testing.T) {
	a, _, _ := setup(t, mockkit.Options{})
	// FLUX.2 needs a 48 GB card: the 24 GB types are dropped, and badged under show_all
	offers, err := a.Offers(ctx, key, imageQuery(provision.OfferQuery{Model: &hostfit.ModelRef{Arch: "flux2"}}))
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range offers {
		if o.VRAMGB < 48 {
			t.Fatalf("%s (%d GB) passed the FLUX.2 floor", o.GPU, o.VRAMGB)
		}
	}
	all, err := a.Offers(ctx, key, imageQuery(provision.OfferQuery{Model: &hostfit.ModelRef{Arch: "flux2"}, ShowAll: true}))
	if err != nil || len(all) <= len(offers) {
		t.Fatalf("show_all %d <= %d (%v)", len(all), len(offers), err)
	}
	for _, o := range all {
		if o.VRAMGB == 24 && (o.Fit.VRAM != hostfit.VRAMNo || !slices.Contains(o.Fit.Codes, "vram_no")) {
			t.Fatalf("%s fit %+v", o.GPU, o.Fit)
		}
	}
}

func TestQuoteImageHost(t *testing.T) {
	a, _, _ := setup(t, mockkit.Options{})
	o, err := a.Quote(ctx, key, OfferID("NVIDIA GeForce RTX 4090", "SECURE", 1), imageQuery(provision.OfferQuery{DiskGB: 100}))
	if err != nil {
		t.Fatal(err)
	}
	if o.Host == nil || o.Host.SM != 89 || !o.Host.PlacementFloor || o.Fit == nil {
		t.Fatalf("quote host %+v fit %+v", o.Host, o.Fit)
	}
}

func TestLaunchSendsMinCuda(t *testing.T) {
	a, m, _ := setup(t, mockkit.Options{})
	spec := provision.LaunchSpec{Label: "aitk-cuda", OfferID: OfferID("NVIDIA GeForce RTX 4090", "SECURE", 1),
		Image: "ghcr.io/geocine/aitoolkit-go:f53a407", DiskGB: 100}
	if _, err := a.Launch(provision.WithLaunchCUDA(ctx, "13.0"), key, spec); err != nil {
		t.Fatal(err)
	}
	var body struct {
		Image string                     `json:"image"`
		GPU   map[string]json.RawMessage `json:"gpu"`
	}
	if err := json.Unmarshal(lastReq(t, m, "POST", "/v2/pods").Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Image != spec.Image || string(body.GPU["minCudaVersion"]) != `"13.0"` {
		t.Fatalf("create body image %q gpu %v", body.Image, body.GPU)
	}
	if _, has := body.GPU["allowedCudaVersions"]; has {
		t.Fatal("allowedCudaVersions sent with minCudaVersion (RunPod answers 400)")
	}
	if p := m.Instances()[0]; p.MinCudaVersion != "13.0" || p.Image != spec.Image {
		t.Fatalf("mock saw %+v", p)
	}
	// without a floor the key is left out
	spec.Label = "aitk-nofloor"
	if _, err := a.Launch(ctx, key, spec); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(lastReq(t, m, "POST", "/v2/pods").Body), "minCudaVersion") {
		t.Fatal("minCudaVersion sent without a floor")
	}
}

func TestMockPullFail(t *testing.T) {
	a, m, clock := setup(t, mockkit.Options{Delays: mockkit.Delays{Provision: 1, Start: 1}})
	m.PullFailText = "denied: requested access to the resource is denied"
	ref := launch(t, a, "aitk-pull")
	clock.Advance(10)
	in, err := a.Get(ctx, key, ref)
	if err != nil {
		t.Fatal(err)
	}
	if in.Status != provision.StatusFailed {
		t.Fatalf("status %s (%s)", in.Status, in.ProviderStatus)
	}
	logs, err := a.Logs(ctx, key, ref.ID, 50)
	if err != nil || !strings.Contains(logs, "requested access to the resource is denied") {
		t.Fatalf("logs %q %v", logs, err)
	}
	if f, ok := imagecheck.ClassifyFailure(logs); !ok || f.Code != "image_denied" {
		t.Fatalf("classified %+v %v", f, ok)
	}
}
