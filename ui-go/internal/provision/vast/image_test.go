package vast

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/vast/vastmock"
)

// imageQuery is q with the filter the service derives for today's default
// image (its built-in requirements: CUDA 13.0, sm 75-120).
func imageQuery(q provision.OfferQuery) provision.OfferQuery {
	img, _ := hostfit.FallbackImage(imagecheck.DefaultRepo + ":latest")
	f := hostfit.FilterFor(img.Requires, hostfit.NeedFor(img, q.Model, nil), q.Model, hostfit.Policy{Fast: q.Fast, ShowAll: q.ShowAll})
	q.Filter, q.ImageInfo = &f, img
	return q
}

func liveEnv(t *testing.T) env {
	e := setup(t, mockkit.Options{})
	e.m.AddOffers(vastmock.LiveOffers()...)
	return e
}

// Synthetic rows of the live fixture.
const (
	rowNullDriver = "91000001"
	rowV100       = "91000003"
	rowJunkCC     = "91000004"
	rowArm64      = "91000005"
	rowSmallDisk  = "91000006"
	rowSlow       = "91000007"
)

func TestOffersImageFilterServerSide(t *testing.T) {
	e := liveEnv(t)
	offers, err := e.c.Offers(context.Background(), testKey, imageQuery(provision.OfferQuery{Limit: 200}))
	if err != nil {
		t.Fatal(err)
	}
	body := decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)
	for _, k := range []string{"driver_vers", "driver_version"} {
		if _, has := body[k]; has {
			t.Fatalf("the search filters on %s", k)
		}
	}
	got, _ := json.Marshal(map[string]any{"cuda": body["cuda_max_good"], "cc": body["compute_cap"], "arch": body["cpu_arch"],
		"gpu": body["gpu_arch"], "disk": body["disk_space"]})
	want := `{"arch":{"eq":"amd64"},"cc":{"in":[800,860,890,900,1000,1030,1200]},"cuda":{"gte":13},"disk":{"gte":100},"gpu":{"eq":"nvidia"}}`
	if string(got) != want {
		t.Fatalf("image terms %s\nwant %s", got, want)
	}
	if len(offers) == 0 {
		t.Fatal("no offers")
	}
	for _, o := range offers {
		if o.Host == nil || o.Fit == nil || o.Fit.Level == "block" || o.Host.SM == 0 {
			t.Fatalf("offer %s host %+v fit %+v", o.ID, o.Host, o.Fit)
		}
		if v, _ := hostfit.ParseVersion(o.Host.CUDAMax); v.Less(hostfit.Version{Major: 13}) {
			t.Fatalf("offer %s on CUDA %s", o.ID, o.Host.CUDAMax)
		}
	}
	have := ids(offers)
	for _, id := range []string{rowV100, rowJunkCC, rowArm64, rowSmallDisk} {
		if slices.Contains(have, id) {
			t.Errorf("row %s passed the image filter", id)
		}
	}
	// driver_vers null does not drop a good host; a slow one stays (Fast is off) with its pull time
	if !slices.Contains(have, rowNullDriver) || !slices.Contains(have, rowSlow) {
		t.Fatalf("offers %v", have)
	}
	for _, o := range offers {
		if o.ID == rowSlow && (o.Fit.Level != "warn" || !slices.Contains(o.Fit.Codes, "pull_slow") || o.Fit.PullMin == nil) {
			t.Fatalf("slow host fit %+v", o.Fit)
		}
	}
}

func TestOffersImageShowAllBadges(t *testing.T) {
	e := liveEnv(t)
	offers, err := e.c.Offers(context.Background(), testKey, imageQuery(provision.OfferQuery{Limit: 200, ShowAll: true}))
	if err != nil {
		t.Fatal(err)
	}
	body := decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)
	for _, k := range []string{"cuda_max_good", "compute_cap", "cpu_arch"} {
		if _, has := body[k]; has {
			t.Fatalf("show_all still sends %s", k)
		}
	}
	var v100 *provision.Offer
	blocks := 0
	for i, o := range offers {
		if o.ID == rowV100 {
			v100 = &offers[i]
		}
		if o.Fit != nil && o.Fit.Level == "block" {
			blocks++
		}
	}
	if v100 == nil || v100.Fit == nil || v100.Fit.Level != "block" || !slices.Contains(v100.Fit.Codes, "gpu_sm") {
		t.Fatalf("V100 under show_all: %+v", v100)
	}
	if blocks < 10 {
		t.Fatalf("%d badged blocks under show_all", blocks)
	}
	// the disk floor still applies under show_all
	if slices.Contains(ids(offers), rowSmallDisk) {
		t.Fatal("a 30 GB host is listed")
	}
}

func TestOffersImageFastAndModel(t *testing.T) {
	e := liveEnv(t)
	q := imageQuery(provision.OfferQuery{Limit: 200, Fast: true, Model: &hostfit.ModelRef{Arch: "flux"}})
	offers, err := e.c.Offers(context.Background(), testKey, q)
	if err != nil {
		t.Fatal(err)
	}
	body := decode(t, e.last(t, "POST", "/api/v0/bundles/").Body)
	got, _ := json.Marshal(map[string]any{"inet": body["inet_down"], "bw": body["disk_bw"], "disk": body["disk_space"]})
	if string(got) != `{"bw":{"gte":300},"disk":{"gte":80},"inet":{"gte":500}}` {
		t.Fatalf("fast and model terms %s", got)
	}
	for _, o := range offers {
		if o.Host.InetDownMbps < hostfit.FastInetDownMbps || o.Fit.VRAM == "" || o.Fit.VRAM == hostfit.VRAMUnknown {
			t.Fatalf("offer %s inet %v vram %q", o.ID, o.Host.InetDownMbps, o.Fit.VRAM)
		}
	}
}

func TestQuoteCarriesHost(t *testing.T) {
	e := setup(t, mockkit.Options{})
	o, err := e.c.Quote(context.Background(), testKey, "57278070", imageQuery(provision.OfferQuery{DiskGB: 100}))
	if err != nil {
		t.Fatal(err)
	}
	if o.Host == nil || o.Host.SM != 89 || o.Host.CUDAMax != "13.4" || o.Fit == nil || o.Fit.Level == "block" {
		t.Fatalf("quote host %+v fit %+v", o.Host, o.Fit)
	}
}

func TestMockRefusesDriverFilter(t *testing.T) {
	e := setup(t, mockkit.Options{})
	for _, k := range []string{"driver_vers", "driver_version"} {
		_, err := search(context.Background(), testKey, map[string]any{k: map[string]any{"gte": 580}})
		if provision.ClassOf(err) != provision.ClassInvalid || !strings.Contains(err.Error(), k) {
			t.Fatalf("%s: %v", k, err)
		}
	}
	_ = e
}

func TestMockKeylessCap(t *testing.T) {
	e := liveEnv(t)
	res, err := http.Get(e.m.URL + "/api/v0/bundles/?q=" + `{"limit":200}`)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out struct {
		Offers []map[string]any `json:"offers"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil || len(out.Offers) != 64 {
		t.Fatalf("keyless search: %d rows, %v (want the 64-row cap)", len(out.Offers), err)
	}
}
