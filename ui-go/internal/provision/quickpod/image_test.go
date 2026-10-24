package quickpod

import (
	"context"
	"slices"
	"testing"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/quickpod/quickpodmock"
)

// imageQuery is q with the filter the service derives for today's default
// image (its built-in requirements: CUDA 13.0, sm 75-120).
func imageQuery(q provision.OfferQuery) provision.OfferQuery {
	img, _ := hostfit.FallbackImage(imagecheck.DefaultRepo + ":latest")
	f := hostfit.FilterFor(img.Requires, hostfit.NeedFor(img, q.Model, nil), q.Model, hostfit.Policy{Fast: q.Fast, ShowAll: q.ShowAll})
	q.Filter, q.ImageInfo = &f, img
	return q
}

// Rows of the live fixture.
const (
	rowGTX1660   = "9000001"
	rowSmallDisk = "9000002"
	rowNoMaxCUDA = "6302336" // max_cuda "", driver_vers 5.95099002e+08
	rowOldDriver = "9000003" // max_cuda "12.8\n": below the image's CUDA 13.0
)

func offerIDs(offers []provision.Offer) []string {
	var out []string
	for _, o := range offers {
		out = append(out, o.ID)
	}
	return out
}

func TestOffersImageClientSide(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	m.AddOffers(quickpodmock.LiveOffers()...)
	offers, err := a.Offers(context.Background(), key, imageQuery(provision.OfferQuery{Limit: 200}))
	if err != nil {
		t.Fatal(err)
	}
	have := offerIDs(offers)
	if len(offers) < 5 || slices.Contains(have, rowGTX1660) || slices.Contains(have, rowSmallDisk) || slices.Contains(have, rowOldDriver) {
		t.Fatalf("offers %v", have)
	}
	for _, o := range offers {
		if o.Host == nil || o.Fit == nil || o.Fit.Level == "block" || o.Host.Provider != "quickpod" {
			t.Fatalf("offer %s host %+v fit %+v", o.ID, o.Host, o.Fit)
		}
		if o.ID == rowNoMaxCUDA && (o.Host.CUDAMax != "13.2" || o.Host.CUDASource != "quickpod.driver_vers" || o.Host.Driver != "595.99.02") {
			t.Fatalf("decoded driver host %+v", o.Host)
		}
		// the hosts report bandwidth: every fit has a pull time
		if o.Host.InetDownMbps > 0 && o.Fit.PullMin == nil {
			t.Fatalf("offer %s: no pull time at %v Mbps", o.ID, o.Host.InetDownMbps)
		}
	}
}

func TestOffersImageShowAll(t *testing.T) {
	m, _ := setup(t, mockkit.Options{})
	m.AddOffers(quickpodmock.LiveOffers()...)
	offers, err := a.Offers(context.Background(), key, imageQuery(provision.OfferQuery{Limit: 200, ShowAll: true}))
	if err != nil {
		t.Fatal(err)
	}
	i := slices.IndexFunc(offers, func(o provision.Offer) bool { return o.ID == rowGTX1660 })
	if i < 0 || offers[i].Fit.Level != "warn" || !slices.Contains(offers[i].Fit.Codes, "gpu_sm_flash_attn") {
		t.Fatalf("GTX 1660 under show_all: %v", offerIDs(offers))
	}
	// a driver too old for the image is listed, badged as a block
	j := slices.IndexFunc(offers, func(o provision.Offer) bool { return o.ID == rowOldDriver })
	if j < 0 || offers[j].Fit.Level != "block" || !slices.Contains(offers[j].Fit.Codes, "host_cuda") {
		t.Fatalf("old driver under show_all: %v", offerIDs(offers))
	}
	// a disk too small for the need stays out even under show_all
	if slices.Contains(offerIDs(offers), rowSmallDisk) {
		t.Fatal("30 GB host listed")
	}
}

func TestQuoteImageHost(t *testing.T) {
	setup(t, mockkit.Options{})
	o, err := a.Quote(context.Background(), key, "6289141", imageQuery(provision.OfferQuery{DiskGB: 100}))
	if err != nil {
		t.Fatal(err)
	}
	if o.Host == nil || o.Host.CUDAMax != "13.1" || o.Host.SM != 120 || o.Fit == nil {
		t.Fatalf("quote host %+v fit %+v", o.Host, o.Fit)
	}
}
