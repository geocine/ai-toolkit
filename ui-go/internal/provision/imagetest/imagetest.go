// Package imagetest points the provisioning image check at in-process mocks
// for tests: registrymock serving its fixtures (the default image published
// as registrymock.Revision) and hfmock for model sizes. Nothing it starts
// leaves the machine.
//
// It sets imagecheck.Default, which provision.ImageChecker reads at call
// time, and never imports provision, so provision's own tests can use it.
package imagetest

import (
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/hostfit/hfmock"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/netguard"
)

// Options shapes the mocks Use starts.
type Options struct {
	Images []registrymock.Image // nil = registrymock.Fixtures()
	Faults registrymock.Faults
	// Unpublished makes the default image private (403 DENIED at the
	// token endpoint), the state before its owner publishes it (D-1).
	Unpublished bool
}

// Env is what Use started.
type Env struct {
	Registry *registrymock.Server
	HF       *hfmock.Server
	Checker  *imagecheck.Client
	Sizer    *hostfit.ModelSizer
}

// Use starts registrymock and hfmock, points the image check
// (imagecheck.Default, hence provision.ImageChecker) and
// hostfit.DefaultSizer at them, and restores both in t.Cleanup.
func Use(t testing.TB, o Options) *Env {
	t.Helper()
	images := o.Images
	if images == nil {
		images = registrymock.Fixtures()
	}
	f := o.Faults
	if o.Unpublished {
		priv := map[string]bool{registrymock.DefaultRepo: true}
		for k, v := range f.Private {
			priv[k] = v
		}
		f.Private = priv
	}
	reg := registrymock.New(registrymock.Options{Images: images, Faults: f})
	reg.StartTLS()
	hf := hfmock.New(hfmock.Options{})
	hf.Start()
	e := &Env{
		Registry: reg,
		HF:       hf,
		Checker:  imagecheck.New(imagecheck.Options{HTTP: reg.Client(), CheckTimeout: 5 * time.Second, Logf: func(string, ...any) {}}),
		Sizer:    hostfit.NewSizer(hostfit.SizerOptions{Base: hf.URL(), HTTP: netguard.NewClient(netguard.Options{AllowLoopback: true})}),
	}
	oldChecker, oldSizer := imagecheck.Default, hostfit.DefaultSizer
	imagecheck.Default, hostfit.DefaultSizer = e.Checker, e.Sizer
	t.Cleanup(func() {
		imagecheck.Default, hostfit.DefaultSizer = oldChecker, oldSizer
		reg.Close()
		hf.Close()
	})
	return e
}
