// Package sim is simulate mode (AITK_PROVISION_SIMULATE=1): the three provider
// mocks run in-process on their own loopback listeners, never on the public
// mux, and accept only a random per-process secret, so only this server's
// provisioning service can create simulated pods. Each running instance gets
// a loopback node (node.go) in front of this server's own handler. Nothing
// calls RunPod, Vast.ai or QuickPod: call.go refuses their hosts while
// provision.Simulated is set.
package sim

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/provision/quickpod"
	"github.com/geocine/aitoolkit/internal/provision/quickpod/quickpodmock"
	"github.com/geocine/aitoolkit/internal/provision/runpod"
	"github.com/geocine/aitoolkit/internal/provision/runpod/runpodmock"
	"github.com/geocine/aitoolkit/internal/provision/vast"
	"github.com/geocine/aitoolkit/internal/provision/vast/vastmock"
)

// Simulated machines come up in DefaultDelay (split 1:2 between allocation
// and container start) and answer AppDelay later, unless
// AITK_PROVISION_SIMULATE_DELAY says otherwise.
const (
	DefaultDelay = 15 * time.Second
	AppDelay     = 4 * time.Second
)

// Enabled reports AITK_PROVISION_SIMULATE (1 or true).
func Enabled() bool {
	v := strings.TrimSpace(os.Getenv("AITK_PROVISION_SIMULATE"))
	return v == "1" || strings.EqualFold(v, "true")
}

// Delays reads AITK_PROVISION_SIMULATE_DELAY: a Go duration or whole seconds,
// the time from create to running. 0 makes every step instant, the app
// included.
func Delays() mockkit.Delays {
	d := DefaultDelay
	if v := strings.TrimSpace(os.Getenv("AITK_PROVISION_SIMULATE_DELAY")); v != "" {
		if p, err := time.ParseDuration(v); err == nil && p >= 0 {
			d = p
		} else if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			d = time.Duration(n) * time.Second
		} else {
			log.Printf("simulate: AITK_PROVISION_SIMULATE_DELAY=%q is not a duration; using %s", v, DefaultDelay)
		}
	}
	app := AppDelay
	if d == 0 {
		app = 0
	}
	return mockkit.Delays{Provision: d / 3, Start: d - d/3, App: app}
}

// Start turns simulate mode on: provision.Simulated, a fresh sim secret, the
// three mocks on 127.0.0.1:0 and the adapters pointed at them, and the image
// check and model sizer on in-process registry and Hugging Face mocks
// (unless AITK_PROVISION_SIMULATE_REGISTRY=real, which reads the public
// registries and Hugging Face anonymously). inner is the handler simulated
// nodes serve (this server below Auth). stop, also run when ctx ends, closes
// the mocks and their nodes and puts the adapters back.
func Start(ctx context.Context, inner http.Handler) (stop func(), err error) {
	return start(ctx, inner, mockkit.RealClock{}, Delays())
}

// The adapter package vars Start replaces, so stop can restore them.
type saved struct {
	rpAPI, rpGraphQL string
	rpProxy          func(string) string
	vastAPI          string
	vastLogs         func(string) bool
	qpAPI            string
	qpProxy          bool
	images           *imagecheck.Client
	sizer            *hostfit.ModelSizer
}

// RealRegistry reports AITK_PROVISION_SIMULATE_REGISTRY=real: simulate
// then checks images against the real public registries.
func RealRegistry() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("AITK_PROVISION_SIMULATE_REGISTRY")), "real")
}

var running sync.Mutex // one simulate session at a time (tests start and stop several)

// active is the running session's mocks, for this package's tests (a
// runpodmock PullFailText, for one).
var active struct {
	rp *runpodmock.Mock
	vm *vastmock.Mock
	qm *quickpodmock.Mock
}

func start(ctx context.Context, inner http.Handler, clock mockkit.Clock, delays mockkit.Delays) (func(), error) {
	if inner == nil {
		return nil, errors.New("simulate: no handler for the simulated nodes")
	}
	if !running.TryLock() {
		return nil, errors.New("simulate: already running")
	}
	b := make([]byte, 32)
	_, _ = rand.Read(b) // crypto/rand.Read does not fail (Go 1.24+ aborts instead)
	secret := base64.RawURLEncoding.EncodeToString(b)

	opts := mockkit.Options{Clock: clock, Delays: delays, SimSecret: secret, Node: factory(inner, clock)}
	rp, vm, qm := runpodmock.New(opts), vastmock.New(opts), quickpodmock.New(opts)
	// The live-shaped rows the image blocks (an old driver, a V100, arm64, a
	// 30 GB disk, a slow link), so Show all has something to refuse, and a
	// template pinned to a build (IMAGE-PLAN §7.3).
	vm.AddOffers(vastmock.LiveOffers()...)
	qm.AddOffers(quickpodmock.LiveOffers()...)
	qm.AddTemplate(quickpodmock.PinnedTemplate(registrymock.Revision))
	rpURL, vmURL, qmURL := rp.Start(), vm.Start(), qm.Start()
	active.rp, active.vm, active.qm = rp, vm, qm
	vastHost := strings.TrimPrefix(vmURL, "http://")

	old := saved{runpod.API, runpod.GraphQL, runpod.ProxyURL, vast.API, vast.LogsHostAllowed, quickpod.API, quickpod.UseProxy,
		imagecheck.Default, hostfit.DefaultSizer}
	if !RealRegistry() {
		// imagecheck never imports provision: simulate wires both here.
		imagecheck.Init(true)
		hostfit.InitSizer(true)
	}
	provision.SetSimSecret(secret)
	provision.Simulated.Store(true)
	runpod.API, runpod.GraphQL = rpURL, rpURL+"/graphql"
	// The one documented non-faithful substitution: a pod's proxy address is
	// its simulated node (runpodmock.NodeURL).
	runpod.ProxyURL = rp.NodeURL
	vast.API = vmURL
	vast.LogsHostAllowed = func(host string) bool { return host == vastHost }
	quickpod.API = qmURL
	quickpod.UseProxy = false // the nodes are plain http on loopback
	log.Printf("Provisioning is simulated: nothing calls RunPod, Vast.ai or QuickPod")

	var once sync.Once
	done := make(chan struct{})
	stop := func() {
		once.Do(func() {
			close(done)
			rp.Close()
			vm.Close()
			qm.Close()
			runpod.API, runpod.GraphQL, runpod.ProxyURL = old.rpAPI, old.rpGraphQL, old.rpProxy
			vast.API, vast.LogsHostAllowed = old.vastAPI, old.vastLogs
			quickpod.API, quickpod.UseProxy = old.qpAPI, old.qpProxy
			imagecheck.Default, hostfit.DefaultSizer = old.images, old.sizer
			active.rp, active.vm, active.qm = nil, nil, nil
			provision.Simulated.Store(false)
			provision.SetSimSecret("")
			running.Unlock()
		})
	}
	go func() {
		select {
		case <-ctx.Done():
			stop()
		case <-done:
		}
	}()
	return stop, nil
}

// WarnIfReachable logs that simulate is for local testing when the server
// listens on more than loopback (host is the --host value).
func WarnIfReachable(host string) {
	switch strings.Trim(strings.ToLower(strings.TrimSpace(host)), "[]") {
	case "127.0.0.1", "::1", "localhost":
		return
	}
	log.Printf("Simulate on a reachable server is for local testing only (--host %s)", host)
}
