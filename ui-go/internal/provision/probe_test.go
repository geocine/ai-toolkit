package provision

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/geocine/aitoolkit/internal/netguard"
)

const probeToken = "aitk_probe_token_0001"

// probeNode is a stand-in aitk server: health answers healthStatus, /v1/system
// answers 200 for the right token, 401 for a wrong one, and without a token
// 401 unless open.
type probeNode struct {
	healthStatus int
	version      string
	token        string // what it accepts
	open         bool
	stripAuth    bool // a proxy that drops Authorization
	hits         atomic.Int32
}

func (n *probeNode) start(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.hits.Add(1)
		if n.stripAuth {
			r.Header.Del("Authorization")
		}
		switch r.URL.Path {
		case "/v1/health":
			w.WriteHeader(n.healthStatus)
			_, _ = w.Write([]byte(`{"ok":true,"version":"` + n.version + `","started_at":"2026-10-24T12:00:00Z"}`))
		case "/v1/system":
			got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
			switch {
			case got == "" && n.open, got != "" && got == n.token:
				_, _ = w.Write([]byte(`{"gpus":[]}`))
			default:
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte(`{"error":{"code":"unauthorized","message":"missing or invalid bearer token"}}`))
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func probeAll(eps []Endpoint, simulated bool) ProbeResult {
	return DefaultProber.Probe(context.Background(), eps, probeToken, simulated)
}

func TestProbeStates(t *testing.T) {
	ready := &probeNode{healthStatus: 200, version: "0.7.3+abc1234", token: probeToken}
	mismatch := &probeNode{healthStatus: 200, version: "x", token: "template-token-0000"}
	open := &probeNode{healthStatus: 200, version: "x", token: probeToken, open: true}
	sick := &probeNode{healthStatus: 503, token: probeToken}
	readyURL, mismatchURL, openURL, sickURL := ready.start(t), mismatch.start(t), open.start(t), sick.start(t)

	res := probeAll([]Endpoint{{URL: readyURL, Via: "direct"}}, true)
	if !res.Health || !res.TokenOK || res.TokenRejected || res.AuthEnforced == nil || !*res.AuthEnforced ||
		res.BaseURL != readyURL || res.Version != "0.7.3+abc1234" || res.Err != "" {
		t.Fatalf("ready: %+v", res)
	}

	res = probeAll([]Endpoint{{URL: mismatchURL}}, true)
	if !res.Health || res.TokenOK || !res.TokenRejected {
		t.Fatalf("token mismatch: %+v", res)
	}

	res = probeAll([]Endpoint{{URL: openURL}}, true)
	if !res.TokenOK || res.AuthEnforced == nil || *res.AuthEnforced {
		t.Fatalf("open server: %+v", res)
	}

	res = probeAll([]Endpoint{{URL: sickURL}}, true)
	if res.Health || res.TokenOK || res.BaseURL != "" || res.Err == "" {
		t.Fatalf("unhealthy: %+v", res)
	}
	if sick.hits.Load() != 1 {
		t.Fatalf("an unhealthy endpoint got %d requests, want only the health check", sick.hits.Load())
	}

	// the first endpoint whose health is 200 wins
	res = probeAll([]Endpoint{{URL: sickURL, Via: "proxy"}, {URL: readyURL, Via: "direct"}}, true)
	if !res.TokenOK || res.BaseURL != readyURL || res.Err != "" {
		t.Fatalf("second endpoint: %+v", res)
	}
	// a healthy endpoint that rejects the token is only a fallback
	res = probeAll([]Endpoint{{URL: mismatchURL}, {URL: readyURL}}, true)
	if res.BaseURL != readyURL || !res.TokenOK || res.TokenRejected {
		t.Fatalf("the endpoint that accepts the token wins: %+v", res)
	}
	res = probeAll([]Endpoint{{URL: mismatchURL}, {URL: sickURL}}, true)
	if res.BaseURL != mismatchURL || !res.Health || !res.TokenRejected || res.Err != "" {
		t.Fatalf("no endpoint accepts the token, the first healthy one is the answer: %+v", res)
	}
	if res := DefaultProber.Probe(context.Background(), []Endpoint{{URL: sickURL}, {URL: mismatchURL}, {URL: readyURL}}, "", true); res.BaseURL != mismatchURL || !res.Health {
		t.Fatalf("with no token the first healthy endpoint wins: %+v", res)
	}

	// a dead endpoint (closed listener) falls through to the next one
	dead := httptest.NewServer(http.NotFoundHandler())
	deadURL := dead.URL
	dead.Close()
	res = probeAll([]Endpoint{{URL: deadURL}, {URL: readyURL}}, true)
	if !res.TokenOK || res.BaseURL != readyURL {
		t.Fatalf("dead first endpoint: %+v", res)
	}
	if res := probeAll(nil, true); res.Health || res.Err != "" {
		t.Fatalf("no endpoints: %+v", res)
	}
}

// A provider proxy that serves the public /v1/health but drops Authorization
// (QuickPod's is UNCONFIRMED) must not hide the direct endpoint behind it.
func TestProbeProxyDropsAuthorization(t *testing.T) {
	proxy := &probeNode{healthStatus: 200, version: "1", token: probeToken, stripAuth: true}
	direct := &probeNode{healthStatus: 200, version: "1", token: probeToken}
	proxyURL, directURL := proxy.start(t), direct.start(t)
	res := probeAll([]Endpoint{{URL: proxyURL, Secure: true, Via: "proxy"}, {URL: directURL, Via: "direct"}}, true)
	if res.BaseURL != directURL || !res.TokenOK || res.TokenRejected || res.Err != "" {
		t.Fatalf("proxy without Authorization, then direct: %+v", res)
	}
	if proxy.hits.Load() == 0 {
		t.Fatal("the proxy endpoint was not tried first")
	}
}

// The probe's clients apply netguard when they connect, not only the URL
// rules before: a proxy hostname that resolves to a private address is
// refused at dial time.
func TestProbeClientsGuardConnections(t *testing.T) {
	n := &probeNode{healthStatus: 200, token: probeToken}
	u := n.start(t)
	p := DefaultProber.(*prober)
	if _, err := p.client(false).Get(u + "/v1/health"); !errors.Is(err, netguard.ErrBlocked) {
		t.Fatalf("strict client on loopback: %v, want netguard.ErrBlocked", err)
	}
	if n.hits.Load() != 0 {
		t.Fatalf("the strict client reached loopback (%d requests)", n.hits.Load())
	}
	if _, err := p.client(true).Get("http://203.0.113.10:8675/v1/health"); !errors.Is(err, netguard.ErrBlocked) {
		t.Fatalf("loopback client on a public address: %v, want netguard.ErrBlocked", err)
	}
	if res, err := p.client(true).Get(u + "/v1/health"); err != nil {
		t.Fatalf("loopback client on loopback: %v", err)
	} else {
		res.Body.Close()
	}
	for _, simulated := range []bool{false, true} {
		c := p.client(simulated)
		if c.CheckRedirect == nil || !errors.Is(c.CheckRedirect(nil, nil), http.ErrUseLastResponse) {
			t.Fatalf("client(%v) follows redirects", simulated)
		}
	}
}

// A real machine is never probed on loopback or a private address: the URL
// rules and netguard refuse it before a request is sent.
func TestProbeRefusesNonPublicEndpoints(t *testing.T) {
	n := &probeNode{healthStatus: 200, token: probeToken}
	u := n.start(t)
	for _, ep := range []string{u, "http://10.0.0.1:8675", "http://169.254.169.254", "http://localhost:8675", "https://evil.example"} {
		res := probeAll([]Endpoint{{URL: ep}}, false)
		if res.Health || res.TokenOK || res.Err == "" {
			t.Fatalf("%s: %+v", ep, res)
		}
	}
	if n.hits.Load() != 0 {
		t.Fatalf("a refused endpoint got %d requests", n.hits.Load())
	}
	// and a simulated machine only ever on 127.0.0.1
	if res := probeAll([]Endpoint{{URL: "http://203.0.113.10:8675"}}, true); res.Health || !strings.Contains(res.Err, "127.0.0.1") {
		t.Fatalf("simulated public: %+v", res)
	}
}

func TestProbeNoRedirectsAndCappedBody(t *testing.T) {
	ready := &probeNode{healthStatus: 200, version: "1", token: probeToken}
	readyURL := ready.start(t)
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, readyURL+r.URL.Path, http.StatusFound)
	}))
	t.Cleanup(redirect.Close)
	if res := probeAll([]Endpoint{{URL: redirect.URL}}, true); res.Health {
		t.Fatalf("followed a redirect: %+v", res)
	}
	if ready.hits.Load() != 0 {
		t.Fatal("the redirect target was contacted")
	}

	long := strings.Repeat("v", 200)
	big := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/health" {
			_, _ = w.Write([]byte(`{"version":"` + long + `"}`))
			return
		}
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(big.Close)
	res := probeAll([]Endpoint{{URL: big.URL}}, true)
	if !res.Health || res.Version != long[:probeVersionMax] || !res.TokenRejected {
		t.Fatalf("long version: %+v", res)
	}

	huge := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"pad":"` + strings.Repeat("x", probeMax) + `","version":"late"}`))
	}))
	t.Cleanup(huge.Close)
	if res := probeAll([]Endpoint{{URL: huge.URL}}, true); !res.Health || res.Version != "" {
		t.Fatalf("over the body cap: %+v", res)
	}
}
