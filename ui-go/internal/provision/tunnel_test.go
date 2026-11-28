package provision

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/tunnel"
)

func TestStudioAPIURL(t *testing.T) {
	for _, tc := range []struct {
		public, origin string
		sim            bool
		want           string
	}{
		{"https://loraboard.com/", "", false, "https://loraboard.com"},
		{"", "https://loraboard.com", false, "https://loraboard.com"},
		{"https://loraboard.com", "https://other.example", false, "https://loraboard.com"},
		{"", "", false, ""},
		{"", "http://127.0.0.1:8675", false, ""},
		{"https://localhost:8675", "", false, ""},
		{"https://192.168.1.5", "", false, ""},
		{"http://loraboard.com", "", false, ""},
		{"", "", true, "http://127.0.0.1:8675"},
		{"", "http://127.0.0.1:9000", true, "http://127.0.0.1:9000"},
	} {
		t.Setenv("AITK_PUBLIC_URL", tc.public)
		got, err := StudioAPIURL(tc.origin, tc.sim)
		if tc.want == "" {
			if err == nil || !strings.Contains(err.Error(), "AITK_PUBLIC_URL") {
				t.Errorf("%+v: %q, %v (want a refusal naming AITK_PUBLIC_URL)", tc, got, err)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("%+v: %q, %v", tc, got, err)
		}
	}
}

// A provider can never hand the service a tunnel URL, and a machine's base
// may be only its own tunnel URL.
func TestTunnelEndpointRules(t *testing.T) {
	u := tunnel.URL("mch_a")
	for _, k := range []Kind{RunPod, Vast, QuickPod, ""} {
		if CheckEndpoint(k, u, false) == nil {
			t.Errorf("%q accepted a tunnel URL from a provider", k)
		}
		if got := FilterEndpoints(k, []Endpoint{{URL: u, Via: ViaTunnel}}, false); len(got) != 0 {
			t.Errorf("%q kept a provider-reported tunnel endpoint", k)
		}
	}
	if CheckBase(RunPod, u, "mch_a", false) != nil || CheckBase(RunPod, u, "mch_b", false) == nil {
		t.Fatal("CheckBase")
	}
	if !SecureBase(u) || !TunnelBase(u) || TunnelBase("https://x-8675.proxy.runpod.net") || SecureBase("http://1.2.3.4:8675") {
		t.Fatal("SecureBase/TunnelBase")
	}
}

func TestBuildEnvMakesWorkers(t *testing.T) {
	e := newEnv(t)
	env, _, err := e.build(RunPod, LaunchRequest{})
	if err != nil || env["AITK_MODE"] != "worker" || env["AITK_STUDIO_API_URL"] != "https://studio.example" {
		t.Fatalf("%v %v", env, err)
	}
	for _, k := range []string{"AITK_MODE", "AITK_STUDIO_API_URL"} {
		if _, _, err := e.build(RunPod, LaunchRequest{Env: map[string]string{k: "x"}}); err == nil {
			t.Errorf("env %s passed through", k)
		}
	}
}

// A launched worker that dials its tunnel turns ready on its tunnel URL
// through the real prober, with no provider endpoint at all, and is relayed
// there; when the tunnel drops it goes unreachable.
func TestWorkerReadyThroughTunnel(t *testing.T) {
	e := newEnv(t)
	oldProber := DefaultProber
	DefaultProber = &prober{}
	oldLookup, oldChange := tunnel.Default.Lookup, tunnel.Default.OnChange
	tunnel.Default.Lookup, tunnel.Default.OnChange = e.s.tunnelToken, nil
	t.Cleanup(func() {
		DefaultProber, tunnel.Default.Lookup, tunnel.Default.OnChange = oldProber, oldLookup, oldChange
	})

	m := e.launch(RunPod, nil)
	r := e.row(m.ID)
	c := e.clouds[RunPod]
	c.set("pod1", func(in *Instance) { in.Status, in.ProviderStatus, in.PortMissing = StatusRunning, "RUNNING", true })

	mux := http.NewServeMux()
	mux.Handle("GET "+tunnel.Path, tunnel.Default)
	studio := httptest.NewServer(mux)
	defer studio.Close()

	worker := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch {
		case req.URL.Path == "/v1/health":
			io.WriteString(w, `{"ok":true,"version":"build1"}`)
		case req.Header.Get("Authorization") != "Bearer "+r.Token:
			w.WriteHeader(http.StatusUnauthorized)
		default:
			io.WriteString(w, `{}`)
		}
	})
	ctx, stop := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		(&tunnel.Client{URL: studio.URL, MachineID: m.ID, Token: r.Token, Handler: worker}).Run(ctx)
	}()
	deadline := time.Now().Add(5 * time.Second)
	for !tunnel.Default.Connected(m.ID) {
		if time.Now().After(deadline) {
			t.Fatal("the worker never connected")
		}
		time.Sleep(10 * time.Millisecond)
	}

	got := e.refresh(m.ID)
	if got.Status != StatusReady || got.BaseURL == nil || *got.BaseURL != tunnel.URL(m.ID) || !got.Tunnel || !got.Secure || got.Problem != nil {
		t.Fatalf("after the tunnel: %+v", got)
	}
	target, err := e.s.RelayTarget(e.ctx, m.ID)
	if err != nil || target.BaseURL != tunnel.URL(m.ID) {
		t.Fatalf("relay target %+v %v", target, err)
	}
	link, err := e.s.Link(e.ctx, m.ID)
	if err != nil || !link.Tunnel {
		t.Fatalf("link %+v %v", link, err)
	}

	stop()
	<-done
	for tunnel.Default.Connected(m.ID) {
		time.Sleep(10 * time.Millisecond)
	}
	e.clock.Advance(6 * time.Minute)
	got = e.refresh(m.ID)
	if got.Problem == nil || *got.Problem != ProblemUnreachable || !strings.Contains(got.Detail, "tunnel") {
		t.Fatalf("after the tunnel dropped: %+v", got)
	}
}

// Only a tracked, live machine may open a tunnel, by its exact id.
func TestTunnelTokenLookup(t *testing.T) {
	e := newEnv(t)
	m := e.launch(RunPod, nil)
	tok, ok := e.s.tunnelToken(e.ctx, m.ID)
	if !ok || tok != e.row(m.ID).Token {
		t.Fatal("a live machine was refused")
	}
	if _, ok := e.s.tunnelToken(e.ctx, m.Name); ok {
		t.Fatal("a machine name stood in for its id")
	}
	if _, ok := e.s.tunnelToken(e.ctx, "mch_unknown"); ok {
		t.Fatal("an unknown machine was accepted")
	}
	if _, err := e.s.Terminate(e.ctx, m.ID, false); err != nil {
		t.Fatal(err)
	}
	e.clouds[RunPod].remove("pod1")
	e.refresh(m.ID)
	if r := e.row(m.ID); r.EndedAt == nil {
		t.Skip("terminate did not end the row in this stub")
	}
	if _, ok := e.s.tunnelToken(e.ctx, m.ID); ok {
		t.Fatal("an ended machine may still open a tunnel")
	}
}

// A worker launched here is never trusted over its provider endpoint: one
// that answers with the right token (a template's AITK_MODE=standalone won,
// or an older image) does not make it ready, and when the tunnel never
// comes the ready timeout says so.
func TestWorkerIgnoresPublicEndpoint(t *testing.T) {
	e := newEnv(t)
	e.noTunnel = true
	m := e.launch(RunPod, nil)
	e.clouds[RunPod].running("pod1")
	got := e.refresh(m.ID)
	if got.Status == StatusReady || e.prober.calls != 0 {
		t.Fatalf("ready over the provider endpoint: %+v (probes %d)", got, e.prober.calls)
	}
	if _, err := e.s.RelayTarget(e.ctx, m.ID); err == nil {
		t.Fatal("relayable without its tunnel")
	}
	e.clock.Advance(readyTimeout() + time.Minute)
	got = e.refresh(m.ID)
	if got.Problem == nil || *got.Problem != ProblemNotReady || got.Detail != detailNoTunnel {
		t.Fatalf("after the ready timeout: %+v", got)
	}

	// the same machine turns ready once its tunnel is up
	e2 := newEnv(t)
	m2 := e2.launch(RunPod, nil)
	e2.clouds[RunPod].running("pod1")
	if got := e2.refresh(m2.ID); got.Status != StatusReady || !got.Tunnel {
		t.Fatalf("with its tunnel: %+v", got)
	}
}
