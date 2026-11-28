package tunnel

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"

	"github.com/geocine/aitoolkit/internal/netguard"
)

// studioAPI serves hub at Path on a loopback test server.
func studioAPI(t *testing.T, hub *Hub) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle("GET "+Path, hub)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func worker(t *testing.T, base, id, token string, h http.Handler) context.CancelFunc {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		(&Client{URL: base, MachineID: id, Token: token, Handler: h}).Run(ctx)
	}()
	t.Cleanup(func() { cancel(); <-done })
	return cancel
}

func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func hubFor(tokens map[string]string) *Hub {
	return &Hub{Lookup: func(_ context.Context, id string) (string, bool) {
		tok, ok := tokens[id]
		return tok, ok
	}}
}

// A request on the studio API to the machine's tunnel URL, through a netguard
// transport with tunnels on, reaches the worker's handler with its headers.
func TestTunnelRoundTrip(t *testing.T) {
	old := Default
	Default = hubFor(map[string]string{"mch_AbC123": "tok"})
	t.Cleanup(func() { Default = old })
	srv := studioAPI(t, Default)

	worker(t, srv.URL, "mch_AbC123", "tok", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, r.Method+" "+r.URL.Path+" "+r.Header.Get("Authorization")+" "+r.RemoteAddr)
	}))
	waitFor(t, "the tunnel", func() bool { return Default.Connected("mch_AbC123") })

	c := netguard.NewClient(netguard.Options{Tunnel: true})
	req, _ := http.NewRequest("GET", URL("mch_AbC123")+"/v1/health", nil)
	req.Header.Set("Authorization", "Bearer tok")
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if got := string(body); got != "GET /v1/health Bearer tok tunnel:0" {
		t.Fatalf("worker saw %q", got)
	}

	// a transport without tunnels refuses the host instead of resolving it
	plain := netguard.NewClient(netguard.Options{})
	if _, err := plain.Get(URL("mch_AbC123") + "/v1/health"); err == nil || !strings.Contains(err.Error(), "tunnel") {
		t.Fatalf("plain transport: %v", err)
	}
}

// A wrong token, an unknown machine and a browser (Origin) are refused.
func TestTunnelRefusals(t *testing.T) {
	hub := hubFor(map[string]string{"mch_a": "right"})
	srv := studioAPI(t, hub)
	for _, tc := range []struct{ id, token string }{{"mch_a", "wrong"}, {"mch_b", "right"}} {
		c := &Client{URL: srv.URL, MachineID: tc.id, Token: tc.token, Handler: http.NotFoundHandler()}
		ws := "ws" + strings.TrimPrefix(srv.URL, "http") + Path
		err := c.once(context.Background(), ws)
		if err == nil || !strings.Contains(err.Error(), "refused") {
			t.Fatalf("%s/%s: %v", tc.id, tc.token, err)
		}
		if hub.Connected(tc.id) {
			t.Fatalf("%s connected", tc.id)
		}
	}
	req, _ := http.NewRequest("GET", srv.URL+Path, nil)
	req.Header.Set(MachineHeader, "mch_a")
	req.Header.Set("Origin", "https://evil.example")
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "websocket")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("browser handshake answered %d", res.StatusCode)
	}
}

// Mismatched tokens never connect.
func TestTokenMismatch(t *testing.T) {
	hub := hubFor(map[string]string{"mch_a": "studio-token"})
	srv := studioAPI(t, hub)
	c := &Client{URL: srv.URL, MachineID: "mch_a", Token: "worker-token", Handler: http.NotFoundHandler()}
	err := c.once(context.Background(), "ws"+strings.TrimPrefix(srv.URL, "http")+Path)
	if err == nil {
		t.Fatal("connected with mismatched tokens")
	}
}

// A reconnect replaces the old tunnel, and Drop closes it.
func TestTunnelReconnectAndDrop(t *testing.T) {
	hub := hubFor(map[string]string{"mch_a": "tok"})
	srv := studioAPI(t, hub)
	cancel := worker(t, srv.URL, "mch_a", "tok", http.NotFoundHandler())
	waitFor(t, "first tunnel", func() bool { return hub.Connected("mch_a") })
	cancel()
	waitFor(t, "disconnect", func() bool { return !hub.Connected("mch_a") })
	worker(t, srv.URL, "mch_a", "tok", http.NotFoundHandler())
	waitFor(t, "second tunnel", func() bool { return hub.Connected("mch_a") })
	hub.Drop("mch_a")
	if hub.Connected("mch_a") {
		t.Fatal("still connected after Drop")
	}
	if _, err := hub.Dial(context.Background(), "mch_a"); err == nil {
		t.Fatal("dialled a dropped tunnel")
	}
}

func TestCheckURL(t *testing.T) {
	for raw, ok := range map[string]bool{
		"https://lora.aivan.io":            true,
		"https://lora.aivan.io/":           true,
		"http://127.0.0.1:8675":            true,
		"http://localhost:8675":            true,
		"http://host.docker.internal:8675": true,
		"http://192.168.1.20:8675":         true,
		"http://lora.aivan.io":             false,
		"http://8.8.8.8":                   false,
		"ftp://x":                          false,
		"https://user@x.example":           false,
		"":                                 false,
	} {
		if _, err := CheckURL(raw); (err == nil) != ok {
			t.Errorf("CheckURL(%q) = %v", raw, err)
		}
	}
}

func TestMachineOf(t *testing.T) {
	id := "mch_AbC123"
	u := URL(id)
	host := strings.TrimPrefix(u, "http://")
	if got, ok := MachineOf(strings.ToUpper(host)); !ok || got != id {
		t.Fatalf("MachineOf(%q) = %q, %v", host, got, ok)
	}
	for _, bad := range []string{"x.tunnel.aitk", ".tunnel.aitk", "a.b.tunnel.aitk", "example.com"} {
		if _, ok := MachineOf(bad); ok {
			t.Errorf("MachineOf(%q) accepted", bad)
		}
	}
	if !IsURL(u+"/", id) || IsURL(u, "mch_other") {
		t.Fatal("IsURL")
	}
}

// A worker refuses a studio API that accepts it without proving the token.
func TestWorkerChecksStudioAPI(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET "+Path, func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer c.CloseNow()
		ctx := r.Context()
		wsjson.Write(ctx, c, hello{V: 1, Challenge: "c"})
		var a answer
		wsjson.Read(ctx, c, &a)
		wsjson.Write(ctx, c, done{OK: true, MAC: "forged"})
		time.Sleep(100 * time.Millisecond)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	c := &Client{URL: srv.URL, MachineID: "mch_a", Token: "tok", Handler: http.NotFoundHandler()}
	err := c.once(context.Background(), "ws"+strings.TrimPrefix(srv.URL, "http")+Path)
	if err == nil || !strings.Contains(err.Error(), "couldn't prove") {
		t.Fatalf("impostor studio API: %v", err)
	}
}
