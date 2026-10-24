package relay

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/netguard"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/v1api"
)

const (
	relayToken   = "relay-token-123"
	machineToken = "aitk_machine_token"
)

// seenReq is what the stub machine received.
type seenReq struct {
	Method, Path, Query, Body string
	Header                    http.Header
}

// stubMachine is a GPU server on loopback that records each request.
type stubMachine struct {
	*httptest.Server
	mu   sync.Mutex
	seen []seenReq
}

func newMachine(t *testing.T) *stubMachine {
	m := &stubMachine{}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/events/stream", func(w http.ResponseWriter, r *http.Request) {
		m.record(r)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: hello\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done() // never ends on its own, like the event stream
	})
	mux.HandleFunc("/go-home", func(w http.ResponseWriter, r *http.Request) {
		m.record(r)
		http.Redirect(w, r, "/studio/", http.StatusFound)
	})
	mux.HandleFunc("/go-base", func(w http.ResponseWriter, r *http.Request) {
		m.record(r)
		http.Redirect(w, r, m.URL+"/v1/health?x=1", http.StatusFound)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		m.record(r)
		w.Header().Set("Set-Cookie", "session=machine")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("X-Machine", "yes")
		w.WriteHeader(http.StatusCreated)
		io.WriteString(w, "machine says hi")
	})
	m.Server = httptest.NewServer(mux)
	t.Cleanup(m.Close)
	return m
}

func (m *stubMachine) record(r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	m.mu.Lock()
	m.seen = append(m.seen, seenReq{r.Method, r.URL.EscapedPath(), r.URL.RawQuery, string(b), r.Header.Clone()})
	m.mu.Unlock()
}

func (m *stubMachine) last(t *testing.T) seenReq {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.seen) == 0 {
		t.Fatal("the machine was not reached")
	}
	return m.seen[len(m.seen)-1]
}

func (m *stubMachine) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.seen)
}

// newRelayServer serves the relay route behind httpx.Auth, the way the
// relaying server does, with AI_TOOLKIT_AUTH set and target as every machine.
func newRelayServer(t *testing.T, target provision.RelayTarget) *httptest.Server {
	t.Setenv("AI_TOOLKIT_AUTH", relayToken)
	t.Setenv("AITK_PUBLIC_URL", "")
	mux := http.NewServeMux()
	for _, m := range []string{"GET", "POST", "PUT", "PATCH", "DELETE"} {
		mux.HandleFunc(m+" /v1/machines/{id}/relay/{path...}", func(w http.ResponseWriter, r *http.Request) {
			tt := target
			tt.MachineID = r.PathValue("id")
			Serve(w, r, tt, r.PathValue("path"), httpx.Authenticated(r))
		})
	}
	srv := httptest.NewServer(httpx.Auth(mux))
	t.Cleanup(srv.Close)
	return srv
}

func send(t *testing.T, method, u, body string, header map[string]string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(method, u, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range header {
		req.Header.Set(k, v)
	}
	c := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { res.Body.Close() })
	return res
}

func errCode(t *testing.T, res *http.Response) string {
	t.Helper()
	var p struct {
		Error struct{ Code string } `json:"error"`
	}
	if err := json.NewDecoder(res.Body).Decode(&p); err != nil {
		t.Fatal(err)
	}
	return p.Error.Code
}

func TestRelayPassesThrough(t *testing.T) {
	m := newMachine(t)
	c := newRelayServer(t, provision.RelayTarget{BaseURL: m.URL, Token: machineToken, Simulated: true})
	res := send(t, "POST", c.URL+"/v1/machines/mch_1/relay/v1/datasets/a%2Fb/files?x=1&y=two", "payload", map[string]string{
		"Authorization":      "Bearer " + relayToken,
		"Origin":             "https://studio.example",
		"Cookie":             "session=secret",
		"X-Forwarded-Prefix": "/evil",
		"Forwarded":          "for=1.2.3.4",
	})
	body, _ := io.ReadAll(res.Body)
	if res.StatusCode != 201 || string(body) != "machine says hi" || res.Header.Get("X-Machine") != "yes" {
		t.Fatalf("%d %q %v", res.StatusCode, body, res.Header)
	}
	if res.Header.Get("Set-Cookie") != "" || res.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Errorf("machine cookie or CORS headers leaked: %v", res.Header)
	}
	got := m.last(t)
	if got.Method != "POST" || got.Path != "/v1/datasets/a%2Fb/files" || got.Query != "x=1&y=two" || got.Body != "payload" {
		t.Errorf("request changed on the way: %+v", got)
	}
	h := got.Header
	if h.Get("Authorization") != "Bearer "+machineToken {
		t.Errorf("Authorization = %q, want the machine token", h.Get("Authorization"))
	}
	if h.Get("Origin") != "" || h.Get("Cookie") != "" || h.Get("Forwarded") != "" {
		t.Errorf("client headers reached the machine: %v", h)
	}
	cu, _ := url.Parse(c.URL)
	if h.Get("X-Forwarded-Prefix") != "/v1/machines/mch_1/relay" || h.Get("X-Forwarded-Host") != cu.Host ||
		h.Get("X-Forwarded-Proto") != "http" || h.Get("X-Forwarded-For") == "" {
		t.Errorf("forwarding headers: %v", h)
	}

	// behind its own https proxy the relaying server passes on the origin its
	// clients use (links.Origin), so the machine's absolute links work
	send(t, "GET", c.URL+"/v1/machines/mch_1/relay/v1/system", "", map[string]string{
		"Authorization": "Bearer " + relayToken, "X-Forwarded-Proto": "https", "X-Forwarded-Host": "relay.example",
	})
	if h := m.last(t).Header; h.Get("X-Forwarded-Host") != "relay.example" || h.Get("X-Forwarded-Proto") != "https" {
		t.Errorf("proxied relay server: %v", h)
	}
	t.Setenv("AITK_PUBLIC_URL", "https://public.example/")
	send(t, "GET", c.URL+"/v1/machines/mch_1/relay/v1/system", "", map[string]string{
		"Authorization": "Bearer " + relayToken, "X-Forwarded-Host": "relay.example",
	})
	if h := m.last(t).Header; h.Get("X-Forwarded-Host") != "public.example" || h.Get("X-Forwarded-Proto") != "https" {
		t.Errorf("AITK_PUBLIC_URL: %v", h)
	}
}

func TestRelayStreamsEvents(t *testing.T) {
	m := newMachine(t)
	c := newRelayServer(t, provision.RelayTarget{BaseURL: m.URL, Token: machineToken, Simulated: true})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", c.URL+"/v1/machines/mch_1/relay/v1/events/stream", nil)
	req.Header.Set("Authorization", "Bearer "+relayToken)
	start := time.Now()
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	line, err := bufio.NewReader(res.Body).ReadString('\n')
	if err != nil || line != "data: hello\n" {
		t.Fatalf("%q %v", line, err)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("first event took %v", d)
	}
}

func TestRelayRedirectsStayInside(t *testing.T) {
	m := newMachine(t)
	c := newRelayServer(t, provision.RelayTarget{BaseURL: m.URL, Token: machineToken, Simulated: true})
	auth := map[string]string{"Authorization": "Bearer " + relayToken}
	for path, want := range map[string]string{
		"go-home": "/v1/machines/mch_1/relay/studio/",
		"go-base": "/v1/machines/mch_1/relay/v1/health?x=1",
	} {
		res := send(t, "GET", c.URL+"/v1/machines/mch_1/relay/"+path, "", auth)
		if res.StatusCode != http.StatusFound || res.Header.Get("Location") != want {
			t.Errorf("%s: %d %q, want %q", path, res.StatusCode, res.Header.Get("Location"), want)
		}
	}
	if n := m.count(); n != 2 {
		t.Errorf("the relay followed a redirect: %d machine requests", n)
	}
	for loc, want := range map[string]string{
		"https://elsewhere.example/x": "https://elsewhere.example/x",
		"//elsewhere.example/x":       "//elsewhere.example/x",
		"relative":                    "relative",
		"http://1.2.3.4:8675":         "/v1/machines/m/relay/",
	} {
		if got := rewriteLocation(loc, "http://1.2.3.4:8675", "/v1/machines/m/relay"); got != want {
			t.Errorf("rewriteLocation(%q) = %q, want %q", loc, got, want)
		}
	}
}

func TestRelayRefusals(t *testing.T) {
	m := newMachine(t)
	cases := []struct {
		name   string
		target provision.RelayTarget
		code   string
	}{
		{"no address", provision.RelayTarget{Token: machineToken, Simulated: true}, v1api.CodeConflict},
		{"private IP", provision.RelayTarget{BaseURL: "http://10.0.0.1:8675", Token: machineToken}, v1api.CodeRemoteFailed},
		{"metadata IP", provision.RelayTarget{BaseURL: "http://169.254.169.254", Token: machineToken}, v1api.CodeRemoteFailed},
		{"loopback, not simulated", provision.RelayTarget{BaseURL: m.URL, Token: machineToken}, v1api.CodeRemoteFailed},
		{"simulated, not loopback", provision.RelayTarget{BaseURL: "http://10.0.0.1:8675", Token: machineToken, Simulated: true}, v1api.CodeRemoteFailed},
		{"a path in the base", provision.RelayTarget{BaseURL: m.URL + "/x", Token: machineToken, Simulated: true}, v1api.CodeRemoteFailed},
	}
	for _, c := range cases {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("GET", "/v1/machines/mch_1/relay/v1/system", nil)
		Serve(w, r, c.target, "v1/system", true)
		res := w.Result()
		if got := errCode(t, res); got != c.code || res.StatusCode != v1api.StatusOf(c.code) {
			t.Errorf("%s: %d %s, want %s", c.name, res.StatusCode, got, c.code)
		}
	}
	if m.count() != 0 {
		t.Error("a refused target was dialled")
	}

	// a simulated machine that is gone: 502
	gone := httptest.NewServer(http.NotFoundHandler())
	base := gone.URL
	gone.Close()
	w := httptest.NewRecorder()
	Serve(w, httptest.NewRequest("GET", "/v1/machines/mch_1/relay/v1/system", nil),
		provision.RelayTarget{MachineID: "mch_1", BaseURL: base, Simulated: true}, "v1/system", true)
	if res := w.Result(); res.StatusCode != 502 || errCode(t, res) != v1api.CodeRemoteFailed {
		t.Errorf("unreachable machine: %d", res.StatusCode)
	}
}

// A proxy hostname passes CheckEndpoint, so only netguard stands between it
// and a private address it resolves to: the dial is refused, and the client
// gets 502 with the blocked-address message, not "couldn't reach".
func TestRelayDialBlocked(t *testing.T) {
	const base = "https://abc-8675.proxy.runpod.net"
	if err := provision.CheckEndpoint("", base, false); err != nil {
		t.Fatalf("the proxy address should pass CheckEndpoint: %v", err)
	}
	var lookups atomic.Int32
	res := netguard.StaticResolver(map[string][]netip.Addr{"abc-8675.proxy.runpod.net": {netip.MustParseAddr("10.0.0.1")}})
	dial := res.Dial
	res.Dial = func(ctx context.Context, network, addr string) (net.Conn, error) {
		lookups.Add(1)
		return dial(ctx, network, addr)
	}
	oldOptions := guardOptions
	guardOptions = func(simulated bool) netguard.Options {
		o := oldOptions(simulated)
		o.Resolver = res
		return o
	}
	resetTransports := func() {
		transportsMu.Lock()
		clear(transports)
		transportsMu.Unlock()
	}
	resetTransports()
	t.Cleanup(func() { guardOptions = oldOptions; resetTransports() })

	w := httptest.NewRecorder()
	Serve(w, httptest.NewRequest("GET", "/v1/machines/mch_1/relay/v1/system", nil),
		provision.RelayTarget{MachineID: "mch_1", BaseURL: base, Token: machineToken}, "v1/system", true)
	r := w.Result()
	var p struct {
		Error struct{ Code, Message string } `json:"error"`
	}
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		t.Fatal(err)
	}
	if r.StatusCode != 502 || p.Error.Code != v1api.CodeRemoteFailed ||
		p.Error.Message != "the machine's address is not one the relay may dial" {
		t.Errorf("dial-time block: %d %+v", r.StatusCode, p.Error)
	}
	if lookups.Load() == 0 {
		t.Error("the hostname was never resolved: the refusal came before the dial")
	}
}

// Without the relaying server's token only the machine's public reads and signed
// links get through, and they carry no credentials.
func TestRelayPublicPassthrough(t *testing.T) {
	m := newMachine(t)
	c := newRelayServer(t, provision.RelayTarget{BaseURL: m.URL, Token: machineToken, Simulated: true})
	relay := c.URL + "/v1/machines/x/relay"

	res := send(t, "GET", relay+"/api/img/a/b.png", "", nil)
	if res.StatusCode != 201 {
		t.Fatalf("public read: %d", res.StatusCode)
	}
	if got := m.last(t); got.Path != "/api/img/a/b.png" || got.Header.Get("Authorization") != "" {
		t.Errorf("public read reached the machine as %+v", got)
	}

	n := m.count()
	for _, c := range []struct{ method, path string }{
		{"POST", "/api/img/a/b.png"},
		{"GET", "/v1/jobs"},
		{"GET", "/v1/files/content?root=training&path=a"},
	} {
		if res := send(t, c.method, relay+c.path, "", nil); res.StatusCode != 401 {
			t.Errorf("unauthenticated %s %s: %d", c.method, c.path, res.StatusCode)
		}
	}
	if m.count() != n {
		t.Error("an unauthenticated private request reached the machine")
	}

	res = send(t, "GET", relay+"/v1/files/content?root=training&path=a&exp=99&sig=machine-signed", "", nil)
	if res.StatusCode != 201 {
		t.Fatalf("signed link: %d", res.StatusCode)
	}
	if got := m.last(t); got.Path != "/v1/files/content" || got.Query != "root=training&path=a&exp=99&sig=machine-signed" ||
		got.Header.Get("Authorization") != "" {
		t.Errorf("signed link reached the machine as %+v", got)
	}

	// a wrong relay token is refused, never swapped for the machine's
	if res := send(t, "GET", relay+"/v1/jobs", "", map[string]string{"Authorization": "Bearer nope"}); res.StatusCode != 401 {
		t.Errorf("wrong token: %d", res.StatusCode)
	}
	// the right one gets the machine token on a private route
	send(t, "GET", relay+"/v1/jobs", "", map[string]string{"Authorization": "Bearer " + relayToken})
	if got := m.last(t); got.Path != "/v1/jobs" || got.Header.Get("Authorization") != "Bearer "+machineToken {
		t.Errorf("authenticated relay reached the machine as %+v", got)
	}
}
