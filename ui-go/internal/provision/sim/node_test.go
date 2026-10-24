package sim

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/links"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

const nodeToken = "aitk_node_token_0001"

// stubInner answers like this server below Auth and reports the
// Authorization header it was handed.
func stubInner() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/health", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"ok":true,"version":"test"}`))
	})
	echo := func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{"auth": r.Header.Get("Authorization")})
	}
	for _, p := range []string{"GET /v1/system", "GET /v1/files/content", "GET /api/img/", "POST /api/img/upload", "GET /{$}", "POST /mcp"} {
		mux.HandleFunc(p, echo)
	}
	return mux
}

func startNode(t *testing.T, clock mockkit.Clock, spec mockkit.NodeSpec) string {
	t.Helper()
	base, stop := factory(stubInner(), clock)(spec)
	if !strings.HasPrefix(base, "http://127.0.0.1:") {
		t.Fatalf("node base %q is not loopback", base)
	}
	t.Cleanup(stop)
	return base
}

func get(t *testing.T, url string, header map[string]string) (*http.Response, string) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	for k, v := range header {
		req.Header.Set(k, v)
	}
	return do(t, req)
}

func do(t *testing.T, req *http.Request) (*http.Response, string) {
	t.Helper()
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	return res, string(b)
}

func spec(env map[string]string) mockkit.NodeSpec {
	return mockkit.NodeSpec{Kind: "runpod", InstanceID: "pod1", Env: env, GPU: "RTX 4090", GPUCount: 1, VRAMGB: 24}
}

func TestNodeAuth(t *testing.T) {
	base := startNode(t, mockkit.RealClock{}, spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken}))

	res, body := get(t, base+"/v1/system", map[string]string{"Authorization": "Bearer wrong"})
	if res.StatusCode != 401 || strings.TrimSpace(body) != `{"error":{"code":"unauthorized","message":"missing or invalid bearer token"}}` {
		t.Fatalf("wrong token: %d %s", res.StatusCode, body)
	}
	if res, _ := get(t, base+"/v1/system", nil); res.StatusCode != 401 {
		t.Fatalf("no token: %d", res.StatusCode)
	}
	res, body = get(t, base+"/v1/system", map[string]string{"Authorization": "Bearer " + nodeToken})
	if res.StatusCode != 200 || !strings.Contains(body, `"auth":""`) {
		t.Fatalf("right token: %d %s (Authorization must not reach inner)", res.StatusCode, body)
	}
	if res, body := get(t, base+"/v1/health", nil); res.StatusCode != 200 || !strings.Contains(body, `"version":"test"`) {
		t.Fatalf("public health: %d %s", res.StatusCode, body)
	}
	// health is public only for GET
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/health", nil)
	if res, _ := do(t, req); res.StatusCode != 401 {
		t.Fatalf("POST health without a token: %d", res.StatusCode)
	}
}

// The node gates paths like httpx.Auth: media reads, the SPA and signed
// links pass without a token; writes and the rest of /api need it, with the
// upstream /api 401 body.
func TestNodeAuthPublicRoutes(t *testing.T) {
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close(); state.DB = nil; links.Reset() })
	base := startNode(t, mockkit.RealClock{}, spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken}))

	if res, body := get(t, base+"/api/img/x.png", nil); res.StatusCode != 200 {
		t.Fatalf("GET /api/img/x.png without a token: %d %s", res.StatusCode, body)
	}
	head, _ := http.NewRequest(http.MethodHead, base+"/v1/health", nil)
	if res, _ := do(t, head); res.StatusCode != 200 {
		t.Fatalf("HEAD /v1/health without a token: %d", res.StatusCode)
	}
	if res, _ := get(t, base+"/", nil); res.StatusCode != 200 {
		t.Fatalf("GET / (the SPA) without a token: %d", res.StatusCode)
	}
	up, _ := http.NewRequest(http.MethodPost, base+"/api/img/upload", nil)
	res, body := do(t, up)
	if res.StatusCode != 401 || strings.TrimSpace(body) != `{"error":"Unauthorized"}` {
		t.Fatalf("POST /api/img/upload without a token: %d %s", res.StatusCode, body)
	}
	up.Header.Set("Authorization", "Bearer "+nodeToken)
	if res, _ := do(t, up); res.StatusCode != 200 {
		t.Fatalf("POST /api/img/upload with the token: %d", res.StatusCode)
	}

	signed, _, err := links.Sign(context.Background(), "/v1/files/content?root=training&path=a.txt", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if res, body := get(t, base+signed, nil); res.StatusCode != 200 {
		t.Fatalf("signed link without a token: %d %s", res.StatusCode, body)
	}
	if res, _ := get(t, base+strings.Replace(signed, "a.txt", "b.txt", 1), nil); res.StatusCode != 401 {
		t.Fatalf("a link edited after signing: %d", res.StatusCode)
	}
	if res, _ := get(t, base+"/v1/files/content?root=training&path=a.txt", nil); res.StatusCode != 401 {
		t.Fatalf("unsigned /v1/files/content without a token: %d", res.StatusCode)
	}
}

// /mcp reaches inner with this server's own token, because code mode replays
// it to this server's outer Auth; every other path gets none.
func TestNodeMCPCarriesServerAuth(t *testing.T) {
	t.Setenv("AI_TOOLKIT_AUTH", "server_token_0001")
	base := startNode(t, mockkit.RealClock{}, spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken}))
	req, _ := http.NewRequest(http.MethodPost, base+"/mcp", strings.NewReader(`{}`))
	if res, _ := do(t, req); res.StatusCode != 401 {
		t.Fatalf("/mcp without the node token: %d", res.StatusCode)
	}
	req, _ = http.NewRequest(http.MethodPost, base+"/mcp", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer "+nodeToken)
	if res, body := do(t, req); res.StatusCode != 200 || !strings.Contains(body, `"auth":"Bearer server_token_0001"`) {
		t.Fatalf("/mcp with the node token: %d %s", res.StatusCode, body)
	}
	res, body := get(t, base+"/v1/system", map[string]string{"Authorization": "Bearer " + nodeToken})
	if res.StatusCode != 200 || !strings.Contains(body, `"auth":""`) {
		t.Fatalf("/v1/system: %d %s (only /mcp gets this server's token)", res.StatusCode, body)
	}
}

// An open node refuses cross-site writes the way httpx.CORS does on an open
// server, so a page in the user's browser cannot drive this server through it.
func TestNodeOpenRefusesCrossSiteWrites(t *testing.T) {
	open := spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken, "AITK_CORS_ORIGINS": "https://studio.example"})
	open.OpenNode = true
	base := startNode(t, mockkit.RealClock{}, open)
	post := func(origin string) (*http.Response, string) {
		req, _ := http.NewRequest(http.MethodPost, base+"/mcp", strings.NewReader(`{}`))
		req.Header.Set("Content-Type", "text/plain")
		if origin != "" {
			req.Header.Set("Origin", origin)
		}
		return do(t, req)
	}
	res, body := post("https://evil.example")
	if res.StatusCode != 403 || strings.TrimSpace(body) != `{"error":{"code":"forbidden","message":"cross-origin request from https://evil.example (allow it with AITK_CORS_ORIGINS)"}}` {
		t.Fatalf("cross-site POST to an open node: %d %s", res.StatusCode, body)
	}
	for _, origin := range []string{"", "https://studio.example", base} {
		if res, body := post(origin); res.StatusCode != 200 {
			t.Fatalf("POST with Origin %q: %d %s", origin, res.StatusCode, body)
		}
	}
	if res, _ := get(t, base+"/v1/system", map[string]string{"Origin": "https://evil.example"}); res.StatusCode != 200 {
		t.Fatalf("cross-site GET: %d", res.StatusCode)
	}
	// a node with a token needs no such check: the page has no token
	closed := startNode(t, mockkit.RealClock{}, spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken}))
	req, _ := http.NewRequest(http.MethodPost, closed+"/mcp", strings.NewReader(`{}`))
	req.Header.Set("Origin", "https://evil.example")
	if res, _ := do(t, req); res.StatusCode != 401 {
		t.Fatalf("cross-site POST to a node with a token: %d", res.StatusCode)
	}
}

func TestNodeOpen(t *testing.T) {
	open := spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken})
	open.OpenNode = true
	base := startNode(t, mockkit.RealClock{}, open)
	if res, _ := get(t, base+"/v1/system", nil); res.StatusCode != 200 {
		t.Fatalf("open node without a token: %d", res.StatusCode)
	}
	// a container with no AI_TOOLKIT_AUTH is open, like the real server
	base = startNode(t, mockkit.RealClock{}, spec(map[string]string{}))
	if res, _ := get(t, base+"/v1/system", nil); res.StatusCode != 200 {
		t.Fatalf("node without a token env: %d", res.StatusCode)
	}
}

func TestNodeCORS(t *testing.T) {
	base := startNode(t, mockkit.RealClock{}, spec(map[string]string{
		"AI_TOOLKIT_AUTH": nodeToken, "AITK_CORS_ORIGINS": "https://studio.example, http://localhost:5173/"}))
	auth := "Bearer " + nodeToken

	res, _ := get(t, base+"/v1/system", map[string]string{"Authorization": auth, "Origin": "https://studio.example"})
	if res.Header.Get("Access-Control-Allow-Origin") != "https://studio.example" {
		t.Fatalf("recorded origin: ACAO %q", res.Header.Get("Access-Control-Allow-Origin"))
	}
	res, _ = get(t, base+"/v1/system", map[string]string{"Authorization": auth, "Origin": "http://localhost:5173"})
	if res.Header.Get("Access-Control-Allow-Origin") != "http://localhost:5173" {
		t.Fatalf("trailing slash in the list: ACAO %q", res.Header.Get("Access-Control-Allow-Origin"))
	}
	res, _ = get(t, base+"/v1/system", map[string]string{"Authorization": auth, "Origin": "https://evil.example"})
	if res.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("an origin outside the list got ACAO %q", res.Header.Get("Access-Control-Allow-Origin"))
	}

	pre, _ := http.NewRequest(http.MethodOptions, base+"/v1/jobs", nil)
	pre.Header.Set("Origin", "https://studio.example")
	pre.Header.Set("Access-Control-Request-Method", "POST")
	res, _ = do(t, pre)
	if res.StatusCode != 204 || !strings.Contains(res.Header.Get("Access-Control-Allow-Headers"), "Authorization") ||
		!strings.Contains(res.Header.Get("Access-Control-Allow-Methods"), "PATCH") {
		t.Fatalf("preflight: %d %v", res.StatusCode, res.Header)
	}
	pre.Header.Set("Origin", "https://evil.example")
	if res, _ := do(t, pre); res.StatusCode == 204 || res.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("preflight from another origin: %d %v", res.StatusCode, res.Header)
	}

	star := startNode(t, mockkit.RealClock{}, spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken, "AITK_CORS_ORIGINS": "*"}))
	res, _ = get(t, star+"/v1/health", map[string]string{"Origin": "https://any.example"})
	if res.Header.Get("Access-Control-Allow-Origin") != "https://any.example" {
		t.Fatalf("*: ACAO %q", res.Header.Get("Access-Control-Allow-Origin"))
	}
}

func TestNodeStartingThenUp(t *testing.T) {
	clock := mockkit.NewFakeClock(time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC))
	s := spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken})
	s.AppDelay = 4 * time.Second
	base := startNode(t, clock, s)

	res, body := get(t, base+"/v1/health", nil)
	if res.StatusCode != 503 || strings.TrimSpace(body) != `{"error":{"code":"internal","message":"starting"}}` {
		t.Fatalf("during AppDelay: %d %s", res.StatusCode, body)
	}
	if res, _ := get(t, base+"/v1/system", map[string]string{"Authorization": "Bearer " + nodeToken}); res.StatusCode != 503 {
		t.Fatalf("system during AppDelay: %d", res.StatusCode)
	}
	clock.Advance(4 * time.Second)
	if res, _ := get(t, base+"/v1/health", nil); res.StatusCode != 200 {
		t.Fatalf("after AppDelay: %d", res.StatusCode)
	}
}

func TestNodeStopClosesListener(t *testing.T) {
	base, stop := factory(stubInner(), mockkit.RealClock{})(spec(nil))
	if res, _ := get(t, base+"/v1/health", nil); res.StatusCode != 200 {
		t.Fatalf("health: %d", res.StatusCode)
	}
	stop()
	stop() // idempotent
	if _, err := http.Get(base + "/v1/health"); err == nil {
		t.Fatal("the node still answers after stop")
	}
}

type nodeCudaCheck struct {
	OK       bool     `json:"ok"`
	Torch    string   `json:"torch"`
	ArchList []string `json:"arch_list"`
	Devices  []struct {
		Index      int    `json:"index"`
		Name       string `json:"name"`
		Capability string `json:"capability"`
		Driver     string `json:"driver"`
	} `json:"devices"`
	Error   *string `json:"error"`
	Failure *string `json:"failure"`
}

// The node answers GET /v1/system/cuda itself (inner would run this host's
// Python), behind its token, with one device per GPU and the failure the
// fault asks for.
func TestNodeCUDA(t *testing.T) {
	auth := map[string]string{"Authorization": "Bearer " + nodeToken}
	for _, tc := range []struct {
		fault, failure, msg, driver string
	}{
		{mockkit.CUDAOK, "", "", "580.95.05"},
		{mockkit.CUDANoKernel, "no_kernel_image", "no kernel image is available", "580.95.05"},
		{mockkit.CUDAOldDriver, "driver_too_old", "driver on your system is too old", "535.104.05"},
	} {
		s := spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken})
		s.GPUCount = 2
		s = s.WithFaults(mockkit.Faults{CUDA: tc.fault})
		base := startNode(t, mockkit.RealClock{}, s)
		if res, _ := get(t, base+"/v1/system/cuda", nil); res.StatusCode != 401 {
			t.Fatalf("%q: no token: %d", tc.fault, res.StatusCode)
		}
		res, body := get(t, base+"/v1/system/cuda?refresh=1", auth)
		var c nodeCudaCheck
		if res.StatusCode != 200 || json.Unmarshal([]byte(body), &c) != nil {
			t.Fatalf("%q: %d %s", tc.fault, res.StatusCode, body)
		}
		if c.OK != (tc.fault == mockkit.CUDAOK) || c.Torch == "" || len(c.ArchList) == 0 || len(c.Devices) != 2 {
			t.Fatalf("%q: %s", tc.fault, body)
		}
		for i, d := range c.Devices {
			if d.Index != i || d.Name != "RTX 4090" || d.Capability != "8.9" || d.Driver != tc.driver {
				t.Fatalf("%q: device %d %+v", tc.fault, i, d)
			}
		}
		if tc.failure == "" {
			if c.Failure != nil || c.Error != nil {
				t.Fatalf("ok node: %s", body)
			}
			continue
		}
		if c.Failure == nil || *c.Failure != tc.failure || c.Error == nil || !strings.Contains(*c.Error, tc.msg) {
			t.Fatalf("%q: %s", tc.fault, body)
		}
	}
	// not while the app is still starting
	clock := mockkit.NewFakeClock(time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC))
	s := spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken})
	s.AppDelay = time.Second
	if res, _ := get(t, startNode(t, clock, s)+"/v1/system/cuda", auth); res.StatusCode != 503 {
		t.Fatalf("during AppDelay: %d", res.StatusCode)
	}
}

// /v1/health reports the build the launch ref names, the simulated
// registry's build for a moving tag or a digest, and an older one with
// Faults.StaleImage; the rest of inner's answer is kept.
func TestNodeHealthVersion(t *testing.T) {
	for _, tc := range []struct {
		image string
		stale bool
		want  string
	}{
		{"", false, "test"}, // no launch ref: inner's own version
		{"ghcr.io/geocine/aitoolkit-go:f53a407", false, "f53a407"},
		{"ghcr.io/geocine/aitoolkit-go:c126c12", false, "c126c12"},
		{"ghcr.io/geocine/aitoolkit-go:latest", false, registrymock.Revision},
		{"ghcr.io/geocine/aitoolkit-go@sha256:" + strings.Repeat("ab", 32), false, registrymock.Revision},
		{"ghcr.io/geocine/aitoolkit-go:f53a407", true, StaleVersion},
		{"", true, StaleVersion},
	} {
		s := spec(map[string]string{"AI_TOOLKIT_AUTH": nodeToken})
		s.Image = tc.image
		s = s.WithFaults(mockkit.Faults{StaleImage: tc.stale})
		res, body := get(t, startNode(t, mockkit.RealClock{}, s)+"/v1/health", nil)
		var h map[string]any
		if res.StatusCode != 200 || json.Unmarshal([]byte(body), &h) != nil || h["version"] != tc.want || h["ok"] != true {
			t.Fatalf("%q stale=%v: %d %s, want version %q", tc.image, tc.stale, res.StatusCode, body, tc.want)
		}
	}
	if StaleVersion == registrymock.Revision {
		t.Fatal("StaleVersion must differ from the published build")
	}
}
