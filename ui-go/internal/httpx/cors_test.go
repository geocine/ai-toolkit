package httpx

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCORS(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) })
	h := CORS(ok)
	do := func(method, path, origin string, preflight bool) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, nil)
		if origin != "" {
			req.Header.Set("Origin", origin)
		}
		if preflight {
			req.Header.Set("Access-Control-Request-Method", "POST")
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}

	t.Setenv("AITK_CORS_ORIGINS", "")
	if rec := do("GET", "/v1/jobs", "http://studio.local", false); rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("CORS must be off by default")
	}

	t.Setenv("AITK_CORS_ORIGINS", "http://localhost:5174, https://studio.example.com/")
	rec := do("OPTIONS", "/v1/jobs", "https://studio.example.com", true)
	if rec.Code != http.StatusNoContent || rec.Header().Get("Access-Control-Allow-Origin") != "https://studio.example.com" ||
		rec.Header().Get("Access-Control-Allow-Headers") == "" {
		t.Fatalf("preflight: %d %v", rec.Code, rec.Header())
	}
	rec = do("GET", "/v1/jobs", "http://localhost:5174", false)
	if rec.Code != http.StatusTeapot || rec.Header().Get("Access-Control-Allow-Origin") != "http://localhost:5174" {
		t.Fatalf("simple request: %d %v", rec.Code, rec.Header())
	}
	if rec := do("GET", "/v1/jobs", "https://evil.example", false); rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("unlisted origin must not be allowed")
	}
	if rec := do("GET", "/index.html", "http://localhost:5174", false); rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("the frontend itself is not a CORS path")
	}

	t.Setenv("AITK_CORS_ORIGINS", "*")
	if rec := do("OPTIONS", "/mcp", "https://anything.example", true); rec.Code != http.StatusNoContent {
		t.Fatalf("wildcard preflight %d", rec.Code)
	}
}

// An https studio calling a studio API on http://127.0.0.1 gets Chrome's
// private-network preflight answered; other preflights do not carry it.
func TestCORSPrivateNetwork(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) })
	h := CORS(ok)
	preflight := func(origin string, private bool) *httptest.ResponseRecorder {
		req := httptest.NewRequest("OPTIONS", "/v1/providers", nil)
		req.Header.Set("Origin", origin)
		req.Header.Set("Access-Control-Request-Method", "GET")
		if private {
			req.Header.Set("Access-Control-Request-Private-Network", "true")
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}
	t.Setenv("AITK_CORS_ORIGINS", "https://studio.example")
	rec := preflight("https://studio.example", true)
	if rec.Code != http.StatusNoContent || rec.Header().Get("Access-Control-Allow-Private-Network") != "true" {
		t.Fatalf("private-network preflight: %d %v", rec.Code, rec.Header())
	}
	if !strings.Contains(rec.Header().Get("Access-Control-Expose-Headers"), "Retry-After") {
		t.Fatalf("Retry-After not exposed: %v", rec.Header())
	}
	if rec := preflight("https://studio.example", false); rec.Header().Get("Access-Control-Allow-Private-Network") != "" {
		t.Fatal("a plain preflight must not allow private network access")
	}
	if rec := preflight("https://evil.example", true); rec.Header().Get("Access-Control-Allow-Private-Network") != "" {
		t.Fatal("an unlisted origin must not get private network access")
	}
}

// A page on another site may not change anything here, even on a server
// without a token; the same origin, an allowed one, and reads still pass.
func TestCrossSiteWritesRefused(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) })
	h := CORS(ok)
	do := func(method, origin, host, fwd string) int {
		req := httptest.NewRequest(method, "/v1/jobs", nil)
		req.Host = host
		if origin != "" {
			req.Header.Set("Origin", origin)
		}
		if fwd != "" {
			req.Header.Set("X-Forwarded-Host", fwd)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code
	}
	t.Setenv("AITK_CORS_ORIGINS", "https://studio.example.com")
	t.Setenv("AI_TOOLKIT_AUTH", "")
	for _, c := range []struct {
		method, origin, host, fwd string
		want                      int
	}{
		{"POST", "https://evil.example", "pod-8675.proxy.runpod.net", "", http.StatusForbidden},
		{"DELETE", "http://attacker.local:8000", "localhost:8675", "", http.StatusForbidden},
		{"POST", "https://pod-8675.proxy.runpod.net", "pod-8675.proxy.runpod.net", "", http.StatusTeapot},
		{"POST", "https://pod-8675.proxy.runpod.net", "127.0.0.1:8675", "pod-8675.proxy.runpod.net", http.StatusTeapot},
		{"POST", "https://studio.example.com", "pod-8675.proxy.runpod.net", "", http.StatusTeapot},
		{"POST", "", "localhost:8675", "", http.StatusTeapot},
		{"GET", "https://evil.example", "localhost:8675", "", http.StatusTeapot},
	} {
		if got := do(c.method, c.origin, c.host, c.fwd); got != c.want {
			t.Errorf("%s from %q to %q: %d, want %d", c.method, c.origin, c.host, got, c.want)
		}
	}
	// with a token the page cannot authenticate anyway: left to auth
	t.Setenv("AI_TOOLKIT_AUTH", "secret")
	if got := do("POST", "https://evil.example", "localhost:8675", ""); got != http.StatusTeapot {
		t.Errorf("with a token: %d", got)
	}
}

func TestIsOpen(t *testing.T) {
	t.Setenv("AI_TOOLKIT_AUTH", "")
	if !IsOpen() {
		t.Fatal("no token and no keys: open")
	}
	t.Setenv("AI_TOOLKIT_AUTH", "secret")
	if IsOpen() {
		t.Fatal("a token closes the server")
	}
}
