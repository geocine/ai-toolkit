package httpx

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func TestNeedsToken(t *testing.T) {
	for _, c := range []struct {
		method, target string
		want           bool
	}{
		{"GET", "/api/img/a.png", false},
		{"HEAD", "/api/files/x.safetensors", false},
		{"GET", "/v1/health", false},
		{"POST", "/api/img/caption", true},
		{"POST", "/api/img/upload", true},
		{"POST", "/api/img/delete", true},
		{"POST", "/api/files/delete", true},
		{"GET", "/v1/jobs", true},
		{"POST", "/mcp", true},
		{"GET", "/studio/", false},
		// the relay passes the machine's public reads and signed links through
		{"GET", "/v1/machines/mch_1/relay/api/img/a/b.png", false},
		{"HEAD", "/v1/machines/mch_1/relay/api/audio/art/x.jpg", false},
		{"GET", "/v1/machines/mch_1/relay/v1/health", false},
		{"GET", "/v1/machines/mch_1/relay/v1/files/content?root=training&path=a&exp=1&sig=abc", false},
		{"GET", "/v1/machines/mch_1/relay/v1/jobs/j1/bundle?sig=abc", false},
		{"POST", "/v1/machines/mch_1/relay/api/img/a.png", true},
		{"GET", "/v1/machines/mch_1/relay/v1/jobs", true},
		{"GET", "/v1/machines/mch_1/relay/v1/files/content?root=training&path=a", true},
		{"GET", "/v1/machines/mch_1/relay/v1/keys?sig=abc", true},
		{"GET", "/v1/machines/mch_1/relay/api/img/../../v1/jobs", true},
		{"GET", "/v1/machines/mch_1/relay", true},
	} {
		r := httptest.NewRequest(c.method, c.target, nil)
		if got := needsToken(r); got != c.want {
			t.Errorf("%s %s: needsToken = %v, want %v", c.method, c.target, got, c.want)
		}
	}
}

// Authenticated is set only when Auth matched this server's own credentials,
// or the server is open and the peer is this machine.
func TestAuthenticated(t *testing.T) {
	dir := testutil.TempDir(t)
	if err := state.Open(filepath.Join(dir, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close(); state.DB = nil })

	var seen *bool
	h := Auth(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ok := Authenticated(r)
		seen = &ok
	}))
	do := func(target, remote, token string, headers ...string) (code int, authed, reached bool) {
		seen = nil
		r := httptest.NewRequest("GET", target, nil)
		r.RemoteAddr = remote
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		for i := 0; i+1 < len(headers); i += 2 {
			r.Header.Set(headers[i], headers[i+1])
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if seen == nil {
			return w.Code, false, false
		}
		return w.Code, *seen, true
	}
	const local, remote = "127.0.0.1:5000", "198.51.100.7:5000"
	const img = "/v1/machines/mch_1/relay/api/img/a.png"

	t.Setenv("AI_TOOLKIT_AUTH", "")
	if _, a, ok := do("/v1/jobs", local, ""); !ok || !a {
		t.Error("open server, loopback peer: authenticated")
	}
	if _, a, ok := do("/v1/jobs", remote, ""); !ok || a {
		t.Error("open server, remote peer: not authenticated")
	}
	if _, a, ok := do("/v1/jobs", local, "", "X-Forwarded-For", "203.0.113.9"); !ok || a {
		t.Error("open server behind a proxy: not authenticated")
	}

	t.Setenv("AI_TOOLKIT_AUTH", "admin-token-123")
	if _, a, ok := do("/v1/jobs", remote, "admin-token-123"); !ok || !a {
		t.Error("the admin token is authenticated")
	}
	if code, _, ok := do("/v1/jobs", local, ""); ok || code != 401 {
		t.Errorf("no token on a guarded route: %d", code)
	}
	if code, _, ok := do("/v1/jobs", local, "wrong"); ok || code != 401 {
		t.Errorf("wrong token on a guarded route: %d", code)
	}
	if _, a, ok := do(img, remote, ""); !ok || a {
		t.Error("a public relay read passes, unauthenticated")
	}
	if _, a, ok := do(img, remote, "admin-token-123"); !ok || !a {
		t.Error("a public relay read with the token is authenticated")
	}
	if _, a, ok := do("/v1/machines/mch_1/relay/v1/files/content?root=training&path=a&exp=1&sig=abc", remote, ""); !ok || a {
		t.Error("a signed relay link passes, unauthenticated")
	}
	if code, _, ok := do("/v1/machines/mch_1/relay/v1/jobs", remote, ""); ok || code != 401 {
		t.Errorf("an unauthenticated relay of a private route: %d", code)
	}

	_, key, err := state.CreateKey(context.Background(), "agent")
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("AI_TOOLKIT_AUTH", "")
	if _, a, ok := do("/v1/jobs", remote, key); !ok || !a {
		t.Error("an API key is authenticated")
	}
	if code, _, ok := do("/v1/jobs", local, ""); ok || code != 401 {
		t.Errorf("with a key the server is not open, even from loopback: %d", code)
	}
}
