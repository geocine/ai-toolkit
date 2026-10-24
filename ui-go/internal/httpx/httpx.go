// Package httpx holds the HTTP plumbing shared by every handler package:
// JSON responses shaped like NextResponse.json, the route registry the
// parity tests check against upstream's route.ts files, and auth.
package httpx

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/geocine/aitoolkit/internal/links"
	"github.com/geocine/aitoolkit/internal/state"
)

// JSON writes v like NextResponse.json(v, {status}).
func JSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		log.Printf("httpx: encode response: %v", err)
	}
}

// OK writes v with status 200.
func OK(w http.ResponseWriter, v any) { JSON(w, http.StatusOK, v) }

// Error writes {"error": msg} with status.
func Error(w http.ResponseWriter, status int, msg string) {
	JSON(w, status, map[string]string{"error": msg})
}

// Internal is an error the Next route did not catch: Next answers those with
// a bare 500. Logged here so the cause is not lost.
func Internal(w http.ResponseWriter, r *http.Request, err error) {
	log.Printf("%s %s: %v", r.Method, r.URL.Path, err)
	w.WriteHeader(http.StatusInternalServerError)
}

// DecodeJSON reads the request body into v (request.json()).
func DecodeJSON(r *http.Request, v any) error {
	return json.NewDecoder(r.Body).Decode(v)
}

var (
	routesMu sync.Mutex
	routes   []string
)

// Handle registers h on mux for pattern ("GET /api/jobs/{jobID}/start") and
// records the pattern for the route-coverage test.
func Handle(mux *http.ServeMux, pattern string, h http.HandlerFunc) {
	routesMu.Lock()
	routes = append(routes, pattern)
	routesMu.Unlock()
	mux.HandleFunc(pattern, h)
}

// Routes returns every registered pattern, sorted.
func Routes() []string {
	routesMu.Lock()
	defer routesMu.Unlock()
	out := append([]string(nil), routes...)
	sort.Strings(out)
	return out
}

// Public routes are fetched by plain <img>/<audio> tags, which cannot send
// the bearer token. Only reads are public: the writes under the same
// prefixes (POST /api/img/caption, /api/img/upload, /api/img/delete,
// /api/files/delete) need the token like any other route.
var publicRoutes = []string{"/api/img/", "/api/files/", "/api/audio/art/", "/v1/health"}

// guarded are the path prefixes that need a token when auth is on.
var guarded = []string{"/api/", "/v1/", "/mcp"}

// relayRoute matches a GPU server relayed through this one
// (/v1/machines/{id}/relay/{inner}); group 1 is the inner path.
var relayRoute = regexp.MustCompile(`^/v1/machines/[^/]+/relay(/.*)$`)

type authKey struct{}

// Authenticated reports whether Auth let r through on this server's own
// credentials (AI_TOOLKIT_AUTH or an API key), or the server is open and r
// came from this machine. A public read or a signed link is not
// authenticated: the relay forwards those with no credentials.
func Authenticated(r *http.Request) bool {
	ok, _ := r.Context().Value(authKey{}).(bool)
	return ok
}

func withAuthenticated(r *http.Request) *http.Request {
	return r.WithContext(context.WithValue(r.Context(), authKey{}, true))
}

// Auth mirrors ui/src/middleware.ts and extends it to /v1 and /mcp: when
// AI_TOOLKIT_AUTH is set or any API key exists, those routes (except the
// public ones) need `Authorization: Bearer <token>`, where the token is
// AI_TOOLKIT_AUTH or an active API key. With neither, the server is open,
// like upstream. A signed download link (internal/links) stands in for the
// token on the GET it was signed for. Requests that matched a token (or, on
// an open server, came from this machine) are marked for Authenticated.
func Auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		want := os.Getenv("AI_TOOLKIT_AUTH")
		if want == "" && !state.HasKeys() {
			if loopbackPeer(r) {
				r = withAuthenticated(r)
			}
			next.ServeHTTP(w, r)
			return
		}
		token := ""
		if parts := strings.Split(r.Header.Get("Authorization"), " "); len(parts) > 1 {
			token = parts[1]
		}
		if token != "" && ((want != "" && subtle.ConstantTimeCompare([]byte(token), []byte(want)) == 1) ||
			state.VerifyKey(r.Context(), token)) {
			next.ServeHTTP(w, withAuthenticated(r))
			return
		}
		if !needsToken(r) || (token == "" && links.Valid(r)) {
			next.ServeHTTP(w, r)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/") {
			JSON(w, http.StatusUnauthorized, map[string]string{"error": "Unauthorized"})
		} else {
			JSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{
				"code": "unauthorized", "message": "missing or invalid bearer token"}})
		}
	})
}

func needsToken(r *http.Request) bool {
	p := r.URL.Path
	if r.Method == http.MethodGet || r.Method == http.MethodHead {
		if publicRoute(p) || publicRelay(r) {
			return false
		}
	}
	for _, g := range guarded {
		if strings.HasPrefix(p, g) {
			return true
		}
	}
	return false
}

func publicRoute(p string) bool {
	for _, pr := range publicRoutes {
		if strings.HasPrefix(p, pr) {
			return true
		}
	}
	return false
}

// publicRelay: a GET/HEAD through the relay to one of the machine's own public
// routes, or carrying a signature for a path a link may point at. The relay
// forwards it with no credentials and the machine checks its route or its
// own signature, so passing it here grants nothing. A path with dot segments
// is never public (the router would redirect it elsewhere).
func publicRelay(r *http.Request) bool {
	p := r.URL.Path
	m := relayRoute.FindStringSubmatch(p)
	if m == nil || path.Clean(p) != p {
		return false
	}
	if publicRoute(m[1]) {
		return true
	}
	if r.URL.Query().Get("sig") == "" {
		return false
	}
	em := relayRoute.FindStringSubmatch(r.URL.EscapedPath())
	return em != nil && links.Allowed(em[1])
}

// loopbackPeer is true for a direct connection from this machine: a loopback
// RemoteAddr and no proxy header saying the request came from elsewhere.
func loopbackPeer(r *http.Request) bool {
	for _, h := range []string{"X-Forwarded-For", "X-Real-Ip", "Forwarded"} {
		if r.Header.Get(h) != "" {
			return false
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
