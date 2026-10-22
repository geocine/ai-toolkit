// Package httpx holds the HTTP plumbing shared by every handler package:
// JSON responses shaped like NextResponse.json, the route registry the
// parity tests check against upstream's route.ts files, and auth.
package httpx

import (
	"crypto/subtle"
	"encoding/json"
	"log"
	"net/http"
	"os"
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

// Auth mirrors ui/src/middleware.ts and extends it to /v1 and /mcp: when
// AI_TOOLKIT_AUTH is set or any API key exists, those routes (except the
// public ones) need `Authorization: Bearer <token>`, where the token is
// AI_TOOLKIT_AUTH or an active API key. With neither, the server is open,
// like upstream. A signed download link (internal/links) stands in for the
// token on the GET it was signed for.
func Auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		want := os.Getenv("AI_TOOLKIT_AUTH")
		if !needsToken(r.Method, r.URL.Path) || (want == "" && !state.HasKeys()) {
			next.ServeHTTP(w, r)
			return
		}
		token := ""
		if parts := strings.Split(r.Header.Get("Authorization"), " "); len(parts) > 1 {
			token = parts[1]
		}
		if token == "" && links.Valid(r) {
			next.ServeHTTP(w, r)
			return
		}
		ok := token != "" && ((want != "" && subtle.ConstantTimeCompare([]byte(token), []byte(want)) == 1) ||
			state.VerifyKey(r.Context(), token))
		if !ok {
			if strings.HasPrefix(r.URL.Path, "/api/") {
				JSON(w, http.StatusUnauthorized, map[string]string{"error": "Unauthorized"})
			} else {
				JSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{
					"code": "unauthorized", "message": "missing or invalid bearer token"}})
			}
			return
		}
		next.ServeHTTP(w, r)
	})
}

func needsToken(method, path string) bool {
	if method == http.MethodGet || method == http.MethodHead {
		for _, p := range publicRoutes {
			if strings.HasPrefix(path, p) {
				return false
			}
		}
	}
	for _, p := range guarded {
		if strings.HasPrefix(path, p) {
			return true
		}
	}
	return false
}
