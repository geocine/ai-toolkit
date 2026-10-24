package sim

import (
	"crypto/subtle"
	"errors"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/links"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Factory starts one simulated GPU server per running instance: a loopback
// listener that answers like the aitk image would (its own token and CORS
// from the instance env) and serves through inner, this server's handler
// below Auth. A simulated machine is therefore this server under another
// address and token.
func Factory(inner http.Handler) mockkit.NodeFactory {
	return factory(inner, mockkit.RealClock{})
}

// factory takes the clock that times AppDelay (tests pass a FakeClock).
func factory(inner http.Handler, clock mockkit.Clock) mockkit.NodeFactory {
	return func(n mockkit.NodeSpec) (string, func()) {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			log.Printf("simulate: node for %s %s: %v", n.Kind, n.InstanceID, err)
			return "", func() {}
		}
		srv := &http.Server{Handler: nodeHandler(n, inner, clock), ReadHeaderTimeout: 10 * time.Second}
		go func() {
			if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Printf("simulate: node for %s %s: %v", n.Kind, n.InstanceID, err)
			}
		}()
		var once sync.Once
		return "http://" + ln.Addr().String(), func() { once.Do(func() { _ = srv.Close() }) }
	}
}

// nodeHandler is the simulated aitk server in front of inner, gated the way
// httpx.CORS and httpx.Auth gate the real one, with the token and origins
// from the instance env instead of this process's:
//   - CORS from Env["AITK_CORS_ORIGINS"] (exact origin or "*"), preflights
//     answered with httpx.CORS's allow lists;
//   - an open node (Faults.OpenNode, or no token in the env, as the real
//     server) refuses cross-site writes with httpx.CORS's 403;
//   - 503 "starting" for AppDelay after the node starts;
//   - /api/, /v1/ and /mcp need Bearer Env["AI_TOOLKIT_AUTH"], except GET and
//     HEAD under /api/img/, /api/files/, /api/audio/art/ and /v1/health, and
//     a signed link (internal/links) on the GET it was signed for; every
//     other path (the SPA) is public;
//   - Authorization is stripped before inner, which runs below Auth. On /mcp
//     it becomes this server's AI_TOOLKIT_AUTH instead, because code mode
//     replays it to this server's outer handler (CORS + Auth); the node
//     already grants everything below Auth, so that escalates nothing.
func nodeHandler(n mockkit.NodeSpec, inner http.Handler, clock mockkit.Clock) http.Handler {
	readyAt := clock.Now().Add(n.AppDelay)
	token := n.Env["AI_TOOLKIT_AUTH"]
	origins := n.Env["AITK_CORS_ORIGINS"]
	open := n.OpenNode || token == ""
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && corsPath(r.URL.Path) && unsafeMethod(r.Method) && open && !sameOrigin(origin, r) && !originAllowed(origins, origin) {
			v1api.JSON(w, http.StatusForbidden, map[string]any{"error": map[string]string{
				"code": "forbidden", "message": "cross-origin request from " + origin + " (allow it with AITK_CORS_ORIGINS)"}})
			return
		}
		if origin != "" && corsPath(r.URL.Path) && originAllowed(origins, origin) {
			h := w.Header()
			h.Set("Access-Control-Allow-Origin", origin)
			h.Add("Vary", "Origin")
			h.Set("Access-Control-Expose-Headers", "Content-Disposition, Content-Range, Content-Length, ETag, Accept-Ranges")
			if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
				h.Set("Access-Control-Allow-Methods", "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS")
				h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Content-Range, Last-Event-ID, Range, If-None-Match, Accept, Mcp-Session-Id, Mcp-Protocol-Version")
				h.Set("Access-Control-Max-Age", "600")
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}
		if clock.Now().Before(readyAt) {
			v1api.JSON(w, http.StatusServiceUnavailable, map[string]any{"error": map[string]string{"code": v1api.CodeInternal, "message": "starting"}})
			return
		}
		if !open && needsToken(r.Method, r.URL.Path) && !bearerOK(r, token) && !(bearer(r) == "" && links.Valid(r)) {
			if strings.HasPrefix(r.URL.Path, "/api/") {
				v1api.JSON(w, http.StatusUnauthorized, map[string]string{"error": "Unauthorized"})
			} else {
				v1api.Fail(w, v1api.CodeUnauthorized, "missing or invalid bearer token")
			}
			return
		}
		r = r.Clone(r.Context())
		r.Header.Del("Authorization")
		if ctl := os.Getenv("AI_TOOLKIT_AUTH"); ctl != "" && strings.HasPrefix(r.URL.Path, "/mcp") {
			r.Header.Set("Authorization", "Bearer "+ctl)
		}
		inner.ServeHTTP(w, r)
	})
}

func bearerOK(r *http.Request, want string) bool {
	got := bearer(r)
	return got != "" && subtle.ConstantTimeCompare([]byte(got), []byte(want)) == 1
}

// bearer is the token as httpx.Auth reads it: the second word of
// Authorization ("" for none).
func bearer(r *http.Request) string {
	if parts := strings.Split(r.Header.Get("Authorization"), " "); len(parts) > 1 {
		return parts[1]
	}
	return ""
}

// needsToken, corsPath, unsafeMethod, sameOrigin and originAllowed mirror
// internal/httpx (Auth and CORS), with the origins from the instance env
// instead of this process's.
var (
	publicRoutes = []string{"/api/img/", "/api/files/", "/api/audio/art/", "/v1/health"}
	guarded      = []string{"/api/", "/v1/", "/mcp"}
)

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

func corsPath(p string) bool {
	return strings.HasPrefix(p, "/api/") || strings.HasPrefix(p, "/v1/") || p == "/mcp"
}

func unsafeMethod(m string) bool {
	return m != http.MethodGet && m != http.MethodHead && m != http.MethodOptions
}

func sameOrigin(origin string, r *http.Request) bool {
	u, err := url.Parse(origin)
	if err != nil || u.Host == "" {
		return false
	}
	if strings.EqualFold(u.Host, r.Host) {
		return true
	}
	for _, h := range strings.Split(r.Header.Get("X-Forwarded-Host"), ",") {
		if h = strings.TrimSpace(h); h != "" && strings.EqualFold(u.Host, h) {
			return true
		}
	}
	return false
}

func originAllowed(list, origin string) bool {
	for _, o := range strings.Split(list, ",") {
		o = strings.TrimRight(strings.TrimSpace(o), "/")
		if o == "*" || (o != "" && strings.EqualFold(o, origin)) {
			return true
		}
	}
	return false
}
