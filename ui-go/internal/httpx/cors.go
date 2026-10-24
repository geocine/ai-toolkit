package httpx

import (
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/geocine/aitoolkit/internal/state"
)

// CORS lets browser apps on other origins (the studio, a desktop app's
// webview, ...) call /api, /v1 and /mcp. Allowed origins come from
// AITK_CORS_ORIGINS: a comma list of origins, or "*" for any. Unset means
// same-origin only, like upstream. Preflights are answered here, before auth,
// because browsers send them without the Authorization header.
func CORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && corsPath(r.URL.Path) && unsafeMethod(r.Method) && open() && !sameOrigin(origin, r) && !originAllowed(origin) {
			// a page on another site posting here (CSRF): browsers send such
			// "simple" requests without a preflight, and a server without a
			// token would act on them. With a token they cannot (it is a
			// header the page does not have), so only an open server checks.
			JSON(w, http.StatusForbidden, map[string]any{"error": map[string]string{
				"code": "forbidden", "message": "cross-origin request from " + origin + " (allow it with AITK_CORS_ORIGINS)"}})
			return
		}
		if origin == "" || !corsPath(r.URL.Path) || !originAllowed(origin) {
			next.ServeHTTP(w, r)
			return
		}
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", origin)
		h.Add("Vary", "Origin")
		// Retry-After: a cross-origin studio shows "try again in N s" on rate_limited
		h.Set("Access-Control-Expose-Headers", "Content-Disposition, Content-Range, Content-Length, ETag, Accept-Ranges, Retry-After")
		if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
			h.Set("Access-Control-Allow-Methods", "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS")
			h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Content-Range, Last-Event-ID, Range, If-None-Match, Accept, Mcp-Session-Id, Mcp-Protocol-Version")
			h.Set("Access-Control-Max-Age", "600")
			// Chrome's private-network preflight: an https page calling a
			// controller on http://127.0.0.1 asks first
			if strings.EqualFold(r.Header.Get("Access-Control-Request-Private-Network"), "true") {
				h.Set("Access-Control-Allow-Private-Network", "true")
			}
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// open: no token configured and no API key, so requests need no credentials.
func open() bool { return os.Getenv("AI_TOOLKIT_AUTH") == "" && !state.HasKeys() }

// IsOpen: no AI_TOOLKIT_AUTH and no API key.
func IsOpen() bool { return open() }

func unsafeMethod(m string) bool {
	return m != http.MethodGet && m != http.MethodHead && m != http.MethodOptions
}

// sameOrigin: the Origin names the host the request was sent to (directly, or
// the one a proxy forwarded).
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

func corsPath(p string) bool {
	return strings.HasPrefix(p, "/api/") || strings.HasPrefix(p, "/v1/") || p == "/mcp"
}

func originAllowed(origin string) bool {
	for _, o := range strings.Split(os.Getenv("AITK_CORS_ORIGINS"), ",") {
		o = strings.TrimRight(strings.TrimSpace(o), "/")
		if o == "*" || (o != "" && strings.EqualFold(o, origin)) {
			return true
		}
	}
	return false
}
