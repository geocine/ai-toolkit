package httpx

import (
	"net/http"
	"os"
	"strings"
)

// CORS lets browser apps on other origins (the studio, a desktop app's
// webview, ...) call /api, /v1 and /mcp. Allowed origins come from
// AITK_CORS_ORIGINS: a comma list of origins, or "*" for any. Unset means
// same-origin only, like upstream. Preflights are answered here, before auth,
// because browsers send them without the Authorization header.
func CORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin == "" || !corsPath(r.URL.Path) || !originAllowed(origin) {
			next.ServeHTTP(w, r)
			return
		}
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", origin)
		h.Add("Vary", "Origin")
		h.Set("Access-Control-Expose-Headers", "Content-Disposition, Content-Range, Content-Length, ETag, Accept-Ranges")
		if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
			h.Set("Access-Control-Allow-Methods", "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS")
			h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Last-Event-ID, Range, If-None-Match, Accept, Mcp-Session-Id, Mcp-Protocol-Version")
			h.Set("Access-Control-Max-Age", "600")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
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
