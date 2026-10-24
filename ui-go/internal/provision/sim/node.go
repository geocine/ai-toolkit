package sim

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
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
//     already grants everything below Auth, so that escalates nothing;
//   - GET /v1/system/cuda answers from n.CUDA instead of running this
//     host's Python (nodeCUDA), and GET /v1/health reports the build the
//     launch ref implies, or StaleVersion (nodeVersion).
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
		if r.Method == http.MethodGet && r.URL.Path == "/v1/system/cuda" {
			v1api.OK(w, nodeCUDA(n, clock.Now()))
			return
		}
		r = r.Clone(r.Context())
		r.Header.Del("Authorization")
		if ctl := os.Getenv("AI_TOOLKIT_AUTH"); ctl != "" && strings.HasPrefix(r.URL.Path, "/mcp") {
			r.Header.Set("Authorization", "Bearer "+ctl)
		}
		if r.Method == http.MethodGet && r.URL.Path == "/v1/health" && (n.Image != "" || n.StaleImage) {
			serveHealth(w, r, inner, nodeVersion(n))
			return
		}
		inner.ServeHTTP(w, r)
	})
}

// StaleVersion is the build a node with Faults.StaleImage reports in
// /v1/health: older than any launch ref names, as when the host ran a cached
// copy of the image.
const StaleVersion = "0a1b2c3"

// shaTag is a tag that names a build (the image's :<revision> tag).
var shaTag = regexp.MustCompile(`^[0-9a-f]{7,40}$`)

// nodeVersion is the build the node's image runs: the :<revision> tag of the
// launch ref, else (a moving tag or a digest) the build the simulated
// registry publishes; StaleVersion with Faults.StaleImage.
func nodeVersion(n mockkit.NodeSpec) string {
	if n.StaleImage {
		return StaleVersion
	}
	if ref, err := imagecheck.ParseRef(n.Image); err == nil && shaTag.MatchString(ref.Tag) {
		return ref.Tag
	}
	return registrymock.Revision
}

// serveHealth is inner's /v1/health with its version replaced.
func serveHealth(w http.ResponseWriter, r *http.Request, inner http.Handler, version string) {
	rec := httptest.NewRecorder()
	inner.ServeHTTP(rec, r)
	var body map[string]any
	if rec.Code != http.StatusOK || json.Unmarshal(rec.Body.Bytes(), &body) != nil {
		for k, v := range rec.Header() {
			w.Header()[k] = v
		}
		w.WriteHeader(rec.Code)
		_, _ = w.Write(rec.Body.Bytes())
		return
	}
	body["version"] = version
	v1api.OK(w, body)
}

// What the simulated node's torch and driver report: the default image's
// build (registrymock.GoLabels).
const (
	nodeTorch     = "2.13.0+cu130"
	nodeCUDA13    = "13.0"
	nodeDriver    = "580.95.05"
	nodeOldDriver = "535.104.05"
)

var nodeArchList = []string{"sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"}

// nodeCUDA is the node's GET /v1/system/cuda, in the CudaCheck shape of
// handlers/v1/system_cuda.go: one device per GPU of the offer, failing the
// way Faults.CUDA asks.
func nodeCUDA(n mockkit.NodeSpec, now time.Time) map[string]any {
	driver := nodeDriver
	if n.CUDA == mockkit.CUDAOldDriver {
		driver = nodeOldDriver
	}
	capability := ""
	if g, ok := hostfit.Lookup(n.GPU); ok && g.SM > 0 {
		capability = fmt.Sprintf("%d.%d", g.SM/10, g.SM%10)
	}
	devices := make([]map[string]any, max(n.GPUCount, 1))
	for i := range devices {
		devices[i] = map[string]any{"index": i, "name": n.GPU, "capability": capability, "driver": driver}
	}
	var errText, failure any
	switch n.CUDA {
	case mockkit.CUDAOK:
	case mockkit.CUDANoKernel:
		errText, failure = "AcceleratorError: CUDA error: no kernel image is available for execution on the device", "no_kernel_image"
	case mockkit.CUDAOldDriver:
		errText, failure = "RuntimeError: The NVIDIA driver on your system is too old (found version 12020). Please update your GPU driver.", "driver_too_old"
	default:
		errText, failure = "simulated CUDA failure: "+n.CUDA, n.CUDA
	}
	return map[string]any{
		"ok": n.CUDA == mockkit.CUDAOK, "checked_at": v1api.Time(now), "torch": nodeTorch, "cuda": nodeCUDA13,
		"arch_list": nodeArchList, "devices": devices, "error": errText, "failure": failure,
	}
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
