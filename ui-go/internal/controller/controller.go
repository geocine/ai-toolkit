// Package controller is the no-GPU mode: this server creates and tracks GPU
// servers but never trains. main turns it on (--controller or
// AITK_CONTROLLER), re-roots the data dir, skips the trainer's loops and puts
// Gate in front of the mux (docs/controller.md).
package controller

import (
	"errors"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/v1api"
)

var enabled atomic.Bool

// Enable turns controller mode on for the life of the process.
func Enable() { enabled.Store(true) }

// Enabled reports whether this server is a controller.
func Enabled() bool { return enabled.Load() }

// SetForTest switches the mode and returns a restore func for t.Cleanup, so one
// test enabling controller mode never leaks into the rest of its package.
func SetForTest(on bool) (restore func()) {
	prev := enabled.Swap(on)
	return func() { enabled.Store(prev) }
}

// FromEnv: AITK_CONTROLLER is 1, true or yes (case-insensitive).
func FromEnv() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AITK_CONTROLLER"))) {
	case "1", "true", "yes":
		return true
	}
	return false
}

// DataDir is flag > AITK_CONTROLLER_DIR > os.UserConfigDir()/aitk-controller
// (Windows %AppData%\aitk-controller, macOS ~/Library/Application Support/aitk-controller,
// Linux ~/.config/aitk-controller). Created with mode 0700 (chmod'ed to 0700 if it exists).
func DataDir(flagValue string) (string, error) {
	dir := strings.TrimSpace(flagValue)
	if dir == "" {
		dir = strings.TrimSpace(os.Getenv("AITK_CONTROLLER_DIR"))
	}
	if dir == "" {
		base, err := os.UserConfigDir()
		if err != nil {
			return "", errors.New("controller: no user config dir (" + err.Error() + "); pass --data-dir or set AITK_CONTROLLER_DIR")
		}
		dir = filepath.Join(base, "aitk-controller")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(abs, 0o700); err != nil {
		return "", err
	}
	// MkdirAll leaves an existing dir as it was; the keys live here
	if err := os.Chmod(abs, 0o700); err != nil {
		return "", err
	}
	return abs, nil
}

// SkippedLoops are bg loops that need the trainer, job folders or dataset folders.
var SkippedLoops = []string{"bundles", "bundles-bootstrap", "portable", "lifecycle", "uploads", "uploads-auto", "photocheck-gc"}

// RunningLoops are the bg loops reviewed as safe on a controller. A test fails
// when a registered loop is in neither list, so a new loop cannot slip in unreviewed.
var RunningLoops = []string{"machines", "notify"}

// BindRefusal is why a controller will not listen on a reachable address
// without a token.
const BindRefusal = "Controller mode on a reachable address holds your provider keys: set AI_TOOLKIT_AUTH, or listen on 127.0.0.1."

// CheckBind refuses a non-loopback host without an env token. API keys cannot
// exist yet on a fresh database, so only AI_TOOLKIT_AUTH counts.
func CheckBind(host, token string) error {
	if loopbackName(host) || token != "" {
		return nil
	}
	return errors.New(BindRefusal)
}

// loopbackName: localhost or a loopback IP ("" and 0.0.0.0 are every
// interface, so not loopback).
func loopbackName(h string) bool {
	h = strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(h), "["), "]")
	if strings.EqualFold(h, "localhost") {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// HostAllowed: Host is 127.0.0.1, [::1] or localhost (any port), the host of
// AITK_PUBLIC_URL, or a host listed in AITK_ALLOWED_HOSTS (comma list, host or
// host:port, case-insensitive). An entry without a port matches any port.
func HostAllowed(host string) bool {
	host = strings.TrimSpace(host)
	if host == "" {
		return false
	}
	name := host
	if h, _, err := net.SplitHostPort(host); err == nil {
		name = h
	}
	if loopbackName(name) {
		return true
	}
	// the operator declared the public name, so a rebinding page cannot use
	// it; MCP's in-process calls carry it as their Host (links.Origin)
	if u, err := url.Parse(strings.TrimSpace(os.Getenv("AITK_PUBLIC_URL"))); err == nil && u.Host != "" && hostMatches(u.Host, host, name) {
		return true
	}
	for _, a := range strings.Split(os.Getenv("AITK_ALLOWED_HOSTS"), ",") {
		if a = strings.TrimSpace(a); a != "" && hostMatches(a, host, name) {
			return true
		}
	}
	return false
}

// hostMatches: entry (host or host:port) names host as sent, or its name
// without the port.
func hostMatches(entry, host, name string) bool {
	return strings.EqualFold(entry, host) || strings.EqualFold(entry, name)
}

// Refusal answers a request that needs the trainer.
const Refusal = "This server is a controller (no GPU, no trainer): it creates and tracks GPU servers. Do this on a GPU server: add one on the Servers page, or create one with machines.launch."

// HostRefusal answers an open controller reached under a name it does not know.
const HostRefusal = "This controller only answers on 127.0.0.1 or localhost. Set AITK_ALLOWED_HOSTS to add a name."

// writeRoots are refused for every method but GET and HEAD: their reads answer
// from the empty schema and the empty data dir.
var writeRoots = []string{"/v1/jobs", "/v1/queues", "/v1/datasets", "/v1/bundles", "/v1/uploads", "/v1/lifecycle", "/v1/portable", "/v1/files"}

// Allowed reports whether a request is served in controller mode. It refuses
// only what is known to need the trainer; unknown paths fall through to the
// mux's not_found.
func Allowed(method, path string) bool {
	if strings.HasPrefix(path, "/api/") || strings.HasPrefix(path, "/v1/system/") {
		return false
	}
	if !strings.HasPrefix(path, "/v1/") || method == http.MethodGet || method == http.MethodHead {
		return true
	}
	for _, root := range writeRoots {
		if path == root || strings.HasPrefix(path, root+"/") {
			return false
		}
	}
	return true
}

// Gate answers what Allowed refuses, and on an open server (httpx.IsOpen())
// refuses a Host that HostAllowed rejects (DNS rebinding). Outside controller
// mode it is a no-op.
func Gate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !Enabled() {
			next.ServeHTTP(w, r)
			return
		}
		// a page on evil.com re-resolved to 127.0.0.1 is same-origin to CORS
		// and a loopback peer to the open-server guard; only its Host gives it
		// away. With a token the page cannot send the bearer header.
		if httpx.IsOpen() && !HostAllowed(r.Host) {
			refuse(w, r, http.StatusForbidden, v1api.CodeForbidden, HostRefusal)
			return
		}
		if !Allowed(r.Method, r.URL.Path) {
			refuse(w, r, http.StatusConflict, v1api.CodeConflict, Refusal)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// refuse answers in the shape the path's clients read: upstream's
// {"error": msg} under /api/, the /v1 envelope everywhere else.
func refuse(w http.ResponseWriter, r *http.Request, status int, code, msg string) {
	if strings.HasPrefix(r.URL.Path, "/api/") {
		httpx.JSON(w, status, map[string]string{"error": msg})
		return
	}
	v1api.Fail(w, code, msg)
}
