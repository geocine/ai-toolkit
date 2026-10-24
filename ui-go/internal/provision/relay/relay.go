// Package relay reverse-proxies /v1/machines/{id}/relay/{path...} to a machine.
//
// A studio served over https cannot call a plain-http GPU server (Vast,
// QuickPod direct), so it goes through the server that holds the provider
// keys instead: the browser leg stays on that server's origin and token, and
// that server dials the machine through the netguard with the machine's own
// token. Nothing is escalated: a request that did not pass the relaying
// server's auth (a public media GET, a signed link) reaches the machine with
// no credentials and the machine decides.
package relay

import (
	"context"
	"errors"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/links"
	"github.com/geocine/aitoolkit/internal/netguard"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// headerTimeout bounds the wait for the machine's response headers. It is
// long because some machine routes answer only when their work is done (a
// caption, a sync job step); SSE and downloads send headers at once and then
// stream without limit.
const headerTimeout = 5 * time.Minute

// Two shared transports, so connections to a machine are reused across
// requests. The relay never honours HTTP(S)_PROXY and has no whole-request
// timeout (SSE and uploads stream as long as they need).
var (
	transportsMu sync.Mutex
	transports   = map[bool]http.RoundTripper{}
)

// guardOptions is the netguard setup for a machine; a test swaps it (and
// clears transports) to resolve a proxy hostname through a fake resolver.
var guardOptions = func(simulated bool) netguard.Options {
	return netguard.Options{AllowLoopback: simulated, HeaderTimeout: headerTimeout}
}

func transport(simulated bool) http.RoundTripper {
	transportsMu.Lock()
	defer transportsMu.Unlock()
	if t, ok := transports[simulated]; ok {
		return t
	}
	t := netguard.Transport(guardOptions(simulated))
	transports[simulated] = t
	return t
}

// Prefix is the path on the relaying server a machine is relayed under.
func Prefix(machineID string) string { return "/v1/machines/" + machineID + "/relay" }

// relayPath splits the escaped request path into the machine id segment and
// the escaped inner path ("" or "/..."), keeping %2F and friends as sent so a
// signed link's path reaches the machine byte for byte.
var relayPath = regexp.MustCompile(`^/v1/machines/[^/]+/relay(/.*)?$`)

// escapedInner is the inner path as the client escaped it, falling back to
// escaping the decoded one (a request that did not come through the router).
func escapedInner(r *http.Request, inner string) string {
	if m := relayPath.FindStringSubmatch(r.URL.EscapedPath()); m != nil {
		return strings.TrimPrefix(m[1], "/")
	}
	return strings.TrimPrefix((&url.URL{Path: "/" + inner}).EscapedPath(), "/")
}

// stripped are the request headers a client cannot pass to the machine: its
// own credentials and cookies for the relaying server, its origin (the machine
// needs no CORS), and any forwarding claims (the relay sets its own).
var stripped = []string{"Authorization", "Cookie", "Origin", "Forwarded", "X-Real-Ip"}

// Serve proxies r to t.BaseURL + "/" + inner. authenticated reports whether
// the caller passed the relaying server's Auth (httpx.Authenticated).
func Serve(w http.ResponseWriter, r *http.Request, t provision.RelayTarget, inner string, authenticated bool) {
	if t.BaseURL == "" {
		v1api.Fail(w, v1api.CodeConflict, "this machine has no address yet")
		return
	}
	base := strings.TrimRight(t.BaseURL, "/")
	if err := provision.CheckEndpoint("", base, t.Simulated); err != nil {
		log.Printf("relay %s: %v", t.MachineID, err)
		v1api.Fail(w, v1api.CodeRemoteFailed, "the machine's address is not one the relay may dial")
		return
	}
	target, err := url.Parse(base + "/" + escapedInner(r, inner))
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "bad relay path")
		return
	}
	target.RawQuery = r.URL.RawQuery
	prefix := Prefix(t.MachineID)
	// The relaying server's own origin as its clients reach it (AITK_PUBLIC_URL,
	// else what its front proxy says), so the machine's absolute links point
	// at the relaying server.
	self, _ := url.Parse(links.Origin(r, os.Getenv("AITK_PUBLIC_URL")))

	p := &httputil.ReverseProxy{
		// Rewrite (not Director) also drops the client's X-Forwarded-* and
		// hop-by-hop headers from the outbound request.
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.Out.URL = target
			pr.Out.Host = ""
			for _, h := range stripped {
				pr.Out.Header.Del(h)
			}
			for k := range pr.Out.Header {
				if strings.HasPrefix(http.CanonicalHeaderKey(k), "X-Forwarded-") {
					pr.Out.Header.Del(k)
				}
			}
			// X-Forwarded-For marks the request as proxied, so the machine
			// never mistakes a relayed client for a local one.
			pr.SetXForwarded()
			if self != nil && self.Host != "" {
				pr.Out.Header.Set("X-Forwarded-Host", self.Host)
				pr.Out.Header.Set("X-Forwarded-Proto", self.Scheme)
			}
			pr.Out.Header.Set("X-Forwarded-Prefix", prefix)
			if authenticated && t.Token != "" {
				pr.Out.Header.Set("Authorization", "Bearer "+t.Token)
			}
		},
		Transport:     transport(t.Simulated),
		FlushInterval: -1,
		ModifyResponse: func(res *http.Response) error {
			res.Header.Del("Set-Cookie")
			// The relaying server's own CORS headers apply; a second set from the
			// machine would make the browser reject the response.
			for k := range res.Header {
				if strings.HasPrefix(http.CanonicalHeaderKey(k), "Access-Control-") {
					res.Header.Del(k)
				}
			}
			if loc := res.Header.Get("Location"); loc != "" {
				res.Header.Set("Location", rewriteLocation(loc, base, prefix))
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if errors.Is(err, context.Canceled) && r.Context().Err() != nil {
				return // the client went away
			}
			log.Printf("relay %s: %v", t.MachineID, err)
			msg := "couldn't reach the machine"
			if errors.Is(err, netguard.ErrBlocked) {
				msg = "the machine's address is not one the relay may dial"
			}
			v1api.Fail(w, v1api.CodeRemoteFailed, msg)
		},
	}
	p.ServeHTTP(w, r)
}

// rewriteLocation keeps a machine redirect inside the relay: a redirect to
// the machine's own base or to one of its absolute paths comes back under
// the prefix. Anything else (another host) passes unchanged; the relay never
// follows redirects itself.
func rewriteLocation(loc, base, prefix string) string {
	switch {
	case loc == base:
		return prefix + "/"
	case strings.HasPrefix(loc, base+"/"), strings.HasPrefix(loc, base+"?"):
		return prefix + strings.TrimPrefix(loc, base)
	case strings.HasPrefix(loc, "/") && !strings.HasPrefix(loc, "//"):
		return prefix + loc
	}
	return loc
}
