package provision

import (
	"errors"
	"fmt"
	"net/netip"
	"net/url"
	"regexp"

	"github.com/geocine/aitoolkit/internal/netguard"
)

// Proxy addresses the providers document for port 8675 (spec: runpod.md §7,
// quickpod.md §6.3). A machine URL is only ever one of these or a literal
// public IP, so a provider answer cannot point the probe or the relay at a
// name of its choosing.
var (
	runpodProxyRe   = regexp.MustCompile(`^https://[a-z0-9]+-8675\.proxy\.runpod\.net$`)
	quickpodProxyRe = regexp.MustCompile(`^https://[a-z0-9]+-8675\.proxy[12]?\.quickpod\.io$`)
)

// ErrEndpoint is wrapped by every CheckEndpoint refusal.
var ErrEndpoint = errors.New("machine endpoint not allowed")

// CheckEndpoint applies the per-provider URL rules to a machine endpoint
// before anything dials it (PLAN §7). The URL is a base: http or https, no
// userinfo, path, query or fragment.
//   - simulated: the host is exactly 127.0.0.1 over http (the in-process node),
//     and nothing else;
//   - RunPod: its https proxy only;
//   - Vast: a direct endpoint on a literal public IP;
//   - QuickPod: its https proxy, or a direct literal public IP.
//
// kind "" accepts what any provider could report (the probe and the relay
// know the endpoint, not its provider). netguard still checks the resolved
// address at connect time.
func CheckEndpoint(kind Kind, raw string, simulated bool) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrEndpoint, err)
	}
	switch {
	case u.Scheme != "http" && u.Scheme != "https":
		return fmt.Errorf("%w: %q is not http or https", ErrEndpoint, raw)
	case u.User != nil:
		return fmt.Errorf("%w: %s carries credentials", ErrEndpoint, u.Redacted())
	case u.Host == "" || u.Opaque != "":
		return fmt.Errorf("%w: %q has no host", ErrEndpoint, raw)
	case u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.ForceQuery:
		return fmt.Errorf("%w: %q is not a base URL", ErrEndpoint, raw)
	}
	if simulated {
		if u.Scheme == "http" && u.Hostname() == "127.0.0.1" && u.Port() != "" {
			return nil
		}
		return fmt.Errorf("%w: a simulated machine is http://127.0.0.1:<port>, not %q", ErrEndpoint, raw)
	}
	origin := u.Scheme + "://" + u.Host
	proxy := func(re *regexp.Regexp) bool { return re.MatchString(origin) }
	switch kind {
	case RunPod:
		if proxy(runpodProxyRe) {
			return nil
		}
		return fmt.Errorf("%w: %q is not a RunPod proxy address", ErrEndpoint, raw)
	case QuickPod:
		if proxy(quickpodProxyRe) {
			return nil
		}
		return checkDirect(raw, u)
	case Vast:
		return checkDirect(raw, u)
	case "":
		if proxy(runpodProxyRe) || proxy(quickpodProxyRe) {
			return nil
		}
		return checkDirect(raw, u)
	}
	return fmt.Errorf("%w: unknown provider %q", ErrEndpoint, kind)
}

// FilterEndpoints keeps the endpoints that pass CheckEndpoint for kind, in
// order. The service runs it on a machine's endpoints before probing or
// relaying, so a RunPod record carrying a direct IP, or a Vast record
// carrying a proxy host, is never sent the machine token.
func FilterEndpoints(kind Kind, eps []Endpoint, simulated bool) []Endpoint {
	var out []Endpoint
	for _, ep := range eps {
		if CheckEndpoint(kind, ep.URL, simulated) == nil {
			out = append(out, ep)
		}
	}
	return out
}

// checkDirect: a direct endpoint names a literal public IP, never a hostname.
func checkDirect(raw string, u *url.URL) error {
	a, err := netip.ParseAddr(u.Hostname())
	if err != nil {
		return fmt.Errorf("%w: a direct endpoint needs a literal IP, not %q", ErrEndpoint, raw)
	}
	if a.Zone() != "" {
		return fmt.Errorf("%w: %q has an IPv6 zone", ErrEndpoint, raw)
	}
	if err := netguard.CheckAddr(a, false); err != nil {
		return fmt.Errorf("%w: %v", ErrEndpoint, err)
	}
	return nil
}
