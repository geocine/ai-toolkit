// Package netguard dials only public unicast addresses, checked at connect
// time (defeats DNS rebinding). Leaf package: stdlib only.
//
// The provisioning probe and relay use it to reach GPU servers, and the image
// check uses it to reach registries and their blob CDNs, so a hostile name,
// redirect or provider answer cannot point the server at itself, the LAN or a
// cloud metadata address. With Options.Proxy the guard checks the proxy's own
// address and refuses a literal-IP destination before it reaches the proxy;
// a hostname sent through a proxy is resolved by the proxy and cannot be
// checked here.
package netguard

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Options configures Transport and NewClient. The zero value is the strict
// guard with the default timeouts.
type Options struct {
	AllowLoopback      bool          // tests and simulate: ONLY 127.0.0.1 / ::1 are allowed, everything else refused
	AllowLoopbackProxy bool          // a loopback HTTP(S)_PROXY (the image check sets it from env AITK_IMAGECHECK_ALLOW_LOOPBACK_PROXY=1)
	Proxy              bool          // honour http.ProxyFromEnvironment (imagecheck: true; provision probe/relay: false)
	Timeout            time.Duration // whole request; 0 = default 10 s, NoTimeout = none (imagecheck: it uses ctx)
	DialTimeout        time.Duration // default 5 s
	TLSTimeout         time.Duration // default 5 s
	HeaderTimeout      time.Duration // default 8 s
	Resolver           *net.Resolver // nil = net.DefaultResolver; tests pass StaticResolver
}

// NoTimeout as Options.Timeout leaves the whole request bounded only by its
// context.
const NoTimeout time.Duration = -1

const (
	defaultTimeout       = 10 * time.Second
	defaultDialTimeout   = 5 * time.Second
	defaultTLSTimeout    = 5 * time.Second
	defaultHeaderTimeout = 8 * time.Second
)

// ErrBlocked is wrapped by every refusal, so callers can tell a guarded
// address from a network failure with errors.Is.
var ErrBlocked = errors.New("netguard: address not allowed")

// Ranges the netip predicates do not cover, the IPv6 prefixes that carry an
// IPv4 address, the two loopback addresses the loopback flag allows, and the
// proxy seam.
var (
	cgnat      = netip.MustParsePrefix("100.64.0.0/10")
	thisNet    = netip.MustParsePrefix("0.0.0.0/8")
	reserved4  = netip.MustParsePrefix("240.0.0.0/4")    // incl. 255.255.255.255
	nat64      = netip.MustParsePrefix("64:ff9b::/96")   // well-known NAT64: the IPv4 address is the last 4 bytes
	nat64Local = netip.MustParsePrefix("64:ff9b:1::/48") // local-use NAT64: the embedding is site-specific, so all of it is refused
	sixToFour  = netip.MustParsePrefix("2002::/16")      // 6to4: the IPv4 address is bytes 2-5
	loopback4  = netip.MustParseAddr("127.0.0.1")
	loopback6  = netip.IPv6Loopback()
	errNoAddr  = fmt.Errorf("%w: not an IP address", ErrBlocked)
	proxyFunc  = http.ProxyFromEnvironment // tests swap it: ProxyFromEnvironment reads the env only once
)

// CheckAddr refuses loopback (unless allowLoopback), RFC 1918, CGNAT 100.64/10,
// link-local 169.254/16 and fe80::/10, ULA fc00::/7, multicast, unspecified,
// and with allowLoopback everything that is not loopback. A NAT64 or 6to4
// address is judged by the IPv4 address it carries.
func CheckAddr(a netip.Addr, allowLoopback bool) error {
	a = a.Unmap()
	if !a.IsValid() {
		return errNoAddr
	}
	if allowLoopback {
		if a == loopback4 || a == loopback6 {
			return nil
		}
		return fmt.Errorf("%w: %s is not loopback", ErrBlocked, a)
	}
	if nat64Local.Contains(a) {
		return fmt.Errorf("%w: %s is local-use NAT64", ErrBlocked, a)
	}
	if v4, ok := embedded4(a); ok {
		if err := CheckAddr(v4, false); err != nil {
			return fmt.Errorf("%w (carried in %s)", err, a)
		}
	}
	switch {
	case a.IsLoopback():
		return fmt.Errorf("%w: %s is loopback", ErrBlocked, a)
	case a.IsPrivate(): // 10/8, 172.16/12, 192.168/16, fc00::/7
		return fmt.Errorf("%w: %s is private", ErrBlocked, a)
	case cgnat.Contains(a):
		return fmt.Errorf("%w: %s is carrier-grade NAT", ErrBlocked, a)
	case a.IsLinkLocalUnicast(), a.IsLinkLocalMulticast():
		return fmt.Errorf("%w: %s is link-local", ErrBlocked, a)
	case a.IsMulticast(), a.IsInterfaceLocalMulticast():
		return fmt.Errorf("%w: %s is multicast", ErrBlocked, a)
	case a.IsUnspecified(), thisNet.Contains(a), reserved4.Contains(a):
		return fmt.Errorf("%w: %s is not a unicast address", ErrBlocked, a)
	}
	return nil
}

// embedded4 is the IPv4 address a NAT64 (64:ff9b::/96) or 6to4 (2002::/16)
// address carries.
func embedded4(a netip.Addr) (netip.Addr, bool) {
	b := a.As16()
	switch {
	case nat64.Contains(a):
		return netip.AddrFrom4([4]byte(b[12:16])), true
	case sixToFour.Contains(a):
		return netip.AddrFrom4([4]byte(b[2:6])), true
	}
	return netip.Addr{}, false
}

// Control is a net.Dialer.Control hook applying CheckAddr to the resolved address.
func Control(allowLoopback bool) func(network, address string, c syscall.RawConn) error {
	return control(func(a netip.Addr) error { return CheckAddr(a, allowLoopback) })
}

func control(check func(netip.Addr) error) func(network, address string, c syscall.RawConn) error {
	return func(network, address string, _ syscall.RawConn) error {
		ap, err := netip.ParseAddrPort(address)
		if err != nil {
			return fmt.Errorf("%w: %q", errNoAddr, address)
		}
		return check(ap.Addr())
	}
}

// Transport is an http.Transport whose every connection goes through the
// guard. With o.Proxy the proxy itself is dialled through the guard too; a
// loopback proxy passes only with o.AllowLoopbackProxy. A destination given as
// a literal IP is checked before the proxy is asked, so a refused address
// never reaches it; a destination hostname is resolved by the proxy.
func Transport(o Options) *http.Transport {
	dialTimeout := orDefault(o.DialTimeout, defaultDialTimeout)
	strict := &net.Dialer{Timeout: dialTimeout, KeepAlive: 30 * time.Second, Resolver: o.Resolver, Control: Control(o.AllowLoopback)}
	t := &http.Transport{
		DialContext:           strict.DialContext,
		TLSHandshakeTimeout:   orDefault(o.TLSTimeout, defaultTLSTimeout),
		ResponseHeaderTimeout: orDefault(o.HeaderTimeout, defaultHeaderTimeout),
		MaxIdleConns:          32,
		MaxIdleConnsPerHost:   4,
		IdleConnTimeout:       90 * time.Second,
		ExpectContinueTimeout: time.Second,
		ForceAttemptHTTP2:     true,
	}
	if !o.Proxy {
		return t
	}
	var proxies sync.Map // "host:port" of every proxy this transport was handed
	t.Proxy = func(r *http.Request) (*url.URL, error) {
		if a, err := netip.ParseAddr(r.URL.Hostname()); err == nil {
			if err := CheckAddr(a, o.AllowLoopback); err != nil {
				return nil, err
			}
		}
		u, err := proxyFunc(r)
		if u != nil {
			proxies.Store(hostPort(u), true)
		}
		return u, err
	}
	if o.AllowLoopbackProxy {
		lenient := &net.Dialer{Timeout: dialTimeout, KeepAlive: 30 * time.Second, Resolver: o.Resolver,
			Control: control(func(a netip.Addr) error {
				if CheckAddr(a, true) == nil {
					return nil
				}
				return CheckAddr(a, o.AllowLoopback)
			})}
		t.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			if _, ok := proxies.Load(addr); ok {
				return lenient.DialContext(ctx, network, addr)
			}
			return strict.DialContext(ctx, network, addr)
		}
	}
	return t
}

// NewClient never follows redirects (CheckRedirect = http.ErrUseLastResponse).
func NewClient(o Options) *http.Client {
	c := &http.Client{
		Transport:     Transport(o),
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	if o.Timeout >= 0 {
		c.Timeout = orDefault(o.Timeout, defaultTimeout)
	}
	return c
}

func orDefault(d, def time.Duration) time.Duration {
	if d > 0 {
		return d
	}
	return def
}

// hostPort is the address the transport dials for a proxy URL, matching what
// it passes to DialContext.
func hostPort(u *url.URL) string {
	port := u.Port()
	if port == "" {
		switch strings.ToLower(u.Scheme) {
		case "https":
			port = "443"
		case "socks5", "socks5h":
			port = "1080"
		default:
			port = "80"
		}
	}
	return net.JoinHostPort(u.Hostname(), port)
}

// StaticResolver answers DNS lookups from hosts (name -> addresses) without
// touching the network; a name it does not know is "no such host". It is for
// tests that need a hostname resolving to a private address, so the guard is
// exercised at connect time rather than at parse time.
func StaticResolver(hosts map[string][]netip.Addr) *net.Resolver {
	return &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, _, _ string) (net.Conn, error) {
			client, server := net.Pipe()
			go serveDNS(server, hosts)
			return client, nil
		},
	}
}

// serveDNS answers DNS-over-TCP framed queries (net.Pipe is a stream, so the
// Go resolver length-prefixes each message) until the pipe closes.
func serveDNS(c net.Conn, hosts map[string][]netip.Addr) {
	defer c.Close()
	for {
		var n [2]byte
		if _, err := io.ReadFull(c, n[:]); err != nil {
			return
		}
		q := make([]byte, binary.BigEndian.Uint16(n[:]))
		if _, err := io.ReadFull(c, q); err != nil {
			return
		}
		resp := dnsAnswer(q, hosts)
		if resp == nil {
			return
		}
		out := binary.BigEndian.AppendUint16(nil, uint16(len(resp)))
		if _, err := c.Write(append(out, resp...)); err != nil {
			return
		}
	}
}

// dnsAnswer builds the reply to a single-question query: the matching A or
// AAAA records, or NXDOMAIN for an unknown name.
func dnsAnswer(q []byte, hosts map[string][]netip.Addr) []byte {
	if len(q) < 12 {
		return nil
	}
	// Question: labels, then type and class.
	i := 12
	var labels []string
	for i < len(q) && q[i] != 0 {
		l := int(q[i])
		if i+1+l > len(q) {
			return nil
		}
		labels = append(labels, string(q[i+1:i+1+l]))
		i += 1 + l
	}
	if i+5 > len(q) {
		return nil
	}
	qEnd := i + 5
	qtype := binary.BigEndian.Uint16(q[i+1:])
	name := strings.ToLower(strings.Join(labels, "."))

	addrs, known := hosts[name]
	var answers [][]byte
	for _, a := range addrs {
		a = a.Unmap()
		switch {
		case qtype == 1 && a.Is4():
			b := a.As4()
			answers = append(answers, b[:])
		case qtype == 28 && a.Is6():
			b := a.As16()
			answers = append(answers, b[:])
		}
	}
	flags := uint16(0x8180) // response, recursion desired and available
	if !known {
		flags |= 3 // NXDOMAIN
	}
	resp := make([]byte, 0, qEnd+len(answers)*28)
	resp = append(resp, q[0], q[1])
	resp = binary.BigEndian.AppendUint16(resp, flags)
	resp = binary.BigEndian.AppendUint16(resp, 1)
	resp = binary.BigEndian.AppendUint16(resp, uint16(len(answers)))
	resp = append(resp, 0, 0, 0, 0)
	resp = append(resp, q[12:qEnd]...)
	for _, rdata := range answers {
		resp = append(resp, 0xC0, 12) // name: pointer to the question
		resp = binary.BigEndian.AppendUint16(resp, qtype)
		resp = binary.BigEndian.AppendUint16(resp, 1) // IN
		resp = binary.BigEndian.AppendUint32(resp, 60)
		resp = binary.BigEndian.AppendUint16(resp, uint16(len(rdata)))
		resp = append(resp, rdata...)
	}
	return resp
}
