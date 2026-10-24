package netguard

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"testing"
	"time"
)

func TestCheckAddr(t *testing.T) {
	refused := []string{
		"127.0.0.1", "127.8.9.10", "::1",
		"10.0.0.1", "172.16.0.1", "172.31.255.255", "192.168.1.1",
		"100.64.0.1", "100.127.255.254",
		"169.254.169.254", "fe80::1",
		"fc00::1", "fd12:3456::1",
		"224.0.0.1", "239.255.255.250", "ff02::1", "ff05::2",
		"0.0.0.0", "0.1.2.3", "::", "255.255.255.255", "240.0.0.1",
		"::ffff:10.0.0.1", "::ffff:127.0.0.1", "::ffff:169.254.169.254",
		// NAT64 and 6to4 addresses are judged by the IPv4 address they carry.
		"64:ff9b::a00:1", "64:ff9b::a9fe:a9fe", "64:ff9b::7f00:1", "64:ff9b:1::a00:1",
		"2002:a00:1::", "2002:a9fe:a9fe::1", "2002:c0a8:101::",
	}
	for _, s := range refused {
		if err := CheckAddr(netip.MustParseAddr(s), false); !errors.Is(err, ErrBlocked) {
			t.Errorf("CheckAddr(%s) = %v, want ErrBlocked", s, err)
		}
	}
	allowed := []string{"8.8.8.8", "1.1.1.1", "100.128.0.1", "172.32.0.1", "192.169.0.1", "2606:4700::1111", "::ffff:8.8.4.4",
		"64:ff9b::808:808", "2002:808:808::1"}
	for _, s := range allowed {
		if err := CheckAddr(netip.MustParseAddr(s), false); err != nil {
			t.Errorf("CheckAddr(%s) = %v, want nil", s, err)
		}
	}
	if err := CheckAddr(netip.Addr{}, false); !errors.Is(err, ErrBlocked) {
		t.Errorf("zero addr: %v", err)
	}

	// The loopback flag allows exactly the two loopback addresses and nothing else.
	for _, s := range []string{"127.0.0.1", "::1", "::ffff:127.0.0.1"} {
		if err := CheckAddr(netip.MustParseAddr(s), true); err != nil {
			t.Errorf("loopback-only CheckAddr(%s) = %v", s, err)
		}
	}
	for _, s := range []string{"127.0.0.2", "8.8.8.8", "10.0.0.1", "169.254.169.254", "::"} {
		if err := CheckAddr(netip.MustParseAddr(s), true); !errors.Is(err, ErrBlocked) {
			t.Errorf("loopback-only CheckAddr(%s) = %v, want ErrBlocked", s, err)
		}
	}
}

func TestControlRejectsNonIP(t *testing.T) {
	if err := Control(false)("tcp", "example.com:443", nil); !errors.Is(err, ErrBlocked) {
		t.Fatalf("hostname reached Control: %v", err)
	}
	if err := Control(false)("tcp4", "93.184.216.34:443", nil); err != nil {
		t.Fatalf("public address: %v", err)
	}
}

// offline is a resolver that knows only these names, so no test touches DNS.
func offline(hosts map[string]string) *net.Resolver {
	m := map[string][]netip.Addr{}
	for name, a := range hosts {
		m[name] = []netip.Addr{netip.MustParseAddr(a)}
	}
	return StaticResolver(m)
}

func get(t *testing.T, c *http.Client, u string) (*http.Response, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Do(req)
	if resp != nil {
		resp.Body.Close()
	}
	return resp, err
}

func TestClientRefusesPrivateTargets(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) }))
	defer srv.Close()
	res := offline(map[string]string{"rebind.test": "10.0.0.7", "self.test": "127.0.0.1"})
	strict := NewClient(Options{Resolver: res})

	// Control runs before connect, so none of these sends a packet.
	for _, u := range []string{
		srv.URL,                          // 127.0.0.1
		"http://10.0.0.1/",               // RFC 1918
		"http://169.254.169.254/latest/", // cloud metadata
		"http://[::1]:1/",
		"http://[fe80::1%25lo]:1/",
		"http://rebind.test/", // a name resolving to 10.x is checked after resolution
	} {
		if _, err := get(t, strict, u); !errors.Is(err, ErrBlocked) {
			t.Errorf("GET %s: %v, want ErrBlocked", u, err)
		}
	}

	// The loopback flag reaches the test server, by IP and through the resolver,
	// and still refuses everything else.
	loop := NewClient(Options{AllowLoopback: true, Resolver: res})
	if resp, err := get(t, loop, srv.URL); err != nil || resp.StatusCode != http.StatusTeapot {
		t.Fatalf("loopback by IP: %v %v", resp, err)
	}
	_, port, _ := net.SplitHostPort(srv.Listener.Addr().String())
	if resp, err := get(t, loop, "http://self.test:"+port+"/"); err != nil || resp.StatusCode != http.StatusTeapot {
		t.Fatalf("loopback by name: %v %v", resp, err)
	}
	for _, u := range []string{"http://rebind.test/", "http://10.0.0.1/", "http://8.8.8.8:1/"} {
		if _, err := get(t, loop, u); !errors.Is(err, ErrBlocked) {
			t.Errorf("loopback-only GET %s: %v, want ErrBlocked", u, err)
		}
	}
	if _, err := get(t, loop, "http://unknown.test/"); err == nil || errors.Is(err, ErrBlocked) {
		t.Errorf("unknown name: %v, want a lookup error", err)
	}
}

func TestNoRedirects(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.Redirect(w, r, "http://169.254.169.254/latest/meta-data/", http.StatusFound)
	}))
	defer srv.Close()
	resp, err := get(t, NewClient(Options{AllowLoopback: true}), srv.URL)
	if err != nil || resp.StatusCode != http.StatusFound || hits != 1 {
		t.Fatalf("redirect: %v %v hits=%d", resp, err, hits)
	}
	if loc := resp.Header.Get("Location"); loc == "" {
		t.Fatal("the redirect is handed back to the caller")
	}
}

func TestProxyRules(t *testing.T) {
	var proxied []string
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxied = append(proxied, r.URL.Host)
		w.WriteHeader(http.StatusTeapot)
	}))
	defer proxy.Close()
	proxyURL, _ := url.Parse(proxy.URL)

	saved := proxyFunc
	t.Cleanup(func() { proxyFunc = saved })
	proxyFunc = func(r *http.Request) (*url.URL, error) {
		if r.URL.Hostname() == "direct.test" {
			return nil, nil
		}
		return proxyURL, nil
	}
	res := offline(map[string]string{"direct.test": "127.0.0.1"})

	// A loopback proxy is refused unless explicitly allowed.
	if _, err := get(t, NewClient(Options{Proxy: true, Resolver: res}), "http://registry.test/v2/"); !errors.Is(err, ErrBlocked) {
		t.Fatalf("loopback proxy without the flag: %v", err)
	}
	if len(proxied) != 0 {
		t.Fatalf("proxy reached: %v", proxied)
	}

	// Allowed: the request goes through the proxy, which resolves the target itself.
	c := NewClient(Options{Proxy: true, AllowLoopbackProxy: true, Resolver: res})
	if resp, err := get(t, c, "http://registry.test/v2/"); err != nil || resp.StatusCode != http.StatusTeapot {
		t.Fatalf("through the proxy: %v %v", resp, err)
	}
	if len(proxied) != 1 || proxied[0] != "registry.test" {
		t.Fatalf("proxied %v", proxied)
	}

	// A literal private, link-local or loopback destination is refused before it
	// reaches the proxy, which would otherwise fetch it from its own network.
	for _, u := range []string{"http://10.0.0.1/", "https://169.254.169.254/latest/", "http://[64:ff9b::a00:1]/", "http://127.0.0.1:1/"} {
		if _, err := get(t, c, u); !errors.Is(err, ErrBlocked) {
			t.Errorf("proxied GET %s: %v, want ErrBlocked", u, err)
		}
	}
	if len(proxied) != 1 {
		t.Fatalf("a refused literal reached the proxy: %v", proxied)
	}
	// A public literal still goes through the proxy.
	if resp, err := get(t, c, "http://8.8.8.8/"); err != nil || resp.StatusCode != http.StatusTeapot || len(proxied) != 2 {
		t.Fatalf("public literal through the proxy: %v %v %v", resp, err, proxied)
	}
	proxied = proxied[:1]

	// The proxy exception covers only the proxy: a direct dial to loopback is still refused.
	_, port, _ := net.SplitHostPort(proxy.Listener.Addr().String())
	if _, err := get(t, c, "http://direct.test:"+port+"/"); !errors.Is(err, ErrBlocked) {
		t.Fatalf("direct loopback with the proxy flag: %v", err)
	}

	// Without Proxy the environment's proxy is ignored.
	if _, err := get(t, NewClient(Options{AllowLoopbackProxy: true, Resolver: res}), "http://registry.test/v2/"); err == nil || len(proxied) != 1 {
		t.Fatalf("Proxy:false used the proxy: %v %v", err, proxied)
	}
}

func TestDefaults(t *testing.T) {
	c := NewClient(Options{})
	if c.Timeout != 10*time.Second || c.CheckRedirect == nil {
		t.Fatalf("client defaults: %v", c.Timeout)
	}
	tr := c.Transport.(*http.Transport)
	if tr.MaxIdleConnsPerHost != 4 || tr.TLSHandshakeTimeout != 5*time.Second || tr.ResponseHeaderTimeout != 8*time.Second || tr.Proxy != nil {
		t.Fatalf("transport defaults: %+v", tr)
	}
	if c := NewClient(Options{Timeout: NoTimeout}); c.Timeout != 0 {
		t.Fatalf("NoTimeout: %v", c.Timeout)
	}
	if c := NewClient(Options{Timeout: 3 * time.Second, HeaderTimeout: time.Second}); c.Timeout != 3*time.Second ||
		c.Transport.(*http.Transport).ResponseHeaderTimeout != time.Second {
		t.Fatal("explicit timeouts are kept")
	}
	if Transport(Options{Proxy: true}).Proxy == nil {
		t.Fatal("Proxy:true sets a proxy func")
	}
}
