package provision

import (
	"errors"
	"slices"
	"testing"
)

func TestEndpointRules(t *testing.T) {
	cases := []struct {
		kind Kind
		url  string
		sim  bool
		ok   bool
	}{
		// RunPod: its https proxy only
		{RunPod, "https://abc123xyz-8675.proxy.runpod.net", false, true},
		{RunPod, "http://abc123xyz-8675.proxy.runpod.net", false, false},
		{RunPod, "https://abc123xyz-8888.proxy.runpod.net", false, false},
		{RunPod, "https://ABC-8675.proxy.runpod.net", false, false},
		{RunPod, "https://abc-8675.proxy.runpod.net.evil.example", false, false},
		{RunPod, "https://evil.example/abc-8675.proxy.runpod.net", false, false},
		{RunPod, "https://abc-8675.proxy.runpod.net:8443", false, false},
		{RunPod, "http://203.0.113.10:21829", false, false},
		// QuickPod: proxy (proxy, proxy1, proxy2) or a direct literal IP
		{QuickPod, "https://5e1f9f29-8675.proxy.quickpod.io", false, true},
		{QuickPod, "https://5e1f9f29-8675.proxy1.quickpod.io", false, true},
		{QuickPod, "https://5e1f9f29-8675.proxy2.quickpod.io", false, true},
		{QuickPod, "https://5e1f9f29-8675.proxy3.quickpod.io", false, false},
		{QuickPod, "https://5e1f9f29-8675.proxy.runpod.net", false, false},
		{QuickPod, "http://203.0.113.11:21829", false, true},
		{QuickPod, "http://node.quickpod.io:21829", false, false},
		// Vast: direct literal public IPs
		{Vast, "http://203.0.113.10:41234", false, true},
		{Vast, "http://[2001:db8::10]:41234", false, true},
		{Vast, "https://abc-8675.proxy.runpod.net", false, false},
		{Vast, "http://ssh4.vast.ai:41234", false, false},
		{Vast, "http://10.0.0.1:41234", false, false},
		{Vast, "http://192.168.1.5:41234", false, false},
		{Vast, "http://100.64.0.1:41234", false, false},
		{Vast, "http://169.254.169.254", false, false},
		{Vast, "http://127.0.0.1:41234", false, false},
		{Vast, "http://[::1]:41234", false, false},
		{Vast, "http://[fe80::1%25eth0]:41234", false, false},
		{Vast, "http://0.0.0.0:41234", false, false},
		// shape, for every kind
		{Vast, "ftp://203.0.113.10:41234", false, false},
		{Vast, "http://user:pw@203.0.113.10:41234", false, false},
		{Vast, "http://203.0.113.10:41234/v1", false, false},
		{Vast, "http://203.0.113.10:41234/", false, false},
		{Vast, "http://203.0.113.10:41234?x=1", false, false},
		{Vast, "http://203.0.113.10:41234#x", false, false},
		{Vast, "http://", false, false},
		{Vast, "203.0.113.10:41234", false, false},
		{"", "https://abc-8675.proxy.runpod.net", false, true},
		{"", "https://abc-8675.proxy2.quickpod.io", false, true},
		{"", "http://203.0.113.10:41234", false, true},
		{"", "http://example.com:41234", false, false},
		{"lambda", "http://203.0.113.10:41234", false, false},
		// simulated: http://127.0.0.1:<port> and nothing else
		{RunPod, "http://127.0.0.1:41000", true, true},
		{Vast, "http://127.0.0.1:41000", true, true},
		{"", "http://127.0.0.1:41000", true, true},
		{RunPod, "https://abc-8675.proxy.runpod.net", true, false},
		{Vast, "http://203.0.113.10:41234", true, false},
		{Vast, "https://127.0.0.1:41000", true, false},
		{Vast, "http://127.0.0.1", true, false},
		{Vast, "http://localhost:41000", true, false},
		{Vast, "http://[::1]:41000", true, false},
		{Vast, "http://127.0.0.2:41000", true, false},
		{Vast, "http://u:p@127.0.0.1:41000", true, false},
	}
	for _, c := range cases {
		err := CheckEndpoint(c.kind, c.url, c.sim)
		if (err == nil) != c.ok {
			t.Errorf("CheckEndpoint(%q, %q, sim=%v) = %v, want ok=%v", c.kind, c.url, c.sim, err, c.ok)
		}
		if err != nil && !errors.Is(err, ErrEndpoint) {
			t.Errorf("CheckEndpoint(%q, %q): %v does not wrap ErrEndpoint", c.kind, c.url, err)
		}
	}
}

// The adapters' own endpoint shapes pass their kind's rules.
func TestEndpointRulesAcceptAdapterShapes(t *testing.T) {
	for _, c := range []struct {
		kind Kind
		url  string
	}{
		{RunPod, "https://" + "k8s7q2xw3c1v9d" + "-8675.proxy.runpod.net"},
		{QuickPod, "https://" + "41000abc" + "-8675.proxy.quickpod.io"},
		{QuickPod, "http://[2001:db8::11]:21829"},
		{Vast, "http://203.0.113.10:20001"},
	} {
		if err := CheckEndpoint(c.kind, c.url, false); err != nil {
			t.Errorf("%s %s: %v", c.kind, c.url, err)
		}
	}
}

// The service narrows a machine's endpoints to its provider's rules before
// the probe, so the token never goes to an endpoint that provider cannot
// report (the probe itself checks only the union, kind "").
func TestEndpointFilterPerProvider(t *testing.T) {
	rpProxy := Endpoint{URL: "https://abc123-8675.proxy.runpod.net", Secure: true, Via: "proxy"}
	qpProxy := Endpoint{URL: "https://41000abc-8675.proxy.quickpod.io", Secure: true, Via: "proxy"}
	direct := Endpoint{URL: "http://203.0.113.10:20001", Via: "direct"}
	urls := func(eps []Endpoint) []string {
		var out []string
		for _, ep := range eps {
			out = append(out, ep.URL)
		}
		return out
	}
	for _, c := range []struct {
		kind Kind
		in   []Endpoint
		want []string
	}{
		{RunPod, []Endpoint{direct, rpProxy}, []string{rpProxy.URL}},
		{RunPod, []Endpoint{direct}, nil},
		{Vast, []Endpoint{qpProxy, rpProxy, direct}, []string{direct.URL}},
		{QuickPod, []Endpoint{qpProxy, direct, rpProxy}, []string{qpProxy.URL, direct.URL}},
		{"bogus", []Endpoint{direct}, nil},
	} {
		if got := urls(FilterEndpoints(c.kind, c.in, false)); !slices.Equal(got, c.want) {
			t.Errorf("%s %v: %v, want %v", c.kind, urls(c.in), got, c.want)
		}
	}
	// a simulated machine is its loopback node whatever the provider
	sim := Endpoint{URL: "http://127.0.0.1:40001", Via: "direct"}
	if got := FilterEndpoints(RunPod, []Endpoint{sim, rpProxy}, true); len(got) != 1 || got[0] != sim {
		t.Fatalf("simulated: %v", got)
	}

	// a RunPod record whose only endpoint is a direct IP is never probed
	n := &probeNode{healthStatus: 200, token: probeToken}
	u := n.start(t)
	res := probeAll(FilterEndpoints(RunPod, []Endpoint{{URL: u, Via: "direct"}}, false), false)
	if res.Health || res.TokenOK || n.hits.Load() != 0 {
		t.Fatalf("RunPod direct endpoint: %+v, %d requests", res, n.hits.Load())
	}
}
