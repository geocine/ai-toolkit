package notify

import "testing"

// PublicURL derives the server's address from each provider's env, in order.
func TestPublicURLFromProvider(t *testing.T) {
	for _, k := range []string{"PORT", "RUNPOD_POD_ID", "PUBLIC_IPADDR", "VAST_TCP_PORT_8675", "QUICKPOD_PORT_8675", "QUICKPOD_PORT_9000"} {
		t.Setenv(k, "")
	}
	if got := PublicURL(Config{}); got != "" {
		t.Fatalf("no provider env: %q", got)
	}
	if got := PublicURL(Config{PublicURL: "https://srv.example/"}); got != "https://srv.example" {
		t.Fatalf("configured: %q", got)
	}

	// QuickPod: the host IP and the external port mapped to 8675
	t.Setenv("PUBLIC_IPADDR", "203.0.113.7")
	if got := PublicURL(Config{}); got != "" {
		t.Fatalf("an IP alone: %q", got)
	}
	t.Setenv("QUICKPOD_PORT_8675", "40123")
	if got := PublicURL(Config{}); got != "http://203.0.113.7:40123" {
		t.Fatalf("quickpod: %q", got)
	}
	// it follows PORT like Vast.ai does
	t.Setenv("PORT", "9000")
	if got := PublicURL(Config{}); got != "" {
		t.Fatalf("quickpod with PORT=9000 and no mapping: %q", got)
	}
	t.Setenv("QUICKPOD_PORT_9000", "40200")
	if got := PublicURL(Config{}); got != "http://203.0.113.7:40200" {
		t.Fatalf("quickpod with PORT=9000: %q", got)
	}
	t.Setenv("PORT", "")

	// Vast.ai comes first when both map the port, RunPod before either
	t.Setenv("VAST_TCP_PORT_8675", "31000")
	if got := PublicURL(Config{}); got != "http://203.0.113.7:31000" {
		t.Fatalf("vast: %q", got)
	}
	t.Setenv("RUNPOD_POD_ID", "abc123")
	if got := PublicURL(Config{}); got != "https://abc123-8675.proxy.runpod.net" {
		t.Fatalf("runpod: %q", got)
	}
	if got := PublicURL(Config{PublicURL: "https://srv.example"}); got != "https://srv.example" {
		t.Fatalf("configured wins: %q", got)
	}
}
