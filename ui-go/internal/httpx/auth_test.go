package httpx

import "testing"

func TestNeedsToken(t *testing.T) {
	for _, c := range []struct {
		method, path string
		want         bool
	}{
		{"GET", "/api/img/a.png", false},
		{"HEAD", "/api/files/x.safetensors", false},
		{"GET", "/v1/health", false},
		{"POST", "/api/img/caption", true},
		{"POST", "/api/img/upload", true},
		{"POST", "/api/img/delete", true},
		{"POST", "/api/files/delete", true},
		{"GET", "/v1/jobs", true},
		{"POST", "/mcp", true},
		{"GET", "/studio/", false},
	} {
		if got := needsToken(c.method, c.path); got != c.want {
			t.Errorf("%s %s: needsToken = %v, want %v", c.method, c.path, got, c.want)
		}
	}
}
