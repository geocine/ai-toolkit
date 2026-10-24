package registrymock

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

// TestMain is the network tripwire: nothing here may use the default transport.
func TestMain(m *testing.M) {
	http.DefaultTransport = tripwire{}
	os.Exit(m.Run())
}

type tripwire struct{}

func (tripwire) RoundTrip(r *http.Request) (*http.Response, error) {
	panic("registrymock test reached the network: " + r.URL.Host)
}

func start(t *testing.T, f Faults) (*Server, *http.Client) {
	t.Helper()
	s := New(Options{Images: Fixtures(), Faults: f})
	s.StartTLS()
	t.Cleanup(s.Close)
	return s, s.Client()
}

func get(t *testing.T, c *http.Client, method, url, auth string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(method, url, nil)
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	resp, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func TestRegistryShapes(t *testing.T) {
	s, c := start(t, Faults{})

	// /v2/ challenges with the real realm.
	resp := get(t, c, "GET", "https://ghcr.io/v2/", "")
	if resp.StatusCode != 401 || !strings.Contains(resp.Header.Get("WWW-Authenticate"), `realm="https://ghcr.io/token"`) {
		t.Fatalf("/v2/: %d %q", resp.StatusCode, resp.Header.Get("WWW-Authenticate"))
	}
	// GHCR's token has no expires_in; Hub's does.
	resp = get(t, c, "GET", "https://ghcr.io/token?scope=repository:geocine/aitoolkit-go:pull&service=ghcr.io", "")
	var tok map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&tok)
	if _, ok := tok["expires_in"]; ok || tok["token"] == nil {
		t.Fatalf("GHCR token %v", tok)
	}
	resp = get(t, c, "GET", "https://auth.docker.io/token?scope=repository:ostris/aitoolkit:pull&service=registry.docker.io", "")
	tok = nil
	_ = json.NewDecoder(resp.Body).Decode(&tok)
	if tok["expires_in"] != float64(300) || tok["access_token"] == nil {
		t.Fatalf("Hub token %v", tok)
	}

	// Manifests need the token, carry their digest, and 404 in the real shape.
	url := "https://ghcr.io/v2/geocine/aitoolkit-go/manifests/latest"
	if resp := get(t, c, "HEAD", url, ""); resp.StatusCode != 401 {
		t.Fatalf("no token: %d", resp.StatusCode)
	}
	resp = get(t, c, "GET", url, "Bearer mock-token.geocine/aitoolkit-go")
	body, _ := io.ReadAll(resp.Body)
	sum := sha256.Sum256(body)
	if resp.Header.Get("Docker-Content-Digest") != "sha256:"+hex.EncodeToString(sum[:]) || resp.Header.Get("Content-Type") != MediaOCIIndex {
		t.Fatalf("manifest headers %v", resp.Header)
	}
	if s.Digest("", DefaultRepo, "latest") != s.Digest("", DefaultRepo, Revision) {
		t.Fatal("latest and the sha tag differ")
	}
	resp = get(t, c, "GET", "https://ghcr.io/v2/geocine/aitoolkit-go/manifests/nope", "Bearer mock-token.geocine/aitoolkit-go")
	body, _ = io.ReadAll(resp.Body)
	if resp.StatusCode != 404 || !strings.Contains(string(body), "MANIFEST_UNKNOWN") {
		t.Fatalf("missing tag: %d %s", resp.StatusCode, body)
	}

	// Hub sends rate-limit headers; registries send no CORS headers.
	resp = get(t, c, "HEAD", "https://registry-1.docker.io/v2/ostris/aitoolkit/manifests/latest", "Bearer mock-token.ostris/aitoolkit")
	if resp.Header.Get("ratelimit-limit") != "100;w=21600" || resp.Header.Get("ratelimit-remaining") != "100;w=21600" {
		t.Fatalf("Hub headers %v", resp.Header)
	}
	if resp := get(t, c, "OPTIONS", url, ""); resp.StatusCode != 405 || resp.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("OPTIONS: %d", resp.StatusCode)
	}
	if !strings.Contains(string(mustBody(t, get(t, c, "GET", "https://ghcr.io/v2/geocine/aitoolkit-go/tags/list", "Bearer mock-token.geocine/aitoolkit-go"))), Revision) {
		t.Fatal("tag list lacks the sha tag")
	}
}

func mustBody(t *testing.T, resp *http.Response) []byte {
	t.Helper()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestPrivateAndCDN(t *testing.T) {
	_, c := start(t, Faults{Private: map[string]bool{DefaultRepo: true}, RedirectBlobs: "cdn"})
	resp := get(t, c, "GET", "https://ghcr.io/token?scope=repository:geocine/aitoolkit-go:pull", "")
	if resp.StatusCode != 403 || !strings.Contains(string(mustBody(t, resp)), "DENIED") {
		t.Fatalf("private repo token: %d", resp.StatusCode)
	}

	// The CDN rejects a request that carries the registry token.
	_, c = start(t, Faults{RedirectBlobs: "cdn"})
	var cfgDigest string
	for _, img := range Fixtures() {
		if img.Repo == "geocine/aitoolkit" && img.Tag == "latest" {
			cfgDigest = digestOf(img.Config)
		}
	}
	resp = get(t, c, "GET", "https://ghcr.io/v2/geocine/aitoolkit/blobs/"+cfgDigest, "Bearer mock-token.geocine/aitoolkit")
	loc := resp.Header.Get("Location")
	if resp.StatusCode != 307 || !strings.HasPrefix(loc, "https://"+GHCRCDN+"/") {
		t.Fatalf("blob redirect: %d %q", resp.StatusCode, loc)
	}
	if resp := get(t, c, "GET", loc, "Bearer leaked"); resp.StatusCode != 400 {
		t.Fatalf("CDN with Authorization: %d", resp.StatusCode)
	}
	if resp := get(t, c, "GET", loc, ""); resp.StatusCode != 200 {
		t.Fatalf("CDN: %d", resp.StatusCode)
	}

	// Hosts the mock does not serve go through the dial guard.
	req, _ := http.NewRequest("GET", "https://"+PrivateCDN+"/x", nil)
	if _, err := c.Do(req); err == nil || !strings.Contains(err.Error(), "not allowed") {
		t.Fatalf("private CDN host: %v", err)
	}
}
