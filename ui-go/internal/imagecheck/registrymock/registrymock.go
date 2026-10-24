// Package registrymock is a fake GHCR and Docker Hub for the image check:
// token endpoints, manifests (OCI index, Docker manifest list, bare
// manifest), config blobs behind CDN redirects, tag lists and Docker Hub
// rate-limit headers, shaped like the real registries' answers.
//
// It is a non-test package so simulate mode can mount it in-process
// (imagecheck.Init(true)). It never imports imagecheck.
package registrymock

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/netguard"
)

// Registry and CDN hosts the mock answers for. Client() sends them all here.
const (
	GHCR       = "ghcr.io"
	Hub        = "registry-1.docker.io"
	HubAuth    = "auth.docker.io"
	GHCRCDN    = "pkg-containers.githubusercontent.com"
	Cloudfront = "production.cloudfront.docker.com"
	Cloudflare = "production.cloudflare.docker.com"
	// PrivateCDN is a CDN host that resolves to 10.0.0.1 (RedirectBlobs
	// "private"): the client's dial guard must refuse it.
	PrivateCDN = "cdn.internal.test"
)

var mockHosts = map[string]bool{GHCR: true, Hub: true, HubAuth: true, GHCRCDN: true, Cloudfront: true, Cloudflare: true}

// Media types the mock serves.
const (
	MediaOCIIndex    = "application/vnd.oci.image.index.v1+json"
	MediaDockerList  = "application/vnd.docker.distribution.manifest.list.v2+json"
	MediaOCIManifest = "application/vnd.oci.image.manifest.v1+json"
	MediaDockerV2    = "application/vnd.docker.distribution.manifest.v2+json"
	mediaOCIConfig   = "application/vnd.oci.image.config.v1+json"
	mediaDockerCfg   = "application/vnd.docker.container.image.v1+json"
	mediaOCILayer    = "application/vnd.oci.image.layer.v1.tar+gzip"
	mediaDockerLayer = "application/vnd.docker.image.rootfs.diff.tar.gzip"
)

// Layer is one layer blob of a manifest. Only the size matters: layer blob
// GETs answer 404 (the check never fetches them).
type Layer struct {
	Digest string
	Size   int64
}

// Image is one tag the mock serves.
type Image struct {
	Host          string   // GHCR (default) or Hub
	Repo, Tag     string   // "geocine/aitoolkit-go", "f53a407"
	Platforms     []string // default ["linux/amd64"]; an attestation entry is added unless NoAttestation
	Config        []byte   // a real config JSON; its digest is computed
	Layers        []Layer  // layer list from a real manifest; wins over LayerSizes
	LayerSizes    []int64  // fake layers when Layers is empty: digests are made up from repo, position and size
	Index         bool     // serve an index (true) or a bare manifest (false)
	SchemaV2List  bool     // Docker media types (manifest list v2) instead of OCI
	NoAttestation bool
}

// Faults bends the mock away from the happy path.
type Faults struct {
	Private        map[string]bool // repo -> 403 DENIED at the token endpoint (GHCR shape)
	SlowMs         int             // delay every response (cut short when the request is cancelled)
	RateLimitAfter int             // Docker Hub: 429 after N manifest GETs; HEAD never counts
	Remaining      int             // starting Docker Hub ratelimit-remaining (default 100), with "100;w=21600"
	RedirectBlobs  string          // "" | "cdn" (the host's own CDN) | "cloudfront" | "cloudflare" | "http" (insecure) | "loop" | "private"
	BadDigest      bool            // manifest GETs answer a body that doesn't hash to its digest
	OversizeConfig bool            // manifests declare a config size above 1 MiB
	Status         map[string]int  // path -> forced status (e.g. 500, 502)
	Realm          string          // realm advertised in WWW-Authenticate (default the real one); the client must ignore it. Client() routes its host here, so a dial shows in Requests
	Challenge      bool            // every /v2/ request answers the 401 challenge, even with a valid token (a token the registry no longer honours)
}

// Options configures New.
type Options struct {
	Images []Image
	Faults Faults
}

// Request is one request the mock saw. The token value is never kept.
type Request struct {
	Method  string
	Host    string
	Path    string
	Accept  string
	HasAuth bool
	Basic   bool // an Authorization: Basic header (the check must never send one)
}

type served struct {
	body      []byte
	mediaType string
}

// Server is the fake registry: an http.Handler routing by Host header.
type Server struct {
	faults    Faults
	mu        sync.Mutex
	manifests map[string]served // host|repo|ref (tag or digest)
	blobs     map[string][]byte // host|repo|digest
	tags      map[string][]string
	requests  []Request
	gets      int
	remaining int
	ts        *httptest.Server
}

// New builds the mock from images. Start it with StartTLS.
func New(o Options) *Server {
	s := &Server{
		faults:    o.Faults,
		manifests: map[string]served{},
		blobs:     map[string][]byte{},
		tags:      map[string][]string{},
		remaining: o.Faults.Remaining,
	}
	if s.remaining == 0 {
		s.remaining = 100
	}
	for _, img := range o.Images {
		s.add(img)
	}
	return s
}

func digestOf(b []byte) string {
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}

func (s *Server) add(img Image) {
	host := img.Host
	if host == "" {
		host = GHCR
	}
	platforms := img.Platforms
	if len(platforms) == 0 {
		platforms = []string{"linux/amd64"}
	}
	manifestType, indexType, configType, layerType := MediaOCIManifest, MediaOCIIndex, mediaOCIConfig, mediaOCILayer
	if img.SchemaV2List {
		manifestType, indexType, configType, layerType = MediaDockerV2, MediaDockerList, mediaDockerCfg, mediaDockerLayer
	}
	layers := img.Layers
	if len(layers) == 0 {
		for i, size := range img.LayerSizes {
			layers = append(layers, Layer{digestOf(fmt.Appendf(nil, "layer|%s|%d|%d", img.Repo, i, size)), size})
		}
	}
	prefix := host + "|" + img.Repo + "|"

	type desc struct {
		MediaType   string            `json:"mediaType"`
		Digest      string            `json:"digest"`
		Size        int64             `json:"size"`
		Annotations map[string]string `json:"annotations,omitempty"`
		Platform    map[string]string `json:"platform,omitempty"`
	}
	var children []desc
	var top served
	for _, p := range platforms {
		parts := strings.Split(p, "/")
		cfg := configFor(img.Config, parts)
		cfgDigest := digestOf(cfg)
		s.blobs[prefix+cfgDigest] = cfg
		cfgSize := int64(len(cfg))
		if s.faults.OversizeConfig {
			cfgSize = 2 << 20
		}
		ls := make([]desc, len(layers))
		for i, l := range layers {
			ls[i] = desc{MediaType: layerType, Digest: l.Digest, Size: l.Size}
		}
		body, _ := json.MarshalIndent(map[string]any{
			"schemaVersion": 2,
			"mediaType":     manifestType,
			"config":        desc{MediaType: configType, Digest: cfgDigest, Size: cfgSize},
			"layers":        ls,
		}, "", "  ")
		d := digestOf(body)
		s.manifests[prefix+d] = served{body, manifestType}
		plat := map[string]string{"architecture": parts[1], "os": parts[0]}
		if len(parts) > 2 {
			plat["variant"] = parts[2]
		}
		children = append(children, desc{MediaType: manifestType, Digest: d, Size: int64(len(body)), Platform: plat})
		if top.body == nil {
			top = served{body, manifestType}
		}
	}
	if img.Index {
		if !img.NoAttestation {
			att, _ := json.Marshal(map[string]any{"schemaVersion": 2, "mediaType": MediaOCIManifest,
				"config": desc{MediaType: "application/vnd.oci.empty.v1+json", Digest: digestOf([]byte("{}")), Size: 2}, "layers": []desc{}})
			ad := digestOf(att)
			s.manifests[prefix+ad] = served{att, MediaOCIManifest}
			children = append(children, desc{MediaType: MediaOCIManifest, Digest: ad, Size: int64(len(att)),
				Annotations: map[string]string{"vnd.docker.reference.digest": children[0].Digest, "vnd.docker.reference.type": "attestation-manifest"},
				Platform:    map[string]string{"architecture": "unknown", "os": "unknown"}})
		}
		body, _ := json.MarshalIndent(map[string]any{"schemaVersion": 2, "mediaType": indexType, "manifests": children}, "", "  ")
		top = served{body, indexType}
		s.manifests[prefix+digestOf(body)] = top
	}
	s.manifests[prefix+img.Tag] = top
	s.tags[host+"|"+img.Repo] = append(s.tags[host+"|"+img.Repo], img.Tag)
}

// configFor is cfg for one platform: the bytes as given when they already
// say that platform, else a copy with architecture/os/variant rewritten.
func configFor(cfg []byte, platform []string) []byte {
	var m map[string]any
	if json.Unmarshal(cfg, &m) != nil {
		return cfg
	}
	variant, _ := m["variant"].(string)
	want := ""
	if len(platform) > 2 {
		want = platform[2]
	}
	if m["os"] == platform[0] && m["architecture"] == platform[1] && variant == want {
		return cfg
	}
	m["os"], m["architecture"] = platform[0], platform[1]
	delete(m, "variant")
	if want != "" {
		m["variant"] = want
	}
	out, _ := json.Marshal(m)
	return out
}

// Digest is the digest a tag points at (index or manifest), "" when unknown.
func (s *Server) Digest(host, repo, tag string) string {
	if host == "" {
		host = GHCR
	}
	if m, ok := s.manifests[host+"|"+repo+"|"+tag]; ok {
		return digestOf(m.body)
	}
	return ""
}

// StartTLS starts an httptest TLS server and returns its 127.0.0.1:port.
func (s *Server) StartTLS() string {
	s.ts = httptest.NewUnstartedServer(s)
	s.ts.StartTLS()
	return s.ts.Listener.Addr().String()
}

// Close stops the server.
func (s *Server) Close() {
	if s.ts != nil {
		s.ts.Close()
	}
}

// Client trusts the mock's certificate and dials every registry and CDN host
// it serves to the mock. Any other host goes through the netguard dial guard
// with a static resolver (PrivateCDN -> 10.0.0.1), so it is refused rather
// than reaching the network. It never follows redirects.
func (s *Server) Client() *http.Client {
	pool := x509.NewCertPool()
	pool.AddCert(s.ts.Certificate())
	addr := s.ts.Listener.Addr().String()
	realmHost := ""
	if u, err := url.Parse(s.faults.Realm); err == nil {
		realmHost = u.Hostname()
	}
	direct := &net.Dialer{Timeout: 5 * time.Second}
	guarded := &net.Dialer{
		Timeout:  5 * time.Second,
		Resolver: netguard.StaticResolver(map[string][]netip.Addr{PrivateCDN: {netip.MustParseAddr("10.0.0.1")}}),
		Control:  netguard.Control(false),
	}
	return &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, hostport string) (net.Conn, error) {
				host, _, _ := net.SplitHostPort(hostport)
				if mockHosts[host] || (realmHost != "" && host == realmHost) {
					return direct.DialContext(ctx, network, addr)
				}
				return guarded.DialContext(ctx, network, hostport)
			},
			// httptest's certificate names example.com; every mock host shares it.
			TLSClientConfig:       &tls.Config{RootCAs: pool, ServerName: "example.com"},
			ResponseHeaderTimeout: 10 * time.Second,
		},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

// Requests is every request seen so far, in order.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Request(nil), s.requests...)
}

// Reset forgets the recorded requests.
func (s *Server) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = nil
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	host := r.Host
	if h, _, err := net.SplitHostPort(host); err == nil {
		host = h
	}
	auth := r.Header.Get("Authorization")
	s.mu.Lock()
	s.requests = append(s.requests, Request{Method: r.Method, Host: host, Path: r.URL.Path, Accept: r.Header.Get("Accept"),
		HasAuth: auth != "", Basic: strings.HasPrefix(auth, "Basic ")})
	s.mu.Unlock()

	if s.faults.SlowMs > 0 {
		select {
		case <-time.After(time.Duration(s.faults.SlowMs) * time.Millisecond):
		case <-r.Context().Done():
			return
		}
	}
	if st, ok := s.faults.Status[r.URL.Path]; ok {
		http.Error(w, http.StatusText(st), st)
		return
	}
	if r.Method == http.MethodOptions { // registries send no CORS headers
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	switch host {
	case GHCR:
		if r.URL.Path == "/token" {
			s.token(w, r, GHCR)
			return
		}
		s.registry(w, r, GHCR)
	case HubAuth:
		s.token(w, r, Hub)
	case Hub:
		s.registry(w, r, Hub)
	case GHCRCDN, Cloudfront, Cloudflare:
		s.cdn(w, r)
	default:
		http.NotFound(w, r)
	}
}

func writeErrors(w http.ResponseWriter, status int, code, message, detail string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"errors": []map[string]string{{"code": code, "message": message, "detail": detail}}})
}

// token answers the anonymous token flow: GHCR's {"token"} shape (no
// expires_in) or Docker Hub's {token, access_token, expires_in: 300, issued_at}.
func (s *Server) token(w http.ResponseWriter, r *http.Request, registry string) {
	scope := r.URL.Query().Get("scope") // repository:<repo>:pull
	repo := strings.TrimSuffix(strings.TrimPrefix(scope, "repository:"), ":pull")
	if repo == "" || repo == scope {
		writeErrors(w, http.StatusBadRequest, "UNSUPPORTED", "bad scope", scope)
		return
	}
	_, known := s.tags[registry+"|"+repo]
	// GHCR answers 403 DENIED for private and for missing repos alike.
	if s.faults.Private[repo] || (registry == GHCR && !known) {
		writeErrors(w, http.StatusForbidden, "DENIED", "requested access to the resource is denied", "")
		return
	}
	tok := "mock-token." + repo
	w.Header().Set("Content-Type", "application/json")
	if registry == GHCR {
		_ = json.NewEncoder(w).Encode(map[string]string{"token": tok})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"token": tok, "access_token": tok, "expires_in": 300, "issued_at": time.Now().UTC().Format(time.RFC3339)})
}

func (s *Server) challenge(w http.ResponseWriter, registry, repo string) {
	realm, service := "https://ghcr.io/token", "ghcr.io"
	if registry == Hub {
		realm, service = "https://auth.docker.io/token", "registry.docker.io"
	}
	if s.faults.Realm != "" {
		realm = s.faults.Realm
	}
	v := fmt.Sprintf(`Bearer realm=%q,service=%q`, realm, service)
	if repo != "" {
		v += fmt.Sprintf(`,scope="repository:%s:pull"`, repo)
	}
	w.Header().Set("WWW-Authenticate", v)
	writeErrors(w, http.StatusUnauthorized, "UNAUTHORIZED", "authentication required", "")
}

// registry serves /v2/: manifests, blobs and tag lists, behind a bearer token.
func (s *Server) registry(w http.ResponseWriter, r *http.Request, registry string) {
	path := strings.TrimPrefix(r.URL.Path, "/v2/")
	if path == "" || path == r.URL.Path {
		s.challenge(w, registry, "")
		return
	}
	var repo, kind, ref string
	for _, k := range []string{"/manifests/", "/blobs/", "/tags/list"} {
		if i := strings.LastIndex(path, k); i > 0 {
			repo, kind, ref = path[:i], strings.Trim(k, "/"), path[i+len(k):]
			break
		}
	}
	if repo == "" {
		http.NotFound(w, r)
		return
	}
	if s.faults.Challenge || r.Header.Get("Authorization") != "Bearer mock-token."+repo {
		s.challenge(w, registry, repo)
		return
	}
	switch kind {
	case "manifests":
		s.manifest(w, r, registry, repo, ref)
	case "blobs":
		s.blob(w, r, registry, repo, ref)
	default:
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"name": repo, "tags": s.tags[registry+"|"+repo]})
	}
}

func (s *Server) manifest(w http.ResponseWriter, r *http.Request, registry, repo, ref string) {
	m, ok := s.manifests[registry+"|"+repo+"|"+ref]
	if !ok {
		writeErrors(w, http.StatusNotFound, "MANIFEST_UNKNOWN", "manifest unknown", "unknown tag="+ref)
		return
	}
	if registry == Hub {
		s.mu.Lock()
		if r.Method == http.MethodGet {
			if s.faults.RateLimitAfter > 0 && s.gets >= s.faults.RateLimitAfter {
				s.mu.Unlock()
				writeErrors(w, http.StatusTooManyRequests, "TOOMANYREQUESTS", "You have reached your unauthenticated pull rate limit.", "")
				return
			}
			s.gets++
			s.remaining = max(s.remaining-1, 0)
		}
		remaining := s.remaining
		s.mu.Unlock()
		w.Header().Set("ratelimit-limit", "100;w=21600")
		w.Header().Set("ratelimit-remaining", strconv.Itoa(remaining)+";w=21600")
		w.Header().Set("docker-ratelimit-source", "203.0.113.7")
	}
	body := m.body
	w.Header().Set("Content-Type", m.mediaType)
	w.Header().Set("Docker-Content-Digest", digestOf(body))
	if s.faults.BadDigest && r.Method == http.MethodGet {
		body = append(append([]byte(nil), body...), '\n')
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	if r.Method == http.MethodHead {
		return
	}
	_, _ = w.Write(body)
}

// blob serves config blobs, or redirects to the CDN the way GHCR (307 to
// pkg-containers.githubusercontent.com) and Docker Hub (to cloudfront or
// cloudflare) do.
func (s *Server) blob(w http.ResponseWriter, r *http.Request, registry, repo, digest string) {
	key := registry + "|" + repo + "|" + digest
	b, ok := s.blobs[key]
	if !ok {
		writeErrors(w, http.StatusNotFound, "BLOB_UNKNOWN", "blob unknown to registry", digest)
		return
	}
	cdn := ""
	switch s.faults.RedirectBlobs {
	case "":
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(b)
		return
	case "cdn":
		cdn = "https://" + GHCRCDN
		if registry == Hub {
			cdn = "https://" + Cloudfront
		}
	case "cloudfront", "loop":
		cdn = "https://" + Cloudfront
	case "cloudflare":
		cdn = "https://" + Cloudflare
	case "http":
		cdn = "http://" + Cloudfront
	case "private":
		cdn = "https://" + PrivateCDN
	}
	loc := cdn + "/blob/" + url.PathEscape(key) + "?sig=mock-signature&expires=600"
	if s.faults.RedirectBlobs == "loop" {
		loc = cdn + "/loop/0?k=" + url.QueryEscape(key)
	}
	http.Redirect(w, r, loc, http.StatusTemporaryRedirect)
}

// cdn serves signed blob URLs. A request carrying Authorization is refused,
// so a client that leaks the registry token fails the test.
func (s *Server) cdn(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "" {
		http.Error(w, "Only one auth mechanism allowed; only the X-Amz-Algorithm query parameter, Signature query string parameter or the Authorization header should be specified", http.StatusBadRequest)
		return
	}
	if n, ok := strings.CutPrefix(r.URL.Path, "/loop/"); ok {
		i, _ := strconv.Atoi(n)
		http.Redirect(w, r, fmt.Sprintf("/loop/%d?k=%s", i+1, url.QueryEscape(r.URL.Query().Get("k"))), http.StatusTemporaryRedirect)
		return
	}
	key, err := url.PathUnescape(strings.TrimPrefix(r.URL.Path, "/blob/"))
	if err != nil || r.URL.Query().Get("sig") == "" {
		http.Error(w, "AccessDenied", http.StatusForbidden)
		return
	}
	b, ok := s.blobs[key]
	if !ok {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	_, _ = w.Write(b)
}
