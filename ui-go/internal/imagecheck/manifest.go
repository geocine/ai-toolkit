package imagecheck

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Media types the check reads. Anything else is a bad_response.
const (
	mediaOCIIndex     = "application/vnd.oci.image.index.v1+json"
	mediaDockerList   = "application/vnd.docker.distribution.manifest.list.v2+json"
	mediaOCIManifest  = "application/vnd.oci.image.manifest.v1+json"
	mediaDockerV2     = "application/vnd.docker.distribution.manifest.v2+json"
	manifestAccept    = mediaOCIIndex + ", " + mediaDockerList + ", " + mediaOCIManifest + ", " + mediaDockerV2
	attestationAnnKey = "vnd.docker.reference.type"
)

// Body caps (IMAGE-PLAN §2.3).
const (
	maxTokenBody    = 64 << 10
	maxManifestBody = 4 << 20
	maxConfigBody   = 1 << 20
	maxTagsBody     = 1 << 20
	maxRedirects    = 3
)

// unpackRatio is unpacked / compressed bytes for an image without the
// base_unpacked label (measured 2.08x on the fork base).
const unpackRatio = 2.1

// goLayers is how many layers ui-go/docker/Dockerfile adds above the base
// (aitk-server, web/dist, studio/dist, start-go.sh, the start.sh edit). The
// base layer count is not labelled, so the last goLayers layers are taken as
// "above the base": close enough for a disk estimate.
const goLayers = 5

// keptEnv are the only config env entries echoed back: nothing secret.
var keptEnv = []string{"CUDA_VERSION", "NVIDIA_REQUIRE_CUDA", "NVIDIA_DRIVER_CAPABILITIES"}

type descriptor struct {
	MediaType   string            `json:"mediaType"`
	Digest      string            `json:"digest"`
	Size        int64             `json:"size"`
	Annotations map[string]string `json:"annotations"`
	Platform    *struct {
		Architecture string `json:"architecture"`
		OS           string `json:"os"`
		Variant      string `json:"variant"`
	} `json:"platform"`
}

type manifest struct {
	SchemaVersion int          `json:"schemaVersion"`
	MediaType     string       `json:"mediaType"`
	Manifests     []descriptor `json:"manifests"` // index / list
	Config        descriptor   `json:"config"`    // image manifest
	Layers        []descriptor `json:"layers"`
}

type imageConfig struct {
	Architecture string    `json:"architecture"`
	OS           string    `json:"os"`
	Variant      string    `json:"variant"`
	Created      time.Time `json:"created"`
	Config       struct {
		Env          []string            `json:"Env"`
		Labels       map[string]string   `json:"Labels"`
		ExposedPorts map[string]struct{} `json:"ExposedPorts"`
	} `json:"config"`
	History []struct {
		CreatedBy string `json:"created_by"`
	} `json:"history"`
}

func (q *request) url(path string) string { return "https://" + q.reg.API + "/v2/" + q.ref.Repo + path }

func (q *request) manifestRef() string {
	if q.ref.Digest != "" {
		return q.ref.Digest
	}
	return q.ref.Tag
}

// send makes one request through the client. Network failures become
// KindUnreachable; the response body is the caller's to close.
func (q *request) send(ctx context.Context, method, u, accept string, bearer bool) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, nil)
	if err != nil {
		return nil, Errorf(KindBadResponse, "bad request URL")
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	if bearer && q.token != "" {
		req.Header.Set("Authorization", "Bearer "+q.token)
	}
	q.n.Add(1)
	resp, err := q.c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, &Error{Kind: KindUnreachable, Message: "the registry did not answer in time", cause: ctx.Err()}
		}
		return nil, &Error{Kind: KindUnreachable, Message: "couldn't reach " + req.URL.Host + ": " + scrub(err), cause: err}
	}
	return resp, nil
}

// scrub is a network error's text without any URL query string (signed CDN
// URLs carry credentials there).
func scrub(err error) string {
	var ue *url.Error
	if errors.As(err, &ue) {
		return ue.Err.Error()
	}
	return err.Error()
}

// statusError maps a registry status to a Kind. 2xx is nil.
func statusError(resp *http.Response, what string) *Error {
	s := resp.StatusCode
	switch {
	case s >= 200 && s < 300:
		return nil
	case s == http.StatusUnauthorized || s == http.StatusForbidden:
		return &Error{Kind: KindDenied, Status: s, Message: what + ": access denied (private or missing)"}
	case s == http.StatusNotFound:
		return &Error{Kind: KindNotFound, Status: s, Message: what + ": not found"}
	case s == http.StatusTooManyRequests:
		return &Error{Kind: KindRateLimited, Status: s, Message: what + ": rate limited"}
	case s >= 500:
		return &Error{Kind: KindUnreachable, Status: s, Message: what + ": registry error"}
	}
	return &Error{Kind: KindBadResponse, Status: s, Message: what + ": unexpected status"}
}

// readCapped reads at most limit bytes; more is a bad_response.
func readCapped(r io.Reader, limit int64, what string) ([]byte, error) {
	b, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, &Error{Kind: KindUnreachable, Message: what + ": " + scrub(err), cause: err}
	}
	if int64(len(b)) > limit {
		return nil, Errorf(KindBadResponse, "%s is larger than %d bytes", what, limit)
	}
	return b, nil
}

func sha(b []byte) string {
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}

// auth gets an anonymous pull token for the repo from the hardcoded realm,
// cached for expires_in-10 s (60 s when absent, as on GHCR). Basic
// credentials are never sent.
func (q *request) auth(ctx context.Context) error {
	key := q.reg.API + "|" + q.ref.Repo
	if t, ok := q.c.cache.token(key, q.c.now()); ok {
		q.token = t
		return nil
	}
	u := q.reg.Realm + "?" + url.Values{
		"service": {q.reg.Service},
		"scope":   {"repository:" + q.ref.Repo + ":pull"},
	}.Encode()
	resp, err := q.send(ctx, http.MethodGet, u, "application/json", false)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if e := statusError(resp, "token"); e != nil {
		return e
	}
	body, err := readCapped(resp.Body, maxTokenBody, "token response")
	if err != nil {
		return err
	}
	var tok struct {
		Token       string `json:"token"`
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
	}
	if json.Unmarshal(body, &tok) != nil {
		return Errorf(KindBadResponse, "token response is not JSON")
	}
	q.token = tok.Token
	if q.token == "" {
		q.token = tok.AccessToken
	}
	if q.token == "" {
		return Errorf(KindBadResponse, "token response has no token")
	}
	ttl := tokenTTL
	if tok.ExpiresIn > 10 {
		ttl = time.Duration(tok.ExpiresIn-10) * time.Second
	}
	q.c.cache.storeToken(key, q.token, ttl, q.c.now())
	return nil
}

// head resolves the tag (or digest) to the digest it points at, and reads
// the Docker Hub rate-limit headers. HEADs are free on Docker Hub.
func (q *request) head(ctx context.Context) (string, *RateLimit, error) {
	resp, err := q.send(ctx, http.MethodHead, q.url("/manifests/"+q.manifestRef()), manifestAccept, true)
	if err != nil {
		return "", nil, err
	}
	resp.Body.Close()
	if e := statusError(resp, "manifest"); e != nil {
		return "", nil, e
	}
	digest := resp.Header.Get("Docker-Content-Digest")
	if digest == "" {
		digest = q.ref.Digest
	}
	if digest == "" {
		return "", nil, Errorf(KindBadResponse, "the registry sent no Docker-Content-Digest")
	}
	if q.ref.Digest != "" && digest != q.ref.Digest {
		return "", nil, Errorf(KindBadResponse, "the registry answered a different digest")
	}
	return digest, parseRateLimit(resp.Header), nil
}

// parseRateLimit reads "ratelimit-limit: 100;w=21600" and friends; nil when
// the registry sent none (GHCR).
func parseRateLimit(h http.Header) *RateLimit {
	limit, window := splitLimit(h.Get("ratelimit-limit"))
	if limit < 0 {
		return nil
	}
	remaining, w2 := splitLimit(h.Get("ratelimit-remaining"))
	if window == 0 {
		window = w2
	}
	return &RateLimit{Limit: limit, Remaining: max(remaining, 0), WindowS: window, Source: h.Get("docker-ratelimit-source")}
}

func splitLimit(v string) (n, window int) {
	if v == "" {
		return -1, 0
	}
	num, rest, _ := strings.Cut(v, ";")
	n, err := strconv.Atoi(strings.TrimSpace(num))
	if err != nil {
		return -1, 0
	}
	if w, ok := strings.CutPrefix(strings.TrimSpace(rest), "w="); ok {
		window, _ = strconv.Atoi(w)
	}
	return n, window
}

// getManifest fetches a manifest and checks it hashes to want (when known).
func (q *request) getManifest(ctx context.Context, ref, want string) (*manifest, string, error) {
	resp, err := q.send(ctx, http.MethodGet, q.url("/manifests/"+ref), manifestAccept, true)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if e := statusError(resp, "manifest"); e != nil {
		return nil, "", e
	}
	body, err := readCapped(resp.Body, maxManifestBody, "manifest")
	if err != nil {
		return nil, "", err
	}
	got := sha(body)
	if h := resp.Header.Get("Docker-Content-Digest"); h != "" && h != got {
		return nil, "", Errorf(KindBadResponse, "the manifest does not hash to its Docker-Content-Digest")
	}
	if want != "" && want != got {
		return nil, "", Errorf(KindBadResponse, "the manifest does not hash to %s", want)
	}
	var m manifest
	if json.Unmarshal(body, &m) != nil {
		return nil, "", Errorf(KindBadResponse, "the manifest is not JSON")
	}
	if m.MediaType == "" {
		m.MediaType = strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0])
	}
	return &m, got, nil
}

// inspect runs the registry flow of IMAGE-PLAN §2.3 for one platform.
func (q *request) inspect(ctx context.Context, platform string) (*Image, error) {
	if err := q.auth(ctx); err != nil {
		return nil, err
	}
	digest, rl, err := q.head(ctx)
	if err != nil {
		return nil, err
	}
	if img, ok := q.c.cache.image(digest, platform); ok { // read before for this same platform
		img = img.clone()
		img.RateLimit = rl
		return img, nil
	}
	if q.reg.API == hubAPI && rl != nil && rl.Remaining <= hubMinRemaining {
		return nil, &Error{Kind: KindRateLimited, Message: fmt.Sprintf("Docker Hub's anonymous limit has %d pulls left", rl.Remaining)}
	}

	top, _, err := q.getManifest(ctx, digest, digest)
	if err != nil {
		return nil, err
	}
	img := &Image{
		Repo: q.ref.Name(), Tag: q.ref.Tag, Digest: digest, Platform: platform, RateLimit: rl,
		Platforms: []string{}, LayerDigests: []string{}, ExposedPorts: []string{},
	}
	m := top
	switch top.MediaType {
	case mediaOCIIndex, mediaDockerList:
		var pick *descriptor
		for i, d := range top.Manifests {
			if d.Platform == nil || d.Annotations[attestationAnnKey] == "attestation-manifest" ||
				d.Platform.OS == "unknown" || d.Platform.Architecture == "unknown" {
				continue
			}
			name := d.Platform.OS + "/" + d.Platform.Architecture
			if d.Platform.Variant != "" {
				name += "/" + d.Platform.Variant
			}
			img.Platforms = append(img.Platforms, name)
			if pick == nil && platformMatches(platform, d.Platform.OS, d.Platform.Architecture, d.Platform.Variant) {
				pick = &top.Manifests[i]
			}
		}
		if pick == nil {
			return nil, &Error{Kind: KindNoPlatform, Message: "no " + platform + " manifest (has " + strings.Join(img.Platforms, ", ") + ")",
				Platforms: slices.Clone(img.Platforms)}
		}
		if m, _, err = q.getManifest(ctx, pick.Digest, pick.Digest); err != nil {
			return nil, err
		}
		img.PlatformDigest = pick.Digest
		if m.MediaType != mediaOCIManifest && m.MediaType != mediaDockerV2 {
			return nil, Errorf(KindBadResponse, "the platform manifest has media type %q", m.MediaType)
		}
	case mediaOCIManifest, mediaDockerV2:
		img.PlatformDigest = digest
	default:
		return nil, Errorf(KindBadResponse, "unsupported manifest media type %q", top.MediaType)
	}

	// Layers: unique blobs only (an empty layer can appear twice).
	seen := map[string]bool{}
	var sizes []int64
	for _, l := range m.Layers {
		img.LayerDigests = append(img.LayerDigests, l.Digest)
		if seen[l.Digest] {
			continue
		}
		seen[l.Digest] = true
		sizes = append(sizes, l.Size)
		img.DownloadBytes += l.Size
		img.LargestLayer = max(img.LargestLayer, l.Size)
	}
	img.Layers = len(m.Layers)

	if m.Config.Size > maxConfigBody {
		return nil, Errorf(KindBadResponse, "the config blob declares %d bytes (limit %d)", m.Config.Size, maxConfigBody)
	}
	raw, err := q.blob(ctx, m.Config.Digest)
	if err != nil {
		return nil, err
	}
	var cfg imageConfig
	if json.Unmarshal(raw, &cfg) != nil {
		return nil, Errorf(KindBadResponse, "the config blob is not JSON")
	}
	if len(img.Platforms) == 0 { // a bare manifest: the config says what it is for
		name := cfg.OS + "/" + cfg.Architecture
		if cfg.Variant != "" {
			name += "/" + cfg.Variant
		}
		img.Platforms = []string{name}
		if !platformMatches(platform, cfg.OS, cfg.Architecture, cfg.Variant) {
			return nil, &Error{Kind: KindNoPlatform, Message: "no " + platform + " manifest (has " + name + ")", Platforms: []string{name}}
		}
	}
	fillConfig(img, &cfg)
	img.Requires = Derive(img)
	img.GoServer = img.Requires.API >= 1
	img.UnpackedBytes, img.UnpackedSource = unpacked(img, sizes)
	img.CheckedAt = q.c.now().UTC()
	return img, nil
}

// platformMatches compares "os/arch[/variant]"; the variant is ignored for amd64.
func platformMatches(want, os, arch, variant string) bool {
	parts := strings.Split(want, "/")
	if len(parts) < 2 || parts[0] != os || parts[1] != arch {
		return false
	}
	return len(parts) < 3 || arch == "amd64" || parts[2] == variant
}

// fillConfig copies the allowed parts of the config: labels with the two
// known prefixes, three env names, exposed ports, and whether any history
// line installs aitk-server (the history itself is dropped).
func fillConfig(img *Image, cfg *imageConfig) {
	img.Created = cfg.Created
	img.Labels = map[string]string{}
	for k, v := range cfg.Config.Labels {
		if strings.HasPrefix(k, labelPrefixOCI) || strings.HasPrefix(k, labelPrefixAITK) {
			img.Labels[k] = v
		}
	}
	img.Env = map[string]string{}
	for _, kv := range cfg.Config.Env {
		k, v, _ := strings.Cut(kv, "=")
		if slices.Contains(keptEnv, k) {
			img.Env[k] = v
		}
	}
	for p := range cfg.Config.ExposedPorts {
		img.ExposedPorts = append(img.ExposedPorts, p)
	}
	slices.Sort(img.ExposedPorts)
	img.Revision = strings.TrimSpace(img.Labels[LabelRevision])
	img.GoServer = img.Labels[LabelServer] == "go"
	for _, h := range cfg.History {
		if strings.Contains(h.CreatedBy, "aitk-server") {
			img.GoServer = true
			break
		}
	}
}

// unpacked estimates the bytes on disk after extraction: the labelled base
// size plus 2.1x the Go layers, else 2.1x the whole download. A label under
// 1.5x the base's compressed layers is not trusted: under Docker's
// containerd image store the build script read the compressed size there
// (6.6 GB for a 13.7 GB base).
func unpacked(img *Image, sizes []int64) (int64, string) {
	if base, err := strconv.ParseInt(strings.TrimSpace(img.Labels[LabelBaseUnpacked]), 10, 64); err == nil && base > 0 {
		var top, below int64
		cut := max(len(sizes)-goLayers, 0)
		for _, s := range sizes[cut:] {
			top += s
		}
		for _, s := range sizes[:cut] {
			below += s
		}
		if base*2 >= below*3 {
			return base + int64(float64(top)*unpackRatio), "label"
		}
	}
	return int64(float64(img.DownloadBytes) * unpackRatio), "estimate"
}

// blob fetches a blob (the config) by digest. Registries answer with a
// redirect to a signed CDN URL; it is followed by hand, at most 3 hops, only
// to https, without the Authorization header, and always through the
// client's dial guard.
func (q *request) blob(ctx context.Context, digest string) ([]byte, error) {
	u := q.url("/blobs/" + digest)
	bearer := true
	for hop := 0; ; hop++ {
		resp, err := q.send(ctx, http.MethodGet, u, "", bearer)
		if err != nil {
			return nil, err
		}
		switch resp.StatusCode {
		case http.StatusMovedPermanently, http.StatusFound, http.StatusSeeOther, http.StatusTemporaryRedirect, http.StatusPermanentRedirect:
			loc := resp.Header.Get("Location")
			resp.Body.Close()
			if hop >= maxRedirects {
				return nil, Errorf(KindBadResponse, "the config blob redirected more than %d times", maxRedirects)
			}
			next, err := url.Parse(u)
			if err == nil {
				next, err = next.Parse(loc)
			}
			if err != nil || loc == "" {
				return nil, Errorf(KindBadResponse, "the config blob redirect has no usable Location")
			}
			if next.Scheme != "https" {
				return nil, Errorf(KindBadResponse, "the config blob redirected to %s, not https", next.Scheme)
			}
			u, bearer = next.String(), false
			continue
		}
		defer resp.Body.Close()
		if e := statusError(resp, "config blob"); e != nil {
			return nil, e
		}
		body, err := readCapped(resp.Body, maxConfigBody, "config blob")
		if err != nil {
			return nil, err
		}
		if sha(body) != digest {
			return nil, Errorf(KindBadResponse, "the config blob does not hash to %s", digest)
		}
		return body, nil
	}
}

// tagList is the repo's tag list (one page of up to 200).
func (q *request) tagList(ctx context.Context) ([]string, error) {
	if err := q.auth(ctx); err != nil {
		return nil, err
	}
	resp, err := q.send(ctx, http.MethodGet, q.url("/tags/list?n=200"), "application/json", true)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if e := statusError(resp, "tag list"); e != nil {
		return nil, e
	}
	body, err := readCapped(resp.Body, maxTagsBody, "tag list")
	if err != nil {
		return nil, err
	}
	var tl struct {
		Tags []string `json:"tags"`
	}
	if json.Unmarshal(body, &tl) != nil {
		return nil, Errorf(KindBadResponse, "the tag list is not JSON")
	}
	return tl.Tags, nil
}
