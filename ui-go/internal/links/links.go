// Package links signs download URLs (docs/headless.md, "Signed links"): a
// GET of exactly the signed path and query is served without a bearer token
// until the link expires. Browsers use them for multi-GB downloads (a plain
// <a href> streams to disk, no blob in memory) and agents to hand files to
// people. The HMAC key is generated once and kept in the fork database.
package links

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
)

// TTL bounds.
const (
	DefaultTTL = time.Hour
	MinTTL     = time.Minute
	MaxTTL     = 7 * 24 * time.Hour
)

const keySetting = "links.key"

var (
	keyMu sync.Mutex
	key   []byte
)

// signingKey loads (creating once) the HMAC key.
func signingKey(ctx context.Context) ([]byte, error) {
	keyMu.Lock()
	defer keyMu.Unlock()
	if key != nil {
		return key, nil
	}
	if state.DB == nil {
		return nil, errors.New("links: fork database not open")
	}
	v, ok, err := state.Setting(ctx, keySetting)
	if err != nil {
		return nil, err
	}
	if ok {
		if k, err := base64.RawStdEncoding.DecodeString(v); err == nil && len(k) >= 32 {
			key = k
			return key, nil
		}
	}
	k := make([]byte, 32)
	if _, err := rand.Read(k); err != nil {
		return nil, err
	}
	if err := state.SetSetting(ctx, keySetting, base64.RawStdEncoding.EncodeToString(k)); err != nil {
		return nil, err
	}
	key = k
	return key, nil
}

// Reset forgets the cached key (tests open a new database).
func Reset() {
	keyMu.Lock()
	key = nil
	keyMu.Unlock()
}

// targets are what a link may point at: downloads only.
var (
	exactTargets  = []string{"/v1/files/content", "/v1/bundles/content"}
	prefixTargets = []string{"/api/img/", "/api/files/", "/api/audio/art/"}
	bundleTarget  = regexp.MustCompile(`^/v1/jobs/[^/]+/bundle$`)
)

// Allowed reports whether a link may target the (escaped) path.
func Allowed(escapedPath string) bool {
	for _, t := range exactTargets {
		if escapedPath == t {
			return true
		}
	}
	for _, p := range prefixTargets {
		if strings.HasPrefix(escapedPath, p) {
			return true
		}
	}
	return bundleTarget.MatchString(escapedPath)
}

// ErrPath: the target is not a server path.
var ErrPath = errors.New("path must be a server path such as /v1/files/content?root=training&path=...")

// ErrTarget: the path is not something a link can point at.
var ErrTarget = errors.New("links can only point at GET /v1/files/content, GET /v1/jobs/{id}/bundle, GET /v1/bundles/content and the /api/img|files|audio/art media routes")

func mac(k []byte, escapedPath string, q url.Values) string {
	h := hmac.New(sha256.New, k)
	h.Write([]byte(escapedPath + "?" + q.Encode()))
	return base64.RawURLEncoding.EncodeToString(h.Sum(nil))
}

// Sign returns target ("/v1/...?a=b", a path with an optional query) with
// exp and sig appended, valid for ttl.
func Sign(ctx context.Context, target string, ttl time.Duration) (string, time.Time, error) {
	u, err := url.Parse(target)
	if err != nil || u.Scheme != "" || u.Host != "" || !strings.HasPrefix(u.Path, "/") {
		return "", time.Time{}, ErrPath
	}
	if !Allowed(u.EscapedPath()) {
		return "", time.Time{}, ErrTarget
	}
	k, err := signingKey(ctx)
	if err != nil {
		return "", time.Time{}, err
	}
	ttl = max(MinTTL, min(ttl, MaxTTL))
	exp := time.Now().Add(ttl).Truncate(time.Second)
	q := u.Query()
	q.Del("sig")
	q.Set("exp", strconv.FormatInt(exp.Unix(), 10))
	q.Set("sig", mac(k, u.EscapedPath(), q))
	return u.EscapedPath() + "?" + q.Encode(), exp.UTC(), nil
}

// Valid reports whether r is a GET/HEAD carrying a valid, unexpired
// signature for exactly its path and query.
func Valid(r *http.Request) bool {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		return false
	}
	q := r.URL.Query()
	sig := q.Get("sig")
	if sig == "" || len(q["sig"]) != 1 || !Allowed(r.URL.EscapedPath()) {
		return false
	}
	exp, err := strconv.ParseInt(q.Get("exp"), 10, 64)
	if err != nil || time.Now().Unix() > exp {
		return false
	}
	k, err := signingKey(r.Context())
	if err != nil {
		return false
	}
	q.Del("sig")
	want := mac(k, r.URL.EscapedPath(), q)
	return subtle.ConstantTimeCompare([]byte(sig), []byte(want)) == 1
}

// relayPrefix is the X-Forwarded-Prefix a controller's relay sets
// (internal/provision/relay); no other prefix is honoured.
var relayPrefix = regexp.MustCompile(`^/v1/machines/[A-Za-z0-9_]+/relay$`)

// Origin is the base URL clients reach this server at: AITK_PUBLIC_URL when
// set, else the scheme://host the request (or the proxy in front of it)
// says. Reached through a controller's relay, it is the controller's
// scheme://host plus the relay prefix, so absolute links point back through
// the relay.
func Origin(r *http.Request, publicURL string) string {
	if publicURL != "" {
		return strings.TrimRight(publicURL, "/")
	}
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	if p := r.Header.Get("X-Forwarded-Proto"); p != "" {
		scheme = strings.TrimSpace(strings.Split(p, ",")[0])
	}
	host := r.Host
	if h := r.Header.Get("X-Forwarded-Host"); h != "" {
		host = strings.TrimSpace(strings.Split(h, ",")[0])
	}
	origin := scheme + "://" + host
	if p := r.Header.Get("X-Forwarded-Prefix"); relayPrefix.MatchString(p) {
		origin += p
	}
	return origin
}
