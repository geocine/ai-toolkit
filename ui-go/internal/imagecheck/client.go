package imagecheck

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"regexp"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/netguard"
)

// Registry is one registry the check may talk to. Realms are hardcoded and
// never read from WWW-Authenticate, so a registry cannot point the server at
// another host.
type Registry struct{ API, Realm, Service string }

// DefaultRegistries is the allowlist: GHCR and Docker Hub. Anything else is
// a warn ("can't be checked") and no request is made.
var DefaultRegistries = map[string]Registry{
	"ghcr.io":              {API: "ghcr.io", Realm: "https://ghcr.io/token", Service: "ghcr.io"},
	"registry-1.docker.io": {API: "registry-1.docker.io", Realm: "https://auth.docker.io/token", Service: "registry.docker.io"},
}

const (
	defaultPlatform     = "linux/amd64"
	defaultCheckTimeout = 15 * time.Second
	hubAPI              = "registry-1.docker.io"
	hubMinRemaining     = 5 // stop before a GET that would count against the server's IP
)

// Options configures New. The zero value is the production client.
type Options struct {
	HTTP         *http.Client        // nil = netguard client (public addresses only, proxy honoured, no redirects)
	Registries   map[string]Registry // nil = DefaultRegistries
	Now          func() time.Time    // tests: fake clock
	CheckTimeout time.Duration       // whole Inspect; default 15 s
	Logf         func(format string, args ...any)
}

// Client reads image facts from registries with anonymous pulls, through
// in-memory caches shared by every caller.
type Client struct {
	http    *http.Client
	regs    map[string]Registry
	now     func() time.Time
	timeout time.Duration
	logf    func(format string, args ...any)
	cache   *cache
}

// New builds a client. Without o.HTTP it dials only public addresses
// (netguard), honours HTTP(S)_PROXY, and follows no redirect on its own.
func New(o Options) *Client {
	c := &Client{http: o.HTTP, regs: o.Registries, now: o.Now, timeout: o.CheckTimeout, logf: o.Logf, cache: newCache()}
	if c.http == nil {
		c.http = netguard.NewClient(netguard.Options{
			Proxy:              true,
			AllowLoopbackProxy: os.Getenv("AITK_IMAGECHECK_ALLOW_LOOPBACK_PROXY") == "1",
			Timeout:            netguard.NoTimeout, // Inspect bounds itself with CheckTimeout
		})
	}
	if c.regs == nil {
		c.regs = DefaultRegistries
	}
	if c.now == nil {
		c.now = time.Now
	}
	if c.timeout <= 0 {
		c.timeout = defaultCheckTimeout
	}
	if c.logf == nil {
		c.logf = log.Printf
	}
	return c
}

// Default is the client every caller uses, read at call time (never copied).
// Init sets it; simulate mode points it at an in-process registrymock.
var Default *Client

var (
	initMu  sync.Mutex
	simMock *registrymock.Server
)

// Init sets Default. It is idempotent: a later call replaces the earlier
// client (and stops the simulated registry it started). With simulate the
// registries are an in-process registrymock serving Fixtures(), so nothing
// leaves the machine.
func Init(simulate bool) {
	initMu.Lock()
	defer initMu.Unlock()
	if simMock != nil {
		simMock.Close()
		simMock = nil
	}
	if !simulate {
		Default = New(Options{})
		return
	}
	simMock = registrymock.New(registrymock.Options{Images: registrymock.Fixtures()})
	simMock.StartTLS()
	Default = New(Options{HTTP: simMock.Client()})
}

// request is one Inspect's state: the parsed ref, its registry, the token
// and a request count for the log line.
type request struct {
	c     *Client
	ref   Ref
	reg   Registry
	token string
	n     atomic.Int32
}

// Inspect reads ref's platform manifest and config: at most 5 requests on a
// cold digest, 1 HEAD on a warm one, none on a fresh ref. platform defaults
// to linux/amd64; refresh re-resolves the tag. Every failure is an *Error.
func (c *Client) Inspect(ctx context.Context, ref, platform string, refresh bool) (*Image, error) {
	r, err := ParseRef(ref)
	if err != nil {
		return nil, err
	}
	if platform == "" {
		platform = defaultPlatform
	}
	reg, ok := c.regs[r.API]
	if !ok {
		return nil, (&Error{Kind: KindNotAllowed, Message: "registry " + r.Domain + " is not on the allowlist"}).withRef(r)
	}
	key := r.String() + "|" + platform
	if refresh {
		c.cache.forget(key)
	} else if img, nerr := c.cache.fresh(key, c.now()); img != nil || nerr != nil {
		if nerr != nil {
			return nil, nerr
		}
		return c.answer(img, ref, r), nil
	}
	img, err := c.cache.do(ctx, key, func() (*Image, error) {
		// The fetch is shared by every caller waiting on key, so it outlives
		// any one of them (a studio request cut short by its debounce) and is
		// bounded by CheckTimeout alone.
		ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), c.timeout)
		defer cancel()
		start := time.Now()
		q := &request{c: c, ref: r, reg: reg}
		img, err := q.inspect(ctx, platform)
		var e *Error
		if errors.As(err, &e) {
			e.withRef(r)
			c.cache.remember(key, e, c.now())
			return nil, e
		}
		if err != nil {
			return nil, err
		}
		c.cache.storeImage(key, img, c.now())
		if q.n.Load() > 1 { // a warm digest is one HEAD: not worth a line
			c.logf("imagecheck: %s -> %s (%d requests, %d ms)", r, img.Digest, q.n.Load(), time.Since(start).Milliseconds())
		}
		return img, nil
	})
	if err != nil {
		return nil, withRef(err, r)
	}
	return c.answer(img, ref, r), nil
}

// answer is a private copy of a cached image, labelled with what was asked.
func (c *Client) answer(img *Image, asked string, r Ref) *Image {
	out := img.clone()
	out.Ref, out.Tag = asked, r.Tag
	return out
}

// HeadDigest resolves ref to the digest its tag points at with one HEAD,
// which Docker Hub does not count against the pull limit.
func (c *Client) HeadDigest(ctx context.Context, ref string) (string, error) {
	r, err := ParseRef(ref)
	if err != nil {
		return "", err
	}
	reg, ok := c.regs[r.API]
	if !ok {
		return "", (&Error{Kind: KindNotAllowed, Message: "registry " + r.Domain + " is not on the allowlist"}).withRef(r)
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	q := &request{c: c, ref: r, reg: reg}
	if err := q.auth(ctx); err != nil {
		return "", withRef(err, r)
	}
	digest, _, err := q.head(ctx)
	if err != nil {
		return "", withRef(err, r)
	}
	return digest, nil
}

func withRef(err error, r Ref) error {
	var e *Error
	if errors.As(err, &e) {
		e.withRef(r)
	}
	return err
}

// Tag shapes the tag list keeps or drops (IMAGE-PLAN §1.2).
var (
	shaTag     = regexp.MustCompile(`^[0-9a-f]{7,12}$`)
	stampTag   = regexp.MustCompile(`^[0-9]{10}$`) // <yymmddHHMM>: redundant with <sha>
	channelTag = regexp.MustCompile(`^cu[0-9]{3}$`)
	versionTag = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)
)

const (
	defaultTagLimit = 8
	maxTagLimit     = 20
	maxShaTags      = 12
)

// Tags lists repo's tags worth offering (latest, flavor channels, versions
// and the newest-looking build tags), inspecting at most limit of them
// (default 8, max 20) through the shared caches, newest first. A tag that
// fails to inspect is left out.
func (c *Client) Tags(ctx context.Context, repo string, limit int) ([]TagInfo, error) {
	r, err := ParseRef(repo)
	if err != nil {
		return nil, err
	}
	reg, ok := c.regs[r.API]
	if !ok {
		return nil, (&Error{Kind: KindNotAllowed, Message: "registry " + r.Domain + " is not on the allowlist"}).withRef(r)
	}
	if limit <= 0 {
		limit = defaultTagLimit
	}
	limit = min(limit, maxTagLimit)

	listCtx, cancel := context.WithTimeout(ctx, c.timeout)
	q := &request{c: c, ref: r, reg: reg}
	tags, err := q.tagList(listCtx)
	cancel()
	if err != nil {
		return nil, withRef(err, r)
	}

	var head, shas []string
	for _, t := range tags {
		switch {
		case t == "latest" || channelTag.MatchString(t) || versionTag.MatchString(t):
			head = append(head, t)
		case stampTag.MatchString(t):
		case shaTag.MatchString(t):
			shas = append(shas, t)
		}
	}
	sort.SliceStable(head, func(i, j int) bool { return tagRank(head[i]) < tagRank(head[j]) })
	if len(shas) > maxShaTags {
		shas = shas[len(shas)-maxShaTags:] // registries list oldest first
	}
	slices.Reverse(shas)
	candidates := append(head, shas...)
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}

	out := []TagInfo{}
	for _, t := range candidates {
		img, err := c.Inspect(ctx, r.WithTag(t).String(), "", false)
		if err != nil {
			continue
		}
		out = append(out, TagInfo{
			Tag: t, Digest: img.Digest, Created: img.Created,
			Flavor: img.Requires.Flavor, CUDAMin: img.Requires.CUDAMin, Revision: img.Revision,
			Moving: IsMovingTag(t),
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Created.After(out[j].Created) })
	return out, nil
}

func tagRank(t string) int {
	switch {
	case t == "latest":
		return 0
	case channelTag.MatchString(t):
		return 1
	}
	return 2
}

// IsMovingTag reports whether tag is a channel later builds overwrite
// (latest, cuXXX, <version>), as opposed to an immutable <sha> tag.
func IsMovingTag(tag string) bool {
	return tag == "latest" || channelTag.MatchString(tag) || versionTag.MatchString(tag)
}
