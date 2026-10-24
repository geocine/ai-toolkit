package imagecheck

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
	"github.com/geocine/aitoolkit/internal/netguard"
)

// TestMain is the network tripwire: any request through the default
// transport fails, so a test that forgets the mock cannot reach a registry.
func TestMain(m *testing.M) {
	http.DefaultTransport = tripwire{}
	os.Exit(m.Run())
}

type tripwire struct{}

func (tripwire) RoundTrip(r *http.Request) (*http.Response, error) {
	panic("imagecheck test reached the network: " + r.URL.Host)
}

const goRef = DefaultRepo + ":" + registrymock.Revision

// clock is a fake time source tests move by hand.
type clock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *clock) now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *clock) add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// newMock starts a registrymock with Fixtures() and returns a client wired to
// it and the fake clock it runs on.
func newMock(t *testing.T, f registrymock.Faults) (*registrymock.Server, *Client, *clock) {
	t.Helper()
	srv := registrymock.New(registrymock.Options{Images: registrymock.Fixtures(), Faults: f})
	srv.StartTLS()
	t.Cleanup(srv.Close)
	clk := &clock{t: time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC)}
	c := New(Options{HTTP: srv.Client(), Now: clk.now, CheckTimeout: 5 * time.Second, Logf: t.Logf})
	return srv, c, clk
}

func count(reqs []registrymock.Request, method, pathPart string) int {
	n := 0
	for _, r := range reqs {
		if (method == "" || r.Method == method) && strings.Contains(r.Path, pathPart) {
			n++
		}
	}
	return n
}

func mustInspect(t *testing.T, c *Client, ref string) *Image {
	t.Helper()
	img, err := c.Inspect(context.Background(), ref, "", false)
	if err != nil {
		t.Fatalf("Inspect(%s): %v", ref, err)
	}
	return img
}

func TestInspectGoImage(t *testing.T) {
	srv, c, _ := newMock(t, registrymock.Faults{})
	img := mustInspect(t, c, goRef)

	if img.Ref != goRef || img.Repo != DefaultRepo || img.Tag != registrymock.Revision {
		t.Fatalf("ref fields: %q %q %q", img.Ref, img.Repo, img.Tag)
	}
	if want := srv.Digest("", registrymock.DefaultRepo, registrymock.Revision); img.Digest != want || img.PlatformDigest == want {
		t.Fatalf("digest %s (want %s), platform digest %s", img.Digest, want, img.PlatformDigest)
	}
	if !slices.Equal(img.Platforms, []string{"linux/amd64"}) || img.Platform != "linux/amd64" {
		t.Fatalf("platforms %v (attestation must be skipped)", img.Platforms)
	}
	if !img.GoServer || img.Revision != registrymock.Revision {
		t.Fatalf("go server %v revision %q", img.GoServer, img.Revision)
	}
	if img.Layers != 35 || img.LargestLayer != 2_798_692_972 {
		t.Fatalf("layers %d largest %d", img.Layers, img.LargestLayer)
	}
	if img.UnpackedSource != "label" || img.UnpackedBytes < 13_689_223_168 || img.UnpackedBytes > 13_689_223_168+100_000_000 {
		t.Fatalf("unpacked %d (%s)", img.UnpackedBytes, img.UnpackedSource)
	}
	r := img.Requires
	if r.CUDAMin != "13.0" || r.DriverMin != 580 || r.API != 1 || r.Port != 8675 || r.Flavor != "cu130" || r.Torch != "2.13.0+cu130" ||
		!slices.Equal(r.SM, []int{75, 80, 86, 90, 100, 120}) || !slices.Equal(r.FlashAttnSM, []int{80, 90, 100, 120}) || len(r.PTX) != 0 {
		t.Fatalf("requires %+v", r)
	}
	if r.Sources["sm"] != "label" || r.Sources["cuda_min"] != "env" || r.Sources["driver_min"] != "label" {
		t.Fatalf("sources %v", r.Sources)
	}

	// Only the two label prefixes and three env names survive; planted secrets never do.
	for k := range img.Labels {
		if !strings.HasPrefix(k, "org.opencontainers.image.") && !strings.HasPrefix(k, "io.github.geocine.aitk.") {
			t.Errorf("label %q kept", k)
		}
	}
	if _, ok := img.Labels["authors"]; ok {
		t.Error("authors label kept")
	}
	if len(img.Env) != 3 || img.Env["CUDA_VERSION"] != "13.0.3" {
		t.Errorf("env %v", img.Env)
	}
	b, _ := json.Marshal(img)
	for _, s := range []string{"planted", "HF_TOKEN", "AWS_SECRET", "203.0.113", "layer_digests"} {
		if strings.Contains(string(b), s) {
			t.Errorf("JSON carries %q", s)
		}
	}
	if !slices.Equal(img.ExposedPorts, []string{"8675/tcp"}) {
		t.Errorf("ports %v", img.ExposedPorts)
	}
	if checks := ImageChecks(img); len(checks) != 0 {
		t.Errorf("checks %+v", checks)
	}

	// Basic credentials are never sent, and every manifest request says what it accepts.
	for _, req := range srv.Requests() {
		if req.Basic {
			t.Errorf("Basic auth sent to %s%s", req.Host, req.Path)
		}
		if strings.Contains(req.Path, "/manifests/") && !strings.Contains(req.Accept, "application/vnd.oci.image.index.v1+json") {
			t.Errorf("manifest request without Accept: %+v", req)
		}
	}
}

func TestInspectOtherImages(t *testing.T) {
	_, c, _ := newMock(t, registrymock.Faults{})

	// Upstream on Docker Hub: deduplicated layers, CUDA floor from env, SM from the torch table.
	ostris := mustInspect(t, c, "ostris/aitoolkit:latest")
	if ostris.DownloadBytes != 6_577_255_697 || ostris.Layers != 28 {
		t.Fatalf("ostris bytes %d layers %d (want 6577255697 unique of 6577255729)", ostris.DownloadBytes, ostris.Layers)
	}
	if ostris.GoServer || ostris.Requires.API != 0 || ostris.Requires.Port != 8675 || ostris.Requires.CUDAMin != "13.0" ||
		ostris.Requires.Sources["sm"] != "table" || !slices.Equal(ostris.Requires.SM, []int{75, 80, 86, 90, 100, 120}) {
		t.Fatalf("ostris %+v go=%v", ostris.Requires, ostris.GoServer)
	}
	if ostris.UnpackedSource != "estimate" || ostris.RateLimit == nil || ostris.RateLimit.Limit != 100 || ostris.RateLimit.WindowS != 21600 {
		t.Fatalf("ostris unpacked %s rate limit %+v", ostris.UnpackedSource, ostris.RateLimit)
	}
	if got := ImageChecks(ostris); len(got) != 1 || got[0].Code != "image_not_go" || got[0].Level != LevelBlock {
		t.Fatalf("ostris checks %+v", got)
	}

	// The Next.js images.
	next := mustInspect(t, c, "ghcr.io/geocine/aitoolkit:0.11.1")
	if next.GoServer || next.Requires.CUDAMin != "12.8" || next.Requires.DriverMin != 570 || next.DownloadBytes < 10_000_000_000 {
		t.Fatalf("0.11.1 %+v go=%v bytes=%d", next.Requires, next.GoServer, next.DownloadBytes)
	}
	if latest := mustInspect(t, c, "ghcr.io/geocine/aitoolkit"); latest.GoServer || latest.Requires.CUDAMin != "13.0" {
		t.Fatalf("aitoolkit:latest %+v", latest.Requires)
	}

	// A Go image without labels, served as a bare manifest: known from history only.
	old := mustInspect(t, c, DefaultRepo+":oldsha1")
	if !old.GoServer || old.Requires.Sources["api"] != "history" || old.PlatformDigest != old.Digest || old.Revision != "" {
		t.Fatalf("oldsha1 go=%v sources=%v", old.GoServer, old.Requires.Sources)
	}
	if old.Requires.Sources["sm"] != "table" || old.Requires.Sources["flash_attn_sm"] != "table" || !slices.Equal(old.Platforms, []string{"linux/amd64"}) {
		t.Fatalf("oldsha1 %+v platforms %v", old.Requires, old.Platforms)
	}

	// A synthetic cu126 flavor.
	cu126 := mustInspect(t, c, DefaultRepo+":cu126test")
	if cu126.Requires.CUDAMin != "12.6" || cu126.Requires.Flavor != "cu126" || !cu126.Requires.Runs(70) || cu126.Requires.Runs(120) {
		t.Fatalf("cu126 %+v", cu126.Requires)
	}

	// A Docker manifest list without CUDA: amd64 picked from three platforms, no SM.
	alpine := mustInspect(t, c, "alpine:3.20")
	if alpine.GoServer || len(alpine.Requires.SM) != 0 || alpine.Requires.CUDAMin != "" || len(alpine.Platforms) != 3 {
		t.Fatalf("alpine %+v platforms %v", alpine.Requires, alpine.Platforms)
	}
	var sawInfo bool
	for _, ch := range ImageChecks(alpine) {
		if ch.Code == "image_sm_unknown" {
			sawInfo = ch.Level == LevelInfo
		}
	}
	if !sawInfo {
		t.Fatal("alpine: no image_sm_unknown info")
	}
}

func TestInspectFailures(t *testing.T) {
	srv, c, _ := newMock(t, registrymock.Faults{})
	ctx := context.Background()

	_, err := c.Inspect(ctx, DefaultRepo+":arm64only", "", false)
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindNoPlatform || !slices.Equal(e.Platforms, []string{"linux/arm64"}) {
		t.Fatalf("arm64only: %v", err)
	}
	if ch := ErrorCheck(err); ch.Code != "image_platform" || ch.Level != LevelBlock ||
		ch.Message != "This image is built for arm64 only. Rented GPU servers here run amd64." {
		t.Fatalf("arm64only check %+v", ch)
	}
	if img, err := c.Inspect(ctx, DefaultRepo+":arm64only", "linux/arm64", false); err != nil || img.Platform != "linux/arm64" {
		t.Fatalf("arm64only as arm64: %v", err)
	}
	// The arm64 read is cached, but never handed to an amd64 caller.
	if img, err := c.Inspect(ctx, DefaultRepo+":arm64only", "", false); KindOf(err) != KindNoPlatform {
		t.Fatalf("arm64only after an arm64 read: %v %+v", err, img)
	}

	_, err = c.Inspect(ctx, DefaultRepo+":0.14.0", "", false)
	if KindOf(err) != KindNotFound {
		t.Fatalf("missing tag: %v", err)
	}
	if ch := ErrorCheck(err); ch.Code != "image_not_found" || ch.Level != LevelBlock || ch.Message != "There's no tag 0.14.0 for this image." {
		t.Fatalf("missing tag check %+v", ch)
	}
	// With the tag list, the fix names the newest build and offers it.
	tags, terr := c.Tags(ctx, DefaultRepo, 0)
	if terr != nil {
		t.Fatal(terr)
	}
	ch := NotFoundCheck(err, tags)
	if ch.Code != "image_not_found" || ch.Message != "There's no tag 0.14.0 for this image." ||
		ch.Fix != "Newest build: "+registrymock.Version+" · "+registrymock.Revision+"." ||
		ch.Action == nil || ch.Action.Kind != "use_image" || ch.Action.Value != goRef {
		t.Fatalf("missing tag check with tags %+v %+v", ch, ch.Action)
	}
	if ch := NotFoundCheck(errors.New("boom"), tags); ch.Code != "image_unreachable" || ch.Action != nil {
		t.Fatalf("NotFoundCheck on another error %+v", ch)
	}

	// A repo GHCR doesn't know is denied at the token step, like a private one.
	_, err = c.Inspect(ctx, "ghcr.io/someone/else:1", "", false)
	if ch := ErrorCheck(err); KindOf(err) != KindDenied || ch.Code != "image_denied" ||
		ch.Message != "The image ghcr.io/someone/else wasn't found, or it's private." {
		t.Fatalf("unknown repo: %v %+v", err, ch)
	}

	// A registry outside the allowlist: a warn, and no request at all.
	srv.Reset()
	_, err = c.Inspect(ctx, "quay.io/org/img:1", "", false)
	if ch := ErrorCheck(err); KindOf(err) != KindNotAllowed || ch.Code != "image_unchecked" || ch.Level != LevelWarn ||
		ch.Message != "Images on quay.io can't be checked." {
		t.Fatalf("quay: %v %+v", err, ch)
	}
	if n := len(srv.Requests()); n != 0 {
		t.Fatalf("quay made %d requests", n)
	}

	_, err = c.Inspect(ctx, "Not An Image", "", false)
	if ch := ErrorCheck(err); ch.Code != "image_ref" || ch.Level != LevelBlock {
		t.Fatalf("bad ref %+v", ch)
	}
}

// One platform's read never answers for another: the image cache is keyed
// by digest and platform.
func TestCacheKeepsPlatformsApart(t *testing.T) {
	ctx := context.Background()
	_, c, _ := newMock(t, registrymock.Faults{})
	if _, err := c.Inspect(ctx, DefaultRepo+":arm64only", "linux/arm64", false); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"", "linux/amd64"} {
		if img, err := c.Inspect(ctx, DefaultRepo+":arm64only", p, false); KindOf(err) != KindNoPlatform {
			t.Fatalf("arm64only as %q after arm64: %v %+v", p, err, img)
		}
	}

	amd := mustInspect(t, c, "alpine:3.20")
	arm, err := c.Inspect(ctx, "alpine:3.20", "linux/arm64/v8", false)
	if err != nil {
		t.Fatal(err)
	}
	if arm.Platform != "linux/arm64/v8" || arm.PlatformDigest == amd.PlatformDigest || arm.Digest != amd.Digest {
		t.Fatalf("alpine arm64 after amd64: platform %s, platform digest %s (amd64 %s)", arm.Platform, arm.PlatformDigest, amd.PlatformDigest)
	}
	if again := mustInspect(t, c, "alpine:3.20"); again.Platform != defaultPlatform || again.PlatformDigest != amd.PlatformDigest {
		t.Fatalf("alpine amd64 after arm64: %s %s", again.Platform, again.PlatformDigest)
	}
}

// A caller that gives up leaves early without failing the others sharing
// its fetch.
func TestSharedFetchOutlivesFirstCaller(t *testing.T) {
	srv, c, _ := newMock(t, registrymock.Faults{SlowMs: 100})
	first, cancel := context.WithCancel(context.Background())
	firstErr := make(chan error, 1)
	start := time.Now()
	go func() {
		_, err := c.Inspect(first, goRef, "", false)
		firstErr <- err
	}()
	time.AfterFunc(150*time.Millisecond, cancel)
	time.Sleep(30 * time.Millisecond) // the first caller's fetch is in flight
	img, err := c.Inspect(context.Background(), goRef, "", false)
	if err != nil || !img.GoServer {
		t.Fatalf("second caller failed with the first one: %v", err)
	}
	err = <-firstErr
	if e := (*Error)(nil); !errors.As(err, &e) || e.Kind != KindUnreachable || e.Repo != DefaultRepo {
		t.Fatalf("cancelled caller: %v", err)
	}
	if d := time.Since(start); d < 400*time.Millisecond {
		t.Fatalf("the fetch finished in %v: the mock was not slow enough to prove anything", d)
	}
	if n := count(srv.Requests(), "HEAD", "/manifests/"); n != 1 {
		t.Fatalf("the callers made %d HEADs, not one shared fetch", n)
	}
}

// IMAGE-PLAN §8 I2: the realm a 401 challenge advertises is never dialled.
// The mock challenges every /v2/ request (so the client does see the
// hostile realm) and routes the realm's host to itself, so a dial would be
// recorded.
func TestHostileRealmNeverDialled(t *testing.T) {
	srv, c, _ := newMock(t, registrymock.Faults{Realm: "https://evil.example/token", Challenge: true})
	_, err := c.Inspect(context.Background(), goRef, "", false)
	if KindOf(err) != KindDenied {
		t.Fatalf("err %v", err)
	}
	reqs := srv.Requests()
	if count(reqs, "HEAD", "/manifests/") != 1 {
		t.Fatalf("the client never met the challenge: %+v", reqs)
	}
	for _, r := range reqs {
		if r.Host == "evil.example" || (strings.HasSuffix(r.Path, "/token") && r.Host != "ghcr.io") {
			t.Fatalf("advertised realm was dialled: %+v", r)
		}
	}
	if n := count(reqs, "GET", "/token"); n != 1 {
		t.Fatalf("%d token fetches", n)
	}
}

// MASTER §3.4: the default image, unpublished or private, gets the "not
// published yet" copy.
func TestDefaultRepoDenied(t *testing.T) {
	_, c, _ := newMock(t, registrymock.Faults{Private: map[string]bool{registrymock.DefaultRepo: true}})
	_, err := c.Inspect(context.Background(), DefaultRepo+":latest", "", false)
	if KindOf(err) != KindDenied {
		t.Fatalf("err %v", err)
	}
	ch := ErrorCheck(err)
	want := Check{Level: LevelBlock, Code: "image_denied",
		Message: "The server image ghcr.io/geocine/aitoolkit-go isn't published yet, or it's still private.",
		Fix:     "Its owner publishes it once with ui-go/scripts/build_and_push.sh --push and makes the package public. Until then GPU servers can't start it."}
	if ch.Level != want.Level || ch.Code != want.Code || ch.Message != want.Message || ch.Fix != want.Fix || ch.Action != nil {
		t.Fatalf("check\n got %+v\nwant %+v", ch, want)
	}
}

func TestTokenFlow(t *testing.T) {
	srv, c, clk := newMock(t, registrymock.Faults{Realm: "https://evil.example/token"})
	ctx := context.Background()
	mustInspect(t, c, goRef)
	mustInspect(t, c, DefaultRepo+":oldsha1")
	if n := count(srv.Requests(), "GET", "/token"); n != 1 {
		t.Fatalf("GHCR token fetched %d times within its 60 s", n)
	}
	for _, r := range srv.Requests() {
		if r.Host == "evil.example" || (strings.HasSuffix(r.Path, "/token") && r.Host != "ghcr.io") {
			t.Fatalf("advertised realm was dialled: %+v", r)
		}
	}
	clk.add(61 * time.Second)
	if _, err := c.Inspect(ctx, DefaultRepo+":cu126test", "", false); err != nil {
		t.Fatal(err)
	}
	if n := count(srv.Requests(), "GET", "/token"); n != 2 {
		t.Fatalf("GHCR token not refreshed after 60 s (%d fetches)", n)
	}

	// Docker Hub's expires_in 300 caches 290 s.
	srv.Reset()
	if _, err := c.HeadDigest(ctx, "ostris/aitoolkit:latest"); err != nil {
		t.Fatal(err)
	}
	clk.add(289 * time.Second)
	if _, err := c.HeadDigest(ctx, "ostris/aitoolkit:latest"); err != nil {
		t.Fatal(err)
	}
	if n := count(srv.Requests(), "GET", "/token"); n != 1 {
		t.Fatalf("Hub token fetched %d times within 290 s", n)
	}
	clk.add(2 * time.Second)
	if _, err := c.HeadDigest(ctx, "ostris/aitoolkit:latest"); err != nil {
		t.Fatal(err)
	}
	if n := count(srv.Requests(), "GET", "/token"); n != 2 {
		t.Fatalf("Hub token not refreshed after 290 s (%d fetches)", n)
	}
	if n := count(srv.Requests(), "GET", "/manifests/"); n != 0 {
		t.Fatalf("HeadDigest made %d GETs", n)
	}
}

func TestBadResponses(t *testing.T) {
	ctx := context.Background()
	_, c, _ := newMock(t, registrymock.Faults{BadDigest: true})
	if _, err := c.Inspect(ctx, goRef, "", false); KindOf(err) != KindBadResponse {
		t.Fatalf("bad digest: %v", err)
	}
	_, c, _ = newMock(t, registrymock.Faults{OversizeConfig: true})
	if _, err := c.Inspect(ctx, goRef, "", false); KindOf(err) != KindBadResponse || !strings.Contains(err.Error(), "declares") {
		t.Fatalf("oversize config: %v", err)
	}
	_, c, _ = newMock(t, registrymock.Faults{Status: map[string]int{"/v2/geocine/aitoolkit-go/manifests/latest": 502}})
	if _, err := c.Inspect(ctx, DefaultRepo+":latest", "", false); KindOf(err) != KindUnreachable {
		t.Fatalf("502: %v", err)
	} else if ch := ErrorCheck(err); ch.Level != LevelWarn || ch.Code != "image_unreachable" {
		t.Fatalf("502 check %+v", ch)
	}

	// An oversize body is cut at the cap and errors.
	if _, err := readCapped(strings.NewReader(strings.Repeat("x", 101)), 100, "manifest"); KindOf(err) != KindBadResponse {
		t.Fatalf("oversize body: %v", err)
	}
	if b, err := readCapped(strings.NewReader(strings.Repeat("x", 100)), 100, "manifest"); err != nil || len(b) != 100 {
		t.Fatalf("body at the cap: %v", err)
	}
}

func TestBlobRedirects(t *testing.T) {
	ctx := context.Background()
	for _, mode := range []string{"cdn", "cloudfront", "cloudflare"} {
		srv, c, _ := newMock(t, registrymock.Faults{RedirectBlobs: mode})
		for _, ref := range []string{goRef, "ostris/aitoolkit:latest"} {
			if _, err := c.Inspect(ctx, ref, "", false); err != nil {
				t.Fatalf("%s %s: %v", mode, ref, err)
			}
		}
		cdn := 0
		for _, r := range srv.Requests() {
			if strings.HasPrefix(r.Path, "/blob/") {
				cdn++
				if r.HasAuth {
					t.Fatalf("%s: Authorization sent to the CDN %s", mode, r.Host)
				}
			}
		}
		if cdn != 2 {
			t.Fatalf("%s: %d CDN requests, want 2", mode, cdn)
		}
	}

	_, c, _ := newMock(t, registrymock.Faults{RedirectBlobs: "http"})
	if _, err := c.Inspect(ctx, goRef, "", false); KindOf(err) != KindBadResponse || !strings.Contains(err.Error(), "not https") {
		t.Fatalf("http redirect: %v", err)
	}
	srv, c, _ := newMock(t, registrymock.Faults{RedirectBlobs: "loop"})
	if _, err := c.Inspect(ctx, goRef, "", false); KindOf(err) != KindBadResponse || !strings.Contains(err.Error(), "more than 3") {
		t.Fatalf("redirect loop: %v", err)
	}
	if n := count(srv.Requests(), "GET", "/loop/"); n != 3 {
		t.Fatalf("followed %d loop hops, want 3", n)
	}
	_, c, _ = newMock(t, registrymock.Faults{RedirectBlobs: "private"})
	_, err := c.Inspect(ctx, goRef, "", false)
	if KindOf(err) != KindUnreachable || !errors.Is(err, netguard.ErrBlocked) {
		t.Fatalf("redirect to a private address: %v", err)
	}
	if strings.Contains(err.Error(), "sig=") {
		t.Fatalf("error leaks the signed query: %v", err)
	}
}

func TestRateLimit(t *testing.T) {
	ctx := context.Background()
	srv, c, _ := newMock(t, registrymock.Faults{Remaining: 92})
	for range 3 {
		if _, err := c.HeadDigest(ctx, "ostris/aitoolkit:latest"); err != nil {
			t.Fatal(err)
		}
	}
	img := mustInspect(t, c, "ostris/aitoolkit:latest")
	// The HEADs did not count; Inspect's HEAD saw 92 before its two GETs.
	if img.RateLimit == nil || img.RateLimit.Remaining != 92 || img.RateLimit.Limit != 100 || img.RateLimit.WindowS != 21600 {
		t.Fatalf("rate limit %+v", img.RateLimit)
	}
	if n := count(srv.Requests(), "GET", "/manifests/"); n != 2 {
		t.Fatalf("%d manifest GETs", n)
	}

	// Nearly used: stop before a GET counts.
	srv, c, _ = newMock(t, registrymock.Faults{Remaining: 5})
	_, err := c.Inspect(ctx, "ostris/aitoolkit:latest", "", false)
	if KindOf(err) != KindRateLimited {
		t.Fatalf("remaining 5: %v", err)
	}
	if n := count(srv.Requests(), "GET", "/manifests/"); n != 0 {
		t.Fatalf("remaining 5 still made %d GETs", n)
	}
	if ch := ErrorCheck(err); ch.Level != LevelWarn || ch.Code != "image_rate_limited" {
		t.Fatalf("rate limited check %+v", ch)
	}

	// A 429 on GET.
	_, c, _ = newMock(t, registrymock.Faults{RateLimitAfter: 1})
	if _, err := c.Inspect(ctx, "ostris/aitoolkit:latest", "", false); KindOf(err) != KindRateLimited {
		t.Fatalf("429: %v", err)
	}

	h := http.Header{}
	h.Set("ratelimit-limit", "100;w=21600")
	h.Set("ratelimit-remaining", "76;w=21600")
	h.Set("docker-ratelimit-source", "203.0.113.7")
	if rl := parseRateLimit(h); *rl != (RateLimit{100, 76, 21600, "203.0.113.7"}) {
		t.Fatalf("parse %+v", rl)
	}
	if parseRateLimit(http.Header{}) != nil {
		t.Fatal("GHCR (no headers) gave a rate limit")
	}
}

func TestCache(t *testing.T) {
	ctx := context.Background()
	srv, c, clk := newMock(t, registrymock.Faults{})

	// 10 concurrent Inspects share one fetch.
	var wg sync.WaitGroup
	errs := make(chan error, 10)
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := c.Inspect(ctx, goRef, "", false)
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if n := count(srv.Requests(), "HEAD", "/manifests/"); n != 1 {
		t.Fatalf("10 concurrent inspects made %d HEADs", n)
	}

	// A fresh ref needs nothing; refresh re-resolves with one HEAD; another
	// tag of the same digest is one HEAD too (the warm-digest path).
	srv.Reset()
	mustInspect(t, c, goRef)
	if n := len(srv.Requests()); n != 0 {
		t.Fatalf("fresh ref made %d requests", n)
	}
	if _, err := c.Inspect(ctx, goRef, "", true); err != nil {
		t.Fatal(err)
	}
	latest := mustInspect(t, c, DefaultRepo+":latest")
	reqs := srv.Requests()
	if len(reqs) != 2 || reqs[0].Method != "HEAD" || reqs[1].Method != "HEAD" {
		t.Fatalf("refresh + warm digest: %+v", reqs)
	}
	if latest.Tag != "latest" || latest.Ref != DefaultRepo+":latest" || latest.Revision != registrymock.Revision {
		t.Fatalf("warm copy kept the old ref: %+v", latest)
	}

	// The ref cache lasts 10 minutes.
	clk.add(11 * time.Minute)
	srv.Reset()
	mustInspect(t, c, goRef)
	if n := count(srv.Requests(), "HEAD", ""); n != 1 {
		t.Fatalf("stale ref made %d HEADs", n)
	}

	// A caller cannot change the cached copy.
	a := mustInspect(t, c, goRef)
	a.Labels["io.github.geocine.aitk.sm"] = "junk"
	a.Requires.SM[0] = 1
	if b := mustInspect(t, c, goRef); b.Labels["io.github.geocine.aitk.sm"] == "junk" || b.Requires.SM[0] != 75 {
		t.Fatal("cached image was mutated through a returned copy")
	}

	// not_found is remembered for 60 s.
	srv.Reset()
	for range 3 {
		if _, err := c.Inspect(ctx, DefaultRepo+":typo", "", false); KindOf(err) != KindNotFound {
			t.Fatal(err)
		}
	}
	if n := count(srv.Requests(), "HEAD", ""); n != 1 {
		t.Fatalf("negative cache: %d HEADs", n)
	}
	clk.add(61 * time.Second)
	if _, err := c.Inspect(ctx, DefaultRepo+":typo", "", false); KindOf(err) != KindNotFound {
		t.Fatal(err)
	}
	if n := count(srv.Requests(), "HEAD", ""); n != 2 {
		t.Fatalf("negative cache outlived 60 s: %d HEADs", n)
	}
}

func TestLRUEvictsOldest(t *testing.T) {
	l := newLRU(lruSize)
	for i := range lruSize + 1 {
		l.put(fmt.Sprintf("sha256:%d", i), &Image{})
	}
	if _, ok := l.get("sha256:0"); ok {
		t.Fatal("oldest entry survived 257 inserts")
	}
	if _, ok := l.get("sha256:1"); !ok {
		t.Fatal("second entry evicted")
	}
	if l.order.Len() != lruSize {
		t.Fatalf("len %d", l.order.Len())
	}
}

func TestInspectTimeout(t *testing.T) {
	srv := registrymock.New(registrymock.Options{Images: registrymock.Fixtures(), Faults: registrymock.Faults{SlowMs: 2000}})
	srv.StartTLS()
	t.Cleanup(srv.Close)
	c := New(Options{HTTP: srv.Client(), CheckTimeout: 200 * time.Millisecond, Logf: t.Logf})
	start := time.Now()
	_, err := c.Inspect(context.Background(), goRef, "", false)
	if KindOf(err) != KindUnreachable {
		t.Fatalf("err %v", err)
	}
	if d := time.Since(start); d > 300*time.Millisecond {
		t.Fatalf("Inspect took %v past a 200 ms deadline", d)
	}
}

func TestTags(t *testing.T) {
	_, c, _ := newMock(t, registrymock.Faults{})
	tags, err := c.Tags(context.Background(), DefaultRepo, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]TagInfo{}
	for _, ti := range tags {
		got[ti.Tag] = ti
	}
	for _, want := range []string{"latest", "cu130", registrymock.Version, registrymock.Revision} {
		ti, ok := got[want]
		if !ok {
			t.Fatalf("tag %s missing from %v", want, tags)
		}
		if ti.Moving != (want != registrymock.Revision) || ti.Flavor != "cu130" || ti.CUDAMin != "13.0" || ti.Revision != registrymock.Revision {
			t.Errorf("tag %+v", ti)
		}
	}
	for _, dropped := range []string{"oldsha1", "cu126test", "arm64only"} {
		if _, ok := got[dropped]; ok {
			t.Errorf("tag %s should be dropped", dropped)
		}
	}
	if tags, _ := c.Tags(context.Background(), DefaultRepo, 2); len(tags) != 2 {
		t.Fatalf("limit 2 gave %d", len(tags))
	}
}

func TestInitSimulate(t *testing.T) {
	prev := Default
	t.Cleanup(func() { Init(false); Default = prev })
	Init(true)
	first := Default
	Init(true) // idempotent: replaces the client and its mock
	if Default == first {
		t.Fatal("second Init kept the old client")
	}
	img, err := Default.Inspect(context.Background(), DefaultRepo+":latest", "", false)
	if err != nil || !img.GoServer || img.Revision != registrymock.Revision {
		t.Fatalf("simulated registry: %v", err)
	}
}
