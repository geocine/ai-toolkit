// Package v1images serves /v1/images (check an image before a GPU server
// runs it, list its tags, pick the image for a host) and /v1/hosts (the GPU
// table with the default image's verdict, VRAM and disk hints for a model).
// The work is done by internal/imagecheck and internal/hostfit; offers are
// looked up through provision.Default. See docs/headless.md, "Images and
// hosts".
//
// A check never fails with a 5xx because a registry is down: it answers 200
// with the trouble as a warn.
package v1images

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// ServerBuild is the build this server runs (a short sha, or "dev") and the
// ai-toolkit version it carries, for the "same build" image of
// images.recommend. handlers/v1 points it at what /v1/health reports.
var ServerBuild = func() (revision, version string) { return "", "" }

// Register adds the routes. The image check and the model sizer are set up
// here unless something (simulate mode, a test) already set them.
func Register(mux *http.ServeMux) {
	if imagecheck.Default == nil {
		imagecheck.Init(false)
	}
	if hostfit.DefaultSizer == nil {
		hostfit.InitSizer(false)
	}

	httpx.Handle(mux, "POST /v1/images/check", checkImage)
	httpx.Handle(mux, "GET /v1/images/tags", imageTags)
	httpx.Handle(mux, "POST /v1/images/recommend", recommendImage)
	httpx.Handle(mux, "GET /v1/hosts/gpus", hostGPUs)
	httpx.Handle(mux, "POST /v1/hosts/fit", hostFit)
}

// ---- shapes ----

// OfferRef names one offer of providers.offers.
type OfferRef struct {
	Provider string `json:"provider"`
	OfferID  string `json:"offer_id"`
}

// ImageCheck is the answer of POST /v1/images/check (the /v1 ImageCheck).
type ImageCheck struct {
	OK        bool               `json:"ok"`    // false when any check blocks
	Level     string             `json:"level"` // worst level: ok | info | warn | block
	Image     *imagecheck.Image  `json:"image"` // null when the image could not be read
	LaunchRef *string            `json:"launch_ref"`
	Pinned    *string            `json:"pinned"`
	Host      *hostfit.Host      `json:"host"`
	Need      *hostfit.Need      `json:"need"`
	Pull      *hostfit.Pull      `json:"pull"`
	VRAM      *hostfit.VRAMFit   `json:"vram"`
	Checks    []imagecheck.Check `json:"checks"`
}

// GPUInfo is one row of GET /v1/hosts/gpus: the table row plus the default
// image's verdict on it.
type GPUInfo struct {
	hostfit.GPU
	Verdict string `json:"verdict"` // ok | info | warn | block
	Note    string `json:"note"`
}

// HostFit is the answer of POST /v1/hosts/fit.
type HostFit struct {
	Model hostfit.ModelRef `json:"model"`
	Tiers []hostfit.Tier   `json:"tiers"`
	Need  hostfit.Need     `json:"need"`
}

// ---- helpers ----

// respond writes err: an adapter error through provision.V1 (with
// Retry-After on 429), else a v1api.Problem or 500.
func respond(w http.ResponseWriter, r *http.Request, err error) {
	if code, msg, after, ok := provision.V1(err); ok {
		v1api.FailAfter(w, code, msg, after)
		return
	}
	v1api.Respond(w, r, err)
}

func invalid(msg string) error { return v1api.Errorf(v1api.CodeInvalidRequest, msg) }

// decodeOptional reads an optional JSON body: an empty body leaves v as is.
func decodeOptional(w http.ResponseWriter, r *http.Request, v any) bool {
	body, err := io.ReadAll(io.LimitReader(r.Body, v1api.MaxJSONBody))
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "could not read the request body")
		return false
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return true
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	return v1api.Decode(w, r, v)
}

// checker is the image check client, read at call time. nil only when
// nothing set it up; every read then fails as unreachable.
func checker() *imagecheck.Client { return imagecheck.Default }

func inspect(ctx context.Context, ref, platform string, refresh bool) (*imagecheck.Image, error) {
	ic := checker()
	if ic == nil {
		return nil, imagecheck.Errorf(imagecheck.KindUnreachable, "no image check is configured on this server")
	}
	return ic.Inspect(ctx, ref, platform, refresh)
}

// errorCheck is the check for an image that could not be read; a missing
// tag names the newest build when the repo's tags can be listed.
func errorCheck(ctx context.Context, err error) imagecheck.Check {
	var e *imagecheck.Error
	if ic := checker(); ic != nil && errors.As(err, &e) && e.Kind == imagecheck.KindNotFound && e.Repo != "" {
		if tags, terr := ic.Tags(ctx, e.Repo, 0); terr == nil {
			return imagecheck.NotFoundCheck(err, tags)
		}
	}
	return imagecheck.ErrorCheck(err)
}

// model is m, or nil when it names no arch.
func model(m *hostfit.ModelRef) *hostfit.ModelRef {
	if m == nil || strings.TrimSpace(m.Arch) == "" {
		return nil
	}
	return m
}

// diskGB validates a disk size (0 = not chosen) with the launch's bounds
// (provision.Service), so a check never passes a size a launch refuses.
func diskGB(gb int) error {
	if gb != 0 && (gb < 20 || gb > 2000) {
		return invalid("disk_gb must be from 20 to 2000")
	}
	return nil
}

// completeHost fills what a caller-given host leaves out from the GPU
// table: compute capability, class, vendor, CPU architecture and the
// nominal VRAM, so {gpu_name: "RTX 4090"} is enough to judge.
func completeHost(h hostfit.Host) hostfit.Host {
	row, ok := hostfit.Lookup(h.GPUName)
	if !ok {
		return h
	}
	if h.SM == 0 && row.SM > 0 {
		h.SM, h.SMSource = row.SM, "table"
	}
	if h.GPUClass == "" {
		h.GPUClass = row.Class
	}
	if h.GPUVendor == "" {
		h.GPUVendor = row.Vendor
	}
	if h.CPUArch == "" {
		_, h.CPUArch, _ = strings.Cut(row.Platform, "/")
	}
	if h.VRAMGB == 0 {
		h.VRAMGB = row.VRAMGB
	}
	return h
}

// target is the host a check or a recommendation judges, and the provider
// that would run the launch ("" when unknown).
type target struct {
	host *hostfit.Host
	kind provision.Kind
}

// resolveHost reads the host of an offer (re-quoted through the 30 s quote
// cache, which also judges image and model) or takes the caller's host
// facts. An offer wins over host. Neither leaves the host nil: only the
// image-side checks run.
func resolveHost(ctx context.Context, o *OfferRef, h *hostfit.Host, image string, m *hostfit.ModelRef, disk int) (target, error) {
	if o != nil {
		k, ok := provision.ParseKind(strings.TrimSpace(o.Provider))
		if ok {
			_, ok = provision.Lookup(k)
		}
		if !ok {
			return target{}, invalid("offer.provider is not a provider this server can rent from")
		}
		id := strings.TrimSpace(o.OfferID)
		if id == "" {
			return target{}, invalid("offer.offer_id is required")
		}
		mgr := provision.Default
		if mgr == nil {
			return target{}, invalid("GPU providers aren't available on this server")
		}
		off, err := mgr.Quote(ctx, k, id, provision.OfferQuery{Image: image, Model: m, DiskGB: disk})
		if err != nil {
			return target{}, err
		}
		t := target{kind: k}
		if off.Host != nil {
			host := *off.Host
			t.host = &host
		}
		return t, nil
	}
	if h == nil || h.IsZero() {
		return target{}, nil
	}
	host := completeHost(*h)
	k, _ := provision.ParseKind(host.Provider)
	return target{host: &host, kind: k}, nil
}

// digestRef: the provider accepts repo@sha256 references.
func digestRef(k provision.Kind) bool {
	c, ok := provision.Lookup(k)
	return ok && c.Info().Caps.ImageDigestRef == "yes"
}

// ---- POST /v1/images/check ----

type checkRequest struct {
	Image    string            `json:"image"`
	Platform string            `json:"platform"`
	Host     *hostfit.Host     `json:"host"`
	Offer    *OfferRef         `json:"offer"`
	Model    *hostfit.ModelRef `json:"model"`
	DiskGB   int               `json:"disk_gb"`
	Refresh  bool              `json:"refresh"`
}

func checkImage(w http.ResponseWriter, r *http.Request) {
	var in checkRequest
	if !v1api.Decode(w, r, &in) {
		return
	}
	ref := strings.TrimSpace(in.Image)
	if ref == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "image is required")
		return
	}
	if err := diskGB(in.DiskGB); err != nil {
		respond(w, r, err)
		return
	}
	m := model(in.Model)
	t, err := resolveHost(r.Context(), in.Offer, in.Host, ref, m, in.DiskGB)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, check(r.Context(), checkInput{ref: ref, platform: strings.TrimSpace(in.Platform), refresh: in.Refresh,
		target: t, model: m, diskGB: in.DiskGB}))
}

type checkInput struct {
	ref, platform string
	refresh       bool
	target        // the host (nil: image-side checks only) and the provider of the launch
	model         *hostfit.ModelRef
	diskGB        int
}

// check judges image ref (on platform, default linux/amd64) and, with a
// host, whether that host runs it with diskGB of disk, the same way a
// launch re-checks it (IMAGE-PLAN §4.5): the image-side checks or why the
// image could not be read, then the host-side checks. When the default
// image cannot be read its built-in requirements still judge the host
// (MASTER §3.10.3). launch_ref is what a launch on kind would send; it is
// null when anything blocks or the image was not read.
func check(ctx context.Context, in checkInput) ImageCheck {
	ref, host, kind, m, disk := in.ref, in.host, in.kind, in.model, in.diskGB
	out := ImageCheck{Checks: []imagecheck.Check{}}
	img, err := inspect(ctx, ref, in.platform, in.refresh)
	judged := img // what judges the host: the image, or the default image's fallback
	if err != nil {
		out.Checks = append(out.Checks, errorCheck(ctx, err))
		if in.platform == "" || in.platform == "linux/amd64" {
			judged, _ = hostfit.FallbackImage(ref)
		}
	} else {
		out.Image = img
		out.Checks = append(out.Checks, imagecheck.ImageChecks(img)...)
	}

	need := hostfit.NeedFor(judged, m, hostfit.DefaultSizer)
	if judged != nil {
		out.Need = &need
	}
	if host != nil {
		h := *host
		out.Host = &h
		out.Checks = append(out.Checks, hostfit.Fit(judged, h, hostfit.FitOptions{Need: &need, DiskGB: disk, Model: m, Launch: true})...)
		if judged != nil {
			var modelGB float64
			if m != nil {
				modelGB = need.ModelGB
			}
			out.Pull = hostfit.PullFor(judged, h, modelGB)
		}
		if m != nil && h.VRAMGB > 0 {
			v := hostfit.VRAMFor(*m, h.VRAMGB)
			out.VRAM = &v
		}
	}

	if img != nil {
		if rf, perr := imagecheck.ParseRef(ref); perr == nil && img.Digest != "" {
			pinned := rf.Pinned(img.Digest)
			out.Pinned = &pinned
		}
		if imagecheck.Worst(out.Checks) != imagecheck.LevelBlock {
			launch := ref
			// QuickPod runs the template's image as it is: createpod has no image field
			if kind != provision.QuickPod {
				var pin *imagecheck.Check
				launch, pin = hostfit.LaunchRef(ctx, checker(), img, ref, digestRef(kind))
				if pin != nil {
					out.Checks = append(out.Checks, *pin)
				}
			}
			out.LaunchRef = &launch
		}
	}
	out.Level = imagecheck.Worst(out.Checks)
	out.OK = out.Level != imagecheck.LevelBlock
	return out
}

// ---- GET /v1/images/tags ----

func imageTags(w http.ResponseWriter, r *http.Request) {
	repo := strings.TrimSpace(r.URL.Query().Get("image"))
	if repo == "" {
		repo = imagecheck.DefaultRepo
	}
	limit := 0
	if v := strings.TrimSpace(r.URL.Query().Get("limit")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > 20 {
			v1api.Fail(w, v1api.CodeInvalidRequest, "limit must be a whole number from 1 to 20")
			return
		}
		limit = n
	}
	ic := checker()
	if ic == nil {
		v1api.Fail(w, v1api.CodeRemoteFailed, "Couldn't check the image right now.")
		return
	}
	items, err := ic.Tags(r.Context(), repo, limit)
	if err != nil {
		registryFail(w, err)
		return
	}
	v1api.Items(w, items)
}

// registryFail answers a registry failure outside a check: a bad or
// unchecked reference is 400, a missing or private repo 404, anything else
// (unreachable, rate limited) 502. The message is the check's copy.
func registryFail(w http.ResponseWriter, err error) {
	c := imagecheck.ErrorCheck(err)
	switch imagecheck.KindOf(err) {
	case imagecheck.KindBadRef, imagecheck.KindNotAllowed:
		v1api.Fail(w, v1api.CodeInvalidRequest, c.Message)
	case imagecheck.KindDenied, imagecheck.KindNotFound:
		v1api.Fail(w, v1api.CodeNotFound, c.Message)
	default:
		v1api.Fail(w, v1api.CodeRemoteFailed, c.Message)
	}
}

// ---- POST /v1/images/recommend ----

type recommendRequest struct {
	Offer      *OfferRef         `json:"offer"`
	Host       *hostfit.Host     `json:"host"`
	Model      *hostfit.ModelRef `json:"model"`
	Prefer     string            `json:"prefer"`
	Image      string            `json:"image"`
	TemplateID string            `json:"template_id"`
	DiskGB     int               `json:"disk_gb"`
}

func recommendImage(w http.ResponseWriter, r *http.Request) {
	var in recommendRequest
	if !decodeOptional(w, r, &in) {
		return
	}
	prefer := strings.TrimSpace(in.Prefer)
	if prefer != "" && prefer != "same" && prefer != "latest" {
		v1api.Fail(w, v1api.CodeInvalidRequest, `prefer must be "same" or "latest"`)
		return
	}
	if err := diskGB(in.DiskGB); err != nil {
		respond(w, r, err)
		return
	}
	ic := checker()
	if ic == nil {
		v1api.Fail(w, v1api.CodeRemoteFailed, "Couldn't check the image right now.")
		return
	}
	ctx := r.Context()
	m := model(in.Model)
	custom := strings.TrimSpace(in.Image)
	t, err := resolveHost(ctx, in.Offer, in.Host, custom, m, in.DiskGB)
	if err != nil {
		respond(w, r, err)
		return
	}
	var tmplImage string
	if id := strings.TrimSpace(in.TemplateID); id != "" {
		if tmplImage, err = templateImage(ctx, id); err != nil {
			respond(w, r, err)
			return
		}
	}
	rev, ver := ServerBuild()
	rin := hostfit.ImageRecommendInput{Model: m, Prefer: prefer, ServerRevision: rev, ServerVersion: ver,
		Custom: custom, TemplateImage: tmplImage, DiskGB: in.DiskGB, DigestRef: digestRef(t.kind)}
	if t.host != nil {
		rin.Host = *t.host
	}
	rec, err := hostfit.RecommendImage(ctx, ic, hostfit.DefaultSizer, rin)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, rec)
}

// templateImage is the image a QuickPod template runs.
func templateImage(ctx context.Context, id string) (string, error) {
	if _, ok := provision.Lookup(provision.QuickPod); !ok || provision.Default == nil {
		return "", invalid("template_id needs QuickPod, which isn't enabled on this server")
	}
	items, err := provision.Default.Templates(ctx, provision.QuickPod)
	if err != nil {
		return "", err
	}
	i := slices.IndexFunc(items, func(t provision.Template) bool { return t.ID == id })
	if i < 0 {
		return "", v1api.Errorf(v1api.CodeNotFound, "unknown template")
	}
	if strings.TrimSpace(items[i].Image) == "" {
		return "", invalid("that template names no image")
	}
	return items[i].Image, nil
}

// ---- GET /v1/hosts/gpus ----

// The default image's verdict on every GPU is resolved once and kept: the
// image as read for gpusTTL, its built-in fallback (not published yet,
// unreachable) for gpusFallbackTTL so a fix shows soon. The image is read
// with a context of its own, bounded by gpusResolveTimeout, so a caller who
// leaves does not turn a working registry into a cached fallback.
const (
	gpusTTL            = 10 * time.Minute
	gpusFallbackTTL    = time.Minute
	gpusResolveTimeout = 20 * time.Second
)

var gpusCache struct {
	mu      sync.Mutex
	items   []GPUInfo
	expires time.Time
	client  *imagecheck.Client // the client they were judged with (a swap re-judges)
}

func hostGPUs(w http.ResponseWriter, r *http.Request) {
	v1api.Items(w, GPUs(r.Context()))
}

// GPUs is the GPU table with the default image's verdict on each GPU.
func GPUs(ctx context.Context) []GPUInfo {
	gpusCache.mu.Lock()
	defer gpusCache.mu.Unlock()
	now := time.Now()
	if gpusCache.items != nil && now.Before(gpusCache.expires) && gpusCache.client == checker() {
		return slices.Clone(gpusCache.items)
	}
	ttl, keep := gpusTTL, true
	rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), gpusResolveTimeout)
	img, err := inspect(rctx, provision.DefaultImage, "", false)
	cancel()
	if err != nil {
		img, _ = hostfit.FallbackImage(provision.DefaultImage)
		// a caller who left is no sign the registry is down: answer, keep nothing
		ttl, keep = gpusFallbackTTL, ctx.Err() == nil
	}
	rows := hostfit.GPUs()
	items := make([]GPUInfo, len(rows))
	for i, row := range rows {
		items[i] = GPUInfo{GPU: row}
		items[i].Verdict, items[i].Note = gpuVerdict(img, row)
	}
	if keep {
		gpusCache.items, gpusCache.expires, gpusCache.client = items, now.Add(ttl), checker()
	}
	return slices.Clone(items)
}

// gpuVerdict is what img says about a GPU on its own: CPU architecture,
// vendor and compute capability. The driver is a fact of each host, not of
// the GPU, so the CUDA checks are left out. A GPU the image runs through a
// lower build of the same architecture says so; sm_75, which offers hide by
// default, is a warn.
func gpuVerdict(img *imagecheck.Image, row hostfit.GPU) (verdict, note string) {
	if img == nil {
		return "ok", ""
	}
	_, arch, _ := strings.Cut(row.Platform, "/")
	h := hostfit.Host{GPUName: row.Name, SM: row.SM, SMSource: "table", GPUClass: row.Class, GPUVendor: row.Vendor,
		CPUArch: arch, VRAMGB: row.VRAMGB}
	var checks []imagecheck.Check
	for _, c := range hostfit.Fit(img, h, hostfit.FitOptions{}) {
		if !strings.HasPrefix(c.Code, "host_cuda") && c.Code != "runpod_no_cuda_capacity" {
			checks = append(checks, c)
		}
	}
	verdict = imagecheck.Worst(checks)
	if verdict != imagecheck.LevelBlock && verdict != imagecheck.LevelWarn {
		f := hostfit.FilterFor(img.Requires, hostfit.Need{}, nil, hostfit.Policy{})
		if slices.Contains(f.SMHidden, row.SM) {
			checks = append(checks, imagecheck.Check{Level: imagecheck.LevelWarn, Code: "gpu_sm_hidden",
				Message: fmt.Sprintf("The %s (sm_%d) is hidden from offers by default: training on it is slow and may not fit.", row.Name, row.SM)})
			verdict = imagecheck.LevelWarn
		}
	}
	for _, level := range []string{imagecheck.LevelBlock, imagecheck.LevelWarn, imagecheck.LevelInfo} {
		for _, c := range checks {
			if c.Level == level {
				return verdict, c.Message
			}
		}
	}
	if b := buildFor(row.SM, img.Requires.SM); b > 0 && b != row.SM {
		note = fmt.Sprintf("Runs via the sm_%d build", b)
	}
	return verdict, note
}

// buildFor is the SASS target a GPU of sm runs: the highest same-major
// target at or below it; 0 when the image has none.
func buildFor(sm int, sass []int) int {
	best := 0
	for _, s := range sass {
		if s/10 == sm/10 && s%10 <= sm%10 && s > best {
			best = s
		}
	}
	return best
}

// ---- POST /v1/hosts/fit ----

type fitGPU struct {
	Name   string  `json:"name"`
	VRAMGB float64 `json:"vram_gb"`
	Count  int     `json:"count"`
}

type fitRequest struct {
	Model *hostfit.ModelRef `json:"model"`
	Image string            `json:"image"`
	GPUs  []fitGPU          `json:"gpus"`
	Host  *hostfit.Host     `json:"host"`
}

const maxFitGPUs = 32

func hostFit(w http.ResponseWriter, r *http.Request) {
	var in fitRequest
	if !v1api.Decode(w, r, &in) {
		return
	}
	m := model(in.Model)
	if m == nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "model.arch is required")
		return
	}
	if len(in.GPUs) > maxFitGPUs {
		v1api.Fail(w, v1api.CodeInvalidRequest, fmt.Sprintf("at most %d gpus", maxFitGPUs))
		return
	}
	var tiers []hostfit.Tier
	for i, g := range in.GPUs {
		vram := g.VRAMGB
		if vram == 0 {
			if row, ok := hostfit.Lookup(g.Name); ok {
				vram = row.VRAMGB
			}
		}
		if !(vram > 0) || vram > 1024 {
			v1api.Fail(w, v1api.CodeInvalidRequest, fmt.Sprintf("gpus[%d] needs vram_gb (1 to 1024) or a known name", i))
			return
		}
		tiers = append(tiers, tierFor(*m, vram, g.Count))
	}
	if in.Host != nil && !in.Host.IsZero() {
		if h := completeHost(*in.Host); h.VRAMGB > 0 {
			tiers = append(tiers, tierFor(*m, h.VRAMGB, h.GPUCount))
		}
	}
	if tiers == nil {
		tiers = hostfit.Tiers(*m)
	}
	ref := strings.TrimSpace(in.Image)
	if ref == "" {
		ref = provision.DefaultImage
	}
	img, err := inspect(r.Context(), ref, "", false)
	if err != nil {
		img, _ = hostfit.FallbackImage(ref)
	}
	v1api.OK(w, HostFit{Model: *m, Tiers: tiers, Need: hostfit.NeedFor(img, m, hostfit.DefaultSizer)})
}

// tierFor is the verdict for one GPU of vramGB; a machine with more GPUs
// still trains each job on one.
func tierFor(m hostfit.ModelRef, vramGB float64, count int) hostfit.Tier {
	t := hostfit.Tier{VRAMGB: vramGB, VRAMFit: hostfit.VRAMFor(m, vramGB)}
	if count > 1 {
		t.Note += " Each job uses one GPU."
	}
	return t
}
