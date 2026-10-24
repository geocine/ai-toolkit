package provision

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/netguard"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// ImageChecker is the image check the offers decoration and the launch
// re-check go through, read at call time. A nil client reads no image: the
// default image is then judged from its built-in requirements and other
// images are not judged. Tests point it at registrymock (imagetest.Use).
var ImageChecker = func() *imagecheck.Client { return imagecheck.Default }

// imageView is what one offers call or launch knows about its image.
type imageView struct {
	ref      string            // as asked
	img      *imagecheck.Image // read, or the default image's fallback; nil when neither
	fallback bool              // img is hostfit.FallbackImage: the registry could not be read
	err      error             // why the image could not be read; nil when it was
	need     hostfit.Need
}

// inspectImage reads ref through ImageChecker (cached there) and sizes the
// disk a launch needs with model. When the default image cannot be read
// (not published yet, private, unreachable) its fallback stands in, so
// hosts are still judged (MASTER §3.10.3).
func inspectImage(ctx context.Context, ref string, model *hostfit.ModelRef) imageView {
	v := imageView{ref: ref}
	if ic := ImageChecker(); ic != nil {
		v.img, v.err = ic.Inspect(ctx, ref, "", false)
	} else {
		v.err = imagecheck.Errorf(imagecheck.KindUnreachable, "no image check is configured on this server")
	}
	if v.err != nil {
		v.img = nil
		if fb, ok := hostfit.FallbackImage(ref); ok {
			v.img, v.fallback = fb, true
		}
	}
	v.need = hostfit.NeedFor(v.img, model, hostfit.DefaultSizer)
	return v
}

// withImageFilter fills q.Filter and q.ImageInfo from q.Image (DefaultImage
// when empty) for the adapters. An image that cannot be read and has no
// fallback leaves both nil: adapters then filter only on their own fields
// and offers carry no fit.
func withImageFilter(ctx context.Context, q OfferQuery) OfferQuery {
	ref := strings.TrimSpace(q.Image)
	if ref == "" {
		ref = DefaultImage
	}
	v := inspectImage(ctx, ref, q.Model)
	if v.img == nil {
		q.Filter, q.ImageInfo = nil, nil
		return q
	}
	f := hostfit.FilterFor(v.img.Requires, v.need, q.Model, hostfit.Policy{Fast: q.Fast, ShowAll: q.ShowAll})
	q.Filter, q.ImageInfo = &f, v.img
	return q
}

// DecorateOffer sets o.Host to h and, when q carries the service's image
// filter, o.Fit: the host-side checks of the image on h plus the VRAM
// verdict for q.Model. The image's own failure, if any, is never in the
// fit (the Image row shows it once). Adapters call it for every offer and
// quote they return.
func DecorateOffer(o *Offer, h hostfit.Host, q OfferQuery) {
	host := h
	o.Host, o.Fit = &host, nil
	if q.Filter == nil {
		return
	}
	need := hostfit.Need{DiskGB: q.Filter.MinDiskGB, ImagePeakGB: q.Filter.ImagePeakGB, WorkGB: hostfit.WorkGB}
	vram := hostfit.VRAMFit{Verdict: hostfit.VRAMUnknown}
	if q.Model != nil && h.VRAMGB > 0 {
		vram = hostfit.VRAMFor(*q.Model, h.VRAMGB)
	}
	fit := hostfit.OfferFitFor(q.ImageInfo, h, need, vram)
	o.Fit = &fit
}

// KeepOffer applies q's image filter to h client-side (hostfit.Keeps): true
// without a filter.
func KeepOffer(h hostfit.Host, q OfferQuery) bool {
	if q.Filter == nil {
		return true
	}
	keep, _ := hostfit.Keeps(*q.Filter, h)
	return keep
}

// ---- the launch re-check (IMAGE-PLAN §4.5) ----

type launchCUDAKey struct{}

// WithLaunchCUDA hands the image's CUDA floor to the adapter's Launch:
// RunPod sends it as gpu.minCudaVersion, so RunPod places the pod only on a
// host whose driver runs the image.
func WithLaunchCUDA(ctx context.Context, cudaMin string) context.Context {
	if cudaMin == "" {
		return ctx
	}
	return context.WithValue(ctx, launchCUDAKey{}, cudaMin)
}

// LaunchCUDAMin is the CUDA floor ("13.0") of the image a Launch starts; ""
// when the image says none.
func LaunchCUDAMin(ctx context.Context) string {
	v, _ := ctx.Value(launchCUDAKey{}).(string)
	return v
}

// ImageBlockedError refuses a launch the image cannot run (422
// image_blocked); details.checks are every check of the image on the
// re-quoted host. No provider call is made.
type ImageBlockedError struct {
	Message string
	Checks  []imagecheck.Check
}

func (e *ImageBlockedError) Error() string { return e.Message }

// V1Details is the 422 body (v1machines.DetailedError).
func (e *ImageBlockedError) V1Details() (code, message string, details any) {
	return v1api.CodeImageBlocked, e.Message, map[string]any{"checks": e.Checks}
}

// warnImageSkipped is the image warning of a launch that skipped its check.
const warnImageSkipped = "image check skipped by request"

// skippable are the image-side blocks skip_image_check waives for a custom
// image: the studio can't drive it, but the user asked for it on purpose.
// Host-side blocks and an image the host cannot pull never are.
var skippable = map[string]bool{"image_not_go": true, "image_port": true}

// imageSnapshot is the machines.spec image_check JSON.
type imageSnapshot struct {
	Image     string             `json:"image"`
	LaunchRef string             `json:"launch_ref"`
	Digest    string             `json:"digest,omitempty"`
	Revision  string             `json:"revision,omitempty"`
	Level     string             `json:"level"`
	Checks    []imagecheck.Check `json:"checks"`
	Fallback  bool               `json:"fallback,omitempty"`
	Skipped   bool               `json:"skipped,omitempty"`
	CheckedAt time.Time          `json:"checked_at"`
}

// launchImage is the outcome of a launch's image check.
type launchImage struct {
	launchRef string
	digest    string
	cudaMin   string
	warnings  []string // image_warnings: every warn-level check
	machine   []string // Machine.warnings: the registry could not be read (IMAGE-PLAN §4.5 step 4)
	pullS     float64
	snapshot  json.RawMessage
}

// checkLaunchImage judges image on the re-quoted offer at diskGB (IMAGE-PLAN
// §4.5): any block refuses the launch with ImageBlockedError, except an
// image-side block of a custom image the caller skipped. An image that
// cannot be read is a warning, unless the registry says it is missing or
// private (D-17: the default image then still blocks).
func checkLaunchImage(ctx context.Context, k Kind, v imageView, offer Offer, diskGB int, skip bool, now time.Time) (launchImage, error) {
	checks := []imagecheck.Check{}
	if v.err != nil {
		checks = append(checks, imageErrorCheck(ctx, v.err))
	}
	if v.img != nil && !v.fallback {
		checks = append(checks, imagecheck.ImageChecks(v.img)...)
	}
	if offer.Host != nil {
		checks = append(checks, hostfit.Fit(v.img, *offer.Host, hostfit.FitOptions{Need: &v.need, DiskGB: diskGB, Launch: true})...)
	}
	var blocks []imagecheck.Check
	waived := true
	for _, c := range checks {
		if c.Level == imagecheck.LevelBlock {
			blocks = append(blocks, c)
			if !skippable[c.Code] {
				waived = false
			}
		}
	}
	repo := v.ref
	if r, err := imagecheck.ParseRef(v.ref); err == nil {
		repo = r.Name()
	}
	skipped := skip && (len(blocks) == 0 || waived && repo != imagecheck.DefaultRepo)
	if len(blocks) > 0 && !skipped {
		msg := "The image can't run on this GPU server: " + blocks[0].Message
		if skip {
			msg += " skip_image_check doesn't bypass this."
		}
		return launchImage{}, &ImageBlockedError{Message: msg, Checks: checks}
	}

	out := launchImage{launchRef: v.ref}
	if v.img != nil && !v.fallback {
		out.digest = v.img.Digest
		// QuickPod runs the template's image as it is: createpod has no image field
		if k != QuickPod {
			ref, info := hostfit.LaunchRef(ctx, ImageChecker(), v.img, v.ref, capsOf(k).ImageDigestRef == "yes")
			out.launchRef = ref
			if info != nil {
				checks = append(checks, *info)
			}
		}
	}
	if v.img != nil {
		out.cudaMin = v.img.Requires.CUDAMin
		if offer.Host != nil {
			if p := hostfit.PullFor(v.img, *offer.Host, 0); p != nil {
				out.pullS = p.ImageMin * 60
			}
		}
	}
	if skipped {
		out.warnings = append(out.warnings, warnImageSkipped)
	}
	for _, c := range checks {
		if c.Level == imagecheck.LevelWarn {
			out.warnings = append(out.warnings, c.Message)
			if c.Code == "image_unreachable" {
				out.machine = append(out.machine, "Couldn't check the image before launch; problems will show in the provider's log.")
			}
		}
	}
	snap := imageSnapshot{Image: v.ref, LaunchRef: out.launchRef, Digest: out.digest, Level: imagecheck.Worst(checks),
		Checks: checks, Fallback: v.fallback, Skipped: skipped, CheckedAt: now.UTC()}
	if v.img != nil && !v.fallback {
		snap.Revision = v.img.Revision
	}
	out.snapshot, _ = json.Marshal(snap)
	return out, nil
}

// imageErrorCheck is the check for an image that could not be read; a
// missing tag names the newest build when the repo's tags can be listed.
func imageErrorCheck(ctx context.Context, err error) imagecheck.Check {
	var e *imagecheck.Error
	if ic := ImageChecker(); ic != nil && errors.As(err, &e) && e.Kind == imagecheck.KindNotFound && e.Repo != "" {
		if tags, terr := ic.Tags(ctx, e.Repo, 0); terr == nil {
			return imagecheck.NotFoundCheck(err, tags)
		}
	}
	return imagecheck.ErrorCheck(err)
}

// snapshotOf reads a machine's image_check snapshot (zero when absent).
func snapshotOf(r *row) imageSnapshot {
	var s imageSnapshot
	if len(r.Spec.ImageCheck) > 0 {
		_ = json.Unmarshal(r.Spec.ImageCheck, &s)
	}
	return s
}

// ---- after launch (IMAGE-PLAN §4.6) ----

// Pull watchdog: a machine still allocating or pulling after
// max(3 x the predicted pull, 15 min) gets pull_slow.
const (
	pullSlowFactor = 3
	pullSlowMin    = 15 * time.Minute
)

func pullSlowAfter(pullS float64) time.Duration {
	return max(time.Duration(pullS*pullSlowFactor*float64(time.Second)), pullSlowMin)
}

func pullSlowDetail(k Kind) string {
	d := "The image is taking far longer than this host's bandwidth predicts. Keep waiting, or terminate it."
	if k == Vast {
		d += " Vast.ai doesn't bill while it's loading."
	}
	return d
}

// failureDetail classifies a provider failure text (status, detail, the
// log tail) into a machine problem and its copy. A pull the registry
// refused for the default image gets the "not published yet" copy (MASTER
// §3.4). ok is false when nothing known matches.
func failureDetail(text, launchRef string, diskGB int) (Problem, string, bool) {
	f, ok := imagecheck.ClassifyFailure(text)
	if !ok {
		return "", "", false
	}
	f = f.WithDisk(max(diskGB+diskGB/2, hostfit.MinDiskGBUnknownModel))
	detail := f.Message
	if f.Code == "image_denied" {
		repo := launchRef
		if r, err := imagecheck.ParseRef(launchRef); err == nil {
			repo = r.Name()
		}
		if repo == imagecheck.DefaultRepo {
			msg, fix := imagecheck.DeniedCopy(repo)
			detail = msg + " " + fix
		}
	}
	if f.Problem == imagecheck.ProblemCUDA {
		return ProblemCUDAFailed, detail, true
	}
	return ProblemImagePullFailed, detail, true
}

// shaBuild is a build id: a short or full git sha.
var shaBuild = regexp.MustCompile(`^[0-9a-f]{7,40}$`)

// expectedBuild is the build a machine should report in /v1/health: the
// sha tag of its launch ref, else the revision the image check read.
func expectedBuild(r *row) string {
	if ref, err := imagecheck.ParseRef(r.ImageLaunchRef); err == nil && shaBuild.MatchString(ref.Tag) {
		return ref.Tag
	}
	return snapshotOf(r).Revision
}

// sameBuild: one sha is a prefix of the other (short and full forms).
func sameBuild(a, b string) bool {
	if len(a) > len(b) {
		a, b = b, a
	}
	return a != "" && strings.HasPrefix(b, a)
}

func buildWarning(running, want string) string {
	return fmt.Sprintf("The server is running build %s, not %s (the host may have had an older copy cached).", running, want)
}

// cudaResult is the part of a machine's GET /v1/system/cuda the service reads.
type cudaResult struct {
	Found   bool    // false: the route is missing (an older image); skipped silently
	OK      bool    `json:"ok"`
	Error   *string `json:"error"`
	Failure *string `json:"failure"`
}

// CUDA check timing: one call waits at most cudaCallTimeout (the machine
// bounds its check at 90 s); a machine that never answers is left alone
// after cudaTries calls.
const (
	cudaCallTimeout = 100 * time.Second
	cudaTries       = 3
)

var cudaClients struct {
	once             sync.Once
	strict, loopback *http.Client
}

// cudaClient is the client for a machine's CUDA check. The machine sends no
// headers until its check is done, so the header timeout is the whole call's,
// not netguard's 8 s default.
func cudaClient(simulated bool) *http.Client {
	cudaClients.once.Do(func() {
		o := netguard.Options{Timeout: cudaCallTimeout + 5*time.Second, HeaderTimeout: cudaCallTimeout}
		cudaClients.strict = netguard.NewClient(o)
		o.AllowLoopback = true
		cudaClients.loopback = netguard.NewClient(o)
	})
	if simulated {
		return cudaClients.loopback
	}
	return cudaClients.strict
}

// cudaCheck asks a ready machine whether its trainer's CUDA runs (tests swap it).
var cudaCheck = func(ctx context.Context, baseURL, token string, simulated bool) (cudaResult, error) {
	c := cudaClient(simulated)
	ctx, cancel := context.WithTimeout(ctx, cudaCallTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/system/cuda", nil)
	if err != nil {
		return cudaResult{}, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	res, err := c.Do(req)
	if err != nil {
		return cudaResult{}, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, probeMax))
	if err != nil {
		return cudaResult{}, err
	}
	switch res.StatusCode {
	case http.StatusNotFound:
		return cudaResult{}, nil
	case http.StatusOK:
	default:
		return cudaResult{}, fmt.Errorf("GET /v1/system/cuda answered %s", http.StatusText(res.StatusCode))
	}
	out := cudaResult{Found: true}
	if err := json.Unmarshal(body, &out); err != nil {
		return cudaResult{}, fmt.Errorf("GET /v1/system/cuda: unreadable answer")
	}
	return out, nil
}

// cudaDetail is the copy for a failed CUDA check.
func cudaDetail(res cudaResult) string {
	text := ""
	if res.Error != nil {
		text = *res.Error
	}
	if f, ok := imagecheck.ClassifyFailure(text); ok && f.Problem == imagecheck.ProblemCUDA {
		return f.Message + " Terminate it and pick another GPU."
	}
	if res.Failure != nil && *res.Failure == "no_gpu" {
		return "The trainer sees no GPU on this server. Terminate it and pick another offer."
	}
	if text == "" {
		return "The trainer's CUDA check failed. Terminate it and pick another GPU."
	}
	return "The trainer's CUDA check failed: " + Clip(text, 200)
}
