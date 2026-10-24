package hostfit

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/imagecheck"
)

// ImageRecommendInput is what the image resolver picks for.
type ImageRecommendInput struct {
	Repo           string    // default DefaultRepo
	Host           Host      // zero = no host: image-side checks only
	Model          *ModelRef // disk need and VRAM hints
	Prefer         string    // "same" (default) | "latest"
	ServerRevision string    // the build this server runs (handlers/v1 version()); a short or full sha
	ServerVersion  string    // the upstream version this server carries ("0.13.34"), for the skew line
	Custom         string    // a user-typed ref: the only candidate
	TemplateImage  string    // QuickPod: the chosen template's image_path:version_tag, the only image it can run
	DiskGB         int       // the disk the launch will ask for; 0 = the host's maximum
	DigestRef      bool      // the provider accepts repo@sha256 references (Caps.ImageDigestRef "yes")
}

// ImageChoice is one candidate image and its verdict for the host.
type ImageChoice struct {
	Channel   string   `json:"channel"` // same | latest | flavor | custom | template
	Label     string   `json:"label"`   // "AI Toolkit 0.13.34 · build f53a407"
	Image     string   `json:"image"`   // the reference as the user picks it
	LaunchRef *string  `json:"launch_ref"`
	Digest    string   `json:"digest"`
	Revision  string   `json:"revision,omitempty"`
	Flavor    string   `json:"flavor,omitempty"`
	Level     string   `json:"level"` // ok | info | warn | block
	Why       []string `json:"why"`

	Checks []imagecheck.Check `json:"-"` // every check for this candidate
	Info   *imagecheck.Image  `json:"-"` // nil when the image could not be read
	err    error              // why the image could not be read
}

// ImageRecommendation is the resolver's answer: the choice (null when every
// candidate blocks), the other candidates, and the checks for the choice
// (or why nothing fits).
type ImageRecommendation struct {
	Choice       *ImageChoice            `json:"choice"`
	Alternatives []ImageChoice           `json:"alternatives"`
	Checks       []imagecheck.Check      `json:"checks"`
	Action       *imagecheck.CheckAction `json:"action,omitempty"` // filter_hosts when nothing fits for host reasons
}

var (
	shaRev     = regexp.MustCompile(`^[0-9a-f]{7,40}$`)
	flavorChan = regexp.MustCompile(`^cu[0-9]{3}$`)
)

// flavorTagLimit bounds the tags inspected to find flavor channels. Tags
// lists latest first, then the cuXXX channels, so a few cover them.
const flavorTagLimit = 6

// hostSide are the block codes a host filter can fix.
var hostSide = []string{"host_cuda", "gpu_sm", "host_arch", "gpu_vendor", "disk_image"}

// RecommendImage picks the image for a host and says why (IMAGE-PLAN §4):
// a QuickPod template's image or a custom image is the only candidate;
// otherwise the build this server runs when it is published, then the
// newest build, then other flavor channels. The choice is the first
// candidate with no block; a different trainer version is never chosen
// silently (its first why line says so). Registry trouble is a check, never
// an error: the error is only for a cancelled context or no client.
func RecommendImage(ctx context.Context, ic *imagecheck.Client, sizer *ModelSizer, in ImageRecommendInput) (ImageRecommendation, error) {
	if ic == nil {
		return ImageRecommendation{}, errors.New("hostfit: no image check client")
	}
	repo := in.Repo
	if repo == "" {
		repo = DefaultRepo
	}
	if r, err := imagecheck.ParseRef(repo); err == nil {
		repo = r.Name()
	}
	rev := in.ServerRevision
	if shaRev.MatchString(rev) {
		rev = rev[:7]
	} else {
		rev = ""
	}
	r := &resolver{ic: ic, sizer: sizer, in: in, rev: rev}

	var cands []ImageChoice
	var notes []imagecheck.Check
	var tagAlt *ImageChoice // QuickPod: the template's image with a newer tag, an alternative only
	switch {
	case in.TemplateImage != "":
		c := r.evaluate(ctx, "template", in.TemplateImage)
		cands = append(cands, c)
		if alt, note, ok := r.templateTag(ctx, c); ok {
			tagAlt = &alt
			notes = append(notes, note)
		}
	case in.Custom != "":
		cands = append(cands, r.evaluate(ctx, "custom", in.Custom))
	default:
		order := []string{"same", "latest"}
		if in.Prefer == "latest" {
			order = []string{"latest", "same"}
		}
		for _, ch := range order {
			switch ch {
			case "same":
				if rev == "" {
					continue
				}
				c := r.evaluate(ctx, "same", repo+":"+rev)
				if imagecheck.KindOf(c.err) == imagecheck.KindNotFound {
					notes = append(notes, imagecheck.Check{Level: imagecheck.LevelInfo, Code: "image_same_unpublished",
						Message: fmt.Sprintf("This server's build (%s) isn't published; using the newest build.", rev)})
					continue
				}
				cands = appendNew(cands, c)
			case "latest":
				cands = appendNew(cands, r.evaluate(ctx, "latest", repo+":latest"))
			}
		}
		if err := ctx.Err(); err != nil {
			return ImageRecommendation{}, err
		}
		if tags, err := ic.Tags(ctx, repo, flavorTagLimit); err == nil {
			for _, t := range tags {
				if flavorChan.MatchString(t.Tag) && !hasDigest(cands, t.Digest) {
					cands = appendNew(cands, r.evaluate(ctx, "flavor", repo+":"+t.Tag))
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return ImageRecommendation{}, err
	}

	out := ImageRecommendation{Alternatives: []ImageChoice{}, Checks: []imagecheck.Check{}}
	choice := slices.IndexFunc(cands, func(c ImageChoice) bool { return c.Level != imagecheck.LevelBlock })
	if choice < 0 {
		if len(cands) > 0 {
			first := cands[0]
			allHost := true
			for _, c := range first.Checks {
				if c.Level == imagecheck.LevelBlock {
					out.Checks = append(out.Checks, c)
					allHost = allHost && slices.Contains(hostSide, c.Code)
				}
			}
			if allHost && len(out.Checks) > 0 {
				out.Action = &imagecheck.CheckAction{Kind: "filter_hosts"}
			}
		}
		out.Alternatives = append(out.Alternatives, cands...)
		for i := range out.Alternatives {
			r.whyChannel(&out.Alternatives[i], cands, false)
		}
		if tagAlt != nil {
			r.whyChannel(tagAlt, cands, false)
			out.Alternatives = append(out.Alternatives, *tagAlt)
			out.Checks = append(out.Checks, notes...)
			if tagAlt.Level != imagecheck.LevelBlock { // the template's tag is the fix
				out.Action = &imagecheck.CheckAction{Kind: "edit_template", Value: tagAlt.Revision}
			}
		}
		return out, nil
	}
	c := cands[choice]
	for i, alt := range cands {
		if i != choice {
			r.whyChannel(&alt, cands, false)
			out.Alternatives = append(out.Alternatives, alt)
		}
	}
	if tagAlt != nil {
		r.whyChannel(tagAlt, cands, false)
		out.Alternatives = append(out.Alternatives, *tagAlt)
	}
	r.whyChannel(&c, cands, true)
	out.Checks = append(out.Checks, c.Checks...)
	out.Checks = append(out.Checks, notes...)
	if skew, ok := r.skew(c, cands); ok {
		out.Checks = append(out.Checks, skew)
	}
	out.Choice = &c
	return out, nil
}

// resolver carries one RecommendImage call.
type resolver struct {
	ic    *imagecheck.Client
	sizer *ModelSizer
	in    ImageRecommendInput
	rev   string // this server's build, short sha; "" when unknown
}

// evaluate inspects ref and judges it for the host: the image-side checks
// (or why it could not be read), the host-side checks when there is a host
// and the image did not block, and the launch reference.
func (r *resolver) evaluate(ctx context.Context, channel, ref string) ImageChoice {
	c := ImageChoice{Channel: channel, Image: ref, Label: ref, Why: []string{}}
	img, err := r.ic.Inspect(ctx, ref, "", false)
	c.err = err
	var checks []imagecheck.Check
	hostImg, judgeHost := img, true
	if err != nil {
		ec := imagecheck.ErrorCheck(err)
		if imagecheck.KindOf(err) == imagecheck.KindNotFound && channel != "same" { // a missing "same" build is skipped
			if rf, perr := imagecheck.ParseRef(ref); perr == nil {
				tags, _ := r.ic.Tags(ctx, rf.Name(), 0)
				ec = imagecheck.NotFoundCheck(err, tags)
			}
		}
		checks = append(checks, ec)
		hostImg, _ = FallbackImage(ref) // nil for other repos: image-independent host checks only
		judgeHost = ec.Level != imagecheck.LevelBlock
	} else {
		c.Info = img
		c.Digest, c.Revision, c.Flavor = img.Digest, img.Revision, img.Requires.Flavor
		c.Label = imageLabel(img, ref)
		checks = append(checks, imagecheck.ImageChecks(img)...)
	}
	if judgeHost && !r.in.Host.IsZero() {
		need := NeedFor(hostImg, r.in.Model, r.sizer)
		checks = append(checks, Fit(hostImg, r.in.Host, FitOptions{Need: &need, DiskGB: r.in.DiskGB, Model: r.in.Model})...)
	}
	if imagecheck.Worst(checks) != imagecheck.LevelBlock {
		launch := ref
		if channel != "template" && img != nil {
			var pin *imagecheck.Check
			launch, pin = LaunchRef(ctx, r.ic, img, ref, r.in.DigestRef)
			if pin != nil {
				checks = append(checks, *pin)
			}
		}
		c.LaunchRef = &launch
	}
	c.Checks = checks
	c.Level = imagecheck.Worst(checks)
	c.Why = r.why(c, hostImg)
	return c
}

// appendNew adds c unless an earlier candidate is the same image (digest).
func appendNew(cands []ImageChoice, c ImageChoice) []ImageChoice {
	if c.Digest != "" && hasDigest(cands, c.Digest) {
		return cands
	}
	return append(cands, c)
}

func hasDigest(cands []ImageChoice, digest string) bool {
	return digest != "" && slices.ContainsFunc(cands, func(c ImageChoice) bool { return c.Digest == digest })
}

// imageLabel is "AI Toolkit 0.13.34 · build f53a407" for a Go-server image
// (plus the flavor when it is not cu130), else the reference.
func imageLabel(img *imagecheck.Image, ref string) string {
	if !img.GoServer {
		return ref
	}
	s := "AI Toolkit"
	if v := goVersion(img); v != "" {
		s += " " + v
	}
	switch {
	case img.Revision != "":
		s += " · build " + img.Revision
	case img.Tag != "":
		s += " · " + img.Tag
	}
	if f := img.Requires.Flavor; f != "" && f != "cu130" {
		s += " · " + f
	}
	return s
}

// goVersion is the upstream version a Go-server image carries (its OCI
// version label). Images without the Go labels inherit Ubuntu's, so only a
// labelled Go image counts.
func goVersion(img *imagecheck.Image) string {
	if img.Labels[imagecheck.LabelServer] != "go" {
		return ""
	}
	return img.Labels[imagecheck.LabelVersion]
}

// build is "0.13.34 · f53a407", or the part that is known.
func build(version, rev string) string {
	switch {
	case version != "" && rev != "":
		return version + " · " + rev
	case rev != "":
		return rev
	}
	return version
}

// why are the fact lines after the channel line (added by whyChannel):
// driver, GPU, size, pin.
func (r *resolver) why(c ImageChoice, img *imagecheck.Image) []string {
	var out []string
	h := r.in.Host
	if img != nil && img.Requires.CUDAMin != "" {
		line := "Needs CUDA " + img.Requires.CUDAMin
		switch {
		case h.CUDAMax != "":
			line += "; this host has " + h.CUDAMax
		case h.PlacementFloor:
			line += "; RunPod places it on a host that has it"
		}
		out = append(out, line)
	}
	if img != nil && h.SM > 0 && img.Requires.Runs(h.SM) {
		if b := buildFor(h.SM, img.Requires.SM); b != h.SM {
			out = append(out, fmt.Sprintf("Runs on sm_%d through the sm_%d build", h.SM, b))
		} else {
			out = append(out, fmt.Sprintf("Runs on sm_%d (%s)", h.SM, gpuDisplay(h)))
		}
	}
	if c.Info != nil {
		line := fmt.Sprintf("%.1f GB download", float64(c.Info.DownloadBytes)/1e9)
		if p := PullFor(c.Info, h, 0); p != nil {
			line += fmt.Sprintf(", about %s at %.0f Mbps", mins(p.ImageMin), h.InetDownMbps)
		}
		out = append(out, line)
	}
	if c.LaunchRef != nil && c.Info != nil && c.Revision != "" && strings.HasSuffix(*c.LaunchRef, ":"+c.Revision) {
		out = append(out, fmt.Sprintf("Pinned to build %s (%s)", c.Revision, shortDigest(c.Digest)))
	} else if c.LaunchRef != nil && strings.Contains(*c.LaunchRef, "@sha256:") {
		out = append(out, "Pinned to "+shortDigest(c.Digest))
	}
	return out
}

// whyChannel puts the channel line first. Only the choice says it is a
// newer build than this server (it replaces a build that cannot run).
func (r *resolver) whyChannel(c *ImageChoice, cands []ImageChoice, chosen bool) {
	var line string
	switch c.Channel {
	case "same":
		line = "Same build as this server"
	case "latest":
		line = "Newest build"
		sameTried := slices.ContainsFunc(cands, func(o ImageChoice) bool { return o.Channel == "same" && o.Level == imagecheck.LevelBlock })
		if chosen && sameTried && r.rev != "" && c.Revision != "" && c.Revision != r.rev {
			line = fmt.Sprintf("Newer build than this server (%s → %s)", build(r.in.ServerVersion, r.rev), build(goVersion(c.Info), c.Revision))
		}
	case "flavor":
		line = fmt.Sprintf("Different flavor (%s) for this host's driver", c.Flavor)
	case "custom":
		line = "Custom image"
	case "template":
		line = "The image of the QuickPod template"
		if c.Image != r.in.TemplateImage {
			line = "Edit the template's version tag to " + c.Revision
		}
	}
	c.Why = append([]string{line}, c.Why...)
}

// skew is the image_skew info when the choice is a newer build than this
// server while this server's own build would also run.
func (r *resolver) skew(c ImageChoice, cands []ImageChoice) (imagecheck.Check, bool) {
	if c.Channel != "latest" || r.rev == "" || c.Revision == "" || c.Revision == r.rev {
		return imagecheck.Check{}, false
	}
	i := slices.IndexFunc(cands, func(o ImageChoice) bool { return o.Channel == "same" && o.Level != imagecheck.LevelBlock })
	if i < 0 {
		return imagecheck.Check{}, false
	}
	return imagecheck.Check{Level: imagecheck.LevelInfo, Code: "image_skew",
		Message: fmt.Sprintf("This is a newer build than this server (%s → %s).", r.rev, c.Revision),
		Action:  &imagecheck.CheckAction{Kind: "use_image", Value: cands[i].Image}}, true
}

// templateTag: a QuickPod template whose tag is not the newest build gets
// an alternative and an info with the edit_template action (QuickPod runs
// only the template's image, so the fix is to edit its version tag). The
// alternative is never the choice: the studio cannot launch it directly.
func (r *resolver) templateTag(ctx context.Context, tmpl ImageChoice) (ImageChoice, imagecheck.Check, bool) {
	ref, err := imagecheck.ParseRef(tmpl.Image)
	if err != nil {
		return ImageChoice{}, imagecheck.Check{}, false
	}
	latest := r.evaluate(ctx, "template", ref.WithTag("latest").String())
	rev := latest.Revision
	if rev == "" || ref.Tag == rev || ref.Digest != "" || latest.Level == imagecheck.LevelBlock && tmpl.Level != imagecheck.LevelBlock {
		return ImageChoice{}, imagecheck.Check{}, false
	}
	latest.Label = "Edit the template's version tag to " + rev
	latest.Image = ref.WithTag(rev).String()
	if latest.LaunchRef != nil { // nil when the newer tag blocks too
		launch := latest.Image
		latest.LaunchRef = &launch
	}
	note := imagecheck.Check{Level: imagecheck.LevelInfo, Code: "image_template_tag",
		Message: fmt.Sprintf("Edit the template's version tag to %s.", rev),
		Fix:     "QuickPod runs the template's image; its tag decides the build.",
		Action:  &imagecheck.CheckAction{Kind: "edit_template", Value: rev}}
	return latest, note, true
}

// LaunchRef is the reference a launch sends for img, asked as asked
// (IMAGE-PLAN §4.4): repo@digest when the provider accepts digests;
// repo:<revision> for a Go-server image whose revision tag points at the
// same digest (an info says what a moving tag resolved to); otherwise the
// tag as asked, with an info that it can move.
func LaunchRef(ctx context.Context, ic *imagecheck.Client, img *imagecheck.Image, asked string, digestRef bool) (string, *imagecheck.Check) {
	r, err := imagecheck.ParseRef(asked)
	if err != nil || img == nil {
		return asked, nil
	}
	if r.Digest != "" {
		return r.Pinned(r.Digest), nil
	}
	if digestRef && img.Digest != "" {
		return r.Pinned(img.Digest), nil
	}
	if img.GoServer && img.Revision != "" {
		if r.Tag == img.Revision {
			return r.String(), nil
		}
		pinned := r.WithTag(img.Revision)
		if d, err := ic.HeadDigest(ctx, pinned.String()); err == nil && d == img.Digest {
			return pinned.String(), &imagecheck.Check{Level: imagecheck.LevelInfo, Code: "image_unpinned",
				Message: fmt.Sprintf("%s is build %s today.", r.Tag, img.Revision),
				Fix:     "This server will run exactly that build."}
		}
	}
	if shaRev.MatchString(r.Tag) {
		return r.String(), nil
	}
	return r.String(), &imagecheck.Check{Level: imagecheck.LevelInfo, Code: "image_unpinned",
		Message: "This tag can move.", Fix: "The server will run what it points at when the host pulls it."}
}

// shortDigest is "sha256:d89f…59e30b".
func shortDigest(d string) string {
	hex, ok := strings.CutPrefix(d, "sha256:")
	if !ok || len(hex) < 12 {
		return d
	}
	return "sha256:" + hex[:4] + "…" + hex[len(hex)-6:]
}
