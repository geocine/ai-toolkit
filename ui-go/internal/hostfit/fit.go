package hostfit

import (
	"fmt"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/imagecheck"
)

// FitOptions are what a check knows beyond the image and the host.
type FitOptions struct {
	Need   *Need     // nil = NeedFor(img, Model, nil) (table model sizes)
	DiskGB int       // the disk the launch asks for; 0 = judge the host's maximum
	Model  *ModelRef // VRAM and RAM hints; nil = none
	Launch bool      // a launch, not the GPU picker: vram_no is an info, not a warn
}

// Fit is the host-side checks of running img on host h (IMAGE-PLAN §3.2,
// §3.3, §3.5, §3.6, §5): CPU architecture, GPU vendor and architecture,
// driver and CUDA, disk, pull time and VRAM, with the copy of IMAGE-PLAN
// §6.4. The image-side checks are imagecheck.ImageChecks. A nil img (it
// could not be read and has no fallback, FallbackImage) runs only the
// checks that need nothing from the image: GPU vendor, VRAM and RAM (its
// size is unknown, so no disk or pull checks). Callers without any host facts
// skip Fit.
func Fit(img *imagecheck.Image, h Host, o FitOptions) []imagecheck.Check {
	checks := []imagecheck.Check{}
	var req *imagecheck.Requirements
	if img != nil {
		req = &img.Requires
	}
	name := gpuDisplay(h)

	if req != nil {
		if c, ok := archCheck(req.Platform, h); ok {
			checks = append(checks, c)
		}
	}
	vendor := h.GPUVendor
	if vendor == vendorAMD {
		checks = append(checks, imagecheck.Check{Level: imagecheck.LevelBlock, Code: "gpu_vendor",
			Message: "AMD GPUs aren't supported by this image.", Fix: "Pick an NVIDIA GPU.",
			Action: &imagecheck.CheckAction{Kind: "pick_gpu"}})
	} else if req != nil && (len(req.SM) > 0 || len(req.PTX) > 0) {
		checks = append(checks, smChecks(*req, h, name)...)
	}
	if req != nil {
		checks = append(checks, cudaChecks(*req, h, name)...)
	}

	need := o.Need
	if need == nil {
		n := NeedFor(img, o.Model, nil)
		need = &n
	}
	if img != nil { // disk and pull time depend on the image's size
		checks = append(checks, diskChecks(*need, h.DiskGB, o.DiskGB, o.Model)...)
		var modelGB float64
		if o.Model != nil {
			modelGB = need.ModelGB
		}
		checks = append(checks, pullChecks(img, PullFor(img, h, modelGB), o.Model)...)
	}
	if o.Model != nil && h.VRAMGB > 0 {
		checks = append(checks, vramChecks(VRAMFor(*o.Model, h.VRAMGB), *o.Model, h, o.Launch)...)
		if o.Model.LowVRAM && h.RAMGB > 0 && need.ModelGB > 0 && h.RAMGB < need.ModelGB {
			checks = append(checks, imagecheck.Check{Level: imagecheck.LevelInfo, Code: "ram_tight",
				Message: fmt.Sprintf("This host has %.0f GB RAM; offloading %s may swap.", h.RAMGB, ArchName(o.Model.Arch))})
		}
	}
	return checks
}

// gpuDisplay is the GPU's table name, else the provider's, else "GPU".
func gpuDisplay(h Host) string {
	if row, ok := Lookup(h.GPUName); ok {
		return row.Name
	}
	if h.GPUName != "" {
		return h.GPUName
	}
	return "GPU"
}

// archOfHost is the host's CPU architecture: the provider's, else the GPU
// table's platform; "" when unknown.
func archOfHost(h Host) string {
	switch a := strings.ToLower(h.CPUArch); a {
	case "x86_64", "x86-64", "x64":
		return "amd64"
	case "aarch64":
		return "arm64"
	case "":
		if row, ok := Lookup(h.GPUName); ok {
			_, arch, _ := strings.Cut(row.Platform, "/")
			return arch
		}
		return ""
	default:
		return a
	}
}

func archCheck(platform string, h Host) (imagecheck.Check, bool) {
	_, want, _ := strings.Cut(platform, "/")
	if want == "" {
		want = "amd64"
	}
	got := archOfHost(h)
	if got == "" || got == want {
		return imagecheck.Check{}, false
	}
	what := "This host is " + got
	if h.GPUName != "" {
		what += " (" + gpuDisplay(h) + ")"
	}
	return imagecheck.Check{Level: imagecheck.LevelBlock, Code: "host_arch",
		Message: what + ". The image runs on " + want + " only.", Fix: "Pick an x86 host.",
		Action: &imagecheck.CheckAction{Kind: "pick_gpu"}}, true
}

// smChecks judge the GPU architecture against the image's torch build.
func smChecks(req imagecheck.Requirements, h Host, name string) []imagecheck.Check {
	sm := h.SM
	if sm == 0 {
		return []imagecheck.Check{{Level: imagecheck.LevelInfo, Code: "gpu_unknown",
			Message: "Couldn't tell which architecture this GPU is."}}
	}
	if !req.Runs(sm) {
		return []imagecheck.Check{{Level: imagecheck.LevelBlock, Code: "gpu_sm",
			Message: fmt.Sprintf("The %s (sm_%d) isn't supported by this image's PyTorch build.", name, sm),
			Fix:     "Pick an RTX 30-series or newer, an A100, L40S or H100.",
			Action:  &imagecheck.CheckAction{Kind: "filter_hosts"}}}
	}
	if len(req.FlashAttnSM) > 0 && !imagecheck.RunsOn(sm, req.FlashAttnSM) {
		msg := fmt.Sprintf("The %s (sm_%d) runs PyTorch but not flash attention", name, sm)
		if sm < 80 {
			msg += ", and has no native bf16"
		}
		return []imagecheck.Check{{Level: imagecheck.LevelWarn, Code: "gpu_sm_flash_attn", Message: msg + ".",
			Fix:    "Training will be slow or may not fit; RTX 30-series or newer is recommended.",
			Action: &imagecheck.CheckAction{Kind: "pick_gpu"}}}
	}
	if row, ok := Lookup(h.GPUName); ok && row.Unverified {
		return []imagecheck.Check{{Level: imagecheck.LevelInfo, Code: "gpu_sm_unverified",
			Message: fmt.Sprintf("The %s runs this image's sm_%d build.", name, buildFor(sm, req.SM)),
			Fix:     "Not yet tested on this GPU."}}
	}
	return nil
}

// buildFor is the SASS target a GPU of sm runs: the highest same-major
// target at or below it (sm itself when the image has no such target).
func buildFor(sm int, sass []int) int {
	best := 0
	for _, s := range sass {
		if s/10 == sm/10 && s%10 <= sm%10 && s > best {
			best = s
		}
	}
	if best == 0 {
		return sm
	}
	return best
}

// cudaChecks judge the host's driver against the image's CUDA floor
// (IMAGE-PLAN §3.2): below it blocks on every GPU class; unknown is a warn,
// or an info where RunPod enforces the floor at placement; on RunPod a
// catalog that lists no available version at the floor is a capacity warn.
func cudaChecks(req imagecheck.Requirements, h Host, name string) []imagecheck.Check {
	min, ok := ParseVersion(req.CUDAMin)
	if !ok {
		return nil
	}
	driverMin := req.DriverMin
	if driverMin == 0 {
		driverMin, _ = BranchForCUDA(min)
	}
	have, known := ParseVersion(h.CUDAMax)
	switch {
	case h.PlacementFloor && (known && have.Less(min) || !known && h.CUDASource == "runpod.catalog"): // none free at the floor
		return []imagecheck.Check{{Level: imagecheck.LevelWarn, Code: "runpod_no_cuda_capacity",
			Message: fmt.Sprintf("No %s hosts with CUDA %s+ are free right now.", name, min),
			Fix:     "The launch may fail with no capacity; try another GPU or later.",
			Action:  &imagecheck.CheckAction{Kind: "pick_gpu"}}}
	case known && have.Less(min):
		c := imagecheck.Check{Level: imagecheck.LevelBlock, Code: "host_cuda",
			Message: fmt.Sprintf("This host's driver supports CUDA %s. The image needs %s (driver %d or newer).", have, min, driverMin),
			Fix:     "Pick a host with a newer driver.",
			Action:  &imagecheck.CheckAction{Kind: "filter_hosts"}}
		branch := 0
		if major, _, _, ok := ParseDriver(h.Driver); ok {
			branch = major
		} else if b, ok := BranchForCUDA(have); ok {
			branch = b
		}
		compat := slices.ContainsFunc(req.CompatBranches, func(b imagecheck.BrandRange) bool { return b.Branch == branch })
		if compat && (h.GPUClass == classPro || h.GPUClass == classDatacenter) {
			c.Message = fmt.Sprintf("This %s's driver (%d) supports CUDA %s. The image needs %s.", name, branch, have, min)
			c.Fix = "Forward compatibility might start it, but it isn't verified for this image."
		}
		return []imagecheck.Check{c}
	case !known && h.PlacementFloor:
		return []imagecheck.Check{{Level: imagecheck.LevelInfo, Code: "host_cuda_placement",
			Message: fmt.Sprintf("RunPod only places this on hosts with CUDA %s or newer.", min)}}
	case !known:
		return []imagecheck.Check{{Level: imagecheck.LevelWarn, Code: "host_cuda_unknown",
			Message: "Couldn't read this host's driver version.",
			Fix:     fmt.Sprintf("If it's older than %d, the server won't start.", driverMin)}}
	}
	return nil
}

// vramChecks: a model that doesn't fit is a warn in the picker and an info
// at launch (the user may know their config better than the table); tight
// is an info naming the recipe. VRAM never blocks.
func vramChecks(v VRAMFit, m ModelRef, h Host, launch bool) []imagecheck.Check {
	name := ArchName(m.Arch)
	size := fmt.Sprintf("%.0f GB", h.VRAMGB) // below the smallest tier: the card's own size
	if i := tierIndex(h.VRAMGB); i >= 0 {
		size = fmt.Sprintf("%.0f GB", TierSizes[i])
	}
	switch v.Verdict {
	case VRAMNo:
		level := imagecheck.LevelWarn
		if launch {
			level = imagecheck.LevelInfo
		}
		c := imagecheck.Check{Level: level, Code: "vram_no", Message: fmt.Sprintf("%s doesn't fit in %s.", name, size),
			Action: &imagecheck.CheckAction{Kind: "pick_gpu"}}
		if t := firstFitting(m); t > 0 {
			c.Fix = fmt.Sprintf("Pick %s %.0f GB or larger GPU.", article(t), t)
		}
		return []imagecheck.Check{c}
	case VRAMTight:
		fix := "Set it in the job's model section."
		if v.Source == "repo" {
			fix = fmt.Sprintf("The %s example config uses it.", size)
		}
		if h.GPUCount > 1 {
			fix += " Each job uses one GPU."
		}
		return []imagecheck.Check{{Level: imagecheck.LevelInfo, Code: "vram_tight",
			Message: fmt.Sprintf("%s fits in %s only with %s.", name, size, v.Recipe), Fix: fix}}
	}
	return nil
}

// firstFitting is the smallest table size where m fits or is tight (0 =
// none), the same rule as MinVRAMGB and the recommender's start tier.
func firstFitting(m ModelRef) float64 {
	for _, t := range Tiers(m) {
		if t.Verdict == VRAMFits || t.Verdict == VRAMTight {
			return t.VRAMGB
		}
	}
	return 0
}

// article is "an" before a size said with a vowel sound ("an 80 GB",
// "an 11 GB", "an 8 GB"), else "a".
func article(gb float64) string {
	switch n := int(gb); {
	case n == 8, n == 11, n == 18, n >= 80 && n < 90, n >= 800 && n < 900:
		return "an"
	}
	return "a"
}

// OfferFitFor is the badge one offer gets: the host-side checks for img on
// h with the given disk need, plus the VRAM verdict (vram_no is a warn in
// the picker). A nil img runs only the image-independent checks and has
// no pull time.
func OfferFitFor(img *imagecheck.Image, h Host, need Need, vram VRAMFit) OfferFit {
	checks := Fit(img, h, FitOptions{Need: &need})
	switch vram.Verdict {
	case VRAMNo:
		checks = append(checks, imagecheck.Check{Level: imagecheck.LevelWarn, Code: "vram_no"})
	case VRAMTight:
		checks = append(checks, imagecheck.Check{Level: imagecheck.LevelInfo, Code: "vram_tight"})
	}
	fit := OfferFit{Level: imagecheck.Worst(checks), Codes: []string{}, VRAM: vram.Verdict}
	if fit.VRAM == "" {
		fit.VRAM = VRAMUnknown
	}
	for _, c := range checks {
		fit.Codes = append(fit.Codes, c.Code)
	}
	if p := PullFor(img, h, 0); p != nil && img != nil {
		m := p.ImageMin
		fit.PullMin = &m
	}
	fit.Summary = summary(checks, h, fit.PullMin)
	return fit
}

// summaries are the short badge lines, by check code.
var summaries = map[string]func(h Host) string{
	"host_arch":               func(h Host) string { return archOfHost(h) + " host" },
	"gpu_vendor":              func(Host) string { return "AMD GPU" },
	"gpu_sm":                  func(h Host) string { return fmt.Sprintf("GPU not supported (sm_%d)", h.SM) },
	"host_cuda":               func(h Host) string { return fmt.Sprintf("Driver too old (CUDA %s)", h.CUDAMax) },
	"disk_image":              func(h Host) string { return fmt.Sprintf("Disk too small (%.0f GB)", h.DiskGB) },
	"host_cuda_unknown":       func(Host) string { return "Driver unknown" },
	"runpod_no_cuda_capacity": func(Host) string { return "No hosts free at this CUDA" },
	"gpu_sm_flash_attn":       func(h Host) string { return fmt.Sprintf("No flash attention (sm_%d)", h.SM) },
	"pull_slow":               func(Host) string { return "Slow download" },
	"disk_model":              func(h Host) string { return fmt.Sprintf("Small disk (%.0f GB)", h.DiskGB) },
	"vram_no":                 func(Host) string { return "Too little VRAM" },
}

// summary is the line for the worst check, else "Fits" or the pull time.
func summary(checks []imagecheck.Check, h Host, pullMin *float64) string {
	for _, level := range []string{imagecheck.LevelBlock, imagecheck.LevelWarn} {
		for _, c := range checks {
			if f, ok := summaries[c.Code]; ok && c.Level == level {
				return f(h)
			}
		}
	}
	if pullMin != nil {
		return fmt.Sprintf("≈ %s to pull", mins(*pullMin))
	}
	return "Fits"
}

// FallbackImage is the default image as the checks see it when its
// registry cannot be read (not published yet, private, unreachable):
// FallbackRequirements and today's measured sizes (image peak 20.3 GB), so
// hosts are still judged (MASTER §3.10.3). Other repos have no fallback.
func FallbackImage(ref string) (*imagecheck.Image, bool) {
	req, ok := imagecheck.FallbackRequirements(ref)
	if !ok {
		return nil, false
	}
	return &imagecheck.Image{
		Ref: ref, Repo: DefaultRepo, Platform: req.Platform, Platforms: []string{req.Platform},
		DownloadBytes: defaultDownloadBytes, UnpackedBytes: defaultUnpackedBytes, UnpackedSource: "estimate",
		Labels: map[string]string{}, Env: map[string]string{}, ExposedPorts: []string{fmt.Sprintf("%d/tcp", imagecheck.ServerPort)},
		GoServer: true, Requires: req,
	}, true
}
