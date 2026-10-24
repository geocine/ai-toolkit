package hostfit

import (
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
)

// On the 61 public Vast offers exactly the 11 below CUDA 13.0 block on the
// driver; the synthetic V100 blocks on its GPU although its driver is new.
func TestFitVastFixture(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	img := goImage(t, c)
	offers := vastOffers(t)
	cuda := 0
	for _, o := range offers[:61] {
		checks := Fit(img, FromVast(o), FitOptions{})
		if hasCode(checks, "host_cuda") {
			cuda++
			if imagecheck.Worst(checks) != imagecheck.LevelBlock {
				t.Errorf("%v: host_cuda is not a block", o["id"])
			}
		}
		if hasCode(checks, "gpu_sm") || hasCode(checks, "host_arch") || hasCode(checks, "gpu_unknown") {
			t.Errorf("%v: unexpected %v", o["id"], codesOf(checks))
		}
	}
	if cuda != 11 {
		t.Errorf("host_cuda on %d offers, want 11", cuda)
	}

	v100 := Fit(img, FromVast(vastRow(t, 90000003)), FitOptions{})
	if c := checkOf(t, v100, "gpu_sm"); c.Level != imagecheck.LevelBlock || !strings.Contains(c.Message, "Tesla V100 (sm_70)") {
		t.Errorf("V100: %+v", c)
	}
	if hasCode(v100, "host_cuda") {
		t.Error("V100 with CUDA 13.0 blocked on the driver")
	}
	if c := checkOf(t, Fit(img, FromVast(vastRow(t, 90000004)), FitOptions{}), "gpu_sm"); !strings.Contains(c.Message, "Quadro P4000 (sm_61)") {
		t.Errorf("junk compute_cap: %+v", c)
	}
	if c := checkOf(t, Fit(img, FromVast(vastRow(t, 90000005)), FitOptions{}), "host_arch"); !strings.Contains(c.Message, "arm64 (GH200)") {
		t.Errorf("GH200: %+v", c)
	}

	old := Fit(img, FromVast(vastRow(t, 38849861)), FitOptions{}) // RTX 3090, CUDA 12.8
	if c := checkOf(t, old, "host_cuda"); c.Message != "This host's driver supports CUDA 12.8. The image needs 13.0 (driver 580 or newer)." ||
		c.Fix != "Pick a host with a newer driver." || c.Action == nil || c.Action.Kind != "filter_hosts" {
		t.Errorf("host_cuda copy: %+v", c)
	}
}

// A data-center GPU on a branch the container gate excepts keeps the block
// but says forward compatibility might start it.
func TestFitCompatBranch(t *testing.T) {
	img := fallbackImage(t)
	img.Requires.CompatBranches = []imagecheck.BrandRange{{Brand: "tesla", Branch: 570}}
	h := Host{GPUName: "NVIDIA A100-SXM4-80GB", SM: 80, GPUClass: "datacenter", GPUVendor: "nvidia", CPUArch: "amd64", CUDAMax: "12.8", Driver: "570.211.01"}
	c := checkOf(t, Fit(img, h, FitOptions{}), "host_cuda")
	if c.Level != imagecheck.LevelBlock || c.Message != "This A100 SXM4 80GB's driver (570) supports CUDA 12.8. The image needs 13.0." ||
		!strings.Contains(c.Fix, "Forward compatibility") {
		t.Errorf("compat: %+v", c)
	}
	// A GeForce on the same branch gets the plain copy.
	h.GPUName, h.GPUClass = "RTX 4090", "geforce"
	if c := checkOf(t, Fit(img, h, FitOptions{}), "host_cuda"); strings.Contains(c.Fix, "Forward") {
		t.Errorf("geforce compat: %+v", c)
	}
}

func TestFitGPUArchitectures(t *testing.T) {
	img := fallbackImage(t)
	fit := func(name string) []imagecheck.Check {
		h := Host{GPUName: name, CUDAMax: "13.2", CPUArch: "amd64"}
		h.fillFromTable()
		return Fit(img, h, FitOptions{})
	}
	if c := checkOf(t, fit("Tesla T4"), "gpu_sm_flash_attn"); c.Level != imagecheck.LevelWarn || !strings.Contains(c.Message, "T4 (sm_75)") ||
		!strings.Contains(c.Message, "no native bf16") {
		t.Errorf("T4: %+v", c)
	}
	if c := checkOf(t, fit("NVIDIA B300 SXM6 AC"), "gpu_sm_unverified"); c.Level != imagecheck.LevelInfo || c.Message != "The B300 runs this image's sm_100 build." {
		t.Errorf("B300: %+v", c)
	}
	if c := checkOf(t, fit("AMD Instinct MI300X OAM"), "gpu_vendor"); c.Level != imagecheck.LevelBlock {
		t.Errorf("MI300X: %+v", c)
	}
	if c := checkOf(t, fit("Mystery Accelerator"), "gpu_unknown"); c.Level != imagecheck.LevelInfo {
		t.Errorf("unknown GPU: %+v", c)
	}
	for _, ok := range []string{"RTX 4090", "NVIDIA GeForce RTX 5090", "NVIDIA L40S", "NVIDIA A100 80GB PCIe", "NVIDIA H100 NVL", "NVIDIA B200", "RTX 3090"} {
		if checks := fit(ok); imagecheck.Worst(checks) == imagecheck.LevelBlock || hasCode(checks, "gpu_sm_flash_attn") {
			t.Errorf("%s: %v", ok, codesOf(checks))
		}
	}
	// No SM information in the image: no SM checks at all.
	custom := *img
	custom.Requires.SM, custom.Requires.PTX = nil, nil
	if checks := Fit(&custom, Host{GPUName: "Tesla V100", SM: 70, CUDAMax: "13.0"}, FitOptions{}); hasCode(checks, "gpu_sm") {
		t.Errorf("custom image without SM: %v", codesOf(checks))
	}
}

func TestFitCUDARules(t *testing.T) {
	img := fallbackImage(t)
	base := Host{GPUName: "RTX 4090", SM: 89, GPUVendor: "nvidia", CPUArch: "amd64"}
	unknown := Fit(img, base, FitOptions{})
	if c := checkOf(t, unknown, "host_cuda_unknown"); c.Level != imagecheck.LevelWarn || c.Fix != "If it's older than 580, the server won't start." {
		t.Errorf("unknown: %+v", c)
	}
	placed := base
	placed.PlacementFloor = true
	if c := checkOf(t, Fit(img, placed, FitOptions{}), "host_cuda_placement"); c.Level != imagecheck.LevelInfo ||
		c.Message != "RunPod only places this on hosts with CUDA 13.0 or newer." {
		t.Errorf("placement: %+v", c)
	}
	// The catalog lists CUDA versions but none is free: capacity, not placement.
	busy := FromRunPodGPU(RunPodGPUType{ID: "NVIDIA GeForce RTX 4090", Memory: 24,
		CUDAVersions: []RunPodCUDAVersion{{"13.0", false}, {"13.2", false}}}, "13.0")
	if c := checkOf(t, Fit(img, busy, FitOptions{}), "runpod_no_cuda_capacity"); c.Level != imagecheck.LevelWarn ||
		c.Message != "No RTX 4090 hosts with CUDA 13.0+ are free right now." {
		t.Errorf("all unavailable: %+v", c)
	}
	placed.CUDAMax = "12.8"
	checks := Fit(img, placed, FitOptions{})
	if c := checkOf(t, checks, "runpod_no_cuda_capacity"); c.Level != imagecheck.LevelWarn || c.Message != "No RTX 4090 hosts with CUDA 13.0+ are free right now." {
		t.Errorf("capacity: %+v", c)
	}
	if hasCode(checks, "host_cuda") {
		t.Error("RunPod capacity blocked as host_cuda")
	}
	fine := base
	fine.CUDAMax = "13.0"
	if checks := Fit(img, fine, FitOptions{}); len(checks) != 0 {
		t.Errorf("CUDA 13.0 host: %v", codesOf(checks))
	}
}

func TestFitDiskPullVRAM(t *testing.T) {
	img := fallbackImage(t)
	flux := &ModelRef{Arch: "flux"}
	h := Host{GPUName: "RTX 4090", SM: 89, GPUVendor: "nvidia", CPUArch: "amd64", CUDAMax: "13.2", VRAMGB: 23.99,
		DiskGB: 16, InetDownMbps: 73, RAMGB: 16}
	checks := Fit(img, h, FitOptions{Model: flux})
	if c := checkOf(t, checks, "disk_image"); c.Level != imagecheck.LevelBlock ||
		c.Message != "20 GB is needed just to download and unpack the image; this host has 16 GB." || c.Fix != "Pick a host with at least 80 GB." {
		t.Errorf("disk_image: %+v", c)
	}
	if c := checkOf(t, checks, "pull_slow"); c.Level != imagecheck.LevelWarn ||
		c.Message != "Expect about 26 minutes to download the 6.6 GB image on this host (73 Mbps)." ||
		c.Fix != "Hosts above 500 Mbps take about 5 minutes." {
		t.Errorf("pull_slow: %+v", c)
	}

	h.DiskGB, h.InetDownMbps, h.DiskBWMBps = 50, 1376, 2842
	checks = Fit(img, h, FitOptions{Model: flux})
	if c := checkOf(t, checks, "disk_model"); c.Level != imagecheck.LevelWarn ||
		c.Message != "50 GB fits the image but not FLUX.1 (about 34 GB) plus checkpoints." || c.Fix != "Pick a host with at least 80 GB." {
		t.Errorf("disk_model: %+v", c)
	}
	if c := checkOf(t, checks, "pull_eta"); c.Message != "About 3 minutes to download the image, then about 7 minutes for FLUX.1." {
		t.Errorf("pull_eta: %+v", c)
	}
	// The launch asks for less than the host allows: set_disk.
	h.DiskGB = 500
	if c := checkOf(t, Fit(img, h, FitOptions{Model: flux, DiskGB: 50}), "disk_model"); c.Action == nil || c.Action.Kind != "set_disk" || c.Action.Value != 80 {
		t.Errorf("set_disk: %+v", c)
	}
	if c := checkOf(t, Fit(img, h, FitOptions{Model: flux, DiskGB: 20}), "disk_image"); c.Action == nil || c.Action.Kind != "set_disk" {
		t.Errorf("asked below the image: %+v", c)
	}

	flux2 := &ModelRef{Arch: "flux2", LowVRAM: true}
	checks = Fit(img, h, FitOptions{Model: flux2})
	if c := checkOf(t, checks, "vram_no"); c.Level != imagecheck.LevelWarn || c.Message != "FLUX.2 doesn't fit in 24 GB." || c.Fix != "Pick a 48 GB or larger GPU." {
		t.Errorf("vram_no: %+v", c)
	}
	if c := checkOf(t, checks, "ram_tight"); c.Message != "This host has 16 GB RAM; offloading FLUX.2 may swap." {
		t.Errorf("ram_tight: %+v", c)
	}
	if c := checkOf(t, Fit(img, h, FitOptions{Model: flux2, Launch: true}), "vram_no"); c.Level != imagecheck.LevelInfo {
		t.Errorf("vram_no at launch: %+v", c)
	}
	// Below the smallest tier the copy names the card's own size; the fix
	// takes "an" before 80.
	small := h
	small.VRAMGB = 12
	if c := checkOf(t, Fit(img, small, FitOptions{Model: flux}), "vram_no"); c.Message != "FLUX.1 doesn't fit in 12 GB." || c.Fix != "Pick a 24 GB or larger GPU." {
		t.Errorf("vram_no at 12 GB: %+v", c)
	}
	if c := checkOf(t, Fit(img, small, FitOptions{Model: &ModelRef{Arch: "hidream"}}), "vram_no"); c.Fix != "Pick a 48 GB or larger GPU." {
		t.Errorf("hidream: %+v", c)
	}
	for gb, want := range map[float64]string{80: "an", 8: "an", 11: "an", 48: "a", 16: "a", 24: "a", 100: "a"} {
		if got := article(gb); got != want {
			t.Errorf("article(%v) = %s", gb, got)
		}
	}
	if c := checkOf(t, Fit(img, h, FitOptions{Model: &ModelRef{Arch: "qwen_image"}}), "vram_tight"); c.Level != imagecheck.LevelInfo ||
		!strings.Contains(c.Message, "Qwen-Image fits in 24 GB only with 3-bit quantization") {
		t.Errorf("vram_tight: %+v", c)
	}
	// VRAM never blocks.
	if imagecheck.Worst(Fit(img, h, FitOptions{Model: &ModelRef{Arch: "hidream"}})) == imagecheck.LevelBlock {
		t.Error("VRAM blocked")
	}
}

func TestOfferFitFor(t *testing.T) {
	img := fallbackImage(t)
	need := NeedFor(img, &ModelRef{Arch: "flux"}, nil)
	fast := FromVast(vastRow(t, 38341566)) // 5090, 1376 Mbps, 612 GB
	fit := OfferFitFor(img, fast, need, VRAMFor(ModelRef{Arch: "flux"}, fast.VRAMGB))
	if fit.Level != "info" || fit.Summary != "≈ 3 min to pull" || fit.PullMin == nil || *fit.PullMin != 2.8 || fit.VRAM != "fits" {
		t.Errorf("fast 5090: %+v", fit)
	}
	old := OfferFitFor(img, FromVast(vastRow(t, 38849861)), need, VRAMFit{})
	if old.Level != "block" || old.Summary != "Driver too old (CUDA 12.8)" || old.VRAM != "unknown" || !strings.Contains(strings.Join(old.Codes, ","), "host_cuda") {
		t.Errorf("old driver: %+v", old)
	}
	slow := OfferFitFor(img, FromVast(vastRow(t, 90000007)), need, VRAMFit{Verdict: "fits"})
	if slow.Level != "warn" || slow.Summary != "Slow download" {
		t.Errorf("slow host: %+v", slow)
	}
	runpod := OfferFitFor(img, FromRunPodGPU(RunPodGPUType{ID: "NVIDIA GeForce RTX 4090", Memory: 24}, "13.0"), need, VRAMFit{Verdict: "fits"})
	if runpod.Level != "info" || runpod.Summary != "Fits" || runpod.PullMin != nil || runpod.Codes[0] != "host_cuda_placement" {
		t.Errorf("runpod: %+v", runpod)
	}
	nov := OfferFitFor(img, fast, need, VRAMFit{Verdict: "no"})
	if nov.Level != "warn" || nov.Summary != "Too little VRAM" {
		t.Errorf("vram no: %+v", nov)
	}
	// No image at all (another repo that could not be read): host-only checks.
	// Its size is unknown: no disk or pull checks, and no pull time.
	if f := OfferFitFor(nil, FromVast(vastRow(t, 38849861)), need, VRAMFit{}); f.Level == "block" || f.Codes == nil || f.PullMin != nil {
		t.Errorf("nil image: %+v", f)
	}
	tiny := fast
	tiny.DiskGB, tiny.InetDownMbps = 16, 73
	if checks := Fit(nil, tiny, FitOptions{Need: &need}); hasCode(checks, "disk_image") || hasCode(checks, "pull_slow") || hasCode(checks, "pull_eta") {
		t.Errorf("nil image disk/pull: %v", codesOf(checks))
	}
}

func TestFallbackImage(t *testing.T) {
	img := fallbackImage(t)
	if ImagePeakGB(img) != DefaultImagePeakGB || !img.GoServer || img.Requires.CUDAMin != "13.0" || img.Requires.Source != "fallback" {
		t.Errorf("fallback: %+v", img)
	}
	if _, ok := FallbackImage("ostris/aitoolkit:latest"); ok {
		t.Error("fallback for another repo")
	}
	// The measured fallback sizes match what the registry fixture reports.
	_, c := newMock(t, nil, registrymock.Faults{})
	if real := goImage(t, c); ImagePeakGB(real) != DefaultImagePeakGB {
		t.Errorf("fixture peak %.1f, fallback %.1f", ImagePeakGB(real), DefaultImagePeakGB)
	}
}
