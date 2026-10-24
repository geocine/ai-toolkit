package hostfit

import (
	"testing"

	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
)

func TestPullFor(t *testing.T) {
	img := fallbackImage(t)
	p := PullFor(img, Host{InetDownMbps: 1376, DiskBWMBps: 2842}, 0)
	if p == nil || p.ImageMin != 2.8 || p.ModelMin != nil || p.Basis != "1376 Mbps, 2842 MB/s disk" || !p.Estimate {
		t.Fatalf("fast host: %+v", p)
	}
	p = PullFor(img, Host{InetDownMbps: 73}, 33.7)
	if p == nil || p.ImageMin < 24.5 || p.ImageMin > 26 || p.Basis != "73 Mbps" || p.ModelMin == nil || *p.ModelMin < 120 {
		t.Fatalf("slow host: %+v", p)
	}
	if PullFor(img, Host{}, 30) != nil {
		t.Error("no bandwidth, but a pull estimate")
	}
	// An unread image uses today's sizes.
	if a, b := PullFor(nil, Host{InetDownMbps: 1376, DiskBWMBps: 2842}, 0), PullFor(img, Host{InetDownMbps: 1376, DiskBWMBps: 2842}, 0); a.ImageMin != b.ImageMin {
		t.Errorf("nil image %v, fallback %v", a.ImageMin, b.ImageMin)
	}
}

// The disk need follows IMAGE-PLAN §3.5 with today's image (20.3 GB peak);
// without a model or with an unknown one it is at least 100 GB.
func TestNeedFor(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	img := goImage(t, c)
	for arch, want := range map[string]int{"flux": 80, "qwen_image": 110, "wan22_14b": 180, "flux2": 170} {
		n := NeedFor(img, &ModelRef{Arch: arch}, nil)
		if n.DiskGB != want || n.ModelSource != "table" || n.ImagePeakGB != 20.3 || n.WorkGB != 15 {
			t.Errorf("%s: %+v, want %d GB", arch, n, want)
		}
	}
	for _, m := range []*ModelRef{nil, {Arch: "mystery"}, {Arch: "hidream"}, {Arch: "ltx2"}} {
		n := NeedFor(img, m, nil)
		if n.DiskGB < MinDiskGBUnknownModel || n.ModelSource != "default" {
			t.Errorf("%+v: %+v", m, n)
		}
	}
	if n := NeedFor(nil, nil, nil); n.ImagePeakGB != DefaultImagePeakGB || n.DiskGB != 100 {
		t.Errorf("unread image: %+v", n)
	}
	// A 50 GB host is too small for the no-model floor (a warn), and a 16 GB
	// one cannot even unpack the image (a block).
	n := NeedFor(img, nil, nil)
	if c := diskChecks(n, 50, 0, nil); len(c) != 1 || c[0].Code != "disk_model" ||
		c[0].Message != "50 GB fits the image but leaves little room for a base model and checkpoints." {
		t.Errorf("50 GB host: %+v", c)
	}
	if c := diskChecks(n, 16, 0, nil); len(c) != 1 || c[0].Code != "disk_image" {
		t.Errorf("16 GB host: %+v", c)
	}
	if c := diskChecks(n, 0, 0, nil); len(c) != 0 {
		t.Errorf("RunPod (disk chosen at launch): %+v", c)
	}
	if c := diskChecks(n, 200, 100, nil); len(c) != 0 {
		t.Errorf("enough disk: %+v", c)
	}
}

func TestVRAMFor(t *testing.T) {
	cases := []struct {
		m       ModelRef
		vram    float64
		verdict string
		source  string
	}{
		{ModelRef{Arch: "flux"}, 24, "fits", "repo"},
		{ModelRef{Arch: "flux"}, 22.49, "fits", "repo"}, // L4 23034 MiB
		{ModelRef{Arch: "flux"}, 16, "no", "estimate"},
		{ModelRef{Arch: "flux", NameOrPath: "black-forest-labs/FLUX.1-dev"}, 24, "tight", "repo"}, // quantize turned off
		{ModelRef{Arch: "flux", NameOrPath: "black-forest-labs/FLUX.1-dev", Quantize: true}, 24, "fits", "repo"},
		{ModelRef{Arch: "flux_kontext"}, 24, "tight", "repo"},
		{ModelRef{Arch: "flux2"}, 32, "no", "estimate"},
		{ModelRef{Arch: "flux2"}, 48, "tight", "estimate"},
		{ModelRef{Arch: "flux2"}, 80, "fits", "estimate"},
		{ModelRef{Arch: "qwen_image"}, 24, "tight", "repo"},
		{ModelRef{Arch: "qwen_image", QType: "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors", LowVRAM: true}, 24, "fits", "repo"},
		{ModelRef{Arch: "qwen_image", QType: "qfloat8", LowVRAM: true}, 24, "tight", "repo"},
		{ModelRef{Arch: "qwen_image_edit"}, 24, "no", "estimate"},
		{ModelRef{Arch: "qwen_image_edit"}, 32, "fits", "repo"},
		{ModelRef{Arch: "qwen_image_edit_plus"}, 32, "fits", "repo"},
		{ModelRef{Arch: "wan22_14b"}, 24, "tight", "repo"},
		{ModelRef{Arch: "wan22_14b", QType: "uint4", LowVRAM: true}, 24, "fits", "repo"},
		{ModelRef{Arch: "hidream"}, 24, "no", "repo"},
		{ModelRef{Arch: "hidream"}, 48, "fits", "repo"},
		{ModelRef{Arch: "hidream"}, 47.99, "fits", "repo"}, // a modded 48 GB 4090 lands on the 48 tier
		{ModelRef{Arch: "ltx2"}, 32, "tight", "vendor"},
		{ModelRef{Arch: "ltx2"}, 80, "fits", "vendor"},
		{ModelRef{Arch: "zimage"}, 16, "fits", "estimate"},
		{ModelRef{Arch: "zimage", Resolution: 1024}, 16, "tight", "estimate"},
		{ModelRef{Arch: "flux"}, 141, "fits", "estimate"},
		{ModelRef{Arch: "flux"}, 12, "no", "estimate"},
		{ModelRef{Arch: "mystery"}, 24, "unknown", ""},
	}
	for _, c := range cases {
		got := VRAMFor(c.m, c.vram)
		if got.Verdict != c.verdict || got.Source != c.source || got.Note == "" {
			t.Errorf("%+v at %v GB: %+v; want %s/%s", c.m, c.vram, got, c.verdict, c.source)
		}
		if got.Verdict == "tight" && got.Recipe == "" {
			t.Errorf("%+v at %v GB: tight without a recipe", c.m, c.vram)
		}
	}
	if n := VRAMFor(ModelRef{Arch: "flux"}, 24).Note; n != "FLUX.1 fits in 24 GB; with 8-bit quantization and gradient checkpointing (train_lora_flux_24gb.yaml)." {
		t.Errorf("note: %q", n)
	}
}

func TestTiersAndMinVRAM(t *testing.T) {
	tiers := Tiers(ModelRef{Arch: "flux2"})
	if len(tiers) != 5 || tiers[2].VRAMGB != 32 || tiers[2].Verdict != "no" || tiers[3].Verdict != "tight" {
		t.Fatalf("flux2 tiers: %+v", tiers)
	}
	for arch, want := range map[string]float64{"flux": 24 * 0.9, "flux2": 48 * 0.9, "qwen_image_edit": 32 * 0.9, "hidream": 48 * 0.9, "zimage": 16 * 0.9, "mystery": 0} {
		if got := MinVRAMGB(&ModelRef{Arch: arch}); got != want {
			t.Errorf("MinVRAMGB(%s) = %v, want %v", arch, got, want)
		}
	}
	if MinVRAMGB(nil) != 0 {
		t.Error("MinVRAMGB(nil)")
	}
}
