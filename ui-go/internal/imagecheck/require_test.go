package imagecheck

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
)

var compatBrands = []string{"unknown", "grid", "tesla", "nvidia", "quadro", "quadrortx", "nvidiartx", "vapps", "vpc", "vcs", "vws", "cloudgaming"}

// fixtureEnv is NVIDIA_REQUIRE_CUDA from a saved real config.
func fixtureEnv(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("registrymock/testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Config struct{ Env []string } `json:"config"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	for _, kv := range cfg.Config.Env {
		if v, ok := strings.CutPrefix(kv, "NVIDIA_REQUIRE_CUDA="); ok {
			return v
		}
	}
	t.Fatalf("%s has no NVIDIA_REQUIRE_CUDA", name)
	return ""
}

func TestParseRequireCUDA(t *testing.T) {
	cases := []struct {
		file     string
		min      Version
		branches []int
	}{
		{"geocine_latest_config.json", Version{13, 0}, []int{535, 550, 565, 570, 575}},
		{"geocine_0.11.1_config.json", Version{12, 8}, []int{470, 535, 550, 560, 565}},
	}
	for _, c := range cases {
		min, ex, err := ParseRequireCUDA(fixtureEnv(t, c.file))
		if err != nil || min != c.min {
			t.Fatalf("%s: %v %v", c.file, min, err)
		}
		var want []BrandRange
		for _, b := range c.branches {
			for _, brand := range compatBrands {
				want = append(want, BrandRange{brand, b})
			}
		}
		if !slices.Equal(ex, want) {
			t.Errorf("%s: exceptions\n got %v\nwant %v", c.file, ex, want)
		}
		for _, e := range ex {
			if e.Brand == "geforce" || e.Brand == "geforcertx" || e.Brand == "titan" {
				t.Errorf("%s: %s is not a gate exception", c.file, e.Brand)
			}
		}
	}
	if _, _, err := ParseRequireCUDA(""); err == nil {
		t.Error("empty env parsed")
	}
	if _, _, err := ParseRequireCUDA("brand=tesla,driver>=535,driver<536"); err == nil {
		t.Error("env without cuda>= parsed")
	}
}

func TestDerive(t *testing.T) {
	cases := []struct {
		name string
		img  Image
		want Requirements
	}{
		{"labels win", Image{Platform: "linux/amd64", Labels: registrymock.GoLabels(), Env: map[string]string{}},
			Requirements{CUDAMin: "13.0", DriverMin: 580, SM: []int{75, 80, 86, 90, 100, 120}, FlashAttnSM: []int{80, 90, 100, 120}, API: 1, Port: 8675, Flavor: "cu130", Torch: "2.13.0+cu130"}},
		{"env only (upstream)", Image{Env: map[string]string{"NVIDIA_REQUIRE_CUDA": "cuda>=13.0 brand=tesla,driver>=535,driver<536", "CUDA_VERSION": "13.0.3"}, ExposedPorts: []string{"8675/tcp"}},
			Requirements{CUDAMin: "13.0", DriverMin: 580, SM: []int{75, 80, 86, 90, 100, 120}, FlashAttnSM: []int{}, API: 0, Port: 8675, Flavor: "cu130"}},
		{"CUDA_VERSION only", Image{Env: map[string]string{"CUDA_VERSION": "12.6.3"}},
			Requirements{CUDAMin: "12.6", DriverMin: 560, SM: []int{50, 60, 70, 75, 80, 86, 90}, FlashAttnSM: []int{}, Flavor: "cu126"}},
		{"a Go image from history", Image{GoServer: true, Env: map[string]string{"NVIDIA_REQUIRE_CUDA": "cuda>=13.0", "CUDA_VERSION": "13.0.3"}},
			Requirements{CUDAMin: "13.0", DriverMin: 580, SM: []int{75, 80, 86, 90, 100, 120}, FlashAttnSM: []int{80, 90, 100, 120}, API: 1, Flavor: "cu130"}},
		{"no CUDA (alpine)", Image{Env: map[string]string{}},
			Requirements{SM: []int{}, FlashAttnSM: []int{}}},
		{"flash-attn label present but empty", Image{Labels: map[string]string{LabelServer: "go", LabelSM: "80,90", LabelFlashAttnSM: "", LabelCUDA: "13.1"}},
			Requirements{CUDAMin: "13.1", DriverMin: 590, SM: []int{80, 90}, FlashAttnSM: []int{}, API: 1}},
	}
	for _, c := range cases {
		got := Derive(&c.img)
		if got.CUDAMin != c.want.CUDAMin || got.DriverMin != c.want.DriverMin || !slices.Equal(got.SM, c.want.SM) ||
			!slices.Equal(got.FlashAttnSM, c.want.FlashAttnSM) || got.API != c.want.API || got.Port != c.want.Port ||
			got.Flavor != c.want.Flavor || got.Torch != c.want.Torch {
			t.Errorf("%s:\n got %+v\nwant %+v", c.name, got, c.want)
		}
		if got.CompatBranches == nil || got.PTX == nil || got.Sources == nil {
			t.Errorf("%s: nil slices or maps encode as null", c.name)
		}
	}
}

func TestRunsOn(t *testing.T) {
	torch := []int{75, 80, 86, 90, 100, 120}
	flash := []int{80, 90, 100, 120}
	cases := []struct {
		sm   int
		list []int
		want bool
	}{
		{89, torch, true}, {70, torch, false}, {103, torch, true}, {121, torch, true}, {120, torch, true}, {61, torch, false}, {0, torch, false},
		{75, flash, false}, {86, flash, true}, {89, flash, true},
	}
	for _, c := range cases {
		if got := RunsOn(c.sm, c.list); got != c.want {
			t.Errorf("RunsOn(%d, %v) = %v", c.sm, c.list, got)
		}
	}
	if !(Requirements{PTX: []int{80}}).Runs(120) || (Requirements{PTX: []int{80}}).Runs(75) {
		t.Error("PTX targets run on later GPUs only")
	}
}

func TestFallbackRequirements(t *testing.T) {
	for _, repo := range []string{DefaultRepo, DefaultRepo + ":latest", DefaultRepo + ":f53a407"} {
		r, ok := FallbackRequirements(repo)
		if !ok || r.Source != "fallback" || r.CUDAMin != "13.0" || r.Platform != "linux/amd64" ||
			!slices.Equal(r.SM, []int{75, 80, 86, 90, 100, 120}) || r.DriverMin != 580 || r.API != 1 {
			t.Fatalf("%s: %+v %v", repo, r, ok)
		}
	}
	for _, repo := range []string{"ghcr.io/geocine/aitoolkit", "ostris/aitoolkit", "quay.io/x/y", "not valid"} {
		if _, ok := FallbackRequirements(repo); ok {
			t.Errorf("%s has a fallback", repo)
		}
	}
	// A derived image has no Source.
	if r := Derive(&Image{Labels: registrymock.GoLabels()}); r.Source != "" {
		t.Errorf("derived source %q", r.Source)
	}
}

func TestClassifyFailure(t *testing.T) {
	cases := []struct {
		text, code, problem string
	}{
		{"Error response from daemon: manifest unknown", "image_not_found", ProblemImagePull},
		{"manifest for ghcr.io/geocine/aitoolkit-go:nope not found", "image_not_found", ProblemImagePull},
		{"pull access denied for x, repository does not exist or may require 'docker login'", "image_denied", ProblemImagePull},
		{"failed to authorize: 401 Unauthorized", "image_denied", ProblemImagePull},
		{"Authentication required", "image_denied", ProblemImagePull},
		{"no matching manifest for linux/arm64/v8 in the manifest list entries", "image_platform", ProblemImagePull},
		{"exec /start.sh: exec format error", "image_platform", ProblemImagePull},
		{"write /var/lib/docker/tmp/x: no space left on device", "disk_full", ProblemImagePull},
		{"nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0, please update your driver", "driver_too_old", ProblemCUDA},
		{"CUDA driver version is insufficient for CUDA runtime version", "driver_too_old", ProblemCUDA},
		{"The NVIDIA driver on your system is too old (found version 12080)", "driver_too_old", ProblemCUDA},
		{"RuntimeError: CUDA error: no kernel image is available for execution on the device", "no_kernel_image", ProblemCUDA},
		{"NVIDIA H20 with CUDA capability sm_90 is not compatible with the current PyTorch installation.", "no_kernel_image", ProblemCUDA},
	}
	for _, c := range cases {
		f, ok := ClassifyFailure(c.text)
		if !ok || f.Code != c.code || f.Problem != c.problem || f.Message == "" {
			t.Errorf("%q -> %+v %v, want %s/%s", c.text, f, ok, c.code, c.problem)
		}
	}
	platform := map[string]string{
		"no matching manifest for linux/amd64 in the manifest list entries":    "The image isn't built for amd64.",
		"no matching manifest for linux/arm64 in the manifest list entries":    "The host isn't amd64.",
		"no matching manifest for linux/arm64/v8 in the manifest list entries": "The host isn't amd64.",
		"exec /start.sh: exec format error":                                    "The host isn't amd64.",
	}
	for text, msg := range platform {
		if f, _ := ClassifyFailure(text); f.Code != "image_platform" || f.Problem != ProblemImagePull || f.Message != msg {
			t.Errorf("%q -> %+v, want image_platform %q", text, f, msg)
		}
	}
	if f, ok := ClassifyFailure("container started"); ok {
		t.Errorf("benign text classified as %+v", f)
	}
	f, _ := ClassifyFailure("no space left on device")
	if got := f.WithDisk(80).Message; got != "The disk filled up while pulling. Use at least 80 GB." {
		t.Errorf("disk message %q", got)
	}
}

func TestDeniedCopy(t *testing.T) {
	msg, fix := DeniedCopy(DefaultRepo + ":latest")
	if !strings.Contains(msg, "isn't published yet") || !strings.Contains(fix, "build_and_push.sh --push") {
		t.Fatalf("default repo copy %q / %q", msg, fix)
	}
	msg, _ = DeniedCopy("ghcr.io/other/img")
	if msg != "The image ghcr.io/other/img wasn't found, or it's private." {
		t.Fatalf("other repo copy %q", msg)
	}
}
