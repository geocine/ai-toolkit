package hostfit

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/imagecheck"
)

var update = flag.Bool("update", false, "rewrite the golden files in testdata/golden")

// golden compares v's indented JSON with testdata/golden/name.
func golden(t *testing.T, name string, v any) {
	t.Helper()
	got, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	path := "testdata/golden/" + name
	if *update {
		if err := os.MkdirAll("testdata/golden", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run with -update to create it)", err)
	}
	if !bytes.Equal(bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n")), got) {
		t.Errorf("%s differs:\n%s", name, got)
	}
}

func TestFilterFor(t *testing.T) {
	req := goRequirements()
	need := NeedFor(nil, &ModelRef{Arch: "flux"}, nil)
	f := FilterFor(req, need, &ModelRef{Arch: "flux"}, Policy{})
	if f.CUDAMin != (Version{13, 0}) || f.Platform != "linux/amd64" || f.Vendor != "nvidia" || f.MinDiskGB != 80 || f.MinVRAMGB != 21.6 {
		t.Errorf("filter: %+v", f)
	}
	if !slices.Equal(f.SMAllowed, []int{80, 86, 89, 90, 100, 103, 120}) {
		t.Errorf("SMAllowed %v", f.SMAllowed)
	}
	if f.MinInetDownMbps != 0 || f.MinDiskBWMBps != 0 {
		t.Errorf("bandwidth floors without Fast: %+v", f)
	}
	turing := FilterFor(req, need, nil, Policy{AllowTuring: true, Fast: true})
	if !slices.Contains(turing.SMAllowed, 75) || turing.MinInetDownMbps != FastInetDownMbps || turing.MinDiskBWMBps != FastDiskBWMBps || turing.MinVRAMGB != 0 {
		t.Errorf("turing/fast filter: %+v", turing)
	}
	custom := req
	custom.SM, custom.PTX = nil, nil
	if f := FilterFor(custom, need, nil, Policy{}); f.SMAllowed != nil {
		t.Errorf("no SM info, but SMAllowed %v", f.SMAllowed)
	}
}

// The Vast bodies: cuda_max_good is the only driver gate, compute_cap
// excludes 750 unless AllowTuring and never holds 1210 (GB10, arm64).
func TestVastQuery(t *testing.T) {
	req := goRequirements()
	base := map[string]any{"verified": map[string]any{"eq": true}, "rentable": map[string]any{"eq": true},
		"num_gpus": map[string]any{"eq": 1}, "gpu_ram": map[string]any{"gte": 16384, "lt": 49152},
		"driver_version": map[string]any{"gte": "580"}, "order": [][]string{{"dph_total", "asc"}}, "limit": 50}
	noModel := NeedFor(nil, nil, nil)
	flux := &ModelRef{Arch: "flux"}
	fluxNeed := NeedFor(nil, flux, nil)
	cases := map[string]Filter{
		"vast_no_model.json": FilterFor(req, noModel, nil, Policy{}),
		"vast_flux.json":     FilterFor(req, fluxNeed, flux, Policy{}),
		"vast_fast.json":     FilterFor(req, fluxNeed, flux, Policy{Fast: true}),
		"vast_show_all.json": FilterFor(req, fluxNeed, flux, Policy{ShowAll: true}),
		"vast_turing.json":   FilterFor(req, noModel, nil, Policy{AllowTuring: true}),
	}
	for name, f := range cases {
		q := VastQuery(f, base)
		golden(t, name, q)
		b, _ := json.Marshal(q)
		if strings.Contains(string(b), "driver_vers") || strings.Contains(string(b), "driver_version") {
			t.Errorf("%s sends a driver key", name)
		}
		if strings.Contains(string(b), "1210") {
			t.Errorf("%s allows sm_121", name)
		}
		if strings.Contains(string(b), "750") != (name == "vast_turing.json") {
			t.Errorf("%s: compute_cap 750 present = %v", name, strings.Contains(string(b), "750"))
		}
	}
	if _, ok := base["gpu_arch"]; ok {
		t.Error("VastQuery modified its base")
	}
	// The stricter bound wins over the adapter's own.
	q := VastQuery(FilterFor(req, fluxNeed, flux, Policy{}), base)
	if ram := q["gpu_ram"].(map[string]any); ram["gte"] != 22118 || ram["lt"] != 49152 {
		t.Errorf("gpu_ram merge: %v", ram)
	}
	if q := VastQuery(FilterFor(req, noModel, nil, Policy{}), map[string]any{"gpu_ram": map[string]any{"gte": 40960}}); q["gpu_ram"].(map[string]any)["gte"] != 40960 {
		t.Errorf("a stricter base gpu_ram lost: %v", q["gpu_ram"])
	}
}

func TestRunPodFilter(t *testing.T) {
	f := FilterFor(goRequirements(), NeedFor(nil, nil, nil), nil, Policy{})
	if got := RunPodGPUConfig(f); got != "13.0" {
		t.Errorf("minCudaVersion %q", got)
	}
	if RunPodGPUConfig(Filter{}) != "" {
		t.Error("no floor, but a minCudaVersion")
	}
	for id, want := range map[string]string{
		"NVIDIA GeForce RTX 4090": "", "NVIDIA B300 SXM6 AC": "", "NVIDIA RTX PRO 6000 Blackwell Server Edition": "",
		"Tesla V100-SXM2-16GB": "gpu_sm", "AMD Instinct MI300X OAM": "gpu_vendor", "NVIDIA GH200 480GB": "host_arch",
		"Tesla T4": "gpu_sm_flash_attn", "Mystery GPU": "",
	} {
		ok, why := RunPodAllowedType(f, id)
		if ok != (want == "") || why != want {
			t.Errorf("%s: %v %q, want %q", id, ok, why, want)
		}
	}
}

func TestQuickPodKeep(t *testing.T) {
	req := goRequirements()
	f := FilterFor(req, NeedFor(nil, nil, nil), nil, Policy{})
	for _, row := range quickPodOffers(t) {
		h := FromQuickPod(row)
		keep, level := QuickPodKeep(f, h)
		// sm_75: the image runs it, the policy hides it (D-8): Fit's warn.
		if strings.Contains(h.GPUName, "1660") && (keep || level != "warn") {
			t.Errorf("GTX 1660: %v %s", keep, level)
		}
		if h.DiskGB == 30 && (keep || level != "warn") {
			t.Errorf("30 GB row kept: %v %s", keep, level)
		}
		if h.CUDAMax != "" && h.SM >= 80 && h.DiskGB >= 100 && (!keep || level != "ok") {
			t.Errorf("%s dropped: %v %s", h.GPUName, keep, level)
		}
	}
	// Unknown CUDA keeps the host with a warn; a CUDA below the floor drops it.
	unknown := Host{GPUName: "NVIDIA GeForce RTX 3090", SM: 86, GPUVendor: "nvidia", CPUArch: "amd64", DiskGB: 500}
	if keep, level := Keeps(f, unknown); !keep || level != "warn" {
		t.Errorf("unknown CUDA: %v %s", keep, level)
	}
	old := unknown
	old.CUDAMax = "12.8"
	if keep, level := Keeps(f, old); keep || level != "block" {
		t.Errorf("CUDA 12.8: %v %s", keep, level)
	}
	// Under ShowAll hosts the image cannot run are kept; level is the badge.
	all := f
	all.ShowAll = true
	if keep, level := Keeps(all, old); !keep || level != "block" {
		t.Errorf("show all: %v %s", keep, level)
	}

	// The level matches Fit: a T4 is the flash-attention warn, a disk
	// below the image peak blocks, one below the need warns, and only the
	// disk floor survives ShowAll.
	good := old
	good.CUDAMax = "13.2"
	t4 := good
	t4.GPUName, t4.SM = "Tesla T4", 75
	if keep, level := Keeps(f, t4); keep || level != "warn" {
		t.Errorf("T4: %v %s", keep, level)
	}
	if keep, level := Keeps(all, t4); !keep || level != "warn" {
		t.Errorf("T4 under show all: %v %s", keep, level)
	}
	if checks := Fit(fallbackImage(t), t4, FitOptions{}); imagecheck.Worst(checks) != "warn" {
		t.Errorf("Fit T4: %v", codesOf(checks))
	}
	tiny := good
	tiny.DiskGB = 16
	for _, filter := range []Filter{f, all} {
		if keep, level := Keeps(filter, tiny); keep || level != "block" {
			t.Errorf("16 GB disk (show all %v): %v %s", filter.ShowAll, keep, level)
		}
	}
	small := good
	small.DiskGB = 50
	if keep, level := Keeps(all, small); keep || level != "warn" {
		t.Errorf("50 GB disk under show all: %v %s", keep, level)
	}
	// VRAM below the floor is dropped, and kept under ShowAll (vram_no is a warn).
	fluxF := FilterFor(req, NeedFor(nil, &ModelRef{Arch: "flux"}, nil), &ModelRef{Arch: "flux"}, Policy{})
	card := good
	card.VRAMGB = 12
	if keep, level := Keeps(fluxF, card); keep || level != "warn" {
		t.Errorf("12 GB card: %v %s", keep, level)
	}
	fluxF.ShowAll = true
	if keep, _ := Keeps(fluxF, card); !keep {
		t.Error("12 GB card dropped under show all")
	}
	// RunPod: a catalog with no free version at the floor is the capacity warn.
	busy := FromRunPodGPU(RunPodGPUType{ID: "NVIDIA GeForce RTX 4090", Memory: 24,
		CUDAVersions: []RunPodCUDAVersion{{"13.0", false}, {"13.2", false}}}, "13.0")
	if keep, level := Keeps(f, busy); !keep || level != "warn" {
		t.Errorf("RunPod no capacity: %v %s", keep, level)
	}
	// Fast: a slow reported link drops; an unreported one passes (D-15).
	fast := FilterFor(req, NeedFor(nil, nil, nil), nil, Policy{Fast: true})
	fine := unknown
	fine.CUDAMax, fine.InetDownMbps, fine.DiskBWMBps = "13.2", 73, 0
	if keep, _ := Keeps(fast, fine); keep {
		t.Error("73 Mbps kept under Fast")
	}
	fine.InetDownMbps = 0
	if keep, level := Keeps(fast, fine); !keep || level != "ok" {
		t.Errorf("unreported bandwidth: %v %s", keep, level)
	}
}
