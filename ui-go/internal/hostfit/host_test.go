package hostfit

import (
	"strings"
	"testing"
)

func TestParseVersionAndDriver(t *testing.T) {
	for in, want := range map[string]Version{"13.1\n": {13, 1}, "13.0": {13, 0}, " 12.8 ": {12, 8}, "13.0.3": {13, 0}} {
		if got, ok := ParseVersion(in); !ok || got != want {
			t.Errorf("ParseVersion(%q) = %v, %v; want %v", in, got, ok, want)
		}
	}
	for _, in := range []string{"", "13", "x.y", "0"} {
		if _, ok := ParseVersion(in); ok {
			t.Errorf("ParseVersion(%q) ok", in)
		}
	}
	if maj, min, patch, ok := ParseDriver("610.57.04"); !ok || maj != 610 || min != 57 || patch != 4 {
		t.Errorf("ParseDriver(610.57.04) = %d %d %d %v", maj, min, patch, ok)
	}
	if maj, min, patch, ok := ParseDriver("580.142"); !ok || maj != 580 || min != 142 || patch != 0 {
		t.Errorf("ParseDriver(580.142) = %d %d %d %v", maj, min, patch, ok)
	}
	for _, in := range []string{"", "13.0", "abc", "1.2.3.4"} {
		if _, _, _, ok := ParseDriver(in); ok {
			t.Errorf("ParseDriver(%q) ok", in)
		}
	}
	if (Version{12, 8}).Less(Version{12, 8}) || !(Version{12, 8}).Less(Version{13, 0}) || (Version{13, 1}).Less(Version{13, 0}) {
		t.Error("Version.Less")
	}
}

func TestDecodeQuickPodDriver(t *testing.T) {
	for in, want := range map[string]string{"5.90048001e+08": "590.48.01", "5.95099002e+08": "595.99.02", "580142000": "580.142.00"} {
		if got, ok := DecodeQuickPodDriver(in); !ok || got != want {
			t.Errorf("DecodeQuickPodDriver(%q) = %q, %v; want %q", in, got, ok, want)
		}
	}
	for _, in := range []string{"0", "", "junk", "13.1"} {
		if _, ok := DecodeQuickPodDriver(in); ok {
			t.Errorf("DecodeQuickPodDriver(%q) ok", in)
		}
	}
}

func TestBranchTable(t *testing.T) {
	for branch, want := range map[int]Version{535: {12, 2}, 570: {12, 8}, 580: {13, 0}, 595: {13, 2}, 615: {13, 4}} {
		if got, ok := MaxCUDAForBranch(branch); !ok || got != want {
			t.Errorf("MaxCUDAForBranch(%d) = %v, %v", branch, got, ok)
		}
	}
	if _, ok := MaxCUDAForBranch(999); ok {
		t.Error("unknown branch known")
	}
	if b, ok := BranchForCUDA(Version{13, 0}); !ok || b != 580 {
		t.Errorf("BranchForCUDA(13.0) = %d", b)
	}
	if b, ok := BranchForCUDA(Version{12, 3}); !ok || b != 550 {
		t.Errorf("BranchForCUDA(12.3) = %d", b)
	}
}

// Every Vast offer's driver value is the cuda_max_good: driver_vers (null
// or stale) has no effect, junk compute_cap falls back to the name table
// and inet_down is kept as Mbps.
func TestFromVast(t *testing.T) {
	for _, o := range vastOffers(t) {
		h := FromVast(o)
		if h.CUDASource != "vast.cuda_max_good" || h.CUDAMax == "" || h.Provider != "vast" {
			t.Errorf("%v: cuda %q from %q", o["id"], h.CUDAMax, h.CUDASource)
		}
		if h.SM == 0 {
			t.Errorf("%v %v: no SM", o["id"], o["gpu_name"])
		}
	}

	// driver_vers null on a 610 host, and stale (535) on a 595 host: no effect.
	if h := FromVast(vastRow(t, 90000001)); h.CUDAMax != "13.3" || h.Driver != "610.57.04" {
		t.Errorf("null driver_vers: %+v", h)
	}
	if h := FromVast(vastRow(t, 90000002)); h.CUDAMax != "13.2" || h.Driver != "595.71.05" {
		t.Errorf("stale driver_vers: %+v", h)
	}
	// compute_cap 243 is junk: the name table decides.
	if h := FromVast(vastRow(t, 90000004)); h.SM != 61 || h.SMSource != "table" || h.GPUClass != "pro" {
		t.Errorf("junk compute_cap: %+v", h)
	}
	h := FromVast(vastRow(t, 38341566))
	if h.SM != 120 || h.SMSource != "provider" || h.InetDownMbps != 1375.8 || h.DiskBWMBps != 2842 || h.GPUVendor != "nvidia" || h.CPUArch != "amd64" {
		t.Errorf("5090 row: %+v", h)
	}
	if h.VRAMGB < 31.8 || h.VRAMGB > 31.9 || h.DiskGB != 612 || h.RAMGB < 62 || h.RAMGB > 63 {
		t.Errorf("5090 sizes: %+v", h)
	}
	if h := FromVast(vastRow(t, 90000005)); h.CPUArch != "arm64" {
		t.Errorf("GH200 arch %q", h.CPUArch)
	}
}

func TestFromQuickPod(t *testing.T) {
	rows := quickPodOffers(t)
	h := FromQuickPod(rows[0]) // max_cuda "13.1\n", driver_vers "5.90048001e+08"
	if h.CUDAMax != "13.1" || h.CUDASource != "quickpod.max_cuda" || h.Driver != "590.48.01" || h.SM != 120 || h.Provider != "quickpod" {
		t.Errorf("5060 Ti row: %+v", h)
	}
	if h.DiskGB != 444 || h.InetDownMbps != 732.17 || h.DiskBWMBps != 698.35 || h.GPUClass != "geforce" {
		t.Errorf("5060 Ti sizes: %+v", h)
	}
	// max_cuda "" with driver_vers 595.99.02: the branch says 13.2.
	h = FromQuickPod(rows[8])
	if h.CUDAMax != "13.2" || h.CUDASource != "quickpod.driver_vers" || h.Driver != "595.99.02" || h.DiskBWMBps != 0 {
		t.Errorf("3090 empty max_cuda: %+v", h)
	}
	// A "0" driver with a max_cuda: the CUDA comes from max_cuda, no driver.
	h = FromQuickPod(rows[12])
	if h.CUDAMax != "13.3" || h.Driver != "" {
		t.Errorf("driver 0 row: %+v", h)
	}
	// "" max_cuda and "0" driver: unknown.
	raw := rows[12]
	raw["_machines"].(map[string]any)["max_cuda"] = ""
	if h := FromQuickPod(raw); h.CUDAMax != "" || h.CUDASource != "" {
		t.Errorf("unknown driver: %+v", h)
	}
	// Disk: the smaller of max_disk_size and avail_disk_space.
	r := map[string]any{"gpu_type": "NVIDIA GeForce RTX 4090", "max_disk_size": 900.0,
		"_machines": map[string]any{"avail_disk_space": 300.0, "cpu_arch": "amd64"}}
	if h := FromQuickPod(r); h.DiskGB != 300 || h.SM != 89 {
		t.Errorf("disk cross-check: %+v", h)
	}
	for _, row := range rows {
		if h := FromQuickPod(row); h.SM == 0 {
			t.Errorf("%v: no SM", row["gpu_type"])
		}
	}
}

// Every RunPod id the create enum accepts is in the GPU table.
func TestFromRunPodGPU(t *testing.T) {
	var ids []string
	readJSON(t, "runpod_gpu_ids.json", &ids)
	if len(ids) < 40 {
		t.Fatalf("only %d ids", len(ids))
	}
	for _, id := range ids {
		if _, ok := Lookup(id); !ok {
			t.Errorf("RunPod id %q not in the GPU table", id)
		}
		manufacturer := "NVIDIA"
		if strings.HasPrefix(id, "AMD") {
			manufacturer = "AMD"
		}
		h := FromRunPodGPU(RunPodGPUType{ID: id, Manufacturer: manufacturer, Memory: 24}, "13.0")
		if !h.PlacementFloor || h.CUDASource != "runpod.placement" || h.CUDAMax != "" {
			t.Errorf("%s: %+v", id, h)
		}
	}
	if h := FromRunPodGPU(RunPodGPUType{ID: "AMD Instinct MI300X OAM", Manufacturer: "AMD", Memory: 192}, "13.0"); h.GPUVendor != "amd" {
		t.Errorf("MI300X vendor %q", h.GPUVendor)
	}
	if h := FromRunPodGPU(RunPodGPUType{ID: "NVIDIA GH200 480GB", Manufacturer: "NVIDIA", Memory: 96}, "13.0"); h.CPUArch != "arm64" {
		t.Errorf("GH200 arch %q", h.CPUArch)
	}
	h := FromRunPodGPU(RunPodGPUType{ID: "NVIDIA GeForce RTX 4090", Memory: 24, CUDAVersions: []RunPodCUDAVersion{
		{"12.8", true}, {"13.0", true}, {"13.2", true}, {"13.4", false}}}, "13.0")
	if h.CUDAMax != "13.2" || h.CUDASource != "runpod.catalog" || h.SM != 89 || h.CPUArch != "amd64" || h.VRAMGB != 24 {
		t.Errorf("catalog 4090: %+v", h)
	}
	// Versions listed, none available: the catalog spoke (no capacity), CUDAMax unknown.
	busy := FromRunPodGPU(RunPodGPUType{ID: "NVIDIA GeForce RTX 4090", Memory: 24, CUDAVersions: []RunPodCUDAVersion{
		{"13.0", false}, {"13.2", false}}}, "13.0")
	if busy.CUDAMax != "" || busy.CUDASource != "runpod.catalog" || !busy.PlacementFloor {
		t.Errorf("all unavailable: %+v", busy)
	}
	if h := FromRunPodGPU(RunPodGPUType{ID: "NVIDIA L4"}, ""); h.PlacementFloor {
		t.Error("no floor sent, but PlacementFloor")
	}
}

func TestLookupAndSpeed(t *testing.T) {
	for name, want := range map[string]float64{"NVIDIA GeForce RTX 4090": 1.0, "RTX 4090": 1.0, "NVIDIA L4": 0.30,
		"NVIDIA H100 80GB HBM3": 2.0, "RTX 5090": 1.35, "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1.40} {
		if got, src, ok := Speed(name); !ok || got != want || src == "" {
			t.Errorf("Speed(%q) = %v %q %v; want %v", name, got, src, ok, want)
		}
	}
	if _, _, ok := Speed("Imaginary GPU 9000"); ok {
		t.Error("unknown GPU has a speed")
	}
	if row, ok := Lookup("NVIDIA GeForce RTX 5090 Laptop GPU"); !ok || row.Name != "RTX 5090 Laptop GPU" {
		t.Errorf("laptop suffix lost: %+v", row)
	}
	if row, ok := Lookup("Tesla V100-SXM2-16GB"); !ok || row.SM != 70 {
		t.Errorf("V100: %+v", row)
	}
	if row, _ := Lookup("NVIDIA B300 SXM6 AC"); row.SM != 103 || !row.Unverified {
		t.Errorf("B300: %+v", row)
	}
}

// Every speed has a documented source; no alias names two GPUs; every
// non-blocked RunPod id has a speed.
func TestGPUTable(t *testing.T) {
	seen := map[string]string{}
	for _, row := range GPUs() {
		if row.Speed > 0 {
			if _, ok := SpeedSources[row.SpeedSource]; !ok {
				t.Errorf("%s: speed source %q unknown", row.Name, row.SpeedSource)
			}
		} else if row.SpeedSource != "" {
			t.Errorf("%s: no speed but source %q", row.Name, row.SpeedSource)
		}
		for _, n := range append([]string{row.Name}, row.Aliases...) {
			k := normalise(n)
			if other, dup := seen[k]; dup && other != row.Name {
				t.Errorf("alias %q names both %s and %s", n, other, row.Name)
			}
			seen[k] = row.Name
		}
		if row.Vendor == vendorNVIDIA && row.SM == 0 {
			t.Errorf("%s: no SM", row.Name)
		}
	}
	var ids []string
	readJSON(t, "runpod_gpu_ids.json", &ids)
	req := goRequirements()
	for _, id := range ids {
		row, _ := Lookup(id)
		blocked := row.Vendor != vendorNVIDIA || row.Platform != platformAMD64 || !req.Runs(row.SM)
		if !blocked && row.Speed <= 0 {
			t.Errorf("RunPod %s runs the image but has no speed", id)
		}
	}
}
