// Package hostfit matches an image (imagecheck) to a GPU host: driver and
// CUDA, GPU architecture, disk, pull time and VRAM. It imports imagecheck but
// never provision or gpupick (both import it); adapters convert their
// raw offers into a Host.
package hostfit

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// Host is the provider-neutral facts about one GPU host or offer.
// A zero field means unknown.
type Host struct {
	Provider       string  `json:"provider,omitempty"`
	GPUName        string  `json:"gpu_name,omitempty"`
	GPUCount       int     `json:"gpu_count,omitempty"`
	VRAMGB         float64 `json:"vram_gb,omitempty"`        // per GPU, MiB/1024
	SM             int     `json:"sm,omitempty"`             // 86, 120; 0 = unknown
	SMSource       string  `json:"sm_source,omitempty"`      // "provider" (Vast compute_cap) | "table"
	GPUClass       string  `json:"gpu_class,omitempty"`      // geforce | pro | datacenter
	GPUVendor      string  `json:"gpu_vendor,omitempty"`     // nvidia | amd
	CPUArch        string  `json:"cpu_arch,omitempty"`       // amd64 | arm64
	CUDAMax        string  `json:"cuda_max,omitempty"`       // "13.3"; "" = unknown
	CUDASource     string  `json:"cuda_source,omitempty"`    // vast.cuda_max_good | quickpod.max_cuda | quickpod.driver_vers | runpod.catalog | runpod.placement
	Driver         string  `json:"driver,omitempty"`         // "610.57.04", display only
	DiskGB         float64 `json:"disk_gb,omitempty"`        // most this host can allocate; 0 = chosen at launch (RunPod)
	InetDownMbps   float64 `json:"inet_down_mbps,omitempty"` // 0 = unknown (not reported; passes the fast-host rule)
	DiskBWMBps     float64 `json:"disk_bw_mbps,omitempty"`   // 0 = unknown (not reported; passes the fast-host rule)
	RAMGB          float64 `json:"ram_gb,omitempty"`
	PlacementFloor bool    `json:"placement_floor,omitempty"` // provider enforces the CUDA floor at create (RunPod minCudaVersion)
}

// IsZero reports a host with no facts at all (no host was given).
func (h Host) IsZero() bool { return h == Host{} }

// fillFromTable completes what the provider did not say from the GPU table:
// SM (when the provider gave none it can be trusted on), class, vendor and,
// when the provider names no CPU architecture, the platform.
func (h *Host) fillFromTable() {
	row, ok := Lookup(h.GPUName)
	if !ok {
		return
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
}

// FromVast reads a Vast bundles offer. compute_cap is trusted only when it
// lands on a known compute capability (junk such as 243 falls back to the
// name table); driver_vers is ignored (null or stale on many hosts) and
// cuda_max_good is the driver gate. inet_down is read as Mbps (UNCONFIRMED
// unit) and disk_bw as MB/s; 0 or missing means not reported.
func FromVast(raw map[string]any) Host {
	h := Host{
		Provider:     "vast",
		GPUName:      str(raw["gpu_name"]),
		GPUCount:     int(num(raw["num_gpus"])),
		VRAMGB:       num(raw["gpu_ram"]) / 1024,
		GPUVendor:    strings.ToLower(str(raw["gpu_arch"])),
		CPUArch:      strings.ToLower(str(raw["cpu_arch"])),
		DiskGB:       num(raw["disk_space"]),
		InetDownMbps: num(raw["inet_down"]),
		DiskBWMBps:   num(raw["disk_bw"]),
		RAMGB:        num(raw["cpu_ram"]) / 1024,
	}
	if cc := int(num(raw["compute_cap"])); cc > 0 && cc%10 == 0 && isKnownSM(cc/10) {
		h.SM, h.SMSource = cc/10, "provider"
	}
	if c := num(raw["cuda_max_good"]); c > 0 {
		h.CUDAMax, h.CUDASource = fmt.Sprintf("%.1f", c), "vast.cuda_max_good"
	}
	if d := str(raw["driver_version"]); d != "" {
		if _, _, _, ok := ParseDriver(d); ok {
			h.Driver = d
		}
	}
	h.fillFromTable()
	return h
}

// FromQuickPod reads a QuickPod /rentable offer (live shape: the machine
// nested under _machines). max_cuda is a string ("13.1\n", "13.0" or "");
// when it is empty the float-encoded driver_vers gives the branch, and "0"
// leaves the CUDA version unknown. Disk is the smaller of the offer's
// max_disk_size and the machine's avail_disk_space. Reliability is the
// provisioning adapter's to normalise.
func FromQuickPod(raw map[string]any) Host {
	m, _ := raw["_machines"].(map[string]any)
	if m == nil {
		m = map[string]any{}
	}
	h := Host{
		Provider:     "quickpod",
		GPUName:      str(raw["gpu_type"]),
		GPUCount:     int(num(raw["num_gpus"])),
		VRAMGB:       num(raw["gpu_vram"]) / 1024,
		CPUArch:      strings.ToLower(str(m["cpu_arch"])),
		InetDownMbps: num(m["inet_down"]),
		DiskBWMBps:   num(m["disk_bw"]),
		RAMGB:        num(m["cpu_ram"]),
	}
	h.DiskGB = num(raw["max_disk_size"])
	if avail := num(m["avail_disk_space"]); avail > 0 && (h.DiskGB == 0 || avail < h.DiskGB) {
		h.DiskGB = avail
	}
	driver, driverOK := DecodeQuickPodDriver(str(m["driver_vers"]))
	if driverOK {
		h.Driver = driver
	}
	if v, ok := ParseVersion(str(m["max_cuda"])); ok {
		h.CUDAMax, h.CUDASource = v.String(), "quickpod.max_cuda"
	} else if driverOK {
		major, _, _, _ := ParseDriver(driver)
		if v, ok := MaxCUDAForBranch(major); ok {
			h.CUDAMax, h.CUDASource = v.String(), "quickpod.driver_vers"
		}
	}
	h.fillFromTable()
	return h
}

// RunPodGPUType is one GPU type of RunPod's v2 catalog
// (GET /v2/catalog/gpus?include=AVAILABILITY&product=POD).
type RunPodGPUType struct {
	ID           string              `json:"id"`           // "NVIDIA GeForce RTX 4090": the gpu.id a create sends
	Name         string              `json:"name"`         // "RTX 4090"
	Manufacturer string              `json:"manufacturer"` // NVIDIA | AMD | UNKNOWN
	Memory       float64             `json:"memory"`       // VRAM GB
	CUDAVersions []RunPodCUDAVersion `json:"cudaVersions"` // with include=AVAILABILITY (the user's key) only
}

// RunPodCUDAVersion is one entry of a catalog GPU type's cudaVersions.
type RunPodCUDAVersion struct {
	Version   string `json:"version"`
	Available bool   `json:"available"`
}

// FromRunPodGPU reads a RunPod catalog GPU type. RunPod reports no driver,
// bandwidth or compute capability: SM comes from the table, CUDAMax is the
// highest available cudaVersions entry (only with the user's key; a
// catalog that lists versions with none available leaves CUDAMax empty
// with CUDASource "runpod.catalog"), and
// PlacementFloor says RunPod itself places the pod on a host with at least
// cudaMin (the create body's gpu.minCudaVersion); "" sends no floor.
func FromRunPodGPU(gpu RunPodGPUType, cudaMin string) Host {
	h := Host{
		Provider:       "runpod",
		GPUName:        gpu.ID,
		VRAMGB:         gpu.Memory,
		PlacementFloor: cudaMin != "",
	}
	if strings.EqualFold(gpu.Manufacturer, "AMD") {
		h.GPUVendor = vendorAMD
	}
	var best Version
	for _, c := range gpu.CUDAVersions {
		if v, ok := ParseVersion(c.Version); ok && c.Available && best.Less(v) {
			best = v
		}
	}
	switch {
	case !best.IsZero():
		h.CUDAMax, h.CUDASource = best.String(), "runpod.catalog"
	case len(gpu.CUDAVersions) > 0:
		// The catalog lists versions but none is free: no capacity, not unknown.
		h.CUDASource = "runpod.catalog"
	case h.PlacementFloor:
		h.CUDASource = "runpod.placement"
	}
	h.fillFromTable()
	return h
}

// num reads a JSON number (or a numeric string); anything else is 0.
func num(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case json.Number:
		f, _ := x.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(strings.TrimSpace(x), 64)
		return f
	}
	return 0
}

// str reads a JSON string; anything else is "".
func str(v any) string {
	s, _ := v.(string)
	return strings.TrimSpace(s)
}
