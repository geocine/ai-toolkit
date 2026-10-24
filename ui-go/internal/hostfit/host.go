// Package hostfit matches an image (imagecheck) to a GPU host: driver and
// CUDA, GPU architecture, disk, pull time and VRAM. It imports imagecheck but
// never provision or gpupick (both import it); adapters convert their
// raw offers into a Host.
package hostfit

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
