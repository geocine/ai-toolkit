package hostfit

import (
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/imagecheck"
)

// FilterFor is what an image (with its disk need, a model and the user's
// policy) asks of a host. SMAllowed is every known compute capability on
// amd64 the image's torch build runs, minus sm_75 unless AllowTuring (then
// in SMHidden); it is empty (no SM filter) when the image says nothing
// about its GPUs.
func FilterFor(req imagecheck.Requirements, need Need, model *ModelRef, policy Policy) Filter {
	f := Filter{
		Platform:    req.Platform,
		Vendor:      vendorNVIDIA,
		MinDiskGB:   need.DiskGB,
		ImagePeakGB: need.ImagePeakGB,
		MinVRAMGB:   MinVRAMGB(model),
		AllowTuring: policy.AllowTuring,
		ShowAll:     policy.ShowAll,
	}
	if f.Platform == "" {
		f.Platform = platformAMD64
	}
	if v, ok := ParseVersion(req.CUDAMin); ok {
		f.CUDAMin = v
	}
	if len(req.SM) > 0 || len(req.PTX) > 0 {
		for _, sm := range knownSM(f.Platform) {
			switch {
			case !req.Runs(sm):
			case sm/10 == 7 && !policy.AllowTuring:
				f.SMHidden = append(f.SMHidden, sm)
			default:
				f.SMAllowed = append(f.SMAllowed, sm)
			}
		}
	}
	if policy.Fast {
		f.MinInetDownMbps, f.MinDiskBWMBps = FastInetDownMbps, FastDiskBWMBps
	}
	return f
}

// VastQuery merges f into the bundles search body the Vast adapter already
// built (base is not modified). cuda_max_good is the only driver gate:
// driver_vers and driver_version are never sent (null or stale on many
// hosts). compute_cap uses "in" over SMAllowed x 10, which also drops junk
// values. gpu_ram is MiB. Under ShowAll the image-side terms (CUDA,
// compute_cap, cpu_arch) and the VRAM floor are left out and the offers
// are badged instead; disk_space and the fast-host terms stay (Keeps
// follows the same rule). Where base already filters a field, the
// stricter bound wins.
func VastQuery(f Filter, base map[string]any) map[string]any {
	out := maps.Clone(base)
	if out == nil {
		out = map[string]any{}
	}
	if f.Vendor != "" {
		merge(out, "gpu_arch", map[string]any{"eq": f.Vendor})
	}
	if !f.ShowAll {
		if _, arch, ok := strings.Cut(f.Platform, "/"); ok {
			merge(out, "cpu_arch", map[string]any{"eq": arch})
		}
		if !f.CUDAMin.IsZero() {
			merge(out, "cuda_max_good", map[string]any{"gte": float64(f.CUDAMin.Major) + float64(f.CUDAMin.Minor)/10})
		}
		if len(f.SMAllowed) > 0 {
			ccs := make([]any, len(f.SMAllowed))
			for i, sm := range f.SMAllowed {
				ccs[i] = sm * 10
			}
			merge(out, "compute_cap", map[string]any{"in": ccs})
		}
	}
	if f.MinDiskGB > 0 {
		merge(out, "disk_space", map[string]any{"gte": f.MinDiskGB})
	}
	if f.MinVRAMGB > 0 && !f.ShowAll {
		merge(out, "gpu_ram", map[string]any{"gte": int(math.Floor(f.MinVRAMGB * 1024))})
	}
	if f.MinInetDownMbps > 0 {
		merge(out, "inet_down", map[string]any{"gte": f.MinInetDownMbps})
	}
	if f.MinDiskBWMBps > 0 {
		merge(out, "disk_bw", map[string]any{"gte": f.MinDiskBWMBps})
	}
	delete(out, "driver_vers")
	delete(out, "driver_version")
	return out
}

// merge sets body[field] to ops, keeping the stricter of two lower ("gte",
// "gt") or upper ("lte", "lt") bounds when body already has the field.
func merge(body map[string]any, field string, ops map[string]any) {
	have, ok := body[field].(map[string]any)
	if !ok {
		body[field] = ops
		return
	}
	merged := maps.Clone(have)
	for op, v := range ops {
		old, had := merged[op]
		switch {
		case !had:
			merged[op] = v
		case op == "gte" || op == "gt":
			if num(v) > num(old) {
				merged[op] = v
			}
		case op == "lte" || op == "lt":
			if num(v) < num(old) {
				merged[op] = v
			}
		default:
			merged[op] = v
		}
	}
	body[field] = merged
}

// RunPodGPUConfig is the create body's gpu.minCudaVersion ("13.0"): the
// image's CUDA floor, sent always and alone (allowedCudaVersions is never
// sent: RunPod answers 400 when both are). "" when the image has no floor.
func RunPodGPUConfig(f Filter) (minCudaVersion string) {
	if f.CUDAMin.IsZero() {
		return ""
	}
	return f.CUDAMin.String()
}

// RunPodAllowedType reports whether a RunPod catalog GPU type can run the
// image: an NVIDIA amd64 GPU whose table SM is allowed. A GPU the table
// does not know is kept (its fit says gpu_unknown). why is the Fit code of
// the reason: an sm_75 type the image runs but the policy hides is
// gpu_sm_flash_attn (a warn when forced, D-8), not gpu_sm.
func RunPodAllowedType(f Filter, gpuID string) (ok bool, why string) {
	row, known := Lookup(gpuID)
	if !known {
		return true, ""
	}
	switch {
	case f.Vendor != "" && row.Vendor != f.Vendor:
		return false, "gpu_vendor"
	case f.Platform != "" && row.Platform != f.Platform:
		return false, "host_arch"
	case slices.Contains(f.SMHidden, row.SM):
		return false, "gpu_sm_flash_attn"
	case len(f.SMAllowed) > 0 && !slices.Contains(f.SMAllowed, row.SM):
		return false, "gpu_sm"
	}
	return true, ""
}

// QuickPodKeep applies f to a QuickPod offer client-side (QuickPod has no
// server-side filters): see Keeps.
func QuickPodKeep(f Filter, h Host) (keep bool, level string) { return Keeps(f, h) }

// Keeps applies f to one host client-side, for adapters that cannot send
// it, the same way VastQuery filters Vast. level is the worst level Fit
// gives the host, for badges: block when the image cannot run there (CPU
// architecture, vendor, an SM the build lacks, a CUDA version below the
// floor, a disk below the image peak); warn for an sm_75 GPU the policy
// hides (D-8), an unknown CUDA version, no free capacity at RunPod's
// floor, a disk below the need, VRAM below the floor or a reported link
// below the fast-host floor; info for a GPU of unknown architecture. Hosts the
// image cannot run, hidden SMs and VRAM below the floor are dropped unless
// ShowAll; a disk below the need or a fast-host failure is dropped even
// under ShowAll (Vast keeps disk_space and inet_down then too). An unknown
// CUDA version is kept. Unreported bandwidth (0) passes (MASTER D-15).
func Keeps(f Filter, h Host) (keep bool, level string) {
	var seen []imagecheck.Check
	flag := func(l string) { seen = append(seen, imagecheck.Check{Level: l}) }
	incompatible, hard := false, false
	arch := archOfHost(h)
	if _, want, ok := strings.Cut(f.Platform, "/"); ok && arch != "" && arch != want {
		flag(imagecheck.LevelBlock)
		incompatible = true
	}
	vendor := h.GPUVendor
	if f.Vendor != "" && vendor != "" && vendor != f.Vendor {
		flag(imagecheck.LevelBlock)
		incompatible = true
	}
	if len(f.SMAllowed)+len(f.SMHidden) > 0 && vendor != vendorAMD && !slices.Contains(f.SMAllowed, h.SM) {
		switch {
		case h.SM == 0:
			flag(imagecheck.LevelInfo) // gpu_unknown
		case slices.Contains(f.SMHidden, h.SM):
			flag(imagecheck.LevelWarn) // gpu_sm_flash_attn, shown only when forced
		default:
			flag(imagecheck.LevelBlock)
		}
		incompatible = true
	}
	if !f.CUDAMin.IsZero() {
		v, known := ParseVersion(h.CUDAMax)
		switch {
		case h.PlacementFloor && (known && v.Less(f.CUDAMin) || !known && h.CUDASource == "runpod.catalog"):
			flag(imagecheck.LevelWarn) // runpod_no_cuda_capacity
		case h.PlacementFloor:
		case !known:
			flag(imagecheck.LevelWarn) // host_cuda_unknown
		case v.Less(f.CUDAMin):
			flag(imagecheck.LevelBlock)
			incompatible = true
		}
	}
	if h.DiskGB > 0 {
		switch {
		case f.ImagePeakGB > 0 && h.DiskGB < f.ImagePeakGB+imageHeadroomGB:
			flag(imagecheck.LevelBlock) // disk_image
			hard = true
		case f.MinDiskGB > 0 && h.DiskGB < float64(f.MinDiskGB):
			flag(imagecheck.LevelWarn) // disk_model
			hard = true
		}
	}
	if f.MinVRAMGB > 0 && h.VRAMGB > 0 && h.VRAMGB < f.MinVRAMGB {
		flag(imagecheck.LevelWarn) // vram_no
		incompatible = true
	}
	if f.MinInetDownMbps > 0 && h.InetDownMbps > 0 && h.InetDownMbps < f.MinInetDownMbps ||
		f.MinDiskBWMBps > 0 && h.DiskBWMBps > 0 && h.DiskBWMBps < f.MinDiskBWMBps {
		flag(imagecheck.LevelWarn) // pull_slow
		hard = true
	}
	return !hard && (!incompatible || f.ShowAll), imagecheck.Worst(seen)
}
