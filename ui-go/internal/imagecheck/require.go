package imagecheck

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Label keys the image check reads (IMAGE-PLAN §1.3). Only these two
// prefixes survive into Image.Labels.
const (
	labelPrefixOCI  = "org.opencontainers.image."
	labelPrefixAITK = "io.github.geocine.aitk."

	LabelServer       = labelPrefixAITK + "server"
	LabelAPI          = labelPrefixAITK + "api"
	LabelRevision     = labelPrefixAITK + "revision"
	LabelFlavor       = labelPrefixAITK + "flavor"
	LabelCUDA         = labelPrefixAITK + "cuda"
	LabelDriverMin    = labelPrefixAITK + "driver_min"
	LabelTorch        = labelPrefixAITK + "torch"
	LabelSM           = labelPrefixAITK + "sm"
	LabelPTX          = labelPrefixAITK + "ptx"
	LabelFlashAttnSM  = labelPrefixAITK + "sm.flash_attn"
	LabelPort         = labelPrefixAITK + "port"
	LabelBaseUnpacked = labelPrefixAITK + "size.base_unpacked"
	LabelVersion      = labelPrefixOCI + "version"
)

// ServerPort is the port the Go server listens on in the image.
const ServerPort = 8675

// APIVersion is the /v1 major version this server speaks (label api).
const APIVersion = 1

// Version is a CUDA version, major.minor.
type Version struct{ Major, Minor int }

func (v Version) String() string { return fmt.Sprintf("%d.%d", v.Major, v.Minor) }

// parseVersion reads "13.0", "13.0.3" or "13.1\n" as major.minor.
func parseVersion(s string) (Version, bool) {
	parts := strings.Split(strings.TrimSpace(s), ".")
	if len(parts) < 2 {
		return Version{}, false
	}
	maj, err1 := strconv.Atoi(parts[0])
	min, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil || maj < 0 || min < 0 {
		return Version{}, false
	}
	return Version{maj, min}, true
}

// driverBranches is the driver branch that first runs each CUDA version
// (IMAGE-PLAN §3.2; hostfit holds the full table). Branches not yet seen on
// live offers are UNCONFIRMED.
var driverBranches = []struct {
	cuda   Version
	branch int
}{
	{Version{12, 0}, 525}, {Version{12, 2}, 535}, {Version{12, 4}, 550}, {Version{12, 6}, 560},
	{Version{12, 7}, 565}, {Version{12, 8}, 570}, {Version{12, 9}, 575}, {Version{13, 0}, 580},
	{Version{13, 1}, 590}, {Version{13, 2}, 595}, {Version{13, 3}, 610}, {Version{13, 4}, 615},
}

// driverFor is the lowest driver branch that runs CUDA v, at least the floor
// of v's major (13.x needs R580, 12.x R525). 0 when v is unknown.
func driverFor(v Version) int {
	floor := map[int]int{12: 525, 13: 580}[v.Major]
	for _, b := range driverBranches {
		if b.cuda.Major == v.Major && b.cuda.Minor >= v.Minor || b.cuda.Major > v.Major {
			return max(b.branch, floor)
		}
	}
	return floor
}

// ParseRequireCUDA reads the NVIDIA container toolkit's start gate,
// "cuda>=13.0 brand=tesla,driver>=535,driver<536 ...": the CUDA floor and the
// brand/branch exceptions, deduplicated in order of appearance.
func ParseRequireCUDA(env string) (min Version, exceptions []BrandRange, err error) {
	found := false
	seen := map[BrandRange]bool{}
	for _, clause := range strings.Fields(env) {
		if v, ok := strings.CutPrefix(clause, "cuda>="); ok {
			if min, found = parseVersion(v); !found {
				return Version{}, nil, fmt.Errorf("bad cuda clause %q", clause)
			}
			continue
		}
		var br BrandRange
		for _, part := range strings.Split(clause, ",") {
			switch {
			case strings.HasPrefix(part, "brand="):
				br.Brand = strings.TrimPrefix(part, "brand=")
			case strings.HasPrefix(part, "driver>="):
				br.Branch, _ = strconv.Atoi(strings.TrimPrefix(part, "driver>="))
			}
		}
		if br.Brand != "" && br.Branch > 0 && !seen[br] {
			seen[br] = true
			exceptions = append(exceptions, br)
		}
	}
	if !found {
		return Version{}, nil, errors.New("no cuda>= clause")
	}
	return min, exceptions, nil
}

// torchSM is the SASS list of each torch wheel flavor, for images without an
// sm label (pytorch .ci/manywheel/build_env_setup.py at v2.13.0). cu129 is
// UNCONFIRMED.
var torchSM = map[string][]int{
	"cu130": {75, 80, 86, 90, 100, 120},
	"cu129": {75, 80, 86, 90, 100, 120},
	"cu126": {50, 60, 70, 75, 80, 86, 90},
}

// flashAttnCU130 is the SASS list of the prebuilt flash-attn 2.8.3 wheel the
// cu130 Go-server image installs.
var flashAttnCU130 = []int{80, 90, 100, 120}

// flavorOf is "cu130" for torch "2.13.0+cu130" or CUDA_VERSION "13.0.3".
func flavorOf(torch, cudaVersion string) string {
	if _, f, ok := strings.Cut(torch, "+"); ok && strings.HasPrefix(f, "cu") {
		return f
	}
	if v, ok := parseVersion(cudaVersion); ok {
		return fmt.Sprintf("cu%d%d", v.Major, v.Minor)
	}
	return ""
}

// Derive reads what img needs from a host: labels first, then config env and
// history, then built-in tables (IMAGE-PLAN §2.4). Sources says where each
// field came from.
func Derive(img *Image) Requirements {
	lab := func(k string) (string, bool) {
		v, ok := img.Labels[k]
		return strings.TrimSpace(v), ok && strings.TrimSpace(v) != ""
	}
	r := Requirements{
		Platform:       img.Platform,
		CompatBranches: []BrandRange{},
		SM:             []int{},
		PTX:            []int{},
		FlashAttnSM:    []int{},
		Sources:        map[string]string{},
	}
	if r.Platform != "" {
		r.Sources["platform"] = "manifest"
	}

	// CUDA floor: the container gate first, since that is what refuses to start.
	gate, exceptions, gateErr := ParseRequireCUDA(img.Env["NVIDIA_REQUIRE_CUDA"])
	switch {
	case gateErr == nil:
		r.CUDAMin, r.Sources["cuda_min"] = gate.String(), "env"
		r.CompatBranches = append(r.CompatBranches, exceptions...)
	default:
		if v, ok := lab(LabelCUDA); ok {
			if pv, ok := parseVersion(v); ok {
				r.CUDAMin, r.Sources["cuda_min"] = pv.String(), "label"
			}
		}
		if r.CUDAMin == "" {
			if pv, ok := parseVersion(img.Env["CUDA_VERSION"]); ok {
				r.CUDAMin, r.Sources["cuda_min"] = pv.String(), "env"
			}
		}
	}

	if v, ok := lab(LabelDriverMin); ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			r.DriverMin, r.Sources["driver_min"] = n, "label"
		}
	}
	if r.DriverMin == 0 && r.CUDAMin != "" {
		v, _ := parseVersion(r.CUDAMin)
		if n := driverFor(v); n > 0 {
			r.DriverMin, r.Sources["driver_min"] = n, "table"
		}
	}

	if v, ok := lab(LabelTorch); ok {
		r.Torch, r.Sources["torch"] = v, "label"
	}
	if v, ok := lab(LabelFlavor); ok {
		r.Flavor, r.Sources["flavor"] = v, "label"
	} else if f := flavorOf(r.Torch, img.Env["CUDA_VERSION"]); f != "" {
		r.Flavor, r.Sources["flavor"] = f, "env"
		if r.Torch != "" {
			r.Sources["flavor"] = "label"
		}
	}

	// The server API: label api, else label server=go or a history line
	// mentioning aitk-server (an older Go image without labels).
	if v, ok := lab(LabelAPI); ok {
		if n, err := strconv.Atoi(v); err == nil {
			r.API, r.Sources["api"] = n, "label"
		}
	}
	if r.API == 0 {
		if v, _ := lab(LabelServer); v == "go" {
			r.API, r.Sources["api"] = APIVersion, "label"
		} else if img.GoServer {
			r.API, r.Sources["api"] = APIVersion, "history"
		}
	}

	if v, ok := lab(LabelSM); ok {
		r.SM, r.Sources["sm"] = parseInts(v), "label"
	} else if sm, ok := torchSM[flavorOf(r.Torch, "")]; ok {
		r.SM, r.Sources["sm"] = slices.Clone(sm), "table"
	} else if sm, ok := torchSM[r.Flavor]; ok {
		r.SM, r.Sources["sm"] = slices.Clone(sm), "table"
	}
	if v, ok := img.Labels[LabelPTX]; ok {
		r.PTX, r.Sources["ptx"] = parseInts(v), "label"
	}
	if v, ok := img.Labels[LabelFlashAttnSM]; ok {
		// Present but empty means flash-attn is not installed.
		r.FlashAttnSM, r.Sources["flash_attn_sm"] = parseInts(v), "label"
	} else if r.API >= 1 && r.Flavor == "cu130" {
		r.FlashAttnSM, r.Sources["flash_attn_sm"] = slices.Clone(flashAttnCU130), "table"
	}

	if v, ok := lab(LabelPort); ok {
		if n, err := strconv.Atoi(v); err == nil {
			r.Port, r.Sources["port"] = n, "label"
		}
	}
	if r.Port == 0 && slices.Contains(img.ExposedPorts, fmt.Sprintf("%d/tcp", ServerPort)) {
		r.Port, r.Sources["port"] = ServerPort, "config"
	}
	return r
}

// parseInts reads "75,80,86" (spaces allowed); junk entries are skipped.
func parseInts(s string) []int {
	out := []int{}
	for _, f := range strings.FieldsFunc(s, func(c rune) bool { return c == ',' || c == ' ' || c == ';' }) {
		if n, err := strconv.Atoi(f); err == nil && n > 0 {
			out = append(out, n)
		}
	}
	return out
}

// FallbackRequirements is what the default image needs when its registry
// cannot be read (not published yet, private, unreachable): today's cu130
// build (IMAGE-PLAN §0.1). Hosts are then still judged, and the image
// failure is shown once where the image is checked. Other repos have no
// fallback.
func FallbackRequirements(repo string) (Requirements, bool) {
	if r, err := ParseRef(repo); err != nil || r.Name() != DefaultRepo {
		return Requirements{}, false
	}
	sources := map[string]string{}
	for _, k := range []string{"platform", "cuda_min", "driver_min", "sm", "flash_attn_sm", "torch", "flavor", "port", "api"} {
		sources[k] = "default"
	}
	return Requirements{
		Platform:       "linux/amd64",
		CUDAMin:        "13.0",
		DriverMin:      580,
		CompatBranches: []BrandRange{},
		SM:             slices.Clone(torchSM["cu130"]),
		PTX:            []int{},
		FlashAttnSM:    slices.Clone(flashAttnCU130),
		Torch:          "2.13.0+cu130",
		Flavor:         "cu130",
		Port:           ServerPort,
		API:            APIVersion,
		Sources:        sources,
		Source:         "fallback",
	}, true
}

// RunsOn reports whether a GPU of compute capability sm (CC x 10: 89 for
// 8.9, 120 for 12.0) runs code built for the SASS targets in sass: a cubin
// for sm_X.Y runs on sm_X.Z when Z >= Y (same major).
func RunsOn(sm int, sass []int) bool {
	if sm <= 0 {
		return false
	}
	for _, s := range sass {
		if s/10 == sm/10 && s%10 <= sm%10 {
			return true
		}
	}
	return false
}

// Runs reports whether the image's torch build runs on sm: a SASS target of
// the same major, or a PTX target at or below it (JIT-compiled).
func (r Requirements) Runs(sm int) bool {
	if RunsOn(sm, r.SM) {
		return true
	}
	for _, p := range r.PTX {
		if sm >= p {
			return true
		}
	}
	return false
}
