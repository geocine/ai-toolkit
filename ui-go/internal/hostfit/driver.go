package hostfit

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/imagecheck"
)

// DefaultRepo is the server image studio-managed GPU servers run.
const DefaultRepo = imagecheck.DefaultRepo

func (v Version) String() string { return fmt.Sprintf("%d.%d", v.Major, v.Minor) }

// IsZero reports an unknown version.
func (v Version) IsZero() bool { return v == Version{} }

// Less reports v < w.
func (v Version) Less(w Version) bool {
	return v.Major < w.Major || v.Major == w.Major && v.Minor < w.Minor
}

// ParseVersion reads "13.1\n", "13.0" or "13.0.3" as major.minor. "" and
// anything without a minor are not versions.
func ParseVersion(s string) (Version, bool) {
	parts := strings.Split(strings.TrimSpace(s), ".")
	if len(parts) < 2 {
		return Version{}, false
	}
	maj, err1 := strconv.Atoi(parts[0])
	min, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil || maj <= 0 || min < 0 {
		return Version{}, false
	}
	return Version{maj, min}, true
}

// ParseDriver reads a driver version, "580.142" or "610.57.04". The patch
// is 0 when absent.
func ParseDriver(s string) (major, minor, patch int, ok bool) {
	parts := strings.Split(strings.TrimSpace(s), ".")
	if len(parts) < 2 || len(parts) > 3 {
		return 0, 0, 0, false
	}
	nums := make([]int, 3)
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil || n < 0 {
			return 0, 0, 0, false
		}
		nums[i] = n
	}
	if nums[0] < 100 {
		return 0, 0, 0, false
	}
	return nums[0], nums[1], nums[2], true
}

// DecodeQuickPodDriver turns QuickPod's float-encoded driver_vers
// ("5.90048001e+08" = 590048001) into "590.48.01". "0", "" and junk are not
// drivers.
func DecodeQuickPodDriver(s string) (string, bool) {
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || f < 1e8 || f >= 1e10 {
		return "", false
	}
	n := int64(f + 0.5)
	major, minor, patch := n/1_000_000, n/1000%1000, n%1000
	return fmt.Sprintf("%d.%02d.%02d", major, minor, patch), true
}

// branches is the driver branch -> highest CUDA it runs (IMAGE-PLAN §3.2).
// The 535, 560, 570, 575, 580, 590, 595, 610 and 615 rows match
// cuda_max_good / max_cuda on live offers; 525, 550 and 565 come from
// NVIDIA's release notes and are UNCONFIRMED until seen.
var branches = []struct {
	branch int
	cuda   Version
}{
	{525, Version{12, 0}}, {535, Version{12, 2}}, {550, Version{12, 4}}, {560, Version{12, 6}},
	{565, Version{12, 7}}, {570, Version{12, 8}}, {575, Version{12, 9}}, {580, Version{13, 0}},
	{590, Version{13, 1}}, {595, Version{13, 2}}, {610, Version{13, 3}}, {615, Version{13, 4}},
}

// MaxCUDAForBranch is the highest CUDA a driver branch runs. An unknown
// branch means the CUDA version is unknown.
func MaxCUDAForBranch(major int) (Version, bool) {
	for _, b := range branches {
		if b.branch == major {
			return b.cuda, true
		}
	}
	return Version{}, false
}

// BranchForCUDA is the lowest driver branch that runs CUDA v.
func BranchForCUDA(v Version) (int, bool) {
	for _, b := range branches {
		if !b.cuda.Less(v) && b.cuda.Major == v.Major {
			return b.branch, true
		}
	}
	return 0, false
}
