// Package imagecheck reads what a Docker image needs from a host (CUDA floor,
// GPU architectures, platform, size) from its registry, without pulling it, so
// a GPU server is never started on an image that cannot run there.
//
// It never imports provision or hostfit: both import it.
package imagecheck

import "time"

// DefaultRepo is the server image studio-managed GPU servers run. It is
// published publicly, so it is read with anonymous pulls.
const DefaultRepo = "ghcr.io/geocine/aitoolkit-go"

// RateLimit is what a registry said about its pull limit (Docker Hub only).
type RateLimit struct {
	Limit     int    `json:"limit"` // ratelimit-limit "100;w=21600" -> 100
	Remaining int    `json:"remaining"`
	WindowS   int    `json:"window_s"` // from w=
	Source    string `json:"-"`        // docker-ratelimit-source (an IP); never logged or sent
}

// Image is what one platform manifest of a reference says about itself.
type Image struct {
	Ref            string            `json:"ref"`  // as asked
	Repo           string            `json:"repo"` // "ghcr.io/geocine/aitoolkit-go"
	Tag            string            `json:"tag,omitempty"`
	Digest         string            `json:"digest"`          // what the tag points at (index or manifest)
	PlatformDigest string            `json:"platform_digest"` // the linux/amd64 manifest
	Platforms      []string          `json:"platforms"`       // attestation (unknown/unknown) entries skipped
	Platform       string            `json:"platform"`        // the one inspected
	DownloadBytes  int64             `json:"download_bytes"`  // sum of unique layer blobs (deduplicated by digest)
	LargestLayer   int64             `json:"largest_layer_bytes"`
	Layers         int               `json:"layers"`
	LayerDigests   []string          `json:"-"` // for the base-layer reuse estimate and tests
	UnpackedBytes  int64             `json:"unpacked_bytes"`
	UnpackedSource string            `json:"unpacked_source"` // "label" | "estimate" (2.1x compressed, measured 2.08x)
	Created        time.Time         `json:"created"`
	Labels         map[string]string `json:"labels"`             // org.opencontainers.image.* and io.github.geocine.aitk.* only
	Env            map[string]string `json:"env"`                // CUDA_VERSION, NVIDIA_REQUIRE_CUDA, NVIDIA_DRIVER_CAPABILITIES only
	ExposedPorts   []string          `json:"exposed_ports"`      // "8675/tcp"
	GoServer       bool              `json:"go_server"`          // label server=go, or a history created_by mentioning aitk-server
	Revision       string            `json:"revision,omitempty"` // label revision (short sha)
	Requires       Requirements      `json:"requires"`
	CheckedAt      time.Time         `json:"checked_at"`
	RateLimit      *RateLimit        `json:"rate_limit,omitempty"`
}

// BrandRange is one container-gate exception of NVIDIA_REQUIRE_CUDA
// ("brand=tesla,driver>=535,driver<536").
type BrandRange struct {
	Brand  string `json:"brand"`  // "tesla"
	Branch int    `json:"branch"` // 535 (from driver>=535,driver<536)
}

// Requirements is what an image needs from a host, read from its labels, env
// and history.
type Requirements struct {
	Platform       string            `json:"platform"`         // "linux/amd64"
	CUDAMin        string            `json:"cuda_min"`         // "13.0"
	DriverMin      int               `json:"driver_min"`       // 580 (display)
	CompatBranches []BrandRange      `json:"compat_branches"`  // container-gate exceptions; advisory, never a pass
	SM             []int             `json:"sm"`               // SASS: 75,80,86,90,100,120
	PTX            []int             `json:"ptx"`              // usually empty
	FlashAttnSM    []int             `json:"flash_attn_sm"`    // 80,90,100,120; empty = not installed or unknown
	Torch          string            `json:"torch"`            // "2.13.0+cu130"
	Flavor         string            `json:"flavor"`           // "cu130"
	Port           int               `json:"port"`             // 8675
	API            int               `json:"api"`              // 1; 0 = not a Go-server image
	Sources        map[string]string `json:"sources"`          // field -> "label" | "env" | "history" | "table" | "default"
	Source         string            `json:"source,omitempty"` // "fallback" = built in, the image could not be read; "" = derived from the image
}

// TagInfo is one tag of a repository, for the image picker.
type TagInfo struct {
	Tag      string    `json:"tag"`
	Digest   string    `json:"digest"`
	Created  time.Time `json:"created"`
	Flavor   string    `json:"flavor,omitempty"`
	CUDAMin  string    `json:"cuda_min,omitempty"`
	Revision string    `json:"revision,omitempty"`
	Moving   bool      `json:"moving"` // latest, cuXXX, <version>; false for <sha>, <yymmddHHMM>
}
