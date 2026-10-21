package v1

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"time"

	"github.com/geocine/aitoolkit/internal/apicache"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/sysstats"
	"github.com/geocine/aitoolkit/internal/v1api"
)

var startedAt = time.Now()

// buildVersion is set at link time (-X) where there is no .git, as in the image.
var buildVersion string

// version is the server build: buildVersion, else the VCS revision Go
// embedded, else "dev".
func version() string {
	if buildVersion != "" {
		return buildVersion
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		rev, dirty := "", false
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				rev = s.Value
			case "vcs.modified":
				dirty = s.Value == "true"
			}
		}
		if len(rev) > 12 {
			rev = rev[:12]
		}
		if rev != "" {
			if dirty {
				rev += "-dirty"
			}
			return rev
		}
	}
	return "dev"
}

var upstreamVersionRe = regexp.MustCompile(`(?m)^VERSION\s*=\s*["']([^"']+)["']`)

// upstreamVersion is ai-toolkit's version.py VERSION.
func upstreamVersion() *string {
	b, err := os.ReadFile(filepath.Join(paths.ToolkitRoot, "version.py"))
	if err != nil {
		return nil
	}
	if m := upstreamVersionRe.FindSubmatch(b); m != nil {
		v := string(m[1])
		return &v
	}
	return nil
}

func health(w http.ResponseWriter, r *http.Request) {
	v1api.OK(w, map[string]any{
		"ok": true, "version": version(), "upstream_version": upstreamVersion(),
		"started_at": v1api.Time(startedAt),
	})
}

var cpuLoad sysstats.LoadSampler

type diskView struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	Free  uint64 `json:"free"`
	Total uint64 `json:"total"`
}

func system(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	gpu, err := apicache.Cached("v1-gpu", apicache.DefaultStale, nil, func() (sysstats.GPUResponse, error) {
		return sysstats.RouteGPUInfo(ctx)
	})
	if err != nil {
		gpu = sysstats.GPUResponse{GPUs: []sysstats.GPUInfo{}, Error: err.Error()}
	}
	cpu, err := apicache.Cached("v1-cpu", apicache.DefaultStale, nil, func() (sysstats.CPUInfo, error) {
		temp, _ := sysstats.ReadCPUTemperature()
		return sysstats.BuildCPUInfo(cpuLoad.Sample(), temp)
	})
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, map[string]any{
		"gpus": gpu.GPUs, "has_nvidia_smi": gpu.HasNvidiaSmi, "cpu": cpu,
		"disks": disks(ctx), "toolkit_root": paths.ToolkitRoot,
	})
}

// disks reports free space where training output and datasets live.
func disks(ctx context.Context) []diskView {
	out := []diskView{}
	for _, root := range fileRoots {
		dir, err := rootDir(ctx, root)
		if err != nil {
			continue
		}
		// a root not created yet is measured on its nearest existing parent
		probe := dir
		for {
			if _, err := os.Stat(probe); err == nil || filepath.Dir(probe) == probe {
				break
			}
			probe = filepath.Dir(probe)
		}
		free, total, err := diskSpace(probe)
		if err != nil {
			continue
		}
		out = append(out, diskView{Name: root, Path: dir, Free: free, Total: total})
	}
	return out
}
