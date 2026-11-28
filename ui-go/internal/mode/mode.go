// Package mode is what this server is, from AITK_MODE (docs/gateway-design.md
// §1). It decides which UIs, which MCP endpoint and which listen address the
// server has; nothing else turns those on or off.
package mode

import (
	"fmt"
	"os"
	"strings"
	"sync/atomic"
)

type Mode string

const (
	// Worker: trainer only, reached only through its tunnel to the studio API
	// at AITK_STUDIO_API_URL. No console, studio or /mcp; listens on loopback.
	Worker Mode = "worker"
	// Standalone: an ordinary AI Toolkit server: console at /, /v1, /mcp.
	Standalone Mode = "standalone"
	// Studio: the studio and the trainer on one node; no console.
	Studio Mode = "studio"
	// StudioAPI: the studio without a trainer, managing GPU servers.
	StudioAPI Mode = "studio-api"
)

var current atomic.Value // Mode

// image is "gpu" in the GPU image's binary (ui-go/docker/Dockerfile sets it
// with -ldflags -X). That image refuses studio-api: the studio API is its own
// image, ghcr.io/geocine/aitoolkit-studio-api.
var image string

// Current is the mode Set chose (Standalone before that).
func Current() Mode {
	if m, ok := current.Load().(Mode); ok {
		return m
	}
	return Standalone
}

// Set records the mode for Current.
func Set(m Mode) { current.Store(m) }

// FromEnv reads AITK_MODE. studioAPI is the older switch (--studio-api or
// AITK_STUDIO_API=1), an alias for studio-api. Unset is standalone; the
// images set their own default (worker for the GPU image, studio-api for the
// slim one).
func FromEnv(studioAPI bool) (Mode, error) {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("AITK_MODE")))
	if raw == "" {
		if studioAPI {
			return StudioAPI, checkImage(StudioAPI)
		}
		return Standalone, nil
	}
	m := Mode(raw)
	switch m {
	case Worker, Standalone, Studio, StudioAPI:
	default:
		return "", fmt.Errorf("AITK_MODE=%s isn't a mode: use worker, standalone, studio or studio-api", raw)
	}
	if studioAPI && m != StudioAPI {
		return "", fmt.Errorf("AITK_MODE=%s conflicts with --studio-api / AITK_STUDIO_API: drop one", raw)
	}
	return m, checkImage(m)
}

func checkImage(m Mode) error {
	if m == StudioAPI && image == "gpu" {
		return fmt.Errorf("this is the GPU image: the studio API runs in its own image, ghcr.io/geocine/aitoolkit-studio-api. Here use AITK_MODE=worker, standalone or studio")
	}
	return nil
}

// Console reports the console (ui-go/web) is served at /.
func (m Mode) Console() bool { return m == Standalone }

// StudioUI reports the studio is served at /studio/.
func (m Mode) StudioUI() bool { return m == Studio || m == StudioAPI }

// MCP reports this server answers /mcp itself.
func (m Mode) MCP() bool { return m != Worker }

// Trainer reports this server runs jobs.
func (m Mode) Trainer() bool { return m != StudioAPI }
