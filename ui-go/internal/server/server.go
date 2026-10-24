// Package server assembles the API: every handler package's routes on one mux.
package server

import (
	"net/http"

	"github.com/geocine/aitoolkit/handlers/datasets"
	"github.com/geocine/aitoolkit/handlers/inference"
	"github.com/geocine/aitoolkit/handlers/jobfiles"
	"github.com/geocine/aitoolkit/handlers/jobs"
	"github.com/geocine/aitoolkit/handlers/media"
	"github.com/geocine/aitoolkit/handlers/queue"
	"github.com/geocine/aitoolkit/handlers/settings"
	"github.com/geocine/aitoolkit/handlers/system"
	"github.com/geocine/aitoolkit/handlers/v1"
	"github.com/geocine/aitoolkit/handlers/v1images"
	"github.com/geocine/aitoolkit/handlers/v1machines"
	"github.com/geocine/aitoolkit/handlers/v1portable"
	"github.com/geocine/aitoolkit/handlers/v1recommend"
	"github.com/geocine/aitoolkit/handlers/v1remotes"
	"github.com/geocine/aitoolkit/handlers/v1templates"
	"github.com/geocine/aitoolkit/handlers/v1unattended"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/mcpserver"
)

// NewMux registers every route: the upstream-compatible /api (parity-checked
// against upstream), then the headless /v1 and /mcp (docs/headless.md).
func NewMux() *http.ServeMux {
	mux := http.NewServeMux()
	jobs.Register(mux)
	jobfiles.Register(mux)
	queue.Register(mux)
	settings.Register(mux)
	media.Register(mux)
	datasets.Register(mux)
	system.Register(mux)
	inference.Register(mux)

	v1.Register(mux)
	v1remotes.Register(mux)
	v1portable.Register(mux)
	v1templates.Register(mux)
	v1unattended.Register(mux)
	v1machines.Register(mux)
	v1images.Register(mux)
	v1recommend.Register(mux)
	events.Register(mux)
	mcpserver.Register(mux)
	return mux
}
