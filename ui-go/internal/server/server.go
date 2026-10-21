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
)

// NewMux registers every API route.
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
	return mux
}
