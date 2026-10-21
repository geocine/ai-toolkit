// Package v1templates serves /v1/templates: job configs built from upstream's
// own UI definitions (internal/templates, docs/headless.md Templates).
package v1templates

import (
	"net/http"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/templates"
	"github.com/geocine/aitoolkit/internal/v1api"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /v1/templates", list)
	httpx.Handle(mux, "POST /v1/templates/render", render)
}

func list(w http.ResponseWriter, r *http.Request) {
	c, err := templates.List(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, c)
}

func render(w http.ResponseWriter, r *http.Request) {
	var req templates.RenderRequest
	if !v1api.Decode(w, r, &req) {
		return
	}
	ctx := r.Context()
	datasets, err := settings.DatasetsRoot(ctx)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	training, err := settings.TrainingFolder(ctx)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	res, err := templates.Render(ctx, req, templates.Options{DatasetsRoot: datasets, TrainingFolder: training})
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	v1api.OK(w, res)
}
