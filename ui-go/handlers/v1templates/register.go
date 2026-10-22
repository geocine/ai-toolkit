// Package v1templates serves /v1/templates: job configs built from upstream's
// own UI definitions, the keys the trainer reads, a check for keys it would
// ignore, and upstream's example configs (internal/templates,
// docs/headless.md Templates).
package v1templates

import (
	"net/http"
	"strings"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/templates"
	"github.com/geocine/aitoolkit/internal/v1api"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /v1/templates", list)
	httpx.Handle(mux, "POST /v1/templates/render", render)
	httpx.Handle(mux, "GET /v1/templates/fields", fields)
	httpx.Handle(mux, "POST /v1/templates/check", check)
	httpx.Handle(mux, "GET /v1/templates/examples", examples)
	httpx.Handle(mux, "GET /v1/templates/examples/{name}", example)
	httpx.Handle(mux, "GET /v1/templates/captioners", captioners)
}

func captioners(w http.ResponseWriter, r *http.Request) {
	c, err := templates.Captioners(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, c)
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

// fields is the whole reference, or with ?section= that section and the ones
// nested in it (section=train also gives train.ema_config).
func fields(w http.ResponseWriter, r *http.Request) {
	ref := templates.Fields()
	if s := r.URL.Query().Get("section"); s != "" {
		keep := ref.Sections[:0:0]
		for _, sec := range ref.Sections {
			if sec.Path == s || strings.HasPrefix(sec.Path, s+".") || strings.HasPrefix(sec.Path, s+"[]") {
				keep = append(keep, sec)
			}
		}
		if len(keep) == 0 {
			v1api.Fail(w, v1api.CodeNotFound, "no section "+s+"; sections: "+sectionNames(ref))
			return
		}
		ref.Sections = keep
	}
	v1api.OK(w, ref)
}

func sectionNames(ref templates.Reference) string {
	names := make([]string, 0, len(ref.Sections))
	for _, s := range ref.Sections {
		names = append(names, s.Path)
	}
	return strings.Join(names, ", ")
}

func check(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Config any `json:"config"`
	}
	if !v1api.Decode(w, r, &req) {
		return
	}
	if req.Config == nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "config is required")
		return
	}
	v1api.OK(w, templates.Check(req.Config))
}

func examples(w http.ResponseWriter, r *http.Request) {
	items, err := templates.Examples()
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func example(w http.ResponseWriter, r *http.Request) {
	ex, err := templates.GetExample(r.PathValue("name"))
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	v1api.OK(w, ex)
}
