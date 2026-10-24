package v1

import (
	"net/http"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/photocheck"
)

// Photo checks (photos.go, photos_fix.go) join the /v1 routes through the
// extraRoutes seam. docs/headless.md, Datasets, "Photo checks" has the shapes.
func init() { addRoutes(registerPhotos) }

func registerPhotos(mux *http.ServeMux) {
	// the daily photo-cache GC registers with the routes, so it starts with
	// the other loops (bg.Start); controller mode skips it
	photocheck.StartGC()

	httpx.Handle(mux, "POST /v1/photos/assess", assessPhotos)
	httpx.Handle(mux, "GET /v1/datasets/{name}/photos/check", checkPhotos)
	httpx.Handle(mux, "POST /v1/datasets/{name}/photos/fix", fixPhotos)
	httpx.Handle(mux, "PUT /v1/datasets/{name}/photos/settings", putPhotoSettings)
	httpx.Handle(mux, "DELETE /v1/datasets/{name}/photos/analysis", forgetPhotos)
}
