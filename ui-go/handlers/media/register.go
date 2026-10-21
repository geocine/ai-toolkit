// Package media ports the media and file routes: serving dataset/training
// files (ui/cron/fileServer.ts plus the Next routes under api/img, api/files,
// api/audio/art), captions, image upload/delete, file delete and zips.
package media

import (
	"net/http"
	"strings"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/settings"
)

// roots is swappable so tests need no database.
var roots = settings.Roots

func datasetsRoot(r *http.Request) (string, error) {
	rt, err := roots(r.Context())
	return rt.Datasets, err
}

func Register(mux *http.ServeMux) {
	// GET patterns also answer HEAD, like the front file server
	httpx.Handle(mux, "GET /api/img/{imagePath...}", func(w http.ResponseWriter, r *http.Request) { serveFile(w, r, prefixImg) })
	httpx.Handle(mux, "GET /api/files/{filePath...}", func(w http.ResponseWriter, r *http.Request) { serveFile(w, r, prefixFiles) })
	httpx.Handle(mux, "GET /api/audio/art/{audioPath...}", func(w http.ResponseWriter, r *http.Request) { serveFile(w, r, prefixArt) })
	httpx.Handle(mux, "POST /api/img/caption", imgCaption)
	httpx.Handle(mux, "POST /api/img/delete", imgDelete)
	httpx.Handle(mux, "POST /api/img/upload", imgUpload)
	httpx.Handle(mux, "POST /api/files/delete", filesDelete)
	httpx.Handle(mux, "POST /api/zip", zipRoute)
	httpx.Handle(mux, "POST /api/caption/get", captionGet)
	httpx.Handle(mux, "POST /api/caption/getBatch", captionGetBatch)
}

// Intercept serves GET/HEAD file requests before ServeMux sees them, like the
// front file server does. ServeMux cleans paths and redirects a literal `//`
// (what a proxy that decodes %2F forwards for /api/img/%2Fapp%2F...) to a
// double-escaped Location; serving here keeps those requests working.
func Intercept(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet || r.Method == http.MethodHead {
			p := r.URL.EscapedPath()
			for _, prefix := range []string{prefixFiles, prefixImg, prefixArt} {
				if strings.HasPrefix(p, prefix) {
					serveFile(w, r, prefix)
					return
				}
			}
		}
		next.ServeHTTP(w, r)
	})
}
