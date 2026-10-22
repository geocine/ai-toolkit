package v1

import (
	"errors"
	"net/http"
	"os"
	"time"

	"github.com/geocine/aitoolkit/internal/links"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// createLink signs a download URL: GET of exactly that path works without a
// bearer token until expires_at (internal/links).
func createLink(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Path string `json:"path"`
		TTLS *int64 `json:"ttl_s"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	if b.Path == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "path is required, e.g. /v1/files/content?root=training&path=job/file.safetensors")
		return
	}
	ttl := links.DefaultTTL
	if b.TTLS != nil {
		ttl = time.Duration(*b.TTLS) * time.Second
		if ttl < links.MinTTL || ttl > links.MaxTTL {
			v1api.Fail(w, v1api.CodeInvalidRequest, "ttl_s must be between 60 and 604800")
			return
		}
	}
	signed, exp, err := links.Sign(r.Context(), b.Path, ttl)
	if errors.Is(err, links.ErrTarget) || errors.Is(err, links.ErrPath) {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, map[string]any{"url": links.Origin(r, os.Getenv("AITK_PUBLIC_URL")) + signed, "path": signed,
		"expires_at": v1api.Time(exp)})
}
