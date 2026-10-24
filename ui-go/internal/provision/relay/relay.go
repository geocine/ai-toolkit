// Package relay reverse-proxies /v1/machines/{id}/relay/{path...} to a machine.
package relay

import (
	"net/http"

	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Serve proxies r to t.BaseURL + "/" + inner. authenticated reports whether
// the caller passed the controller's Auth (httpx.Authenticated).
func Serve(w http.ResponseWriter, r *http.Request, t provision.RelayTarget, inner string, authenticated bool) {
	v1api.Fail(w, v1api.CodeNotFound, "relay not available in this build")
}
