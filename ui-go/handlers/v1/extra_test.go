package v1

import (
	"net/http"
	"slices"
	"testing"

	"github.com/geocine/aitoolkit/internal/httpx"
)

// a route a feature file adds with addRoutes is served by Register
func TestExtraRoutes(t *testing.T) {
	saved := slices.Clone(extraRoutes)
	t.Cleanup(func() { extraRoutes = saved })
	addRoutes(func(mux *http.ServeMux) {
		// plain HandleFunc keeps the test route out of httpx.Routes()
		mux.HandleFunc("GET /v1/extra-test/{id}", func(w http.ResponseWriter, r *http.Request) {
			httpx.OK(w, map[string]string{"id": r.PathValue("id")})
		})
	})
	e := setup(t)
	r := e.call(t, "GET", "/v1/extra-test/x1", nil)
	want(t, r, 200, "")
	if m := r.json(t); m["id"] != "x1" {
		t.Fatalf("extra route: %v", m)
	}
	// the built-in routes are still there
	want(t, e.call(t, "GET", "/v1/health", nil), 200, "")
}
