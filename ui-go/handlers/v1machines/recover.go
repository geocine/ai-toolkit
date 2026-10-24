package v1machines

import (
	"net/http"

	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Recovery (PLAN §12): the aitk-labelled instances on a provider account and
// adopting one this server does not track. The service implements it as
// provision.Recoverer, asserted here.

func recoverer(w http.ResponseWriter) (provision.Recoverer, bool) {
	rec, ok := provision.Default.(provision.Recoverer)
	if !ok {
		v1api.Fail(w, v1api.CodeNotFound, "this server can't look up provider instances")
	}
	return rec, ok
}

func instances(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok {
		return
	}
	rec, ok := recoverer(w)
	if !ok {
		return
	}
	items, err := rec.Instances(r.Context(), k)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.Items(w, items)
}

// adopt hands this server a machine token and starts watching a billing
// machine, so the open-server guard applies.
func adopt(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok || !guard(w, r) {
		return
	}
	rec, ok := recoverer(w)
	if !ok {
		return
	}
	var in provision.AdoptRequest
	if !v1api.Decode(w, r, &in) {
		return
	}
	m, err := rec.Adopt(r.Context(), k, in)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.Created(w, m)
}
