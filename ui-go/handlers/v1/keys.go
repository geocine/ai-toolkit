package v1

import (
	"net/http"
	"strings"

	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/v1api"
)

func listKeys(w http.ResponseWriter, r *http.Request) {
	keys, err := state.ListKeys(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Items(w, keys)
}

// createKey returns the secret once; only its hash is stored.
func createKey(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Name string `json:"name"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	b.Name = strings.TrimSpace(b.Name)
	if b.Name == "" || len(b.Name) > 100 {
		v1api.Fail(w, v1api.CodeInvalidRequest, "name is required (at most 100 characters)")
		return
	}
	k, secret, err := state.CreateKey(r.Context(), b.Name)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.Created(w, map[string]any{
		"id": k.ID, "name": k.Name, "prefix": k.Prefix, "created_at": k.CreatedAt, "key": secret,
	})
}

func deleteKey(w http.ResponseWriter, r *http.Request) {
	ok, err := state.RevokeKey(r.Context(), r.PathValue("id"))
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if !ok {
		v1api.Fail(w, v1api.CodeNotFound, "key not found")
		return
	}
	v1api.NoContent(w)
}
