// Package settings ports ui/src/app/api/settings/route.ts and api/auth/route.ts.
package settings

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/paths"
	cfg "github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /api/settings", get)
	httpx.Handle(mux, "POST /api/settings", post)
	// reaching this handler means the auth middleware already let it through
	httpx.Handle(mux, "GET /api/auth", func(w http.ResponseWriter, r *http.Request) {
		httpx.OK(w, map[string]bool{"isAuthenticated": true})
	})
}

func get(w http.ResponseWriter, r *http.Request) {
	rows, err := store.ListSettings(r.Context())
	if err != nil {
		httpx.Error(w, http.StatusInternalServerError, "Failed to fetch settings")
		return
	}
	out := map[string]string{}
	for _, s := range rows {
		out[s.Key] = s.Value
	}
	if out["TRAINING_FOLDER"] == "" {
		out["TRAINING_FOLDER"] = paths.DefaultTrainFolder
	}
	if out["DATASETS_FOLDER"] == "" {
		out["DATASETS_FOLDER"] = paths.DefaultDatasetsFolder
	}
	// MODELS_PATH from the env always takes precedence over the setting
	if env := os.Getenv("MODELS_PATH"); strings.TrimSpace(env) != "" {
		out["MODELS_PATH"] = env
	} else if out["MODELS_PATH"] == "" {
		out["MODELS_PATH"] = paths.DefaultModelsFolder
	}
	httpx.OK(w, out)
}

var postKeys = []string{"HF_TOKEN", "TRAINING_FOLDER", "DATASETS_FOLDER", "MODELS_PATH"}

// post upserts the four settings. Like Prisma's upsert, a key missing from the
// body leaves an existing row alone but cannot create one, and a non-string
// value is rejected; either fails the request after the other writes ran.
func post(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var body map[string]json.RawMessage
	if err := httpx.DecodeJSON(r, &body); err != nil {
		httpx.Error(w, http.StatusInternalServerError, "Failed to update settings")
		return
	}
	var failed error
	for _, key := range postKeys {
		raw, present := body[key]
		if !present {
			if _, exists, err := store.GetSetting(ctx, key); err != nil || !exists {
				failed = errors.Join(failed, errors.New(key+" is required"))
			}
			continue
		}
		var v string
		if err := json.Unmarshal(raw, &v); err != nil || string(raw) == "null" {
			failed = errors.Join(failed, errors.New(key+" must be a string"))
			continue
		}
		if err := store.UpsertSetting(ctx, key, v); err != nil {
			failed = errors.Join(failed, err)
		}
	}
	if failed != nil {
		log.Printf("POST /api/settings: %v", failed)
		httpx.Error(w, http.StatusInternalServerError, "Failed to update settings")
		return
	}
	cfg.Flush()
	httpx.OK(w, map[string]bool{"success": true})
}
