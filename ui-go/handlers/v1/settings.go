package v1

import (
	"context"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strconv"

	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

type settingsView struct {
	TrainingFolder        string `json:"training_folder"`
	DatasetsFolder        string `json:"datasets_folder"`
	DataRoot              string `json:"data_root"`
	ModelsPath            string `json:"models_path"`
	HFTokenSet            bool   `json:"hf_token_set"`
	AutoResumeDefault     bool   `json:"auto_resume_default"`
	MaxAutoResumesDefault int    `json:"max_auto_resumes_default"`
}

func currentSettings(ctx context.Context) (settingsView, error) {
	var v settingsView
	var err error
	if v.TrainingFolder, err = settings.TrainingFolder(ctx); err != nil {
		return v, err
	}
	if v.DatasetsFolder, err = settings.DatasetsRoot(ctx); err != nil {
		return v, err
	}
	if v.DataRoot, err = settings.DataRoot(ctx); err != nil {
		return v, err
	}
	if v.ModelsPath, err = settings.ModelsPath(ctx); err != nil {
		return v, err
	}
	if abs, err := filepath.Abs(v.ModelsPath); err == nil {
		v.ModelsPath = abs
	}
	tok, err := settings.HFToken(ctx)
	if err != nil {
		return v, err
	}
	v.HFTokenSet = tok != ""
	v.AutoResumeDefault, v.MaxAutoResumesDefault = false, 3
	if s, ok, _ := state.Setting(ctx, state.SettingAutoResumeDefault); ok {
		v.AutoResumeDefault = s == "true"
	}
	if s, ok, _ := state.Setting(ctx, state.SettingMaxAutoResumesDefault); ok {
		if n, err := strconv.Atoi(s); err == nil {
			v.MaxAutoResumesDefault = n
		}
	}
	return v, nil
}

func getSettings(w http.ResponseWriter, r *http.Request) {
	v, err := currentSettings(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, v)
}

// upstreamKeys maps /v1 setting names to upstream's Settings rows.
var upstreamKeys = map[string]string{
	"training_folder": "TRAINING_FOLDER",
	"datasets_folder": "DATASETS_FOLDER",
	"data_root":       "DATA_ROOT",
	"models_path":     "MODELS_PATH",
	"hf_token":        "HF_TOKEN",
}

func patchSettings(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var body map[string]json.RawMessage
	if !v1api.Decode(w, r, &body) {
		return
	}
	// validate everything before writing anything
	upstream := map[string]string{}
	fork := map[string]string{}
	for k, raw := range body {
		if row, ok := upstreamKeys[k]; ok {
			var s string
			if err := json.Unmarshal(raw, &s); err != nil {
				v1api.Fail(w, v1api.CodeInvalidRequest, k+" must be a string")
				return
			}
			upstream[row] = s
			continue
		}
		switch k {
		case "auto_resume_default":
			var b bool
			if err := json.Unmarshal(raw, &b); err != nil {
				v1api.Fail(w, v1api.CodeInvalidRequest, k+" must be a boolean")
				return
			}
			fork[state.SettingAutoResumeDefault] = strconv.FormatBool(b)
		case "max_auto_resumes_default":
			var n int
			if err := json.Unmarshal(raw, &n); err != nil || n < 0 {
				v1api.Fail(w, v1api.CodeInvalidRequest, k+" must be a non-negative integer")
				return
			}
			fork[state.SettingMaxAutoResumesDefault] = strconv.Itoa(n)
		default:
			v1api.Fail(w, v1api.CodeInvalidRequest, "unknown setting "+k)
			return
		}
	}
	for row, v := range upstream {
		if err := store.UpsertSetting(ctx, row, v); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	for k, v := range fork {
		if err := state.SetSetting(ctx, k, v); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	settings.Flush()
	getSettings(w, r)
}
