package templates

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/dop251/goja"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Option is one choice of a select in upstream's caption dialog.
type Option struct {
	Value any    `json:"value"`
	Label string `json:"label"`
}

// Captioner is one captioner as ui/src/helpers/captionOptions.ts defines it.
// Defaults are the dotted config paths selecting it sets (its prompt, model,
// file extensions, ...).
type Captioner struct {
	Name               string            `json:"name"`
	Label              string            `json:"label"`
	Group              string            `json:"group"`
	Defaults           map[string]any    `json:"defaults"`
	ModelOptions       []Option          `json:"model_options"`
	Model2Options      []Option          `json:"model2_options"`
	CaptionPrompts     map[string]string `json:"caption_prompts"`
	AdditionalSections []string          `json:"additional_sections"`
	SupportsLoras      bool              `json:"supports_loras"`
	CloudLoras         []any             `json:"cloud_loras"`
	NoCompile          bool              `json:"no_compile"`
	MinNewTokens       *int              `json:"min_new_tokens"`
	MultiLinePrompts   bool              `json:"multi_line_prompts"`
	// PythonAccepted: extensions_built_in/captioner registers this name
	PythonAccepted bool `json:"python_accepted"`
}

// CaptionCatalog is GET /v1/templates/captioners.
type CaptionCatalog struct {
	Captioners    []Captioner         `json:"captioners"`
	DefaultConfig any                 `json:"default_config"`
	Options       map[string][]Option `json:"options"`
}

// CaptionRequest is what to caption with. Model, Prompt and Recaption are
// shorthands for caption.model_name_or_path, caption.caption_prompt and
// caption.recaption; Overrides keys are dotted paths, relative to
// config.process[0] unless they start with "config.".
type CaptionRequest struct {
	Captioner string         `json:"captioner"`
	Model     *string        `json:"model"`
	Prompt    *string        `json:"prompt"`
	Recaption *bool          `json:"recaption"`
	Overrides map[string]any `json:"overrides"`
}

var captionerUID = regexp.MustCompile(`(?m)^\s*uid\s*=\s*["']([A-Za-z0-9_]+)["']`)

// pythonCaptioners are the captioner process types the toolkit registers.
func pythonCaptioners() map[string]bool {
	out := map[string]bool{}
	b, err := os.ReadFile(filepath.Join(paths.ToolkitRoot, "extensions_built_in", "captioner", "__init__.py"))
	if err != nil {
		return out
	}
	for _, m := range captionerUID.FindAllStringSubmatch(string(b), -1) {
		out[m[1]] = true
	}
	return out
}

// Captioners is GET /v1/templates/captioners.
func Captioners(context.Context) (CaptionCatalog, error) {
	e, err := load()
	if err != nil {
		return CaptionCatalog{}, err
	}
	e.mu.Lock()
	fn, ok := goja.AssertFunction(e.vm.Get("__aitkCaptioners"))
	if !ok {
		e.mu.Unlock()
		return CaptionCatalog{}, fmt.Errorf("templates driver lacks captioners")
	}
	out, err := fn(goja.Undefined())
	e.mu.Unlock()
	if err != nil {
		return CaptionCatalog{}, fmt.Errorf("upstream captioners: %w", err)
	}
	var c CaptionCatalog
	if err := json.Unmarshal([]byte(out.String()), &c); err != nil {
		return CaptionCatalog{}, err
	}
	python := pythonCaptioners()
	for i := range c.Captioners {
		c.Captioners[i].PythonAccepted = python[c.Captioners[i].Name]
	}
	return c, nil
}

// RenderCaption builds a caption job config for the folder, the way the
// dataset page's caption dialog starts one.
func RenderCaption(ctx context.Context, req CaptionRequest, folder, name string) (RenderResult, error) {
	cat, err := Captioners(ctx)
	if err != nil {
		return RenderResult{}, err
	}
	warnings := []string{}
	if req.Captioner != "" {
		var found *Captioner
		names := make([]string, 0, len(cat.Captioners))
		for i := range cat.Captioners {
			names = append(names, cat.Captioners[i].Name)
			if cat.Captioners[i].Name == req.Captioner {
				found = &cat.Captioners[i]
			}
		}
		if found == nil {
			return RenderResult{}, v1api.Errorf(v1api.CodeInvalidRequest,
				fmt.Sprintf("unknown captioner %q; known: %s", req.Captioner, strings.Join(names, ", ")))
		}
		if !found.PythonAccepted {
			warnings = append(warnings, fmt.Sprintf("captioner %q is offered by the upstream UI but extensions_built_in/captioner does not register it", req.Captioner))
		}
	}
	overrides := map[string]any{}
	for k, v := range req.Overrides {
		if !strings.HasPrefix(k, "config.") {
			k = "config.process[0]." + k
		}
		if !overridePath.MatchString(k) {
			return RenderResult{}, v1api.Errorf(v1api.CodeInvalidRequest,
				fmt.Sprintf("override key %q is not a dotted path like caption.max_res", k))
		}
		overrides[k] = v
	}
	if req.Model != nil {
		overrides["config.process[0].caption.model_name_or_path"] = *req.Model
	}
	if req.Prompt != nil {
		overrides["config.process[0].caption.caption_prompt"] = *req.Prompt
	}
	if req.Recaption != nil {
		overrides["config.process[0].caption.recaption"] = *req.Recaption
	}

	e, err := load()
	if err != nil {
		return RenderResult{}, err
	}
	payload, _ := json.Marshal(map[string]any{"captioner": req.Captioner, "path": folder, "name": name, "overrides": overrides})
	e.mu.Lock()
	fn, _ := goja.AssertFunction(e.vm.Get("__aitkRenderCaption"))
	out, err := fn(goja.Undefined(), e.vm.ToValue(string(payload)))
	e.mu.Unlock()
	if err != nil {
		return RenderResult{}, v1api.Errorf(v1api.CodeInvalidConfig, "render failed: "+err.Error())
	}
	var res struct {
		Config  any    `json:"config"`
		Applied string `json:"captioner_applied"`
	}
	if err := json.Unmarshal([]byte(out.String()), &res); err != nil {
		return RenderResult{}, err
	}
	if req.Captioner != "" && res.Applied != req.Captioner {
		return RenderResult{}, v1api.Errorf(v1api.CodeInvalidConfig,
			fmt.Sprintf("upstream did not apply captioner %q (got %q)", req.Captioner, res.Applied))
	}
	return RenderResult{Config: res.Config, Warnings: warnings}, nil
}
