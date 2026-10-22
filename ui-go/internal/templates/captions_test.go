package templates

import (
	"context"
	"errors"
	"testing"

	"github.com/geocine/aitoolkit/internal/v1api"
)

func TestCaptioners(t *testing.T) {
	useRepo(t)
	cat, err := Captioners(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var vl *Captioner
	for i := range cat.Captioners {
		if !cat.Captioners[i].PythonAccepted {
			t.Errorf("%s: not registered in extensions_built_in/captioner", cat.Captioners[i].Name)
		}
		if cat.Captioners[i].Name == "Qwen3VLCaptioner" {
			vl = &cat.Captioners[i]
		}
	}
	if vl == nil || len(vl.ModelOptions) == 0 || vl.Defaults["config.process[0].caption.caption_prompt"] == nil {
		t.Fatalf("Qwen3VLCaptioner = %+v", vl)
	}
	if len(cat.Options["quantization"]) == 0 || cat.DefaultConfig == nil {
		t.Errorf("options %v, default %v", cat.Options, cat.DefaultConfig)
	}
}

func TestRenderCaption(t *testing.T) {
	useRepo(t)
	ctx := context.Background()
	prompt := "a short caption"
	res, err := RenderCaption(ctx, CaptionRequest{Captioner: "Qwen3VLCaptioner", Prompt: &prompt,
		Overrides: map[string]any{"caption.max_new_tokens": 64}}, "/data/set", "set_caption")
	if err != nil {
		t.Fatal(err)
	}
	if v := get(t, res.Config, "config", "name"); v != "set_caption" {
		t.Errorf("name = %v", v)
	}
	if v := get(t, res.Config, "config", "process", 0, "type"); v != "Qwen3VLCaptioner" {
		t.Errorf("type = %v", v)
	}
	for k, want := range map[string]any{"path_to_caption": "/data/set", "caption_prompt": prompt, "max_new_tokens": float64(64)} {
		if v := get(t, res.Config, "config", "process", 0, "caption", k); v != want {
			t.Errorf("caption.%s = %v, want %v", k, v, want)
		}
	}

	// no captioner: upstream's default caption job
	res, err = RenderCaption(ctx, CaptionRequest{}, "/data/set", "")
	if err != nil || get(t, res.Config, "config", "process", 0, "type") != "Qwen3OmniCaptioner" {
		t.Fatalf("default: %v %v", err, res.Config)
	}

	var p *v1api.Problem
	if _, err := RenderCaption(ctx, CaptionRequest{Captioner: "nope"}, "/d", ""); !errors.As(err, &p) || p.Code != v1api.CodeInvalidRequest {
		t.Errorf("unknown captioner: %v", err)
	}
	if _, err := RenderCaption(ctx, CaptionRequest{Overrides: map[string]any{"caption.bad key": 1}}, "/d", ""); !errors.As(err, &p) {
		t.Errorf("bad override: %v", err)
	}
}
