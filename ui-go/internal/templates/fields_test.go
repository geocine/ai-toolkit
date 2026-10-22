package templates

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/v1api"
)

func section(t *testing.T, ref Reference, path string) map[string]Field {
	t.Helper()
	for _, s := range ref.Sections {
		if s.Path == path {
			out := map[string]Field{}
			for _, f := range s.Fields {
				out[f.Key] = f
			}
			return out
		}
	}
	t.Fatalf("no section %s", path)
	return nil
}

func TestFieldsFromToolkit(t *testing.T) {
	useRepo(t)
	ref := Fields()
	if len(ref.Errors) != 0 {
		t.Fatalf("errors: %v", ref.Errors)
	}
	train := section(t, ref, "train")
	if f := train["loss_type"]; f.Default != "'mse'" || !strings.Contains(f.Doc, "mae") {
		t.Errorf("train.loss_type = %+v", f)
	}
	if f := train["steps"]; f.Default != "1000" || f.Type != "int" {
		t.Errorf("train.steps = %+v", f)
	}
	if _, ok := train["use_ema"]; ok {
		t.Error("use_ema is read from train.ema_config, not train")
	}
	if _, ok := section(t, ref, "train.ema_config")["ema_decay"]; !ok {
		t.Error("train.ema_config lacks ema_decay")
	}
	// a default spread over two lines, with a comment inside the call
	if f := section(t, ref, "sample")["refiner_start_at"]; f.Default != "0.5" || !strings.Contains(f.Doc, "refiner") {
		t.Errorf("sample.refiner_start_at = %+v", f)
	}
	proc := section(t, ref, "process")
	for _, k := range []string{"trigger_word", "training_folder", "sqlite_db_path", "type"} {
		if _, ok := proc[k]; !ok {
			t.Errorf("process lacks %s", k)
		}
	}
	if _, ok := proc["train"]; ok {
		t.Error("sections are not process fields")
	}
	if _, ok := section(t, ref, "datasets[]")["folder_path"]; !ok {
		t.Error("datasets[] lacks folder_path")
	}
}

// the two mistakes in the user's Krea configs, plus a typo in a list item
func TestCheckFindsIgnoredKeys(t *testing.T) {
	useRepo(t)
	cfg := map[string]any{"job": "extension", "config": map[string]any{"name": "x", "process": []any{map[string]any{
		"type": "diffusion_trainer", "trigger_word": "jihan", "loss_type": "mse",
		"train":    map[string]any{"steps": 3000, "ema_config": nil, "use_ema": true, "ma_decay": 0.99},
		"model":    map[string]any{"arch": "krea2", "name_or_path": "krea/Krea-2-Raw"},
		"datasets": []any{map[string]any{"folder_path": "/d", "caption_ext": "txt"}},
		"sample":   map[string]any{"samples": []any{map[string]any{"promt": "a cat"}}},
	}}}}
	got := map[string]string{}
	for _, p := range Check(cfg).Problems {
		got[p.Path] = p.Fix
	}
	want := map[string]string{
		"config.process[0].loss_type":               "it is read in train",
		"config.process[0].train.use_ema":           "it is read in train.ema_config",
		"config.process[0].train.ma_decay":          "did you mean train.ema_config.ema_decay?",
		"config.process[0].sample.samples[0].promt": "did you mean prompt?",
	}
	if len(got) != len(want) {
		t.Errorf("problems = %v", got)
	}
	for k, fix := range want {
		if got[k] != fix {
			t.Errorf("%s: fix = %q, want %q", k, got[k], fix)
		}
	}

	// fixed, and a process that is not a trainer, have nothing to report
	procs := cfg["config"].(map[string]any)["process"].([]any)
	p := procs[0].(map[string]any)
	tr := p["train"].(map[string]any)
	tr["loss_type"] = p["loss_type"]
	delete(p, "loss_type")
	tr["ema_config"] = map[string]any{"use_ema": true, "ema_decay": 0.99}
	delete(tr, "use_ema")
	delete(tr, "ma_decay")
	p["sample"] = map[string]any{"samples": []any{map[string]any{"prompt": "a cat"}}}
	cfg["config"].(map[string]any)["process"] = append(procs, map[string]any{"type": "to_folder", "anything": 1})
	if res := Check(cfg); len(res.Problems) != 0 {
		t.Errorf("fixed config: %+v", res.Problems)
	}
	if res := Check(map[string]any{"job": "extension"}); len(res.Problems) != 1 || res.Problems[0].Path != "config.process" {
		t.Errorf("no process: %+v", res.Problems)
	}
}

func TestRenderChecksOverrides(t *testing.T) {
	useRepo(t)
	res, err := Render(context.Background(), RenderRequest{Arch: "sdxl", Datasets: []string{"x"},
		Overrides: map[string]any{"config.process[0].loss_type": "mae"}}, Options{DatasetsRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(res.Warnings, "\n"), "config.process[0].loss_type: loss_type is not read at the process level") {
		t.Fatalf("warnings = %v", res.Warnings)
	}
}

func TestExamples(t *testing.T) {
	useRepo(t)
	list, err := Examples()
	if err != nil || len(list) == 0 {
		t.Fatalf("examples: %v %v", list, err)
	}
	var flux *Example
	for i := range list {
		if list[i].Name == "train_lora_flux_24gb" {
			flux = &list[i]
		}
	}
	if flux == nil || flux.Network != "lora" || flux.Model == "" {
		t.Fatalf("flux example = %+v", flux)
	}
	ex, err := GetExample(flux.Name)
	if err != nil || !strings.Contains(ex.YAML, "#") || ex.Config == nil || len(Check(ex.Config).Problems) != 0 {
		t.Fatalf("flux = %v %+v", err, Check(ex.Config).Problems)
	}
	var p *v1api.Problem
	if _, err := GetExample("../config_modules"); !errors.As(err, &p) || p.Code != v1api.CodeNotFound {
		t.Errorf("unknown example: %v", err)
	}
}
