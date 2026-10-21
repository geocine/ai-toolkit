package templates

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// the tests read this repo's real upstream ui/ and extensions
func useRepo(t *testing.T) {
	t.Helper()
	root, _ := filepath.Abs(filepath.Join("..", "..", ".."))
	if _, err := os.Stat(filepath.Join(root, "ui", "src", "app", "jobs", "new", "jobConfig.ts")); err != nil {
		t.Skipf("upstream ui not found: %v", err)
	}
	paths.ToolkitRoot = root
	cacheMu.Lock()
	cached = nil
	cacheMu.Unlock()
}

func get(t *testing.T, v any, path ...any) any {
	t.Helper()
	cur := v
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, ok := cur.(map[string]any)
			if !ok {
				t.Fatalf("at %v: not an object (%T)", path, cur)
			}
			cur = m[k]
		case int:
			a, ok := cur.([]any)
			if !ok || k >= len(a) {
				t.Fatalf("at %v: not an array long enough", path)
			}
			cur = a[k]
		}
	}
	return cur
}

func archByName(c Catalog, name string) *Arch {
	for i := range c.Archs {
		if c.Archs[i].Name == name {
			return &c.Archs[i]
		}
	}
	return nil
}

func TestListFromUpstream(t *testing.T) {
	useRepo(t)
	c, err := List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Errors) > 0 {
		t.Fatalf("module errors: %v", c.Errors)
	}
	if len(c.Archs) < 20 {
		t.Fatalf("only %d archs", len(c.Archs))
	}
	for _, name := range []string{"flex1", "sdxl", "sd15", "flux"} {
		if archByName(c, name) == nil {
			t.Errorf("arch %s missing", name)
		}
	}
	// default config straight from jobConfig.ts (migrated)
	if got := get(t, c.DefaultConfig, "config", "process", 0, "model", "arch"); got != "flex1" {
		t.Errorf("default arch = %v", got)
	}
	if got := get(t, c.DefaultConfig, "config", "process", 0, "type"); got != "diffusion_trainer" {
		t.Errorf("default type = %v", got)
	}
	if a := archByName(c, "flex1"); !a.PythonAccepted || a.PythonArch != "flux" {
		t.Errorf("flex1 python = %v %v", a.PythonArch, a.PythonAccepted)
	}
	if a := archByName(c, "sdxl"); !a.PythonAccepted {
		t.Error("sdxl should be accepted")
	}
	if a := archByName(c, "sd15"); a.PythonAccepted {
		t.Error("sd15 is not a Python arch (sd1 is)")
	}
	var rejected []string
	for _, a := range c.Archs {
		if !a.PythonAccepted {
			rejected = append(rejected, a.Name)
		}
	}
	t.Logf("%d archs; not accepted by Python: %v", len(c.Archs), rejected)
}

func TestRenderSDXL(t *testing.T) {
	useRepo(t)
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "my_set"), 0o755)
	res, err := Render(context.Background(), RenderRequest{
		Arch: "sdxl", Name: "my_lora", Datasets: []string{"my_set", "second"},
		Overrides: map[string]any{"config.process[0].train.steps": 500, "config.process[0].sample.samples[0].prompt": "x"},
	}, Options{DatasetsRoot: dir, TrainingFolder: "/out"})
	if err != nil {
		t.Fatal(err)
	}
	cfg := res.Config
	p0 := []any{"config", "process", 0}
	at := func(rest ...any) any { return get(t, cfg, append(append([]any{}, p0...), rest...)...) }
	if at("model", "arch") != "sdxl" {
		t.Errorf("arch = %v", at("model", "arch"))
	}
	// sdxl's defaults from extensions_built_in/diffusion_models/ui.tsx
	c, _ := List(context.Background())
	for key, pair := range archByName(c, "sdxl").Defaults {
		if strings.Contains(key, "datasets[x]") || strings.Contains(key, "samples") {
			continue
		}
		want := pair.([]any)[0]
		path := strings.TrimPrefix(key, "config.process[0].")
		var parts []any
		for _, seg := range strings.Split(path, ".") {
			parts = append(parts, seg)
		}
		if key == "config.process[0].train.steps" {
			want = float64(500)
		}
		got := at(parts...)
		gb, _ := json.Marshal(got)
		wb, _ := json.Marshal(want)
		if string(gb) != string(wb) {
			t.Errorf("%s = %s, want %s", key, gb, wb)
		}
	}
	if at("train", "steps") != float64(500) {
		t.Errorf("steps override = %v", at("train", "steps"))
	}
	if at("training_folder") != "/out" {
		t.Errorf("training_folder = %v", at("training_folder"))
	}
	if got := get(t, cfg, "config", "name"); got != "my_lora" {
		t.Errorf("name = %v", got)
	}
	ds := at("datasets").([]any)
	if len(ds) != 2 || ds[0].(map[string]any)["folder_path"] != filepath.Join(dir, "my_set") ||
		ds[1].(map[string]any)["folder_path"] != filepath.Join(dir, "second") {
		t.Errorf("datasets = %v", ds)
	}
	if at("sample", "samples", 0, "prompt") != "x" {
		t.Error("sample prompt override not applied")
	}
	// the missing second dataset is a warning, not an error
	if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0], `"second"`) {
		t.Errorf("warnings = %v", res.Warnings)
	}
}

func TestRenderFluxAndDefaultArch(t *testing.T) {
	useRepo(t)
	dir := t.TempDir()
	for _, arch := range []string{"flux", "flex1", ""} {
		res, err := Render(context.Background(), RenderRequest{Arch: arch, Name: "n", Datasets: []string{"d"}},
			Options{DatasetsRoot: dir})
		if err != nil {
			t.Fatalf("%q: %v", arch, err)
		}
		want := arch
		if want == "" {
			want = "flex1"
		}
		if got := get(t, res.Config, "config", "process", 0, "model", "arch"); got != want {
			t.Errorf("%q: arch = %v", arch, got)
		}
		for _, w := range res.Warnings {
			if strings.Contains(w, "Python") {
				t.Errorf("%q: unexpected python warning %s", arch, w)
			}
		}
	}
}

func TestRenderSD15Warns(t *testing.T) {
	useRepo(t)
	res, err := Render(context.Background(), RenderRequest{Arch: "sd15", Datasets: []string{"x"}}, Options{DatasetsRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range res.Warnings {
		if strings.Contains(w, `"sd15"`) && strings.Contains(w, `"sd1"`) {
			found = true
		}
	}
	if !found {
		t.Fatalf("no sd15 -> sd1 warning: %v", res.Warnings)
	}
	// upstream's own value is kept, not silently changed
	if got := get(t, res.Config, "config", "process", 0, "model", "arch"); got != "sd15" {
		t.Errorf("arch = %v", got)
	}
}

func TestRenderRejects(t *testing.T) {
	useRepo(t)
	opt := Options{DatasetsRoot: t.TempDir()}
	cases := []RenderRequest{
		{Arch: "nope"},
		{Arch: "sdxl", Overrides: map[string]any{"config.process[0]].x": 1}},
		{Arch: "sdxl", Datasets: []string{"../etc"}},
	}
	for _, c := range cases {
		_, err := Render(context.Background(), c, opt)
		var p *v1api.Problem
		if !errors.As(err, &p) || p.Code != v1api.CodeInvalidRequest {
			t.Errorf("%+v: err = %v", c, err)
		}
	}
}
