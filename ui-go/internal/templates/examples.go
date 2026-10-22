package templates

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"go.yaml.in/yaml/v3"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Example is one of upstream's example configs (config/examples).
type Example struct {
	Name    string `json:"name"` // the file name without extension
	File    string `json:"file"` // relative to the toolkit root
	Arch    string `json:"arch,omitempty"`
	Model   string `json:"model,omitempty"`
	Network string `json:"network,omitempty"`
	Error   string `json:"error,omitempty"` // the file does not parse
}

// ExampleConfig is GET /v1/templates/examples/{name}: the file as written and
// parsed.
type ExampleConfig struct {
	Example
	YAML   string `json:"yaml"`
	Config any    `json:"config"`
}

func examplesDir() string { return filepath.Join(paths.ToolkitRoot, "config", "examples") }

// Examples lists config/examples (no subfolders), by name.
func Examples() ([]Example, error) {
	entries, err := os.ReadDir(examplesDir())
	if os.IsNotExist(err) {
		return []Example{}, nil
	}
	if err != nil {
		return nil, err
	}
	out := []Example{}
	for _, e := range entries {
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if e.IsDir() || ext != ".yaml" && ext != ".yml" {
			continue
		}
		ex, _, _ := readExample(e.Name())
		out = append(out, ex)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// GetExample reads one example by the name Examples lists.
func GetExample(name string) (ExampleConfig, error) {
	list, err := Examples()
	if err != nil {
		return ExampleConfig{}, err
	}
	for _, e := range list {
		if e.Name == name {
			ex, text, cfg := readExample(filepath.Base(e.File))
			return ExampleConfig{Example: ex, YAML: text, Config: cfg}, nil
		}
	}
	names := make([]string, 0, len(list))
	for _, e := range list {
		names = append(names, e.Name)
	}
	return ExampleConfig{}, v1api.Errorf(v1api.CodeNotFound, fmt.Sprintf("no example %q; examples: %s", name, strings.Join(names, ", ")))
}

func readExample(file string) (Example, string, any) {
	ex := Example{Name: strings.TrimSuffix(file, filepath.Ext(file)), File: filepath.ToSlash(filepath.Join("config", "examples", file))}
	b, err := os.ReadFile(filepath.Join(examplesDir(), file))
	if err != nil {
		return ex, "", nil
	}
	var cfg any
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		ex.Error = err.Error()
		return ex, string(b), nil
	}
	top, _ := cfg.(map[string]any)
	if conf, ok := top["config"].(map[string]any); ok {
		if procs, ok := conf["process"].([]any); ok && len(procs) > 0 {
			p, _ := procs[0].(map[string]any)
			model, _ := p["model"].(map[string]any)
			network, _ := p["network"].(map[string]any)
			ex.Arch, _ = model["arch"].(string)
			ex.Model, _ = model["name_or_path"].(string)
			ex.Network, _ = network["type"].(string)
		}
	}
	return ex, string(b), cfg
}
