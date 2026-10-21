// Package templates builds job configs from upstream's own UI definitions:
// ui/src/app/jobs/new/jobConfig.ts (the default job), utils.ts
// (handleModelArchChange, what selecting a model arch does) and the extension
// ui modules (internal/extui) that define the archs. The TypeScript is
// bundled with esbuild and evaluated in goja at runtime, so templates follow
// upstream after a rebase with no edits here. See docs/headless.md, Templates.
package templates

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dop251/goja"
	"github.com/evanw/esbuild/pkg/api"

	"github.com/geocine/aitoolkit/internal/extui"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/v1api"
)

//go:embed driver.js
var driverJS string

// entry pulls in exactly the upstream modules the new-job page uses.
const entry = `
import { defaultJobConfig, defaultDatasetConfig, migrateJobConfig } from '@/app/jobs/new/jobConfig';
import { handleModelArchChange } from '@/app/jobs/new/utils';
import { setNestedValue } from '@/utils/hooks';
import { objectCopy } from '@/utils/basic';
import * as defaultSamples from '@/helpers/defaultSamples';
import * as options from '@/app/jobs/new/options';
(globalThis as any).__aitkUpstream = {
  defaultJobConfig, defaultDatasetConfig, migrateJobConfig, handleModelArchChange,
  setNestedValue, objectCopy, defaultSamples, options,
};
`

// stubModule stands in for npm packages the bundle reaches (react, next,
// icon libraries): only data is needed, never rendering.
const stubModule = `
var h = {
  get: function (t, p) { if (p === '__esModule') return true; if (p === Symbol.toPrimitive) return function () { return ''; }; return s; },
  apply: function () { return s; }, construct: function () { return s; },
};
var s = new Proxy(function () {}, h);
module.exports = s;
`

// Arch is one model arch as the upstream UI defines it, plus whether the
// Python trainer accepts its name.
type Arch struct {
	Name               string         `json:"name"`
	Label              string         `json:"label"`
	Group              string         `json:"group"`
	Source             string         `json:"source"`
	Defaults           map[string]any `json:"defaults"`
	DisableSections    []string       `json:"disable_sections"`
	AdditionalSections []string       `json:"additional_sections"`
	Controls           []any          `json:"controls"`
	IsVideoModel       bool           `json:"is_video_model"`
	// PythonArch is the name after the trainer's own normalization
	// (toolkit/config_modules.py: ":tag" stripped, flex1 -> flux).
	PythonArch     string `json:"python_arch"`
	PythonAccepted bool   `json:"python_accepted"`
}

// Catalog is GET /v1/templates.
type Catalog struct {
	Archs         []Arch   `json:"archs"`
	DefaultConfig any      `json:"default_config"`
	Errors        []string `json:"errors"`
}

// RenderRequest is POST /v1/templates/render.
type RenderRequest struct {
	Arch      string         `json:"arch"`
	Name      string         `json:"name"`
	Datasets  []string       `json:"datasets"`
	Overrides map[string]any `json:"overrides"`
}

// RenderResult is the rendered config plus anything the caller should know.
type RenderResult struct {
	Config   any      `json:"config"`
	Warnings []string `json:"warnings"`
}

// Options are the environment a render runs in.
type Options struct {
	DatasetsRoot   string
	TrainingFolder string
}

type engine struct {
	mu      sync.Mutex
	vm      *goja.Runtime
	inputs  map[string]time.Time // bundle inputs -> mtime
	modules map[string]float64   // extension module id -> mtime
	catalog Catalog
	python  map[string]bool
}

var (
	cacheMu sync.Mutex
	cached  *engine
)

func uiSrc() string { return filepath.Join(paths.ToolkitRoot, "ui", "src") }

// load returns the engine, rebuilding it when any source changed.
func load() (*engine, error) {
	cacheMu.Lock()
	defer cacheMu.Unlock()
	mods := extui.List(paths.ToolkitRoot)
	if cached != nil && cached.fresh(mods) {
		return cached, nil
	}
	e, err := build(mods)
	if err != nil {
		return nil, err
	}
	cached = e
	return e, nil
}

func (e *engine) fresh(mods extui.Result) bool {
	for p, t := range e.inputs {
		st, err := os.Stat(p)
		if err != nil || !st.ModTime().Equal(t) {
			return false
		}
	}
	if len(mods.Modules) != len(e.modules) {
		return false
	}
	for _, m := range mods.Modules {
		if e.modules[m.ID] != m.MtimeMs {
			return false
		}
	}
	return pythonFresh()
}

var aliasRe = regexp.MustCompile(`^@/`)

func resolveFile(base string) (string, bool) {
	for _, cand := range []string{base, base + ".ts", base + ".tsx", base + ".js", base + ".jsx",
		filepath.Join(base, "index.ts"), filepath.Join(base, "index.tsx"), filepath.Join(base, "index.js")} {
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, true
		}
	}
	return "", false
}

func bundle() (string, map[string]time.Time, error) {
	src := uiSrc()
	plugin := api.Plugin{
		Name: "aitk-upstream",
		Setup: func(b api.PluginBuild) {
			b.OnResolve(api.OnResolveOptions{Filter: `^@/`}, func(a api.OnResolveArgs) (api.OnResolveResult, error) {
				p, ok := resolveFile(filepath.Join(src, filepath.FromSlash(aliasRe.ReplaceAllString(a.Path, ""))))
				if !ok {
					return api.OnResolveResult{}, fmt.Errorf("cannot resolve %s", a.Path)
				}
				return api.OnResolveResult{Path: p}, nil
			})
			// bare package imports (react, next/..., @scope/pkg): stubbed
			b.OnResolve(api.OnResolveOptions{Filter: `^([^./@]|@[^/])`}, func(a api.OnResolveArgs) (api.OnResolveResult, error) {
				return api.OnResolveResult{Path: a.Path, Namespace: "aitk-stub"}, nil
			})
			b.OnLoad(api.OnLoadOptions{Filter: `.*`, Namespace: "aitk-stub"}, func(api.OnLoadArgs) (api.OnLoadResult, error) {
				s := stubModule
				return api.OnLoadResult{Contents: &s, Loader: api.LoaderJS}, nil
			})
		},
	}
	res := api.Build(api.BuildOptions{
		Stdin:         &api.StdinOptions{Contents: entry, ResolveDir: src, Sourcefile: "aitk-templates-entry.ts", Loader: api.LoaderTS},
		AbsWorkingDir: src,
		Bundle:        true,
		Write:         false,
		Format:        api.FormatIIFE,
		Platform:      api.PlatformNeutral,
		Target:        api.ES2017,
		JSX:           api.JSXAutomatic,
		Metafile:      true,
		LogLevel:      api.LogLevelSilent,
		Plugins:       []api.Plugin{plugin},
	})
	if len(res.Errors) > 0 {
		msgs := make([]string, 0, len(res.Errors))
		for _, m := range res.Errors {
			msg := m.Text
			if m.Location != nil {
				msg = fmt.Sprintf("%s:%d: %s", m.Location.File, m.Location.Line, m.Text)
			}
			msgs = append(msgs, msg)
		}
		return "", nil, fmt.Errorf("bundle upstream UI sources: %s", strings.Join(msgs, "; "))
	}
	if len(res.OutputFiles) == 0 {
		return "", nil, fmt.Errorf("bundle upstream UI sources: no output")
	}
	var meta struct {
		Inputs map[string]json.RawMessage `json:"inputs"`
	}
	inputs := map[string]time.Time{}
	if json.Unmarshal([]byte(res.Metafile), &meta) == nil {
		for p := range meta.Inputs {
			if strings.Contains(p, ":") && !filepath.IsAbs(p) {
				continue // namespaced (stubs, stdin)
			}
			full := p
			if !filepath.IsAbs(full) {
				full = filepath.Join(src, filepath.FromSlash(p))
			}
			if st, err := os.Stat(full); err == nil {
				inputs[full] = st.ModTime()
			}
		}
	}
	return string(res.OutputFiles[0].Contents), inputs, nil
}

// nodePlatform is process.platform as Node reports it.
func nodePlatform() string {
	if runtime.GOOS == "windows" {
		return "win32"
	}
	return runtime.GOOS
}

func build(mods extui.Result) (*engine, error) {
	code, inputs, err := bundle()
	if err != nil {
		return nil, err
	}
	vm := goja.New()
	// jobConfig.ts's isMac() reads process.platform (the server's platform)
	vm.Set("process", map[string]any{"platform": nodePlatform(), "env": map[string]any{}})
	noop := func(goja.FunctionCall) goja.Value { return goja.Undefined() }
	vm.Set("console", map[string]any{"log": noop, "info": noop, "warn": noop, "error": noop, "debug": noop})
	if _, err := vm.RunScript("aitk-templates-bundle.js", code); err != nil {
		return nil, fmt.Errorf("evaluate upstream UI bundle: %w", err)
	}
	if _, err := vm.RunScript("aitk-templates-driver.js", driverJS); err != nil {
		return nil, fmt.Errorf("evaluate templates driver: %w", err)
	}

	type modIn struct {
		ID   string `json:"id"`
		Code string `json:"code"`
	}
	in := make([]modIn, 0, len(mods.Modules))
	modTimes := map[string]float64{}
	for _, m := range mods.Modules {
		in = append(in, modIn{ID: m.ID, Code: m.Code})
		modTimes[m.ID] = m.MtimeMs
	}
	modsJSON, _ := json.Marshal(in)
	loadFn, _ := goja.AssertFunction(vm.Get("__aitkLoadModules"))
	out, err := loadFn(goja.Undefined(), vm.ToValue(string(modsJSON)))
	if err != nil {
		return nil, fmt.Errorf("evaluate extension ui modules: %w", err)
	}
	var loaded struct {
		Errors []string `json:"errors"`
		Archs  []Arch   `json:"archs"`
	}
	if err := json.Unmarshal([]byte(out.String()), &loaded); err != nil {
		return nil, err
	}
	for _, e := range mods.Errors {
		loaded.Errors = append(loaded.Errors, e.ID+": "+e.Error)
	}
	defFn, _ := goja.AssertFunction(vm.Get("__aitkDefaultConfig"))
	defOut, err := defFn(goja.Undefined())
	if err != nil {
		return nil, fmt.Errorf("default job config: %w", err)
	}
	var def any
	if err := json.Unmarshal([]byte(defOut.String()), &def); err != nil {
		return nil, err
	}

	python := pythonArchs()
	for i := range loaded.Archs {
		a := &loaded.Archs[i]
		a.PythonArch = normalizePythonArch(a.Name)
		a.PythonAccepted = python[a.PythonArch]
		if a.Defaults == nil {
			a.Defaults = map[string]any{}
		}
	}
	if loaded.Errors == nil {
		loaded.Errors = []string{}
	}
	return &engine{
		vm: vm, inputs: inputs, modules: modTimes, python: python,
		catalog: Catalog{Archs: loaded.Archs, DefaultConfig: def, Errors: loaded.Errors},
	}, nil
}

// List is GET /v1/templates.
func List(context.Context) (Catalog, error) {
	e, err := load()
	if err != nil {
		return Catalog{}, err
	}
	return e.catalog, nil
}

var overridePath = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*$`)

// Render is POST /v1/templates/render. Problems the caller can fix come back
// as *v1api.Problem.
func Render(_ context.Context, req RenderRequest, opt Options) (RenderResult, error) {
	e, err := load()
	if err != nil {
		return RenderResult{}, err
	}
	warnings := []string{}
	var arch *Arch
	if req.Arch != "" {
		for i := range e.catalog.Archs {
			if e.catalog.Archs[i].Name == req.Arch {
				arch = &e.catalog.Archs[i]
			}
		}
		if arch == nil {
			names := make([]string, 0, len(e.catalog.Archs))
			for _, a := range e.catalog.Archs {
				names = append(names, a.Name)
			}
			return RenderResult{}, v1api.Errorf(v1api.CodeInvalidRequest,
				fmt.Sprintf("unknown arch %q; known archs: %s", req.Arch, strings.Join(names, ", ")))
		}
		if !arch.PythonAccepted {
			warnings = append(warnings, pythonWarning(arch, e.python))
		}
	}
	for k := range req.Overrides {
		if !overridePath.MatchString(k) {
			return RenderResult{}, v1api.Errorf(v1api.CodeInvalidRequest,
				fmt.Sprintf("override key %q is not a dotted path like config.process[0].train.steps", k))
		}
	}
	datasetPaths := make([]string, 0, len(req.Datasets))
	for _, name := range req.Datasets {
		if name == "" || strings.ContainsAny(name, `/\`) || name == "." || name == ".." {
			return RenderResult{}, v1api.Errorf(v1api.CodeInvalidRequest, fmt.Sprintf("invalid dataset name %q", name))
		}
		p := filepath.Join(opt.DatasetsRoot, name)
		if st, err := os.Stat(p); err != nil || !st.IsDir() {
			warnings = append(warnings, fmt.Sprintf("dataset %q does not exist yet under %s", name, opt.DatasetsRoot))
		}
		datasetPaths = append(datasetPaths, p)
	}
	if len(req.Datasets) == 0 {
		warnings = append(warnings, "no datasets given; datasets[0].folder_path keeps the upstream placeholder")
	}

	payload, _ := json.Marshal(map[string]any{
		"arch": req.Arch, "name": req.Name, "dataset_paths": datasetPaths,
		"overrides": req.Overrides, "training_folder": opt.TrainingFolder,
	})
	e.mu.Lock()
	fn, _ := goja.AssertFunction(e.vm.Get("__aitkRender"))
	out, err := fn(goja.Undefined(), e.vm.ToValue(string(payload)))
	e.mu.Unlock()
	if err != nil {
		return RenderResult{}, v1api.Errorf(v1api.CodeInvalidConfig, "render failed: "+err.Error())
	}
	var res struct {
		Config      any    `json:"config"`
		ArchApplied string `json:"arch_applied"`
	}
	if err := json.Unmarshal([]byte(out.String()), &res); err != nil {
		return RenderResult{}, err
	}
	if req.Arch != "" && res.ArchApplied != req.Arch {
		return RenderResult{}, v1api.Errorf(v1api.CodeInvalidConfig,
			fmt.Sprintf("upstream did not apply arch %q (the default config's arch %q is not in the arch list)", req.Arch, res.ArchApplied))
	}
	return RenderResult{Config: res.Config, Warnings: warnings}, nil
}

func pythonWarning(a *Arch, python map[string]bool) string {
	known := make([]string, 0, len(python))
	for k := range python {
		known = append(known, k)
	}
	sort.Strings(known)
	msg := fmt.Sprintf("arch %q is offered by the upstream UI but the Python trainer does not accept it "+
		"(toolkit/util/get_model.py would raise \"Unknown model arch\")", a.Name)
	if s := closest(a.PythonArch, known); s != "" {
		msg += fmt.Sprintf("; did you mean %q? set it with overrides {\"config.process[0].model.arch\": %q}", s, s)
	}
	return msg
}

// closest is the known name with the smallest edit distance (<= 2).
func closest(name string, known []string) string {
	best, bestD := "", 3
	for _, k := range known {
		if d := editDistance(name, k); d < bestD {
			best, bestD = k, d
		}
	}
	return best
}

func editDistance(a, b string) int {
	prev := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur := make([]int, len(b)+1)
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(b)]
}
