// Package extui discovers and transpiles extension UI modules at request
// time. Mirrors ui/src/server/extensionUi.ts: every package under
// extensions_built_in/ and extensions/ may ship a ui.tsx (or .ts/.jsx/.js)
// exporting AI_TOOLKIT_UI_MODELS. Files are re-read when their mtime changes
// and transpiled to CommonJS (esbuild here, SWC upstream); the browser
// evaluates them with a fixed set of importable modules.
package extui

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/evanw/esbuild/pkg/api"
)

var (
	extensionDirs = []string{"extensions_built_in", "extensions"}
	uiFileNames   = []string{"ui.tsx", "ui.ts", "ui.jsx", "ui.js"}
)

type Module struct {
	ID      string  `json:"id"` // "<dir>/<package>"
	File    string  `json:"file"`
	MtimeMs float64 `json:"mtimeMs"`
	Code    string  `json:"code"` // CommonJS source
}

type Error struct {
	ID    string `json:"id"`
	File  string `json:"file"`
	Error string `json:"error"`
}

type Result struct {
	Modules []Module `json:"modules"`
	Errors  []Error  `json:"errors"`
}

var (
	mu    sync.Mutex
	cache = map[string]Module{}
)

func loaderFor(file string) api.Loader {
	switch {
	case strings.HasSuffix(file, ".tsx"):
		return api.LoaderTSX
	case strings.HasSuffix(file, ".ts"):
		return api.LoaderTS
	default:
		// SWC parses .js and .jsx as ecmascript with JSX enabled
		return api.LoaderJSX
	}
}

// Transpile converts one ui module to CommonJS with the automatic JSX runtime.
func Transpile(file, source string) (string, error) {
	res := api.Transform(source, api.TransformOptions{
		Sourcefile: file,
		Loader:     loaderFor(file),
		JSX:        api.JSXAutomatic,
		Target:     api.ES2020,
		Format:     api.FormatCommonJS,
		Sourcemap:  api.SourceMapNone,
	})
	if len(res.Errors) > 0 {
		msgs := api.FormatMessages(res.Errors, api.FormatMessagesOptions{Kind: api.ErrorMessage})
		return "", &transpileError{strings.TrimSpace(strings.Join(msgs, "\n"))}
	}
	return string(res.Code), nil
}

type transpileError struct{ msg string }

func (e *transpileError) Error() string { return e.msg }

func findUIFile(pkgDir string) string {
	for _, name := range uiFileNames {
		f := filepath.Join(pkgDir, name)
		if st, err := os.Stat(f); err == nil && st.Mode().IsRegular() {
			return f
		}
	}
	return ""
}

// List returns every extension UI module under root, built-ins first,
// packages sorted by name.
func List(root string) Result {
	mu.Lock()
	defer mu.Unlock()
	out := Result{Modules: []Module{}, Errors: []Error{}}
	seen := map[string]bool{}
	for _, dir := range extensionDirs {
		entries, err := os.ReadDir(filepath.Join(root, dir))
		if err != nil {
			continue
		}
		var pkgs []string
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() && !strings.HasPrefix(name, ".") && !strings.HasPrefix(name, "_") {
				pkgs = append(pkgs, name)
			}
		}
		sort.Strings(pkgs)
		for _, pkg := range pkgs {
			file := findUIFile(filepath.Join(root, dir, pkg))
			if file == "" {
				continue
			}
			id := dir + "/" + pkg
			seen[id] = true
			st, err := os.Stat(file)
			if err != nil {
				delete(cache, id)
				out.Errors = append(out.Errors, Error{ID: id, File: file, Error: err.Error()})
				continue
			}
			mtimeMs := float64(st.ModTime().UnixNano()) / 1e6
			if c, ok := cache[id]; ok && c.File == file && c.MtimeMs == mtimeMs {
				out.Modules = append(out.Modules, c)
				continue
			}
			source, err := os.ReadFile(file)
			if err == nil {
				var code string
				code, err = Transpile(file, string(source))
				if err == nil {
					mod := Module{ID: id, File: file, MtimeMs: mtimeMs, Code: code}
					cache[id] = mod
					out.Modules = append(out.Modules, mod)
					continue
				}
			}
			delete(cache, id)
			out.Errors = append(out.Errors, Error{ID: id, File: file, Error: err.Error()})
		}
	}
	for id := range cache {
		if !seen[id] {
			delete(cache, id)
		}
	}
	return out
}
