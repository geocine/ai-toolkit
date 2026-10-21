package templates

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
)

// The Python trainer resolves model.arch in toolkit/util/get_model.py:
// classes registered by extensions (AI_TOOLKIT_MODELS) and the built-ins
// match on their class attribute `arch`, and LEGACY_ARCHS go to the
// StableDiffusion monolith; anything else raises "Unknown model arch". Read
// statically (importing the models needs torch).

var (
	legacyBlock = regexp.MustCompile(`(?s)LEGACY_ARCHS\s*=\s*\{(.*?)\}`)
	quoted      = regexp.MustCompile(`["']([^"']+)["']`)
	// a class body attribute: exactly one indent level, `arch = "x"`
	classArch = regexp.MustCompile(`(?m)^(?:    |\t)arch\s*(?::\s*\w+\s*)?=\s*["']([^"']+)["']\s*(?:#.*)?$`)
)

var pythonScanDirs = []string{"extensions_built_in", "extensions", filepath.Join("toolkit", "models")}

var (
	pyMu    sync.Mutex
	pyFiles map[string]time.Time
)

// pythonArchs scans the toolkit for archs the trainer accepts.
func pythonArchs() map[string]bool {
	archs := map[string]bool{}
	files := map[string]time.Time{}
	getModel := filepath.Join(paths.ToolkitRoot, "toolkit", "util", "get_model.py")
	if b, err := os.ReadFile(getModel); err == nil {
		if st, err := os.Stat(getModel); err == nil {
			files[getModel] = st.ModTime()
		}
		if m := legacyBlock.FindSubmatch(b); m != nil {
			for _, q := range quoted.FindAllSubmatch(m[1], -1) {
				archs[string(q[1])] = true
			}
		}
	}
	for _, dir := range pythonScanDirs {
		filepath.WalkDir(filepath.Join(paths.ToolkitRoot, dir), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if n := d.Name(); n == "__pycache__" || strings.HasPrefix(n, ".") || n == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(p, ".py") {
				return nil
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return nil
			}
			if info, err := d.Info(); err == nil {
				files[p] = info.ModTime()
			}
			for _, m := range classArch.FindAllSubmatch(b, -1) {
				archs[string(m[1])] = true
			}
			return nil
		})
	}
	pyMu.Lock()
	pyFiles = files
	pyMu.Unlock()
	return archs
}

// pythonFresh reports whether no scanned Python file changed (new files in
// the scanned dirs are picked up on the next rebuild of the ui sources).
func pythonFresh() bool {
	pyMu.Lock()
	defer pyMu.Unlock()
	for p, t := range pyFiles {
		st, err := os.Stat(p)
		if err != nil || !st.ModTime().Equal(t) {
			return false
		}
	}
	return true
}

// normalizePythonArch applies toolkit/config_modules.py's ModelConfig arch
// normalization: a ":tag" suffix is dropped and flex1 means flux.
func normalizePythonArch(arch string) string {
	if i := strings.Index(arch, ":"); i >= 0 {
		arch = arch[:i]
	}
	if arch == "flex1" {
		return "flux"
	}
	return arch
}
