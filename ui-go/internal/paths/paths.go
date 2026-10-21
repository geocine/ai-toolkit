// Package paths locates the ai-toolkit checkout the server works against.
// Mirrors ui/cron/paths.ts and ui/cron/pythonPath.ts.
package paths

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// ToolkitRoot is the ai-toolkit checkout: the directory holding run.py.
// AI_TOOLKIT_ROOT overrides. Otherwise the parent of the working directory
// (the server runs from ui-go/, like the Next UI runs from ui/), falling back
// to the working directory itself when that is the checkout.
var ToolkitRoot = resolveRoot()

var (
	DefaultTrainFolder    = filepath.Join(ToolkitRoot, "output")
	DefaultDatasetsFolder = filepath.Join(ToolkitRoot, "datasets")
	DefaultDataRoot       = filepath.Join(ToolkitRoot, "data")
	DefaultModelsFolder   = filepath.Join(ToolkitRoot, "models")
	DBPath                = filepath.Join(ToolkitRoot, "aitk_db.db")
)

func resolveRoot() string {
	if v := strings.TrimSpace(os.Getenv("AI_TOOLKIT_ROOT")); v != "" {
		if abs, err := filepath.Abs(v); err == nil {
			return abs
		}
		return v
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	parent := filepath.Dir(cwd)
	if fileExists(filepath.Join(parent, "run.py")) {
		return parent
	}
	if fileExists(filepath.Join(cwd, "run.py")) {
		return cwd
	}
	return parent
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

var isWindows = runtime.GOOS == "windows"

// ResolvePython returns the interpreter for running toolkit scripts: the
// checkout's .venv or venv when present, else python on PATH.
func ResolvePython() string {
	var candidates []string
	if isWindows {
		candidates = []string{
			filepath.Join(ToolkitRoot, ".venv", "Scripts", "python.exe"),
			filepath.Join(ToolkitRoot, "venv", "Scripts", "python.exe"),
		}
	} else {
		candidates = []string{
			filepath.Join(ToolkitRoot, ".venv", "bin", "python"),
			filepath.Join(ToolkitRoot, "venv", "bin", "python"),
		}
	}
	for _, c := range candidates {
		if fileExists(c) {
			return c
		}
	}
	if isWindows {
		return "python.exe"
	}
	return "python3"
}

// ResolveDetachedPython is the interpreter for jobs that outlive the server.
// On Windows a detached console app gets a visible console window, so use
// the pythonw.exe next to the resolved interpreter when it exists.
func ResolveDetachedPython() string {
	p := ResolvePython()
	if !isWindows {
		return p
	}
	w := filepath.Join(filepath.Dir(p), "pythonw.exe")
	if fileExists(w) {
		return w
	}
	return p
}
