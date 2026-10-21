// Package safepath resolves requested file paths against allowed roots.
package safepath

import (
	"path/filepath"
	"strings"
)

func within(resolved string, allowed []string) bool {
	for _, dir := range allowed {
		root := abs(dir)
		if resolved == root || strings.HasPrefix(resolved, root+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

func abs(p string) string {
	if a, err := filepath.Abs(p); err == nil {
		return a
	}
	return filepath.Clean(p)
}

// Resolve returns the absolute path for an already URL-decoded file path when
// it lies under one of allowed, or "" when it does not. `..` segments are
// collapsed before the check. A path that arrived without its leading root
// (a proxy decoded `%2F` and collapsed the resulting `//`, so
// `/mnt/out/x.png` shows up as `mnt/out/x.png`) is retried against each
// allowed root's filesystem root.
func Resolve(filePath string, allowed []string) string {
	// URLs carry `/`; Windows paths may mix in `\`. Clean normalizes both.
	filePath = filepath.FromSlash(filePath)
	candidates := []string{filePath}
	if !filepath.IsAbs(filePath) {
		for _, dir := range allowed {
			if vol := rootOf(abs(dir)); vol != "" {
				candidates = append(candidates, filepath.Join(vol, filePath))
			}
		}
	}
	seen := map[string]bool{}
	for _, c := range candidates {
		if seen[c] {
			continue
		}
		seen[c] = true
		r := abs(c)
		if within(r, allowed) {
			return r
		}
	}
	return ""
}

// rootOf is path.parse(p).root: "/" on posix, "C:\" or "\\server\share\" on Windows.
func rootOf(p string) string {
	vol := filepath.VolumeName(p)
	rest := p[len(vol):]
	if strings.HasPrefix(rest, string(filepath.Separator)) {
		return vol + string(filepath.Separator)
	}
	return vol
}
