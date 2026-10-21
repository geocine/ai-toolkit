package datasets

import (
	"github.com/geocine/aitoolkit/internal/collation"
	"math"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"unicode"
)

// Helpers reproducing the JavaScript semantics the upstream routes rely on.

// localeCompare is String.prototype.localeCompare under Node's ICU, which
// upstream uses to sort file lists.
func localeCompare(a, b string) int { return collation.Compare(a, b) }

// replaceUTF16 replaces every UTF-16 code unit of s for which keep is false
// with repl, like a JS /[^...]/g replace: a character outside the BMP counts twice.
func replaceUTF16(s string, keep func(rune) bool, repl string) string {
	var b strings.Builder
	for _, r := range s {
		if keep(r) {
			b.WriteRune(r)
			continue
		}
		b.WriteString(repl)
		if r > 0xFFFF {
			b.WriteString(repl)
		}
	}
	return b.String()
}

// jsTrim is String.prototype.trim (Unicode whitespace plus BOM).
func jsTrim(s string) string {
	return strings.TrimFunc(s, func(r rune) bool { return unicode.IsSpace(r) || r == 0xFEFF })
}

var jsDecimal = regexp.MustCompile(`^[+-]?(Infinity|\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)$`)

// jsNumber is Number(raw) for a query param; ok=false is NaN. A missing param
// is null, and Number(null) is 0.
func jsNumber(raw string, present bool) (float64, bool) {
	if !present {
		return 0, true
	}
	s := jsTrim(raw)
	if s == "" {
		return 0, true
	}
	if len(s) > 2 && s[0] == '0' {
		base := 0
		switch s[1] {
		case 'x', 'X':
			base = 16
		case 'o', 'O':
			base = 8
		case 'b', 'B':
			base = 2
		}
		if base != 0 {
			n, err := strconv.ParseUint(s[2:], base, 64)
			if err != nil || strings.Contains(s[2:], "_") {
				return 0, false
			}
			return float64(n), true
		}
	}
	if !jsDecimal.MatchString(s) {
		return 0, false
	}
	f, err := strconv.ParseFloat(strings.TrimPrefix(s, "+"), 64)
	if err != nil && !math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

// isSafeInteger is Number.isSafeInteger.
func isSafeInteger(f float64) bool {
	return !math.IsInf(f, 0) && !math.IsNaN(f) && f == math.Trunc(f) && math.Abs(f) <= 1<<53-1
}

// jsResolve is path.resolve(root, name) for an absolute root.
func jsResolve(root, name string) string {
	p := filepath.FromSlash(name)
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	if runtime.GOOS == "windows" {
		if filepath.VolumeName(p) != "" { // drive-relative, e.g. "D:foo"
			if abs, err := filepath.Abs(p); err == nil {
				return abs
			}
			return filepath.Clean(p)
		}
		if strings.HasPrefix(p, `\`) { // rooted on root's drive
			return filepath.Clean(filepath.VolumeName(root) + p)
		}
	}
	return filepath.Join(root, p)
}

// directChild resolves name against root and reports whether it is a direct
// child of root (upstream's check: rejects "..", absolute paths and root itself).
func directChild(root, name string) (string, bool) {
	p := jsResolve(root, name)
	return p, filepath.Dir(p) == root && p != root
}
