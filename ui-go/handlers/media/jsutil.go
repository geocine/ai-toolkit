package media

import (
	"errors"
	"math"
	"net/url"
	"regexp"
	"strings"
	"unicode/utf8"
)

// The upstream routes lean on a few JavaScript semantics that differ from Go's
// stdlib equivalents; these reproduce them.

// jsParseInt is parseInt(s, 10): optional leading whitespace and sign, then the
// longest run of digits; NaN (ok=false) when there are none.
func jsParseInt(s string) (int64, bool) {
	s = strings.TrimLeft(s, " \t\n\r\v\f")
	neg := false
	if s != "" && (s[0] == '+' || s[0] == '-') {
		neg = s[0] == '-'
		s = s[1:]
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 {
		return 0, false
	}
	var n int64
	for _, c := range s[:i] {
		if n > (math.MaxInt64-9)/10 {
			n = math.MaxInt64
			break
		}
		n = n*10 + int64(c-'0')
	}
	if neg {
		n = -n
	}
	return n, true
}

var errURI = errors.New("URIError: URI malformed")

// decodeURIComponent fails on malformed escapes and on escapes that do not
// form valid UTF-8, like JavaScript's (`+` stays `+`).
func decodeURIComponent(s string) (string, error) {
	out, err := url.PathUnescape(s)
	if err != nil || !utf8.ValidString(out) {
		return "", errURI
	}
	return out, nil
}

// captionExt is upstream's `((ext || "txt") as string).replace(/^\.+/, "").trim() || "txt"`.
// A truthy non-string ext throws in upstream (no .replace), reported as an error.
func captionExt(ext any) (string, error) {
	if !truthy(ext) {
		return "txt", nil
	}
	s, ok := ext.(string)
	if !ok {
		return "", errors.New("ext.replace is not a function")
	}
	s = strings.TrimSpace(strings.TrimLeft(s, "."))
	if s == "" {
		return "txt", nil
	}
	return s, nil
}

func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case float64:
		return x != 0 && !math.IsNaN(x)
	}
	return true
}

var extRe = regexp.MustCompile(`\.[^/.]+$`)

// stripExt is `p.replace(/\.[^/.]+$/, "")`: drops the last extension; note it
// only treats `/` as a separator, so a Windows `a.b\c` loses `.b\c`.
func stripExt(p string) string {
	return extRe.ReplaceAllString(p, "")
}

// str returns body[key] when it is a string; a missing key or another type is
// what makes upstream throw a TypeError on string methods.
func str(body map[string]any, key string) (string, bool) {
	s, ok := body[key].(string)
	return s, ok
}
