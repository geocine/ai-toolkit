package jobfiles

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// Helpers that reproduce the JavaScript semantics the upstream routes lean
// on (Buffer.toString('utf-8'), Number(), parseInt(), Array.prototype.sort),
// so responses match byte for byte where it matters.

// decodeUTF8 decodes like Buffer.toString('utf-8') / the WHATWG decoder:
// each maximal invalid subsequence becomes one U+FFFD. (Go's utf8 package
// replaces byte by byte, which differs on truncated multi-byte characters,
// e.g. a log read that starts or ends mid progress-bar glyph.)
func decodeUTF8(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	var out strings.Builder
	out.Grow(len(b))
	n := len(b)
	for i := 0; i < n; {
		c := b[i]
		if c < 0x80 {
			out.WriteByte(c)
			i++
			continue
		}
		need := 0
		lo, hi := byte(0x80), byte(0xBF)
		switch {
		case c >= 0xC2 && c <= 0xDF:
			need = 1
		case c == 0xE0:
			need, lo = 2, 0xA0
		case (c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF:
			need = 2
		case c == 0xED:
			need, hi = 2, 0x9F
		case c == 0xF0:
			need, lo = 3, 0x90
		case c >= 0xF1 && c <= 0xF3:
			need = 3
		case c == 0xF4:
			need, hi = 3, 0x8F
		default:
			out.WriteRune(utf8.RuneError)
			i++
			continue
		}
		j, ok := i+1, true
		for k := 0; k < need; k++ {
			if j >= n {
				ok = false
				break
			}
			d := b[j]
			if k == 0 {
				if d < lo || d > hi {
					ok = false
					break
				}
			} else if d < 0x80 || d > 0xBF {
				ok = false
				break
			}
			j++
		}
		if ok {
			out.Write(b[i:j])
		} else {
			// consume the lead and the valid continuations; the offending
			// byte (if any) is decoded on its own next
			out.WriteRune(utf8.RuneError)
		}
		i = j
	}
	return out.String()
}

var decimalLiteral = regexp.MustCompile(`^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`)

// jsNumber is Number(string): trimmed, "" is 0, Infinity and 0x/0o/0b
// literals accepted, anything else unparsable is NaN.
func jsNumber(s string) float64 {
	s = strings.TrimFunc(s, isJSSpace)
	if s == "" {
		return 0
	}
	switch s {
	case "Infinity", "+Infinity":
		return math.Inf(1)
	case "-Infinity":
		return math.Inf(-1)
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
			v := 0.0
			for _, ch := range s[2:] {
				d, err := strconv.ParseUint(string(ch), base, 8)
				if err != nil {
					return math.NaN()
				}
				v = v*float64(base) + float64(d)
			}
			return v
		}
	}
	if !decimalLiteral.MatchString(s) {
		return math.NaN()
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		return math.NaN()
	}
	return v
}

// jsParseInt is parseInt(s, 10): leading whitespace, optional sign, then the
// longest run of digits; NaN when there are none.
func jsParseInt(s string) float64 {
	s = strings.TrimLeftFunc(s, isJSSpace)
	neg := false
	if s != "" && (s[0] == '+' || s[0] == '-') {
		neg = s[0] == '-'
		s = s[1:]
	}
	end := 0
	for end < len(s) && s[end] >= '0' && s[end] <= '9' {
		end++
	}
	if end == 0 {
		return math.NaN()
	}
	v, _ := strconv.ParseFloat(s[:end], 64)
	if neg {
		v = -v
	}
	return v
}

func isJSSpace(r rune) bool {
	switch r {
	case '\t', '\n', '\v', '\f', '\r', ' ', 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF:
		return true
	}
	return r >= 0x2000 && r <= 0x200A
}

// toNumber is Number(v) for a value decoded from JSON (via parseBody).
func toNumber(v any) float64 {
	switch x := v.(type) {
	case nil:
		return 0
	case bool:
		if x {
			return 1
		}
		return 0
	case float64:
		return x
	case string:
		return jsNumber(x)
	case []any:
		return jsNumber(jsString(x))
	}
	return math.NaN() // objects: "[object Object]"
}

// jsString is String(v) for JSON values (used for Number([..]) coercion).
func jsString(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case bool:
		return strconv.FormatBool(x)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case string:
		return x
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = jsString(e)
		}
		return strings.Join(parts, ",")
	}
	return "[object Object]"
}

// parseBody is request.json(): the body text (BOM stripped, decoded like
// fetch's UTF-8 decode) parsed strictly, numbers as float64 like JSON.parse.
func parseBody(r *http.Request) (any, error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	raw = bytes.TrimPrefix(raw, []byte("\xEF\xBB\xBF"))
	dec := json.NewDecoder(strings.NewReader(decodeUTF8(raw)))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("unexpected data after JSON body")
	}
	return numbersToFloat(v), nil
}

func numbersToFloat(v any) any {
	switch x := v.(type) {
	case json.Number:
		f, _ := strconv.ParseFloat(string(x), 64) // out of range -> ±Inf, like JSON.parse
		return f
	case []any:
		for i := range x {
			x[i] = numbersToFloat(x[i])
		}
	case map[string]any:
		for k := range x {
			x[k] = numbersToFloat(x[k])
		}
	}
	return v
}

// jsonNum is a number as JSON.stringify writes it: NaN and ±Infinity become null.
func jsonNum(f float64) any {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil
	}
	return f
}

// jsLess orders strings like Array.prototype.sort's default (UTF-16 code units).
func jsLess(a, b string) bool {
	ua, ub := utf16.Encode([]rune(a)), utf16.Encode([]rune(b))
	for i := 0; i < len(ua) && i < len(ub); i++ {
		if ua[i] != ub[i] {
			return ua[i] < ub[i]
		}
	}
	return len(ua) < len(ub)
}

// jsSlice is s.slice(0, end) with JavaScript's negative-index rule, over
// UTF-16 code units (filenames here are compared, not displayed).
func jsSlice(s string, end int) string {
	u := utf16.Encode([]rune(s))
	if end < 0 {
		end += len(u)
	}
	end = max(0, min(end, len(u)))
	return string(utf16.Decode(u[:end]))
}

// lastIndexOf is s.lastIndexOf(sub) in UTF-16 code units.
func lastIndexOf(s, sub string) int {
	u, su := utf16.Encode([]rune(s)), utf16.Encode([]rune(sub))
	for i := len(u) - len(su); i >= 0; i-- {
		match := true
		for k := range su {
			if u[i+k] != su[k] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}
