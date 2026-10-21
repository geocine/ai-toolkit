// Package sysstats reads CPU, memory and GPU stats the way the upstream Next
// server does (ui/src/server/cpuStats.ts, monitor.ts, api/cpu, api/gpu).
package sysstats

import (
	"encoding/json"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// Num is a JavaScript number: NaN and ±Infinity serialize as null, like
// JSON.stringify.
type Num float64

var NaN = Num(math.NaN())

func (n Num) IsNaN() bool { return math.IsNaN(float64(n)) }

func (n Num) MarshalJSON() ([]byte, error) {
	f := float64(n)
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return []byte("null"), nil
	}
	if f == 0 {
		return []byte("0"), nil // no "-0"
	}
	return json.Marshal(f)
}

// Round is JavaScript's Math.round (halves round toward +Infinity).
func Round(f float64) float64 { return math.Floor(f + 0.5) }

func trimJSLeft(s string) string {
	return strings.TrimLeftFunc(s, func(r rune) bool { return unicode.IsSpace(r) || r == 0xFEFF })
}

// ParseInt is JavaScript's parseInt(s) (radix 10, or 16 with a 0x prefix).
func ParseInt(s string) Num {
	s = trimJSLeft(s)
	neg := false
	if s != "" && (s[0] == '+' || s[0] == '-') {
		neg = s[0] == '-'
		s = s[1:]
	}
	base := 10
	if len(s) >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		base = 16
		s = s[2:]
	}
	end := 0
	for end < len(s) {
		c := s[end]
		isDigit := c >= '0' && c <= '9'
		if base == 16 {
			isDigit = isDigit || (c|0x20 >= 'a' && c|0x20 <= 'f')
		}
		if !isDigit {
			break
		}
		end++
	}
	if end == 0 {
		return NaN
	}
	v, err := strconv.ParseFloat(s[:end], 64)
	if base == 16 {
		var u uint64
		u, err = strconv.ParseUint(s[:end], 16, 64)
		v = float64(u)
	}
	if err != nil {
		return NaN
	}
	if neg {
		v = -v
	}
	return Num(v)
}

var floatPrefix = regexp.MustCompile(`^[+-]?(Infinity|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)`)

// ParseFloat is JavaScript's parseFloat(s).
func ParseFloat(s string) Num {
	m := floatPrefix.FindString(trimJSLeft(s))
	if m == "" {
		return NaN
	}
	switch strings.TrimLeft(m, "+-") {
	case "Infinity":
		if strings.HasPrefix(m, "-") {
			return Num(math.Inf(-1))
		}
		return Num(math.Inf(1))
	}
	v, err := strconv.ParseFloat(m, 64)
	if err != nil {
		// an exponent too large for float64 still parses as ±Inf in JS
		if ne, ok := err.(*strconv.NumError); ok && ne.Err == strconv.ErrRange {
			return Num(v)
		}
		return NaN
	}
	return Num(v)
}

// orZero is `x || 0` for a number.
func orZero(n Num) Num {
	if n.IsNaN() {
		return 0
	}
	return n
}
