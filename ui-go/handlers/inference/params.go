package inference

import (
	"strings"
	"unicode/utf8"
)

// searchParams is URLSearchParams: ordered pairs, parsed and serialized
// with application/x-www-form-urlencoded rules (url.Values sorts keys and
// escapes differently, which would reorder what the engine receives).
type searchParams [][2]string

func parseSearchParams(raw string) searchParams {
	var out searchParams
	for _, seq := range strings.Split(raw, "&") {
		if seq == "" {
			continue
		}
		name, value, _ := strings.Cut(seq, "=")
		out = append(out, [2]string{formDecode(name), formDecode(value)})
	}
	return out
}

func (p searchParams) get(name string) (string, bool) {
	for _, kv := range p {
		if kv[0] == name {
			return kv[1], true
		}
	}
	return "", false
}

func (p *searchParams) del(name string) {
	kept := (*p)[:0]
	for _, kv := range *p {
		if kv[0] != name {
			kept = append(kept, kv)
		}
	}
	*p = kept
}

func (p searchParams) String() string {
	parts := make([]string, len(p))
	for i, kv := range p {
		parts[i] = formEncode(kv[0]) + "=" + formEncode(kv[1])
	}
	return strings.Join(parts, "&")
}

func unhex(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}

// formDecode: '+' is a space; bad percent sequences stay literal; invalid
// UTF-8 becomes U+FFFD.
func formDecode(s string) string {
	s = strings.ReplaceAll(s, "+", " ")
	buf := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '%' && i+2 < len(s) {
			hi, ok1 := unhex(s[i+1])
			lo, ok2 := unhex(s[i+2])
			if ok1 && ok2 {
				buf = append(buf, hi<<4|lo)
				i += 2
				continue
			}
		}
		buf = append(buf, s[i])
	}
	if utf8.Valid(buf) {
		return string(buf)
	}
	return strings.ToValidUTF8(string(buf), string(utf8.RuneError))
}

// formEncode is the urlencoded byte serializer: alphanumerics and *-._ pass,
// space becomes '+', everything else is %XX.
func formEncode(s string) string {
	const hex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == ' ':
			b.WriteByte('+')
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '*', c == '-', c == '.', c == '_':
			b.WriteByte(c)
		default:
			b.WriteByte('%')
			b.WriteByte(hex[c>>4])
			b.WriteByte(hex[c&15])
		}
	}
	return b.String()
}
