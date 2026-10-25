package quickpod

import (
	"net"
	"regexp"
	"strconv"
	"strings"
)

// tag is an HTML tag: the host agent writes each address as a link.
var tag = regexp.MustCompile(`<[^>]*>`)

// Parse8675 finds the host address of container port 8675 in a pod's
// port_mappings. The host agent writes one entry per published port,
// `8675 -> <a target="_new" href="http://IP:PORT">IP:PORT</a> |`; the tags
// are dropped first, and plain "8675 -> IP:PORT" entries (with "/tcp" on the
// left, or "," between entries) read the same.
func Parse8675(s string) (ip string, port int, ok bool) {
	s = tag.ReplaceAllString(s, "")
	for _, entry := range strings.FieldsFunc(s, func(r rune) bool { return r == '|' || r == ',' }) {
		left, right, found := strings.Cut(entry, "->")
		if !found {
			continue
		}
		left = strings.TrimSpace(left)
		left, _, _ = strings.Cut(left, "/")
		if left != "8675" {
			continue
		}
		host, p, err := net.SplitHostPort(strings.TrimSpace(right))
		if err != nil {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n <= 0 || n > 65535 || net.ParseIP(host) == nil {
			continue
		}
		return host, n, true
	}
	return "", 0, false
}
