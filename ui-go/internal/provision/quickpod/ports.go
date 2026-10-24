package quickpod

import (
	"net"
	"strconv"
	"strings"
)

// Parse8675 finds the host address of container port 8675 in a pod's
// port_mappings. The format is reconstructed from the console's parser
// (spec: quickpod.md §6.3): entries "<internal> -> <ip>:<external>"
// separated by "|" (UNCONFIRMED: "," is accepted too), the left side a bare
// port that may carry "/tcp", with any amount of space around the parts.
func Parse8675(s string) (ip string, port int, ok bool) {
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
