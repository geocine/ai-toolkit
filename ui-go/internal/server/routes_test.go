package server

import (
	"regexp"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/codemode"
	"github.com/geocine/aitoolkit/internal/httpx"
)

// notOps are /v1 routes the code-mode client deliberately has no op for.
var notOps = map[string]string{
	"GET /v1/events/stream": "an SSE stream never ends inside one sandboxed call; events.list with wait_ms long-polls the same events",
	"GET /v1/openapi.json":  "describes the HTTP API; agents read the client's declaration from aitk_api instead",
}

var param = regexp.MustCompile(`\{[^}]*\}`)

func normalize(method, path string) string { return method + " " + param.ReplaceAllString(path, "{}") }

// Every /v1 route has a codemode op (MCP parity: what a UI can do, an agent
// can do), and every op names a route that exists.
func TestEveryV1RouteHasAnOp(t *testing.T) {
	NewMux()
	ops := map[string]bool{}
	for _, o := range codemode.Ops {
		ops[normalize(o.Method, o.Path)] = true
	}
	routes := map[string]bool{}
	for _, pattern := range httpx.Routes() {
		method, path, ok := strings.Cut(pattern, " ")
		if !ok || !strings.HasPrefix(path, "/v1/") {
			continue
		}
		key := normalize(method, path)
		routes[key] = true
		if ops[key] {
			continue
		}
		if why, ok := notOps[pattern]; ok {
			t.Logf("no op for %s: %s", pattern, why)
			continue
		}
		t.Errorf("route %s has no codemode op (internal/codemode/ops.go); add one or list it in notOps with a reason", pattern)
	}
	for key := range ops {
		if !routes[key] {
			t.Errorf("codemode op %s names no registered route", key)
		}
	}
	for pattern := range notOps {
		method, path, _ := strings.Cut(pattern, " ")
		if !routes[normalize(method, path)] {
			t.Errorf("notOps lists %s, which is not a route any more", pattern)
		}
	}
}
