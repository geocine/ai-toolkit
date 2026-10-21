package mcpserver

import "net/http"

// api is the full server handler (auth included) that sandboxed code calls
// in-process; set by main once the mux is built.
var api http.Handler

// SetAPI installs the handler /v1 calls from the code-mode sandbox go to.
func SetAPI(h http.Handler) { api = h }

// Register serves /mcp (filled in by the MCP work).
func Register(mux *http.ServeMux) {}
