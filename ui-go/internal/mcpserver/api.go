// Package mcpserver serves /mcp: MCP over Streamable HTTP in code mode
// (docs/headless.md). Two tools instead of one per endpoint: aitk_api returns
// the TypeScript declaration of the `aitk` client, aitk_run executes code
// against it in the internal/codemode sandbox.
package mcpserver

import "net/http"

// api is the full server handler (auth included) that sandboxed code calls
// in-process; set by main once the mux is built.
var api http.Handler

// SetAPI installs the handler /v1 calls from the code-mode sandbox go to.
func SetAPI(h http.Handler) { api = h }
