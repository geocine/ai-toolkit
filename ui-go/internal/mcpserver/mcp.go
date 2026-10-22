package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/geocine/aitoolkit/internal/codemode"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/links"
)

// Version is reported to MCP clients.
const Version = "1.0.0"

const instructions = `ai-toolkit training server, code mode.

Instead of one tool per endpoint, write a short TypeScript script against the typed ` + "`aitk`" + ` client:
1. Call aitk_api (optionally with a search term) to read the client's TypeScript declaration.
2. Call aitk_run with code using it. Top-level await and return work; the returned value is your result.
3. Call aitk_view with sample or dataset image paths (as the API returns them) to look at them.

The script runs in a sandbox with only aitk.*, console.log and sleep(ms): no filesystem, network or processes.
Typical flow: aitk.datasets.create -> add images (datasets.importUrl by URL, datasets.putFile with base64 for small files)
-> aitk.templates.render({arch, name, datasets}) -> aitk.jobs.create({name, gpu_ids, config, policy}) -> aitk.jobs.start(id)
To build or change a config: aitk.templates.fields({ section }) lists every key the trainer reads with its default and where
it goes, aitk.templates.examples / example(name) are upstream's starting points, and aitk.templates.check({ config }) finds keys
the trainer would ignore (wrong section, typos). Check before jobs.create.
To caption a dataset with a model: aitk.templates.captioners() lists upstream's captioners (model choices, named prompts),
then aitk.datasets.caption(name, { captioner, model, prompt }) queues a caption job; datasets.get(name).caption_job follows it.
-> poll aitk.jobs.get(id) or aitk.jobs.wait(id); look at aitk.jobs.samples(id) with aitk_view.
Run bundles move a whole run (dataset + checkpoints + optimizer state) between servers: aitk.bundles.download(jobId) gives a
signed URL, aitk.bundles.import({ url | remote+key | upload_id, start: true }) continues it here, aitk.bundles.push / list
manage a remote's library. To hand a person a file, sign it: aitk.files.link({ root, path }) or aitk.links.create.
Training runs for a long time: start it, return, and check back with later runs instead of waiting in one.`

const apiNotes = `// Usage: every method returns a Promise; await it. Methods with path params take them positionally,
// then one params object: aitk.jobs.log(id, { offset: -4096 }). Failed calls throw AitkError {code, status, message}.
// Results are the /v1 JSON bodies (lists are { items: [...] }).
// Binary in: base64 params (datasets.putFile, uploadFiles, uploadArchive, bundles.uploadChunk; 16 MiB per call) or
// let the server fetch a URL (datasets.importUrl, bundles.import({ url })).
// Binary out: methods returning Link give a signed URL ({ url, expires_at }) that downloads without a token;
// to see images yourself use the aitk_view tool. events.list({ after, wait_ms }) long-polls for new events.
`

// APIInput is aitk_api's input.
type APIInput struct {
	Search string `json:"search,omitempty" jsonschema:"only list operations whose group, name, path or description contains this"`
}

// RunInput is aitk_run's input.
type RunInput struct {
	Code      string `json:"code" jsonschema:"TypeScript or JavaScript using the aitk client; top-level await and return are allowed"`
	TimeoutMs int    `json:"timeout_ms,omitempty" jsonschema:"wall-time limit in ms (default 120000, max 600000)"`
}

func newServer() *mcp.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "aitk", Version: Version}, &mcp.ServerOptions{Instructions: instructions})
	mcp.AddTool(s, &mcp.Tool{
		Name:        "aitk_api",
		Description: "TypeScript declaration of the `aitk` client (every /v1 operation with typed params and results). Read it before writing code for aitk_run.",
	}, func(_ context.Context, _ *mcp.CallToolRequest, in APIInput) (*mcp.CallToolResult, any, error) {
		text := apiNotes + codemode.Declaration(in.Search)
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: text}}}, nil, nil
	})
	mcp.AddTool(s, &mcp.Tool{
		Name: "aitk_run",
		Description: "Run TypeScript against the `aitk` client in a sandbox and return {result, logs, error}. " +
			"Only aitk.*, console and sleep are available. Limits: 120 s by default (timeout_ms up to 600000), 256 aitk calls, 1 MiB of logs.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, in RunInput) (*mcp.CallToolResult, RunOutput, error) {
		if api == nil {
			return nil, RunOutput{}, fmt.Errorf("server API not configured")
		}
		auth, origin := "", ""
		if req.Extra != nil && req.Extra.Header != nil {
			auth, origin = req.Extra.Header.Get("Authorization"), req.Extra.Header.Get(originHeader)
		}
		res := codemode.Run(ctx, in.Code, codemode.Options{
			Timeout: time.Duration(in.TimeoutMs) * time.Millisecond,
			Call:    codemode.HandlerCaller(api, auth, origin),
		})
		out := RunOutput{Logs: res.Logs, Error: res.Error, Calls: res.Calls, DurationMs: res.DurationMs}
		json.Unmarshal(res.Result, &out.Result)
		text, _ := json.MarshalIndent(out, "", "  ")
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: string(text)}}, IsError: res.Error != nil}, out, nil
	})
	mcp.AddTool(s, &mcp.Tool{
		Name: "aitk_view",
		Description: "Look at images: sample or dataset image paths as the API returns them (a sample's path; a dataset's path joined with an item's file, " +
			"or <dataset>/<file>). Returns JPEG previews scaled to max_px (default 512, max 1024), up to 8 per call.",
	}, view)
	return s
}

// RunOutput is aitk_run's structured result.
type RunOutput struct {
	Result     any                 `json:"result" jsonschema:"the value the script returned (JSON)"`
	Logs       string              `json:"logs"`
	Error      *codemode.ErrorInfo `json:"error"`
	Calls      int                 `json:"calls"`
	DurationMs int64               `json:"duration_ms"`
}

// Handler is the /mcp endpoint (stateless Streamable HTTP: tools only).
func Handler() http.Handler {
	s := newServer()
	return mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return s },
		&mcp.StreamableHTTPOptions{Stateless: true})
}

// originHeader carries the scheme://host the MCP client reached us at into
// the tool handlers (set here, never trusted from the client), so links
// signed from code mode are absolute URLs the agent can hand out.
const originHeader = "X-Aitk-Origin"

// Register serves /mcp. The auth middleware in front of the mux guards it.
func Register(mux *http.ServeMux) {
	h := Handler()
	httpx.Handle(mux, "/mcp", func(w http.ResponseWriter, r *http.Request) {
		r.Header.Set(originHeader, links.Origin(r, os.Getenv("AITK_PUBLIC_URL")))
		h.ServeHTTP(w, r)
	})
}
