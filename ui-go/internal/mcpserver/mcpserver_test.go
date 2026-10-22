package mcpserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type authTransport struct{ token string }

func (a authTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("Authorization", "Bearer "+a.token)
	return http.DefaultTransport.RoundTrip(r)
}

func TestMCPEndToEnd(t *testing.T) {
	// a stand-in /v1 that echoes what it received
	var gotAuth string
	SetAPI(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/jobs/j1":
			json.NewEncoder(w).Encode(map[string]any{"id": "j1", "state": "running", "step": 42})
		default:
			w.WriteHeader(404)
			json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": "not_found", "message": "nope"}})
		}
	}))
	t.Cleanup(func() { SetAPI(nil) })

	mux := http.NewServeMux()
	Register(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	ctx := context.Background()
	client := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "1"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint: srv.URL + "/mcp", HTTPClient: &http.Client{Transport: authTransport{"tok"}}, MaxRetries: -1,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	if !strings.Contains(session.InitializeResult().Instructions, "code mode") {
		t.Error("instructions missing")
	}
	tools, err := session.ListTools(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	names := []string{}
	for _, tl := range tools.Tools {
		names = append(names, tl.Name)
	}
	if strings.Join(names, ",") != "aitk_api,aitk_run,aitk_view" {
		t.Fatalf("tools = %v", names)
	}

	text := func(r *mcp.CallToolResult) string {
		var b strings.Builder
		for _, c := range r.Content {
			if tc, ok := c.(*mcp.TextContent); ok {
				b.WriteString(tc.Text)
			}
		}
		return b.String()
	}

	apiRes, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "aitk_api", Arguments: map[string]any{"search": "checkpoint"}})
	if err != nil {
		t.Fatal(err)
	}
	if decl := text(apiRes); !strings.Contains(decl, "checkpoints(id: string") || !strings.Contains(decl, "declare const aitk") {
		t.Fatalf("aitk_api:\n%s", decl)
	}

	runRes, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "aitk_run", Arguments: map[string]any{
		"code": `const j: { step: number } = await aitk.jobs.get("j1"); console.log("step", j.step); return j.step * 2;`,
	}})
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Result json.RawMessage `json:"result"`
		Logs   string          `json:"logs"`
		Error  any             `json:"error"`
	}
	if err := json.Unmarshal([]byte(text(runRes)), &out); err != nil {
		t.Fatal(err)
	}
	if runRes.IsError || string(out.Result) != "84" || out.Logs != "step 42\n" || gotAuth != "Bearer tok" {
		t.Fatalf("aitk_run = %s (auth %q)", text(runRes), gotAuth)
	}

	failRes, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "aitk_run", Arguments: map[string]any{"code": `await aitk.jobs.get("zz")`}})
	if err != nil {
		t.Fatal(err)
	}
	if !failRes.IsError || !strings.Contains(text(failRes), "not_found") {
		t.Fatalf("failing run = %s", text(failRes))
	}
}
