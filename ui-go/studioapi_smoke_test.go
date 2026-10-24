package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/geocine/aitoolkit/internal/bg"
	"github.com/geocine/aitoolkit/internal/mcpserver"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/server"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/studioapi"
	"github.com/geocine/aitoolkit/internal/testutil"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// NewMux registers routes and loops in package globals: build it once.
var (
	muxOnce sync.Once
	theMux  *http.ServeMux
)

func testMux() *http.ServeMux {
	muxOnce.Do(func() {
		theMux = server.NewMux()
		theMux.Handle("/", studioAPIRoot())
	})
	return theMux
}

// Every background loop is reviewed for studio API mode: a new loop has to
// be listed as skipped or as safe to run.
func TestStudioAPILoopsReviewed(t *testing.T) {
	testMux()
	known := append(append([]string{}, studioapi.SkippedLoops...), studioapi.RunningLoops...)
	names := bg.Names()
	if len(names) == 0 {
		t.Fatal("no loops registered")
	}
	for _, n := range names {
		if !slices.Contains(known, n) {
			t.Errorf("bg loop %q is neither in studioapi.SkippedLoops nor in studioapi.RunningLoops", n)
		}
	}
}

func TestStudioAPISmoke(t *testing.T) {
	tmp := testutil.TempDir(t)
	// a checkout-like place the binary was started from: its datasets must
	// stay out of sight
	os.MkdirAll(filepath.Join(tmp, "datasets", "x"), 0o755)
	os.WriteFile(filepath.Join(tmp, "datasets", "x", "a.jpg"), []byte("x"), 0o644)
	cwd := filepath.Join(tmp, "ui-go")
	os.MkdirAll(cwd, 0o755)
	t.Chdir(cwd)
	saved := []string{paths.ToolkitRoot, paths.DefaultTrainFolder, paths.DefaultDatasetsFolder, paths.DefaultDataRoot,
		paths.DefaultModelsFolder, paths.DBPath}
	t.Cleanup(func() {
		paths.ToolkitRoot, paths.DefaultTrainFolder, paths.DefaultDatasetsFolder, paths.DefaultDataRoot, paths.DefaultModelsFolder, paths.DBPath =
			saved[0], saved[1], saved[2], saved[3], saved[4], saved[5]
	})
	paths.SetRoot(tmp)

	for _, k := range []string{"AI_TOOLKIT_AUTH", "AI_TOOLKIT_GO_DB", "AITK_ALLOWED_HOSTS", "AITK_STUDIO_API_DIR", "AITK_PUBLIC_URL"} {
		t.Setenv(k, "")
	}
	t.Cleanup(studioapi.SetForTest(true))
	data := filepath.Join(tmp, "studio-api")
	dir, err := openStudioAPI(data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		store.DB.Close()
		state.DB.Close()
	})
	if dir != data || paths.DBPath != filepath.Join(data, "aitk_db.db") {
		t.Fatalf("data dir %q, db %q", dir, paths.DBPath)
	}

	_, handler := newHandler(testMux())
	mcpserver.SetAPI(handler)
	t.Cleanup(func() { mcpserver.SetAPI(nil) })
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	noRedirect := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

	type reply struct {
		code int
		body []byte
		hdr  http.Header
	}
	do := func(method, path, host string) reply {
		t.Helper()
		req, _ := http.NewRequest(method, srv.URL+path, strings.NewReader("{}"))
		req.Header.Set("Content-Type", "application/json")
		if host != "" {
			req.Host = host
		}
		res, err := noRedirect.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		b, _ := io.ReadAll(res.Body)
		return reply{res.StatusCode, b, res.Header}
	}
	errCode := func(r reply) string {
		var e struct {
			Error struct{ Code, Message string } `json:"error"`
		}
		json.Unmarshal(r.body, &e)
		return e.Error.Code
	}

	r := do("GET", "/v1/health", "")
	var health struct{ Mode string }
	if json.Unmarshal(r.body, &health); r.code != 200 || health.Mode != "studio-api" {
		t.Fatalf("health: %d %s", r.code, r.body)
	}
	if r := do("GET", "/v1/providers", ""); r.code != 200 {
		t.Fatalf("providers: %d %s", r.code, r.body)
	}
	r = do("GET", "/v1/datasets", "")
	var sets struct{ Items []any }
	if err := json.Unmarshal(r.body, &sets); r.code != 200 || err != nil || sets.Items == nil || len(sets.Items) != 0 {
		t.Fatalf("datasets must be the data dir's empty list: %d %s", r.code, r.body)
	}
	if r := do("POST", "/v1/jobs", ""); r.code != 409 || errCode(r) != "conflict" || !strings.Contains(string(r.body), "studio API") {
		t.Fatalf("POST /v1/jobs: %d %s", r.code, r.body)
	}
	for _, p := range []string{"/v1/providers/vast/bogus", "/v1/nope"} {
		if r := do("GET", p, ""); r.code != 404 || errCode(r) != "not_found" {
			t.Errorf("GET %s: %d %s", p, r.code, r.body)
		}
	}
	if r := do("GET", "/", ""); r.code != http.StatusFound || r.hdr.Get("Location") != "/studio/" {
		t.Errorf("GET /: %d %v", r.code, r.hdr)
	}
	if r := do("GET", "/x", ""); r.code != 404 {
		t.Errorf("GET /x: %d", r.code)
	}
	if r := do("GET", "/v1/health", "evil.com:8675"); r.code != 403 || errCode(r) != "forbidden" {
		t.Errorf("rebinding Host: %d %s", r.code, r.body)
	}

	// MCP: code mode reaches the studio API's routes, and trainer ops get the
	// studio API sentence back
	ctx := context.Background()
	client := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "1"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: srv.URL + "/mcp", MaxRetries: -1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	run := func(code string) (*mcp.CallToolResult, string) {
		t.Helper()
		res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "aitk_run", Arguments: map[string]any{"code": code}})
		if err != nil {
			t.Fatal(err)
		}
		var b strings.Builder
		for _, c := range res.Content {
			if tc, ok := c.(*mcp.TextContent); ok {
				b.WriteString(tc.Text)
			}
		}
		return res, b.String()
	}
	// providers.list returns every provider kind, so at least one item
	listProviders := func(label string) {
		t.Helper()
		res, text := run(`const p = await aitk.providers.list(); return Array.isArray(p.items) ? p.items.length : -1;`)
		var out struct {
			Result any
			Error  any
		}
		if err := json.Unmarshal([]byte(text), &out); err != nil || res.IsError || out.Error != nil {
			t.Fatalf("providers.list over MCP (%s): %s", label, text)
		}
		if n, ok := out.Result.(float64); !ok || n < 1 {
			t.Fatalf("providers.list over MCP (%s) returned no provider kinds: %s", label, text)
		}
	}
	listProviders("loopback")
	if res, text := run(`return await aitk.jobs.create({ name: "x", gpu_ids: "0", config: {} });`); !res.IsError || !strings.Contains(text, "This server is the studio API") {
		t.Fatalf("jobs.create over MCP: %s", text)
	}
	// with a public URL (notification links) the in-process calls carry its
	// host: the operator declared it, so the Host check lets it through
	t.Setenv("AITK_PUBLIC_URL", "https://notify.example")
	listProviders("AITK_PUBLIC_URL set")
	t.Setenv("AITK_PUBLIC_URL", "")

	if runtime.GOOS == "windows" {
		return // file modes are not enforced on Windows
	}
	if st, _ := os.Stat(data); st.Mode().Perm() != 0o700 {
		t.Errorf("data dir mode %o", st.Mode().Perm())
	}
	for _, f := range []string{"aitk_db.db", "aitk_go.db"} {
		if st, err := os.Stat(filepath.Join(data, f)); err != nil {
			t.Error(err)
		} else if st.Mode().Perm() != 0o600 {
			t.Errorf("%s mode %o", f, st.Mode().Perm())
		}
	}
}
