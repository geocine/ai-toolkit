package codemode

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

type recorded struct {
	method, path, query, body, ctype string
}

// fakeAPI answers like /v1 would, recording what it got.
func fakeAPI(t *testing.T, calls *[]recorded) Caller {
	return func(_ context.Context, method, path string, q url.Values, body []byte, ctype string) (Response, error) {
		*calls = append(*calls, recorded{method, path, q.Encode(), string(body), ctype})
		json200 := func(v any) (Response, error) {
			b, _ := json.Marshal(v)
			return Response{Status: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: b}, nil
		}
		switch {
		case method == "POST" && path == "/v1/datasets":
			var in map[string]any
			json.Unmarshal(body, &in)
			return json200(map[string]any{"name": in["name"], "images": 0})
		case method == "POST" && path == "/v1/jobs/j1/start":
			return json200(map[string]any{"id": "j1", "state": "queued"})
		case method == "GET" && path == "/v1/jobs/missing":
			return Response{Status: 404, Body: []byte(`{"error":{"code":"not_found","message":"job missing not found"}}`)}, nil
		case method == "GET" && strings.HasPrefix(path, "/v1/jobs/"):
			return json200(map[string]any{"id": strings.TrimPrefix(path, "/v1/jobs/"), "state": "completed"})
		case method == "GET" && path == "/v1/files/content":
			return Response{Status: 200, Body: []byte("a caption")}, nil
		case method == "DELETE":
			return Response{Status: 204}, nil
		}
		return json200(map[string]any{"items": []any{}})
	}
}

func TestTypeScriptAsyncAndCalls(t *testing.T) {
	var calls []recorded
	res := Run(context.Background(), `
interface Out { ds: string; state: string; caption: string; n: number }
const ds = await aitk.datasets.create({ name: "my set" });
const started = await aitk.jobs.start("j1");
const caption: string = await aitk.files.readText({ root: "datasets", path: "my set/a.txt" });
await aitk.datasets.setCaption("my set", "sub/a b.png", { caption: "hi" });
await aitk.jobs.delete("j1", { files: true });
const jobs = await Promise.all([aitk.jobs.get("a"), aitk.jobs.get({ id: "b" })]);
await sleep(10);
console.log("started", started.state, { n: jobs.length });
const out: Out = { ds: ds.name, state: started.state, caption, n: jobs.length };
return out;
`, Options{Call: fakeAPI(t, &calls)})
	if res.Error != nil {
		t.Fatalf("error: %+v\nlogs: %s", res.Error, res.Logs)
	}
	if string(res.Result) != `{"ds":"my set","state":"queued","caption":"a caption","n":2}` {
		t.Errorf("result = %s", res.Result)
	}
	if res.Logs != "started queued {\"n\":2}\n" {
		t.Errorf("logs = %q", res.Logs)
	}
	want := []recorded{
		{"POST", "/v1/datasets", "", `{"name":"my set"}`, "application/json"},
		{"POST", "/v1/jobs/j1/start", "", "{}", "application/json"},
		{"GET", "/v1/files/content", "path=my+set%2Fa.txt&root=datasets", "", ""},
		{"PUT", "/v1/datasets/my%20set/captions/sub/a%20b.png", "", `{"caption":"hi"}`, "application/json"},
		{"DELETE", "/v1/jobs/j1", "files=true", "", ""},
		{"GET", "/v1/jobs/a", "", "", ""},
		{"GET", "/v1/jobs/b", "", "", ""},
	}
	if len(calls) != len(want) {
		t.Fatalf("calls = %+v", calls)
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Errorf("call %d = %+v, want %+v", i, calls[i], want[i])
		}
	}
	if res.Calls != len(want) {
		t.Errorf("calls counted = %d", res.Calls)
	}
}

func TestAPIErrorMapsToAitkError(t *testing.T) {
	var calls []recorded
	res := Run(context.Background(), `
try { await aitk.jobs.get("missing"); } catch (e) {
  if (!(e instanceof AitkError)) throw new Error("not an AitkError");
  return { code: e.code, status: e.status, msg: e.message };
}`, Options{Call: fakeAPI(t, &calls)})
	if res.Error != nil {
		t.Fatalf("%+v", res.Error)
	}
	if string(res.Result) != `{"code":"not_found","status":404,"msg":"job missing not found"}` {
		t.Errorf("result = %s", res.Result)
	}
	// uncaught -> api_error
	res = Run(context.Background(), `await aitk.jobs.get("missing")`, Options{Call: fakeAPI(t, &calls)})
	if res.Error == nil || res.Error.Code != "api_error" || res.Error.Status != 404 || !strings.Contains(res.Error.Message, "not_found") {
		t.Errorf("error = %+v", res.Error)
	}
}

func TestWaitHelper(t *testing.T) {
	var calls []recorded
	res := Run(context.Background(), `return (await aitk.jobs.wait("j9", { poll_ms: 500 })).state`, Options{Call: fakeAPI(t, &calls)})
	if res.Error != nil || string(res.Result) != `"completed"` {
		t.Fatalf("%s %+v", res.Result, res.Error)
	}
}

func TestSyntaxError(t *testing.T) {
	res := Run(context.Background(), "const x: = 1", Options{})
	if res.Error == nil || res.Error.Code != "syntax_error" || !strings.Contains(res.Error.Message, "line 1") {
		t.Fatalf("%+v", res.Error)
	}
	res = Run(context.Background(), `import fs from "fs"`, Options{})
	if res.Error == nil || res.Error.Code != "syntax_error" {
		t.Fatalf("import should not be allowed: %+v", res.Error)
	}
}

func TestTimeout(t *testing.T) {
	start := time.Now()
	res := Run(context.Background(), `for (;;) {}`, Options{Timeout: 300 * time.Millisecond})
	if res.Error == nil || res.Error.Code != "timeout" {
		t.Fatalf("busy loop: %+v", res.Error)
	}
	res = Run(context.Background(), `await sleep(60000); return 1`, Options{Timeout: 300 * time.Millisecond})
	if res.Error == nil || res.Error.Code != "timeout" {
		t.Fatalf("sleep: %+v", res.Error)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("timeouts took %s", time.Since(start))
	}
}

func TestCallLimit(t *testing.T) {
	var calls []recorded
	res := Run(context.Background(), `for (let i = 0; i < 1000; i++) await aitk.jobs.get("x" + i)`, Options{Call: fakeAPI(t, &calls)})
	if res.Error == nil || res.Error.Code != "call_limit" || len(calls) != MaxCalls {
		t.Fatalf("%+v after %d calls", res.Error, len(calls))
	}
}

func TestLogCap(t *testing.T) {
	res := Run(context.Background(), `const s = "x".repeat(100000); for (let i = 0; i < 30; i++) console.log(s); return "done"`, Options{})
	if res.Error != nil || string(res.Result) != `"done"` {
		t.Fatalf("%+v", res.Error)
	}
	if len(res.Logs) > MaxLogBytes+100 || !strings.Contains(res.Logs, "truncated") {
		t.Fatalf("logs %d bytes", len(res.Logs))
	}
}

func TestNoHostAccess(t *testing.T) {
	res := Run(context.Background(), `return [typeof require, typeof process, typeof fetch, typeof setTimeout, typeof XMLHttpRequest, typeof Deno]`, Options{})
	if res.Error != nil || string(res.Result) != `["undefined","undefined","undefined","undefined","undefined","undefined"]` {
		t.Fatalf("%s %+v", res.Result, res.Error)
	}
}

func TestMissingArgument(t *testing.T) {
	var calls []recorded
	res := Run(context.Background(), `await aitk.jobs.get()`, Options{Call: fakeAPI(t, &calls)})
	if res.Error == nil || !strings.Contains(res.Error.Message, "missing argument id") || len(calls) != 0 {
		t.Fatalf("%+v", res.Error)
	}
	res = Run(context.Background(), `await aitk.jobs.create({ name: "x" })`, Options{Call: fakeAPI(t, &calls)})
	if res.Error == nil || !strings.Contains(res.Error.Message, "missing gpu_ids") {
		t.Fatalf("%+v", res.Error)
	}
}

func TestDeclarationCoversEveryOp(t *testing.T) {
	decl := Declaration("")
	for _, o := range Ops {
		if !strings.Contains(decl, "    "+o.Name+"(") {
			t.Errorf("declaration lacks %s.%s", o.Group, o.Name)
		}
	}
	// the declaration itself must be valid TypeScript
	if _, err := Transpile(decl + "\nreturn 1"); err != nil {
		t.Fatalf("declaration does not compile: %v", err)
	}
	if s := Declaration("checkpoint"); !strings.Contains(s, "checkpoints(") || strings.Contains(s, "datasets: {") {
		t.Errorf("search filter wrong:\n%s", s)
	}
}

// HandlerCaller works against a real http.Handler (what mcpserver uses).
func TestHandlerCaller(t *testing.T) {
	var gotAuth string
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"path": r.URL.Path, "body": string(b)})
	})
	call := HandlerCaller(h, "Bearer tok", "")
	res := Run(context.Background(), `return await aitk.templates.render({ arch: "sdxl" })`, Options{Call: call})
	if res.Error != nil || gotAuth != "Bearer tok" || !strings.Contains(string(res.Result), `/v1/templates/render`) {
		t.Fatalf("%s %+v auth=%q", res.Result, res.Error, gotAuth)
	}
	_ = httptest.NewRecorder
}
