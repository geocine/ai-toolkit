package codemode

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// base64 params travel as raw bytes, files as multipart, Link ops sign
// instead of downloading.
func TestBinaryAndLinkModes(t *testing.T) {
	type got struct {
		method, path, query, ctype string
		body                       []byte
	}
	var calls []got
	call := func(_ context.Context, method, path string, q url.Values, body []byte, ctype string) (Response, error) {
		calls = append(calls, got{method, path, q.Encode(), ctype, body})
		out, _ := json.Marshal(map[string]any{"url": "https://node/x?sig=1", "path": path, "expires_at": "2026-01-01T00:00:00Z"})
		return Response{Status: 200, Body: out}, nil
	}
	png := []byte("\x89PNG fake")
	b64 := base64.StdEncoding.EncodeToString(png)
	res := Run(context.Background(), `
await aitk.datasets.putFile("set", "sub/a.png", { base64: "`+b64+`" });
await aitk.datasets.uploadFiles("set", { files: [{ name: "b.png", base64: "data:image/png;base64,`+b64+`" }] });
await aitk.bundles.uploadChunk("bupl_1", { offset: 1048576, base64: "`+b64+`" });
const link = await aitk.bundles.download("job1", { checkpoints: "latest", caches: false, ttl_s: 600 });
const file = await aitk.files.link({ root: "training", path: "job1/a b.safetensors" });
return [link.url, file.path];
`, Options{Call: call})
	if res.Error != nil {
		t.Fatalf("%+v", res.Error)
	}
	if len(calls) != 5 {
		t.Fatalf("calls: %+v", calls)
	}
	if c := calls[0]; c.method != "PUT" || c.path != "/v1/datasets/set/files/sub/a.png" || !bytes.Equal(c.body, png) || c.ctype != "application/octet-stream" {
		t.Fatalf("putFile: %+v", c)
	}
	mt, params, _ := mime.ParseMediaType(calls[1].ctype)
	if mt != "multipart/form-data" {
		t.Fatalf("uploadFiles content type %s", calls[1].ctype)
	}
	part, err := multipart.NewReader(bytes.NewReader(calls[1].body), params["boundary"]).NextPart()
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(part)
	if part.FormName() != "files" || part.FileName() != "b.png" || !bytes.Equal(data, png) {
		t.Fatalf("multipart part %s %s %q", part.FormName(), part.FileName(), data)
	}
	if c := calls[2]; c.path != "/v1/bundles/uploads/bupl_1" || c.query != "offset=1048576" || !bytes.Equal(c.body, png) {
		t.Fatalf("uploadChunk: %+v", c)
	}
	var link map[string]any
	json.Unmarshal(calls[3].body, &link)
	if c := calls[3]; c.method != "POST" || c.path != "/v1/links" || link["path"] != "/v1/jobs/job1/bundle?caches=false&checkpoints=latest" || link["ttl_s"] != float64(600) {
		t.Fatalf("bundles.download: %+v %v", c, link)
	}
	json.Unmarshal(calls[4].body, &link)
	if link["path"] != "/v1/files/content?path=job1%2Fa+b.safetensors&root=training" {
		t.Fatalf("files.link: %v", link)
	}

	for code, want := range map[string]string{
		`await aitk.datasets.putFile("s", "a.png", { base64: "!!!" })`:                "invalid_request",
		`await aitk.datasets.putFile("s", "a.png", { base64: "A".repeat(23000000) })`: "too_large",
		`await aitk.datasets.uploadFiles("s", { files: [] })`:                         "invalid_request",
	} {
		res := Run(context.Background(), code, Options{Call: call})
		if res.Error == nil || !strings.Contains(res.Error.Message, want) {
			t.Errorf("%s: %+v", code, res.Error)
		}
	}
}

// HandlerCaller presents the MCP client's origin as the request's host.
func TestHandlerCallerOrigin(t *testing.T) {
	var host, proto string
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host, proto = r.Host, r.Header.Get("X-Forwarded-Proto")
		w.Write([]byte(`{}`))
	})
	res := Run(context.Background(), `return await aitk.system.health()`, Options{Call: HandlerCaller(h, "", "https://pod-8675.proxy.example")})
	if res.Error != nil || host != "pod-8675.proxy.example" || proto != "https" {
		t.Fatalf("host %q proto %q %+v", host, proto, res.Error)
	}
	_ = httptest.NewRecorder
}

func TestBundleWaitHelper(t *testing.T) {
	n := 0
	call := func(_ context.Context, method, path string, q url.Values, body []byte, ctype string) (Response, error) {
		n++
		st := "running"
		if n > 1 {
			st = "done"
		}
		out, _ := json.Marshal(map[string]any{"id": strings.TrimPrefix(path, "/v1/bundles/pushes/"), "state": st})
		if !strings.HasPrefix(path, "/v1/bundles/pushes/") {
			t.Errorf("wait polled %s", path)
		}
		return Response{Status: 200, Body: out}, nil
	}
	res := Run(context.Background(), `return (await aitk.bundles.wait("bpsh_1", { poll_ms: 500 })).state`, Options{Call: call})
	if res.Error != nil || string(res.Result) != `"done"` {
		t.Fatalf("%s %+v", res.Result, res.Error)
	}
}
