package codemode

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// The in-process caller keeps at most MaxResponseBytes of a response, so a
// big file read through the API is never buffered whole; readText reports
// the real size, and writeText refuses a non-string text instead of
// writing an empty file.
func TestCallerCapsAndTextChecks(t *testing.T) {
	big := strings.Repeat("x", MaxResponseBytes+10)
	var wrote []string
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "PUT" {
			wrote = append(wrote, r.URL.RawQuery)
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"path":"x"}`))
			return
		}
		w.Write([]byte(big[:len(big)/2]))
		w.Write([]byte(big[len(big)/2:]))
	})
	call := HandlerCaller(h, "", "")
	resp, err := call(context.Background(), "GET", "/v1/files/content", nil, nil, "")
	if err != nil || len(resp.Body) != MaxResponseBytes || resp.Size != int64(len(big)) || resp.Status != 200 {
		t.Fatalf("kept %d of %d (size %d, status %d): %v", len(resp.Body), len(big), resp.Size, resp.Status, err)
	}

	res := Run(context.Background(), `return await aitk.files.readText({ root: "training", path: "x/model.safetensors" })`, Options{Call: call})
	if res.Error == nil || !strings.Contains(res.Error.Message, "readText reads at most") {
		t.Fatalf("readText of a big file: %+v", res.Error)
	}
	res = Run(context.Background(), `return await aitk.files.writeText({ root: "data", path: "a.txt", text: 5, overwrite: true })`, Options{Call: call})
	if res.Error == nil || !strings.Contains(res.Error.Message, "text must be a string") || len(wrote) != 0 {
		t.Fatalf("writeText with a number: %+v, wrote %v", res.Error, wrote)
	}
}
