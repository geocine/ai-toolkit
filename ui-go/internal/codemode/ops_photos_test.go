package codemode

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// PhotoTypesTS is studio/src/lib/photoTypes.ts without its comments and
// without `export`; only PhotoFaces.job is typed as the declared Job.
func TestPhotoTypesMatchStudio(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "studio", "src", "lib", "photoTypes.ts"))
	if err != nil {
		t.Fatal(err)
	}
	var want []string
	for _, line := range strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "//") || strings.TrimSpace(line) == "" {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		want = append(want, strings.Replace(line, "job: unknown | null;", "job: Job | null;", 1))
	}
	got := strings.Split(strings.TrimSpace(strings.ReplaceAll(PhotoTypesTS, "\r\n", "\n")), "\n")
	if len(got) != len(want) {
		t.Fatalf("PhotoTypesTS has %d lines, photoTypes.ts %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("PhotoTypesTS differs from photoTypes.ts at line %d:\n got %q\nwant %q", i+1, got[i], want[i])
		}
	}
}

func TestPhotoDeclaration(t *testing.T) {
	decl := Declaration("")
	for _, s := range []string{"type PhotoReview = ", "type UploadResult = ", "  photos: {\n",
		"assess(params: { /** at most 2000 */ items: PhotoMetrics[];", "checkPhotos(name: string, params?: {", "fixPhotos(name: string, params: {",
		"photoSettings(name: string, params?: {", "forgetPhotos(name: string): Promise<null>;",
		"Promise<UploadResult>;", "caption_job: Job | null; photo_job: Job | null };"} {
		if !strings.Contains(decl, s) {
			t.Errorf("declaration lacks %q", s)
		}
	}
	if _, err := Transpile(decl + "\nreturn 1"); err != nil {
		t.Fatalf("declaration does not compile: %v", err)
	}
	// the types come after typesTS, which declares Job
	if strings.Index(decl, "type PhotoReview = ") < strings.Index(decl, "type Job = ") {
		t.Error("PhotoTypesTS is written before typesTS")
	}
}

// Each photo op sends its params where the handler reads them.
func TestPhotoOpsCalls(t *testing.T) {
	var calls []recorded
	code := `
await aitk.datasets.checkPhotos("me", { resolution: [512, 768, 1024], kind: "person", wait_ms: 0, pick: 25 });
await aitk.datasets.fixPhotos("me", { ops: [{ op: "set_aside_flagged" }], apply: true });
await aitk.datasets.photoSettings("me", { reference: null, decisions: { "a.jpg": "keep" } });
await aitk.datasets.forgetPhotos("me");
await aitk.photos.assess({ items: [], dataset: "me" });
await aitk.datasets.uploadFiles("me", { files: [{ name: "a.jpg", base64: "eA==" }], on_conflict: "rename" });
return 1`
	res := Run(context.Background(), code, Options{Call: fakeAPI(t, &calls)})
	if res.Error != nil {
		t.Fatalf("%+v", res.Error)
	}
	want := []recorded{
		{method: "GET", path: "/v1/datasets/me/photos/check", query: "kind=person&pick=25&resolution=512%2C768%2C1024&wait_ms=0"},
		{method: "POST", path: "/v1/datasets/me/photos/fix", body: `{"apply":true,"ops":[{"op":"set_aside_flagged"}]}`},
		{method: "PUT", path: "/v1/datasets/me/photos/settings", body: `{"decisions":{"a.jpg":"keep"},"reference":null}`},
		{method: "DELETE", path: "/v1/datasets/me/photos/analysis"},
		{method: "POST", path: "/v1/photos/assess", body: `{"dataset":"me","items":[]}`},
		{method: "POST", path: "/v1/datasets/me/files", query: "on_conflict=rename"},
	}
	if len(calls) != len(want) {
		t.Fatalf("%d calls, want %d: %+v", len(calls), len(want), calls)
	}
	for i, w := range want {
		c := calls[i]
		if c.method != w.method || c.path != w.path || c.query != w.query || (w.body != "" && c.body != w.body) {
			t.Errorf("call %d: %s %s ?%s %s, want %s %s ?%s %s", i, c.method, c.path, c.query, c.body, w.method, w.path, w.query, w.body)
		}
	}
	if !strings.HasPrefix(calls[5].ctype, "multipart/form-data") {
		t.Errorf("uploadFiles content type %q", calls[5].ctype)
	}
}
