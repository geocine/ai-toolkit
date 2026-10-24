package codemode

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ImageTypesTS is studio/src/lib/serverImageTypes.ts without its comments
// and without `export`, so the agent and the studio see the same shapes.
func TestImageTypesMatchStudio(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "studio", "src", "lib", "serverImageTypes.ts"))
	if err != nil {
		t.Fatal(err)
	}
	var want []string
	for _, line := range strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n") {
		if strings.HasPrefix(line, "//") || strings.TrimSpace(line) == "" {
			continue
		}
		want = append(want, strings.TrimPrefix(line, "export "))
	}
	got := strings.Split(strings.TrimSpace(ImageTypesTS), "\n")
	if len(got) != len(want) {
		t.Fatalf("ImageTypesTS has %d lines, serverImageTypes.ts %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("ImageTypesTS differs from serverImageTypes.ts at type %d:\n got %q\nwant %q", i+1, got[i], want[i])
		}
	}
}

func TestImageDeclaration(t *testing.T) {
	decl := Declaration("")
	for _, s := range []string{"type ImageCheck = ", "type HostFacts = ", "type CudaCheck = ", "  images: {\n", "  hosts: {\n",
		"check(params: { /** e.g. ghcr.io/geocine/aitoolkit-go:latest */ image: string;", "recommend(params?: {", "gpus(): Promise<Items<GpuInfo>>;", "Promise<HostFit>;"} {
		if !strings.Contains(decl, s) {
			t.Errorf("declaration lacks %q", s)
		}
	}
	if _, err := Transpile(decl + "\nreturn 1"); err != nil {
		t.Fatalf("declaration does not compile: %v", err)
	}
	for _, o := range Ops {
		if o.Group+"."+o.Name == "machines.launch" && !strings.Contains(o.Doc, "image_blocked") {
			t.Errorf("machines.launch doc does not say the server re-checks the image: %s", o.Doc)
		}
	}
}
