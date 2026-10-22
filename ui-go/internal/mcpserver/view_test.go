package mcpserver

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestView(t *testing.T) {
	dir := t.TempDir()
	datasets, training := filepath.Join(dir, "datasets"), filepath.Join(dir, "output")
	img := image.NewRGBA(image.Rect(0, 0, 1600, 400))
	for x := 0; x < 1600; x++ {
		img.Set(x, 10, color.RGBA{255, 0, 0, 255})
	}
	var b bytes.Buffer
	png.Encode(&b, img)
	os.MkdirAll(filepath.Join(datasets, "set"), 0o755)
	os.MkdirAll(filepath.Join(training, "job", "samples"), 0o755)
	os.WriteFile(filepath.Join(datasets, "set", "a.png"), b.Bytes(), 0o644)
	sample := filepath.Join(training, "job", "samples", "1__000000100_0.png")
	os.WriteFile(sample, b.Bytes(), 0o644)
	os.WriteFile(filepath.Join(dir, "secret.png"), b.Bytes(), 0o644)
	prev := viewRoots
	viewRoots = func(context.Context) ([]string, error) { return []string{datasets, training}, nil }
	t.Cleanup(func() { viewRoots = prev })

	mux := http.NewServeMux()
	Register(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()
	ctx := context.Background()
	session, err := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "1"}, nil).Connect(ctx,
		&mcp.StreamableClientTransport{Endpoint: srv.URL + "/mcp", MaxRetries: -1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "aitk_view", Arguments: map[string]any{
		"paths": []string{sample, "set/a.png", filepath.Join(dir, "secret.png"), "../secret.png"}, "max_px": 400}})
	if err != nil {
		t.Fatal(err)
	}
	var images int
	var texts []string
	for _, c := range res.Content {
		switch c := c.(type) {
		case *mcp.ImageContent:
			images++
			cfg, err := jpeg.DecodeConfig(bytes.NewReader(c.Data))
			if err != nil || c.MIMEType != "image/jpeg" || cfg.Width != 400 || cfg.Height != 100 {
				t.Fatalf("preview: %v %s %+v", err, c.MIMEType, cfg)
			}
		case *mcp.TextContent:
			texts = append(texts, c.Text)
		}
	}
	all := strings.Join(texts, "\n")
	if res.IsError || images != 2 || strings.Count(all, "not a file inside") != 2 || !strings.Contains(all, "1600x400, shown at 400x100") {
		t.Fatalf("view: %d images, error %v\n%s", images, res.IsError, all)
	}
	res, _ = session.CallTool(ctx, &mcp.CallToolParams{Name: "aitk_view", Arguments: map[string]any{
		"paths": []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"}}})
	if !res.IsError {
		t.Fatal("9 paths accepted")
	}
}
