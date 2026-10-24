package mcpserver

import (
	"bytes"
	"context"
	"encoding/binary"
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

	"github.com/geocine/aitoolkit/internal/imgmeta"
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

// A phone photo stored landscape with EXIF orientation 6 is previewed upright
// (portrait), with its upright size in the note.
func TestViewOrientation(t *testing.T) {
	stored := image.NewRGBA(image.Rect(0, 0, 400, 300))
	for y := 0; y < 300; y++ {
		for x := 0; x < 400; x++ {
			c := color.RGBA{255, 255, 255, 255}
			if x < 100 && y < 100 {
				c = color.RGBA{255, 0, 0, 255} // stored top-left: upright top-right
			}
			stored.Set(x, y, c)
		}
	}
	var b bytes.Buffer
	if err := jpeg.Encode(&b, stored, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}
	data, err := imgmeta.InjectJPEG(b.Bytes(), imgmeta.APP1(imgmeta.EncodeTIFF(imgmeta.Exif{Orientation: 6}, binary.BigEndian)))
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "IMG_0042.jpg")
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	out, note, err := preview(p, 200)
	if err != nil {
		t.Fatal(err)
	}
	if note != "300x400, shown at 150x200" {
		t.Fatalf("note %q", note)
	}
	img, err := jpeg.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatal(err)
	}
	if got := img.Bounds(); got.Dx() != 150 || got.Dy() != 200 {
		t.Fatalf("preview is %dx%d, want 150x200", got.Dx(), got.Dy())
	}
	red := func(x, y int) bool {
		r, g, _, _ := img.At(x, y).RGBA()
		return r > 0xc000 && g < 0x4000
	}
	if !red(130, 20) || red(20, 20) {
		t.Fatal("preview is not turned upright")
	}
	// at full size the note has no "shown at"
	if _, note, _ := preview(p, 1024); note != "300x400" {
		t.Fatalf("full-size note %q", note)
	}
}
