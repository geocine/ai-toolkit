package mcpserver

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/gif"
	"image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/geocine/aitoolkit/internal/safepath"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/thumbs"
)

// aitk_view limits.
const (
	viewMaxPaths    = 8
	viewDefaultPx   = 512
	viewMaxPx       = 1024
	viewMaxBytes    = 8 << 20 // all previews of one call
	viewMaxSource   = 256 << 20
	viewJPEGQuality = 82
)

// ViewInput is aitk_view's input.
type ViewInput struct {
	Paths []string `json:"paths" jsonschema:"image paths as the API returns them: a sample's path, a dataset's path + an item's file, or <dataset>/<file>"`
	MaxPx int      `json:"max_px,omitempty" jsonschema:"longest edge of each preview in pixels (default 512, max 1024)"`
}

// viewRoots are the folders aitk_view may read: datasets first (relative
// paths are tried there, then in the training folder), then the rest.
// Swappable so tests need no database.
var viewRoots = func(ctx context.Context) ([]string, error) {
	rt, err := settings.Roots(ctx)
	if err != nil {
		return nil, err
	}
	return []string{rt.Datasets, rt.Training, rt.Data}, nil
}

// resolveView maps a requested path to a file inside the allowed roots.
func resolveView(roots []string, p string) (string, error) {
	p = strings.TrimSpace(p)
	if p == "" {
		return "", fmt.Errorf("empty path")
	}
	var candidates []string
	if filepath.IsAbs(p) || strings.HasPrefix(p, "/") {
		candidates = []string{p}
	} else {
		for _, r := range roots {
			candidates = append(candidates, filepath.Join(r, filepath.FromSlash(p)))
		}
	}
	for _, c := range candidates {
		full := safepath.Resolve(c, roots)
		if full == "" {
			continue
		}
		if st, err := os.Stat(full); err == nil && st.Mode().IsRegular() {
			return full, nil
		}
	}
	return "", fmt.Errorf("not a file inside the datasets, training or data folders")
}

// preview decodes an image (a video or audio file: its generated thumb) and
// scales it to fit maxPx as JPEG.
func preview(full string, maxPx int) ([]byte, string, error) {
	st, err := os.Stat(full)
	if err != nil {
		return nil, "", err
	}
	src := full
	if _, _, err := imageConfig(full); err != nil {
		// not a decodable image: videos (and audio covers) have a thumb
		if t := thumbs.Ensure(full, thumbs.Base(full), st.ModTime()); t != "" {
			src = t
		} else {
			return nil, "", fmt.Errorf("not an image (and no thumbnail could be made)")
		}
	} else if st.Size() > viewMaxSource {
		return nil, "", fmt.Errorf("image is larger than %d MiB", viewMaxSource>>20)
	}
	data, err := os.ReadFile(src) // a source image is capped at viewMaxSource above
	if err != nil {
		return nil, "", err
	}
	img, err := thumbs.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, "", err
	}
	// Show a phone photo upright, as the studio grid and photos/check do. A
	// generated thumb is upright already and carries no EXIF.
	o := 1
	if src == full {
		o = thumbs.Orientation(data)
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if o >= 5 && o <= 8 {
		w, h = h, w
	}
	fit := thumbs.Orient(thumbs.Fit(img, maxPx), o)
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, fit, &jpeg.Options{Quality: viewJPEGQuality}); err != nil {
		return nil, "", err
	}
	note := fmt.Sprintf("%dx%d", w, h)
	if src != full {
		note = "thumbnail " + note
	}
	if fb := fit.Bounds(); fb.Dx() != w || fb.Dy() != h {
		note += fmt.Sprintf(", shown at %dx%d", fb.Dx(), fb.Dy())
	}
	return buf.Bytes(), note, nil
}

func imageConfig(p string) (image.Config, string, error) {
	f, err := os.Open(p)
	if err != nil {
		return image.Config{}, "", err
	}
	defer f.Close()
	return image.DecodeConfig(f)
}

func view(ctx context.Context, _ *mcp.CallToolRequest, in ViewInput) (*mcp.CallToolResult, any, error) {
	fail := func(msg string) (*mcp.CallToolResult, any, error) {
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: msg}}, IsError: true}, nil, nil
	}
	if len(in.Paths) == 0 {
		return fail("paths is required")
	}
	if len(in.Paths) > viewMaxPaths {
		return fail(fmt.Sprintf("at most %d paths per call", viewMaxPaths))
	}
	maxPx := in.MaxPx
	if maxPx <= 0 {
		maxPx = viewDefaultPx
	}
	maxPx = min(maxPx, viewMaxPx)
	roots, err := viewRoots(ctx)
	if err != nil {
		return nil, nil, err
	}
	var content []mcp.Content
	total, shown := 0, 0
	for _, p := range in.Paths {
		full, err := resolveView(roots, p)
		if err != nil {
			content = append(content, &mcp.TextContent{Text: p + ": " + err.Error()})
			continue
		}
		data, note, err := preview(full, maxPx)
		if err != nil {
			content = append(content, &mcp.TextContent{Text: p + ": " + err.Error()})
			continue
		}
		if total+len(data) > viewMaxBytes {
			content = append(content, &mcp.TextContent{Text: p + ": skipped, the call's previews would exceed 8 MiB (ask for fewer or a smaller max_px)"})
			continue
		}
		total += len(data)
		shown++
		content = append(content, &mcp.TextContent{Text: p + " (" + note + ")"}, &mcp.ImageContent{Data: data, MIMEType: "image/jpeg"})
	}
	return &mcp.CallToolResult{Content: content, IsError: shown == 0}, nil, nil
}
