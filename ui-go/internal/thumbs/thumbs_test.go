package thumbs

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestHasAlpha(t *testing.T) {
	opaquePal := &image.Paletted{Palette: color.Palette{color.RGBA{A: 255}}}
	clearPal := &image.Paletted{Palette: color.Palette{color.RGBA{A: 255}, color.RGBA{}}}
	for name, c := range map[string]struct {
		img  image.Image
		want bool
	}{
		"nrgba":          {image.NewNRGBA(image.Rect(0, 0, 1, 1)), true},
		"rgba":           {image.NewRGBA(image.Rect(0, 0, 1, 1)), false},
		"gray":           {image.NewGray(image.Rect(0, 0, 1, 1)), false},
		"ycbcr":          {image.NewYCbCr(image.Rect(0, 0, 2, 2), image.YCbCrSubsampleRatio420), false},
		"nycbcra":        {image.NewNYCbCrA(image.Rect(0, 0, 2, 2), image.YCbCrSubsampleRatio420), true},
		"opaque palette": {opaquePal, false},
		"clear palette":  {clearPal, true},
	} {
		if got := HasAlpha(c.img); got != c.want {
			t.Errorf("%s: HasAlpha = %v", name, got)
		}
	}
}

func TestEnsureFailureMemoized(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "broken.png")
	os.WriteFile(src, []byte("not a png"), 0o644)
	mtime := time.Now()
	if got := Ensure(src, Base(src), mtime); got != "" {
		t.Fatalf("broken source produced %q", got)
	}
	// replace with a valid image but keep the same mtime key: still memoized
	opaque := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for i := 3; i < len(opaque.Pix); i += 4 {
		opaque.Pix[i] = 0xff
	}
	f, _ := os.Create(src)
	png.Encode(f, opaque)
	f.Close()
	if got := Ensure(src, Base(src), mtime); got != "" {
		t.Fatalf("failure not memoized: %q", got)
	}
	// a new mtime retries
	if got := Ensure(src, Base(src), mtime.Add(time.Second)); got != Base(src)+".jpg" {
		t.Fatalf("retry = %q", got)
	}
	if tmp, _ := filepath.Glob(filepath.Join(dir, ".thumbs", "*.tmp*")); len(tmp) != 0 {
		t.Fatalf("tmp left behind: %v", tmp)
	}
}

func TestUnsupportedExtension(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "notes.txt")
	os.WriteFile(src, []byte("x"), 0o644)
	if got := Ensure(src, Base(src), time.Now()); got != "" {
		t.Fatalf("txt produced %q", got)
	}
}
