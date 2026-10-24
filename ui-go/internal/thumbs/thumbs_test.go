package thumbs

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/jpeg"
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

// orientationTIFF is a TIFF block holding only an orientation tag.
func orientationTIFF(o uint16) []byte {
	var tiff bytes.Buffer
	tiff.WriteString("II*\x00")
	binary.Write(&tiff, binary.LittleEndian, uint32(8)) // IFD0 offset
	binary.Write(&tiff, binary.LittleEndian, uint16(1)) // one entry
	binary.Write(&tiff, binary.LittleEndian, [4]uint16{0x0112, 3, 1, 0})
	binary.Write(&tiff, binary.LittleEndian, [2]uint16{o, 0})
	binary.Write(&tiff, binary.LittleEndian, uint32(0)) // no next IFD
	return tiff.Bytes()
}

// withOrientation inserts an Exif APP1 holding only an orientation tag right
// after the JPEG's SOI marker.
func withOrientation(jpg []byte, o uint16) []byte {
	payload := append([]byte("Exif\x00\x00"), orientationTIFF(o)...)
	seg := []byte{0xFF, 0xE1, 0, 0}
	binary.BigEndian.PutUint16(seg[2:], uint16(len(payload)+2))
	out := append([]byte(nil), jpg[:2]...)
	out = append(out, seg...)
	out = append(out, payload...)
	return append(out, jpg[2:]...)
}

// quadrants is a w x h image with a solid colour in each quarter.
func quadrants(w, h int, tl, tr, bl, br color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := tl
			switch {
			case x >= w/2 && y < h/2:
				c = tr
			case x < w/2 && y >= h/2:
				c = bl
			case x >= w/2 && y >= h/2:
				c = br
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

var (
	red   = color.RGBA{220, 20, 20, 255}
	green = color.RGBA{20, 200, 20, 255}
	blue  = color.RGBA{20, 20, 220, 255}
	white = color.RGBA{250, 250, 250, 255}
)

// near allows for JPEG loss.
func near(c color.Color, want color.RGBA) bool {
	r, g, b, _ := c.RGBA()
	within := func(got uint32, want uint8) bool {
		d := int(got>>8) - int(want)
		return d > -40 && d < 40
	}
	return within(r, want.R) && within(g, want.G) && within(b, want.B)
}

func checkQuadrants(t *testing.T, path string, tl, tr, bl, br color.RGBA) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != Size || b.Dy() != Size {
		t.Fatalf("thumb is %v", b)
	}
	q, r := Size/4, 3*Size/4
	for _, c := range []struct {
		x, y int
		want color.RGBA
		at   string
	}{{q, q, tl, "top left"}, {r, q, tr, "top right"}, {q, r, bl, "bottom left"}, {r, r, br, "bottom right"}} {
		if got := img.At(c.x, c.y); !near(got, c.want) {
			t.Errorf("%s = %v, want %v", c.at, got, c.want)
		}
	}
}

func writeJPEG(t *testing.T, path string, img image.Image, orientation uint16) {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	if orientation != 0 {
		data = withOrientation(data, orientation)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writePNG(t *testing.T, path string, img image.Image) {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestOrientedThumb(t *testing.T) {
	t.Run("orientation 6 gives an upright cover", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "phone.jpg")
		// stored landscape, shown portrait: a quarter turn clockwise puts the
		// stored bottom-left quarter at the top left
		writeJPEG(t, src, quadrants(600, 400, red, green, blue, white), 6)
		st, _ := os.Stat(src)
		got := Ensure(src, Base(src), st.ModTime())
		if got != Base(src)+".jpg" {
			t.Fatalf("Ensure = %q", got)
		}
		checkQuadrants(t, got, blue, red, white, green)
	})

	t.Run("no orientation tag keeps the stored layout", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "plain.jpg")
		writeJPEG(t, src, quadrants(600, 400, red, green, blue, white), 0)
		st, _ := os.Stat(src)
		checkQuadrants(t, Ensure(src, Base(src), st.ModTime()), red, green, blue, white)
	})

	t.Run("a source touched after its thumb regenerates", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "edit.png")
		writePNG(t, src, quadrants(40, 40, red, red, red, red))
		st, _ := os.Stat(src)
		first := Ensure(src, Base(src), st.ModTime())
		if first == "" {
			t.Fatal("no thumb")
		}
		if p, _ := Lookup(Base(src)); p != first {
			t.Fatalf("fresh Lookup = %q", p)
		}
		// a fresh thumb is reused, not rewritten
		before, _ := os.Stat(first)
		if got := Ensure(src, Base(src), st.ModTime()); got != first {
			t.Fatalf("second Ensure = %q", got)
		}
		if after, _ := os.Stat(first); !after.ModTime().Equal(before.ModTime()) {
			t.Fatal("fresh thumb was rewritten")
		}

		// the thumb now predates the source
		old := time.Now().Add(-time.Hour)
		os.Chtimes(first, old, old)
		writePNG(t, src, quadrants(40, 40, blue, blue, blue, blue))
		st, _ = os.Stat(src)
		if p, _ := Lookup(Base(src)); p != "" {
			t.Fatalf("stale Lookup = %q", p)
		}
		got := Ensure(src, Base(src), st.ModTime())
		if got != first {
			t.Fatalf("regenerated = %q", got)
		}
		checkQuadrants(t, got, blue, blue, blue, blue)
		if p, _ := Lookup(Base(src)); p != got {
			t.Fatalf("Lookup after regenerate = %q", p)
		}
	})

	t.Run("a thumb that changes suffix drops the old one", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "logo.png")
		writePNG(t, src, image.NewNRGBA(image.Rect(0, 0, 10, 10))) // has alpha
		st, _ := os.Stat(src)
		if got := Ensure(src, Base(src), st.ModTime()); got != Base(src)+".png" {
			t.Fatalf("alpha thumb = %q", got)
		}
		old := time.Now().Add(-time.Hour)
		os.Chtimes(Base(src)+".png", old, old)
		writePNG(t, src, quadrants(10, 10, red, red, red, red)) // opaque, encodes as RGB
		st, _ = os.Stat(src)
		if got := Ensure(src, Base(src), st.ModTime()); got != Base(src)+".jpg" {
			t.Fatalf("opaque thumb = %q", got)
		}
		if _, err := os.Stat(Base(src) + ".png"); !os.IsNotExist(err) {
			t.Fatalf("old .png thumb left: %v", err)
		}
		if p, _ := Lookup(Base(src)); p != Base(src)+".jpg" {
			t.Fatalf("Lookup = %q", p)
		}
	})

	t.Run("a source dated in the future does not leave its thumb stale", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "skew.png")
		writePNG(t, src, quadrants(10, 10, red, red, red, red))
		future := time.Now().Add(24 * time.Hour)
		os.Chtimes(src, future, future)
		got := Ensure(src, Base(src), future)
		if got == "" {
			t.Fatal("no thumb")
		}
		if p, _ := Lookup(Base(src)); p != got {
			t.Fatalf("Lookup = %q, want %q", p, got)
		}

		// edited now, which is before the future date the thumb carries
		writePNG(t, src, quadrants(10, 10, blue, blue, blue, blue))
		st, _ := os.Stat(src)
		if p, _ := Lookup(Base(src)); p != "" {
			t.Fatalf("Lookup after the edit = %q", p)
		}
		checkQuadrants(t, Ensure(src, Base(src), st.ModTime()), blue, blue, blue, blue)
	})

	t.Run("a source replaced with an older mtime regenerates", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "swap.png")
		writePNG(t, src, quadrants(10, 10, red, red, red, red))
		st, _ := os.Stat(src)
		if Ensure(src, Base(src), st.ModTime()) == "" {
			t.Fatal("no thumb")
		}
		// a copy that keeps its original date (Explorer, cp -p, unzip)
		writePNG(t, src, quadrants(10, 10, blue, blue, blue, blue))
		old := st.ModTime().Add(-48 * time.Hour)
		os.Chtimes(src, old, old)
		if p, _ := Lookup(Base(src)); p != "" {
			t.Fatalf("Lookup after the swap = %q", p)
		}
		checkQuadrants(t, Ensure(src, Base(src), old), blue, blue, blue, blue)
	})

	t.Run("a thumb made before orientation was applied is rebuilt upright", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "legacy.jpg")
		stored := quadrants(600, 400, red, green, blue, white)
		writeJPEG(t, src, stored, 6)
		old := time.Now().Add(-time.Hour)
		os.Chtimes(src, old, old)
		// the old generators cropped the stored pixels and wrote the thumb
		// after the photo, so it is newer and sideways
		os.MkdirAll(filepath.Dir(Base(src)), 0o755)
		writeJPEG(t, Base(src)+".jpg", Cover(stored, false), 0)
		if p, _ := Lookup(Base(src)); p != "" {
			t.Fatalf("legacy Lookup = %q", p)
		}
		got := Ensure(src, Base(src), old)
		if got != Base(src)+".jpg" {
			t.Fatalf("Ensure = %q", got)
		}
		checkQuadrants(t, got, blue, red, white, green)
		if p, _ := Lookup(Base(src)); p != got {
			t.Fatalf("Lookup after rebuild = %q", p)
		}
	})

	t.Run("a webp's exif after a large image chunk is read", func(t *testing.T) {
		chunk := func(fourcc string, data []byte) []byte {
			b := append([]byte(fourcc), 0, 0, 0, 0)
			binary.LittleEndian.PutUint32(b[4:], uint32(len(data)))
			b = append(b, data...)
			if len(data)%2 == 1 {
				b = append(b, 0)
			}
			return b
		}
		vp8x := make([]byte, 10)
		vp8x[0] = 0x08 // EXIF present
		body := []byte("WEBP")
		body = append(body, chunk("VP8X", vp8x)...)
		body = append(body, chunk("VP8L", make([]byte, 400<<10))...)
		body = append(body, chunk("EXIF", orientationTIFF(6))...)
		data := append([]byte("RIFF\x00\x00\x00\x00"), body...)
		binary.LittleEndian.PutUint32(data[4:], uint32(len(body)))
		if got := Orientation(data); got != 6 {
			t.Fatalf("Orientation = %d, want 6", got)
		}
	})

	t.Run("a video keeps an older generator's thumb when ffmpeg is missing", func(t *testing.T) {
		mu.Lock()
		was := ffmpegMissing
		ffmpegMissing = true
		mu.Unlock()
		t.Cleanup(func() {
			mu.Lock()
			ffmpegMissing = was
			mu.Unlock()
		})
		src := filepath.Join(t.TempDir(), "sample.mp4")
		os.WriteFile(src, []byte("not a video"), 0o644)
		os.MkdirAll(filepath.Dir(Base(src)), 0o755)
		writeJPEG(t, Base(src)+".jpg", quadrants(Size, Size, red, red, red, red), 0)
		st, _ := os.Stat(src)
		if got := Ensure(src, Base(src), st.ModTime()); got != Base(src)+".jpg" {
			t.Fatalf("Ensure = %q", got)
		}
	})

	t.Run("a thumb whose mtime the filesystem did not keep is fresh for this process", func(t *testing.T) {
		src := filepath.Join(t.TempDir(), "nfs.png")
		writePNG(t, src, quadrants(10, 10, red, red, red, red))
		st, _ := os.Stat(src)
		got := Ensure(src, Base(src), st.ModTime())
		if got == "" {
			t.Fatal("no thumb")
		}
		// as if Chtimes had been ignored: the thumb keeps its write time
		now := time.Now().Truncate(time.Second)
		os.Chtimes(got, now, now)
		ts, _ := os.Stat(got)
		mu.Lock()
		stamps[got] = stamp{source: st.ModTime(), thumb: ts.ModTime()}
		mu.Unlock()
		t.Cleanup(func() {
			mu.Lock()
			delete(stamps, got)
			mu.Unlock()
		})
		if p, _ := Lookup(Base(src)); p != got {
			t.Fatalf("Lookup = %q", p)
		}
	})
}

// Orient follows the EXIF definition: where the first stored pixel and the
// last pixel of the first stored row end up, and whether the sides swap.
func TestOrient(t *testing.T) {
	const w, h = 3, 2
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	first, rowEnd := color.NRGBA{R: 1, A: 255}, color.NRGBA{R: 2, A: 255}
	src.SetNRGBA(0, 0, first)
	src.SetNRGBA(w-1, 0, rowEnd)
	type pt struct{ x, y int }
	for o, c := range map[int]struct{ first, rowEnd pt }{
		1: {pt{0, 0}, pt{2, 0}},
		2: {pt{2, 0}, pt{0, 0}},
		3: {pt{2, 1}, pt{0, 1}},
		4: {pt{0, 1}, pt{2, 1}},
		5: {pt{0, 0}, pt{0, 2}},
		6: {pt{1, 0}, pt{1, 2}},
		7: {pt{1, 2}, pt{1, 0}},
		8: {pt{0, 2}, pt{0, 0}},
	} {
		out := Orient(src, o)
		if _, ok := out.(*image.NRGBA); !ok {
			t.Errorf("%d: pixel type %T", o, out)
		}
		wantW, wantH := w, h
		if o >= 5 {
			wantW, wantH = h, w
		}
		if b := out.Bounds(); b.Dx() != wantW || b.Dy() != wantH {
			t.Errorf("%d: bounds %v", o, b)
		}
		if got := out.At(c.first.x, c.first.y); got != first {
			t.Errorf("%d: first pixel not at %v (%v there)", o, c.first, got)
		}
		if got := out.At(c.rowEnd.x, c.rowEnd.y); got != rowEnd {
			t.Errorf("%d: row end not at %v (%v there)", o, c.rowEnd, got)
		}
	}
	for _, o := range []int{0, 9} {
		if Orient(src, o) != src {
			t.Errorf("orientation %d changed the image", o)
		}
	}
}
