package photocheck

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"golang.org/x/image/bmp"
	"golang.org/x/image/tiff"

	"github.com/geocine/aitoolkit/internal/imgmeta"
	"github.com/geocine/aitoolkit/internal/testutil"
)

var (
	update      = flag.Bool("update", false, "rewrite testdata/dhash_golden.json")
	fixturesOut = flag.String("fixtures-out", "", "TestWriteFixtures writes the photo fixtures to this folder")
)

// measureBytes measures an encoded image the way Measure does a file.
func measureBytes(t testing.TB, name string, data []byte) Metrics {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Measure(p, name)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return m
}

// deref prints an optional metric.
func deref(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

func hashes(t testing.TB, m Metrics) (uint64, uint64) {
	t.Helper()
	h, err := ParseHash(m.DHash)
	if err != nil {
		t.Fatal(err)
	}
	f, err := ParseHash(m.DHashFlip)
	if err != nil {
		t.Fatal(err)
	}
	return h, f
}

// TestMetrics checks the measurements behind each pixel flag: every
// fixture lands on the side of the section 4.3 threshold its flag needs.
// The judge's own table (fixture -> flags) is in judge_test.go.
func TestMetrics(t *testing.T) {
	const w, h = 1200, 900
	var sharps []float64
	scenes := make([]Metrics, 10)
	for i := range scenes {
		scenes[i] = measureBytes(t, fmt.Sprintf("s%d.jpg", i), testutil.JPEG(testutil.Scene(i+1, w, h), 90))
		sharps = append(sharps, scenes[i].Sharp)
	}
	slices.Sort(sharps)
	median := (sharps[4] + sharps[5]) / 2
	sc := testutil.Scene(1, w, h)
	base := scenes[0]

	for i, m := range scenes {
		if !m.Decoded || m.Error != "" {
			t.Fatalf("scene %d: decoded %v, error %q", i, m.Decoded, m.Error)
		}
		if m.Sharp < 100 {
			t.Errorf("scene %d: sharp %.1f, want a clearly sharp photo", i, m.Sharp)
		}
		if m.LumaMean < 40 || m.LumaMean > 220 || m.ClipHiCenter > 0.25 || m.LumaStd < 25 {
			t.Errorf("scene %d: exposure mean %.1f std %.1f clip_hi_center %.3f", i, m.LumaMean, m.LumaStd, m.ClipHiCenter)
		}
		if m.ChromaSpread < 3 || m.Colorful > 100 {
			t.Errorf("scene %d: chroma %.2f colorful %.1f, want everyday colour", i, m.ChromaSpread, m.Colorful)
		}
		if m.Bars > 0.04 || m.UIBand {
			t.Errorf("scene %d: bars %.3f ui_band %v", i, m.Bars, m.UIBand)
		}
		if m.Noise == nil || *m.Noise > 8 {
			t.Errorf("scene %d: noise %v", i, m.Noise)
		}
		if m.Detail == nil || *m.Detail > 40 {
			t.Errorf("scene %d: detail %v, want a native photo", i, m.Detail)
		}
	}

	t.Run("blurry", func(t *testing.T) {
		m := measureBytes(t, "blur.jpg", testutil.JPEG(testutil.Blur(sc, 3), 90))
		if !(m.Sharp < 0.35*median && m.Sharp < 25) {
			t.Errorf("sharp %.2f against median %.1f", m.Sharp, median)
		}
	})
	t.Run("dark", func(t *testing.T) {
		m := measureBytes(t, "dark.jpg", testutil.JPEG(testutil.Darken(sc), 90))
		if m.LumaMean >= 40 || m.Bars > 0.04 {
			t.Errorf("luma_mean %.1f bars %.3f", m.LumaMean, m.Bars)
		}
	})
	t.Run("bright", func(t *testing.T) {
		m := measureBytes(t, "bright.jpg", testutil.JPEG(testutil.Brighten(sc), 90))
		if !(m.LumaMean > 220 || m.ClipHiCenter > 0.25) || m.Bars > 0.04 {
			t.Errorf("luma_mean %.1f clip_hi_center %.3f bars %.3f", m.LumaMean, m.ClipHiCenter, m.Bars)
		}
	})
	t.Run("white backdrop is not bright and has no bars", func(t *testing.T) {
		m := measureBytes(t, "white.jpg", testutil.JPEG(testutil.WhiteBackdrop(900, 1200), 90))
		if m.LumaMean > 220 || m.ClipHiCenter > 0.25 || m.Bars != 0 {
			t.Errorf("luma_mean %.1f clip_hi_center %.3f bars %.3f", m.LumaMean, m.ClipHiCenter, m.Bars)
		}
		if m.LumaMean < 170 {
			t.Errorf("luma_mean %.1f, want a high-key photo", m.LumaMean)
		}
	})
	t.Run("flat", func(t *testing.T) {
		m := measureBytes(t, "flat.jpg", testutil.JPEG(testutil.Flatten(sc), 90))
		if m.LumaStd >= 25 {
			t.Errorf("luma_std %.1f", m.LumaStd)
		}
	})
	t.Run("gray and sepia", func(t *testing.T) {
		for name, img := range map[string]image.Image{"gray": testutil.Gray(sc), "sepia": testutil.Sepia(sc)} {
			m := measureBytes(t, name+".jpg", testutil.JPEG(img, 90))
			if m.ChromaSpread >= 3 {
				t.Errorf("%s: chroma_spread %.2f", name, m.ChromaSpread)
			}
		}
	})
	t.Run("strong colour", func(t *testing.T) {
		m := measureBytes(t, "sat.jpg", testutil.JPEG(testutil.Saturate(sc), 90))
		if m.Colorful <= 100 {
			t.Errorf("colorful %.1f", m.Colorful)
		}
	})
	t.Run("noisy", func(t *testing.T) {
		m := measureBytes(t, "noise.jpg", testutil.JPEG(testutil.Noise(sc, 12, 1), 90))
		if m.Noise == nil || *m.Noise <= 8 {
			t.Errorf("noise %v", m.Noise)
		}
	})
	t.Run("screenshot", func(t *testing.T) {
		m := measureBytes(t, "ui.png", testutil.PNG(testutil.UIScreen(1170, 2532)))
		if !m.UIBand || m.Width != 1170 || m.Height != 2532 || m.Exif != nil {
			t.Errorf("ui_band %v %dx%d exif %v", m.UIBand, m.Width, m.Height, m.Exif)
		}
		if m.Bars > 0.04 || m.ClipHiCenter > 0.25 {
			t.Errorf("bars %.3f clip_hi_center %.3f", m.Bars, m.ClipHiCenter)
		}
	})
	t.Run("a camera-less screen-size scene has no status bar", func(t *testing.T) {
		m := measureBytes(t, "frame.jpg", testutil.JPEG(testutil.Scene(3, 1920, 1080), 90))
		if m.UIBand || m.Exif != nil || m.Width != 1920 || m.Height != 1080 {
			t.Errorf("ui_band %v exif %v %dx%d", m.UIBand, m.Exif, m.Width, m.Height)
		}
	})
	t.Run("letterbox", func(t *testing.T) {
		m := measureBytes(t, "bars.jpg", testutil.JPEG(testutil.Letterbox(sc, 0.1, color.Black), 90))
		if m.Bars <= 0.04 {
			t.Errorf("bars %.3f", m.Bars)
		}
	})
	// every edge of an evenly underexposed photo is a near-black band that
	// runs a quarter of the way in, but nothing inside differs from it
	t.Run("evenly dark has no bars", func(t *testing.T) {
		m := measureBytes(t, "under.jpg", testutil.JPEG(testutil.Underexpose(sc), 90))
		if m.LumaMean >= 10 || m.Bars != 0 {
			t.Errorf("luma_mean %.1f bars %.3f", m.LumaMean, m.Bars)
		}
	})
	t.Run("a dark scene letterboxed still has bars", func(t *testing.T) {
		m := measureBytes(t, "dark_bars.jpg", testutil.JPEG(testutil.Letterbox(testutil.Darken(sc), 0.1, color.Black), 90))
		if m.LumaMean >= 40 || m.Bars <= 0.04 {
			t.Errorf("luma_mean %.1f bars %.3f", m.LumaMean, m.Bars)
		}
	})
	// bars wider than the scan reaches are told apart by the centre
	t.Run("pillarbox past a quarter of the width", func(t *testing.T) {
		m := measureBytes(t, "pillar.jpg", testutil.JPEG(testutil.Pillarbox(testutil.Scene(1, 600, 900), 1600, color.Black), 90))
		if m.Bars < 0.4 {
			t.Errorf("bars %.3f", m.Bars)
		}
	})
	t.Run("sizes", func(t *testing.T) {
		for _, sz := range [][2]int{{300, 400}, {700, 900}} {
			m := measureBytes(t, "small.jpg", testutil.JPEG(testutil.Scene(2, sz[0], sz[1]), 90))
			if m.Width != sz[0] || m.Height != sz[1] || !m.Decoded {
				t.Errorf("%v: %dx%d decoded %v", sz, m.Width, m.Height, m.Decoded)
			}
		}
	})
	t.Run("jpeg quality", func(t *testing.T) {
		m := measureBytes(t, "q40.jpg", testutil.JPEG(sc, 40))
		if m.JPEGQ <= 0 || m.JPEGQ >= 70 || base.JPEGQ < 87 {
			t.Errorf("jpeg_q %d (q90 copy %d)", m.JPEGQ, base.JPEGQ)
		}
	})
	t.Run("png pixel formats", func(t *testing.T) {
		g16 := measureBytes(t, "g16.png", testutil.Gray16PNG(sc))
		if g16.Mode != "gray16" || g16.PNGColorType != 0 || g16.PNGBitDepth != 16 || !g16.Decoded {
			t.Errorf("gray16: mode %q type %d depth %d", g16.Mode, g16.PNGColorType, g16.PNGBitDepth)
		}
		ga := measureBytes(t, "ga16.png", testutil.GrayAlpha16PNG(sc))
		if ga.Mode == "gray16" || ga.PNGColorType != 4 || ga.PNGBitDepth != 16 || !ga.Decoded || ga.Alpha != 0 {
			t.Errorf("grey+alpha 16: mode %q type %d depth %d alpha %.3f", ga.Mode, ga.PNGColorType, ga.PNGBitDepth, ga.Alpha)
		}
		rgba := measureBytes(t, "rgba16.png", testutil.RGBA16PNG(sc))
		if rgba.Mode != "rgb" || rgba.PNGColorType != 6 || rgba.Alpha != 0 || math.Abs(rgba.LumaMean-base.LumaMean) > 2 {
			t.Errorf("rgba16: mode %q type %d alpha %.3f mean %.1f", rgba.Mode, rgba.PNGColorType, rgba.Alpha, rgba.LumaMean)
		}
		tr := measureBytes(t, "transparent.png", testutil.TransparentPNG(sc))
		if tr.Alpha < 0.3 || tr.Alpha > 0.36 {
			t.Errorf("alpha_transparent %.3f, want about a third", tr.Alpha)
		}
	})
	t.Run("exif orientation", func(t *testing.T) {
		upright := measureBytes(t, "upright.jpg", testutil.JPEG(testutil.Rotate(sc, 1), 90))
		for name, jpg := range map[string][]byte{
			"II": testutil.WithExif(testutil.JPEG(sc, 90), 6, "2024:05:06 07:08:09", "Apple", "iPhone 15"),
			"MM": testutil.WithExifMM(testutil.JPEG(sc, 90), 6, "2024:05:06 07:08:09", "Apple", "iPhone 15"),
		} {
			m := measureBytes(t, "rot.jpg", jpg)
			if m.Orientation != 6 || m.Width != h || m.Height != w {
				t.Errorf("%s: orientation %d %dx%d", name, m.Orientation, m.Width, m.Height)
			}
			want := ExifInfo{Time: "2024-05-06T07:08:09", Make: "Apple", Model: "iPhone 15"}
			if m.Exif == nil || *m.Exif != want {
				t.Errorf("%s: exif %+v", name, m.Exif)
			}
			// The small copies are upright: they hash like the same
			// picture turned by hand and saved without EXIF.
			hr, _ := hashes(t, m)
			hb, _ := hashes(t, upright)
			if Hamming(hr, hb) > 3 || math.Abs(m.Sharp-upright.Sharp) > 0.05*upright.Sharp {
				t.Errorf("%s: dhash %s vs %s, sharp %.1f vs %.1f", name, m.DHash, upright.DHash, m.Sharp, upright.Sharp)
			}
		}
	})
	t.Run("upscaled", func(t *testing.T) {
		big := testutil.Scene(2, 3000, 2000)
		up := measureBytes(t, "up.jpg", testutil.JPEG(testutil.Upscaled(big, 2), 90))
		native := measureBytes(t, "native.jpg", testutil.JPEG(big, 90))
		if up.Detail == nil || *up.Detail <= 40 || native.Detail == nil || *native.Detail > 40 {
			t.Errorf("detail upscaled %v native %v", deref(up.Detail), deref(native.Detail))
		}
		if up.Width != native.Width || up.Height != native.Height {
			t.Errorf("size %dx%d vs %dx%d", up.Width, up.Height, native.Width, native.Height)
		}
	})
	t.Run("generated and animated headers", func(t *testing.T) {
		g := measureBytes(t, "gen.png", testutil.PNGWithText("parameters", "a photo of sks person, Steps: 30"))
		if !g.Generated || !g.Decoded {
			t.Errorf("generated %v decoded %v", g.Generated, g.Decoded)
		}
		a := measureBytes(t, "anim.webp", testutil.AnimatedWebPHeader())
		if !a.Animated || a.Format != "webp" || a.Width != 640 || a.Height != 480 || a.Decoded || a.Error == "" {
			t.Errorf("animated webp: %+v", a)
		}
	})
	t.Run("animated webp trains on frame 1", func(t *testing.T) {
		m := measureBytes(t, "anim2.webp", testutil.AnimatedWebP())
		if !m.Decoded || m.Error != "" || !m.Animated || m.Format != "webp" || m.Width != 32 || m.Height != 24 {
			t.Fatalf("animated webp: decoded %v error %q animated %v %dx%d", m.Decoded, m.Error, m.Animated, m.Width, m.Height)
		}
		if math.Abs(m.LumaMean-64) > 3 || math.Abs(m.Alpha-0.25) > 0.02 {
			t.Errorf("frame 1: luma_mean %.1f alpha %.3f, want grey 64 with a quarter transparent", m.LumaMean, m.Alpha)
		}
	})
	t.Run("lossy webp is full range", func(t *testing.T) {
		// VP8 stores Y 16-235: black and white must still clip.
		m := measureBytes(t, "halves.webp", testutil.WebPHalves())
		if !m.Decoded || math.Abs(m.ClipLo-0.5) > 0.07 || math.Abs(m.ClipHi-0.5) > 0.07 || math.Abs(m.LumaMean-127.5) > 4 {
			t.Errorf("halves: decoded %v clip_lo %.3f clip_hi %.3f luma_mean %.1f", m.Decoded, m.ClipLo, m.ClipHi, m.LumaMean)
		}
		a := measureBytes(t, "alpha.webp", testutil.WebPAlpha())
		if !a.Decoded || a.Mode != "rgb" || math.Abs(a.LumaMean-73.4) > 2 || math.Abs(a.Alpha-0.25) > 0.02 {
			t.Errorf("alpha: decoded %v mode %q luma_mean %.1f alpha %.3f", a.Decoded, a.Mode, a.LumaMean, a.Alpha)
		}
	})
	t.Run("other formats", func(t *testing.T) {
		small := testutil.Scene(1, 128, 96)
		ref := measureBytes(t, "ref.png", testutil.PNG(small))
		gray := image.NewGray(small.Bounds())
		draw.Draw(gray, gray.Bounds(), small, image.Point{}, draw.Src)
		var gifb, bmpb, tiffb bytes.Buffer
		if err := gif.Encode(&gifb, small, nil); err != nil {
			t.Fatal(err)
		}
		if err := bmp.Encode(&bmpb, small); err != nil {
			t.Fatal(err)
		}
		if err := tiff.Encode(&tiffb, small, nil); err != nil {
			t.Fatal(err)
		}
		for _, c := range []struct {
			name, format, mode string
			data               []byte
		}{
			{"s.gif", "gif", "paletted", gifb.Bytes()},
			{"s.bmp", "bmp", "rgb", bmpb.Bytes()},
			{"s.tiff", "tiff", "rgb", tiffb.Bytes()},
			{"gray.jpg", "jpeg", "gray", testutil.JPEG(gray, 95)},
			{"s.webp", "webp", "rgb", testutil.SceneWebP()},
		} {
			m := measureBytes(t, c.name, c.data)
			if !m.Decoded || m.Error != "" || m.Format != c.format || m.Mode != c.mode || m.Width != 128 || m.Height != 96 {
				t.Errorf("%s: decoded %v error %q format %q mode %q %dx%d", c.name, m.Decoded, m.Error, m.Format, m.Mode, m.Width, m.Height)
			}
			if math.Abs(m.LumaMean-ref.LumaMean) > 3 || math.Abs(m.LumaStd-ref.LumaStd) > 3 {
				t.Errorf("%s: luma mean %.1f std %.1f, png copy %.1f %.1f", c.name, m.LumaMean, m.LumaStd, ref.LumaMean, ref.LumaStd)
			}
			if c.format == "webp" && math.Abs(m.Colorful-ref.Colorful) > 0.1*ref.Colorful {
				t.Errorf("webp: colorful %.1f, png copy %.1f (chroma left in limited range?)", m.Colorful, ref.Colorful)
			}
		}
	})
	t.Run("files that do not decode", func(t *testing.T) {
		huge := measureBytes(t, "huge.png", testutil.RawPNG(20000, 20000, 2, 8, nil))
		if huge.Decoded || huge.Error != "too large to analyse" || huge.Width != 20000 || huge.Height != 20000 {
			t.Errorf("huge: decoded %v error %q %dx%d", huge.Decoded, huge.Error, huge.Width, huge.Height)
		}
		junk := measureBytes(t, "junk.jpg", []byte("not an image at all"))
		if junk.Decoded || junk.Error == "" || junk.Format != "other" || junk.SHA256 == "" {
			t.Errorf("junk: %+v", junk)
		}
		if _, err := Measure(filepath.Join(t.TempDir(), "missing.jpg"), "missing.jpg"); err == nil {
			t.Error("a missing file measured without an error")
		}
	})
	t.Run("file facts", func(t *testing.T) {
		data := testutil.JPEG(sc, 90)
		m := measureBytes(t, "facts.jpg", data)
		sum := sha256.Sum256(data)
		if m.SHA256 != hex.EncodeToString(sum[:]) || m.Bytes != int64(len(data)) || m.File != "facts.jpg" ||
			m.Version != MetricsVersion || m.Source != "server" || m.Format != "jpeg" || m.Orientation != 1 {
			t.Errorf("facts: %+v", m)
		}
		if len(DecodeThumb32(m.Thumb32)) != 1024 {
			t.Errorf("thumb32 is %d chars", len(m.Thumb32))
		}
		if l, _ := base64.StdEncoding.DecodeString(m.Layout); len(l) != 48 {
			t.Errorf("layout is %d bytes", len(l))
		}
		if LayoutSim(m.Layout, m.Layout) != 1 {
			t.Error("a layout is not identical to itself")
		}
	})
}

// TestMetricsSniffOnly is MeasureImage on an image whose header was not
// sniffed: the decoded size stands in.
func TestMetricsSniffOnly(t *testing.T) {
	m := MeasureImage(testutil.Scene(4, 320, 240), imgmeta.Sniff{Format: "png", Exif: imgmeta.Exif{Orientation: 1}}, "a.png", 10)
	if !m.Decoded || m.Width != 320 || m.Height != 240 || m.SHA256 != "" || m.Bytes != 10 {
		t.Errorf("%+v", m)
	}
}

func TestMetricsBoxDown(t *testing.T) {
	// a constant plane stays constant at any size
	src := make([]float32, 13*7)
	for i := range src {
		src[i] = 77
	}
	for _, sz := range [][2]int{{5, 3}, {9, 8}, {1, 1}, {13, 7}, {32, 32}} {
		for i, v := range BoxDown(src, 13, 7, sz[0], sz[1]) {
			if math.Abs(float64(v)-77) > 1e-4 {
				t.Fatalf("%v: out[%d] = %v", sz, i, v)
			}
		}
	}
	// an integer factor is the block mean
	g := []float32{0, 2, 4, 6, 1, 3, 5, 7}
	if got := BoxDown(g, 4, 2, 2, 1); !slices.Equal(got, []float32{1.5, 5.5}) {
		t.Errorf("2x2 blocks: %v", got)
	}
	// a fractional factor weighs the edge pixels: 3 -> 2 reads [0, 1.5)
	// and [1.5, 3)
	if got := BoxDown([]float32{0, 3, 6}, 3, 1, 2, 1); math.Abs(float64(got[0])-1) > 1e-6 || math.Abs(float64(got[1])-5) > 1e-6 {
		t.Errorf("3 -> 2: %v", got)
	}
	if BoxDown(nil, 0, 0, 1, 1) != nil || BoxDown(g, 4, 2, 0, 1) != nil {
		t.Error("an empty plane or size gave pixels")
	}
}

func TestMetricsOrient(t *testing.T) {
	// 3 x 2 stored:  a b c / d e f
	src := []float32{1, 2, 3, 4, 5, 6}
	want := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {3, 2, 1, 6, 5, 4},
		3: {6, 5, 4, 3, 2, 1},
		4: {4, 5, 6, 1, 2, 3},
		5: {1, 4, 2, 5, 3, 6},
		6: {4, 1, 5, 2, 6, 3},
		7: {6, 3, 5, 2, 4, 1},
		8: {3, 6, 2, 5, 1, 4},
	}
	for o, w := range want {
		got, ow, oh := orient(src, 3, 2, o)
		if !slices.Equal(got, w) {
			t.Errorf("orientation %d: %v, want %v", o, got, w)
		}
		if (o >= 5) != (ow == 2 && oh == 3) {
			t.Errorf("orientation %d: size %dx%d", o, ow, oh)
		}
	}
}

func TestMetricsLayoutSim(t *testing.T) {
	a := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{128, 128, 128}, 16))
	b := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{128, 153, 128}, 16)) // Δa 25 everywhere
	if s := LayoutSim(a, b); math.Abs(s-0.5) > 1e-9 {
		t.Errorf("ΔE 25: %v, want 0.5", s)
	}
	if LayoutSim(a, "") != 0 || LayoutSim("!!", a) != 0 {
		t.Error("a malformed layout compared")
	}
}

func TestNCC32(t *testing.T) {
	a := make([]byte, 1024)
	for i := range a {
		a[i] = byte(i%32*4 + i/32)
	}
	if v := NCC32(a, a); math.Abs(v-1) > 1e-9 {
		t.Errorf("self: %v", v)
	}
	inv := make([]byte, 1024)
	for i, v := range a {
		inv[i] = 255 - v
	}
	if v := NCC32(a, inv); math.Abs(v+1) > 1e-9 {
		t.Errorf("inverse: %v", v)
	}
	flat := bytes.Repeat([]byte{9}, 1024)
	if NCC32(a, flat) != 0 || NCC32(a, a[:10]) != 0 || NCC32(nil, nil) != 0 {
		t.Error("an undefined NCC was not 0")
	}
	m := Mirror32(a)
	if m[0] != a[31] || m[32] != a[63] || !slices.Equal(Mirror32(m), a) || Mirror32(a[:5]) != nil {
		t.Error("Mirror32 is not a left-right mirror")
	}
}

func TestDHashHex(t *testing.T) {
	for _, h := range []uint64{0, 1, 0x8000000000000000, 0xdeadbeef01234567} {
		s := HashHex(h)
		got, err := ParseHash(s)
		if err != nil || got != h || len(s) != 16 {
			t.Errorf("%x: %q %x %v", h, s, got, err)
		}
	}
	for _, bad := range []string{"", "123", "zzzzzzzzzzzzzzzz", "00000000000000000"} {
		if _, err := ParseHash(bad); err == nil {
			t.Errorf("ParseHash(%q) passed", bad)
		}
	}
	if Hamming(0, 0xFF) != 8 || Hamming(5, 5) != 0 {
		t.Error("Hamming")
	}
}

// TestDHash checks the duplicate hash against the edits it must survive
// and the unrelated photos it must keep apart.
func TestDHash(t *testing.T) {
	const w, h = 1200, 900
	for seed := 1; seed <= 4; seed++ {
		sc := testutil.Scene(seed, w, h)
		a := measureBytes(t, "a.jpg", testutil.JPEG(sc, 90))
		ha, _ := hashes(t, a)
		ta := DecodeThumb32(a.Thumb32)

		half := measureBytes(t, "half.jpg", testutil.JPEG(testutil.Resize(sc, w/2, h/2), 70))
		hh, _ := hashes(t, half)
		if d := Hamming(ha, hh); d > 5 {
			t.Errorf("seed %d: resized 50%% + q70 is %d bits away", seed, d)
		}
		if v := NCC32(ta, DecodeThumb32(half.Thumb32)); v < 0.95 {
			t.Errorf("seed %d: resized NCC %.3f", seed, v)
		}

		flip := measureBytes(t, "flip.jpg", testutil.JPEG(testutil.FlipH(sc), 90))
		hf, ff := hashes(t, flip)
		if d := Hamming(ha, hf); d <= 10 {
			t.Errorf("seed %d: the mirror is only %d bits away directly", seed, d)
		}
		if d := Hamming(ha, ff); d > 5 {
			t.Errorf("seed %d: the mirror is %d bits away mirrored", seed, d)
		}
		if v := NCC32(ta, Mirror32(DecodeThumb32(flip.Thumb32))); v < 0.95 {
			t.Errorf("seed %d: mirrored NCC %.3f", seed, v)
		}

		crop := measureBytes(t, "crop.jpg", testutil.JPEG(testutil.Crop(sc, 0.03), 90))
		hc, _ := hashes(t, crop)
		if d := Hamming(ha, hc); d > 8 {
			t.Errorf("seed %d: a 3%% crop is %d bits away", seed, d)
		}
	}

	// 100 pairs of unrelated scenes never come within the similar range.
	closest := 64
	for i := 0; i < 100; i++ {
		a := measureBytes(t, "x.jpg", testutil.JPEG(testutil.Scene(1000+2*i, 320, 240), 90))
		b := measureBytes(t, "y.jpg", testutil.JPEG(testutil.Scene(1001+2*i, 320, 240), 90))
		ha, _ := hashes(t, a)
		hb, fb := hashes(t, b)
		d := min(Hamming(ha, hb), Hamming(ha, fb))
		closest = min(closest, d)
		if d <= 10 {
			t.Errorf("seeds %d and %d are %d bits apart", 1000+2*i, 1001+2*i, d)
		}
	}
	t.Logf("closest unrelated pair: %d bits", closest)
}

// TestDuplicateConfirmNCC builds two different pictures with the same
// dHash: their 9x8 cell means order the same way, but inside the cells
// they differ. NCC on thumb32 tells them apart, so the pair is not a
// confirmed duplicate (the judge shows it as similar).
func TestDuplicateConfirmNCC(t *testing.T) {
	const w, h = 72, 64 // 8 x 8 pixel cells under the 9 x 8 grid
	a, b := make([]float32, w*h), make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ramp := float32(200 - 10*(x/8)) // falls left to right: every bit set
			qx, qy := x%8 < 4, y%8 < 4
			pa, pb := float32(-60), float32(-60)
			if qx == qy { // checks of 4 x 4 blocks
				pa = 60
			}
			if qx { // left half of the cell light, right half dark
				pb = 60
			}
			a[y*w+x], b[y*w+x] = ramp+pa, ramp+pb
		}
	}
	ha, _ := DHash(a, w, h)
	hb, _ := DHash(b, w, h)
	if d := Hamming(ha, hb); d > 5 {
		t.Fatalf("built grids are %d bits apart", d)
	}
	ta, tb := thumb32(a, w, h), thumb32(b, w, h)
	if v := max(NCC32(ta, tb), NCC32(ta, Mirror32(tb))); v >= 0.95 {
		t.Errorf("NCC %.3f confirms two different pictures", v)
	}
	if v := NCC32(ta, ta); v < 0.999 {
		t.Errorf("self NCC %.3f", v)
	}
}

// golden is testdata/dhash_golden.json, which studio/scripts/check-dhash.mjs
// also reads: the TS box filter and dHash must give the same numbers.
type golden struct {
	Comment   string      `json:"comment"`
	W         int         `json:"w"`
	H         int         `json:"h"`
	Grid      []float32   `json:"grid"`
	Box       []goldenBox `json:"box"`
	DHash     string      `json:"dhash"`
	DHashFlip string      `json:"dhash_flip"`
	Thumb32   string      `json:"thumb32"`
}

type goldenBox struct {
	W   int       `json:"w"`
	H   int       `json:"h"`
	Out []float32 `json:"out"`
}

const goldenPath = "testdata/dhash_golden.json"

// goldenGrid is a 64 x 48 luma grid with broad structure and fine
// variation, all whole numbers so both languages read it exactly.
func goldenGrid() []float32 {
	g := make([]float32, 64*48)
	s := uint32(12345)
	for y := 0; y < 48; y++ {
		for x := 0; x < 64; x++ {
			s = s*1664525 + 1013904223
			v := 128 + 70*math.Sin(float64(x)/9+float64(y)/23) + 30*math.Cos(float64(x*y)/150) + float64(s>>24)/16
			g[y*64+x] = float32(math.Round(math.Min(255, math.Max(0, v))))
		}
	}
	return g
}

func makeGolden() golden {
	g := golden{
		Comment: "photocheck golden: grid is w x h luma, row-major; box[i].out = BoxDown(grid, w, h, box[i].w, box[i].h); " +
			"dhash/dhash_flip = DHash(grid, w, h) as 16 hex chars; thumb32 = base64 of round(BoxDown(grid, w, h, 32, 32)) as uint8. " +
			"Regenerate: go test ./internal/photocheck -run TestWriteDHashGolden -update",
		W: 64, H: 48, Grid: goldenGrid(),
	}
	for _, sz := range [][2]int{{9, 8}, {32, 32}, {27, 20}, {5, 7}, {64, 48}} {
		g.Box = append(g.Box, goldenBox{W: sz[0], H: sz[1], Out: BoxDown(g.Grid, g.W, g.H, sz[0], sz[1])})
	}
	h, f := DHash(g.Grid, g.W, g.H)
	g.DHash, g.DHashFlip = HashHex(h), HashHex(f)
	g.Thumb32 = base64.StdEncoding.EncodeToString(thumb32(g.Grid, g.W, g.H))
	return g
}

// TestWriteDHashGolden rewrites the golden file with -update.
func TestWriteDHashGolden(t *testing.T) {
	if !*update {
		t.Skip("run with -update to rewrite " + goldenPath)
	}
	// One line per field and per box case keeps the file short and its
	// diffs readable.
	g := makeGolden()
	js := func(v any) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	var b bytes.Buffer
	fmt.Fprintf(&b, "{\n \"comment\": %s,\n \"w\": %d,\n \"h\": %d,\n \"grid\": %s,\n \"box\": [\n", js(g.Comment), g.W, g.H, js(g.Grid))
	for i, bx := range g.Box {
		sep := ","
		if i == len(g.Box)-1 {
			sep = ""
		}
		fmt.Fprintf(&b, "  %s%s\n", js(bx), sep)
	}
	fmt.Fprintf(&b, " ],\n \"dhash\": %s,\n \"dhash_flip\": %s,\n \"thumb32\": %s\n}\n", js(g.DHash), js(g.DHashFlip), js(g.Thumb32))
	if !json.Valid(b.Bytes()) {
		t.Fatal("golden is not valid JSON")
	}
	if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(goldenPath, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestGolden(t *testing.T) {
	b, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	var want golden
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	got := makeGolden()
	if !slices.Equal(want.Grid, got.Grid) {
		t.Fatal("the golden grid changed; rerun with -update only if the definition changed (and bump MetricsVersion)")
	}
	if len(want.Box) != len(got.Box) {
		t.Fatalf("%d box cases, want %d", len(got.Box), len(want.Box))
	}
	for i, bx := range want.Box {
		g := got.Box[i]
		if g.W != bx.W || g.H != bx.H || len(g.Out) != len(bx.Out) {
			t.Fatalf("box %d: %dx%d", i, g.W, g.H)
		}
		for j := range bx.Out {
			if math.Abs(float64(g.Out[j]-bx.Out[j])) > 1e-4 {
				t.Errorf("box %dx%d out[%d] = %v, golden %v", bx.W, bx.H, j, g.Out[j], bx.Out[j])
				break
			}
		}
	}
	if got.DHash != want.DHash || got.DHashFlip != want.DHashFlip || got.Thumb32 != want.Thumb32 {
		t.Errorf("hashes %s/%s, golden %s/%s (thumb32 equal: %v)", got.DHash, got.DHashFlip, want.DHash, want.DHashFlip,
			got.Thumb32 == want.Thumb32)
	}
}

// TestWriteFixtures writes the photo fixtures to -fixtures-out, for a
// manual pass: drop them into the studio and compare its flags with
// photos/check after upload.
func TestWriteFixtures(t *testing.T) {
	if *fixturesOut == "" {
		t.Skip("set -fixtures-out=<dir> to write the fixtures")
	}
	sc := testutil.Scene(1, 1600, 1200)
	files := map[string][]byte{
		"scene.jpg":          testutil.JPEG(sc, 90),
		"blurry.jpg":         testutil.JPEG(testutil.Blur(sc, 3), 90),
		"dark.jpg":           testutil.JPEG(testutil.Darken(sc), 90),
		"bright.jpg":         testutil.JPEG(testutil.Brighten(sc), 90),
		"white_backdrop.jpg": testutil.JPEG(testutil.WhiteBackdrop(1200, 1600), 90),
		"flat.jpg":           testutil.JPEG(testutil.Flatten(sc), 90),
		"gray.jpg":           testutil.JPEG(testutil.Gray(sc), 90),
		"sepia.jpg":          testutil.JPEG(testutil.Sepia(sc), 90),
		"saturated.jpg":      testutil.JPEG(testutil.Saturate(sc), 90),
		"noisy.jpg":          testutil.JPEG(testutil.Noise(sc, 12, 1), 90),
		"screenshot.png":     testutil.PNG(testutil.UIScreen(1170, 2532)),
		"frame_1080p.jpg":    testutil.JPEG(testutil.Scene(3, 1920, 1080), 90),
		"letterbox.jpg":      testutil.JPEG(testutil.Letterbox(sc, 0.1, color.Black), 90),
		"tiny.jpg":           testutil.JPEG(testutil.Scene(2, 300, 400), 90),
		"small.jpg":          testutil.JPEG(testutil.Scene(2, 700, 900), 90),
		"compressed.jpg":     testutil.JPEG(sc, 40),
		"gray16.png":         testutil.Gray16PNG(sc),
		"grayalpha16.png":    testutil.GrayAlpha16PNG(sc),
		"transparent.png":    testutil.TransparentPNG(sc),
		"rotated.jpg":        testutil.WithExif(testutil.JPEG(sc, 90), 6, "2024:05:06 07:08:09", "Apple", "iPhone 15"),
		"upscaled.jpg":       testutil.JPEG(testutil.Upscaled(testutil.Scene(2, 3000, 2000), 2), 90),
		"generated.png":      testutil.PNGWithText("parameters", "a photo"),
		"dup_half.jpg":       testutil.JPEG(testutil.Resize(sc, 800, 600), 70),
		"dup_mirror.jpg":     testutil.JPEG(testutil.FlipH(sc), 90),
	}
	if err := testutil.WriteTree(*fixturesOut, files); err != nil {
		t.Fatal(err)
	}
}

// BenchmarkMeasure12MP measures a 4000 x 3000 q90 scene end to end
// (section 17 budget: about 200 ms per core).
func BenchmarkMeasure12MP(b *testing.B) {
	p := filepath.Join(b.TempDir(), "12mp.jpg")
	if err := os.WriteFile(p, testutil.JPEG(testutil.Scene(9, 4000, 3000), 90), 0o644); err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := Measure(p, "12mp.jpg"); err != nil {
			b.Fatal(err)
		}
	}
}
