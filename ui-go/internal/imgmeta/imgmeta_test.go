package imgmeta

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"math/rand"
	"testing"
	"time"

	"golang.org/x/image/bmp"
)

// scene is a small image with enough detail for the encoders.
func scene(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	r := rand.New(rand.NewSource(1))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 255 / w), uint8(y * 255 / h), uint8(r.Intn(256)), 255})
		}
	}
	return img
}

func jpegOf(t testing.TB, img image.Image, q int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := jpeg.Encode(&b, img, &jpeg.Options{Quality: q}); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func pngOf(t testing.TB, img image.Image) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// pngChunk is one PNG chunk with its CRC.
func pngChunk(typ string, data []byte) []byte {
	b := binary.BigEndian.AppendUint32(nil, uint32(len(data)))
	b = append(b, typ...)
	b = append(b, data...)
	return binary.BigEndian.AppendUint32(b, crc32.ChecksumIEEE(b[4:]))
}

// pngWith inserts chunks right after IHDR.
func pngWith(p []byte, chunks ...[]byte) []byte {
	at := len(pngSig) + 8 + 13 + 4
	out := append([]byte{}, p[:at]...)
	for _, c := range chunks {
		out = append(out, c...)
	}
	return append(out, p[at:]...)
}

// pngHeader is a PNG that is only a signature and an IHDR.
func pngHeader(w, h, depth, ctype int) []byte {
	ihdr := binary.BigEndian.AppendUint32(nil, uint32(w))
	ihdr = binary.BigEndian.AppendUint32(ihdr, uint32(h))
	ihdr = append(ihdr, byte(depth), byte(ctype), 0, 0, 0)
	return append(append([]byte{}, pngSig...), pngChunk("IHDR", ihdr)...)
}

// riff builds a WebP file from chunks.
func riff(chunks ...[]byte) []byte {
	var body []byte
	for _, c := range chunks {
		body = append(body, c...)
	}
	b := []byte("RIFF")
	b = binary.LittleEndian.AppendUint32(b, uint32(4+len(body)))
	b = append(b, "WEBP"...)
	return append(b, body...)
}

func webpChunk(fourcc string, data []byte) []byte {
	b := append([]byte(fourcc), binary.LittleEndian.AppendUint32(nil, uint32(len(data)))...)
	b = append(b, data...)
	if len(data)&1 == 1 {
		b = append(b, 0)
	}
	return b
}

// vp8x is a VP8X payload with the given flags and canvas size.
func vp8x(flags byte, w, h int) []byte {
	d := []byte{flags, 0, 0, 0}
	w, h = w-1, h-1
	d = append(d, byte(w), byte(w>>8), byte(w>>16), byte(h), byte(h>>8), byte(h>>16))
	return d
}

// segment is a JPEG marker segment.
func segment(marker byte, payload []byte) []byte {
	b := []byte{0xFF, marker}
	b = binary.BigEndian.AppendUint16(b, uint16(len(payload)+2))
	return append(b, payload...)
}

// afterSOI inserts segments right after SOI.
func afterSOI(j []byte, segs ...[]byte) []byte {
	out := []byte{0xFF, 0xD8}
	for _, s := range segs {
		out = append(out, s...)
	}
	return append(out, j[2:]...)
}

func TestExif(t *testing.T) {
	base := jpegOf(t, scene(32, 24), 90)
	full := Exif{Time: "2024-05-06T07:08:09", SubSec: "123", Make: "Apple", Model: "iPhone 15 Pro", Software: "17.4.1"}
	for _, bo := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		for o := 1; o <= 8; o++ {
			e := full
			e.Orientation = o
			tiff := EncodeTIFF(e, bo)
			got, err := ParseTIFF(tiff)
			if err != nil || got != e {
				t.Fatalf("%v o=%d: ParseTIFF = %+v, %v", bo, o, got, err)
			}
			j, err := InjectJPEG(base, APP1(tiff))
			if err != nil {
				t.Fatal(err)
			}
			if got, ok := FromJPEG(j); !ok || got != e {
				t.Fatalf("%v o=%d: FromJPEG = %+v, %v", bo, o, got, ok)
			}
			if s := SniffHead(j); !s.HasExif || s.Exif != e || s.Width != 32 || s.Height != 24 {
				t.Fatalf("%v o=%d: SniffHead = %+v", bo, o, s)
			}
		}
	}

	// EXIF's own date layout is read into the ISO form; a blank clock is
	// no time; an unknown orientation reads as 1.
	tiff := EncodeTIFF(Exif{Time: "2001:02:03 04:05:06"}, binary.BigEndian)
	if got, _ := ParseTIFF(tiff); got.Time != "2001-02-03T04:05:06" || got.Orientation != 1 {
		t.Fatalf("EXIF layout: %+v", got)
	}
	if got, _ := ParseTIFF(EncodeTIFF(Exif{Time: "0000:00:00 00:00:00", Make: "X"}, binary.LittleEndian)); got.Time != "" {
		t.Fatalf("zero clock: %+v", got)
	}
	if got, _ := ParseTIFF(EncodeTIFF(Exif{Orientation: 9}, binary.LittleEndian)); got.Orientation != 1 {
		t.Fatalf("orientation 9: %+v", got)
	}
	// A fraction on Time stands in for SubSec.
	if got, _ := ParseTIFF(EncodeTIFF(Exif{Time: "2024-05-06T07:08:09.25"}, binary.LittleEndian)); got.SubSec != "25" {
		t.Fatalf("fraction: %+v", got)
	}

	// PNG eXIf and WebP EXIF (with and without the JPEG prefix).
	e6 := Exif{Orientation: 6, Make: "Google", Model: "Pixel 8"}
	p := pngWith(pngOf(t, scene(8, 8)), pngChunk("eXIf", EncodeTIFF(e6, binary.BigEndian)))
	if got, ok := FromPNG(p); !ok || got != e6 {
		t.Fatalf("FromPNG = %+v, %v", got, ok)
	}
	if s := SniffHead(p); !s.HasExif || s.Exif.Orientation != 6 {
		t.Fatalf("png sniff: %+v", s)
	}
	for _, prefix := range [][]byte{nil, exifPrefix} {
		w := riff(webpChunk("VP8X", vp8x(0x08, 640, 480)), webpChunk("EXIF", append(append([]byte{}, prefix...), EncodeTIFF(e6, binary.LittleEndian)...)))
		if got, ok := FromWebP(w); !ok || got != e6 {
			t.Fatalf("FromWebP(prefix %q) = %+v, %v", prefix, got, ok)
		}
		if s := SniffHead(w); s.Format != "webp" || s.Exif.Orientation != 6 || s.Width != 640 || s.Height != 480 {
			t.Fatalf("webp sniff: %+v", s)
		}
	}

	// Missing EXIF.
	if _, ok := FromJPEG(base); ok {
		t.Fatal("plain JPEG reported EXIF")
	}
	if _, ok := FromPNG(pngOf(t, scene(4, 4))); ok {
		t.Fatal("plain PNG reported EXIF")
	}
	if _, ok := FromWebP(riff(webpChunk("VP8X", vp8x(0, 4, 4)))); ok {
		t.Fatal("plain WebP reported EXIF")
	}
	if s := SniffHead(base); s.HasExif || s.Exif.Orientation != 1 {
		t.Fatalf("plain sniff: %+v", s)
	}

	// Every truncation of a file with EXIF: no panic, and never a wrong
	// orientation (a cut block yields 6 or nothing).
	j, _ := InjectJPEG(base, APP1(EncodeTIFF(Exif{Orientation: 6, Make: "M"}, binary.BigEndian)))
	for n := 0; n <= len(j); n++ {
		if e, ok := FromJPEG(j[:n]); ok && e.Orientation != 6 && e.Orientation != 1 {
			t.Fatalf("cut at %d: %+v", n, e)
		}
		SniffHead(j[:n])
	}
	for n := 0; n <= len(p); n++ {
		FromPNG(p[:n])
		SniffHead(p[:n])
	}
}

func TestSniff(t *testing.T) {
	img := scene(40, 30)

	// 16-bit PNGs: only colour type 0 is gray16.
	g16 := image.NewGray16(image.Rect(0, 0, 4, 4))
	rgb16 := image.NewRGBA64(image.Rect(0, 0, 4, 4))
	for i := 0; i < len(rgb16.Pix); i += 8 {
		rgb16.Pix[i+6], rgb16.Pix[i+7] = 0xFF, 0xFF
	}
	for name, c := range map[string]struct {
		file   []byte
		gray16 bool
		ctype  int
	}{
		"gray16":       {pngOf(t, g16), true, 0},
		"gray alpha16": {pngHeader(4, 4, 16, 4), false, 4},
		"rgb16":        {pngOf(t, rgb16), false, 2},
		"gray8":        {pngOf(t, image.NewGray(image.Rect(0, 0, 4, 4))), false, 0},
	} {
		s := SniffHead(c.file)
		if s.Gray16() != c.gray16 || s.PNGColorType != c.ctype || s.Format != "png" || s.Width != 4 {
			t.Errorf("%s: %+v gray16=%v", name, s, s.Gray16())
		}
	}

	// Generator text keys, and EXIF Software.
	plain := pngOf(t, img)
	for key, want := range map[string]bool{"parameters": true, "prompt": true, "workflow": true, "Comment": false, "Title": false} {
		p := pngWith(plain, pngChunk("tEXt", append([]byte(key+"\x00"), "a cat"...)))
		if s := SniffHead(p); s.Generated != want {
			t.Errorf("tEXt %q: generated = %v", key, s.Generated)
		}
	}
	if s := SniffHead(pngWith(plain, pngChunk("iTXt", []byte("parameters\x00\x00\x00\x00\x00steps: 20")))); !s.Generated {
		t.Error("iTXt parameters not generated")
	}
	// NovelAI names itself only in a Software text chunk.
	for name, c := range map[string]struct {
		chunk []byte
		want  bool
	}{
		"tEXt NovelAI":            {pngChunk("tEXt", []byte("Software\x00NovelAI")), true},
		"tEXt GIMP":               {pngChunk("tEXt", []byte("Software\x00GIMP")), false},
		"tEXt lower key":          {pngChunk("tEXt", []byte("software\x00NovelAI")), true},
		"iTXt NovelAI":            {pngChunk("iTXt", []byte("Software\x00\x00\x00en\x00\x00NovelAI")), true},
		"iTXt GIMP":               {pngChunk("iTXt", []byte("Software\x00\x00\x00\x00\x00GIMP 2.10")), false},
		"iTXt compressed NovelAI": {pngChunk("iTXt", []byte("Software\x00\x01\x00\x00\x00NovelAI")), false},
		"Comment NovelAI":         {pngChunk("tEXt", []byte("Comment\x00NovelAI")), false},
	} {
		if s := SniffHead(pngWith(plain, c.chunk)); s.Generated != c.want {
			t.Errorf("%s: generated = %v", name, s.Generated)
		}
	}
	base := jpegOf(t, img, 90)
	for sw, want := range map[string]bool{"ComfyUI": true, "Adobe Firefly": true, "DALL-E 3": true, "Stable Diffusion XL": true, "Adobe Photoshop 25.0": false, "HDR+ 1.0": false} {
		j, _ := InjectJPEG(base, APP1(EncodeTIFF(Exif{Software: sw}, binary.LittleEndian)))
		if s := SniffHead(j); s.Generated != want {
			t.Errorf("Software %q: generated = %v", sw, s.Generated)
		}
	}

	// C2PA: a manifest alone (as camera phones write) is not generated.
	jumbf := func(body string) []byte {
		box := []byte("\x00\x00\x00\x40jumb\x00\x00\x00\x20jumdc2pa\x00\x11\x00\x10")
		return segment(0xEB, append(append([]byte("JP\x00\x01\x00\x00\x00\x01"), box...), body...))
	}
	if s := SniffHead(afterSOI(base, jumbf("c2pa.actions digitalSourceType http://cv.iptc.org/newscodes/digitalsourcetype/computationalCapture"))); s.Generated || s.C2PAGenerated {
		t.Errorf("camera C2PA: %+v", s)
	}
	if s := SniffHead(afterSOI(base, jumbf("c2pa.actions digitalSourceType http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"))); !s.Generated || !s.C2PAGenerated {
		t.Errorf("model C2PA: %+v", s)
	}

	// WebP animation, by the VP8X flag or an ANIM chunk; VP8 and VP8L sizes.
	if s := SniffHead(riff(webpChunk("VP8X", vp8x(0x02, 100, 50)))); !s.Animated || s.Width != 100 || s.Height != 50 {
		t.Errorf("VP8X anim: %+v", s)
	}
	if s := SniffHead(riff(webpChunk("VP8X", vp8x(0, 10, 10)), webpChunk("ANIM", make([]byte, 6)))); !s.Animated {
		t.Errorf("ANIM: %+v", s)
	}
	if s := SniffHead(riff(webpChunk("VP8X", vp8x(0x10, 10, 10)))); s.Animated {
		t.Errorf("still WebP animated: %+v", s)
	}
	vp8 := []byte{0, 0, 0, 0x9D, 0x01, 0x2A}
	vp8 = binary.LittleEndian.AppendUint16(vp8, 320)
	vp8 = binary.LittleEndian.AppendUint16(vp8, 200)
	if s := SniffHead(riff(webpChunk("VP8 ", vp8))); s.Width != 320 || s.Height != 200 || s.Animated {
		t.Errorf("VP8: %+v", s)
	}
	vp8l := binary.LittleEndian.AppendUint32([]byte{0x2F}, uint32(199)|uint32(99)<<14)
	if s := SniffHead(riff(webpChunk("VP8L", vp8l))); s.Width != 200 || s.Height != 100 {
		t.Errorf("VP8L: %+v", s)
	}
	if s := SniffHead(pngWith(plain, pngChunk("acTL", make([]byte, 8)))); !s.Animated {
		t.Errorf("APNG: %+v", s)
	}

	// SOF component count: 3 for stdlib colour, 1 for grey, 4 for CMYK.
	if s := SniffHead(base); s.JPEGComponents != 3 || s.Width != 40 || s.Height != 30 {
		t.Errorf("colour JPEG: %+v", s)
	}
	if s := SniffHead(jpegOf(t, image.NewGray(image.Rect(0, 0, 8, 8)), 90)); s.JPEGComponents != 1 {
		t.Errorf("grey JPEG: %+v", s)
	}
	sof := []byte{8, 0, 16, 0, 24, 4, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1, 4, 0x11, 0}
	cmyk := append(append([]byte{0xFF, 0xD8}, segment(0xC0, sof)...), 0xFF, 0xDA)
	if s := SniffHead(cmyk); s.JPEGComponents != 4 || s.Width != 24 || s.Height != 16 {
		t.Errorf("CMYK: %+v", s)
	}

	// DQT quality for stdlib JPEGs, within 3.
	for _, q := range []int{50, 75, 90} {
		if s := SniffHead(jpegOf(t, img, q)); s.JPEGQuality < q-3 || s.JPEGQuality > q+3 {
			t.Errorf("q%d: estimated %d", q, s.JPEGQuality)
		}
	}
	if JPEGQuality(nil) != 0 || JPEGQuality([]byte{0x01}) != 0 {
		t.Error("empty or chroma-only DQT should be unknown")
	}

	// Formats and their stored sizes.
	var gb, bb bytes.Buffer
	gif.Encode(&gb, image.NewPaletted(image.Rect(0, 0, 7, 5), color.Palette{color.Black, color.White}), nil)
	bmp.Encode(&bb, image.NewRGBA(image.Rect(0, 0, 9, 6)))
	tiffHead := EncodeTIFF(Exif{Make: "Scanner"}, binary.LittleEndian)
	for name, c := range map[string]struct {
		file   []byte
		format string
		w, h   int
	}{
		"gif":           {gb.Bytes(), "gif", 7, 5},
		"bmp":           {bb.Bytes(), "bmp", 9, 6},
		"tiff":          {tiffHead, "tiff", 0, 0},
		"heic":          {ftyp("heic", "mif1", "heic"), "heic", 0, 0},
		"heif mif1":     {ftyp("mif1", "mif1"), "heif", 0, 0},
		"avif":          {ftyp("avif", "mif1", "avif", "miaf"), "avif", 0, 0},
		"jxl stream":    {[]byte{0xFF, 0x0A, 0, 0}, "jxl", 0, 0},
		"jxl container": {[]byte("\x00\x00\x00\x0CJXL \r\n\x87\n\x00\x00"), "jxl", 0, 0},
		"mp4":           {ftyp("isom", "isom", "mp41"), "other", 0, 0},
		"text":          {[]byte("hello"), "other", 0, 0},
		"empty":         {nil, "other", 0, 0},
	} {
		s := SniffHead(c.file)
		if s.Format != c.format || s.Width != c.w || s.Height != c.h || s.Exif.Orientation != 1 {
			t.Errorf("%s: %+v", name, s)
		}
	}
	if s := SniffHead(tiffHead); !s.HasExif || s.Exif.Make != "Scanner" {
		t.Errorf("tiff exif: %+v", s)
	}
}

// ftyp is a ftyp box with a major brand and compatible brands.
func ftyp(major string, compat ...string) []byte {
	body := append([]byte(major), 0, 0, 0, 0)
	for _, c := range compat {
		body = append(body, c...)
	}
	return box("ftyp", body)
}

func box(typ string, body []byte) []byte {
	b := binary.BigEndian.AppendUint32(nil, uint32(8+len(body)))
	b = append(b, typ...)
	return append(b, body...)
}

func fullBox(typ string, version byte, flags uint32, body []byte) []byte {
	h := []byte{version, byte(flags >> 16), byte(flags >> 8), byte(flags)}
	return box(typ, append(h, body...))
}

func infe(id uint16, typ string) []byte {
	b := binary.BigEndian.AppendUint16(nil, id)
	b = append(b, 0, 0) // protection index
	b = append(b, typ...)
	return fullBox("infe", 2, 0, append(b, 0)) // empty item name
}

// heicFile builds a HEIC whose Exif item (id 2) holds payload. With idat
// the payload sits in meta/idat (construction method 1); otherwise in mdat,
// found through an iloc base offset plus an extent offset.
func heicFile(payload []byte, idat bool) []byte {
	head := ftyp("heic", "mif1", "heic")
	iinf := fullBox("iinf", 0, 0, append(binary.BigEndian.AppendUint16(nil, 2),
		append(infe(1, "hvc1"), infe(2, "Exif")...)...))
	ispe := fullBox("ispe", 0, 0, binary.BigEndian.AppendUint32(binary.BigEndian.AppendUint32(nil, 4032), 3024))
	ipma := fullBox("ipma", 0, 0, []byte{0, 0, 0, 1, 0, 1, 1, 0x81})
	iprp := box("iprp", append(box("ipco", ispe), ipma...))
	pitm := fullBox("pitm", 0, 0, []byte{0, 1})
	hdlr := fullBox("hdlr", 0, 0, append(make([]byte, 4), "pict\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"...))

	// iloc v1: offset_size 4, length_size 4, base_offset_size 4, index 0.
	iloc := func(base, off uint32, method byte) []byte {
		b := []byte{0x44, 0x40}
		b = binary.BigEndian.AppendUint16(b, 2) // items
		// item 1: the image, a dummy extent
		b = append(b, 0, 1, 0, 0, 0, 0)
		b = binary.BigEndian.AppendUint32(b, 0)
		b = append(b, 0, 1)
		b = binary.BigEndian.AppendUint32(b, 0)
		b = binary.BigEndian.AppendUint32(b, 1)
		// item 2: Exif
		b = append(b, 0, 2, 0, method, 0, 0)
		b = binary.BigEndian.AppendUint32(b, base)
		b = append(b, 0, 1)
		b = binary.BigEndian.AppendUint32(b, off)
		b = binary.BigEndian.AppendUint32(b, uint32(len(payload)))
		return fullBox("iloc", 1, 0, b)
	}
	build := func(il []byte, extra []byte) []byte {
		meta := fullBox("meta", 0, 0, bytes.Join([][]byte{hdlr, pitm, iinf, il, iprp, extra}, nil))
		return append(append([]byte{}, head...), meta...)
	}
	if idat {
		return build(iloc(0, 3, 1), box("idat", append([]byte{9, 9, 9}, payload...)))
	}
	// The meta box size does not depend on the offsets, so lay it out once
	// to learn where mdat starts.
	pre := build(iloc(0, 0, 0), nil)
	mdatData := len(pre) + 8
	f := build(iloc(uint32(mdatData), 5, 0), nil)
	return append(f, box("mdat", append([]byte("xxxxx"), payload...))...)
}

func TestHEIFExif(t *testing.T) {
	want := Exif{Orientation: 1, Time: "2023-11-12T13:14:15", SubSec: "042", Make: "Apple", Model: "iPhone 14"}
	tiff := EncodeTIFF(want, binary.BigEndian)
	payload := append([]byte{0, 0, 0, 6}, append(append([]byte{}, exifPrefix...), tiff...)...)
	for _, idat := range []bool{false, true} {
		f := heicFile(payload, idat)
		got, err := HEIFExif(bytes.NewReader(f), int64(len(f)))
		if err != nil || !bytes.Equal(got, tiff) {
			t.Fatalf("idat=%v: HEIFExif = %x, %v", idat, got, err)
		}
		if !IsHEIF(f) {
			t.Fatal("IsHEIF false")
		}
		s := SniffHead(f)
		if s.Format != "heic" || s.Width != 4032 || s.Height != 3024 || !s.HasExif || s.Exif != want {
			t.Fatalf("idat=%v: sniff %+v", idat, s)
		}
		// Every truncation errors or succeeds, and never panics.
		for n := 0; n < len(f); n++ {
			HEIFExif(bytes.NewReader(f[:n]), int64(n))
			SniffHead(f[:n])
		}
	}
	// A zero TIFF-header offset works too.
	f := heicFile(append([]byte{0, 0, 0, 0}, tiff...), false)
	if got, err := HEIFExif(bytes.NewReader(f), int64(len(f))); err != nil || !bytes.Equal(got, tiff) {
		t.Fatalf("offset 0: %v", err)
	}

	r := rand.New(rand.NewSource(7))
	for i := 0; i < 200; i++ {
		g := make([]byte, r.Intn(512))
		r.Read(g)
		if i%2 == 0 && len(g) > 32 {
			copy(g, ftyp("heic", "mif1"))
		}
		if _, err := HEIFExif(bytes.NewReader(g), int64(len(g))); err == nil {
			t.Fatalf("garbage %d gave no error", i)
		}
	}
	if _, err := HEIFExif(bytes.NewReader(ftyp("heic")), 20); err == nil {
		t.Fatal("no meta gave no error")
	}
	if IsHEIF([]byte("not heif at all!")) {
		t.Fatal("IsHEIF on text")
	}
}

// An iloc whose offset, length and index sizes are all 0 reads no bytes per
// extent, so a 0xFFFF extent count on each of 4096 items must not loop.
func TestHEIFZeroSizeILOC(t *testing.T) {
	b := []byte{0x00, 0x00} // offset, length, base and index sizes 0
	b = binary.BigEndian.AppendUint16(b, maxHEIFItems)
	for id := 1; id <= maxHEIFItems; id++ {
		b = binary.BigEndian.AppendUint16(b, uint16(id))
		b = append(b, 0, 0, 0xFF, 0xFF) // data_reference_index, extent_count
	}
	iloc := fullBox("iloc", 0, 0, b)
	f := append(ftyp("heic", "mif1"), fullBox("meta", 0, 0, iloc)...)

	meta, err := readMetaBox(bytes.NewReader(f), int64(len(f)))
	if err != nil {
		t.Fatal(err)
	}
	m, err := parseHEIFMeta(meta)
	if err != nil || len(m.items) != maxHEIFItems {
		t.Fatalf("items = %d, %v", len(m.items), err)
	}
	for id, it := range m.items {
		if len(it.extents) > 1 {
			t.Fatalf("item %d: %d extents", id, len(it.extents))
		}
	}
	start := time.Now()
	SniffHead(f)
	HEIFExif(bytes.NewReader(f), int64(len(f)))
	if d := time.Since(start); d > 50*time.Millisecond {
		t.Fatalf("zero-size iloc took %v", d)
	}

	// Sizes 4 with an extent count far past the box's bytes stops too.
	b = []byte{0x44, 0x00}
	b = binary.BigEndian.AppendUint16(b, 1)
	b = append(b, 0, 1, 0, 0, 0xFF, 0xFF)
	m, _ = parseHEIFMeta(append([]byte{0, 0, 0, 0}, fullBox("iloc", 0, 0, b)...))
	if it := m.items[1]; it != nil && it.located {
		t.Fatalf("short iloc located: %+v", it)
	}
}

func TestInjectJPEG(t *testing.T) {
	base := jpegOf(t, scene(48, 32), 85)
	in := Exif{Orientation: 6, Time: "2022-01-02T03:04:05", SubSec: "7", Make: "Apple", Model: "iPhone 13", Software: "16.0"}
	j, err := InjectJPEG(base, BuildAPP1(in))
	if err != nil {
		t.Fatal(err)
	}
	got, ok := FromJPEG(j)
	want := Exif{Orientation: 1, Time: in.Time, SubSec: in.SubSec, Make: in.Make, Model: in.Model}
	if !ok || got != want {
		t.Fatalf("round trip = %+v, %v", got, ok)
	}
	tiff := j[4+2+len(exifPrefix):]
	if bytes.Contains(tiff[:min(len(tiff), 200)], []byte{0x12, 0x01, 0x03, 0x00}) {
		t.Fatal("an Orientation tag was written")
	}
	if _, err := jpeg.Decode(bytes.NewReader(j)); err != nil {
		t.Fatalf("injected JPEG does not decode: %v", err)
	}

	// Injecting again replaces the block, leaving one Exif APP1.
	again, err := InjectJPEG(j, BuildAPP1(Exif{Make: "Canon"}))
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := FromJPEG(again); got.Make != "Canon" || got.Model != "" || got.Time != "" {
		t.Fatalf("replace: %+v", got)
	}
	if bytes.Count(again, exifPrefix) != 1 {
		t.Fatalf("Exif blocks after replace: %d", bytes.Count(again, exifPrefix))
	}

	// A JFIF APP0 stays first.
	jfif := afterSOI(base, segment(0xE0, []byte("JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")))
	k, err := InjectJPEG(jfif, BuildAPP1(Exif{Make: "Nikon"}))
	if err != nil {
		t.Fatal(err)
	}
	if k[3] != 0xE0 || k[2+2+16+1] != 0xE1 {
		t.Fatalf("APP0 not first: % x", k[:24])
	}
	if got, _ := FromJPEG(k); got.Make != "Nikon" {
		t.Fatalf("after APP0: %+v", got)
	}

	// An empty APP1 strips; nothing to write builds nothing.
	stripped, err := InjectJPEG(j, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := FromJPEG(stripped); ok {
		t.Fatal("strip left EXIF")
	}
	if BuildAPP1(Exif{Orientation: 3, Software: "x"}) != nil {
		t.Fatal("BuildAPP1 with nothing to write")
	}

	// Bad input errors.
	if _, err := InjectJPEG([]byte("nope"), nil); err == nil {
		t.Fatal("not a JPEG accepted")
	}
	if _, err := InjectJPEG(base, []byte{0xFF, 0xE1, 0, 2}); err == nil {
		t.Fatal("bad APP1 accepted")
	}
	for n := 0; n < 200; n++ {
		if _, err := InjectJPEG(base[:n], BuildAPP1(in)); err == nil {
			t.Fatalf("cut at %d accepted", n)
		}
	}
}

// fuzzSeeds is the seed corpus: one of each container this package reads.
func fuzzSeeds(t testing.TB) [][]byte {
	img := scene(16, 12)
	j := jpegOf(t, img, 80)
	jx, _ := InjectJPEG(j, APP1(EncodeTIFF(Exif{Orientation: 8, Time: "2020-01-01T00:00:00", Make: "M", Model: "N", Software: "S"}, binary.BigEndian)))
	tiff := EncodeTIFF(Exif{Make: "Apple"}, binary.BigEndian)
	return [][]byte{
		j, jx,
		pngWith(pngOf(t, img), pngChunk("eXIf", tiff), pngChunk("tEXt", []byte("parameters\x00x"))),
		riff(webpChunk("VP8X", vp8x(0x0A, 16, 12)), webpChunk("EXIF", tiff), webpChunk("ANIM", make([]byte, 6))),
		heicFile(append([]byte{0, 0, 0, 0}, tiff...), false),
		heicFile(append([]byte{0, 0, 0, 0}, tiff...), true),
		tiff,
		{0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF, 'E', 'x', 'i', 'f', 0, 0, 'M', 'M', 0, 42, 0xFF, 0xFF, 0xFF, 0xF0},
		{},
	}
}

func FuzzSniffHead(f *testing.F) {
	for _, s := range fuzzSeeds(f) {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		SniffHead(b)
		FromJPEG(b)
		FromPNG(b)
		FromWebP(b)
		ParseTIFF(b)
		JPEGQuality(b)
		IsHEIF(b)
		HEIFExif(bytes.NewReader(b), int64(len(b)))
		InjectJPEG(b, BuildAPP1(Exif{Make: "x"}))
	})
}
