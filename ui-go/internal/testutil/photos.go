package testutil

import (
	"bytes"
	"compress/zlib"
	"encoding/base64"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"golang.org/x/image/draw"

	"github.com/geocine/aitoolkit/internal/imgmeta"
)

// Photo fixtures for the photo checks. Every fixture is generated in code
// from a seed, so tests need no image files.

// Scene is a stand-in for an ordinary photo: a sky gradient across the top,
// a ground gradient below it, hard-edged shapes in muted colours (some
// plain, some striped like fabric or blinds), and a fine per-pixel grain
// like a sensor's. It is sharp at every size, has detail finer than half
// its size (so it never reads as enlarged), keeps everyday colour (not
// black and white, not filtered), and the top band is free of shapes (so
// it never reads as a phone status bar). Different seeds give unrelated
// pictures.
func Scene(seed, w, h int) *image.RGBA {
	rng := rand.New(rand.NewPCG(uint64(seed), 0x9E3779B97F4A7C15))
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	muted := func(lo, hi, chroma float64) [3]float64 {
		y := lo + (hi-lo)*rng.Float64()
		return [3]float64{y + chroma*(2*rng.Float64()-1), y + chroma*(2*rng.Float64()-1), y + chroma*(2*rng.Float64()-1)}
	}
	horizon := 0.25 + 0.2*rng.Float64()
	sky0 := muted(120, 200, 25)
	sky0[2] += 20
	gnd0 := muted(60, 140, 25)
	skew := 15 + 15*rng.Float64() // left-right brightness drift of the sky
	if rng.IntN(2) == 0 {
		skew = -skew
	}
	for y := 0; y < h; y++ {
		fy := float64(y) / float64(h)
		for x := 0; x < w; x++ {
			fx := float64(x) / float64(w)
			var c [3]float64
			if fy < horizon {
				for k := range c {
					c[k] = sky0[k] + 40*fy + skew*(fx-0.5)
				}
			} else {
				for k := range c {
					c[k] = gnd0[k] + 50*(fy-horizon) - 20*fx
				}
			}
			set(img, x, y, c)
		}
	}
	// Shapes stay below the top 8% of the frame. Stripes are half bands
	// wide, so they repeat about every 3 pixels once the photo is scaled
	// to the 1024 sharpness copy (and every 4 pixels in smaller photos).
	top := int(0.08 * float64(h))
	half := max(2, int(math.Round(math.Sqrt(float64(w)*float64(h))/683)))
	for range 30 {
		col := muted(30, 220, 45)
		cx, cy := rng.Float64()*float64(w), float64(top)+rng.Float64()*float64(h-top)
		rx, ry := (0.03+0.10*rng.Float64())*float64(w), (0.03+0.10*rng.Float64())*float64(h)
		round := rng.IntN(2) == 0
		pattern := rng.IntN(3) // 0 plain, 1 vertical stripes, 2 horizontal stripes
		amp := 25 + 30*rng.Float64()
		x0, x1 := max(0, int(cx-rx)), min(w, int(cx+rx)+1)
		y0, y1 := max(top, int(cy-ry)), min(h, int(cy+ry)+1)
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				if round {
					dx, dy := (float64(x)-cx)/rx, (float64(y)-cy)/ry
					if dx*dx+dy*dy > 1 {
						continue
					}
				}
				c := col
				on := false
				switch pattern {
				case 1:
					on = x/half%2 == 0
				case 2:
					on = y/half%2 == 0
				}
				if on {
					c = [3]float64{c[0] + amp, c[1] + amp, c[2] + amp}
				} else if pattern != 0 {
					c = [3]float64{c[0] - amp, c[1] - amp, c[2] - amp}
				}
				set(img, x, y, c)
			}
		}
	}
	// Grain: triangular noise of up to ±grain levels, the same on all channels.
	const grain = 10
	s := uint32(seed)*2654435761 + 1
	for i := 0; i < len(img.Pix); i += 4 {
		s ^= s << 13
		s ^= s >> 17
		s ^= s << 5
		n := int(s&0xFF) + int(s>>8&0xFF) - 255 // -255..255, triangular
		d := n * grain / 255
		for k := 0; k < 3; k++ {
			img.Pix[i+k] = clampByte(float64(int(img.Pix[i+k]) + d))
		}
	}
	return img
}

func set(img *image.RGBA, x, y int, c [3]float64) {
	i := img.PixOffset(x, y)
	img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = clampByte(c[0]), clampByte(c[1]), clampByte(c[2]), 255
}

func clampByte(v float64) uint8 {
	switch {
	case v <= 0:
		return 0
	case v >= 255:
		return 255
	}
	return uint8(math.Round(v))
}

// toRGBA copies any image into a new *image.RGBA at the origin.
func toRGBA(img image.Image) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(out, out.Bounds(), img, b.Min, draw.Src)
	return out
}

// mapPix applies f to every pixel's RGB.
func mapPix(img image.Image, f func(r, g, b float64) (float64, float64, float64)) *image.RGBA {
	out := toRGBA(img)
	for i := 0; i < len(out.Pix); i += 4 {
		r, g, b := f(float64(out.Pix[i]), float64(out.Pix[i+1]), float64(out.Pix[i+2]))
		out.Pix[i], out.Pix[i+1], out.Pix[i+2] = clampByte(r), clampByte(g), clampByte(b)
	}
	return out
}

func luma(r, g, b float64) float64 { return 0.299*r + 0.587*g + 0.114*b }

// Blur is a Gaussian blur with the given sigma in pixels.
func Blur(img image.Image, sigma float64) *image.RGBA {
	out := toRGBA(img)
	rad := int(math.Ceil(3 * sigma))
	if rad < 1 {
		return out
	}
	k := make([]float64, 2*rad+1)
	var sum float64
	for i := range k {
		d := float64(i - rad)
		k[i] = math.Exp(-d * d / (2 * sigma * sigma))
		sum += k[i]
	}
	for i := range k {
		k[i] /= sum
	}
	w, h := out.Rect.Dx(), out.Rect.Dy()
	tmp := make([]float64, len(out.Pix))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for c := 0; c < 3; c++ {
				var s float64
				for i, kv := range k {
					xx := min(max(x+i-rad, 0), w-1)
					s += kv * float64(out.Pix[y*out.Stride+4*xx+c])
				}
				tmp[y*out.Stride+4*x+c] = s
			}
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for c := 0; c < 3; c++ {
				var s float64
				for i, kv := range k {
					yy := min(max(y+i-rad, 0), h-1)
					s += kv * tmp[yy*out.Stride+4*x+c]
				}
				out.Pix[y*out.Stride+4*x+c] = clampByte(s)
			}
		}
	}
	return out
}

// Resize scales to w x h (Catmull-Rom, which filters properly when
// shrinking).
func Resize(img image.Image, w, h int) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.CatmullRom.Scale(out, out.Bounds(), img, img.Bounds(), draw.Src, nil)
	return out
}

// FlipH mirrors left to right.
func FlipH(img image.Image) *image.RGBA {
	src := toRGBA(img)
	w, h := src.Rect.Dx(), src.Rect.Dy()
	out := image.NewRGBA(src.Rect)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			copy(out.Pix[y*out.Stride+4*x:y*out.Stride+4*x+4], src.Pix[y*src.Stride+4*(w-1-x):])
		}
	}
	return out
}

// Rotate turns img clockwise by quarter turns (0-3).
func Rotate(img image.Image, quarters int) *image.RGBA {
	out := toRGBA(img)
	for range (quarters%4 + 4) % 4 {
		w, h := out.Rect.Dx(), out.Rect.Dy()
		r := image.NewRGBA(image.Rect(0, 0, h, w))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				// (x, y) moves to (h-1-y, x)
				copy(r.Pix[x*r.Stride+4*(h-1-y):x*r.Stride+4*(h-1-y)+4], out.Pix[y*out.Stride+4*x:])
			}
		}
		out = r
	}
	return out
}

// Crop keeps the centre, cutting frac of the width and of the height in
// total (half from each side).
func Crop(img image.Image, frac float64) *image.RGBA {
	b := img.Bounds()
	dx, dy := int(math.Round(float64(b.Dx())*frac/2)), int(math.Round(float64(b.Dy())*frac/2))
	r := image.Rect(b.Min.X+dx, b.Min.Y+dy, b.Max.X-dx, b.Max.Y-dy)
	out := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(out, out.Bounds(), img, r.Min, draw.Src)
	return out
}

// Darken is a photo taken in deep shadow: about 3 stops under.
func Darken(img image.Image) *image.RGBA {
	return mapPix(img, func(r, g, b float64) (float64, float64, float64) { return r * 0.18, g * 0.18, b * 0.18 })
}

// Brighten is a photo blown out by about 2 stops: highlights clip to white.
func Brighten(img image.Image) *image.RGBA {
	return mapPix(img, func(r, g, b float64) (float64, float64, float64) { return r*2.5 + 30, g*2.5 + 30, b*2.5 + 30 })
}

// Flatten squeezes the contrast to a fifth around mid grey, like haze.
func Flatten(img image.Image) *image.RGBA {
	f := func(v float64) float64 { return 128 + (v-128)*0.2 }
	return mapPix(img, func(r, g, b float64) (float64, float64, float64) { return f(r), f(g), f(b) })
}

// Gray is black and white.
func Gray(img image.Image) *image.RGBA {
	return mapPix(img, func(r, g, b float64) (float64, float64, float64) {
		y := luma(r, g, b)
		return y, y, y
	})
}

// Sepia is a toned print: grey plus one fixed warm tint.
func Sepia(img image.Image) *image.RGBA {
	return mapPix(img, func(r, g, b float64) (float64, float64, float64) {
		y := luma(r, g, b) * 0.9
		return y + 20, y + 6, y - 14
	})
}

// Saturate pushes colour four times further from grey, like a strong
// filter.
func Saturate(img image.Image) *image.RGBA {
	return mapPix(img, func(r, g, b float64) (float64, float64, float64) {
		y := luma(r, g, b)
		return y + (r-y)*4, y + (g-y)*4, y + (b-y)*4
	})
}

// Noise adds Gaussian grain with the given sigma in levels.
func Noise(img image.Image, sigma float64, seed int) *image.RGBA {
	rng := rand.New(rand.NewPCG(uint64(seed), 7))
	return mapPix(img, func(r, g, b float64) (float64, float64, float64) {
		n := rng.NormFloat64() * sigma
		return r + n, g + n, b + n
	})
}

// Letterbox paints bars of colour c over the top and bottom frac of the
// height, like a video still.
func Letterbox(img image.Image, frac float64, c color.Color) *image.RGBA {
	out := toRGBA(img)
	h := out.Rect.Dy()
	n := int(math.Round(frac * float64(h)))
	draw.Draw(out, image.Rect(0, 0, out.Rect.Dx(), n), image.NewUniform(c), image.Point{}, draw.Src)
	draw.Draw(out, image.Rect(0, h-n, out.Rect.Dx(), h), image.NewUniform(c), image.Point{}, draw.Src)
	return out
}

// WhiteBackdrop is a high-key studio portrait: a person-sized figure in
// mid tones in front of a near-white seamless backdrop that falls off
// toward the edges. It is bright but not blown out and has no bars.
func WhiteBackdrop(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewPCG(11, 13))
	cx := 0.5 * float64(w)
	for y := 0; y < h; y++ {
		fy := float64(y) / float64(h)
		for x := 0; x < w; x++ {
			fx := float64(x) / float64(w)
			d := math.Hypot(fx-0.5, fy-0.45)
			v := 246 - 40*d*d - 10*fy + rng.NormFloat64()*1.5
			c := [3]float64{v, v - 1, v - 3}
			// head
			hx, hy := (float64(x)-cx)/(0.11*float64(w)), (fy-0.3)/0.13
			// shoulders and body down to the bottom edge
			body := fy > 0.42 && math.Abs(float64(x)-cx) < float64(w)*(0.16+0.25*(fy-0.42))
			if hx*hx+hy*hy < 1 || body {
				t := 110 + 40*math.Sin(fx*40)*math.Sin(fy*30) + rng.NormFloat64()*4
				c = [3]float64{t + 25, t, t - 15}
			}
			set(img, x, y, c)
		}
	}
	return img
}

// UIScreen is a phone screenshot: a light app screen with a status bar
// (clock on the left; signal, wifi and battery on the right), a title, and
// a list running edge to edge and down to the bottom: alternating row
// tints, a picture and two lines of text per row.
func UIScreen(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	fill := func(x0, y0, x1, y1 float64, c color.RGBA) {
		r := image.Rect(int(x0*float64(w)), int(y0*float64(h)), int(x1*float64(w)), int(y1*float64(h)))
		draw.Draw(img, r, image.NewUniform(c), image.Point{}, draw.Src)
	}
	ink := color.RGBA{20, 20, 24, 255}
	fill(0, 0, 1, 1, color.RGBA{238, 238, 240, 255})
	// status bar glyphs inside the top 4%
	for i, x := range []float64{0.07, 0.10, 0.135, 0.165} { // "9:41"
		wd := 0.022
		if i == 1 {
			wd = 0.008
		}
		fill(x, 0.012, x+wd, 0.026, ink)
	}
	fill(0.72, 0.014, 0.78, 0.026, ink)                       // signal
	fill(0.80, 0.013, 0.85, 0.026, ink)                       // wifi
	fill(0.88, 0.013, 0.95, 0.026, ink)                       // battery
	fill(0.05, 0.05, 0.6, 0.075, color.RGBA{30, 30, 36, 255}) // title
	for i := 0; i < 8; i++ {
		y := 0.1 + float64(i)*0.1125
		tint := color.RGBA{244, 244, 246, 255}
		if i%2 == 1 {
			tint = color.RGBA{196, 200, 210, 255}
		}
		fill(0, y, 1, y+0.1125, tint)
		fill(0, y+0.015, 0.2, y+0.095, color.RGBA{uint8(60 + 20*i), 120, uint8(200 - 15*i), 255})
		fill(0.25, y+0.025, 0.85, y+0.04, color.RGBA{60, 60, 70, 255})
		fill(0.25, y+0.06, 0.7, y+0.072, color.RGBA{120, 120, 130, 255})
	}
	return img
}

// JPEG encodes at quality q.
func JPEG(img image.Image, q int) []byte {
	var b bytes.Buffer
	if err := jpeg.Encode(&b, img, &jpeg.Options{Quality: q}); err != nil {
		panic(err)
	}
	return b.Bytes()
}

// PNG encodes with the standard library.
func PNG(img image.Image) []byte {
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		panic(err)
	}
	return b.Bytes()
}

// Gray16PNG is img as a 16-bit greyscale PNG (IHDR colour type 0, depth
// 16): the kind the trainer's loader clips to white.
func Gray16PNG(img image.Image) []byte {
	b := img.Bounds()
	g := image.NewGray16(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(g, g.Bounds(), img, b.Min, draw.Src)
	return PNG(g)
}

// GrayAlpha16PNG is img as a 16-bit grey+alpha PNG (colour type 4, depth
// 16), fully opaque. The standard encoder never writes type 4.
func GrayAlpha16PNG(img image.Image) []byte {
	src := toRGBA(img)
	w, h := src.Rect.Dx(), src.Rect.Dy()
	rows := make([][]byte, h)
	for y := range rows {
		r := make([]byte, 0, 4*w)
		for x := 0; x < w; x++ {
			i := y*src.Stride + 4*x
			v := uint16(clampByte(luma(float64(src.Pix[i]), float64(src.Pix[i+1]), float64(src.Pix[i+2])))) * 257
			r = append(r, byte(v>>8), byte(v), 0xFF, 0xFF)
		}
		rows[y] = r
	}
	return RawPNG(w, h, 4, 16, rows)
}

// RGBA16PNG is img as a 16-bit RGBA PNG (colour type 6, depth 16), fully
// opaque.
func RGBA16PNG(img image.Image) []byte {
	src := toRGBA(img)
	w, h := src.Rect.Dx(), src.Rect.Dy()
	rows := make([][]byte, h)
	for y := range rows {
		r := make([]byte, 0, 8*w)
		for x := 0; x < w; x++ {
			i := y*src.Stride + 4*x
			for c := 0; c < 3; c++ {
				r = append(r, src.Pix[i+c], src.Pix[i+c])
			}
			r = append(r, 0xFF, 0xFF)
		}
		rows[y] = r
	}
	return RawPNG(w, h, 6, 16, rows)
}

// TransparentPNG is img as an RGBA PNG whose right third is fully
// transparent (and keeps its colour underneath).
func TransparentPNG(img image.Image) []byte {
	b := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(out, out.Bounds(), img, b.Min, draw.Src)
	for y := 0; y < b.Dy(); y++ {
		for x := b.Dx() * 2 / 3; x < b.Dx(); x++ {
			out.Pix[out.PixOffset(x, y)+3] = 0
		}
	}
	return PNG(out)
}

// RawPNG writes a PNG from filtered-nothing scanlines (filter type 0 is
// added per row) with any colour type and bit depth, plus extra chunks
// (type, data) placed right after IHDR.
func RawPNG(w, h, colorType, depth int, rows [][]byte, extra ...[2]string) []byte {
	var b bytes.Buffer
	b.WriteString("\x89PNG\r\n\x1a\n")
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:], uint32(w))
	binary.BigEndian.PutUint32(ihdr[4:], uint32(h))
	ihdr[8], ihdr[9] = byte(depth), byte(colorType)
	pngChunk(&b, "IHDR", ihdr)
	for _, e := range extra {
		pngChunk(&b, e[0], []byte(e[1]))
	}
	var z bytes.Buffer
	zw := zlib.NewWriter(&z)
	for _, r := range rows {
		zw.Write([]byte{0})
		zw.Write(r)
	}
	zw.Close()
	pngChunk(&b, "IDAT", z.Bytes())
	pngChunk(&b, "IEND", nil)
	return b.Bytes()
}

func pngChunk(b *bytes.Buffer, typ string, data []byte) {
	var n [4]byte
	binary.BigEndian.PutUint32(n[:], uint32(len(data)))
	b.Write(n[:])
	crc := crc32.NewIEEE()
	crc.Write([]byte(typ))
	crc.Write(data)
	b.WriteString(typ)
	b.Write(data)
	binary.BigEndian.PutUint32(n[:], crc.Sum32())
	b.Write(n[:])
}

// PNGWithText is a small scene PNG carrying a tEXt chunk k=v right after
// IHDR (k "parameters", "prompt" or "workflow" marks a generated image).
func PNGWithText(k, v string) []byte {
	p := PNG(Scene(5, 256, 192))
	const ihdrEnd = 8 + 4 + 4 + 13 + 4
	var chunk bytes.Buffer
	pngChunk(&chunk, "tEXt", []byte(k+"\x00"+v))
	out := append([]byte{}, p[:ihdrEnd]...)
	out = append(out, chunk.Bytes()...)
	return append(out, p[ihdrEnd:]...)
}

// WithExif returns jpg with an Exif APP1 (little-endian, "II") holding
// orientation o, DateTimeOriginal t ("2006:01:02 15:04:05" or
// "2006-01-02T15:04:05", "" for none), Make and Model.
func WithExif(jpg []byte, o int, t, make_, model string) []byte {
	return withExif(jpg, o, t, make_, model, binary.LittleEndian)
}

// WithExifMM is WithExif with a big-endian ("MM") TIFF block.
func WithExifMM(jpg []byte, o int, t, make_, model string) []byte {
	return withExif(jpg, o, t, make_, model, binary.BigEndian)
}

func withExif(jpg []byte, o int, t, make_, model string, bo binary.ByteOrder) []byte {
	app1 := imgmeta.APP1(imgmeta.EncodeTIFF(imgmeta.Exif{Orientation: o, Time: t, Make: make_, Model: model}, bo))
	out, err := imgmeta.InjectJPEG(jpg, app1)
	if err != nil {
		panic(err)
	}
	return out
}

// AnimatedWebPHeader is the head of an animated WebP: RIFF, a VP8X chunk
// with the animation bit and a 640x480 canvas, and an ANIM chunk. It is
// not a decodable image.
func AnimatedWebPHeader() []byte {
	var body bytes.Buffer
	body.WriteString("WEBP")
	vp8x := make([]byte, 10)
	vp8x[0] = 0x02 // animation
	put24 := func(b []byte, v int) { b[0], b[1], b[2] = byte(v), byte(v>>8), byte(v>>16) }
	put24(vp8x[4:], 640-1)
	put24(vp8x[7:], 480-1)
	riffChunk(&body, "VP8X", vp8x)
	riffChunk(&body, "ANIM", make([]byte, 6))
	var b bytes.Buffer
	b.WriteString("RIFF")
	var n [4]byte
	binary.LittleEndian.PutUint32(n[:], uint32(body.Len()))
	b.Write(n[:])
	b.Write(body.Bytes())
	return b.Bytes()
}

// Lossy WebP fixtures, made once with Pillow 12.3 (libwebp, quality 90,
// method 6, exact=True for the ones with alpha). Go has no WebP encoder,
// and VP8 stores limited-range YCbCr, which these pin down.
const (
	// webpHalves is 32x16: the left half black, the right half white.
	webpHalves = "UklGRjIAAABXRUJQVlA4ICYAAABQAgCdASogABAALiUYjEYREREBABIlpAAD2UAA/v7uxb/7P9AAAA=="
	// webpAlpha is 32x16 of RGB (30, 100, 50) with the left quarter fully
	// transparent (VP8X, ALPH, VP8).
	webpAlpha = "UklGRm4AAABXRUJQVlA4WAoAAAAQAAAAHwAADwAAQUxQSBEAAAABBjFp0j+1GfxGRARAAACAvwBWUDggNgAAANACAJ0BKiAAEAA+MRSIQqIhIRgEACADBKAHYAARYwAA/vhff/+hRsomKf//zYH7efbzwsAAAA=="
	// webpAnimated is a 32x24 two-frame animation: frame 1 is grey 64
	// with a transparent 16x12 centre (ALPH and VP8), frame 2 is white.
	webpAnimated = "UklGRtoAAABXRUJQVlA4WAoAAAASAAAAHwAAFwAAQU5JTQYAAAAAAAAAAABBTk1GZAAAAAAAAAAAAB8AABcAAGQAAAJBTFBIGAAAAAEPMP8REUIsmOQvvT+CiP4HAACAugEAAFZQOCAsAAAAcAMAnQEqIAAYAC4lGIxGERERAQASJaQAW3FAM4wn1N8l33cAAP7/3ibQAABBTk1GQgAAAAAAAAAAAB8AABcAAGQAAABWUDggKgAAABQDAJ0BKiAAGAAuJRiMRgAAAkS0gAtuKAZn10raCqdoAAD+///E4WgAAA=="
	// webpScene is Scene(1, 128, 96).
	webpScene = `
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
inNQpWwRzA3BYSOWns+ovgAA
`
)

// WebPHalves is a lossy 32x16 WebP, black on the left and white on the
// right.
func WebPHalves() []byte { return mustBase64(webpHalves) }

// WebPAlpha is a lossy 32x16 WebP of RGB (30, 100, 50) whose left quarter
// is transparent.
func WebPAlpha() []byte { return mustBase64(webpAlpha) }

// AnimatedWebP is a real two-frame animated WebP (32x24): frame 1 is grey
// 64 with a transparent 16x12 centre, frame 2 is white.
func AnimatedWebP() []byte { return mustBase64(webpAnimated) }

// SceneWebP is Scene(1, 128, 96) as a lossy WebP.
func SceneWebP() []byte { return mustBase64(webpScene) }

func mustBase64(s string) []byte {
	b, err := base64.StdEncoding.DecodeString(s) // skips the line breaks
	if err != nil {
		panic(err)
	}
	return b
}

func riffChunk(b *bytes.Buffer, fourcc string, data []byte) {
	var n [4]byte
	binary.LittleEndian.PutUint32(n[:], uint32(len(data)))
	b.WriteString(fourcc)
	b.Write(n[:])
	b.Write(data)
	if len(data)%2 == 1 {
		b.WriteByte(0)
	}
}

// Upscaled is img shrunk by factor, a little soft as small copies from
// chat apps are, and enlarged back to its size (bilinear): a small photo
// saved at a larger size. Its detail reads above 40 dB. When the shrunk
// copy is smaller than the 1024 sharpness copy (img under about 2900 x
// 1900 for factor 2) it also reads as soft there, as such a photo would.
func Upscaled(img image.Image, factor int) *image.RGBA {
	b := img.Bounds()
	small := Blur(Resize(img, max(1, b.Dx()/factor), max(1, b.Dy()/factor)), 0.9)
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.BiLinear.Scale(out, out.Bounds(), small, small.Bounds(), draw.Src, nil)
	return out
}

// WriteTree writes files (slash-separated names relative to dir), creating
// folders as needed.
func WriteTree(dir string, files map[string][]byte) error {
	for name, data := range files {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(p, data, 0o644); err != nil {
			return err
		}
	}
	return nil
}
