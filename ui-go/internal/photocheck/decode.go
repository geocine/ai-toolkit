package photocheck

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"os"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	"golang.org/x/image/webp"

	"github.com/geocine/aitoolkit/internal/imgmeta"
	"github.com/geocine/aitoolkit/internal/thumbs"
)

// Measure reads one photo file and measures it (section 3). rel is the
// dataset-relative name stored in Metrics.File. The error is only for a
// file that cannot be read at all; a file that does not decode, or is too
// large to decode, comes back with Decoded false and Error set, plus what
// its header says (format, size, EXIF).
func Measure(path, rel string) (m Metrics, err error) {
	var (
		sha  string
		size int64
		sn   imgmeta.Sniff
	)
	failed := func(msg string) (Metrics, error) {
		m := headerMetrics(sn, rel, size)
		m.SHA256 = sha
		m.Error = msg
		return m, nil
	}
	defer func() {
		// The decoders and the measurements are bounds-checked; this keeps
		// one hostile file from taking the server down if one is not. Once
		// the file is hashed it still gets a row (and is not decoded again).
		if r := recover(); r != nil {
			if sha == "" {
				m, err = Metrics{}, fmt.Errorf("measure %s: %v", rel, r)
				return
			}
			m, err = failed("cannot decode this file")
		}
	}()
	f, err := os.Open(path)
	if err != nil {
		return Metrics{}, err
	}
	defer f.Close()
	head := make([]byte, imgmeta.HeadSize)
	n, err := io.ReadFull(f, head)
	if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
		return Metrics{}, err
	}
	head = head[:n]
	sum := sha256.New()
	sum.Write(head)
	rest, err := io.Copy(sum, f)
	if err != nil {
		return Metrics{}, err
	}
	sn = imgmeta.SniffHead(head)
	size = int64(n) + rest
	sha = hex.EncodeToString(sum.Sum(nil))

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return failed("cannot read this file")
	}
	cfg, _, err := image.DecodeConfig(bufio.NewReader(f))
	if err != nil {
		return failed(decodeError(sn.Format, err))
	}
	if sn.Width == 0 || sn.Height == 0 {
		sn.Width, sn.Height = cfg.Width, cfg.Height
	}
	if int64(cfg.Width)*int64(cfg.Height) > thumbs.MaxPixels {
		return failed("too large to analyse")
	}
	var img image.Image
	if sn.Format == "webp" && sn.Animated {
		// x/image/webp reads stills only; the trainer trains on frame 1.
		img, err = firstWebPFrame(f, size)
		if err != nil {
			return failed(decodeError(sn.Format, err))
		}
	} else {
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return failed("cannot read this file")
		}
		img, _, err = image.Decode(bufio.NewReader(f))
		if err != nil {
			return failed(decodeError(sn.Format, err))
		}
	}
	m = MeasureImage(img, sn, rel, size)
	m.SHA256 = sha
	if sn.Format == "webp" && sn.Animated {
		// The size is the canvas's, as the trainer sees it.
		h := headerMetrics(sn, rel, size)
		m.Width, m.Height = h.Width, h.Height
	}
	return m, nil
}

// firstWebPFrame decodes the first frame of an animated WebP: its ANMF
// chunk's ALPH and VP8 (or VP8L) sub-chunks, rewrapped as a still WebP.
// The frame is measured on its own, at its own size and without the
// canvas around it; the first frame almost always covers the canvas.
func firstWebPFrame(f *os.File, size int64) (image.Image, error) {
	var hdr [8]byte
	for pos := int64(12); pos+8 <= size; {
		if _, err := f.ReadAt(hdr[:], pos); err != nil {
			return nil, err
		}
		n := int64(binary.LittleEndian.Uint32(hdr[4:]))
		if n > size-pos-8 {
			return nil, errWebPFrame
		}
		if string(hdr[:4]) != "ANMF" {
			pos += 8 + n + n&1
			continue
		}
		if n < 16+8 {
			return nil, errWebPFrame
		}
		frame := make([]byte, n)
		if _, err := f.ReadAt(frame, pos+8); err != nil {
			return nil, err
		}
		w := 1 + (int(frame[6]) | int(frame[7])<<8 | int(frame[8])<<16)
		h := 1 + (int(frame[9]) | int(frame[10])<<8 | int(frame[11])<<16)
		var alph, bits []byte // whole sub-chunks, header included
		for sub := frame[16:]; len(sub) >= 8 && bits == nil; {
			m := int64(binary.LittleEndian.Uint32(sub[4:]))
			if m > int64(len(sub))-8 {
				return nil, errWebPFrame
			}
			end := 8 + m
			chunk := sub[:end]
			switch string(sub[:4]) {
			case "ALPH":
				alph = chunk
			case "VP8 ", "VP8L":
				bits = chunk
			}
			end += m & 1
			sub = sub[min(end, int64(len(sub))):]
		}
		if bits == nil {
			return nil, errWebPFrame
		}
		var body bytes.Buffer
		body.WriteString("WEBP")
		if alph != nil && string(bits[:4]) == "VP8 " {
			vp8x := make([]byte, 18)
			copy(vp8x, "VP8X")
			vp8x[4] = 10
			vp8x[8] = 0x10 // alpha
			vp8x[12], vp8x[13], vp8x[14] = byte(w-1), byte((w-1)>>8), byte((w-1)>>16)
			vp8x[15], vp8x[16], vp8x[17] = byte(h-1), byte((h-1)>>8), byte((h-1)>>16)
			body.Write(vp8x)
			writePadded(&body, alph)
		}
		writePadded(&body, bits)
		still := make([]byte, 8, 8+body.Len())
		copy(still, "RIFF")
		binary.LittleEndian.PutUint32(still[4:], uint32(body.Len()))
		still = append(still, body.Bytes()...)
		cfg, err := webp.DecodeConfig(bytes.NewReader(still))
		if err != nil {
			return nil, err
		}
		if int64(cfg.Width)*int64(cfg.Height) > thumbs.MaxPixels {
			return nil, errWebPFrame
		}
		return webp.Decode(bytes.NewReader(still))
	}
	return nil, errWebPFrame
}

var errWebPFrame = errors.New("no readable first frame")

// writePadded writes one RIFF chunk and its pad byte.
func writePadded(b *bytes.Buffer, chunk []byte) {
	b.Write(chunk)
	if len(chunk)%2 == 1 {
		b.WriteByte(0)
	}
}

func decodeError(format string, err error) string {
	if errors.Is(err, image.ErrFormat) {
		return "cannot decode " + format + " here"
	}
	return fmt.Sprintf("cannot decode: %v", err)
}

// headerMetrics is what the sniffed header alone says about a photo.
func headerMetrics(sn imgmeta.Sniff, rel string, bytes int64) Metrics {
	o := sn.Exif.Orientation
	if o < 1 || o > 8 {
		o = 1
	}
	w, h := sn.Width, sn.Height
	if o >= 5 {
		w, h = h, w
	}
	m := Metrics{
		Version:      MetricsVersion,
		File:         rel,
		Bytes:        bytes,
		Format:       sn.Format,
		Width:        w,
		Height:       h,
		Orientation:  o,
		Source:       "server",
		Animated:     sn.Animated,
		JPEGQ:        sn.JPEGQuality,
		PNGColorType: sn.PNGColorType,
		PNGBitDepth:  sn.PNGBitDepth,
		Generated:    sn.Generated,
	}
	if sn.JPEGComponents == 4 {
		m.Mode = "cmyk"
	}
	if sn.HasExif {
		e := sn.Exif
		x := &ExifInfo{Time: e.Time, Make: e.Make, Model: e.Model, Software: e.Software}
		if x.Time != "" && e.SubSec != "" {
			x.Time += "." + (e.SubSec + "000")[:3]
		}
		if *x != (ExifInfo{}) {
			m.Exif = x
		}
	}
	return m
}

// planes holds a decoded photo as stored (before EXIF orientation): the
// full-resolution luma plane, and either the JPEG chroma planes or RGB
// planes for the colour copy, plus alpha when the image has any.
type planes struct {
	w, h    int
	y       []uint8
	ys      int
	cb, cr  []uint8 // YCbCr images: subsampled chroma
	cs      int
	cw, ch  int
	r, g, b []uint8 // other images: stride w
	a       []uint8 // nil when opaque; stride w
	mode    string
}

// lumaOf is the 8-bit luma of 8-bit RGB (section 3).
func lumaOf(r, g, b uint32) uint8 {
	return uint8((19595*r + 38470*g + 7471*b + 32768) >> 16)
}

// toPlanes reads a decoded image into planes. format is the sniffed file
// format: lossy WebP (VP8) stores limited-range BT.601 YCbCr, which Go
// hands back as is, so its planes are expanded to full range first.
func toPlanes(img image.Image, format string) planes {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	p := planes{w: w, h: h, mode: "rgb"}
	studio := format == "webp"
	switch m := img.(type) {
	case *image.YCbCr:
		if p.fromYCbCr(m, studio) {
			return p
		}
	case *image.NYCbCrA:
		if p.fromYCbCr(&m.YCbCr, studio) {
			p.a = make([]uint8, w*h)
			alpha := false
			for y := 0; y < h; y++ {
				row := m.A[y*m.AStride : y*m.AStride+w]
				copy(p.a[y*w:], row)
				for _, v := range row {
					if v != 255 {
						alpha = true
					}
				}
			}
			if !alpha {
				p.a = nil
			}
			return p
		}
	case *image.Gray:
		p.mode = "gray"
		if b.Min == (image.Point{}) {
			p.y, p.ys = m.Pix, m.Stride
			p.r, p.g, p.b = grayRGB(m.Pix, m.Stride, w, h)
			return p
		}
	}
	n := w * h
	p.y, p.ys = make([]uint8, n), w
	p.r, p.g, p.b = make([]uint8, n), make([]uint8, n), make([]uint8, n)
	var alpha bool
	switch m := img.(type) {
	case *image.RGBA:
		for y := 0; y < h; y++ {
			row := m.Pix[(y)*m.Stride : (y)*m.Stride+4*w]
			for x := 0; x < w; x++ {
				i := y*w + x
				r, g, bl, a := row[4*x], row[4*x+1], row[4*x+2], row[4*x+3]
				if a != 255 {
					alpha = true
					if a != 0 {
						// premultiplied: back to straight colour
						r, g, bl = uint8(uint32(r)*255/uint32(a)), uint8(uint32(g)*255/uint32(a)), uint8(uint32(bl)*255/uint32(a))
					}
				}
				p.r[i], p.g[i], p.b[i] = r, g, bl
			}
		}
		if alpha {
			p.a = make([]uint8, n)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					p.a[y*w+x] = m.Pix[y*m.Stride+4*x+3]
				}
			}
		}
	case *image.NRGBA:
		p.a = make([]uint8, n)
		for y := 0; y < h; y++ {
			row := m.Pix[y*m.Stride : y*m.Stride+4*w]
			for x := 0; x < w; x++ {
				i := y*w + x
				p.r[i], p.g[i], p.b[i], p.a[i] = row[4*x], row[4*x+1], row[4*x+2], row[4*x+3]
				if row[4*x+3] != 255 {
					alpha = true
				}
			}
		}
	default:
		switch img.(type) {
		case *image.Gray16:
			p.mode = "gray16"
		case *image.Gray:
			p.mode = "gray"
		case *image.CMYK:
			p.mode = "cmyk"
		case *image.Paletted:
			p.mode = "paletted"
		}
		p.a = make([]uint8, n)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				i := y*w + x
				p.r[i], p.g[i], p.b[i], p.a[i] = c.R, c.G, c.B, c.A
				if c.A != 255 {
					alpha = true
				}
			}
		}
	}
	if !alpha {
		p.a = nil
	}
	for i := range n {
		p.y[i] = lumaOf(uint32(p.r[i]), uint32(p.g[i]), uint32(p.b[i]))
	}
	return p
}

// fromYCbCr takes a YCbCr image's planes as they are, or expanded from
// limited range (Y 16-235, C 16-240) when studio is set. It reports false
// for a layout it does not read directly.
func (p *planes) fromYCbCr(m *image.YCbCr, studio bool) bool {
	cw, ch, ok := chromaSize(m.SubsampleRatio, p.w, p.h)
	if !ok || m.Rect.Min != (image.Point{}) {
		return false
	}
	if !studio {
		p.y, p.ys = m.Y, m.YStride
		p.cb, p.cr, p.cs, p.cw, p.ch = m.Cb, m.Cr, m.CStride, cw, ch
		return true
	}
	p.y, p.ys = expandPlane(m.Y, m.YStride, p.w, p.h, &lumaFull), p.w
	p.cb = expandPlane(m.Cb, m.CStride, cw, ch, &chromaFull)
	p.cr = expandPlane(m.Cr, m.CStride, cw, ch, &chromaFull)
	p.cs, p.cw, p.ch = cw, cw, ch
	return true
}

// Limited-range to full-range tables: y' = (y-16)*255/219 and
// c' = (c-128)*255/224 + 128, rounded and clamped.
var lumaFull, chromaFull = func() (l, c [256]uint8) {
	for i := range 256 {
		l[i] = clampByte(float64(i-16) * 255 / 219)
		c[i] = clampByte(float64(i-128)*255/224 + 128)
	}
	return l, c
}()

func clampByte(v float64) uint8 {
	return uint8(math.Round(math.Min(255, math.Max(0, v))))
}

// expandPlane maps a w x h plane through t into a new plane of stride w.
func expandPlane(src []uint8, stride, w, h int, t *[256]uint8) []uint8 {
	out := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		row := src[y*stride : y*stride+w]
		o := out[y*w : (y+1)*w]
		for x, v := range row {
			o[x] = t[v]
		}
	}
	return out
}

func grayRGB(pix []uint8, stride, w, h int) (r, g, b []uint8) {
	r = make([]uint8, w*h)
	for y := 0; y < h; y++ {
		copy(r[y*w:(y+1)*w], pix[y*stride:y*stride+w])
	}
	return r, r, r
}

// chromaSize is the size of a YCbCr image's chroma planes when the image
// starts at the origin.
func chromaSize(r image.YCbCrSubsampleRatio, w, h int) (int, int, bool) {
	switch r {
	case image.YCbCrSubsampleRatio444:
		return w, h, true
	case image.YCbCrSubsampleRatio422:
		return (w + 1) / 2, h, true
	case image.YCbCrSubsampleRatio420:
		return (w + 1) / 2, (h + 1) / 2, true
	case image.YCbCrSubsampleRatio440:
		return w, (h + 1) / 2, true
	case image.YCbCrSubsampleRatio411:
		return (w + 3) / 4, h, true
	case image.YCbCrSubsampleRatio410:
		return (w + 3) / 4, (h + 1) / 2, true
	}
	return 0, 0, false
}
