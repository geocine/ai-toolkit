package photocheck

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/image/draw"

	"github.com/geocine/aitoolkit/internal/imgmeta"
	"github.com/geocine/aitoolkit/internal/thumbs"
)

// Server conversion (section 11.6). Go converts bmp, gif, tiff, 16-bit grey,
// transparent, CMYK and huge photos itself; HEIC, HEIF and AVIF go through
// the ffmpeg the thumbnails use, when it is 7.1 or newer (tile-grid HEIF).
// Every output is upright, with the camera's date, make and model kept and
// no orientation tag. A JPEG source loses a little at q95: Go has no
// lossless JPEG rotation.

const (
	jpegQuality = 95
	// hugeTarget is what convert scales a huge photo down to.
	hugeTarget = 40_000_000
	// convertMaxPixels caps what convert decodes. It is above
	// thumbs.MaxPixels, because a huge photo is what convert is for.
	convertMaxPixels = 400_000_000
)

// baseFormats are the formats Go converts on its own.
var baseFormats = []string{"bmp", "gif", "tiff"}

// ffmpegFormats are the formats convert hands to ffmpeg.
var ffmpegFormats = []string{"heic", "heif", "avif"}

// ffmpegVersion runs `ffmpeg -version` and returns its first line ("" when
// ffmpeg is missing). Tests replace it.
var ffmpegVersion = func() string {
	exe, env := thumbs.FFmpeg()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-version")
	cmd.Env = env
	thumbs.HideWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	line, _, _ := strings.Cut(string(out), "\n")
	return strings.TrimSpace(line)
}

var versionRe = regexp.MustCompile(`^ffmpeg version n?(\d+)\.(\d+)`)

// ffmpegReadsHEIF reports whether an `ffmpeg -version` first line names
// 7.1 or newer. A git build ("N-118000-g…") is taken as new.
func ffmpegReadsHEIF(line string) bool {
	if strings.HasPrefix(line, "ffmpeg version N-") {
		return true
	}
	m := versionRe.FindStringSubmatch(line)
	if m == nil {
		return false
	}
	major, _ := strconv.Atoi(m[1])
	minor, _ := strconv.Atoi(m[2])
	return major > 7 || (major == 7 && minor >= 1)
}

var convertFormats = sync.OnceValue(detectFormats)

func detectFormats() []string {
	out := slices.Clone(baseFormats)
	if ffmpegReadsHEIF(ffmpegVersion()) {
		out = append(out, ffmpegFormats...)
	}
	return out
}

// ConvertFormats are the formats the server converts for the trainer
// (review.convert): bmp, gif and tiff always; heic, heif and avif when the
// server's ffmpeg is 7.1 or newer.
func ConvertFormats() []string { return slices.Clone(convertFormats()) }

// convertTarget is what convert turns a photo into: the new extension and
// the reason, or ok false when convert leaves it alone. flags are the
// photo's active flags.
func convertTarget(rel string, flags []Flag) (ext, reason string, ok bool) {
	has := func(f Flag) bool { return slices.Contains(flags, f) }
	src := strings.ToLower(filepath.Ext(rel))
	format := strings.TrimPrefix(src, ".")
	if format == "tif" {
		format = "tiff"
	}
	if ConvertibleExts[src] {
		if !slices.Contains(ConvertFormats(), format) {
			return "", "", false
		}
		switch {
		case slices.Contains(ffmpegFormats, format):
			return ".jpg", strings.ToUpper(format) + " to JPEG", true
		case has(FlagTransparent):
			return ".jpg", strings.ToUpper(format) + " to JPEG on white", true
		}
		return ".png", strings.ToUpper(format) + " to PNG", true
	}
	switch {
	case has(FlagHuge):
		return ".jpg", "scaled down to 40 MP", true
	case has(FlagGray16):
		return ".png", "16-bit grey to 8-bit", true
	case has(FlagTransparent):
		return ".jpg", "transparent areas put on white", true
	case has(FlagCMYK):
		return ".jpg", "CMYK to RGB", true
	}
	return "", "", false
}

// convertFile writes src converted to dst (its extension picks JPEG or
// PNG). Huge photos are scaled down to 40 MP. A big decode waits for one
// of the pool's big-decode slots.
func convertFile(src, dst string) error {
	switch strings.ToLower(filepath.Ext(src)) {
	case ".heic", ".heif", ".avif":
		return convertFFmpeg(src, dst)
	}
	defer analysis.holdBig(src)()
	img, ex, err := decodeUpright(src)
	if err != nil {
		return err
	}
	b := img.Bounds()
	if area := float64(b.Dx()) * float64(b.Dy()); area > hugeTarget {
		f := math.Sqrt(hugeTarget / area)
		w, h := max(1, int(float64(b.Dx())*f)), max(1, int(float64(b.Dy())*f))
		small := image.NewRGBA(image.Rect(0, 0, w, h))
		draw.BiLinear.Scale(small, small.Bounds(), img, b, draw.Src, nil)
		img = small
	}
	return writeImage(dst, img, ex)
}

// convertFFmpeg turns a HEIC, HEIF or AVIF into a JPEG through ffmpeg and
// copies the camera's date, make and model from the source's Exif item.
func convertFFmpeg(src, dst string) error {
	tmp := filepath.Join(filepath.Dir(dst), "."+filepath.Base(dst)+".convert.jpg")
	defer os.Remove(tmp)
	exe, env := thumbs.FFmpeg()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-v", "error", "-i", src, "-frames:v", "1", "-q:v", "2", "-y", tmp)
	cmd.Env = env
	thumbs.HideWindow(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg could not convert %s: %v %s", filepath.Base(src), err, strings.TrimSpace(stderr.String()))
	}
	data, err := os.ReadFile(tmp)
	if err != nil {
		return err
	}
	if f, err := os.Open(src); err == nil {
		if st, err := f.Stat(); err == nil {
			if raw, err := imgmeta.HEIFExif(f, st.Size()); err == nil {
				if ex, err := imgmeta.ParseTIFF(raw); err == nil {
					if app1 := imgmeta.BuildAPP1(ex); app1 != nil {
						if out, err := imgmeta.InjectJPEG(data, app1); err == nil {
							data = out
						}
					}
				}
			}
		}
		f.Close()
	}
	return writeFileAtomic(dst, data)
}

// decodeUpright decodes a photo and turns it upright by its EXIF
// orientation. It returns the EXIF the output keeps.
func decodeUpright(src string) (image.Image, imgmeta.Exif, error) {
	data, err := os.ReadFile(src)
	if err != nil {
		return nil, imgmeta.Exif{}, err
	}
	sn := imgmeta.SniffHead(data)
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, sn.Exif, fmt.Errorf("cannot decode %s: %w", filepath.Base(src), err)
	}
	if int64(cfg.Width)*int64(cfg.Height) > convertMaxPixels {
		return nil, sn.Exif, fmt.Errorf("%s is %dx%d, too large to convert here", filepath.Base(src), cfg.Width, cfg.Height)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, sn.Exif, fmt.Errorf("cannot decode %s: %w", filepath.Base(src), err)
	}
	return upright(img, sn.Exif.Orientation), sn.Exif, nil
}

// upright copies img into 8-bit pixels (grey, RGBA, or NRGBA when it has
// transparent pixels) turned upright for an EXIF orientation. With no
// turn, an 8-bit grey or opaque RGBA image comes back as it is and any
// other gets one copy, so a big photo is held in memory at most twice.
func upright(img image.Image, o int) image.Image {
	b := img.Bounds()
	turn := o >= 2 && o <= 8
	if !turn && b.Min == (image.Point{}) {
		switch img.(type) {
		case *image.Gray:
			return img
		case *image.RGBA:
			if opaque(img) {
				return img
			}
		}
	}
	var flat image.Image
	var pix []uint8
	var bpp int
	var alloc func(r image.Rectangle) (image.Image, []uint8, int)
	switch img.(type) {
	case *image.Gray, *image.Gray16:
		g := image.NewGray(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(g, g.Bounds(), img, b.Min, draw.Src)
		flat, pix, bpp = g, g.Pix, 1
		alloc = func(r image.Rectangle) (image.Image, []uint8, int) { m := image.NewGray(r); return m, m.Pix, m.Stride }
	default:
		if thumbs.HasAlpha(img) && !opaque(img) {
			n := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
			draw.Draw(n, n.Bounds(), img, b.Min, draw.Src)
			flat, pix = n, n.Pix
			alloc = func(r image.Rectangle) (image.Image, []uint8, int) { m := image.NewNRGBA(r); return m, m.Pix, m.Stride }
		} else {
			n := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
			draw.Draw(n, n.Bounds(), img, b.Min, draw.Src)
			flat, pix = n, n.Pix
			alloc = func(r image.Rectangle) (image.Image, []uint8, int) { m := image.NewRGBA(r); return m, m.Pix, m.Stride }
		}
		bpp = 4
	}
	if !turn {
		return flat
	}
	w, h := b.Dx(), b.Dy()
	r := image.Rect(0, 0, w, h)
	if o >= 5 {
		r = image.Rect(0, 0, h, w)
	}
	out, dst, stride := alloc(r)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var dx, dy int
			switch o {
			case 2:
				dx, dy = w-1-x, y
			case 3:
				dx, dy = w-1-x, h-1-y
			case 4:
				dx, dy = x, h-1-y
			case 5:
				dx, dy = y, x
			case 6:
				dx, dy = h-1-y, x
			case 7:
				dx, dy = h-1-y, w-1-x
			case 8:
				dx, dy = y, w-1-x
			}
			s := (y*w + x) * bpp
			copy(dst[dy*stride+dx*bpp:dy*stride+dx*bpp+bpp], pix[s:s+bpp])
		}
	}
	return out
}

// opaque reports an image whose every pixel is fully opaque.
func opaque(img image.Image) bool {
	if o, ok := img.(interface{ Opaque() bool }); ok {
		return o.Opaque()
	}
	return false
}

// writeImage encodes img as JPEG q95 (on white when it has transparency,
// with the camera's date, make and model) or PNG, by dst's extension, and
// writes it atomically.
func writeImage(dst string, img image.Image, ex imgmeta.Exif) error {
	var buf bytes.Buffer
	switch strings.ToLower(filepath.Ext(dst)) {
	case ".png":
		if err := png.Encode(&buf, img); err != nil {
			return err
		}
	case ".jpg", ".jpeg":
		if _, ok := img.(*image.Gray); !ok {
			flat := image.NewRGBA(img.Bounds())
			draw.Draw(flat, flat.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
			draw.Draw(flat, flat.Bounds(), img, img.Bounds().Min, draw.Over)
			img = flat
		}
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: jpegQuality}); err != nil {
			return err
		}
		if app1 := imgmeta.BuildAPP1(ex); app1 != nil {
			if out, err := imgmeta.InjectJPEG(buf.Bytes(), app1); err == nil {
				return writeFileAtomic(dst, out)
			}
		}
	default:
		return errors.New("convert writes only JPEG or PNG")
	}
	return writeFileAtomic(dst, buf.Bytes())
}

// writeFileAtomic writes data to a dot-file next to dst and renames it into
// place, so a reader never sees half a photo.
func writeFileAtomic(dst string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(dst), "."+filepath.Base(dst)+".*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, dst); err != nil {
		os.Remove(name)
		return err
	}
	return nil
}
