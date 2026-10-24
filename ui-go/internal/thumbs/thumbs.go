// Package thumbs finds and generates the 300x300 thumbs the UI requests with
// ?thumb=1. Mirrors the generator in ui/cron/fileServer.ts (which matches the
// Python SampleConfig._generate_thumbnail): a center-cropped square written
// atomically into the sibling .thumbs folder as <name>.<ext>.jpg, or
// <name>.<ext>.png when the source has alpha. Image thumbs are turned upright
// by their EXIF orientation. A thumb carries the mtime of the source it was
// made from, and one whose mtime differs is stale: that catches an edit, a
// replacement with an older date, and thumbs made before orientation was
// applied (by older code, ui/cron/fileServer.ts or the sampler).
package thumbs

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	_ "image/gif"

	_ "golang.org/x/image/bmp"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"

	"github.com/geocine/aitoolkit/internal/imgmeta"
	"github.com/geocine/aitoolkit/internal/paths"
)

// Size is the thumb edge length.
const Size = 300

// MaxPixels caps what Decode will decode: a few KB of PNG can declare a
// canvas that needs gigabytes, and the server shares its memory with the
// training run.
const MaxPixels = 100_000_000

// Decode decodes an image after checking the size its header declares.
func Decode(r io.ReadSeeker) (image.Image, error) {
	cfg, _, err := image.DecodeConfig(r)
	if err != nil {
		return nil, err
	}
	if int64(cfg.Width)*int64(cfg.Height) > MaxPixels {
		return nil, fmt.Errorf("image is %dx%d, more than %d pixels", cfg.Width, cfg.Height, MaxPixels)
	}
	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	img, _, err := image.Decode(r)
	return img, err
}

var (
	imageExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".gif": true, ".bmp": true}
	videoExts = map[string]bool{".mp4": true, ".avi": true, ".mov": true, ".mkv": true, ".wmv": true, ".m4v": true, ".flv": true}
	// audio thumbs: the embedded cover when the file has one, else a waveform
	audioExts = map[string]bool{".mp3": true, ".wav": true, ".flac": true, ".ogg": true}
)

// Base is the thumb path without its .png/.jpg suffix: <dir>/.thumbs/<name>.
func Base(sourcePath string) string {
	return filepath.Join(filepath.Dir(sourcePath), ".thumbs", filepath.Base(sourcePath))
}

// source is the file a Base path was made from: <dir>/<name>.
func source(base string) string {
	return filepath.Join(filepath.Dir(filepath.Dir(base)), filepath.Base(base))
}

// Lookup returns an existing thumb for base (.png first, then .jpg), or ""
// when there is none or it was not made from the source as it is now (Ensure
// then makes a new one). When the source cannot be read, any thumb is
// returned.
func Lookup(base string) (string, os.FileInfo) {
	var srcMtime time.Time
	if st, err := os.Stat(source(base)); err == nil {
		srcMtime = st.ModTime()
	}
	return lookup(base, srcMtime)
}

// lookup returns the thumb for base unless its mtime differs from
// sourceMtime (a zero sourceMtime accepts any thumb).
func lookup(base string, sourceMtime time.Time) (string, os.FileInfo) {
	for _, c := range []string{base + ".png", base + ".jpg"} {
		if st, err := os.Stat(c); err == nil && st.Mode().IsRegular() {
			if !sourceMtime.IsZero() && !madeFrom(c, st.ModTime(), sourceMtime) {
				return "", nil
			}
			return c, st
		}
	}
	return "", nil
}

// madeFrom reports whether the thumb at path, with mtime thumbMtime, was made
// from a source with mtime sourceMtime.
func madeFrom(path string, thumbMtime, sourceMtime time.Time) bool {
	if thumbMtime.Equal(sourceMtime) {
		return true
	}
	mu.Lock()
	s, ok := stamps[path]
	mu.Unlock()
	return ok && s.source.Equal(sourceMtime) && s.thumb.Equal(thumbMtime)
}

// Orientation is the EXIF orientation (1..8) of an image file's bytes, 1 when
// it has none. It reads the whole file: a WebP keeps its EXIF chunk after the
// image data. Orient(Fit(img, n), Orientation(data)) is the upright preview.
func Orientation(data []byte) int {
	return imgmeta.SniffHead(data).Exif.Orientation
}

// The manager provisions a portable FFmpeg at <repo>/.ffmpeg (see
// manager/ffmpeg.py); prefer it over PATH. Its Linux build is a shared one, so
// running it directly needs .ffmpeg/lib on LD_LIBRARY_PATH.
var ffmpegExe, ffmpegEnv = resolveFfmpeg()

func resolveFfmpeg() (string, []string) {
	name := "ffmpeg"
	if runtime.GOOS == "windows" {
		name = "ffmpeg.exe"
	}
	local := filepath.Join(paths.ToolkitRoot, ".ffmpeg", "bin", name)
	if st, err := os.Stat(local); err != nil || st.IsDir() {
		return "ffmpeg", nil
	}
	libDir := filepath.Join(paths.ToolkitRoot, ".ffmpeg", "lib")
	if runtime.GOOS != "linux" {
		return local, nil
	}
	if _, err := os.Stat(libDir); err != nil {
		return local, nil
	}
	env := os.Environ()
	if prior := os.Getenv("LD_LIBRARY_PATH"); prior != "" {
		env = append(env, "LD_LIBRARY_PATH="+libDir+string(os.PathListSeparator)+prior)
	} else {
		env = append(env, "LD_LIBRARY_PATH="+libDir)
	}
	return local, env
}

// FFmpeg is the ffmpeg the thumbnails use: the manager's portable build when
// present, else "ffmpeg" from PATH. A nil env means the process environment.
func FFmpeg() (exe string, env []string) {
	return ffmpegExe, ffmpegEnv
}

var (
	mu            sync.Mutex
	ffmpegMissing bool
	// failed generations, keyed on source mtime so a re-written file retries;
	// the gallery re-requests thumbs constantly and must not re-run decoders
	// against a broken file on every poll
	failed   = map[string]bool{}
	inFlight = map[string]*pending{}
	// thumbs whose mtime the filesystem would not set to their source's
	// (Chtimes refused or ignored), so they still count as fresh instead of
	// being made again on every request
	stamps = map[string]stamp{}
	// a gallery burst can request hundreds of missing thumbs at once
	slots = make(chan struct{}, 4)
)

type pending struct {
	done chan struct{}
	path string
}

type stamp struct{ source, thumb time.Time }

// Ensure returns the thumb for sourcePath at base, generating it when it is
// missing or was made from another sourceMtime, or "" when none could be
// generated (unsupported format, no ffmpeg, corrupt file). A video or audio
// source keeps a thumb of any age when ffmpeg is missing: the sampler or the
// old UI may have made it. Concurrent calls for one base share the work.
func Ensure(sourcePath, base string, sourceMtime time.Time) string {
	if p, _ := lookup(base, sourceMtime); p != "" {
		return p
	}
	p := ensure(sourcePath, base, sourceMtime)
	if p == "" && !canGenerate(sourcePath) {
		p, _ = lookup(base, time.Time{})
	}
	return p
}

// canGenerate reports whether generate can make a thumb for sourcePath.
func canGenerate(sourcePath string) bool {
	ext := strings.ToLower(filepath.Ext(sourcePath))
	if imageExts[ext] {
		return true
	}
	mu.Lock()
	defer mu.Unlock()
	return (videoExts[ext] || audioExts[ext]) && !ffmpegMissing
}

func ensure(sourcePath, base string, sourceMtime time.Time) string {
	failKey := fmt.Sprintf("%s:%d", base, sourceMtime.UnixNano())
	mu.Lock()
	if failed[failKey] {
		mu.Unlock()
		return ""
	}
	if p, ok := inFlight[base]; ok {
		mu.Unlock()
		<-p.done
		return p.path
	}
	p := &pending{done: make(chan struct{})}
	inFlight[base] = p
	mu.Unlock()

	slots <- struct{}{}
	written, err := generate(sourcePath, base, sourceMtime)
	<-slots
	if err != nil {
		log.Printf("Failed to generate thumbnail for %s: %v", sourcePath, err)
		written = ""
	}
	mu.Lock()
	if written == "" {
		failed[failKey] = true
	}
	delete(inFlight, base)
	mu.Unlock()
	p.path = written
	close(p.done)
	return written
}

func generate(sourcePath, base string, sourceMtime time.Time) (string, error) {
	ext := strings.ToLower(filepath.Ext(sourcePath))
	isImage, isVideo, isAudio := imageExts[ext], videoExts[ext], audioExts[ext]
	mu.Lock()
	noFfmpeg := ffmpegMissing
	mu.Unlock()
	if (!isImage && !isVideo && !isAudio) || ((isVideo || isAudio) && noFfmpeg) {
		return "", nil
	}

	var img image.Image
	hasAlpha := false
	orientation := 1
	if isImage {
		data, err := os.ReadFile(sourcePath)
		if err != nil {
			return "", err
		}
		// animated formats decode their first frame
		img, err = Decode(bytes.NewReader(data))
		if err != nil {
			return "", err
		}
		hasAlpha = HasAlpha(img)
		orientation = Orientation(data)
	}
	suffix := ".jpg"
	if hasAlpha {
		suffix = ".png"
	}
	thumbPath := base + suffix
	if err := os.MkdirAll(filepath.Dir(thumbPath), 0o755); err != nil {
		return "", err
	}
	// per-process tmp name then atomic rename (same as the Python generator) so
	// a concurrent request never reads a partial thumb; the suffix is what
	// ffmpeg detects its output format from
	tmpPath := fmt.Sprintf("%s.%d.tmp%s", thumbPath, os.Getpid(), suffix)
	var err error
	switch {
	case isImage:
		err = writeImageThumb(img, hasAlpha, orientation, tmpPath)
	case isVideo:
		err = runFfmpeg("-i", sourcePath, "-frames:v", "1", "-vf", squareVF, "-q:v", "2", tmpPath)
	default:
		done := false
		if ext == ".mp3" {
			// embedded cover first (ffmpeg exposes an attached picture as a video stream)
			if runFfmpeg("-i", sourcePath, "-map", "0:v:0", "-frames:v", "1", "-vf", squareVF, "-q:v", "2", tmpPath) == nil {
				done = true
			}
		}
		if !done {
			wave := fmt.Sprintf("color=c=0x1a1a1a:s=%dx%d[bg];[0:a]showwavespic=s=%dx%d:colors=0xfbbf24@0.9[w];[bg][w]overlay=format=auto", Size, Size, Size, Size)
			err = runFfmpeg("-i", sourcePath, "-filter_complex", wave, "-frames:v", "1", "-q:v", "2", tmpPath)
		}
	}
	if err == nil {
		err = os.Rename(tmpPath, thumbPath)
	}
	if err != nil {
		os.Remove(tmpPath)
		return "", err
	}
	// a stale thumb under the other suffix would win in Lookup (.png first)
	if hasAlpha {
		os.Remove(base + ".jpg")
	} else {
		os.Remove(base + ".png")
	}
	stampThumb(thumbPath, sourceMtime)
	return thumbPath, nil
}

// stampThumb sets the thumb's mtime to its source's, which is what makes it
// fresh in lookup. Where the filesystem does not keep that mtime, the pair is
// remembered for this process instead.
func stampThumb(thumbPath string, sourceMtime time.Time) {
	if sourceMtime.IsZero() {
		return
	}
	os.Chtimes(thumbPath, time.Now(), sourceMtime)
	st, err := os.Stat(thumbPath)
	mu.Lock()
	defer mu.Unlock()
	if err == nil && !st.ModTime().Equal(sourceMtime) {
		stamps[thumbPath] = stamp{source: sourceMtime, thumb: st.ModTime()}
	} else {
		delete(stamps, thumbPath)
	}
}

var squareVF = fmt.Sprintf("crop='min(iw,ih)':'min(iw,ih)',scale=%d:%d", Size, Size)

func runFfmpeg(args ...string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, ffmpegExe, append([]string{"-y", "-loglevel", "error"}, args...)...)
	cmd.Env = ffmpegEnv
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	hideWindow(cmd)
	if err := cmd.Run(); err != nil {
		if errors.Is(err, exec.ErrNotFound) || errors.Is(err, os.ErrNotExist) {
			mu.Lock()
			ffmpegMissing = true
			mu.Unlock()
			return err
		}
		return fmt.Errorf("ffmpeg exited with %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	return nil
}

// HasAlpha reports whether img has an alpha channel, like sharp's
// metadata().hasAlpha (a channel property, not whether any pixel is clear).
func HasAlpha(img image.Image) bool {
	switch m := img.(type) {
	case *image.NRGBA, *image.NRGBA64, *image.Alpha, *image.Alpha16, *image.NYCbCrA:
		return true
	case *image.Paletted:
		for _, c := range m.Palette {
			if _, _, _, a := c.RGBA(); a != 0xffff {
				return true
			}
		}
	}
	return false
}

// Cover scales img to fill a Size x Size square and center-crops it
// (sharp's resize with fit: 'cover').
func Cover(img image.Image, hasAlpha bool) draw.Image {
	b := img.Bounds()
	side := min(b.Dx(), b.Dy())
	x0 := b.Min.X + (b.Dx()-side)/2
	y0 := b.Min.Y + (b.Dy()-side)/2
	src := image.Rect(x0, y0, x0+side, y0+side)
	dstRect := image.Rect(0, 0, Size, Size)
	var dst draw.Image
	if hasAlpha {
		dst = image.NewNRGBA(dstRect)
	} else {
		dst = image.NewRGBA(dstRect)
	}
	draw.CatmullRom.Scale(dst, dstRect, img, src, draw.Src, nil)
	return dst
}

func writeImageThumb(img image.Image, hasAlpha bool, orientation int, out string) error {
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	// the crop is centred, so turning the square is the same as cropping the
	// upright image, at thumb size instead of full size
	thumb := Orient(Cover(img, hasAlpha), orientation)
	if hasAlpha {
		err = png.Encode(f, thumb)
	} else {
		err = jpeg.Encode(f, thumb, &jpeg.Options{Quality: 90})
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// Orient turns img upright for an EXIF orientation (1..8): mirrors for 2 and
// 4, a half turn for 3, and for 5-8 a quarter turn (with a mirror for 5 and
// 7) that swaps width and height. Other values return img unchanged. The
// result has img's pixel type when that is RGBA or NRGBA.
func Orient(img draw.Image, orientation int) draw.Image {
	if orientation < 2 || orientation > 8 {
		return img
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.Rect(0, 0, w, h)
	if orientation >= 5 {
		dst = image.Rect(0, 0, h, w)
	}
	var out draw.Image
	if _, ok := img.(*image.NRGBA); ok {
		out = image.NewNRGBA(dst)
	} else {
		out = image.NewRGBA(dst)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var dx, dy int
			switch orientation {
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
			out.Set(dx, dy, img.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	return out
}

// Fit scales img down so neither side exceeds maxPx, keeping its aspect
// (never up), flattened onto white so it encodes as JPEG: the preview an
// agent looks at (MCP aitk_view), where a center crop would hide the edges.
func Fit(img image.Image, maxPx int) *image.RGBA {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if maxPx > 0 && (w > maxPx || h > maxPx) {
		if w >= h {
			w, h = maxPx, max(1, h*maxPx/b.Dx())
		} else {
			w, h = max(1, w*maxPx/b.Dy()), maxPx
		}
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), image.White, image.Point{}, draw.Src)
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, b, draw.Over, nil)
	return dst
}
