// Package thumbs finds and generates the 300x300 thumbs the UI requests with
// ?thumb=1. Mirrors the generator in ui/cron/fileServer.ts (which matches the
// Python SampleConfig._generate_thumbnail): a center-cropped square written
// atomically into the sibling .thumbs folder as <name>.<ext>.jpg, or
// <name>.<ext>.png when the source has alpha.
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

// Lookup returns an existing thumb for base (.png first, then .jpg).
func Lookup(base string) (string, os.FileInfo) {
	for _, c := range []string{base + ".png", base + ".jpg"} {
		if st, err := os.Stat(c); err == nil && st.Mode().IsRegular() {
			return c, st
		}
	}
	return "", nil
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

var (
	mu            sync.Mutex
	ffmpegMissing bool
	// failed generations, keyed on source mtime so a re-written file retries;
	// the gallery re-requests thumbs constantly and must not re-run decoders
	// against a broken file on every poll
	failed   = map[string]bool{}
	inFlight = map[string]*pending{}
	// a gallery burst can request hundreds of missing thumbs at once
	slots = make(chan struct{}, 4)
)

type pending struct {
	done chan struct{}
	path string
}

// Ensure generates the thumb for sourcePath at base when possible and returns
// the path it wrote, or "" when none could be generated (unsupported format,
// no ffmpeg, corrupt file). Concurrent calls for one base share the work.
func Ensure(sourcePath, base string, sourceMtime time.Time) string {
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
	written, err := generate(sourcePath, base)
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

func generate(sourcePath, base string) (string, error) {
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
		err = writeImageThumb(img, hasAlpha, tmpPath)
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
	return thumbPath, nil
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

func writeImageThumb(img image.Image, hasAlpha bool, out string) error {
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	thumb := Cover(img, hasAlpha)
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
