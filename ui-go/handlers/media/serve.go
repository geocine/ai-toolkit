package media

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/safepath"
	"github.com/geocine/aitoolkit/internal/thumbs"
)

// Mirrors serveFile() in ui/cron/fileServer.ts, which serves these routes in
// production (the Next routes under api/{img,files,audio/art} are its
// fallback). /api/img/ serves media inline with ETag revalidation,
// /api/files/ is a forced download, /api/audio/art/ is the audio file's thumb
// (embedded cover / waveform) served like an image.

const (
	prefixImg   = "/api/img/"
	prefixFiles = "/api/files/"
	prefixArt   = "/api/audio/art/"
)

var contentTypes = map[string]string{
	".jpg":         "image/jpeg",
	".jpeg":        "image/jpeg",
	".png":         "image/png",
	".gif":         "image/gif",
	".webp":        "image/webp",
	".svg":         "image/svg+xml",
	".bmp":         "image/bmp",
	".safetensors": "application/octet-stream",
	".zip":         "application/zip",
	// Videos
	".mp4": "video/mp4",
	".avi": "video/x-msvideo",
	".mov": "video/quicktime",
	".mkv": "video/x-matroska",
	".wmv": "video/x-ms-wmv",
	".m4v": "video/x-m4v",
	".flv": "video/x-flv",
	// Audio
	".mp3":  "audio/mpeg",
	".wav":  "audio/wav",
	".flac": "audio/flac",
	".ogg":  "audio/ogg",
}

func contentType(p string) string {
	if t, ok := contentTypes[strings.ToLower(filepath.Ext(p))]; ok {
		return t
	}
	return "application/octet-stream"
}

func plain(w http.ResponseWriter, status int, body string) {
	w.WriteHeader(status)
	io.WriteString(w, body)
}

// requestedPath decodes the part of the raw URL path after prefix one segment
// at a time, so both URL forms resolve to the same file:
//
//	/api/files/%2Fmnt%2Fout%2Fjob%2Ffile.safetensors   (legacy: one segment)
//	/api/files/%2Fmnt%2Fout%2Fjob/file.safetensors     (folder / filename, what the UI emits)
//
// Windows folders arrive as `C%3A%5Cout%5Cjob` and decode to `C:\out\job`.
func requestedPath(r *http.Request, prefix string) (string, error) {
	rest := strings.TrimPrefix(r.URL.EscapedPath(), prefix)
	segs := strings.Split(rest, "/")
	for i, s := range segs {
		d, err := decodeURIComponent(s)
		if err != nil {
			return "", err
		}
		segs[i] = d
	}
	return strings.Join(segs, "/"), nil
}

func serveFile(w http.ResponseWriter, r *http.Request, prefix string) {
	isArt := prefix == prefixArt
	isImg := prefix == prefixImg || isArt

	filePath, err := requestedPath(r, prefix)
	if err != nil {
		log.Printf("Error serving file: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	rt, err := roots(r.Context())
	if err != nil {
		log.Printf("Error serving file: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	allowed := []string{rt.Datasets, rt.Training}
	if isImg {
		allowed = append(allowed, rt.Data)
	}
	resolved := safepath.Resolve(filePath, allowed)
	if resolved == "" {
		log.Printf("Access denied: %s not in %s", filePath, strings.Join(allowed, ", "))
		plain(w, http.StatusForbidden, "Access denied")
		return
	}

	// ?thumb=1 serves the thumb from the sibling .thumbs folder, generating it
	// when missing; falls through to the full file only if generation is not
	// possible. Album art is always the thumb.
	if isArt || (isImg && r.URL.Query().Has("thumb")) {
		base := thumbs.Base(resolved)
		thumbPath, st := thumbs.Lookup(base)
		if st == nil {
			if src, err := os.Stat(resolved); err == nil && src.Mode().IsRegular() {
				if written := thumbs.Ensure(resolved, base, src.ModTime()); written != "" {
					if s, err := os.Stat(written); err == nil {
						thumbPath, st = written, s
					}
				}
			}
		}
		if st != nil && st.Mode().IsRegular() {
			resolved = thumbPath
		} else if isArt {
			serveAlbumArt(w, r, resolved)
			return
		}
	}

	st, err := os.Stat(resolved)
	if err != nil {
		plain(w, http.StatusNotFound, "File not found")
		return
	}
	if !st.Mode().IsRegular() {
		plain(w, http.StatusBadRequest, "Not a file")
		return
	}

	h := w.Header()
	h.Set("Content-Type", contentType(resolved))
	h.Set("Accept-Ranges", "bytes")
	if isImg {
		// weak ETag from inode/size/mtime: cheap and stable enough for
		// revalidating thumbnails/videos the UI requests over and over
		etag := fmt.Sprintf(`W/"%s-%s-%s"`, strconv.FormatUint(fileID(resolved, st), 36),
			strconv.FormatInt(st.Size(), 36), strconv.FormatInt(st.ModTime().UnixNano(), 36))
		h.Set("Cache-Control", "public, max-age=86400, immutable")
		h.Set("ETag", etag)
		if r.Header.Get("If-None-Match") == etag {
			h.Del("Content-Type")
			h.Del("Accept-Ranges")
			w.WriteHeader(http.StatusNotModified)
			return
		}
	} else {
		h.Set("Cache-Control", "public, max-age=86400")
		h.Set("Content-Disposition", `attachment; filename="`+encodeURIComponent(filepath.Base(resolved))+`"`)
		h.Set("X-Content-Type-Options", "nosniff")
	}

	size := st.Size()
	start, end := int64(0), size-1
	status := http.StatusOK
	if rng := r.Header.Get("Range"); rng != "" {
		// open-ended ranges (`bytes=0-`) serve to EOF
		parts := strings.Split(strings.Replace(rng, "bytes=", "", 1), "-")
		s, okS := jsParseInt(parts[0])
		e, okE := size-1, true
		if len(parts) > 1 && parts[1] != "" {
			e, okE = jsParseInt(parts[1])
		}
		if !okS || !okE || s > e || s >= size {
			for _, k := range []string{"Content-Type", "Accept-Ranges", "Cache-Control", "ETag", "Content-Disposition", "X-Content-Type-Options"} {
				h.Del(k)
			}
			h.Set("Content-Range", fmt.Sprintf("bytes */%d", size))
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		// upstream does not clamp; clamping keeps Content-Length truthful
		start, end = s, min(e, size-1)
		status = http.StatusPartialContent
		h.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
	}
	h.Set("Content-Length", strconv.FormatInt(end-start+1, 10))
	w.WriteHeader(status)
	if r.Method == http.MethodHead {
		return
	}
	f, err := os.Open(resolved)
	if err != nil {
		return
	}
	defer f.Close()
	io.Copy(w, io.NewSectionReader(f, start, end-start+1))
}

// encodeURIComponent escapes everything but A-Z a-z 0-9 - _ . ! ~ * ' ( ).
func encodeURIComponent(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.IndexByte("-_.!~*'()", c) >= 0 {
			b.WriteByte(c)
		} else {
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}
