package media

import (
	"errors"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/google/uuid"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/safepath"
)

var mediaExtRe = regexp.MustCompile(`(?i)\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$`)

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// unlink removes a file; like fs.unlink it refuses directories.
func unlink(p string) error {
	if st, err := os.Lstat(p); err == nil && st.IsDir() {
		return errors.New("EISDIR: illegal operation on a directory, unlink '" + p + "'")
	}
	return os.Remove(p)
}

// imgDelete is POST /api/img/delete: remove a dataset/training media file and its .txt caption.
func imgDelete(w http.ResponseWriter, r *http.Request) {
	fail := func(err error) {
		log.Printf("POST /api/img/delete: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to create dataset")
	}
	body, isNull, err := readBody(r)
	if err != nil || isNull {
		fail(errors.Join(err, errors.New("invalid body")))
		return
	}
	rt, err := roots(r.Context())
	if err != nil {
		fail(err)
		return
	}
	imgPath, ok := str(body, "imgPath")
	if !ok {
		fail(errors.New("imgPath must be a string"))
		return
	}
	if !strings.HasPrefix(imgPath, rt.Datasets) && !strings.HasPrefix(imgPath, rt.Training) {
		httpx.Error(w, http.StatusBadRequest, "Invalid image path")
		return
	}
	if !mediaExtRe.MatchString(strings.ToLower(imgPath)) {
		httpx.Error(w, http.StatusBadRequest, "Not an image")
		return
	}
	if !exists(imgPath) {
		httpx.OK(w, map[string]bool{"success": true})
		return
	}
	if err := unlink(imgPath); err != nil {
		fail(err)
		return
	}
	if captionPath := stripExt(imgPath) + ".txt"; exists(captionPath) {
		if err := unlink(captionPath); err != nil {
			fail(err)
			return
		}
	}
	httpx.OK(w, map[string]bool{"success": true})
}

// imgUpload is POST /api/img/upload: save multipart `files` under <DATA_ROOT>/images
// with uuid names, keeping each upload's extension.
func imgUpload(w http.ResponseWriter, r *http.Request) {
	fail := func(err error) {
		log.Printf("Upload error: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Error uploading files")
	}
	rt, err := roots(r.Context())
	if err != nil {
		fail(err)
		return
	}
	if rt.Data == "" {
		httpx.Error(w, http.StatusInternalServerError, "Data root path not found")
		return
	}
	imgRoot := filepath.Join(rt.Data, "images")
	mr, err := r.MultipartReader()
	if err != nil {
		fail(err)
		return
	}
	saved := []string{}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			fail(err)
			return
		}
		if part.FormName() != "files" {
			part.Close()
			continue
		}
		name, isFile := uploadName(part.Header.Get("Content-Disposition"))
		if !isFile {
			// a plain string field: upstream's file.arrayBuffer() throws
			fail(errors.New("files entry is not a file"))
			return
		}
		ext := name[strings.LastIndex(name, ".")+1:]
		if ext == "" {
			ext = "jpg"
		}
		if err := os.MkdirAll(imgRoot, 0o755); err != nil {
			fail(err)
			return
		}
		dest := filepath.Join(imgRoot, uuid.NewString()+"."+ext)
		if err := writeStream(dest, part); err != nil {
			fail(err)
			return
		}
		saved = append(saved, dest)
	}
	if len(saved) == 0 {
		httpx.Error(w, http.StatusBadRequest, "No files provided")
		return
	}
	httpx.OK(w, map[string]any{"message": "Files uploaded successfully", "files": saved})
}

// uploadName is the raw filename parameter (File.name; Go's FileName() would
// strip directories) and whether the part is a file at all.
func uploadName(disposition string) (string, bool) {
	_, params, err := mime.ParseMediaType(disposition)
	if err != nil {
		return "", false
	}
	name, ok := params["filename"]
	return name, ok
}

func writeStream(dest string, src io.Reader) error {
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, src); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// filesDelete is POST /api/files/delete: remove one file under the datasets or training root.
func filesDelete(w http.ResponseWriter, r *http.Request) {
	internal := func(err error) {
		log.Printf("Error deleting file: %v", err)
		textPlain(w, http.StatusInternalServerError, "Internal Server Error")
	}
	body, isNull, err := readBody(r)
	if err != nil || isNull {
		internal(errors.Join(err, errors.New("invalid body")))
		return
	}
	filePath, ok := str(body, "filePath")
	if !ok || filePath == "" {
		textPlain(w, http.StatusBadRequest, "filePath is required")
		return
	}
	decoded, err := decodeURIComponent(filePath)
	if err != nil {
		internal(err)
		return
	}
	rt, err := roots(r.Context())
	if err != nil {
		internal(err)
		return
	}
	allowed := []string{rt.Datasets, rt.Training}
	resolved := safepath.Resolve(decoded, allowed)
	if resolved == "" {
		log.Printf("Access denied: %s not in %s", decoded, strings.Join(allowed, ", "))
		textPlain(w, http.StatusForbidden, "Access denied")
		return
	}
	st, err := os.Stat(resolved)
	if err != nil {
		log.Printf("File not found: %s", resolved)
		textPlain(w, http.StatusNotFound, "File not found")
		return
	}
	if !st.Mode().IsRegular() {
		textPlain(w, http.StatusBadRequest, "Not a file")
		return
	}
	if err := os.Remove(resolved); err != nil {
		internal(err)
		return
	}
	httpx.OK(w, map[string]bool{"success": true})
}
