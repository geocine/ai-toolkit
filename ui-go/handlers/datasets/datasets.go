// Package datasets ports ui/src/app/api/datasets/{list,listImages,upload,create,delete}
// (create and delete are route.tsx upstream) and api/loras{,/upload}.
package datasets

import (
	"bytes"
	"compress/gzip"

	"encoding/json"
	"errors"
	"github.com/andybalholm/brotli"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/settings"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /api/datasets/list", list)
	httpx.Handle(mux, "POST /api/datasets/listImages", listImages)
	httpx.Handle(mux, "POST /api/datasets/upload", upload)
	httpx.Handle(mux, "POST /api/datasets/create", create)
	httpx.Handle(mux, "POST /api/datasets/delete", del)
	httpx.Handle(mux, "GET /api/loras", listLoras)
	httpx.Handle(mux, "POST /api/loras/upload", uploadLora)
}

// list returns the dataset folder names, creating the datasets root if needed.
func list(w http.ResponseWriter, r *http.Request) {
	fail := func(err error) {
		log.Printf("GET /api/datasets/list: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to fetch datasets")
	}
	root, err := settings.DatasetsRoot(r.Context())
	if err != nil {
		fail(err)
		return
	}
	if _, err := os.Stat(root); err != nil {
		if err := os.Mkdir(root, 0o755); err != nil {
			fail(err)
			return
		}
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		fail(err)
		return
	}
	folders := []string{}
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			folders = append(folders, e.Name())
		}
	}
	httpx.OK(w, folders)
}

var mediaExtensions = map[string]bool{
	".png": true, ".jpg": true, ".jpeg": true, ".webp": true, ".mp4": true, ".avi": true, ".mov": true,
	".mkv": true, ".wmv": true, ".m4v": true, ".flv": true, ".mp3": true, ".wav": true, ".flac": true, ".ogg": true,
}

// findMedia lists media files under dir, skipping dot entries and _controls folders.
func findMedia(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out, subdirs []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		p := filepath.Join(dir, name)
		if e.IsDir() {
			if name != "_controls" {
				subdirs = append(subdirs, p)
			}
		} else if e.Type().IsRegular() && mediaExtensions[strings.ToLower(filepath.Ext(name))] {
			out = append(out, p)
		}
	}
	for _, sub := range subdirs {
		nested, err := findMedia(sub)
		if err != nil {
			return nil, err
		}
		out = append(out, nested...)
	}
	return out, nil
}

var (
	acceptsBrotli = regexp.MustCompile(`\bbr\b`)
	acceptsGzip   = regexp.MustCompile(`\bgzip\b`)
)

// listImages answers {root, images}: one shared root (with a trailing
// separator) and each file's path below it, sorted.
func listImages(w http.ResponseWriter, r *http.Request) {
	root, err := settings.DatasetsRoot(r.Context())
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	field, err := bodyField(r, "datasetName")
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	name, ok := field.(string)
	if !ok || jsTrim(name) == "" {
		httpx.Error(w, http.StatusBadRequest, "Invalid dataset name")
		return
	}
	folder, ok := directChild(root, name)
	if !ok {
		httpx.Error(w, http.StatusBadRequest, "Invalid dataset name")
		return
	}
	if _, err := os.Stat(folder); err != nil {
		httpx.Error(w, http.StatusNotFound, "Folder '"+name+"' not found")
		return
	}
	files, err := findMedia(folder)
	if err != nil {
		log.Printf("Error finding images: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to process request")
		return
	}
	sort.SliceStable(files, func(i, j int) bool { return localeCompare(files[i], files[j]) < 0 })

	prefix := folder + string(filepath.Separator)
	images := make([]string, len(files))
	for i, f := range files {
		images[i] = f[len(prefix):]
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.Encode(struct {
		Root   string   `json:"root"`
		Images []string `json:"images"`
	}{prefix, images})
	payload := bytes.TrimRight(buf.Bytes(), "\n")

	w.Header().Set("Content-Type", "application/json")
	accept := r.Header.Get("Accept-Encoding")
	if acceptsBrotli.MatchString(accept) {
		w.Header().Set("Content-Encoding", "br")
		br := brotli.NewWriterLevel(w, brotli.BestCompression)
		br.Write(payload)
		br.Close()
		return
	}
	if acceptsGzip.MatchString(accept) {
		w.Header().Set("Content-Encoding", "gzip")
		gz := gzip.NewWriter(w)
		gz.Write(payload)
		gz.Close()
		return
	}
	w.Write(payload)
}

// uploadName cleans an uploaded file name like upstream's
// name.replace(/[^a-zA-Z0-9.-]/g, '_').
func uploadName(name string) string {
	return replaceUTF16(name, func(r rune) bool {
		return r < 0x80 && (r == '.' || r == '-' || unicode.IsLetter(r) || unicode.IsDigit(r))
	}, "_")
}

type formFile struct {
	name   string // client file name as sent, not reduced to a basename
	tmp    string
	isText bool // a plain string entry under "files"
}

// upload saves multipart "files" into <datasets>/<datasetName>. Parts are
// streamed to temp files first because datasetName may follow the files.
func upload(w http.ResponseWriter, r *http.Request) {
	fail := func(err error) {
		log.Printf("Upload error: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Error uploading files")
	}
	root, err := settings.DatasetsRoot(r.Context())
	if err != nil {
		fail(err)
		return
	}
	if root == "" {
		httpx.Error(w, http.StatusInternalServerError, "Datasets path not found")
		return
	}
	mr, err := r.MultipartReader()
	if err != nil {
		fail(err)
		return
	}
	tmpDir, err := os.MkdirTemp("", "aitk-upload-")
	if err != nil {
		fail(err)
		return
	}
	defer os.RemoveAll(tmpDir)

	var files []formFile
	var datasetName *string
	datasetNameIsFile := false
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			fail(err)
			return
		}
		_, params, _ := mime.ParseMediaType(part.Header.Get("Content-Disposition"))
		filename, isFile := params["filename"]
		switch params["name"] {
		case "files":
			if !isFile {
				io.Copy(io.Discard, part)
				files = append(files, formFile{isText: true})
				break
			}
			tmp, err := os.CreateTemp(tmpDir, "part-")
			if err != nil {
				fail(err)
				return
			}
			_, err = io.Copy(tmp, part)
			tmp.Close()
			if err != nil {
				fail(err)
				return
			}
			files = append(files, formFile{name: filename, tmp: tmp.Name()})
		case "datasetName":
			if datasetName != nil || datasetNameIsFile {
				io.Copy(io.Discard, part)
				break
			}
			if isFile {
				datasetNameIsFile = true
				io.Copy(io.Discard, part)
				break
			}
			v, err := io.ReadAll(part)
			if err != nil {
				fail(err)
				return
			}
			s := string(v)
			datasetName = &s
		default:
			io.Copy(io.Discard, part)
		}
		part.Close()
	}

	if len(files) == 0 {
		httpx.Error(w, http.StatusBadRequest, "No files provided")
		return
	}
	if datasetName == nil || datasetNameIsFile || jsTrim(*datasetName) == "" {
		httpx.Error(w, http.StatusBadRequest, "Invalid dataset name")
		return
	}
	uploadDir, ok := directChild(root, *datasetName)
	if !ok {
		httpx.Error(w, http.StatusBadRequest, "Invalid dataset name")
		return
	}
	if err := os.MkdirAll(uploadDir, 0o755); err != nil {
		fail(err)
		return
	}
	saved := []string{}
	for _, f := range files {
		if f.isText {
			fail(errors.New("files entry is not a file"))
			return
		}
		name := uploadName(f.name)
		if err := moveFile(f.tmp, filepath.Join(uploadDir, name)); err != nil {
			fail(err)
			return
		}
		saved = append(saved, name)
	}
	httpx.OK(w, struct {
		Message string   `json:"message"`
		Files   []string `json:"files"`
	}{"Files uploaded successfully", saved})
}

// moveFile renames src to dst, copying when they are on different volumes.
// dst must not be a directory (writeFile onto a directory fails upstream too).
func moveFile(src, dst string) error {
	if st, err := os.Stat(dst); err == nil && st.IsDir() {
		return errors.New("EISDIR: illegal operation on a directory, open '" + dst + "'")
	}
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// bodyField is `const { key } = await request.json()`: a JSON null body
// throws (error), a non-object body yields undefined (nil).
func bodyField(r *http.Request, key string) (any, error) {
	var v any
	if err := httpx.DecodeJSON(r, &v); err != nil {
		return nil, err
	}
	if v == nil {
		return nil, errors.New("cannot destructure null body")
	}
	if m, ok := v.(map[string]any); ok {
		return m[key], nil
	}
	return nil, nil
}

// datasetSlug lowercases and replaces every run of characters outside
// [a-z0-9] with one underscore (name.toLowerCase().replace(/[^a-z0-9]+/g, '_')).
func datasetSlug(name string) string {
	// JS lowercases U+0130 to "i" + combining dot, which leaves a trailing "_"
	lower := strings.ToLower(strings.ReplaceAll(name, "İ", "i̇"))
	var b strings.Builder
	inRun := false
	for _, r := range lower {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			inRun = false
			continue
		}
		if !inRun {
			b.WriteByte('_')
			inRun = true
		}
	}
	return b.String()
}

func create(w http.ResponseWriter, r *http.Request) {
	fail := func(err error) {
		log.Printf("POST /api/datasets/create: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to create dataset")
	}
	field, err := bodyField(r, "name")
	if err != nil {
		fail(err)
		return
	}
	raw, ok := field.(string)
	if !ok {
		fail(errors.New("name is not a string"))
		return
	}
	name := datasetSlug(raw)
	root, err := settings.DatasetsRoot(r.Context())
	if err != nil {
		fail(err)
		return
	}
	p := filepath.Join(root, name)
	if _, err := os.Stat(p); err != nil {
		if err := os.Mkdir(p, 0o755); err != nil {
			fail(err)
			return
		}
	}
	httpx.OK(w, struct {
		Success bool   `json:"success"`
		Name    string `json:"name"`
	}{true, name})
}

func del(w http.ResponseWriter, r *http.Request) {
	fail := func(err error) {
		log.Printf("POST /api/datasets/delete: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to delete dataset")
	}
	field, err := bodyField(r, "name")
	if err != nil {
		fail(err)
		return
	}
	name, ok := field.(string)
	if !ok || jsTrim(name) == "" {
		httpx.Error(w, http.StatusBadRequest, "Invalid dataset name")
		return
	}
	root, err := settings.DatasetsRoot(r.Context())
	if err != nil {
		fail(err)
		return
	}
	p, ok := directChild(root, name)
	if !ok {
		httpx.Error(w, http.StatusBadRequest, "Invalid dataset name")
		return
	}
	if _, err := os.Stat(p); err == nil {
		if err := os.RemoveAll(p); err != nil {
			fail(err)
			return
		}
	}
	httpx.OK(w, map[string]bool{"success": true})
}
