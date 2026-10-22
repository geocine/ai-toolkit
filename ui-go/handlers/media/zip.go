package media

import (
	"archive/zip"
	"compress/flate"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/geocine/aitoolkit/internal/httpx"
)

// Mirrors ui/src/app/api/zip/route.ts: zip a job's samples folder, or a
// dataset (all files or captions only), and return where the zip landed so
// /api/files/ can serve it.

var captionExts = map[string]bool{".txt": true, ".json": true, ".caption": true}

type zipResult struct {
	OK       bool   `json:"ok"`
	ZipPath  string `json:"zipPath"`
	FileName string `json:"fileName"`
}

// realpath is fs.realpath: absolute, symlinks resolved, must exist.
func realpath(p string) (string, error) {
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(abs)
}

func zipRoute(w http.ResponseWriter, r *http.Request) {
	internal := func(err error) {
		log.Printf("Zip error: %v", err)
		textPlain(w, http.StatusInternalServerError, "Internal Server Error")
	}
	body, isNull, err := readBody(r)
	if err != nil {
		internal(err)
		return
	}
	if !isNull {
		if t, _ := body["zipTarget"].(string); t == "dataset" || t == "dataset_captions" {
			if !truthy(body["datasetName"]) {
				httpx.Error(w, http.StatusBadRequest, "datasetName is required")
				return
			}
			name, ok := body["datasetName"].(string)
			if !ok {
				internal(errors.New("datasetName must be a string"))
				return
			}
			zipDataset(w, r, name, t == "dataset_captions", internal)
			return
		}
	}
	if isNull || !truthy(body["jobName"]) {
		httpx.Error(w, http.StatusBadRequest, "jobName is required")
		return
	}
	jobName, ok := body["jobName"].(string)
	if !ok {
		internal(errors.New("jobName must be a string"))
		return
	}
	rt, err := roots(r.Context())
	if err != nil {
		internal(err)
		return
	}
	trainingRoot, err := realpath(rt.Training)
	if err != nil {
		internal(err)
		return
	}
	// Deliberate deviation: upstream joins jobName unchecked, so "../x" writes
	// samples.zip outside the training root. Treat that like a missing folder.
	if rel, err := filepath.Rel(trainingRoot, filepath.Join(trainingRoot, jobName)); err != nil ||
		rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		textPlain(w, http.StatusNotFound, "Folder not found")
		return
	}
	// a missing samples folder fails here, before the stat below (upstream answers 500)
	folderPath, err := realpath(filepath.Join(trainingRoot, jobName, "samples"))
	if err != nil {
		internal(err)
		return
	}
	outputPath, err := filepath.Abs(filepath.Join(trainingRoot, jobName, "samples.zip"))
	if err != nil {
		internal(err)
		return
	}
	st, err := os.Stat(folderPath)
	if err != nil {
		textPlain(w, http.StatusNotFound, "Folder not found")
		return
	}
	if !st.IsDir() {
		textPlain(w, http.StatusBadRequest, "Not a directory")
		return
	}
	if err := removeIfExists(outputPath); err != nil {
		internal(err)
		return
	}
	// the directory contents, under the folder's base name in the zip
	rootName := filepath.Base(folderPath)
	var files []string
	err = filepath.WalkDir(folderPath, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() {
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		internal(err)
		return
	}
	if err := writeZip(outputPath, folderPath, rootName, files); err != nil {
		internal(err)
		return
	}
	httpx.OK(w, zipResult{OK: true, ZipPath: outputPath, FileName: filepath.Base(outputPath)})
}

func zipDataset(w http.ResponseWriter, r *http.Request, datasetName string, captionsOnly bool, internal func(error)) {
	rt, err := roots(r.Context())
	if err != nil {
		internal(err)
		return
	}
	datasetsRoot, err := realpath(rt.Datasets)
	if err != nil {
		internal(err)
		return
	}
	// basename strips any path segments so the name can't escape the datasets root
	safeName := jsBasename(datasetName)
	folderPath, err := realpath(filepath.Join(datasetsRoot, safeName))
	// "." and ".." survive basename and would zip the datasets root or the toolkit
	if err == nil && (safeName == "" || safeName == "." || safeName == ".." || !underRoot(folderPath, datasetsRoot) || folderPath == datasetsRoot) {
		err = errors.New("not a dataset folder")
	}
	if err != nil {
		httpx.Error(w, http.StatusNotFound, fmt.Sprintf("Dataset '%s' not found", datasetName))
		return
	}
	files, err := collectFiles(folderPath, captionsOnly)
	if err != nil {
		internal(err)
		return
	}
	if len(files) == 0 {
		msg := "Dataset is empty"
		if captionsOnly {
			msg = "No captions found"
		}
		httpx.Error(w, http.StatusNotFound, msg)
		return
	}
	// hidden dir at the datasets root: skipped by the dataset walkers but still
	// under the root, so /api/files/ may serve it
	zipDir := filepath.Join(datasetsRoot, ".zips")
	if err := os.MkdirAll(zipDir, 0o755); err != nil {
		internal(err)
		return
	}
	kind := "dataset"
	if captionsOnly {
		kind = "captions"
	}
	outputPath := filepath.Join(zipDir, safeName+"_"+kind+".zip")
	if err := removeIfExists(outputPath); err != nil {
		internal(err)
		return
	}
	if err := writeZip(outputPath, folderPath, safeName, files); err != nil {
		internal(err)
		return
	}
	httpx.OK(w, zipResult{OK: true, ZipPath: outputPath, FileName: filepath.Base(outputPath)})
}

// jsBasename is path.basename: trailing separators ignored, "" for a root.
func jsBasename(p string) string {
	seps := "/"
	if filepath.Separator == '\\' {
		seps = `/\`
	}
	trimmed := strings.TrimRight(p, seps)
	if trimmed == "" {
		return ""
	}
	return filepath.Base(trimmed)
}

// collectFiles walks dir skipping hidden entries (.thumbs etc); captionsOnly
// keeps only caption files. Symlinks are neither files nor dirs, so skipped.
func collectFiles(dir string, captionsOnly bool) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		p := filepath.Join(dir, e.Name())
		switch {
		case e.IsDir():
			sub, err := collectFiles(p, captionsOnly)
			if err != nil {
				return nil, err
			}
			out = append(out, sub...)
		case e.Type().IsRegular():
			if !captionsOnly || captionExts[strings.ToLower(filepath.Ext(e.Name()))] {
				out = append(out, p)
			}
		}
	}
	return out, nil
}

func removeIfExists(p string) error {
	err := os.RemoveAll(p)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	return err
}

// writeZip stores files under prefix/<path relative to base>, deflated at
// level 9 like archiver's zlib level 9.
func writeZip(outputPath, base, prefix string, files []string) (err error) {
	out, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := out.Close(); err == nil {
			err = cerr
		}
	}()
	zw := zip.NewWriter(out)
	zw.RegisterCompressor(zip.Deflate, func(w io.Writer) (io.WriteCloser, error) {
		return flate.NewWriter(w, flate.BestCompression)
	})
	for _, f := range files {
		rel, err := filepath.Rel(base, f)
		if err != nil {
			return err
		}
		if err := addZipFile(zw, f, filepath.ToSlash(filepath.Join(prefix, rel))); err != nil {
			return err
		}
	}
	return zw.Close()
}

func addZipFile(zw *zip.Writer, path, name string) error {
	st, err := os.Stat(path)
	if err != nil {
		return err
	}
	hdr, err := zip.FileInfoHeader(st)
	if err != nil {
		return err
	}
	hdr.Name = name
	hdr.Method = zip.Deflate
	w, err := zw.CreateHeader(hdr)
	if err != nil {
		return err
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(w, f)
	return err
}
