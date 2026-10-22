package v1

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"io/fs"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/geocine/aitoolkit/internal/collation"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobview"
	"github.com/geocine/aitoolkit/internal/safepath"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/thumbs"
	"github.com/geocine/aitoolkit/internal/v1api"
)

var (
	imageExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".bmp": true,
		".gif": true, ".tif": true, ".tiff": true}
	mediaExts = map[string]bool{".mp4": true, ".mov": true, ".webm": true, ".mkv": true, ".avi": true,
		".m4v": true, ".mp3": true, ".wav": true, ".flac": true, ".ogg": true}
)

func isMedia(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	return imageExts[ext] || mediaExts[ext]
}

// captionPath is <stem>.txt next to a media file (upstream's caption_ext).
func captionPath(file string) string {
	return strings.TrimSuffix(file, filepath.Ext(file)) + ".txt"
}

type datasetView struct {
	Name     string `json:"name"`
	Path     string `json:"path"`
	Images   int    `json:"images"`
	Captions int    `json:"captions"`
	Size     int64  `json:"size"`
	Modified string `json:"modified"`
}

type datasetItem struct {
	File    string  `json:"file"`
	Caption *string `json:"caption"`
	Size    int64   `json:"size"`
	Width   *int    `json:"width,omitempty"`
	Height  *int    `json:"height,omitempty"`
}

// validDatasetName keeps names to a single safe folder.
func validDatasetName(name string) error {
	switch {
	case strings.TrimSpace(name) == "" || strings.TrimSpace(name) != name:
		return errors.New("dataset name may not be empty or padded with spaces")
	case len(name) > 128:
		return errors.New("dataset name is longer than 128 characters")
	case strings.HasPrefix(name, "."):
		return errors.New("dataset name may not start with a dot")
	case strings.ContainsAny(name, `/\:*?"<>|`):
		return errors.New(`dataset name may not contain / \ : * ? " < > |`)
	}
	for _, r := range name {
		if r < 0x20 {
			return errors.New("dataset name may not contain control characters")
		}
	}
	return nil
}

// datasetDir resolves the path's dataset; exists says whether it must exist.
func datasetDir(w http.ResponseWriter, r *http.Request, name string, exists bool) (string, bool) {
	if err := validDatasetName(name); err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return "", false
	}
	root, err := settings.DatasetsRoot(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return "", false
	}
	dir := filepath.Join(root, name)
	if exists {
		if st, err := os.Stat(dir); err != nil || !st.IsDir() {
			v1api.Fail(w, v1api.CodeNotFound, "dataset "+name+" not found")
			return "", false
		}
	}
	return dir, true
}

// walkDataset visits every visible file (hidden files and folders such as
// .thumbs are skipped), with paths relative to dir using "/".
func walkDataset(dir string, fn func(rel string, info fs.FileInfo)) error {
	return filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p == dir {
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		fn(filepath.ToSlash(rel), info)
		return nil
	})
}

func summarize(name, dir string) (datasetView, error) {
	v := datasetView{Name: name, Path: dir}
	st, err := os.Stat(dir)
	if err != nil {
		return v, err
	}
	newest := st.ModTime()
	err = walkDataset(dir, func(rel string, info fs.FileInfo) {
		v.Size += info.Size()
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		if isMedia(rel) {
			v.Images++
		} else if strings.EqualFold(filepath.Ext(rel), ".txt") {
			v.Captions++
		}
	})
	v.Modified = v1api.Time(newest)
	return v, err
}

func listDatasets(w http.ResponseWriter, r *http.Request) {
	root, err := settings.DatasetsRoot(r.Context())
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	entries, err := os.ReadDir(root)
	if err != nil && !os.IsNotExist(err) {
		v1api.Internal(w, r, err)
		return
	}
	items := []datasetView{}
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		v, err := summarize(e.Name(), filepath.Join(root, e.Name()))
		if err != nil {
			continue
		}
		items = append(items, v)
	}
	sort.SliceStable(items, func(i, j int) bool { return collation.Compare(items[i].Name, items[j].Name) < 0 })
	v1api.Items(w, items)
}

func changed(ctx context.Context, name string) {
	events.Emit(ctx, events.DatasetChange, "", map[string]string{"name": name})
}

func createDataset(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Name string `json:"name"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	dir, ok := datasetDir(w, r, b.Name, false)
	if !ok {
		return
	}
	if _, err := os.Stat(dir); err == nil {
		v1api.Fail(w, v1api.CodeConflict, "dataset "+b.Name+" already exists")
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v, err := summarize(b.Name, dir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	changed(r.Context(), b.Name)
	v1api.Created(w, v)
}

func readCaption(file string) *string {
	b, err := os.ReadFile(captionPath(file))
	if err != nil {
		return nil
	}
	s := string(b)
	return &s
}

func itemOf(dir, rel string, info fs.FileInfo) datasetItem {
	full := filepath.Join(dir, filepath.FromSlash(rel))
	it := datasetItem{File: rel, Size: info.Size(), Caption: readCaption(full)}
	if imageExts[strings.ToLower(filepath.Ext(rel))] {
		if f, err := os.Open(full); err == nil {
			if cfg, _, err := image.DecodeConfig(f); err == nil {
				it.Width, it.Height = &cfg.Width, &cfg.Height
			}
			f.Close()
		}
	}
	return it
}

func getDataset(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	q := r.URL.Query()
	start, limit := 0, 500
	if s := q.Get("cursor"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 {
			v1api.Fail(w, v1api.CodeInvalidRequest, "invalid cursor")
			return
		}
		start = n
	}
	if s := q.Get("limit"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n <= 0 {
			v1api.Fail(w, v1api.CodeInvalidRequest, "limit must be a positive integer")
			return
		}
		limit = min(n, 5000)
	}
	type entry struct {
		rel  string
		info fs.FileInfo
	}
	var media []entry
	if err := walkDataset(dir, func(rel string, info fs.FileInfo) {
		if isMedia(rel) {
			media = append(media, entry{rel, info})
		}
	}); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	sort.SliceStable(media, func(i, j int) bool { return collation.Compare(media[i].rel, media[j].rel) < 0 })
	v, err := summarize(name, dir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	items := []datasetItem{}
	end := min(start+limit, len(media))
	for i := start; i < end; i++ {
		items = append(items, itemOf(dir, media[i].rel, media[i].info))
	}
	var next any
	if end < len(media) {
		next = strconv.Itoa(end)
	}
	cj, err := latestCaptionJob(r.Context(), dir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, struct {
		datasetView
		Items      []datasetItem `json:"items"`
		NextCursor any           `json:"next_cursor"`
		CaptionJob *jobview.Job  `json:"caption_job"`
	}{v, items, next, cj})
}

func deleteDataset(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	users, err := jobsUsingDataset(r.Context(), dir)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if len(users) > 0 {
		v1api.Fail(w, v1api.CodeConflict, "the dataset is in use by "+strings.Join(users, ", ")+"; stop it first")
		return
	}
	if err := os.RemoveAll(dir); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	changed(r.Context(), name)
	v1api.NoContent(w)
}

// jobsUsingDataset names the queued, running or stopping jobs that read dir:
// training runs through datasets[].folder_path, caption jobs through their ref.
func jobsUsingDataset(ctx context.Context, dir string) ([]string, error) {
	jobs, err := store.FindJobs(ctx, store.JobQuery{Where: `"status" IN ('queued', 'running', 'stopping')`})
	if err != nil {
		return nil, err
	}
	clean := filepath.Clean(dir)
	var out []string
	for _, j := range jobs {
		if j.JobRef != nil && filepath.Clean(*j.JobRef) == clean {
			out = append(out, j.Name)
			continue
		}
		var cfg struct {
			Config struct {
				Process []struct {
					Datasets []struct {
						FolderPath string `json:"folder_path"`
					} `json:"datasets"`
				} `json:"process"`
			} `json:"config"`
		}
		if json.Unmarshal([]byte(j.JobConfig), &cfg) != nil {
			continue
		}
	process:
		for _, p := range cfg.Config.Process {
			for _, d := range p.Datasets {
				if d.FolderPath != "" && filepath.Clean(d.FolderPath) == clean {
					out = append(out, j.Name)
					break process
				}
			}
		}
	}
	return out, nil
}

// datasetFile resolves {file...} inside the dataset, refusing traversal.
func datasetFile(w http.ResponseWriter, dir, rel string) (string, bool) {
	clean := path.Clean("/" + strings.ReplaceAll(rel, `\`, "/"))
	if clean == "/" || rel == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "file is required")
		return "", false
	}
	p := safepath.Resolve(filepath.Join(dir, filepath.FromSlash(clean[1:])), []string{dir})
	if p == "" || p == dir {
		v1api.Fail(w, v1api.CodeInvalidRequest, "file is outside the dataset")
		return "", false
	}
	return p, true
}

func putCaption(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	file, ok := datasetFile(w, dir, r.PathValue("file"))
	if !ok {
		return
	}
	var b struct {
		Caption *string `json:"caption"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	if b.Caption == nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "caption is required")
		return
	}
	info, err := os.Stat(file)
	if err != nil || info.IsDir() || !isMedia(file) {
		v1api.Fail(w, v1api.CodeNotFound, "no media file "+r.PathValue("file")+" in dataset "+name)
		return
	}
	if err := writeAtomic(captionPath(file), strings.NewReader(*b.Caption)); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	rel, _ := filepath.Rel(dir, file)
	changed(r.Context(), name)
	v1api.OK(w, itemOf(dir, filepath.ToSlash(rel), info))
}

func deleteDatasetFile(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	file, ok := datasetFile(w, dir, r.PathValue("file"))
	if !ok {
		return
	}
	info, err := os.Stat(file)
	if err != nil || info.IsDir() {
		v1api.Fail(w, v1api.CodeNotFound, "no file "+r.PathValue("file")+" in dataset "+name)
		return
	}
	if err := os.Remove(file); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if isMedia(file) {
		os.Remove(captionPath(file))
		base := thumbs.Base(file)
		os.Remove(base + ".png")
		os.Remove(base + ".jpg")
	}
	changed(r.Context(), name)
	v1api.NoContent(w)
}

// writeAtomic writes src to dst through a temp file in dst's folder.
func writeAtomic(dst string, src io.Reader) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".upload-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := io.Copy(tmp, src); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), dst)
}

// uploadName is the base name of an uploaded file, or "" when unusable.
func uploadName(raw string) string {
	name := path.Base(strings.ReplaceAll(raw, `\`, "/"))
	if name == "." || name == "/" || name == ".." || strings.HasPrefix(name, ".") || validDatasetName(name) != nil {
		return ""
	}
	return name
}

func uploadDatasetFiles(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	mr, err := r.MultipartReader()
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "expected multipart/form-data with files[]")
		return
	}
	saved := []string{}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			v1api.Fail(w, v1api.CodeInvalidRequest, "bad multipart body: "+err.Error())
			return
		}
		field := part.FormName()
		if part.FileName() == "" || (field != "files" && field != "files[]" && field != "file") {
			part.Close()
			continue
		}
		fname := uploadName(part.FileName())
		if fname == "" {
			part.Close()
			v1api.Fail(w, v1api.CodeInvalidRequest, "unusable file name "+strconv.Quote(part.FileName()))
			return
		}
		if err := writeAtomic(filepath.Join(dir, fname), part); err != nil {
			part.Close()
			v1api.Internal(w, r, err)
			return
		}
		part.Close()
		saved = append(saved, fname)
	}
	if len(saved) == 0 {
		v1api.Fail(w, v1api.CodeInvalidRequest, "no files in the request (field files[])")
		return
	}
	changed(r.Context(), name)
	v1api.Items(w, saved)
}

// Archive limits.
const (
	maxArchiveFiles = 20000
	maxArchiveBytes = 20 << 30
)

// archiveTemp stores the request's zip (raw body or multipart field "file")
// in a temp file, since zip needs random access.
func archiveTemp(r *http.Request, dir string) (string, error) {
	tmp, err := os.CreateTemp(dir, ".archive-*.zip")
	if err != nil {
		return "", err
	}
	defer tmp.Close()
	var src io.Reader = r.Body
	if mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type")); mt == "multipart/form-data" {
		mr, err := r.MultipartReader()
		if err != nil {
			return tmp.Name(), err
		}
		for {
			part, err := mr.NextPart()
			if err != nil {
				return tmp.Name(), errors.New(`multipart body has no "file" part`)
			}
			if part.FormName() == "file" {
				src = part
				break
			}
			part.Close()
		}
	}
	_, err = io.Copy(tmp, io.LimitReader(src, maxArchiveBytes+1))
	return tmp.Name(), err
}

type archiveEntry struct {
	f   *zip.File
	rel string
}

// planArchive validates every entry before anything is written: a traversal,
// absolute path or symlink rejects the whole archive. Junk (macOS metadata,
// hidden files) is skipped, and one shared top-level folder is stripped.
func planArchive(zr *zip.Reader) ([]archiveEntry, []string, error) {
	var entries []archiveEntry
	var skipped []string
	var total uint64
	for _, f := range zr.File {
		name := strings.ReplaceAll(f.Name, `\`, "/")
		if f.Mode()&fs.ModeSymlink != 0 {
			return nil, nil, v1api.Errorf(v1api.CodeInvalidRequest, "archive contains a symlink: "+f.Name)
		}
		if strings.HasPrefix(name, "/") || (len(name) > 1 && name[1] == ':') {
			return nil, nil, v1api.Errorf(v1api.CodeInvalidRequest, "archive contains an absolute path: "+f.Name)
		}
		for _, seg := range strings.Split(name, "/") {
			if seg == ".." {
				return nil, nil, v1api.Errorf(v1api.CodeInvalidRequest, "archive contains a path leaving the dataset: "+f.Name)
			}
		}
		if f.FileInfo().IsDir() || strings.HasSuffix(name, "/") {
			continue
		}
		clean := path.Clean(name)
		hidden := strings.HasPrefix(clean, "__MACOSX/")
		for _, seg := range strings.Split(clean, "/") {
			if strings.HasPrefix(seg, ".") {
				hidden = true
			}
		}
		if hidden {
			skipped = append(skipped, f.Name)
			continue
		}
		total += f.UncompressedSize64
		entries = append(entries, archiveEntry{f, clean})
	}
	if len(entries) > maxArchiveFiles {
		return nil, nil, v1api.Errorf(v1api.CodeTooLarge, fmt.Sprintf("archive has %d files (max %d)", len(entries), maxArchiveFiles))
	}
	if total > maxArchiveBytes {
		return nil, nil, v1api.Errorf(v1api.CodeTooLarge, "archive unpacks to more than 20 GiB")
	}
	// strip a single folder every file lives under (set.zip -> set/...)
	if len(entries) > 0 {
		top, _, nested := strings.Cut(entries[0].rel, "/")
		shared := nested
		for _, e := range entries {
			t, _, ok := strings.Cut(e.rel, "/")
			if !ok || t != top {
				shared = false
				break
			}
		}
		if shared {
			for i := range entries {
				entries[i].rel = strings.TrimPrefix(entries[i].rel, top+"/")
			}
		}
	}
	return entries, skipped, nil
}

// limitedReader fails instead of truncating when an entry lies about its size.
type limitedReader struct {
	r    io.Reader
	left int64
}

var errEntryTooLarge = errors.New("archive entry is larger than it declares")

func (l *limitedReader) Read(p []byte) (int, error) {
	if l.left <= 0 {
		// the declared size is used up: anything more is a lie
		var one [1]byte
		if n, err := l.r.Read(one[:]); n > 0 {
			return 0, errEntryTooLarge
		} else if err != nil {
			return 0, err
		}
		return 0, io.EOF
	}
	if int64(len(p)) > l.left {
		p = p[:l.left]
	}
	n, err := l.r.Read(p)
	l.left -= int64(n)
	return n, err
}

func uploadDatasetArchive(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	tmp, err := archiveTemp(r, dir)
	if tmp != "" {
		defer os.Remove(tmp)
	}
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "could not read the archive: "+err.Error())
		return
	}
	res, err := extractArchive(tmp, dir)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	changed(r.Context(), name)
	v1api.OK(w, map[string]any{"added": res.Added, "skipped": res.Skipped})
}

type archiveResult struct {
	Added   int      `json:"added"`
	Files   []string `json:"files"`
	Skipped []string `json:"skipped"`
}

// extractArchive safely unpacks the zip at tmp into the dataset folder dir.
func extractArchive(tmp, dir string) (archiveResult, error) {
	res := archiveResult{Files: []string{}, Skipped: []string{}}
	if st, err := os.Stat(tmp); err == nil && st.Size() > maxArchiveBytes {
		return res, v1api.Errorf(v1api.CodeTooLarge, "archive is larger than 20 GiB")
	}
	zr, err := zip.OpenReader(tmp)
	if err != nil {
		return res, v1api.Errorf(v1api.CodeInvalidRequest, "not a zip archive: "+err.Error())
	}
	defer zr.Close()
	entries, skipped, err := planArchive(&zr.Reader)
	if err != nil {
		return res, err
	}
	res.Skipped = append(res.Skipped, skipped...)
	for _, e := range entries {
		dst := safepath.Resolve(filepath.Join(dir, filepath.FromSlash(e.rel)), []string{dir})
		if dst == "" || dst == dir {
			res.Skipped = append(res.Skipped, e.f.Name)
			continue
		}
		rc, err := e.f.Open()
		if err != nil {
			return res, v1api.Errorf(v1api.CodeInvalidRequest, "corrupt archive entry "+e.f.Name+": "+err.Error())
		}
		err = writeAtomic(dst, &limitedReader{r: rc, left: int64(e.f.UncompressedSize64)})
		rc.Close()
		if err != nil {
			return res, v1api.Errorf(v1api.CodeInvalidRequest, "extracting "+e.f.Name+": "+err.Error())
		}
		if !e.f.Modified.IsZero() {
			os.Chtimes(dst, time.Now(), e.f.Modified)
		}
		res.Added++
		res.Files = append(res.Files, e.rel)
	}
	return res, nil
}

// putDatasetFile writes one file (raw body) into the dataset, replacing it.
func putDatasetFile(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	file, ok := datasetFile(w, dir, r.PathValue("file"))
	if !ok {
		return
	}
	rel, _ := filepath.Rel(dir, file)
	for _, seg := range strings.Split(filepath.ToSlash(rel), "/") {
		if strings.HasPrefix(seg, ".") {
			v1api.Fail(w, v1api.CodeInvalidRequest, "hidden file names are reserved")
			return
		}
	}
	if st, err := os.Stat(file); err == nil && st.IsDir() {
		v1api.Fail(w, v1api.CodeConflict, "path is a folder")
		return
	}
	lr := &io.LimitedReader{R: r.Body, N: maxArchiveBytes + 1}
	if err := writeAtomic(file, lr); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if lr.N <= 0 {
		os.Remove(file)
		v1api.Fail(w, v1api.CodeTooLarge, "file is larger than 20 GiB")
		return
	}
	info, err := os.Stat(file)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	changed(r.Context(), name)
	v1api.OK(w, itemOf(dir, filepath.ToSlash(rel), info))
}

var sniffedExts = map[string]string{"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
	"image/gif": ".gif", "image/bmp": ".bmp", "video/mp4": ".mp4", "video/webm": ".webm"}

// importDatasetURL downloads an image (or a zip of images) into the
// dataset: what an agent uses to add pictures it found by URL.
func importDatasetURL(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	name := r.PathValue("name")
	dir, ok := datasetDir(w, r, name, true)
	if !ok {
		return
	}
	var b struct {
		URL     string  `json:"url"`
		File    string  `json:"file"`
		Caption *string `json:"caption"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	u, err := url.Parse(b.URL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "url must be an http(s) URL")
		return
	}
	if b.File != "" && uploadName(b.File) == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "unusable file name "+strconv.Quote(b.File))
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.URL, nil)
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		v1api.Fail(w, v1api.CodeRemoteFailed, "download: "+err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		v1api.Fail(w, v1api.CodeRemoteFailed, "download: "+resp.Status)
		return
	}
	tmp, err := os.CreateTemp(dir, ".download-*")
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	defer os.Remove(tmp.Name())
	n, err := io.Copy(tmp, io.LimitReader(resp.Body, maxArchiveBytes+1))
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		v1api.Fail(w, v1api.CodeRemoteFailed, "download: "+err.Error())
		return
	}
	if n > maxArchiveBytes {
		v1api.Fail(w, v1api.CodeTooLarge, "download is larger than 20 GiB")
		return
	}
	head := make([]byte, 512)
	if f, err := os.Open(tmp.Name()); err == nil {
		k, _ := io.ReadFull(f, head)
		head = head[:k]
		f.Close()
	}
	if bytes.HasPrefix(head, []byte("PK\x03\x04")) {
		res, err := extractArchive(tmp.Name(), dir)
		if err != nil {
			v1api.Respond(w, r, err)
			return
		}
		changed(ctx, name)
		v1api.OK(w, res)
		return
	}
	fname := ""
	if b.File != "" {
		// a file name, never a path: the dataset folder is the only place it may land
		if fname = uploadName(b.File); fname == "" || fname != b.File {
			v1api.Fail(w, v1api.CodeInvalidRequest, "file must be a plain file name (no folders, not starting with a dot)")
			return
		}
	}
	if fname == "" {
		if _, params, err := mime.ParseMediaType(resp.Header.Get("Content-Disposition")); err == nil {
			fname = uploadName(params["filename"])
		}
	}
	if fname == "" {
		fname = uploadName(path.Base(u.Path))
	}
	if !isMedia(fname) {
		// name it after what the bytes are
		ext := sniffedExts[http.DetectContentType(head)]
		if ext == "" {
			v1api.Fail(w, v1api.CodeInvalidRequest, "the URL is neither a zip nor an image, video or audio file")
			return
		}
		if fname == "" {
			fname = "image_" + strconv.FormatInt(time.Now().UnixMilli(), 10)
		}
		fname = strings.TrimSuffix(fname, filepath.Ext(fname)) + ext
	}
	dst := filepath.Join(dir, fname)
	if err := os.Rename(tmp.Name(), dst); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if b.Caption != nil {
		if err := writeAtomic(captionPath(dst), strings.NewReader(*b.Caption)); err != nil {
			v1api.Internal(w, r, err)
			return
		}
	}
	changed(ctx, name)
	v1api.OK(w, archiveResult{Added: 1, Files: []string{fname}, Skipped: []string{}})
}
