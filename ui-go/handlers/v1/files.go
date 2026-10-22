package v1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/safepath"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/thumbs"
	"github.com/geocine/aitoolkit/internal/v1api"
)

var fileRoots = []string{"datasets", "training", "data", "models"}

func rootDir(ctx context.Context, root string) (string, error) {
	switch root {
	case "datasets":
		return settings.DatasetsRoot(ctx)
	case "training":
		return settings.TrainingFolder(ctx)
	case "data":
		return settings.DataRoot(ctx)
	case "models":
		p, err := settings.ModelsPath(ctx)
		if err != nil {
			return "", err
		}
		return filepath.Abs(p)
	}
	return "", v1api.Errorf(v1api.CodeInvalidRequest, "root must be one of "+strings.Join(fileRoots, ", "))
}

// resolve maps ?root=&path= to an absolute path inside that root.
func resolve(r *http.Request) (root, dir, full, rel string, err error) {
	q := r.URL.Query()
	root = q.Get("root")
	if dir, err = rootDir(r.Context(), root); err != nil {
		return
	}
	clean := path.Clean("/" + strings.ReplaceAll(q.Get("path"), `\`, "/"))
	rel = strings.TrimPrefix(clean, "/")
	full = safepath.Resolve(filepath.Join(dir, filepath.FromSlash(rel)), []string{dir})
	if full == "" {
		err = v1api.Errorf(v1api.CodeForbidden, "path is outside the "+root+" root")
	}
	return
}

type fileEntry struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Size     int64  `json:"size"`
	Modified string `json:"modified"`
}

func listFiles(w http.ResponseWriter, r *http.Request) {
	root, _, full, rel, err := resolve(r)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	entries, err := os.ReadDir(full)
	if os.IsNotExist(err) {
		if rel == "" { // a root that was never created is just empty
			v1api.OK(w, map[string]any{"root": root, "path": rel, "items": []fileEntry{}})
			return
		}
		v1api.Fail(w, v1api.CodeNotFound, "no such folder")
		return
	}
	if err != nil {
		if st, serr := os.Stat(full); serr == nil && !st.IsDir() {
			v1api.Fail(w, v1api.CodeInvalidRequest, "path is a file; use /v1/files/content")
			return
		}
		v1api.Internal(w, r, err)
		return
	}
	items := []fileEntry{}
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			continue
		}
		t, size := "file", info.Size()
		if e.IsDir() {
			t, size = "dir", 0
		}
		items = append(items, fileEntry{Name: e.Name(), Type: t, Size: size, Modified: v1api.Time(info.ModTime())})
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].Type != items[j].Type {
			return items[i].Type == "dir"
		}
		return items[i].Name < items[j].Name
	})
	v1api.OK(w, map[string]any{"root": root, "path": rel, "items": items})
}

func getFile(w http.ResponseWriter, r *http.Request) {
	_, _, full, _, err := resolve(r)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	st, err := os.Stat(full)
	if err != nil || st.IsDir() {
		v1api.Fail(w, v1api.CodeNotFound, "no such file")
		return
	}
	serve := full
	if r.URL.Query().Has("thumb") {
		base := thumbs.Base(full)
		if p, _ := thumbs.Lookup(base); p != "" {
			serve = p
		} else if p := thumbs.Ensure(full, base, st.ModTime()); p != "" {
			serve = p
		}
		if serve != full {
			if st, err = os.Stat(serve); err != nil {
				v1api.Internal(w, r, err)
				return
			}
		}
	}
	f, err := os.Open(serve)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	defer f.Close()
	w.Header().Set("ETag", fmt.Sprintf(`"%x-%x"`, st.Size(), st.ModTime().UnixNano()))
	w.Header().Set("Cache-Control", "no-cache")
	if r.URL.Query().Get("download") == "1" || r.URL.Query().Get("download") == "true" {
		w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": filepath.Base(serve)}))
	}
	// ServeContent handles Range, If-None-Match and HEAD
	http.ServeContent(w, r, filepath.Base(serve), st.ModTime(), f)
}

func putFile(w http.ResponseWriter, r *http.Request) {
	root, dir, full, rel, err := resolve(r)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	if full == dir || rel == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "path must name a file")
		return
	}
	if st, err := os.Stat(full); err == nil {
		if st.IsDir() {
			v1api.Fail(w, v1api.CodeConflict, "path is a folder")
			return
		}
		if r.URL.Query().Get("overwrite") != "true" {
			v1api.Fail(w, v1api.CodeConflict, "file exists; pass overwrite=true to replace it")
			return
		}
	}
	h := sha256.New()
	counter := &countWriter{}
	if err := writeAtomic(full, io.TeeReader(r.Body, io.MultiWriter(h, counter))); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, map[string]any{"root": root, "path": rel, "size": counter.n, "sha256": hex.EncodeToString(h.Sum(nil))})
}

type countWriter struct{ n int64 }

func (c *countWriter) Write(p []byte) (int, error) {
	c.n += int64(len(p))
	return len(p), nil
}

func deleteFile(w http.ResponseWriter, r *http.Request) {
	_, dir, full, _, err := resolve(r)
	if err != nil {
		v1api.Respond(w, r, err)
		return
	}
	if full == dir {
		v1api.Fail(w, v1api.CodeInvalidRequest, "refusing to delete a root")
		return
	}
	st, err := os.Stat(full)
	if os.IsNotExist(err) {
		v1api.Fail(w, v1api.CodeNotFound, "no such file")
		return
	}
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if st.IsDir() {
		v1api.Fail(w, v1api.CodeInvalidRequest, "path is a folder; only files can be deleted here")
		return
	}
	if err := os.Remove(full); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.NoContent(w)
}
