package media

import (
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/geocine/aitoolkit/internal/httpx"
)

// readBody is request.json(): a JSON value of any type. obj is nil when the
// body is not an object; isNull reports a literal null (destructuring that
// throws in upstream, unlike other non-objects whose properties are undefined).
func readBody(r *http.Request) (obj map[string]any, isNull bool, err error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, false, err
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, false, err
	}
	if v == nil {
		return nil, true, nil
	}
	obj, _ = v.(map[string]any)
	if obj == nil {
		obj = map[string]any{}
	}
	return obj, false, nil
}

func textPlain(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Content-Type", "text/plain;charset=UTF-8")
	plain(w, status, body)
}

// underRoot is upstream's isUnderRoot: path.resolve then a prefix check.
func underRoot(p, root string) bool {
	resolved, err := filepath.Abs(p)
	if err != nil {
		return false
	}
	return resolved == root || strings.HasPrefix(resolved, root+string(filepath.Separator))
}

// imgCaption is POST /api/img/caption: write the caption file next to an image.
func imgCaption(w http.ResponseWriter, r *http.Request) {
	fail := func(err error) {
		log.Printf("POST /api/img/caption: %v", err)
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
	// make sure the dataset path is in the image path
	if !underRoot(imgPath, rt.Datasets) {
		httpx.Error(w, http.StatusBadRequest, "Invalid image path")
		return
	}
	if _, err := os.Stat(imgPath); err != nil {
		httpx.Error(w, http.StatusNotFound, "Image does not exist")
		return
	}
	ext, err := captionExt(body["ext"])
	if err != nil {
		fail(err)
		return
	}
	caption, ok := str(body, "caption")
	if !ok {
		fail(errors.New(`the "data" argument must be a string`))
		return
	}
	if err := os.WriteFile(stripExt(imgPath)+"."+ext, []byte(caption), 0o666); err != nil {
		fail(err)
		return
	}
	httpx.OK(w, map[string]bool{"success": true})
}

// captionGet is POST /api/caption/get: the caption text for one image ("" when none).
func captionGet(w http.ResponseWriter, r *http.Request) {
	body, isNull, err := readBody(r)
	if err != nil {
		// client aborted before the body was fully sent
		w.WriteHeader(499)
		return
	}
	if isNull {
		httpx.Internal(w, r, errors.New("cannot destructure null body"))
		return
	}
	fail := func(err error) {
		log.Printf("Error getting caption: %v", err)
		textPlain(w, http.StatusInternalServerError, "Error getting caption")
	}
	imgPath, ok := str(body, "imgPath")
	if !ok {
		fail(errors.New("imgPath must be a string"))
		return
	}
	ext, err := captionExt(body["ext"])
	if err != nil {
		fail(err)
		return
	}
	captionPath := stripExt(imgPath) + "." + ext
	root, err := datasetsRoot(r)
	if err != nil {
		fail(err)
		return
	}
	if !underRoot(imgPath, root) {
		log.Printf("Access denied: %s not in %s", imgPath, root)
		textPlain(w, http.StatusForbidden, "Access denied")
		return
	}
	data, err := os.ReadFile(captionPath)
	if errors.Is(err, fs.ErrNotExist) {
		textPlain(w, http.StatusOK, "")
		return
	}
	if err != nil {
		fail(err)
		return
	}
	textPlain(w, http.StatusOK, string(data))
}

// captionGetBatch is POST /api/caption/getBatch: captions for many images, keyed by path.
func captionGetBatch(w http.ResponseWriter, r *http.Request) {
	body, isNull, err := readBody(r)
	if err != nil {
		w.WriteHeader(499)
		return
	}
	if isNull {
		httpx.Internal(w, r, errors.New("cannot destructure null body"))
		return
	}
	imgPaths, ok := body["imgPaths"].([]any)
	if !ok {
		httpx.Error(w, http.StatusBadRequest, "imgPaths must be an array")
		return
	}
	ext, err := captionExt(body["ext"])
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	root, err := datasetsRoot(r)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	captions := map[string]string{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, 32)
	for _, v := range imgPaths {
		imgPath, ok := v.(string)
		if !ok || !underRoot(imgPath, root) {
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			// a missing file or any read error is an empty caption
			data, _ := os.ReadFile(stripExt(imgPath) + "." + ext)
			mu.Lock()
			captions[imgPath] = string(data)
			mu.Unlock()
		}()
	}
	wg.Wait()
	httpx.OK(w, map[string]any{"captions": captions})
}
