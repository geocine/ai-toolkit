package jobfiles

import (
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/httpx"
)

type fileEntry struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// files lists the job folder's .safetensors files (sorted) plus optimizer.pt
// when present. Mirrors jobs/[jobID]/files/route.ts.
func files(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}
	if _, err := os.Stat(folder); err != nil {
		httpx.OK(w, map[string]any{"files": []fileEntry{}})
		return
	}
	entries, err := os.ReadDir(folder)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	var paths []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") {
			paths = append(paths, filepath.Join(folder, e.Name()))
		}
	}
	slices.SortFunc(paths, func(a, b string) int { return cmpJS(a, b) })

	out := make([]fileEntry, 0, len(paths)+1)
	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil {
			httpx.Internal(w, r, err)
			return
		}
		out = append(out, fileEntry{Path: decodeUTF8([]byte(p)), Size: st.Size()})
	}
	// include the optimizer state if it exists
	optimizer := filepath.Join(folder, "optimizer.pt")
	if st, err := os.Stat(optimizer); err == nil {
		out = append(out, fileEntry{Path: decodeUTF8([]byte(optimizer)), Size: st.Size()})
	}
	httpx.OK(w, map[string]any{"files": out})
}

func cmpJS(a, b string) int {
	switch {
	case jsLess(a, b):
		return -1
	case jsLess(b, a):
		return 1
	}
	return 0
}
