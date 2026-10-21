package jobfiles

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/httpx"
)

type pluginStatus struct {
	Exists bool    `json:"exists"`
	HTML   *string `json:"html"`
}

// plugin serves the job's plugin.html raw, so it can be an iframe src.
// ?check=1 only reports whether it exists (to decide if the Plugin tab
// shows). Mirrors jobs/[jobID]/plugin/route.ts.
func plugin(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}
	pluginPath := filepath.Join(folder, "plugin.html")
	if _, err := os.Stat(pluginPath); err != nil {
		httpx.OK(w, pluginStatus{Exists: false})
		return
	}
	if r.URL.Query().Get("check") == "1" {
		httpx.OK(w, pluginStatus{Exists: true})
		return
	}
	b, err := os.ReadFile(pluginPath)
	if err != nil {
		log.Printf("Error reading plugin file: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Error reading plugin file")
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(decodeUTF8(b)))
}
