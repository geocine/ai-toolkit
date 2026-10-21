package jobfiles

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/httpx"
)

// notesGet returns the job's notes.md ("" when there is none). Mirrors
// jobs/[jobID]/notes/route.ts.
func notesGet(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}
	content := ""
	if b, err := os.ReadFile(filepath.Join(folder, "notes.md")); err == nil {
		content = decodeUTF8(b)
	}
	httpx.OK(w, map[string]string{"content": content})
}

// notesPost saves {content} to notes.md, creating the job folder if the job
// never ran. A non-string content saves an empty file.
func notesPost(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}
	body, err := parseBody(r)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	content := ""
	if m, ok := body.(map[string]any); ok {
		if s, ok := m["content"].(string); ok {
			content = s
		}
	}
	notesPath := filepath.Join(folder, "notes.md")
	if err := os.MkdirAll(filepath.Dir(notesPath), 0o755); err != nil {
		log.Printf("Error saving notes file: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Error saving notes file")
		return
	}
	if err := os.WriteFile(notesPath, []byte(content), 0o644); err != nil {
		log.Printf("Error saving notes file: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Error saving notes file")
		return
	}
	httpx.OK(w, map[string]bool{"success": true})
}
