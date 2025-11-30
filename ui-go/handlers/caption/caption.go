package caption

import (
	"context"
	"io/ioutil"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func ServeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	imagePath := strings.TrimPrefix(r.URL.Path, "/api/caption/")
	if imagePath == "" {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	decodedPath, err := url.PathUnescape(imagePath)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	captionPath := decodedPath[:len(decodedPath)-len(filepath.Ext(decodedPath))] + ".txt"
	absCaptionPath, err := filepath.Abs(captionPath)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	defer client.Prisma.Disconnect()
	ctx := context.Background()
	allowedDir, _ := internal.GetDatasetsRoot(ctx, client)
	absAllowedDir, _ := filepath.Abs(allowedDir)
	if !strings.HasPrefix(absCaptionPath, absAllowedDir) || strings.Contains(absCaptionPath, "..") {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte("Access denied"))
		return
	}
	b, err := ioutil.ReadFile(absCaptionPath)
	w.Header().Set("Content-Type", "text/plain")
	if err != nil {
		// If not found, return empty string with 200
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(""))
		return
	}
	w.Write(b)
}
