package img

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func ServeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	imagePath := strings.TrimPrefix(r.URL.Path, "/api/img/")
	if imagePath == "" {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	decodedPath, err := url.PathUnescape(imagePath)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid image path"})
		return
	}
	ctx := context.Background()
	client := db.NewClient()
	defer client.Prisma.Disconnect()
	datasetRoot, err := internal.GetDatasetsRoot(ctx, client)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to get datasets root"})
		return
	}
	trainingRoot, err := internal.GetTrainingFolder(ctx, client)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to get training root"})
		return
	}
	allowedDirs := []string{datasetRoot, trainingRoot}
	isAllowed := false
	for _, dir := range allowedDirs {
		absDir, _ := filepath.Abs(dir)
		absFile, _ := filepath.Abs(decodedPath)
		if strings.HasPrefix(absFile, absDir) && !strings.Contains(decodedPath, "..") {
			isAllowed = true
			break
		}
	}
	if !isAllowed {
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]string{"error": "Access denied"})
		return
	}
	stat, err := os.Stat(decodedPath)
	if os.IsNotExist(err) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "File not found"})
		return
	}
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Internal server error"})
		return
	}
	if !stat.Mode().IsRegular() {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Not a file"})
		return
	}
	ext := strings.ToLower(filepath.Ext(decodedPath))
	contentTypeMap := map[string]string{
		".jpg":  "image/jpeg",
		".jpeg": "image/jpeg",
		".png":  "image/png",
		".gif":  "image/gif",
		".webp": "image/webp",
		".svg":  "image/svg+xml",
		".bmp":  "image/bmp",
		".mp4":  "video/mp4",
		".avi":  "video/x-msvideo",
		".mov":  "video/quicktime",
		".mkv":  "video/x-matroska",
		".wmv":  "video/x-ms-wmv",
		".m4v":  "video/x-m4v",
		".flv":  "video/x-flv",
	}
	contentType := contentTypeMap[ext]
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	b, err := os.ReadFile(decodedPath)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to read file"})
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", strconv.Itoa(len(b)))
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.Write(b)
}
