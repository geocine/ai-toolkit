package files

import (
	"context"
	"encoding/json"
	"fmt"
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
	filePath := strings.TrimPrefix(r.URL.Path, "/api/files/")
	if filePath == "" {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	decodedFilePath, err := url.PathUnescape(filePath)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid file path"})
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
		absFile, _ := filepath.Abs(decodedFilePath)
		if strings.HasPrefix(absFile, absDir) && !strings.Contains(decodedFilePath, "..") {
			isAllowed = true
			break
		}
	}
	if !isAllowed {
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]string{"error": "Access denied"})
		return
	}
	stat, err := os.Stat(decodedFilePath)
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
	ext := strings.ToLower(filepath.Ext(decodedFilePath))
	contentTypeMap := map[string]string{
		".jpg":         "image/jpeg",
		".jpeg":        "image/jpeg",
		".png":         "image/png",
		".gif":         "image/gif",
		".webp":        "image/webp",
		".svg":         "image/svg+xml",
		".bmp":         "image/bmp",
		".safetensors": "application/octet-stream",
		".mp4":         "video/mp4",
		".avi":         "video/x-msvideo",
		".mov":         "video/quicktime",
		".mkv":         "video/x-matroska",
		".wmv":         "video/x-ms-wmv",
		".m4v":         "video/x-m4v",
		".flv":         "video/x-flv",
	}
	contentType := contentTypeMap[ext]
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	filename := filepath.Base(decodedFilePath)
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Accept-Ranges", "bytes")
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.Header().Set("Content-Disposition", "attachment; filename=\""+url.PathEscape(filename)+"\"")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	b, err := os.ReadFile(decodedFilePath)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to read file"})
		return
	}

	rangeHeader := r.Header.Get("Range")
	if rangeHeader != "" && strings.HasPrefix(rangeHeader, "bytes=") {
		// Parse range header
		parts := strings.Split(strings.TrimPrefix(rangeHeader, "bytes="), "-")
		if len(parts) == 2 {
			start, serr := strconv.ParseInt(parts[0], 10, 64)
			end := stat.Size() - 1
			if parts[1] != "" {
				end, _ = strconv.ParseInt(parts[1], 10, 64)
			}
			if serr == nil && start >= 0 && end >= start && end < stat.Size() {
				chunkSize := end - start + 1
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, stat.Size()))
				w.Header().Set("Content-Length", fmt.Sprintf("%d", chunkSize))
				w.WriteHeader(http.StatusPartialContent)
				w.Write(b[start : end+1])
				return
			}
		}
		// If range is invalid, fall through to full file
	}
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(b)))
	w.Write(b)
}
