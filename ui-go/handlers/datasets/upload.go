package datasets

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func UploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseMultipartForm(50 << 20); err != nil { // 50MB limit
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid multipart form"})
		return
	}
	datasetName := r.FormValue("datasetName")
	if datasetName == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Missing datasetName"})
		return
	}
	ctx := context.Background()
	client := db.NewClient()
	defer client.Prisma.Disconnect()
	root, err := internal.GetDatasetsRoot(ctx, client)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to get datasets root"})
		return
	}
	uploadDir := filepath.Join(root, datasetName)
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create upload directory"})
		return
	}
	files := r.MultipartForm.File["files"]
	if len(files) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "No files provided"})
		return
	}
	var savedFiles []string
	cleanRe := regexp.MustCompile(`[^a-zA-Z0-9.-]`)
	for _, fileHeader := range files {
		file, err := fileHeader.Open()
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Error opening file"})
			return
		}
		defer file.Close()
		fileName := cleanRe.ReplaceAllString(fileHeader.Filename, "_")
		filePath := filepath.Join(uploadDir, fileName)
		out, err := os.Create(filePath)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Error saving file"})
			return
		}
		if _, err := io.Copy(out, file); err != nil {
			out.Close()
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Error writing file"})
			return
		}
		out.Close()
		savedFiles = append(savedFiles, fileName)
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Files uploaded successfully",
		"files":   savedFiles,
	})
}
