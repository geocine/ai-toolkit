package img

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
	"github.com/google/uuid"
)

// UploadHandler handles POST /api/img/upload
func UploadHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "Method not allowed"})
		return
	}

	// Parse multipart form with 50MB max memory
	err := r.ParseMultipartForm(50 << 20)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to parse form"})
		return
	}

	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	// Get data root for images
	dataRoot, err := internal.GetDataRoot(ctx, client)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to get data root"})
		return
	}

	imgRoot := filepath.Join(dataRoot, "images")

	// Create images directory if it doesn't exist
	if err := os.MkdirAll(imgRoot, 0755); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create images directory"})
		return
	}

	files := r.MultipartForm.File["files"]
	if len(files) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "No files provided"})
		return
	}

	savedFiles := make([]string, 0, len(files))

	for _, fileHeader := range files {
		file, err := fileHeader.Open()
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to open uploaded file"})
			return
		}
		defer file.Close()

		// Get file extension
		ext := filepath.Ext(fileHeader.Filename)
		if ext == "" {
			ext = ".jpg"
		}

		// Generate unique filename using UUID
		fileName := uuid.New().String() + ext
		filePath := filepath.Join(imgRoot, fileName)

		// Create destination file
		dst, err := os.Create(filePath)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create file"})
			return
		}
		defer dst.Close()

		// Copy file contents
		if _, err := io.Copy(dst, file); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to save file"})
			return
		}

		savedFiles = append(savedFiles, filePath)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Files uploaded successfully",
		"files":   savedFiles,
	})
}
