package datasets

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func ListImagesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		DatasetName string `json:"datasetName"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}
	ctx := context.Background()
	client := db.NewClient()
	defer client.Prisma.Disconnect()
	root, _ := internal.GetDatasetsRoot(ctx, client)
	datasetFolder := filepath.Join(root, body.DatasetName)
	if stat, err := os.Stat(datasetFolder); err != nil || !stat.IsDir() {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Folder '" + body.DatasetName + "' not found"})
		return
	}
	var images []map[string]string
	err := filepath.Walk(datasetFolder, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			ext := strings.ToLower(filepath.Ext(path))
			if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".wmv" || ext == ".m4v" || ext == ".flv" {
				images = append(images, map[string]string{"img_path": path})
			}
		}
		return nil
	})
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to process request"})
		return
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"images": images})
}
