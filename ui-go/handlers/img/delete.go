package img

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

func DeleteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		ImgPath string `json:"imgPath"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}
	if body.ImgPath == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "imgPath required"})
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
	allowedDir, _ := filepath.Abs(datasetRoot)
	absImg, _ := filepath.Abs(body.ImgPath)
	if !strings.HasPrefix(absImg, allowedDir) || strings.Contains(body.ImgPath, "..") {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid image path"})
		return
	}
	if _, err := os.Stat(body.ImgPath); os.IsNotExist(err) {
		json.NewEncoder(w).Encode(map[string]bool{"success": true})
		return
	}
	if err := os.Remove(body.ImgPath); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to delete image"})
		return
	}
	captionPath := body.ImgPath[:len(body.ImgPath)-len(filepath.Ext(body.ImgPath))] + ".txt"
	if _, err := os.Stat(captionPath); err == nil {
		if err := os.Remove(captionPath); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to delete caption"})
			return
		}
	}
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}
