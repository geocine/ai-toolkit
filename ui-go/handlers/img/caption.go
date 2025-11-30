package img

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func CaptionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		ImgPath string `json:"imgPath"`
		Caption string `json:"caption"`
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
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Image does not exist"})
		return
	}
	captionPath := body.ImgPath[:len(body.ImgPath)-len(filepath.Ext(body.ImgPath))] + ".txt"
	if err := ioutil.WriteFile(captionPath, []byte(body.Caption), 0644); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to write caption"})
		return
	}
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}
