package jobs

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func FilesHandler(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	jobID := r.URL.Query().Get("id")
	if jobID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Missing job id"})
		return
	}
	job, err := client.Job.FindUnique(db.Job.ID.Equals(jobID)).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to fetch job"})
		return
	}
	if job == nil {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Job not found"})
		return
	}
	trainingRoot, err := internal.GetTrainingFolder(ctx, client)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to get training root"})
		return
	}
	jobFolder := filepath.Join(trainingRoot, job.Name)
	files := []map[string]interface{}{}
	if stat, err := os.Stat(jobFolder); err == nil && stat.IsDir() {
		entries, _ := os.ReadDir(jobFolder)
		for _, entry := range entries {
			if !entry.IsDir() && filepath.Ext(entry.Name()) == ".safetensors" {
				filePath := filepath.Join(jobFolder, entry.Name())
				info, err := os.Stat(filePath)
				if err == nil {
					files = append(files, map[string]interface{}{
						"path": filePath,
						"size": info.Size(),
					})
				}
			}
		}
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"files": files})
}
