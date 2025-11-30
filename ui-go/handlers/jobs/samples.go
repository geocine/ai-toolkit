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

func SamplesHandler(w http.ResponseWriter, r *http.Request) {
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
	samplesFolder := filepath.Join(trainingRoot, job.Name, "samples")
	samples := []string{}
	if stat, err := os.Stat(samplesFolder); err == nil && stat.IsDir() {
		entries, _ := os.ReadDir(samplesFolder)
		for _, entry := range entries {
			if !entry.IsDir() {
				ext := filepath.Ext(entry.Name())
				if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" {
					samples = append(samples, filepath.Join(samplesFolder, entry.Name()))
				}
			}
		}
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"samples": samples})
}
