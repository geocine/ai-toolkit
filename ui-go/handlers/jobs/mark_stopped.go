package jobs

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/geocine/aitoolkit/prisma/db"
)

// MarkStoppedHandler handles GET /api/jobs/:id/mark_stopped
func MarkStoppedHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Extract jobID from path: /api/jobs/{jobID}/mark_stopped
	path := r.URL.Path
	parts := strings.Split(strings.TrimPrefix(path, "/api/jobs/"), "/")
	if len(parts) < 1 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Job ID required"})
		return
	}
	jobID := parts[0]

	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	// Find the job
	job, err := client.Job.FindUnique(db.Job.ID.Equals(jobID)).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Job not found"})
		return
	}

	// Update job status to stopped
	updatedJob, err := client.Job.FindUnique(db.Job.ID.Equals(jobID)).Update(
		db.Job.Stop.Set(true),
		db.Job.Status.Set("stopped"),
		db.Job.Info.Set("Job stopped"),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update job"})
		return
	}

	// Return the original job (matching Next.js behavior)
	_ = updatedJob
	json.NewEncoder(w).Encode(job)
}
