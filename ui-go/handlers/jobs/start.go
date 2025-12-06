package jobs

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/geocine/aitoolkit/prisma/db"
)

func StartHandler(w http.ResponseWriter, r *http.Request) {
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

	// Get highest queue position and add 1000
	allJobs, err := client.Job.FindMany().OrderBy(db.Job.QueuePosition.Order(db.DESC)).Take(1).Exec(ctx)
	var newQueuePosition int
	if err != nil || len(allJobs) == 0 {
		newQueuePosition = 1000
	} else {
		newQueuePosition = allJobs[0].QueuePosition + 1000
	}

	// Update job with new queue position
	_, err = client.Job.FindUnique(db.Job.ID.Equals(jobID)).Update(
		db.Job.QueuePosition.Set(newQueuePosition),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update queue position"})
		return
	}

	// Make sure the queue exists for this GPU
	queue, _ := client.Queue.FindFirst(db.Queue.GpuIds.Equals(job.GpuIds)).Exec(ctx)
	if queue == nil {
		// Create queue if it doesn't exist
		_, err = client.Queue.CreateOne(
			db.Queue.GpuIds.Set(job.GpuIds),
			db.Queue.IsRunning.Set(false),
		).Exec(ctx)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create queue"})
			return
		}
	}

	// Update job status to queued
	_, err = client.Job.FindUnique(db.Job.ID.Equals(jobID)).Update(
		db.Job.Status.Set("queued"),
		db.Job.Stop.Set(false),
		db.Job.ReturnToQueue.Set(false),
		db.Job.Info.Set("Job queued"),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update job status"})
		return
	}

	json.NewEncoder(w).Encode(job)
}
