package queue

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/geocine/aitoolkit/prisma/db"
)

// StopHandler handles GET /api/queue/:id/stop
func StopHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Extract queueID from path: /api/queue/{queueID}/stop
	path := r.URL.Path
	parts := strings.Split(strings.TrimPrefix(path, "/api/queue/"), "/")
	if len(parts) < 1 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Queue ID required"})
		return
	}
	queueID := parts[0]

	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	// Try to find existing queue
	queue, err := client.Queue.FindUnique(db.Queue.GpuIds.Equals(queueID)).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Queue not found"})
		return
	}

	// Update queue to stopped
	updatedQueue, err := client.Queue.FindUnique(db.Queue.ID.Equals(queue.ID)).Update(
		db.Queue.IsRunning.Set(false),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update queue"})
		return
	}

	json.NewEncoder(w).Encode(updatedQueue)
}
