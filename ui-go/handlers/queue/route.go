package queue

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/geocine/aitoolkit/prisma/db"
)

// GetQueuesHandler handles GET /api/queue
func GetQueuesHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	queues, err := client.Queue.FindMany().OrderBy(db.Queue.GpuIds.Order(db.ASC)).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to fetch queues"})
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{"queues": queues})
}
