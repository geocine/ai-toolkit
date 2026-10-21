// Package queue ports ui/src/app/api/queue/route.ts and queue/[queueID]/{start,stop}.
// A queue is keyed by its gpu_ids string, which is what [queueID] carries.
package queue

import (
	"log"
	"net/http"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/store"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /api/queue", list)
	httpx.Handle(mux, "GET /api/queue/{queueID}/start", start)
	httpx.Handle(mux, "GET /api/queue/{queueID}/stop", stop)
}

func list(w http.ResponseWriter, r *http.Request) {
	queues, err := store.ListQueues(r.Context(), `"gpu_ids" ASC`)
	if err != nil {
		log.Printf("GET /api/queue: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to fetch queue")
		return
	}
	httpx.OK(w, map[string]any{"queues": queues})
}

// start turns the queue on, creating it if needed. Like upstream it returns
// the new queue, or an existing one as it was before the update.
func start(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	gpuIDs := r.PathValue("queueID")
	q, err := store.GetQueueByGPU(ctx, gpuIDs)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if q == nil {
		created, err := store.CreateQueue(ctx, gpuIDs, true)
		if err != nil {
			httpx.Internal(w, r, err)
			return
		}
		httpx.OK(w, created)
		return
	}
	if err := store.SetQueueRunning(ctx, q.ID, true); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	httpx.OK(w, q)
}

func stop(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	q, err := store.GetQueueByGPU(ctx, r.PathValue("queueID"))
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if q == nil {
		httpx.Error(w, http.StatusNotFound, "Queue not found")
		return
	}
	if err := store.SetQueueRunning(ctx, q.ID, false); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	httpx.OK(w, q)
}
