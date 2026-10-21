package v1

import (
	"context"
	"net/http"

	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

type queueView struct {
	GpuIDs       string  `json:"gpu_ids"`
	Running      bool    `json:"running"`
	CurrentJobID *string `json:"current_job_id"`
	Queued       int     `json:"queued"`
}

func queueOf(ctx context.Context, q *store.Queue) (queueView, error) {
	v := queueView{GpuIDs: q.GpuIDs, Running: q.IsRunning}
	cur, err := store.FindFirstJob(ctx, store.JobQuery{
		Where: `"gpu_ids" = ? AND "status" IN ('running', 'stopping')`, Args: []any{q.GpuIDs},
	})
	if err != nil {
		return v, err
	}
	if cur != nil {
		v.CurrentJobID = &cur.ID
	}
	queued, err := store.FindJobs(ctx, store.JobQuery{Where: `"gpu_ids" = ? AND "status" = 'queued'`, Args: []any{q.GpuIDs}})
	v.Queued = len(queued)
	return v, err
}

func listQueues(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	qs, err := store.ListQueues(ctx, `"gpu_ids" ASC`)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	items := make([]queueView, 0, len(qs))
	for _, q := range qs {
		v, err := queueOf(ctx, q)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		items = append(items, v)
	}
	v1api.Items(w, items)
}

// queueAction starts (creating it if needed) or stops a GPU queue.
func queueAction(run bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		gpu := r.PathValue("gpu_ids")
		q, err := store.GetQueueByGPU(ctx, gpu)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		switch {
		case q == nil && !run:
			v1api.Fail(w, v1api.CodeNotFound, "no queue for gpu_ids "+gpu)
			return
		case q == nil:
			if q, err = store.CreateQueue(ctx, gpu, true); err != nil {
				v1api.Internal(w, r, err)
				return
			}
		case q.IsRunning != run:
			if err := store.SetQueueRunning(ctx, q.ID, run); err != nil {
				v1api.Internal(w, r, err)
				return
			}
			q.IsRunning = run
		}
		v, err := queueOf(ctx, q)
		if err != nil {
			v1api.Internal(w, r, err)
			return
		}
		v1api.OK(w, v)
	}
}
