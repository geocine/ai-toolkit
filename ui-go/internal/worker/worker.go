// Package worker runs the job queues. Mirrors ui/cron/worker.ts and
// ui/cron/actions/processQueue.ts: once a second, stop jobs on queues that
// were turned off and start the next queued job on idle running queues.
package worker

import (
	"context"
	"log"
	"time"

	"github.com/geocine/aitoolkit/internal/store"
)

const interval = time.Second

// RelayArg is the hidden subcommand main dispatches to RunRelay (Windows job launch).
const RelayArg = "__aitk_launch_relay"

// StartGuard, when set, may hold a queued job back (portable jobs whose lease
// another server holds). Returning ok=false leaves the job queued with info.
var StartGuard func(ctx context.Context, job *store.Job) (ok bool, info string)

// Run loops until ctx is done. Ticks never overlap: a slow pass delays the next.
func Run(ctx context.Context) {
	log.Printf("Cron worker started with interval: %d ms", interval.Milliseconds())
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := processQueue(ctx); err != nil {
				log.Printf("Error in cron worker loop: %v", err)
			}
		}
	}
}

func processQueue(ctx context.Context) error {
	queues, err := store.ListQueues(ctx, `"id" ASC`)
	if err != nil {
		return err
	}
	for _, q := range queues {
		if !q.IsRunning {
			// stop any running jobs first
			running, err := store.FindJobs(ctx, store.JobQuery{
				Where: `"status" = 'running' AND "gpu_ids" = ?`, Args: []any{q.GpuIDs},
			})
			if err != nil {
				return err
			}
			for _, j := range running {
				log.Printf("Stopping job %s on GPU(s) %s", j.ID, j.GpuIDs)
				if _, err := store.UpdateJob(ctx, j.ID, store.Fields{
					"return_to_queue": true,
					"info":            "Stopping job...",
				}); err != nil {
					return err
				}
			}
			continue
		}

		// first see if one is already running, status of running or stopping
		active, err := store.FindFirstJob(ctx, store.JobQuery{
			Where: `"status" IN ('running', 'stopping') AND "gpu_ids" = ?`, Args: []any{q.GpuIDs},
		})
		if err != nil {
			return err
		}
		if active != nil {
			continue
		}
		next, err := store.FindFirstJob(ctx, store.JobQuery{
			Where: `"status" = 'queued' AND "gpu_ids" = ?`, Args: []any{q.GpuIDs},
			OrderBy: `"queue_position" ASC`,
		})
		if err != nil {
			return err
		}
		if next != nil {
			if StartGuard != nil {
				if ok, info := StartGuard(ctx, next); !ok {
					if next.Info != info {
						store.UpdateJob(ctx, next.ID, store.Fields{"info": info})
					}
					continue
				}
			}
			log.Printf("Starting job %s on GPU(s) %s", next.ID, next.GpuIDs)
			if err := StartJob(ctx, next.ID); err != nil {
				return err
			}
			continue
		}
		log.Printf("No more jobs in queue for GPU(s) %s, stopping queue", q.GpuIDs)
		if err := store.SetQueueRunning(ctx, q.ID, false); err != nil {
			return err
		}
	}
	return nil
}
