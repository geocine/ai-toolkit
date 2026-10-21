package store

import (
	"context"
	"database/sql"
)

// Queue mirrors model Queue in ui/prisma/schema.prisma.
type Queue struct {
	ID        int    `json:"id"`
	GpuIDs    string `json:"gpu_ids"`
	IsRunning bool   `json:"is_running"`
}

var queueColumns = []string{"id", "gpu_ids", "is_running"}

var queueSelect = `SELECT ` + quoteCols(queueColumns) + ` FROM "Queue"`

func scanQueues(rows *sql.Rows) ([]*Queue, error) {
	defer rows.Close()
	out := []*Queue{}
	for rows.Next() {
		var q Queue
		if err := rows.Scan(&q.ID, &q.GpuIDs, &q.IsRunning); err != nil {
			return nil, err
		}
		out = append(out, &q)
	}
	return out, rows.Err()
}

// ListQueues returns every queue ordered by orderBy (e.g. `"id" ASC`).
func ListQueues(ctx context.Context, orderBy string) ([]*Queue, error) {
	rows, err := DB.QueryContext(ctx, queueSelect+" ORDER BY "+orderBy)
	if err != nil {
		return nil, err
	}
	return scanQueues(rows)
}

// GetQueueByGPU returns the queue for gpuIDs, or nil.
func GetQueueByGPU(ctx context.Context, gpuIDs string) (*Queue, error) {
	var q Queue
	err := DB.QueryRowContext(ctx, queueSelect+` WHERE "gpu_ids" = ? LIMIT 1`, gpuIDs).Scan(&q.ID, &q.GpuIDs, &q.IsRunning)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &q, nil
}

// CreateQueue inserts a queue and returns it.
func CreateQueue(ctx context.Context, gpuIDs string, isRunning bool) (*Queue, error) {
	res, err := DB.ExecContext(ctx, `INSERT INTO "Queue" ("gpu_ids", "is_running") VALUES (?, ?)`, gpuIDs, toDB(isRunning))
	if err != nil {
		return nil, mapErr(err)
	}
	id, _ := res.LastInsertId()
	return &Queue{ID: int(id), GpuIDs: gpuIDs, IsRunning: isRunning}, nil
}

// SetQueueRunning updates is_running on the queue with id.
func SetQueueRunning(ctx context.Context, id int, running bool) error {
	res, err := DB.ExecContext(ctx, `UPDATE "Queue" SET "is_running" = ? WHERE "id" = ?`, toDB(running), id)
	if err != nil {
		return mapErr(err)
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}
