package store

import (
	"context"
	"database/sql"
	"strings"

	"github.com/google/uuid"
)

// Job mirrors model Job in ui/prisma/schema.prisma; JSON matches what the
// Next routes return for a Prisma Job.
type Job struct {
	ID            string  `json:"id"`
	Name          string  `json:"name"`
	GpuIDs        string  `json:"gpu_ids"`
	JobConfig     string  `json:"job_config"`
	CreatedAt     Time    `json:"created_at"`
	UpdatedAt     Time    `json:"updated_at"`
	Status        string  `json:"status"`
	Stop          bool    `json:"stop"`
	ReturnToQueue bool    `json:"return_to_queue"`
	Step          int     `json:"step"`
	TotalSteps    *int    `json:"total_steps"`
	Info          string  `json:"info"`
	SpeedString   string  `json:"speed_string"`
	QueuePosition int     `json:"queue_position"`
	Pid           *int    `json:"pid"`
	JobType       string  `json:"job_type"`
	JobRef        *string `json:"job_ref"`
	SaveNow       bool    `json:"save_now"`
	SampleNow     bool    `json:"sample_now"`
}

var jobColumns = []string{
	"id", "name", "gpu_ids", "job_config", "created_at", "updated_at", "status", "stop",
	"return_to_queue", "step", "total_steps", "info", "speed_string", "queue_position",
	"pid", "job_type", "job_ref", "save_now", "sample_now",
}

var jobSelect = `SELECT ` + quoteCols(jobColumns) + ` FROM "Job"`

type rowScanner interface{ Scan(dest ...any) error }

func scanJob(r rowScanner) (*Job, error) {
	var j Job
	var totalSteps, pid sql.NullInt64
	var jobRef sql.NullString
	err := r.Scan(&j.ID, &j.Name, &j.GpuIDs, &j.JobConfig, &j.CreatedAt, &j.UpdatedAt, &j.Status,
		&j.Stop, &j.ReturnToQueue, &j.Step, &totalSteps, &j.Info, &j.SpeedString, &j.QueuePosition,
		&pid, &j.JobType, &jobRef, &j.SaveNow, &j.SampleNow)
	if err != nil {
		return nil, err
	}
	if totalSteps.Valid {
		v := int(totalSteps.Int64)
		j.TotalSteps = &v
	}
	if pid.Valid {
		v := int(pid.Int64)
		j.Pid = &v
	}
	if jobRef.Valid {
		v := jobRef.String
		j.JobRef = &v
	}
	return &j, nil
}

// JobQuery is a WHERE clause (without the keyword) plus its args, and an
// ORDER BY. Empty Where matches every row.
type JobQuery struct {
	Where   string
	Args    []any
	OrderBy string
}

func (q JobQuery) sql(base string) string {
	s := base
	if q.Where != "" {
		s += " WHERE " + q.Where
	}
	if q.OrderBy != "" {
		s += " ORDER BY " + q.OrderBy
	}
	return s
}

// GetJob returns the job with id, or nil when there is none (Prisma findUnique).
func GetJob(ctx context.Context, id string) (*Job, error) {
	return FindFirstJob(ctx, JobQuery{Where: `"id" = ?`, Args: []any{id}})
}

// FindFirstJob returns the first matching job, or nil (Prisma findFirst).
func FindFirstJob(ctx context.Context, q JobQuery) (*Job, error) {
	j, err := scanJob(DB.QueryRowContext(ctx, q.sql(jobSelect)+" LIMIT 1", q.Args...))
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return j, mapErr(err)
}

// FindJobs returns every matching job (Prisma findMany).
func FindJobs(ctx context.Context, q JobQuery) ([]*Job, error) {
	rows, err := DB.QueryContext(ctx, q.sql(jobSelect), q.Args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	jobs := []*Job{}
	for rows.Next() {
		j, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, rows.Err()
}

// NewJob carries the fields a create may set; the rest take the schema defaults.
type NewJob struct {
	Name          string
	GpuIDs        string
	JobConfig     string
	QueuePosition int
	JobType       *string
	JobRef        *string
}

// CreateJob inserts a job the way Prisma create does: uuid id, created_at and
// updated_at set to now, schema defaults for everything else.
func CreateJob(ctx context.Context, n NewJob) (*Job, error) {
	now := Now()
	jobType := "train"
	if n.JobType != nil {
		jobType = *n.JobType
	}
	id := uuid.NewString()
	_, err := DB.ExecContext(ctx, `INSERT INTO "Job" ("id", "name", "gpu_ids", "job_config", "created_at", "updated_at",
		"status", "stop", "return_to_queue", "step", "info", "speed_string", "queue_position", "job_type", "job_ref",
		"save_now", "sample_now") VALUES (?, ?, ?, ?, ?, ?, 'stopped', 0, 0, 0, '', '', ?, ?, ?, 0, 0)`,
		id, n.Name, n.GpuIDs, n.JobConfig, now.UnixMilli(), now.UnixMilli(), n.QueuePosition, jobType, toDB(n.JobRef))
	if err != nil {
		return nil, mapErr(err)
	}
	return GetJob(ctx, id)
}

// UpdateJob applies f to the job and returns the updated row; ErrNotFound when
// the job does not exist (Prisma update). updated_at is bumped like @updatedAt.
func UpdateJob(ctx context.Context, id string, f Fields) (*Job, error) {
	n, err := UpdateJobs(ctx, JobQuery{Where: `"id" = ?`, Args: []any{id}}, f)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, ErrNotFound
	}
	return GetJob(ctx, id)
}

// UpdateJobs applies f to every matching job and returns how many changed (Prisma updateMany).
func UpdateJobs(ctx context.Context, q JobQuery, f Fields) (int64, error) {
	withTime := Fields{"updated_at": Now()}
	for k, v := range f {
		withTime[k] = v
	}
	set, args, err := buildSet("Job", withTime)
	if err != nil {
		return 0, err
	}
	stmt := `UPDATE "Job" SET ` + set
	if q.Where != "" {
		stmt += " WHERE " + q.Where
	}
	res, err := DB.ExecContext(ctx, stmt, append(args, q.Args...)...)
	if err != nil {
		return 0, mapErr(err)
	}
	return res.RowsAffected()
}

// DeleteJob removes the job; ErrNotFound when it does not exist (Prisma delete).
func DeleteJob(ctx context.Context, id string) error {
	res, err := DB.ExecContext(ctx, `DELETE FROM "Job" WHERE "id" = ?`, id)
	if err != nil {
		return mapErr(err)
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}

// MaxQueuePosition is Prisma's aggregate _max.queue_position (0 when empty).
func MaxQueuePosition(ctx context.Context) (int, error) {
	var max sql.NullInt64
	err := DB.QueryRowContext(ctx, `SELECT MAX("queue_position") FROM "Job"`).Scan(&max)
	return int(max.Int64), err
}

// In builds `"col" IN (?, ?, ...)` and its args.
func In(col string, values ...string) (string, []any) {
	ph := strings.TrimSuffix(strings.Repeat("?, ", len(values)), ", ")
	args := make([]any, len(values))
	for i, v := range values {
		args[i] = v
	}
	return `"` + col + `" IN (` + ph + `)`, args
}
