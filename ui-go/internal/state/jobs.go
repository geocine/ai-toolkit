package state

import (
	"context"
	"database/sql"
	"log"
	"strconv"
	"time"
)

// Attempt is one launch of a job.
type Attempt struct {
	N         int        `json:"n"`
	StartedAt time.Time  `json:"started_at"`
	EndedAt   *time.Time `json:"ended_at"`
	EndState  *string    `json:"end_state"`
	EndReason *string    `json:"end_reason"`
	StartStep int        `json:"start_step"`
	EndStep   *int       `json:"end_step"`
	Auto      bool       `json:"auto"`
	Pid       *int       `json:"pid"`
}

func scanAttempt(r interface{ Scan(...any) error }) (*Attempt, error) {
	var a Attempt
	var started int64
	var ended, endStep, pid sql.NullInt64
	var endState, endReason sql.NullString
	if err := r.Scan(&a.N, &started, &ended, &endState, &endReason, &a.StartStep, &endStep, &a.Auto, &pid); err != nil {
		return nil, err
	}
	a.StartedAt = Time(started)
	if ended.Valid {
		t := Time(ended.Int64)
		a.EndedAt = &t
	}
	if endState.Valid {
		a.EndState = &endState.String
	}
	if endReason.Valid {
		a.EndReason = &endReason.String
	}
	if endStep.Valid {
		v := int(endStep.Int64)
		a.EndStep = &v
	}
	if pid.Valid {
		v := int(pid.Int64)
		a.Pid = &v
	}
	return &a, nil
}

const attemptCols = `n, started_at, ended_at, end_state, end_reason, start_step, end_step, auto, pid`

// StartAttempt opens attempt n+1 for the job, closing a dangling open one.
func StartAttempt(ctx context.Context, jobID string, startStep int, auto bool) (*Attempt, error) {
	if _, err := DB.ExecContext(ctx, `UPDATE job_attempts SET ended_at = ?, end_state = 'interrupted',
		end_reason = 'superseded by a new attempt' WHERE job_id = ? AND ended_at IS NULL`, NowMs(), jobID); err != nil {
		return nil, err
	}
	var n int
	if err := DB.QueryRowContext(ctx, `SELECT COALESCE(MAX(n), 0) + 1 FROM job_attempts WHERE job_id = ?`, jobID).Scan(&n); err != nil {
		return nil, err
	}
	if _, err := DB.ExecContext(ctx, `INSERT INTO job_attempts (job_id, n, started_at, start_step, auto) VALUES (?, ?, ?, ?, ?)`,
		jobID, n, NowMs(), startStep, auto); err != nil {
		return nil, err
	}
	return LastAttempt(ctx, jobID)
}

// SetAttemptPid records the trainer pid on the open attempt.
func SetAttemptPid(ctx context.Context, jobID string, pid int) error {
	_, err := DB.ExecContext(ctx, `UPDATE job_attempts SET pid = ? WHERE job_id = ? AND ended_at IS NULL`, pid, jobID)
	return err
}

// EndAttempt closes the open attempt; nil, nil when none was open.
func EndAttempt(ctx context.Context, jobID, endState, reason string, endStep int) (*Attempt, error) {
	res, err := DB.ExecContext(ctx, `UPDATE job_attempts SET ended_at = ?, end_state = ?, end_reason = ?, end_step = ?
		WHERE job_id = ? AND ended_at IS NULL`, NowMs(), endState, reason, endStep, jobID)
	if err != nil {
		return nil, err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return nil, nil
	}
	return LastAttempt(ctx, jobID)
}

// LastAttempt returns the newest attempt, or nil.
func LastAttempt(ctx context.Context, jobID string) (*Attempt, error) {
	a, err := scanAttempt(DB.QueryRowContext(ctx, `SELECT `+attemptCols+` FROM job_attempts
		WHERE job_id = ? ORDER BY n DESC LIMIT 1`, jobID))
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return a, err
}

// Attempts returns every attempt, oldest first.
func Attempts(ctx context.Context, jobID string) ([]*Attempt, error) {
	rows, err := DB.QueryContext(ctx, `SELECT `+attemptCols+` FROM job_attempts WHERE job_id = ? ORDER BY n`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Attempt{}
	for rows.Next() {
		a, err := scanAttempt(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

// ConsecutiveInterruptions counts trailing attempts that ended interrupted.
func ConsecutiveInterruptions(ctx context.Context, jobID string) (int, error) {
	rows, err := DB.QueryContext(ctx, `SELECT COALESCE(end_state, '') FROM job_attempts WHERE job_id = ? ORDER BY n DESC`, jobID)
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return 0, err
		}
		if s != "interrupted" {
			break
		}
		n++
	}
	return n, rows.Err()
}

// DeleteJobState drops attempts and policy for a deleted job.
func DeleteJobState(ctx context.Context, jobID string) error {
	if _, err := DB.ExecContext(ctx, `DELETE FROM job_attempts WHERE job_id = ?`, jobID); err != nil {
		return err
	}
	_, err := DB.ExecContext(ctx, `DELETE FROM job_policies WHERE job_id = ?`, jobID)
	return err
}

// Policy is a job's headless behavior. Nil fields fall back to the defaults.
type Policy struct {
	AutoResume       *bool   `json:"auto_resume"`
	MaxAutoResumes   *int    `json:"max_auto_resumes"`
	AutoUploadRemote *string `json:"auto_upload_remote"`
	// SyncRemote keeps the job portable: its resumable state is pushed there
	// so another server can import and continue it.
	SyncRemote *string `json:"sync_remote"`
}

// GetPolicy returns the stored policy (all nil when none).
func GetPolicy(ctx context.Context, jobID string) (Policy, error) {
	var p Policy
	var ar, mx sql.NullInt64
	var up, sync sql.NullString
	err := DB.QueryRowContext(ctx, `SELECT auto_resume, max_auto_resumes, auto_upload_remote, sync_remote FROM job_policies WHERE job_id = ?`, jobID).
		Scan(&ar, &mx, &up, &sync)
	if err == sql.ErrNoRows {
		return p, nil
	}
	if err != nil {
		return p, err
	}
	if ar.Valid {
		v := ar.Int64 != 0
		p.AutoResume = &v
	}
	if mx.Valid {
		v := int(mx.Int64)
		p.MaxAutoResumes = &v
	}
	if up.Valid {
		p.AutoUploadRemote = &up.String
	}
	if sync.Valid {
		p.SyncRemote = &sync.String
	}
	return p, nil
}

// SetPolicy replaces the job's policy.
func SetPolicy(ctx context.Context, jobID string, p Policy) error {
	_, err := DB.ExecContext(ctx, `INSERT INTO job_policies (job_id, auto_resume, max_auto_resumes, auto_upload_remote, sync_remote)
		VALUES (?, ?, ?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET auto_resume = excluded.auto_resume,
		max_auto_resumes = excluded.max_auto_resumes, auto_upload_remote = excluded.auto_upload_remote,
		sync_remote = excluded.sync_remote`,
		jobID, p.AutoResume, p.MaxAutoResumes, p.AutoUploadRemote, p.SyncRemote)
	return err
}

// Defaults for job policies, stored as fork settings.
const (
	SettingAutoResumeDefault       = "auto_resume_default"
	SettingMaxAutoResumesDefault   = "max_auto_resumes_default"
	SettingSyncRemoteDefault       = "sync_remote_default"
	SettingAutoUploadRemoteDefault = "auto_upload_remote_default"
)

// EffectivePolicy resolves a job's policy against the defaults (auto_resume
// off and 3 resumes unless configured).
func EffectivePolicy(ctx context.Context, jobID string) (autoResume bool, maxResumes int, uploadRemote string, err error) {
	p, err := GetPolicy(ctx, jobID)
	if err != nil {
		return false, 0, "", err
	}
	maxResumes = 3
	if v, ok, _ := Setting(ctx, SettingAutoResumeDefault); ok {
		autoResume = v == "true"
	}
	if v, ok, _ := Setting(ctx, SettingMaxAutoResumesDefault); ok {
		if n, err := strconv.Atoi(v); err == nil {
			maxResumes = n
		}
	}
	if p.AutoResume != nil {
		autoResume = *p.AutoResume
	}
	if p.MaxAutoResumes != nil {
		maxResumes = *p.MaxAutoResumes
	}
	if p.AutoUploadRemote != nil {
		uploadRemote = *p.AutoUploadRemote
	}
	return autoResume, maxResumes, uploadRemote, nil
}

// ApplyDefaultPolicy fills the job's unset sync / auto-upload remotes with
// the configured defaults (AITK_SYNC_REMOTE / AITK_AUTO_UPLOAD_REMOTE). An
// explicit "" means "none" and is kept. Called when a job is created, from
// any API.
func ApplyDefaultPolicy(ctx context.Context, jobID string) {
	p, err := GetPolicy(ctx, jobID)
	if err != nil {
		log.Printf("state: default policy for %s: %v", jobID, err)
		return
	}
	changed := false
	if p.SyncRemote == nil {
		if v, ok, _ := Setting(ctx, SettingSyncRemoteDefault); ok && v != "" {
			p.SyncRemote, changed = &v, true
		}
	}
	if p.AutoUploadRemote == nil {
		if v, ok, _ := Setting(ctx, SettingAutoUploadRemoteDefault); ok && v != "" {
			p.AutoUploadRemote, changed = &v, true
		}
	}
	if !changed {
		return
	}
	if err := SetPolicy(ctx, jobID, p); err != nil {
		log.Printf("state: default policy for %s: %v", jobID, err)
	}
}
