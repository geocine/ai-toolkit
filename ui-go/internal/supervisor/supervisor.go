// Package supervisor watches every job so clients never see one stuck: it
// turns row changes and job-folder writes into events, records attempts,
// notices trainers that died without anyone watching them (the server
// restarted, the machine went down) and resumes them when policy allows.
// See docs/headless.md, "Recovery and resume" and "Events".
package supervisor

import (
	"context"
	"database/sql"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/proc"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/worker"
)

const (
	tick          = time.Second
	scanEvery     = 2 * time.Second
	livenessEvery = 5 * time.Second
	// a running row whose trainer never reported a pid is given this long
	noPidGrace = 2 * time.Minute
	// keep scanning a job's folder this long after it stops (final saves)
	scanAfterEnd = time.Minute
)

type snapshot struct {
	status, info, speed string
	step                int
	totalSteps          *int
	pid                 *int
	name                string
	endedAt             time.Time
}

type folderState struct {
	checkpoints map[string]int64 // name -> size at last scan
	reported    map[string]bool
	samples     map[string]bool
	lossStep    int
}

// Supervisor holds the observer's memory; one per process.
type Supervisor struct {
	prev      map[string]*snapshot
	folders   map[string]*folderState
	requeued  map[string]bool // set when auto-resume queued the job
	lastScan  time.Time
	lastAlive time.Time
}

// Run blocks until ctx is done. The first pass reconciles jobs left over from
// before this process started, without replaying their history as events.
func Run(ctx context.Context) {
	s := &Supervisor{prev: map[string]*snapshot{}, folders: map[string]*folderState{}, requeued: map[string]bool{}}
	if err := s.pass(ctx, true); err != nil {
		log.Printf("supervisor: first pass: %v", err)
	}
	t := time.NewTicker(tick)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := s.pass(ctx, false); err != nil && ctx.Err() == nil {
				log.Printf("supervisor: %v", err)
			}
		}
	}
}

func (s *Supervisor) pass(ctx context.Context, first bool) error {
	jobs, err := store.FindJobs(ctx, store.JobQuery{OrderBy: `"created_at" ASC`})
	if err != nil {
		return err
	}
	now := time.Now()
	seen := map[string]bool{}
	for _, j := range jobs {
		seen[j.ID] = true
		s.observe(ctx, j, first)
	}
	for id, p := range s.prev {
		if !seen[id] {
			events.Emit(ctx, events.JobDeleted, id, map[string]any{"name": p.name})
			delete(s.prev, id)
			delete(s.folders, id)
			delete(s.requeued, id)
		}
	}
	if first || now.Sub(s.lastAlive) >= livenessEvery {
		s.lastAlive = now
		for _, j := range jobs {
			s.checkAlive(ctx, j)
		}
	}
	if first || now.Sub(s.lastScan) >= scanEvery {
		s.lastScan = now
		root, err := settings.TrainingFolder(ctx)
		if err != nil {
			return err
		}
		for _, j := range jobs {
			s.scan(ctx, j, filepath.Join(root, j.Name), first)
		}
	}
	return nil
}

func active(status string) bool {
	return status == "running" || status == "stopping" || status == "queued"
}

// observe diffs the row against the last pass and emits what changed.
func (s *Supervisor) observe(ctx context.Context, j *store.Job, first bool) {
	p, known := s.prev[j.ID]
	cur := &snapshot{status: j.Status, info: j.Info, speed: j.SpeedString, step: j.Step, totalSteps: j.TotalSteps,
		pid: j.Pid, name: j.Name}
	if known {
		cur.endedAt = p.endedAt
	}
	s.prev[j.ID] = cur
	if first {
		// a job found running at boot gets an open attempt if it lacks one
		if j.Status == "running" || j.Status == "stopping" {
			if a, _ := state.LastAttempt(ctx, j.ID); a == nil || a.EndedAt != nil {
				state.StartAttempt(ctx, j.ID, j.Step, false)
			}
		}
		return
	}
	if !known {
		// a job created without a policy gets the configured defaults
		// (AITK_SYNC_REMOTE / AITK_AUTO_UPLOAD_REMOTE), whichever API made it
		state.ApplyDefaultPolicy(ctx, j.ID)
		events.Emit(ctx, events.JobCreated, j.ID, map[string]any{"name": j.Name})
		return
	}
	if p.status != j.Status {
		s.transition(ctx, j, p)
	}
	if p.step != j.Step || p.info != j.Info || p.speed != j.SpeedString || !eqIntPtr(p.totalSteps, j.TotalSteps) {
		events.Emit(ctx, events.JobProgress, j.ID, map[string]any{
			"step": j.Step, "total_steps": j.TotalSteps, "speed": j.SpeedString, "info": j.Info})
	}
	if j.Pid != nil && !eqIntPtr(p.pid, j.Pid) {
		state.SetAttemptPid(ctx, j.ID, *j.Pid)
	}
}

func (s *Supervisor) transition(ctx context.Context, j *store.Job, p *snapshot) {
	from := jobstate.FromStatus(p.status, p.step, p.info)
	to := jobstate.FromStatus(j.Status, j.Step, j.Info)

	switch {
	case j.Status == "running" && p.status != "stopping":
		auto := s.requeued[j.ID]
		delete(s.requeued, j.ID)
		if a, err := state.StartAttempt(ctx, j.ID, j.Step, auto); err == nil {
			events.Emit(ctx, events.JobAttempt, j.ID, a)
		}
	case (p.status == "running" || p.status == "stopping") && !active(j.Status):
		end := jobstate.EndState(j.Status, j.Info)
		if a, err := state.EndAttempt(ctx, j.ID, end, j.Info, j.Step); err == nil && a != nil {
			events.Emit(ctx, events.JobAttempt, j.ID, a)
		}
		if f := s.prev[j.ID]; f != nil {
			f.endedAt = time.Now()
		}
		to = end
		if end == jobstate.Interrupted {
			s.maybeResume(ctx, j)
		}
	case (p.status == "running" || p.status == "stopping") && j.Status == "queued":
		// returned to the queue (the queue was paused): the attempt ends
		// here, as "requeued", which is not an interruption
		if a, err := state.EndAttempt(ctx, j.ID, "requeued", j.Info, j.Step); err == nil && a != nil {
			events.Emit(ctx, events.JobAttempt, j.ID, a)
		}
	}
	events.Emit(ctx, events.JobState, j.ID, map[string]any{"from": from, "to": to, "status": j.Status, "info": j.Info})
}

// checkAlive marks a running job interrupted when its trainer is gone and no
// launcher in this process is watching it (those record exits themselves).
func (s *Supervisor) checkAlive(ctx context.Context, j *store.Job) {
	if j.Status != "running" && j.Status != "stopping" {
		return
	}
	if j.Pid == nil {
		a, _ := state.LastAttempt(ctx, j.ID)
		if a != nil && time.Since(a.StartedAt) < noPidGrace {
			return
		}
		if a == nil && time.Since(j.UpdatedAt.Time) < noPidGrace {
			return
		}
		s.markGone(ctx, j, "Interrupted: the job has no trainer process (the server restarted before it launched)")
		return
	}
	pid := *j.Pid
	if worker.Watched(pid) {
		return
	}
	if proc.Alive(pid) && isTrainer(pid, j) {
		return
	}
	s.markGone(ctx, j, "Interrupted: the trainer process is gone (the server or machine restarted, or it was killed)")
}

func (s *Supervisor) markGone(ctx context.Context, j *store.Job, reason string) {
	status, info := "error", reason
	if j.Status == "stopping" || j.Stop {
		// it was asked to stop; finishing the stop is not an interruption
		status, info = "stopped", "Job stopped"
	}
	n, err := store.UpdateJobs(ctx, store.JobQuery{
		Where: `"id" = ? AND "status" IN ('running', 'stopping')`, Args: []any{j.ID},
	}, store.Fields{"status": status, "info": info, "pid": nil})
	if err != nil {
		log.Printf("supervisor: mark %s: %v", j.ID, err)
		return
	}
	if n > 0 {
		log.Printf("supervisor: job %s (%s): %s", j.ID, j.Name, info)
	}
}

// maybeResume queues an interrupted job again when its policy allows.
func (s *Supervisor) maybeResume(ctx context.Context, j *store.Job) {
	auto, max, _, err := state.EffectivePolicy(ctx, j.ID)
	if err != nil || !auto {
		return
	}
	n, err := state.ConsecutiveInterruptions(ctx, j.ID)
	if err != nil {
		return
	}
	if n > max {
		log.Printf("supervisor: job %s interrupted %d times in a row, not resuming (max_auto_resumes %d)", j.Name, n, max)
		return
	}
	if err := Requeue(ctx, j, "Auto-resuming after interruption"); err != nil {
		log.Printf("supervisor: resume %s: %v", j.Name, err)
		return
	}
	s.requeued[j.ID] = true
	log.Printf("supervisor: job %s queued to resume (interruption %d of %d)", j.Name, n, max)
}

// Requeue puts a job back in its GPU queue (at the end) and makes sure that
// queue runs, like the start route plus starting the queue.
func Requeue(ctx context.Context, j *store.Job, info string) error {
	maxPos, err := store.MaxQueuePosition(ctx)
	if err != nil {
		return err
	}
	if _, err := store.UpdateJob(ctx, j.ID, store.Fields{
		"status": "queued", "stop": false, "return_to_queue": false, "info": info,
		"queue_position": maxPos + 1000, "pid": nil,
	}); err != nil {
		return err
	}
	q, err := store.GetQueueByGPU(ctx, j.GpuIDs)
	if err != nil {
		return err
	}
	if q == nil {
		_, err = store.CreateQueue(ctx, j.GpuIDs, true)
		return err
	}
	if !q.IsRunning {
		return store.SetQueueRunning(ctx, q.ID, true)
	}
	return nil
}

// scan reports new finished checkpoints and samples and new loss points.
func (s *Supervisor) scan(ctx context.Context, j *store.Job, folder string, first bool) {
	p := s.prev[j.ID]
	recentlyEnded := p != nil && !p.endedAt.IsZero() && time.Since(p.endedAt) < scanAfterEnd
	if !first && !active(j.Status) && !recentlyEnded {
		return
	}
	fs := s.folders[j.ID]
	if fs == nil {
		fs = &folderState{checkpoints: map[string]int64{}, reported: map[string]bool{}, samples: map[string]bool{}}
		s.folders[j.ID] = fs
	}
	if cps, err := artifacts.Checkpoints(folder); err == nil {
		for _, c := range cps {
			prevSize, seenBefore := fs.checkpoints[c.Name]
			fs.checkpoints[c.Name] = c.Size
			if first {
				fs.reported[c.Name] = true
				continue
			}
			// report once its size held still for a scan (the save finished)
			if !fs.reported[c.Name] && seenBefore && prevSize == c.Size && c.Size > 0 {
				fs.reported[c.Name] = true
				events.Emit(ctx, events.JobCheckpoint, j.ID, c)
			}
		}
	}
	if smp, err := artifacts.Samples(folder); err == nil {
		for _, sm := range smp {
			if fs.samples[sm.Name] {
				continue
			}
			fs.samples[sm.Name] = true
			if !first {
				events.Emit(ctx, events.JobSample, j.ID, sm)
			}
		}
	}
	s.scanLoss(ctx, j, folder, fs, first)
}

func (s *Supervisor) scanLoss(ctx context.Context, j *store.Job, folder string, fs *folderState, first bool) {
	path := filepath.Join(folder, "loss_log.db")
	if _, err := os.Stat(path); err != nil {
		return
	}
	db, err := sql.Open("sqlite", store.FileURI(path)+"?mode=ro&_pragma=busy_timeout(5000)")
	if err != nil {
		return
	}
	defer db.Close()
	if first {
		db.QueryRowContext(ctx, `SELECT COALESCE(MAX(step), 0) FROM metrics`).Scan(&fs.lossStep)
		return
	}
	rows, err := db.QueryContext(ctx, `SELECT step, key, value_real FROM metrics
		WHERE step > ? AND value_real IS NOT NULL ORDER BY step, key LIMIT 5000`, fs.lossStep)
	if err != nil {
		return
	}
	defer rows.Close()
	type point struct {
		Step  int     `json:"step"`
		Key   string  `json:"key"`
		Value float64 `json:"value"`
	}
	var pts []point
	for rows.Next() {
		var pt point
		if rows.Scan(&pt.Step, &pt.Key, &pt.Value) == nil {
			pts = append(pts, pt)
		}
	}
	if len(pts) == 0 {
		return
	}
	fs.lossStep = pts[len(pts)-1].Step
	events.Emit(ctx, events.JobLoss, j.ID, map[string]any{"points": pts})
}

func eqIntPtr(a, b *int) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

// isTrainer guards against pid reuse: where the OS lets us read a process's
// command line, the pid must be running this job's config.
func isTrainer(pid int, j *store.Job) bool {
	cmdline, ok := commandLine(pid)
	if !ok {
		return true // cannot tell; alive is the best we know
	}
	return strings.Contains(cmdline, j.Name) && strings.Contains(cmdline, ".job_config.json")
}
