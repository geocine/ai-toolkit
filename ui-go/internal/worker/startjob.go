package worker

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/proc"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

// Mirrors ui/cron/actions/startJob.ts.

// StartJob marks the job running synchronously (so one tick never starts two
// jobs) and launches it in the background.
func StartJob(ctx context.Context, jobID string) error {
	job, err := store.GetJob(ctx, jobID)
	if err != nil {
		return err
	}
	if job == nil {
		log.Printf("Job with ID %s not found", jobID)
		return nil
	}
	if _, err := store.UpdateJob(ctx, jobID, store.Fields{
		"status":          "running",
		"stop":            false,
		"return_to_queue": false,
		"info":            "Starting job...",
	}); err != nil {
		return err
	}
	go startAndWatchJob(context.Background(), job)
	return nil
}

func appendJobLog(logPath, msg string) {
	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Printf("Error writing to job log: %v", err)
		return
	}
	defer f.Close()
	if _, err := f.WriteString(msg); err != nil {
		log.Printf("Error writing to job log: %v", err)
	}
}

func markError(ctx context.Context, jobID, info string) {
	if _, err := store.UpdateJob(ctx, jobID, store.Fields{"status": "error", "info": info}); err != nil {
		log.Printf("Error updating job %s: %v", jobID, err)
	}
}

// rotateLog moves an existing log.txt to logs/<n>_log.txt (lowest free n).
func rotateLog(trainingFolder, logPath string) {
	if _, err := os.Stat(logPath); err != nil {
		return
	}
	logsFolder := filepath.Join(trainingFolder, "logs")
	if err := os.MkdirAll(logsFolder, 0o755); err != nil {
		log.Printf("Error moving log file: %v", err)
		return
	}
	num := 0
	for {
		if _, err := os.Stat(filepath.Join(logsFolder, fmt.Sprintf("%d_log.txt", num))); err != nil {
			break
		}
		num++
	}
	if err := os.Rename(logPath, filepath.Join(logsFolder, fmt.Sprintf("%d_log.txt", num))); err != nil {
		log.Printf("Error moving log file: %v", err)
	}
}

// writeJobConfig writes .job_config.json: the stored config with the db path
// (and, for inference engines, the job folder) the Python side needs.
func writeJobConfig(job *store.Job, trainingFolder, configPath string) error {
	cfg, err := jsonx.ParseObject([]byte(job.JobConfig))
	if err != nil {
		return fmt.Errorf("invalid job_config: %w", err)
	}
	process := cfg.Object("config").Array("process")
	if len(process) == 0 {
		return fmt.Errorf("job_config has no config.process[0]")
	}
	p0, ok := process[0].(*jsonx.Object)
	if !ok {
		return fmt.Errorf("job_config config.process[0] is not an object")
	}
	p0.Set("sqlite_db_path", paths.DBPath)
	if job.JobType == "inference" {
		engine := p0.Object("engine")
		if engine == nil {
			engine = jsonx.NewObject()
		}
		engine.Set("job_folder", trainingFolder)
		p0.Set("engine", engine)
	}
	out, err := jsonx.Indent(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(configPath, out, 0o644)
}

func jobEnv(ctx context.Context, job *store.Job) []string {
	env := append(os.Environ(),
		"AITK_JOB_ID="+job.ID,
		"CUDA_DEVICE_ORDER=PCI_BUS_ID",
		"CUDA_VISIBLE_DEVICES="+job.GpuIDs,
		"IS_AI_TOOLKIT_UI=1",
		"PYTHONUNBUFFERED=1", // write Python output immediately so it is not lost on a crash
	)
	if tok, _ := settings.HFToken(ctx); strings.TrimSpace(tok) != "" {
		env = append(env, "HF_TOKEN="+tok)
	}
	// an env MODELS_PATH always wins (it passes through os.Environ)
	if strings.TrimSpace(os.Getenv("MODELS_PATH")) == "" {
		if mp, _ := settings.JobModelsPath(ctx); strings.TrimSpace(mp) != "" {
			env = append(env, "MODELS_PATH="+mp)
		}
	}
	return env
}

func startAndWatchJob(ctx context.Context, job *store.Job) {
	trainingRoot, err := settings.TrainingFolder(ctx)
	if err != nil {
		markError(ctx, job.ID, "Error launching job: "+err.Error())
		return
	}
	trainingFolder := filepath.Join(trainingRoot, job.Name)
	if err := os.MkdirAll(trainingFolder, 0o755); err != nil {
		markError(ctx, job.ID, "Error launching job: "+err.Error())
		return
	}
	configPath := filepath.Join(trainingFolder, ".job_config.json")
	logPath := filepath.Join(trainingFolder, "log.txt")
	rotateLog(trainingFolder, logPath)

	if err := writeJobConfig(job, trainingFolder, configPath); err != nil {
		log.Printf("Error launching process: %v", err)
		appendJobLog(logPath, "Error launching job process: "+err.Error()+"\n")
		markError(ctx, job.ID, "Error launching job: "+err.Error())
		return
	}

	runFile := filepath.Join(paths.ToolkitRoot, "run.py")
	if _, err := os.Stat(runFile); err != nil {
		log.Printf("run.py not found at path: %s", runFile)
		markError(ctx, job.ID, "Error launching job: run.py not found")
		return
	}

	pid, err := launch(launchSpec{
		python:         paths.ResolveDetachedPython(),
		args:           []string{runFile, configPath},
		env:            jobEnv(ctx, job),
		dir:            paths.ToolkitRoot,
		logPath:        logPath,
		trainingFolder: trainingFolder,
		jobID:          job.ID,
	})
	if err != nil {
		log.Printf("Error launching process: %v", err)
		appendJobLog(logPath, "Error launching job process: "+err.Error()+"\n")
		markError(ctx, job.ID, "Error launching job: "+err.Error())
		return
	}

	// Save the PID to the database and a file for future management (stop/inspect)
	if pid > 0 {
		if _, err := store.UpdateJob(ctx, job.ID, store.Fields{"pid": pid}); err != nil {
			log.Printf("Error saving job pid: %v", err)
		}
	}
	pidText := ""
	if pid > 0 {
		pidText = fmt.Sprint(pid)
	}
	if err := os.WriteFile(filepath.Join(trainingFolder, "pid.txt"), []byte(pidText), 0o644); err != nil {
		log.Printf("Error writing pid file: %v", err)
	}
}

// failRunning records a job that died while still marked running; a stopped
// or completed job writes its own ending, so it is left alone.
func failRunning(jobID, logPath, message string) {
	n, err := store.UpdateJobs(context.Background(), store.JobQuery{
		Where: `"id" = ? AND "status" = 'running'`, Args: []any{jobID},
	}, store.Fields{"status": "error", "info": message, "pid": nil})
	if err != nil {
		log.Printf("Error updating job after process exit: %v", err)
		return
	}
	if n > 0 {
		appendJobLog(logPath, "\n"+message+"\n")
	}
}

const (
	jobPollInterval = 2 * time.Second
	logTailBytes    = 4096
	logTailMaxChars = 300
)

var (
	watchedMu sync.Mutex
	watched   = map[int]bool{}
)

func setWatched(pid int, on bool) {
	watchedMu.Lock()
	defer watchedMu.Unlock()
	if on {
		watched[pid] = true
	} else {
		delete(watched, pid)
	}
}

// Watched reports whether this process is watching pid (it launched the job
// and will record its exit). The supervisor only polls jobs nobody watches,
// such as ones still running from before a server restart.
func Watched(pid int) bool {
	watchedMu.Lock()
	defer watchedMu.Unlock()
	return watched[pid]
}

// watchDetached polls a job that is not our child: there is no exit event, so
// a job that dies without updating its own row (OOM kill, hard crash) would
// otherwise sit on 'running' forever.
func watchDetached(pid int, jobID, logPath string) {
	setWatched(pid, true)
	go func() {
		defer setWatched(pid, false)
		t := time.NewTicker(jobPollInterval)
		defer t.Stop()
		for range t.C {
			if proc.Alive(pid) {
				continue
			}
			msg := "Job process exited unexpectedly."
			if tail := readLogTail(logPath); tail != "" {
				msg = "Job process exited unexpectedly. Last log line: " + tail
			}
			failRunning(jobID, logPath, msg)
			return
		}
	}()
}

// readLogTail is the last non-empty line of the log (its last 300 chars):
// for a job that dies on startup that traceback line is the whole diagnosis.
func readLogTail(logPath string) string {
	f, err := os.Open(logPath)
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || st.Size() == 0 {
		return ""
	}
	n := min(st.Size(), logTailBytes)
	buf := make([]byte, n)
	if _, err := f.ReadAt(buf, st.Size()-n); err != nil {
		return ""
	}
	lines := strings.Split(strings.ReplaceAll(string(buf), "\r\n", "\n"), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) == "" {
			continue
		}
		line := []rune(lines[i])
		if len(line) > logTailMaxChars {
			line = line[len(line)-logTailMaxChars:]
		}
		return string(line)
	}
	return ""
}
