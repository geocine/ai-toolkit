package jobs

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func StartHandler(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	jobID := r.URL.Query().Get("id")
	if jobID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Missing job id"})
		return
	}
	job, err := client.Job.FindUnique(db.Job.ID.Equals(jobID)).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to fetch job"})
		return
	}
	if job == nil {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Job not found"})
		return
	}
	// Update job status to running
	_, err = client.Job.FindUnique(db.Job.ID.Equals(jobID)).Update(
		db.Job.Status.Set("running"),
		db.Job.Stop.Set(false),
		db.Job.Info.Set("Starting job..."),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update job status"})
		return
	}

	// Setup training folder
	trainingRoot, err := internal.GetTrainingFolder(ctx, client)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to get training root"})
		return
	}
	trainingFolder := filepath.Join(trainingRoot, job.Name)
	if err := os.MkdirAll(trainingFolder, 0755); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create training folder"})
		return
	}

	// Log file
	logPath := filepath.Join(trainingFolder, "log.txt")
	logsFolder := filepath.Join(trainingFolder, "logs")
	if _, err := os.Stat(logPath); err == nil {
		if err := os.MkdirAll(logsFolder, 0755); err == nil {
			num := 0
			for {
				candidate := filepath.Join(logsFolder, fmt.Sprintf("%d_log.txt", num))
				if _, err := os.Stat(candidate); os.IsNotExist(err) {
					_ = os.Rename(logPath, candidate)
					break
				}
				num++
			}
		}
	}

	// Write config file, update sqlite_db_path
	configPath := filepath.Join(trainingFolder, ".job_config.json")
	var jobConfig map[string]interface{}
	_ = json.Unmarshal([]byte(job.JobConfig), &jobConfig)
	if processArr, ok := jobConfig["config"].(map[string]interface{})["process"].([]interface{}); ok && len(processArr) > 0 {
		if proc, ok := processArr[0].(map[string]interface{}); ok {
			proc["sqlite_db_path"] = filepath.Join(".", "aitk_db.db")
		}
	}
	configBytes, _ := json.MarshalIndent(jobConfig, "", "  ")
	if err := os.WriteFile(configPath, configBytes, 0644); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to write config file"})
		return
	}

	// Python path (support venv)
	toolkitRoot := "."
	pythonPath := "python"
	venvPaths := []string{
		filepath.Join(toolkitRoot, ".venv", "Scripts", "python.exe"),
		filepath.Join(toolkitRoot, ".venv", "bin", "python"),
		filepath.Join(toolkitRoot, "venv", "Scripts", "python.exe"),
		filepath.Join(toolkitRoot, "venv", "bin", "python"),
	}
	for _, p := range venvPaths {
		if _, err := os.Stat(p); err == nil {
			pythonPath = p
			break
		}
	}

	runFilePath := filepath.Join(toolkitRoot, "run.py")
	if _, err := os.Stat(runFilePath); os.IsNotExist(err) {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "run.py not found"})
		return
	}
	args := []string{runFilePath, configPath, "--log", logPath}

	// Environment variables
	env := append(os.Environ(),
		"AITK_JOB_ID="+jobID,
		"CUDA_VISIBLE_DEVICES="+job.GpuIds,
		"IS_AI_TOOLKIT_UI=1",
	)
	hfToken, _ := internal.GetHFToken(ctx, client)
	if strings.TrimSpace(hfToken) != "" {
		env = append(env, "HF_TOKEN="+hfToken)
	}

	cmd := exec.Command(pythonPath, args...)
	cmd.Dir = toolkitRoot
	cmd.Env = env
	if err := cmd.Start(); err != nil {
		client.Job.FindUnique(db.Job.ID.Equals(jobID)).Update(
			db.Job.Status.Set("error"),
			db.Job.Info.Set("Error launching job: "+err.Error()),
		).Exec(ctx)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to start process"})
		return
	}
	go func() {
		err := cmd.Wait()
		if err != nil {
			client.Job.FindUnique(db.Job.ID.Equals(jobID)).Update(
				db.Job.Status.Set("error"),
				db.Job.Info.Set("Error running job: "+err.Error()),
			).Exec(ctx)
		} else {
			client.Job.FindUnique(db.Job.ID.Equals(jobID)).Update(
				db.Job.Status.Set("stopped"),
				db.Job.Info.Set("Job completed"),
			).Exec(ctx)
		}
	}()
	json.NewEncoder(w).Encode(job)
}
