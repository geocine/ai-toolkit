package jobs

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/geocine/aitoolkit/prisma/db"
)

type Job struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	GPUIds    []string  `json:"gpu_ids"`
	JobConfig string    `json:"job_config"`
	CreatedAt time.Time `json:"created_at"`
	Status    string    `json:"status"`
	Info      string    `json:"info"`
}

func GetJobsHandler(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	id := r.URL.Query().Get("id")
	if id != "" {
		job, err := client.Job.FindUnique(
			db.Job.ID.Equals(id),
		).Exec(ctx)
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
		json.NewEncoder(w).Encode(job)
		return
	}
	jobs, err := client.Job.FindMany().OrderBy(db.Job.CreatedAt.Order(db.DESC)).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to fetch jobs"})
		return
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"jobs": jobs})
}

func PostJobsHandler(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	var body struct {
		ID        string      `json:"id"`
		Name      string      `json:"name"`
		GPUIds    []string    `json:"gpu_ids"`
		JobConfig interface{} `json:"job_config"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}
	jobConfigStr, _ := json.Marshal(body.JobConfig)
	gpuIdsStr, _ := json.Marshal(body.GPUIds)
	if body.ID != "" {
		// Update
		job, err := client.Job.FindUnique(db.Job.ID.Equals(body.ID)).Exec(ctx)
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
		updated, err := client.Job.FindUnique(db.Job.ID.Equals(body.ID)).Update(
			db.Job.Name.Set(body.Name),
			db.Job.GpuIds.Set(string(gpuIdsStr)),
			db.Job.JobConfig.Set(string(jobConfigStr)),
		).Exec(ctx)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update job"})
			return
		}
		json.NewEncoder(w).Encode(updated)
		return
	}
	// Create: check for unique name
	exists, err := client.Job.FindFirst(db.Job.Name.Equals(body.Name)).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to check job name"})
		return
	}
	if exists != nil {
		w.WriteHeader(http.StatusConflict)
		json.NewEncoder(w).Encode(map[string]string{"error": "Job name already exists"})
		return
	}
	created, err := client.Job.CreateOne(
		db.Job.Name.Set(body.Name),
		db.Job.GpuIds.Set(string(gpuIdsStr)),
		db.Job.JobConfig.Set(string(jobConfigStr)),
		db.Job.CreatedAt.Set(time.Now()),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create job"})
		return
	}
	json.NewEncoder(w).Encode(created)
}

func marshalJobConfig(cfg interface{}) string {
	b, _ := json.Marshal(cfg)
	return string(b)
}

func generateID() string {
	return time.Now().Format("20060102150405.000000")
}
