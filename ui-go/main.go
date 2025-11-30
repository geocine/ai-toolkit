package main

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/geocine/aitoolkit/handlers/auth"
	"github.com/geocine/aitoolkit/handlers/caption"
	"github.com/geocine/aitoolkit/handlers/cpu"
	"github.com/geocine/aitoolkit/handlers/datasets"
	"github.com/geocine/aitoolkit/handlers/files"
	"github.com/geocine/aitoolkit/handlers/gpu"
	"github.com/geocine/aitoolkit/handlers/img"
	"github.com/geocine/aitoolkit/handlers/jobs"
	"github.com/geocine/aitoolkit/handlers/queue"
	"github.com/geocine/aitoolkit/handlers/settings"
	"github.com/geocine/aitoolkit/handlers/zip"
)

// Middleware to allow public routes and apply custom logic to others
func publicRouteMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		publicRoutes := []string{"/api/img/", "/api/files/"}
		for _, prefix := range publicRoutes {
			if strings.HasPrefix(r.URL.Path, prefix) {
				next.ServeHTTP(w, r)
				return
			}
		}
		// Custom logic for non-public routes can go here (e.g., auth)
		// For now, just allow all
		next.ServeHTTP(w, r)
	})
}

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/jobs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodGet:
			jobs.GetJobsHandler(w, r)
		case http.MethodPost:
			jobs.PostJobsHandler(w, r)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})
	// Add subroutes
	mux.HandleFunc("/api/jobs/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		path := r.URL.Path
		if strings.HasSuffix(path, "/files") {
			jobs.FilesHandler(w, r)
			return
		}
		if strings.HasSuffix(path, "/log") {
			jobs.LogHandler(w, r)
			return
		}
		if strings.HasSuffix(path, "/samples") {
			jobs.SamplesHandler(w, r)
			return
		}
		if strings.HasSuffix(path, "/stop") {
			jobs.StopHandler(w, r)
			return
		}
		if strings.HasSuffix(path, "/start") {
			jobs.StartHandler(w, r)
			return
		}
		if strings.HasSuffix(path, "/delete") {
			jobs.DeleteHandler(w, r)
			return
		}
		if strings.HasSuffix(path, "/mark_stopped") {
			jobs.MarkStoppedHandler(w, r)
			return
		}
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Not found"})
	})

	// Queue routes
	mux.HandleFunc("/api/queue", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			queue.GetQueuesHandler(w, r)
		} else {
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/api/queue/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		path := r.URL.Path
		if strings.HasSuffix(path, "/start") {
			queue.StartHandler(w, r)
			return
		}
		if strings.HasSuffix(path, "/stop") {
			queue.StopHandler(w, r)
			return
		}
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Not found"})
	})
	mux.HandleFunc("/api/settings", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			settings.GetSettingsHandler(w, r)
		case http.MethodPost:
			settings.PostSettingsHandler(w, r)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/api/img/delete", img.DeleteHandler)
	mux.HandleFunc("/api/img/caption", img.CaptionHandler)
	mux.HandleFunc("/api/img/upload", img.UploadHandler)
	mux.HandleFunc("/api/img/", img.ServeHandler)
	mux.HandleFunc("/api/gpu", gpu.RouteHandler)
	mux.HandleFunc("/api/cpu", cpu.RouteHandler)
	mux.HandleFunc("/api/auth", auth.RouteHandler)
	mux.HandleFunc("/api/datasets/listImages", datasets.ListImagesHandler)
	mux.HandleFunc("/api/datasets/upload", datasets.UploadHandler)
	mux.HandleFunc("/api/datasets/create", datasets.CreateHandler)
	mux.HandleFunc("/api/datasets/delete", datasets.DeleteHandler)
	mux.HandleFunc("/api/datasets/list", datasets.ListHandler)
	mux.HandleFunc("/api/files/", files.ServeHandler)
	mux.HandleFunc("/api/caption/", caption.ServeHandler)
	mux.HandleFunc("/api/zip", zip.RouteHandler)

	log.Println("Listening on :8080...")
	log.Fatal(http.ListenAndServe(":8080", publicRouteMiddleware(mux)))
}
