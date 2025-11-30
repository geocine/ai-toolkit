package zip

import (
	"archive/zip"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

type PostBody struct {
	ZipTarget string `json:"zipTarget"`
	JobName   string `json:"jobName"`
}

// RouteHandler handles POST /api/zip
func RouteHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "Method not allowed"})
		return
	}

	var body PostBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	if body.JobName == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "jobName is required"})
		return
	}

	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	// Get training folder
	trainingRoot, err := internal.GetTrainingFolder(ctx, client)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to get training folder"})
		return
	}

	folderPath := filepath.Join(trainingRoot, body.JobName, "samples")
	outputPath := filepath.Join(trainingRoot, body.JobName, "samples.zip")

	// Check if folder exists and is a directory
	stat, err := os.Stat(folderPath)
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Folder not found"})
		return
	}
	if !stat.IsDir() {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Not a directory"})
		return
	}

	// Delete existing zip if it exists
	if _, err := os.Stat(outputPath); err == nil {
		os.Remove(outputPath)
	}

	// Create zip file
	if err := createZip(folderPath, outputPath); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create zip"})
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":       true,
		"zipPath":  outputPath,
		"fileName": filepath.Base(outputPath),
	})
}

func createZip(sourceDir, outputPath string) error {
	zipFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer zipFile.Close()

	archive := zip.NewWriter(zipFile)
	defer archive.Close()

	rootName := filepath.Base(sourceDir)

	return filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Get relative path from sourceDir
		relPath, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}

		// Skip the root directory itself
		if relPath == "." {
			return nil
		}

		// Prefix with root folder name
		zipPath := filepath.Join(rootName, relPath)
		// Use forward slashes for zip entries
		zipPath = filepath.ToSlash(zipPath)

		if info.IsDir() {
			// Create directory entry
			_, err := archive.Create(zipPath + "/")
			return err
		}

		// Create file entry
		writer, err := archive.Create(zipPath)
		if err != nil {
			return err
		}

		// Copy file contents
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()

		_, err = io.Copy(writer, file)
		return err
	})
}
