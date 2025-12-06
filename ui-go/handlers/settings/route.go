package settings

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/geocine/aitoolkit/internal"
	"github.com/geocine/aitoolkit/prisma/db"
)

func GetSettingsHandler(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database: " + err.Error()})
		return
	}
	defer client.Prisma.Disconnect()

	settings, err := client.Settings.FindMany().Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to fetch settings: " + err.Error()})
		return
	}
	settingsObject := make(map[string]string)
	for _, setting := range settings {
		settingsObject[setting.Key] = setting.Value
	}

	// Dynamically get default folders from internal/settings.go
	if settingsObject["TRAINING_FOLDER"] == "" {
		trainFolder, _ := internal.GetTrainingFolder(ctx, client)
		settingsObject["TRAINING_FOLDER"] = trainFolder
	}
	if settingsObject["DATASETS_FOLDER"] == "" {
		datasetsFolder, _ := internal.GetDatasetsRoot(ctx, client)
		settingsObject["DATASETS_FOLDER"] = datasetsFolder
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(settingsObject)
}

func PostSettingsHandler(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	client := db.NewClient()
	if err := client.Prisma.Connect(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to database"})
		return
	}
	defer client.Prisma.Disconnect()

	var body struct {
		HF_TOKEN        string `json:"HF_TOKEN"`
		TRAINING_FOLDER string `json:"TRAINING_FOLDER"`
		DATASETS_FOLDER string `json:"DATASETS_FOLDER"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}
	_, err := client.Settings.UpsertOne(
		db.Settings.Key.Equals("HF_TOKEN"),
	).Update(
		db.Settings.Value.Set(body.HF_TOKEN),
	).Create(
		db.Settings.Key.Set("HF_TOKEN"),
		db.Settings.Value.Set(body.HF_TOKEN),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update HF_TOKEN"})
		return
	}
	_, err = client.Settings.UpsertOne(
		db.Settings.Key.Equals("TRAINING_FOLDER"),
	).Update(
		db.Settings.Value.Set(body.TRAINING_FOLDER),
	).Create(
		db.Settings.Key.Set("TRAINING_FOLDER"),
		db.Settings.Value.Set(body.TRAINING_FOLDER),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update TRAINING_FOLDER"})
		return
	}
	_, err = client.Settings.UpsertOne(
		db.Settings.Key.Equals("DATASETS_FOLDER"),
	).Update(
		db.Settings.Value.Set(body.DATASETS_FOLDER),
	).Create(
		db.Settings.Key.Set("DATASETS_FOLDER"),
		db.Settings.Value.Set(body.DATASETS_FOLDER),
	).Exec(ctx)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to update DATASETS_FOLDER"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}
