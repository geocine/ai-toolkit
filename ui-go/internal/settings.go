package internal

import (
	"context"
	"os"
	"path/filepath"
	"sync"

	"github.com/geocine/aitoolkit/prisma/db"
)

var (
	cache     = make(map[string]string)
	cacheLock sync.RWMutex
)

// getToolkitRoot returns the parent directory of ui-go (the toolkit root)
func getToolkitRoot() string {
	// Get the executable's directory or current working directory
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	// Go up one level from ui-go to get the toolkit root
	return filepath.Dir(cwd)
}

func getDefaultDatasetsFolder() string {
	return filepath.Join(getToolkitRoot(), "datasets")
}

func getDefaultTrainFolder() string {
	return filepath.Join(getToolkitRoot(), "output")
}

func GetDatasetsRoot(ctx context.Context, client *db.PrismaClient) (string, error) {
	const key = "DATASETS_FOLDER"
	cacheLock.RLock()
	if v, ok := cache[key]; ok {
		cacheLock.RUnlock()
		return v, nil
	}
	cacheLock.RUnlock()

	setting, err := client.Settings.FindFirst(
		db.Settings.Key.Equals(key),
	).Exec(ctx)
	value := getDefaultDatasetsFolder()
	if err == nil && setting != nil && setting.Value != "" {
		value = setting.Value
	}
	cacheLock.Lock()
	cache[key] = value
	cacheLock.Unlock()
	return value, nil
}

func GetTrainingFolder(ctx context.Context, client *db.PrismaClient) (string, error) {
	const key = "TRAINING_FOLDER"
	cacheLock.RLock()
	if v, ok := cache[key]; ok {
		cacheLock.RUnlock()
		return v, nil
	}
	cacheLock.RUnlock()

	setting, err := client.Settings.FindFirst(
		db.Settings.Key.Equals(key),
	).Exec(ctx)
	value := getDefaultTrainFolder()
	if err == nil && setting != nil && setting.Value != "" {
		value = setting.Value
	}
	cacheLock.Lock()
	cache[key] = value
	cacheLock.Unlock()
	return value, nil
}

func GetHFToken(ctx context.Context, client *db.PrismaClient) (string, error) {
	const key = "HF_TOKEN"
	cacheLock.RLock()
	if v, ok := cache[key]; ok {
		cacheLock.RUnlock()
		return v, nil
	}
	cacheLock.RUnlock()

	setting, err := client.Settings.FindFirst(
		db.Settings.Key.Equals(key),
	).Exec(ctx)
	value := ""
	if err == nil && setting != nil && setting.Value != "" {
		value = setting.Value
	}
	cacheLock.Lock()
	cache[key] = value
	cacheLock.Unlock()
	return value, nil
}

func GetDataRoot(ctx context.Context, client *db.PrismaClient) (string, error) {
	const key = "DATA_ROOT"
	cacheLock.RLock()
	if v, ok := cache[key]; ok {
		cacheLock.RUnlock()
		return v, nil
	}
	cacheLock.RUnlock()

	setting, err := client.Settings.FindFirst(
		db.Settings.Key.Equals(key),
	).Exec(ctx)
	value := "./data"
	if err == nil && setting != nil && setting.Value != "" {
		value = setting.Value
	}
	cacheLock.Lock()
	cache[key] = value
	cacheLock.Unlock()
	return value, nil
}
