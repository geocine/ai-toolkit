package internal

import (
	"context"
	"sync"

	"github.com/geocine/aitoolkit/prisma/db"
)

var (
	cache                 = make(map[string]string)
	cacheLock             sync.RWMutex
	defaultDatasetsFolder = "./datasets"
	defaultTrainFolder    = "./training_data"
)

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
	value := defaultDatasetsFolder
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
	value := defaultTrainFolder
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
