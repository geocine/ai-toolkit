// Package settings reads the toolkit settings rows with an in-process cache.
// Mirrors ui/src/server/settings.ts (API side) and ui/cron/paths.ts (job
// launch side, which differs for MODELS_PATH).
package settings

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/store"
)

var (
	mu    sync.RWMutex
	cache = map[string]string{}
)

// Flush drops every cached value; call after settings are written.
func Flush() {
	mu.Lock()
	cache = map[string]string{}
	mu.Unlock()
}

func cachedValue(ctx context.Context, key string, compute func(row string, ok bool) string) (string, error) {
	mu.RLock()
	v, ok := cache[key]
	mu.RUnlock()
	if ok {
		return v, nil
	}
	row, found, err := store.GetSetting(ctx, key)
	if err != nil {
		return "", err
	}
	v = compute(row, found && row != "")
	mu.Lock()
	cache[key] = v
	mu.Unlock()
	return v, nil
}

// resolved strips trailing separators like path.resolve, so the routes'
// `root + sep` prefix checks never fail on a stored "/mnt/foo/".
func resolved(p string) string {
	if abs, err := filepath.Abs(p); err == nil {
		return abs
	}
	return filepath.Clean(p)
}

func DatasetsRoot(ctx context.Context) (string, error) {
	return cachedValue(ctx, "DATASETS_FOLDER", func(row string, ok bool) string {
		if ok {
			return resolved(row)
		}
		return resolved(paths.DefaultDatasetsFolder)
	})
}

func TrainingFolder(ctx context.Context) (string, error) {
	return cachedValue(ctx, "TRAINING_FOLDER", func(row string, ok bool) string {
		if ok {
			return resolved(row)
		}
		return resolved(paths.DefaultTrainFolder)
	})
}

func DataRoot(ctx context.Context) (string, error) {
	return cachedValue(ctx, "DATA_ROOT", func(row string, ok bool) string {
		if ok {
			return resolved(row)
		}
		return resolved(paths.DefaultDataRoot)
	})
}

func HFToken(ctx context.Context) (string, error) {
	return cachedValue(ctx, "HF_TOKEN", func(row string, ok bool) string {
		if ok {
			return row
		}
		return ""
	})
}

// ModelsPath is the API-side value: env MODELS_PATH wins, then the setting,
// then <toolkit>/models. Not cached, like upstream.
func ModelsPath(ctx context.Context) (string, error) {
	if v := os.Getenv("MODELS_PATH"); strings.TrimSpace(v) != "" {
		return v, nil
	}
	row, ok, err := store.GetSetting(ctx, "MODELS_PATH")
	if err != nil {
		return "", err
	}
	if ok && row != "" {
		return row, nil
	}
	return paths.DefaultModelsFolder, nil
}

// JobModelsPath is what a launched job gets as MODELS_PATH when the env does
// not already set it: the setting, unless it is empty or the default folder.
func JobModelsPath(ctx context.Context) (string, error) {
	row, ok, err := store.GetSetting(ctx, "MODELS_PATH")
	if err != nil || !ok {
		return "", err
	}
	if row == "" || row == paths.DefaultModelsFolder {
		return "", nil
	}
	return row, nil
}

// AllowedRoots is the set of folders media routes may serve from.
type AllowedRoots struct {
	Datasets string
	Training string
	Data     string
}

func Roots(ctx context.Context) (AllowedRoots, error) {
	var r AllowedRoots
	var err error
	if r.Datasets, err = DatasetsRoot(ctx); err != nil {
		return r, err
	}
	if r.Training, err = TrainingFolder(ctx); err != nil {
		return r, err
	}
	r.Data, err = DataRoot(ctx)
	return r, err
}
