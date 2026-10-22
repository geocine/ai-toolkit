package portable

import (
	"context"
	"time"
)

// Helpers shared with run bundles (internal/bundles, handlers/v1), which pack
// the same resumable state into one zip instead of a remote layout.

// HashFile is the file's sha256 and size, cached by path, size and mtime.
func HashFile(ctx context.Context, p string) (string, int64, error) { return hashFile(ctx, p) }

// WaitStable waits until the file's size stops changing (a save still being
// written), up to limit.
func WaitStable(ctx context.Context, p string, limit time.Duration) error {
	return waitStable(ctx, p, limit)
}

// SnapshotSQLite copies a live sqlite database (the trainer's loss log)
// consistently into dst.
func SnapshotSQLite(ctx context.Context, src, dst string) error { return snapshotSQLite(ctx, src, dst) }

// WriteJobConfig writes .job_config.json the way the worker does at launch.
func WriteJobConfig(dst, config string) error { return writeJobConfig(dst, config) }

// SafeRel accepts slash-separated relative paths that stay inside their folder.
func SafeRel(p string) bool { return safeRel(p) }

// RewriteStrings replaces every string value of a parsed config (jsonx tree)
// for which fn returns true.
func RewriteStrings(v any, fn func(string) (string, bool)) any { return walk(v, fn) }
