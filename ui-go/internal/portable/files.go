package portable

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// hashFile returns the sha256 of path, cached in the fork database by path,
// size and mtime so unchanged dataset files are hashed once.
func hashFile(ctx context.Context, p string) (string, int64, error) {
	st, err := os.Stat(p)
	if err != nil {
		return "", 0, err
	}
	abs, _ := filepath.Abs(p)
	size, mtime := st.Size(), st.ModTime().UnixNano()
	if state.DB != nil {
		var sha string
		var cs, cm int64
		err := state.DB.QueryRowContext(ctx, `SELECT size, mtime_ns, sha256 FROM file_hashes WHERE path = ?`, abs).Scan(&cs, &cm, &sha)
		if err == nil && cs == size && cm == mtime {
			return sha, size, nil
		}
	}
	sha, err := artifacts.SHA256File(p)
	if err != nil {
		return "", 0, err
	}
	if state.DB != nil {
		state.DB.ExecContext(ctx, `INSERT INTO file_hashes (path, size, mtime_ns, sha256) VALUES (?, ?, ?, ?)
			ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime_ns = excluded.mtime_ns, sha256 = excluded.sha256`,
			abs, size, mtime, sha)
	}
	return sha, size, nil
}

// waitStable waits until the file's size stops changing (a save still being
// written), up to limit. Missing file -> error.
func waitStable(ctx context.Context, p string, limit time.Duration) error {
	deadline := time.Now().Add(limit)
	var last int64 = -1
	for {
		st, err := os.Stat(p)
		if err != nil {
			return err
		}
		if st.Size() == last && st.Size() > 0 {
			return nil
		}
		last = st.Size()
		if time.Now().After(deadline) {
			return nil // give up waiting; upload what is there
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(stablePoll):
		}
	}
}

var stablePoll = 2 * time.Second

// snapshotSQLite copies a live sqlite database consistently (the trainer may
// be writing it) into dst using VACUUM INTO on a read-only connection.
func snapshotSQLite(ctx context.Context, src, dst string) error {
	os.Remove(dst)
	db, err := sql.Open("sqlite", store.FileURI(src)+"?mode=ro&_pragma=busy_timeout(30000)")
	if err != nil {
		return err
	}
	defer db.Close()
	if _, err := db.ExecContext(ctx, `VACUUM INTO ?`, dst); err != nil {
		return fmt.Errorf("snapshot %s: %w", filepath.Base(src), err)
	}
	return nil
}

// copyFile snapshots a plain file (log.txt grows while we read it; a copy
// keeps the hash and the upload consistent).
func copyFile(src, dst string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o644)
}
