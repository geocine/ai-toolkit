// Package state is the fork's own SQLite database (aitk_go.db): everything
// the headless server adds (events, API keys, job attempts and policies,
// remotes, uploads, hash cache) lives here so upstream's schema is never
// extended. See docs/headless.md.
package state

import (
	"context"
	"crypto/rand"
	"database/sql"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/store"
	_ "modernc.org/sqlite"
)

// DB is the shared handle; set by Open.
var DB *sql.DB

// Path is where the fork database lives: AI_TOOLKIT_GO_DB, else next to
// upstream's aitk_db.db.
func Path() string {
	if v := strings.TrimSpace(os.Getenv("AI_TOOLKIT_GO_DB")); v != "" {
		return v
	}
	return filepath.Join(paths.ToolkitRoot, "aitk_go.db")
}

// migrations run in order; append only, never edit a shipped one.
var migrations = []string{
	`CREATE TABLE events (
		id      INTEGER PRIMARY KEY AUTOINCREMENT,
		type    TEXT NOT NULL,
		job_id  TEXT,
		at      INTEGER NOT NULL,
		data    TEXT NOT NULL DEFAULT '{}'
	);
	CREATE INDEX events_job ON events (job_id, id);
	CREATE INDEX events_at ON events (at);

	CREATE TABLE api_keys (
		id           TEXT PRIMARY KEY,
		name         TEXT NOT NULL,
		prefix       TEXT NOT NULL,
		hash         TEXT NOT NULL UNIQUE,
		created_at   INTEGER NOT NULL,
		last_used_at INTEGER,
		revoked_at   INTEGER
	);

	CREATE TABLE job_attempts (
		job_id     TEXT NOT NULL,
		n          INTEGER NOT NULL,
		started_at INTEGER NOT NULL,
		ended_at   INTEGER,
		end_state  TEXT,
		end_reason TEXT,
		start_step INTEGER NOT NULL DEFAULT 0,
		end_step   INTEGER,
		auto       INTEGER NOT NULL DEFAULT 0,
		pid        INTEGER,
		PRIMARY KEY (job_id, n)
	);

	CREATE TABLE job_policies (
		job_id             TEXT PRIMARY KEY,
		auto_resume        INTEGER,
		max_auto_resumes   INTEGER,
		auto_upload_remote TEXT
	);

	CREATE TABLE remotes (
		id         TEXT PRIMARY KEY,
		name       TEXT NOT NULL UNIQUE,
		kind       TEXT NOT NULL,
		config     TEXT NOT NULL DEFAULT '{}',
		secret     TEXT NOT NULL DEFAULT '{}',
		created_at INTEGER NOT NULL
	);

	CREATE TABLE uploads (
		id         TEXT PRIMARY KEY,
		remote_id  TEXT NOT NULL,
		job_id     TEXT,
		root       TEXT NOT NULL,
		path       TEXT NOT NULL,
		url        TEXT,
		state      TEXT NOT NULL,
		bytes      INTEGER NOT NULL DEFAULT 0,
		bytes_done INTEGER NOT NULL DEFAULT 0,
		sha256     TEXT,
		error      TEXT,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL
	);
	CREATE INDEX uploads_job ON uploads (job_id);
	CREATE INDEX uploads_state ON uploads (state);

	CREATE TABLE file_hashes (
		path     TEXT PRIMARY KEY,
		size     INTEGER NOT NULL,
		mtime_ns INTEGER NOT NULL,
		sha256   TEXT NOT NULL
	);

	CREATE TABLE settings (
		key   TEXT PRIMARY KEY,
		value TEXT NOT NULL
	);`,
	// portable jobs: the remote a job's resumable state is synced to
	`ALTER TABLE job_policies ADD COLUMN sync_remote TEXT;`,
	// how many older checkpoints the sync remote keeps (-1: all)
	`ALTER TABLE job_policies ADD COLUMN keep_checkpoints INTEGER;`,
	// 4: GPU servers this server launched on RunPod / Vast.ai / QuickPod (keys: settings provision.key.<kind>)
	`CREATE TABLE machines (
	  id TEXT PRIMARY KEY, provider TEXT NOT NULL, provider_id TEXT NOT NULL DEFAULT '',
	  label TEXT NOT NULL, name TEXT NOT NULL, token TEXT NOT NULL,
	  status TEXT NOT NULL, problem TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT '',
	  provider_status TEXT NOT NULL DEFAULT '', pending TEXT NOT NULL DEFAULT '',
	  spec TEXT NOT NULL DEFAULT '{}', base_url TEXT NOT NULL DEFAULT '', endpoints TEXT NOT NULL DEFAULT '[]',
	  image_launch_ref TEXT NOT NULL DEFAULT '', image_digest TEXT NOT NULL DEFAULT '',
	  price_per_hour REAL, stopped_price_per_hour REAL, request_id TEXT,
	  simulated INTEGER NOT NULL DEFAULT 0,
	  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
	  running_at INTEGER, ready_at INTEGER, last_ok_at INTEGER, ended_at INTEGER);
	CREATE UNIQUE INDEX machines_request ON machines(request_id) WHERE request_id IS NOT NULL;
	CREATE UNIQUE INDEX machines_live_name ON machines(name) WHERE ended_at IS NULL;
	CREATE INDEX machines_provider ON machines(provider, ended_at);`,
	// 5: photo checks: per-content analysis (never per path); no face templates
	`CREATE TABLE photo_metrics (
		sha256  TEXT NOT NULL,
		version INTEGER NOT NULL,      -- photocheck.MetricsVersion
		data    TEXT NOT NULL,         -- Metrics JSON (file omitted)
		seen_at INTEGER NOT NULL,      -- unix ms; refreshed at most daily, batched
		PRIMARY KEY (sha256, version)
	);
	CREATE INDEX photo_metrics_seen ON photo_metrics (seen_at);
	CREATE TABLE photo_faces (
		sha256  TEXT NOT NULL,
		algo    INTEGER NOT NULL,      -- photocheck.FaceAlgo
		data    TEXT NOT NULL,         -- FaceData JSON: boxes, landmarks, scores, sharpness, light, text boxes
		seen_at INTEGER NOT NULL,
		PRIMARY KEY (sha256, algo)
	);
	CREATE INDEX photo_faces_seen ON photo_faces (seen_at);
	CREATE TABLE photo_sets (
		dir        TEXT PRIMARY KEY,   -- clean absolute dataset folder
		kind       TEXT,               -- person | style | object (NULL: person)
		reference  TEXT,               -- sha256 of the "this is them" photo
		decisions  TEXT NOT NULL DEFAULT '{}',  -- {sha256: "keep" | "intended"}
		derived    TEXT NOT NULL DEFAULT '{}',  -- {crop sha256: parent sha256}
		identity   TEXT,               -- Identity JSON: set_sig, model, job_id, at, faces, cos (f16 matrix)
		updated_at INTEGER NOT NULL
	);`,
}

// NodeID identifies this server (stable across restarts, new per database):
// portable-job leases name the node holding a job.
func NodeID(ctx context.Context) (string, error) {
	if v, ok, err := Setting(ctx, "node_id"); err != nil || ok {
		return v, err
	}
	id := NewID("node")
	if err := SetSetting(ctx, "node_id", id); err != nil {
		return "", err
	}
	return id, nil
}

// Open opens (creating if needed) the fork database and applies migrations.
func Open(path string) error {
	db, err := sql.Open("sqlite", store.FileURI(path)+"?_pragma=busy_timeout(30000)&_pragma=journal_mode(WAL)")
	if err != nil {
		return err
	}
	if err := db.Ping(); err != nil {
		db.Close()
		return fmt.Errorf("open %s: %w", path, err)
	}
	if err := migrate(context.Background(), db); err != nil {
		db.Close()
		return fmt.Errorf("migrate %s: %w", path, err)
	}
	// it holds API key hashes and remote credentials: owner-only (no-op on Windows)
	for _, p := range []string{path, path + "-wal", path + "-shm"} {
		os.Chmod(p, 0o600)
	}
	DB = db
	return nil
}

func migrate(ctx context.Context, db *sql.DB) error {
	if _, err := db.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)`); err != nil {
		return err
	}
	var version int
	err := db.QueryRowContext(ctx, `SELECT version FROM schema_version`).Scan(&version)
	if err == sql.ErrNoRows {
		if _, err := db.ExecContext(ctx, `INSERT INTO schema_version (version) VALUES (0)`); err != nil {
			return err
		}
	} else if err != nil {
		return err
	}
	for i := version; i < len(migrations); i++ {
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, migrations[i]); err != nil {
			tx.Rollback()
			return fmt.Errorf("migration %d: %w", i+1, err)
		}
		if _, err := tx.ExecContext(ctx, `UPDATE schema_version SET version = ?`, i+1); err != nil {
			tx.Rollback()
			return err
		}
		if err := tx.Commit(); err != nil {
			return err
		}
	}
	return nil
}

// NowMs is the current time in unix milliseconds (the storage format here).
func NowMs() int64 { return time.Now().UnixMilli() }

// Time converts stored milliseconds to time.Time (UTC).
func Time(ms int64) time.Time { return time.UnixMilli(ms).UTC() }

const base62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

// RandomString returns n random base62 characters.
func RandomString(n int) string {
	b := make([]byte, n)
	max := big.NewInt(int64(len(base62)))
	for i := range b {
		v, err := rand.Int(rand.Reader, max)
		if err != nil {
			panic(err)
		}
		b[i] = base62[v.Int64()]
	}
	return string(b)
}

// NewID returns "<prefix>_<16 base62>", e.g. NewID("rem").
func NewID(prefix string) string { return prefix + "_" + RandomString(16) }

// Setting reads a fork setting ("" and false when unset).
func Setting(ctx context.Context, key string) (string, bool, error) {
	var v string
	err := DB.QueryRowContext(ctx, `SELECT value FROM settings WHERE key = ?`, key).Scan(&v)
	if err == sql.ErrNoRows {
		return "", false, nil
	}
	return v, err == nil, err
}

// SetSetting writes a fork setting.
func SetSetting(ctx context.Context, key, value string) error {
	_, err := DB.ExecContext(ctx, `INSERT INTO settings (key, value) VALUES (?, ?)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, value)
	return err
}
