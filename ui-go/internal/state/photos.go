package state

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Photo checks cache their analysis by content (sha256), never by path, so a
// renamed or set-aside photo keeps it. Paths map to content through
// file_hashes, the same cache the job data and portable sync use. The data
// columns hold JSON owned by internal/photocheck; this file only stores it.

// touchAge is how stale seen_at must be before TouchPhotos rewrites it: at
// most one write per photo per day, not one per poll.
const touchAge = 24 * time.Hour

// shaChunk caps the parameters of one IN (...) list.
const shaChunk = 500

// hashKeyPath is the file_hashes key for path: absolute and clean, matching
// portable sync's rows.
func hashKeyPath(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		return abs
	}
	return path
}

// LookupFileSHA256 returns the cached sha256 of path when its size and mtime
// still match the cached row; false when there is none (nothing is hashed).
// info may be nil, then the file is stat'ed.
func LookupFileSHA256(ctx context.Context, path string, info fs.FileInfo) (string, bool, error) {
	if info == nil {
		st, err := os.Stat(path)
		if err != nil {
			return "", false, err
		}
		info = st
	}
	var sum string
	err := DB.QueryRowContext(ctx, `SELECT sha256 FROM file_hashes WHERE path = ? AND size = ? AND mtime_ns = ?`,
		hashKeyPath(path), info.Size(), info.ModTime().UnixNano()).Scan(&sum)
	if err == sql.ErrNoRows {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return sum, true, nil
}

// FileSHA256 returns the sha256 of path, cached in file_hashes by path, size
// and mtime so an unchanged photo is hashed once. info may be nil, then the
// file is stat'ed.
func FileSHA256(ctx context.Context, path string, info fs.FileInfo) (string, error) {
	if info == nil {
		st, err := os.Stat(path)
		if err != nil {
			return "", err
		}
		info = st
	}
	if sum, ok, err := LookupFileSHA256(ctx, path, info); err != nil || ok {
		return sum, err
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	sum := hex.EncodeToString(h.Sum(nil))
	_, err = DB.ExecContext(ctx, `INSERT INTO file_hashes (path, size, mtime_ns, sha256) VALUES (?, ?, ?, ?)
		ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime_ns = excluded.mtime_ns, sha256 = excluded.sha256`,
		hashKeyPath(path), info.Size(), info.ModTime().UnixNano(), sum)
	return sum, err
}

// inChunks calls fn with shas split into lists of at most shaChunk, each with
// its "?, ?, …" placeholder string.
func inChunks(shas []string, fn func(marks string, args []any) error) error {
	for start := 0; start < len(shas); start += shaChunk {
		end := min(start+shaChunk, len(shas))
		args := make([]any, 0, end-start)
		for _, s := range shas[start:end] {
			args = append(args, s)
		}
		if err := fn(strings.TrimSuffix(strings.Repeat("?, ", end-start), ", "), args); err != nil {
			return err
		}
	}
	return nil
}

// getByContent reads data by sha from table (photo_metrics or photo_faces)
// for one key value (version or algo).
func getByContent(ctx context.Context, table, keyCol string, shas []string, key int) (map[string][]byte, error) {
	out := map[string][]byte{}
	err := inChunks(shas, func(marks string, args []any) error {
		rows, err := DB.QueryContext(ctx, `SELECT sha256, data FROM `+table+` WHERE `+keyCol+` = ? AND sha256 IN (`+marks+`)`,
			append([]any{key}, args...)...)
		if err != nil {
			return err
		}
		defer rows.Close()
		for rows.Next() {
			var sha, data string
			if err := rows.Scan(&sha, &data); err != nil {
				return err
			}
			out[sha] = []byte(data)
		}
		return rows.Err()
	})
	return out, err
}

// GetPhotoMetrics returns the stored Metrics JSON by sha256 for one metrics
// version. A sha with no row (or only rows of another version) is absent.
func GetPhotoMetrics(ctx context.Context, shas []string, version int) (map[string][]byte, error) {
	return getByContent(ctx, "photo_metrics", "version", shas, version)
}

// PutPhotoMetrics stores (or replaces) one photo's Metrics JSON and marks it seen.
func PutPhotoMetrics(ctx context.Context, sha string, version int, data []byte) error {
	_, err := DB.ExecContext(ctx, `INSERT INTO photo_metrics (sha256, version, data, seen_at) VALUES (?, ?, ?, ?)
		ON CONFLICT(sha256, version) DO UPDATE SET data = excluded.data, seen_at = excluded.seen_at`,
		sha, version, string(data), NowMs())
	return err
}

// GetPhotoFaces returns the stored FaceData JSON by sha256 for one face algo.
func GetPhotoFaces(ctx context.Context, shas []string, algo int) (map[string][]byte, error) {
	return getByContent(ctx, "photo_faces", "algo", shas, algo)
}

// PutPhotoFaces stores (or replaces) the FaceData JSON of many photos in one
// transaction (one face job's ingest) and marks them seen.
func PutPhotoFaces(ctx context.Context, rows map[string][]byte, algo int) error {
	tx, err := DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	now := NowMs()
	for sha, data := range rows {
		if _, err := tx.ExecContext(ctx, `INSERT INTO photo_faces (sha256, algo, data, seen_at) VALUES (?, ?, ?, ?)
			ON CONFLICT(sha256, algo) DO UPDATE SET data = excluded.data, seen_at = excluded.seen_at`,
			sha, algo, string(data), now); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// TouchPhotos marks the photos' metrics and face rows seen at now, but only
// rows last seen more than a day before now, so a review polled every second
// writes at most once a day per photo.
func TouchPhotos(ctx context.Context, shas []string, now time.Time) error {
	at, cutoff := now.UnixMilli(), now.Add(-touchAge).UnixMilli()
	return inChunks(shas, func(marks string, args []any) error {
		for _, table := range []string{"photo_metrics", "photo_faces"} {
			if _, err := DB.ExecContext(ctx, `UPDATE `+table+` SET seen_at = ? WHERE sha256 IN (`+marks+`) AND seen_at < ?`,
				append(append([]any{at}, args...), cutoff)...); err != nil {
				return err
			}
		}
		return nil
	})
}

// PhotoSet is a dataset folder's photo-check state (photo_sets). Decisions
// map sha256 to "keep" or "intended"; Derived maps a face crop's sha256 to its
// parent's. Identity is the Identity JSON of the last face check, nil when
// none. Kind and Reference are "" when unset.
type PhotoSet struct {
	Dir       string
	Kind      string
	Reference string
	Decisions map[string]string
	Derived   map[string]string
	Identity  []byte
	UpdatedAt int64
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// GetPhotoSet returns the folder's photo-check state, or nil when none.
func GetPhotoSet(ctx context.Context, dir string) (*PhotoSet, error) {
	s := PhotoSet{Dir: filepath.Clean(dir)}
	var kind, reference, identity sql.NullString
	var decisions, derived string
	err := DB.QueryRowContext(ctx, `SELECT kind, reference, decisions, derived, identity, updated_at FROM photo_sets WHERE dir = ?`, s.Dir).
		Scan(&kind, &reference, &decisions, &derived, &identity, &s.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	s.Kind, s.Reference = kind.String, reference.String
	if identity.Valid {
		s.Identity = []byte(identity.String)
	}
	if err := json.Unmarshal([]byte(decisions), &s.Decisions); err != nil {
		return nil, err
	}
	if err := json.Unmarshal([]byte(derived), &s.Derived); err != nil {
		return nil, err
	}
	if s.Decisions == nil {
		s.Decisions = map[string]string{}
	}
	if s.Derived == nil {
		s.Derived = map[string]string{}
	}
	return &s, nil
}

// PutPhotoSet replaces the folder's photo-check state. A zero UpdatedAt is
// set to now (and written back to s).
func PutPhotoSet(ctx context.Context, s *PhotoSet) error {
	if s.UpdatedAt == 0 {
		s.UpdatedAt = NowMs()
	}
	decisions, err := json.Marshal(nonNilMap(s.Decisions))
	if err != nil {
		return err
	}
	derived, err := json.Marshal(nonNilMap(s.Derived))
	if err != nil {
		return err
	}
	var identity any
	if s.Identity != nil {
		identity = string(s.Identity)
	}
	_, err = DB.ExecContext(ctx, `INSERT INTO photo_sets (dir, kind, reference, decisions, derived, identity, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(dir) DO UPDATE SET kind = excluded.kind, reference = excluded.reference,
		decisions = excluded.decisions, derived = excluded.derived, identity = excluded.identity, updated_at = excluded.updated_at`,
		filepath.Clean(s.Dir), nullIfEmpty(s.Kind), nullIfEmpty(s.Reference), string(decisions), string(derived), identity, s.UpdatedAt)
	return err
}

func nonNilMap(m map[string]string) map[string]string {
	if m == nil {
		return map[string]string{}
	}
	return m
}

// DeletePhotoSet drops the folder's photo-check state (no error when none).
func DeletePhotoSet(ctx context.Context, dir string) error {
	_, err := DB.ExecContext(ctx, `DELETE FROM photo_sets WHERE dir = ?`, filepath.Clean(dir))
	return err
}

// ForgetPhotos removes a dataset's analysis: the metrics and face rows of
// shas (every version and algo; pass the set-aside photos' too) and the
// folder's photo_sets row, in one transaction.
func ForgetPhotos(ctx context.Context, dir string, shas []string) error {
	tx, err := DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	err = inChunks(shas, func(marks string, args []any) error {
		for _, table := range []string{"photo_metrics", "photo_faces"} {
			if _, err := tx.ExecContext(ctx, `DELETE FROM `+table+` WHERE sha256 IN (`+marks+`)`, args...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM photo_sets WHERE dir = ?`, filepath.Clean(dir)); err != nil {
		return err
	}
	return tx.Commit()
}

// GCPhotos deletes metrics and face rows last seen before olderThan and
// returns how many went. photo_sets rows are the caller's to prune (by
// PhotoSetDirs, for folders that no longer exist).
func GCPhotos(ctx context.Context, olderThan time.Time) (int64, error) {
	cutoff := olderThan.UnixMilli()
	var total int64
	for _, table := range []string{"photo_metrics", "photo_faces"} {
		res, err := DB.ExecContext(ctx, `DELETE FROM `+table+` WHERE seen_at < ?`, cutoff)
		if err != nil {
			return total, err
		}
		n, _ := res.RowsAffected()
		total += n
	}
	return total, nil
}

// PhotoSetDirs lists every folder with photo-check state, sorted.
func PhotoSetDirs(ctx context.Context) ([]string, error) {
	rows, err := DB.QueryContext(ctx, `SELECT dir FROM photo_sets`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []string{}
	for rows.Next() {
		var d string
		if err := rows.Scan(&d); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}
