// Package store reads and writes the toolkit's SQLite database (aitk_db.db).
//
// The schema belongs to upstream: ui/prisma/schema.prisma, applied with
// `npx prisma db push` from ui/. This package never creates or alters tables;
// it only checks at startup that every column it uses exists. Values are
// stored the way Prisma stores them so the Next UI, the Python trainer and
// this server can share one file: DateTime as unix milliseconds, Boolean as
// 0/1.
//
// The studio API (internal/studioapi) is the one exception: it creates a
// fresh database from the embedded copy of the schema (schema/), kept equal
// to upstream by parity/schema_embed_test.go.
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite"
)

var (
	ErrNotFound        = errors.New("record not found")
	ErrUniqueViolation = errors.New("unique constraint failed")
)

// DB is the shared handle; set by Open.
var DB *sql.DB

const defaultJournalMode = "WAL"

var validJournalModes = []string{"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"}

// Open opens the database at path, applies the journal mode the Next worker
// would (AI_TOOLKIT_DB_JOURNAL_MODE, default WAL) and verifies the schema.
func Open(path string) error {
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("database %s not found: %w (create it with `npx prisma db push` in ui/)", path, err)
	}
	dsn := FileURI(path) + "?_pragma=busy_timeout(30000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return err
	}
	if err := db.Ping(); err != nil {
		db.Close()
		return err
	}
	DB = db
	ensureJournalMode(context.Background())
	return CheckSchema(context.Background())
}

// FileURI is path as a sqlite "file:" URI, with the characters a URI would
// read as syntax (%, ?, #) percent-escaped.
func FileURI(path string) string {
	return "file:" + strings.NewReplacer("%", "%25", "?", "%3f", "#", "%23").Replace(filepath.ToSlash(path))
}

func ensureJournalMode(ctx context.Context) {
	envMode := os.Getenv("AI_TOOLKIT_DB_JOURNAL_MODE")
	target := strings.ToUpper(envMode)
	if target == "" {
		target = defaultJournalMode
	}
	valid := false
	for _, m := range validJournalModes {
		if m == target {
			valid = true
		}
	}
	if !valid {
		log.Printf("Invalid AI_TOOLKIT_DB_JOURNAL_MODE %q, expected one of %s. Using %s.",
			envMode, strings.Join(validJournalModes, ", "), defaultJournalMode)
		target = defaultJournalMode
	}
	var current string
	if err := DB.QueryRowContext(ctx, "PRAGMA journal_mode;").Scan(&current); err != nil {
		log.Printf("Could not check database journal mode: %v", err)
		return
	}
	if strings.ToUpper(current) == target {
		return
	}
	log.Printf("Converting database journal mode from %s to %s...", strings.ToUpper(current), target)
	var result string
	// target is validated against validJournalModes above, safe to interpolate
	if err := DB.QueryRowContext(ctx, "PRAGMA journal_mode = "+target+";").Scan(&result); err != nil {
		log.Printf("Could not convert database journal mode: %v", err)
		return
	}
	if strings.ToUpper(result) == target {
		log.Printf("Database journal mode is now %s.", strings.ToUpper(result))
	} else {
		log.Printf("Could not convert database journal mode to %s, still using %s.", target, strings.ToUpper(result))
	}
}

// Tables lists, per table, every column this package reads or writes. The
// parity tests compare it against ui/prisma/schema.prisma after each rebase.
var Tables = map[string][]string{
	"Job":      jobColumns,
	"Queue":    queueColumns,
	"Settings": settingsColumns,
}

// CheckSchema fails when a table or column this package needs is missing,
// e.g. an older database that upstream's prisma db push has not migrated yet.
func CheckSchema(ctx context.Context) error {
	var missing []string
	for table, cols := range Tables {
		have := map[string]bool{}
		rows, err := DB.QueryContext(ctx, `SELECT name FROM pragma_table_info(?)`, table)
		if err != nil {
			return err
		}
		for rows.Next() {
			var name string
			if err := rows.Scan(&name); err != nil {
				rows.Close()
				return err
			}
			have[name] = true
		}
		rows.Close()
		if len(have) == 0 {
			missing = append(missing, table)
			continue
		}
		for _, c := range cols {
			if !have[c] {
				missing = append(missing, table+"."+c)
			}
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("database schema is out of date, missing %s (run `npx prisma db push` in ui/)", strings.Join(missing, ", "))
	}
	return nil
}

// Fields is a set of column updates. Keys must be real column names from
// this package's own code, never user input; they are checked anyway.
type Fields map[string]any

func buildSet(table string, f Fields) (string, []any, error) {
	allowed := map[string]bool{}
	for _, c := range Tables[table] {
		allowed[c] = true
	}
	parts := make([]string, 0, len(f))
	args := make([]any, 0, len(f))
	for col, v := range f {
		if !allowed[col] {
			return "", nil, fmt.Errorf("store: unknown column %s.%s", table, col)
		}
		parts = append(parts, `"`+col+`" = ?`)
		args = append(args, toDB(v))
	}
	return strings.Join(parts, ", "), args, nil
}

// toDB converts Go values to what Prisma stores.
func toDB(v any) any {
	switch x := v.(type) {
	case bool:
		if x {
			return int64(1)
		}
		return int64(0)
	case *int:
		if x == nil {
			return nil
		}
		return int64(*x)
	case *string:
		if x == nil {
			return nil
		}
		return *x
	case Time:
		return x.UnixMilli()
	}
	return v
}

func mapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, sql.ErrNoRows) {
		return ErrNotFound
	}
	if strings.Contains(err.Error(), "UNIQUE constraint failed") {
		return fmt.Errorf("%w: %v", ErrUniqueViolation, err)
	}
	return err
}

func quoteCols(cols []string) string {
	q := make([]string, len(cols))
	for i, c := range cols {
		q[i] = `"` + c + `"`
	}
	return strings.Join(q, ", ")
}
