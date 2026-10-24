package store

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// PrismaSchema is a verbatim copy of upstream's ui/prisma/schema.prisma.
// parity/schema_embed_test.go fails when upstream changes it.
//
//go:embed schema/schema.prisma
var PrismaSchema string

// SchemaSQL is the DDL `npx prisma db push` creates for PrismaSchema.
// parity/schema_embed_test.go checks its columns and indexes against it.
//
//go:embed schema/schema.sql
var SchemaSQL string

// Create makes a new upstream database at path from SchemaSQL. It refuses an
// existing file: an existing database belongs to prisma db push.
func Create(path string) error {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("database %s already exists", path)
		}
		return err
	}
	f.Close()
	db, err := sql.Open("sqlite", FileURI(path))
	if err != nil {
		os.Remove(path)
		return err
	}
	_, err = db.Exec(SchemaSQL)
	db.Close()
	if err != nil {
		os.Remove(path)
		return fmt.Errorf("create %s: %w", path, err)
	}
	return nil
}

// IsOutOfDate reports whether err is CheckSchema's "missing columns" error,
// the one Recreate answers.
func IsOutOfDate(err error) bool {
	return err != nil && strings.HasPrefix(err.Error(), "database schema is out of date")
}

// Recreate replaces the database at path, made by an older studio API whose
// embedded schema has since followed upstream: it keeps the Settings rows (the
// only data the studio API keeps there), renames the file to
// <path>.bak-<unix>, creates a fresh one from SchemaSQL and puts the rows
// back. Studio API only: a database next to a checkout belongs to
// prisma db push. It closes DB first (the file is renamed); call Open after.
func Recreate(path string) (backup string, err error) {
	if DB != nil {
		DB.Close()
		DB = nil
	}
	rows, err := settingsRows(path)
	if err != nil {
		return "", err
	}
	backup = path + ".bak-" + strconv.FormatInt(time.Now().Unix(), 10)
	if err := os.Rename(path, backup); err != nil {
		return "", err
	}
	// a leftover WAL must not be replayed into the new file
	for _, side := range []string{"-wal", "-shm"} {
		if _, err := os.Stat(path + side); err == nil {
			if err := os.Rename(path+side, backup+side); err != nil {
				return backup, err
			}
		}
	}
	if err := Create(path); err != nil {
		return backup, err
	}
	db, err := sql.Open("sqlite", FileURI(path))
	if err != nil {
		return backup, err
	}
	defer db.Close()
	ctx := context.Background()
	for _, r := range rows {
		if _, err := db.ExecContext(ctx, `INSERT INTO "Settings" ("key", "value") VALUES (?, ?)`, r[0], r[1]); err != nil {
			return backup, fmt.Errorf("restore setting %s: %w", r[0], err)
		}
	}
	return backup, nil
}

// settingsRows reads key/value pairs from the Settings table at path; a
// database without a readable Settings table has none.
func settingsRows(path string) ([][2]string, error) {
	db, err := sql.Open("sqlite", FileURI(path)+"?_pragma=busy_timeout(30000)")
	if err != nil {
		return nil, err
	}
	defer db.Close()
	if err := db.Ping(); err != nil {
		return nil, err
	}
	// fold the WAL into the file, so the renamed backup is whole
	db.Exec(`PRAGMA wal_checkpoint(TRUNCATE)`)
	q, err := db.Query(`SELECT "key", "value" FROM "Settings"`)
	if err != nil {
		return nil, nil
	}
	defer q.Close()
	var out [][2]string
	for q.Next() {
		var k, v sql.NullString
		if err := q.Scan(&k, &v); err != nil {
			return nil, err
		}
		if k.Valid {
			out = append(out, [2]string{k.String, v.String})
		}
	}
	return out, q.Err()
}
