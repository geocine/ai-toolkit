package store

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/testutil"
)

func closeDB(t *testing.T) {
	t.Cleanup(func() {
		if DB != nil {
			DB.Close()
			DB = nil
		}
	})
}

func TestCreate(t *testing.T) {
	path := filepath.Join(testutil.TempDir(t), "aitk_db.db")
	closeDB(t) // after TempDir: cleanups run last first
	if err := Create(path); err != nil {
		t.Fatal(err)
	}
	if err := Open(path); err != nil {
		t.Fatalf("a created database fails CheckSchema: %v", err)
	}
	if err := UpsertSetting(context.Background(), "TRAINING_FOLDER", "/x"); err != nil {
		t.Fatal(err)
	}
	if err := Create(path); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("Create over an existing file = %v", err)
	}
}

func TestRecreate(t *testing.T) {
	dir := testutil.TempDir(t)
	closeDB(t)
	path := filepath.Join(dir, "aitk_db.db")
	// an older schema: Job lacks a column the code now reads
	ddl := strings.Replace(SchemaSQL, `, "save_now" BOOLEAN NOT NULL DEFAULT false`, "", 1)
	if ddl == SchemaSQL {
		t.Fatal("the test's column is no longer in SchemaSQL")
	}
	db, err := sql.Open("sqlite", FileURI(path))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(ddl); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO "Settings" ("key", "value") VALUES ('a', '1'), ('b', '')`); err != nil {
		t.Fatal(err)
	}
	db.Close()

	err = Open(path)
	if !IsOutOfDate(err) {
		t.Fatalf("Open on the old schema = %v", err)
	}
	backup, err := Recreate(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(filepath.Base(backup), "aitk_db.db.bak-") {
		t.Fatalf("backup = %q", backup)
	}
	if baks, _ := filepath.Glob(filepath.Join(dir, "aitk_db.db.bak-*")); len(baks) == 0 {
		t.Fatal("no backup file left")
	}
	if err := Open(path); err != nil {
		t.Fatalf("after Recreate: %v", err)
	}
	ctx := context.Background()
	for k, want := range map[string]string{"a": "1", "b": ""} {
		got, ok, err := GetSetting(ctx, k)
		if err != nil || !ok || got != want {
			t.Errorf("setting %s = %q, %v, %v; want %q", k, got, ok, err, want)
		}
	}
}

func TestIsOutOfDate(t *testing.T) {
	if IsOutOfDate(nil) || IsOutOfDate(sql.ErrNoRows) {
		t.Fatal("IsOutOfDate matches other errors")
	}
}
