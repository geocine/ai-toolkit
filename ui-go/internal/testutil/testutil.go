// Package testutil holds helpers shared by tests.
package testutil

import (
	"os"
	"testing"
	"time"
)

// TempDir is t.TempDir for tests that open SQLite databases: on Windows the
// -wal/-shm files can stay locked for a moment after Close, which makes the
// built-in cleanup fail the test, so removal is retried briefly instead.
func TempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "aitk-test-*")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		var err error
		for i := 0; i < 50; i++ {
			if err = os.RemoveAll(dir); err == nil {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
		t.Logf("could not remove %s: %v", dir, err)
	})
	return dir
}
