package state

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
)

// latest is the schema version this round ships (#4 machines, #5 photo checks).
const latest = 5

// openState opens path through Open and closes it when the test ends.
func openState(t *testing.T, path string) {
	t.Helper()
	if err := Open(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { DB.Close() })
}

// buildAt writes a database that has run only the first n migrations, the
// way an older server left it.
func buildAt(t *testing.T, path string, n int) {
	t.Helper()
	db, err := sql.Open("sqlite", store.FileURI(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	if _, err := db.ExecContext(ctx, `CREATE TABLE schema_version (version INTEGER NOT NULL)`); err != nil {
		t.Fatal(err)
	}
	for _, m := range migrations[:n] {
		if _, err := db.ExecContext(ctx, m); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.ExecContext(ctx, `INSERT INTO schema_version (version) VALUES (?)`, n); err != nil {
		t.Fatal(err)
	}
}

func schemaVersion(t *testing.T) int {
	t.Helper()
	var v int
	if err := DB.QueryRow(`SELECT version FROM schema_version`).Scan(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

func hasTable(t *testing.T, name string) bool {
	t.Helper()
	var n int
	if err := DB.QueryRow(`SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?`, name).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n == 1
}

// insertMachine adds a minimal machine row; requestID and endedAt may be nil.
func insertMachine(id, name string, requestID, endedAt any) error {
	_, err := DB.Exec(`INSERT INTO machines (id, provider, label, name, token, status, request_id, created_at, updated_at, ended_at)
		VALUES (?, 'runpod', 'GPU server', ?, 'tok', 'starting', ?, 1, 1, ?)`, id, name, requestID, endedAt)
	return err
}

func TestFreshDBReachesLatest(t *testing.T) {
	if len(migrations) != latest {
		t.Fatalf("migrations = %d, want %d", len(migrations), latest)
	}
	openState(t, filepath.Join(testutil.TempDir(t), "aitk_go.db"))
	if v := schemaVersion(t); v != latest {
		t.Fatalf("version = %d, want %d", v, latest)
	}
	for _, name := range []string{"machines", "photo_metrics", "photo_faces", "photo_sets"} {
		if !hasTable(t, name) {
			t.Errorf("table %s missing", name)
		}
	}
}

func TestMigrateFromVersion3(t *testing.T) {
	path := filepath.Join(testutil.TempDir(t), "aitk_go.db")
	buildAt(t, path, 3)
	db, err := sql.Open("sqlite", store.FileURI(path))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO job_policies (job_id, sync_remote, keep_checkpoints) VALUES ('j1', 'rem_a', 2)`); err != nil {
		t.Fatal(err)
	}
	db.Close()

	openState(t, path)
	if v := schemaVersion(t); v != latest {
		t.Fatalf("version = %d, want %d", v, latest)
	}
	var remote string
	var keep int
	if err := DB.QueryRow(`SELECT sync_remote, keep_checkpoints FROM job_policies WHERE job_id = 'j1'`).Scan(&remote, &keep); err != nil {
		t.Fatal(err)
	}
	if remote != "rem_a" || keep != 2 {
		t.Fatalf("job policy = %q %d, want rem_a 2", remote, keep)
	}
	for _, name := range []string{"machines", "photo_metrics", "photo_faces", "photo_sets"} {
		if !hasTable(t, name) {
			t.Errorf("table %s missing", name)
		}
	}
}

func TestMachineDefaults(t *testing.T) {
	openState(t, filepath.Join(testutil.TempDir(t), "aitk_go.db"))
	if err := insertMachine("m1", "trainer", nil, nil); err != nil {
		t.Fatal(err)
	}
	var spec, endpoints, launchRef, digest, problem string
	var simulated int
	var price sql.NullFloat64
	if err := DB.QueryRow(`SELECT spec, endpoints, image_launch_ref, image_digest, problem, simulated, price_per_hour FROM machines WHERE id = 'm1'`).
		Scan(&spec, &endpoints, &launchRef, &digest, &problem, &simulated, &price); err != nil {
		t.Fatal(err)
	}
	if spec != "{}" || endpoints != "[]" || launchRef != "" || digest != "" || problem != "" || simulated != 0 || price.Valid {
		t.Fatalf("defaults = %q %q %q %q %q %d %v", spec, endpoints, launchRef, digest, problem, simulated, price)
	}
	if _, err := DB.Exec(`UPDATE machines SET image_launch_ref = 'ghcr.io/geocine/aitoolkit-go@sha256:ab', image_digest = 'sha256:ab' WHERE id = 'm1'`); err != nil {
		t.Fatal(err)
	}
}

func TestMachineLiveNameUnique(t *testing.T) {
	openState(t, filepath.Join(testutil.TempDir(t), "aitk_go.db"))
	if err := insertMachine("m1", "trainer", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := insertMachine("m2", "trainer", nil, nil); err == nil {
		t.Fatal("two live machines share a name")
	}
	if _, err := DB.Exec(`UPDATE machines SET ended_at = 2 WHERE id = 'm1'`); err != nil {
		t.Fatal(err)
	}
	if err := insertMachine("m2", "trainer", nil, nil); err != nil {
		t.Fatalf("name not reusable after the first ended: %v", err)
	}
	if err := insertMachine("m3", "trainer", nil, int64(3)); err != nil {
		t.Fatalf("ended rows may share a name: %v", err)
	}
	if err := insertMachine("m4", "trainer", nil, nil); err == nil {
		t.Fatal("a second live machine took a live name")
	}
}

func TestMachineRequestUnique(t *testing.T) {
	openState(t, filepath.Join(testutil.TempDir(t), "aitk_go.db"))
	if err := insertMachine("m1", "a", "req_1", nil); err != nil {
		t.Fatal(err)
	}
	if err := insertMachine("m2", "b", "req_1", nil); err == nil {
		t.Fatal("duplicate request_id accepted")
	}
	if err := insertMachine("m3", "c", "req_1", int64(2)); err == nil {
		t.Fatal("duplicate request_id accepted on an ended row")
	}
	// rows without a request id never collide
	if err := insertMachine("m4", "d", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := insertMachine("m5", "e", nil, nil); err != nil {
		t.Fatal(err)
	}
}

func TestPhotoTables(t *testing.T) {
	openState(t, filepath.Join(testutil.TempDir(t), "aitk_go.db"))
	sha := "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
	if _, err := DB.Exec(`INSERT INTO photo_metrics (sha256, version, data, seen_at) VALUES (?, 1, ?, ?)`,
		sha, `{"width":1024,"height":1024,"sharpness":0.8}`, NowMs()); err != nil {
		t.Fatal(err)
	}
	if _, err := DB.Exec(`INSERT INTO photo_metrics (sha256, version, data, seen_at) VALUES (?, 1, '{}', 1)`, sha); err == nil {
		t.Fatal("duplicate (sha256, version) accepted")
	}
	// a metrics version bump is a new row, not a conflict
	if _, err := DB.Exec(`INSERT INTO photo_metrics (sha256, version, data, seen_at) VALUES (?, 2, '{}', 1)`, sha); err != nil {
		t.Fatal(err)
	}
	if _, err := DB.Exec(`INSERT INTO photo_faces (sha256, algo, data, seen_at) VALUES (?, 1, ?, ?)`,
		sha, `{"faces":[{"box":[10,20,200,240],"score":0.98}]}`, NowMs()); err != nil {
		t.Fatal(err)
	}
	if _, err := DB.Exec(`INSERT INTO photo_faces (sha256, algo, data, seen_at) VALUES (?, 1, '{}', 1)`, sha); err == nil {
		t.Fatal("duplicate (sha256, algo) accepted")
	}

	dir := filepath.Join(testutil.TempDir(t), "datasets", "ana")
	if _, err := DB.Exec(`INSERT INTO photo_sets (dir, updated_at) VALUES (?, ?)`, dir, NowMs()); err != nil {
		t.Fatal(err)
	}
	var kind, reference, identity sql.NullString
	var decisions, derived string
	if err := DB.QueryRow(`SELECT kind, reference, decisions, derived, identity FROM photo_sets WHERE dir = ?`, dir).
		Scan(&kind, &reference, &decisions, &derived, &identity); err != nil {
		t.Fatal(err)
	}
	if kind.Valid || reference.Valid || identity.Valid || decisions != "{}" || derived != "{}" {
		t.Fatalf("photo_sets defaults = %v %v %q %q %v", kind, reference, decisions, derived, identity)
	}
	if _, err := DB.Exec(`UPDATE photo_sets SET kind = 'person', reference = ?, decisions = ?, derived = ?, identity = ? WHERE dir = ?`,
		sha, `{"`+sha+`":"keep"}`, `{"abc":"`+sha+`"}`, `{"set_sig":"s","model":"m","faces":1}`, dir); err != nil {
		t.Fatal(err)
	}

	// the seen_at indexes exist for the daily touch and the GC
	for _, idx := range []string{"photo_metrics_seen", "photo_faces_seen"} {
		var n int
		if err := DB.QueryRow(`SELECT count(*) FROM sqlite_master WHERE type = 'index' AND name = ?`, idx).Scan(&n); err != nil {
			t.Fatal(err)
		}
		if n != 1 {
			t.Errorf("index %s missing", idx)
		}
	}
	res, err := DB.Exec(`DELETE FROM photo_metrics WHERE seen_at < ?`, 2)
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		t.Fatalf("gc removed %d metrics rows, want 1", n)
	}
}
