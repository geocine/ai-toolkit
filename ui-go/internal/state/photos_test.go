package state

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/testutil"
)

func openPhotoDB(t *testing.T) context.Context {
	t.Helper()
	openState(t, filepath.Join(testutil.TempDir(t), "aitk_go.db"))
	return context.Background()
}

func fakeSHA(i int) string {
	h := sha256.Sum256([]byte(fmt.Sprint(i)))
	return hex.EncodeToString(h[:])
}

func seenAt(t *testing.T, table, sha string) int64 {
	t.Helper()
	var v int64
	if err := DB.QueryRow(`SELECT seen_at FROM `+table+` WHERE sha256 = ?`, sha).Scan(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

func setSeen(t *testing.T, table, sha string, at int64) {
	t.Helper()
	if _, err := DB.Exec(`UPDATE `+table+` SET seen_at = ? WHERE sha256 = ?`, at, sha); err != nil {
		t.Fatal(err)
	}
}

func TestPhotoCacheMetrics(t *testing.T) {
	ctx := openPhotoDB(t)
	a, b, c := fakeSHA(1), fakeSHA(2), fakeSHA(3)
	if err := PutPhotoMetrics(ctx, a, 1, []byte(`{"w":1}`)); err != nil {
		t.Fatal(err)
	}
	if err := PutPhotoMetrics(ctx, b, 1, []byte(`{"w":2}`)); err != nil {
		t.Fatal(err)
	}
	got, err := GetPhotoMetrics(ctx, []string{a, b, c}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || string(got[a]) != `{"w":1}` || string(got[b]) != `{"w":2}` {
		t.Fatalf("get = %q", got)
	}
	if _, ok := got[c]; ok {
		t.Fatal("unknown sha returned")
	}
	// a put replaces the row
	if err := PutPhotoMetrics(ctx, a, 1, []byte(`{"w":3}`)); err != nil {
		t.Fatal(err)
	}
	got, _ = GetPhotoMetrics(ctx, []string{a}, 1)
	if string(got[a]) != `{"w":3}` {
		t.Fatalf("replaced = %q", got[a])
	}
	// a metrics version bump misses, so the photo is measured again
	got, err = GetPhotoMetrics(ctx, []string{a, b}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("version 2 = %q, want none", got)
	}
	got, _ = GetPhotoMetrics(ctx, nil, 1)
	if got == nil || len(got) != 0 {
		t.Fatalf("no shas = %v, want an empty map", got)
	}
}

func TestPhotoCacheManyShas(t *testing.T) {
	ctx := openPhotoDB(t)
	var shas []string
	rows := map[string][]byte{}
	for i := range 1234 {
		s := fakeSHA(i)
		shas = append(shas, s)
		if i%2 == 0 {
			if err := PutPhotoMetrics(ctx, s, 1, []byte(`{}`)); err != nil {
				t.Fatal(err)
			}
			rows[s] = []byte(`{"faces":[]}`)
		}
	}
	if err := PutPhotoFaces(ctx, rows, 1); err != nil {
		t.Fatal(err)
	}
	m, err := GetPhotoMetrics(ctx, shas, 1)
	if err != nil {
		t.Fatal(err)
	}
	f, err := GetPhotoFaces(ctx, shas, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 617 || len(f) != 617 {
		t.Fatalf("metrics %d faces %d, want 617 each", len(m), len(f))
	}
	later := time.Now().Add(48 * time.Hour)
	if err := TouchPhotos(ctx, shas, later); err != nil {
		t.Fatal(err)
	}
	// one row from each of the three 500-sha chunks
	for _, i := range []int{0, 600, 1232} {
		for _, table := range []string{"photo_metrics", "photo_faces"} {
			if v := seenAt(t, table, shas[i]); v != later.UnixMilli() {
				t.Errorf("%s: sha %d seen_at = %d, want %d", table, i, v, later.UnixMilli())
			}
		}
	}
	if err := ForgetPhotos(ctx, "/data/x", shas); err != nil {
		t.Fatal(err)
	}
	m, _ = GetPhotoMetrics(ctx, shas, 1)
	f, _ = GetPhotoFaces(ctx, shas, 1)
	if len(m) != 0 || len(f) != 0 {
		t.Fatalf("after forget: metrics %d faces %d", len(m), len(f))
	}
}

func TestPhotoCacheFaces(t *testing.T) {
	ctx := openPhotoDB(t)
	a, b := fakeSHA(1), fakeSHA(2)
	if err := PutPhotoFaces(ctx, map[string][]byte{a: []byte(`{"faces":[1]}`), b: []byte(`{"faces":[]}`)}, 1); err != nil {
		t.Fatal(err)
	}
	if err := PutPhotoFaces(ctx, map[string][]byte{a: []byte(`{"faces":[2]}`)}, 1); err != nil {
		t.Fatal(err)
	}
	got, err := GetPhotoFaces(ctx, []string{a, b}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if string(got[a]) != `{"faces":[2]}` || string(got[b]) != `{"faces":[]}` {
		t.Fatalf("faces = %q", got)
	}
	if got, _ := GetPhotoFaces(ctx, []string{a}, 2); len(got) != 0 {
		t.Fatalf("algo 2 = %q, want none", got)
	}
	if err := PutPhotoFaces(ctx, nil, 1); err != nil {
		t.Fatalf("empty ingest: %v", err)
	}
}

func TestPhotoCacheTouch(t *testing.T) {
	ctx := openPhotoDB(t)
	fresh, stale := fakeSHA(1), fakeSHA(2)
	for _, s := range []string{fresh, stale} {
		if err := PutPhotoMetrics(ctx, s, 1, []byte(`{}`)); err != nil {
			t.Fatal(err)
		}
		if err := PutPhotoFaces(ctx, map[string][]byte{s: []byte(`{}`)}, 1); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Now()
	recent := now.Add(-23 * time.Hour).UnixMilli()
	old := now.Add(-25 * time.Hour).UnixMilli()
	for _, table := range []string{"photo_metrics", "photo_faces"} {
		setSeen(t, table, fresh, recent)
		setSeen(t, table, stale, old)
	}
	if err := TouchPhotos(ctx, []string{fresh, stale}, now); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"photo_metrics", "photo_faces"} {
		if v := seenAt(t, table, fresh); v != recent {
			t.Errorf("%s: a row seen 23 h ago was rewritten (%d, want %d)", table, v, recent)
		}
		if v := seenAt(t, table, stale); v != now.UnixMilli() {
			t.Errorf("%s: a row seen 25 h ago = %d, want %d", table, v, now.UnixMilli())
		}
	}
	if err := TouchPhotos(ctx, nil, now); err != nil {
		t.Fatal(err)
	}
}

func TestPhotoCacheForget(t *testing.T) {
	ctx := openPhotoDB(t)
	mine, asideSHA, other := fakeSHA(1), fakeSHA(2), fakeSHA(3)
	for _, s := range []string{mine, asideSHA, other} {
		if err := PutPhotoMetrics(ctx, s, 1, []byte(`{}`)); err != nil {
			t.Fatal(err)
		}
		if err := PutPhotoMetrics(ctx, s, 2, []byte(`{}`)); err != nil {
			t.Fatal(err)
		}
		if err := PutPhotoFaces(ctx, map[string][]byte{s: []byte(`{}`)}, 1); err != nil {
			t.Fatal(err)
		}
	}
	root := testutil.TempDir(t)
	dir, keep := filepath.Join(root, "ana"), filepath.Join(root, "bo")
	for _, d := range []string{dir, keep} {
		if err := PutPhotoSet(ctx, &PhotoSet{Dir: d, Kind: "person"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := ForgetPhotos(ctx, dir, []string{mine, asideSHA}); err != nil {
		t.Fatal(err)
	}
	for _, v := range []int{1, 2} {
		got, _ := GetPhotoMetrics(ctx, []string{mine, asideSHA, other}, v)
		if len(got) != 1 || got[other] == nil {
			t.Fatalf("version %d after forget = %q, want only the other dataset's row", v, got)
		}
	}
	if got, _ := GetPhotoFaces(ctx, []string{mine, asideSHA, other}, 1); len(got) != 1 || got[other] == nil {
		t.Fatalf("faces after forget = %q", got)
	}
	if s, err := GetPhotoSet(ctx, dir); err != nil || s != nil {
		t.Fatalf("forgotten set = %+v, %v", s, err)
	}
	if s, err := GetPhotoSet(ctx, keep); err != nil || s == nil {
		t.Fatalf("other set = %+v, %v", s, err)
	}
}

func TestPhotoCacheGC(t *testing.T) {
	ctx := openPhotoDB(t)
	young, old := fakeSHA(1), fakeSHA(2)
	for _, s := range []string{young, old} {
		if err := PutPhotoMetrics(ctx, s, 1, []byte(`{}`)); err != nil {
			t.Fatal(err)
		}
		if err := PutPhotoFaces(ctx, map[string][]byte{s: []byte(`{}`)}, 1); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Now()
	for _, table := range []string{"photo_metrics", "photo_faces"} {
		setSeen(t, table, young, now.Add(-89*24*time.Hour).UnixMilli())
		setSeen(t, table, old, now.Add(-91*24*time.Hour).UnixMilli())
	}
	set := filepath.Join(testutil.TempDir(t), "ana")
	if err := PutPhotoSet(ctx, &PhotoSet{Dir: set, UpdatedAt: 1}); err != nil {
		t.Fatal(err)
	}
	n, err := GCPhotos(ctx, now.Add(-90*24*time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("gc removed %d rows, want 2", n)
	}
	if got, _ := GetPhotoMetrics(ctx, []string{young, old}, 1); len(got) != 1 || got[young] == nil {
		t.Fatalf("metrics after gc = %q", got)
	}
	if got, _ := GetPhotoFaces(ctx, []string{young, old}, 1); len(got) != 1 || got[young] == nil {
		t.Fatalf("faces after gc = %q", got)
	}
	// photo_sets are pruned by folder, not by age
	if s, _ := GetPhotoSet(ctx, set); s == nil {
		t.Fatal("gc dropped a photo set")
	}
}

func TestPhotoSet(t *testing.T) {
	ctx := openPhotoDB(t)
	root := testutil.TempDir(t)
	dir := filepath.Join(root, "ana")
	if s, err := GetPhotoSet(ctx, dir); err != nil || s != nil {
		t.Fatalf("missing set = %+v, %v", s, err)
	}

	// an empty set stores NULLs and empty maps, and gets a time
	s := &PhotoSet{Dir: dir}
	if err := PutPhotoSet(ctx, s); err != nil {
		t.Fatal(err)
	}
	if s.UpdatedAt == 0 {
		t.Fatal("UpdatedAt not set")
	}
	got, err := GetPhotoSet(ctx, dir)
	if err != nil || got == nil {
		t.Fatalf("get = %+v, %v", got, err)
	}
	if got.Kind != "" || got.Reference != "" || got.Identity != nil || len(got.Decisions) != 0 || len(got.Derived) != 0 ||
		got.Decisions == nil || got.Derived == nil || got.UpdatedAt != s.UpdatedAt {
		t.Fatalf("empty set = %+v", got)
	}
	var kindNull, refNull, idNull bool
	if err := DB.QueryRow(`SELECT kind IS NULL, reference IS NULL, identity IS NULL FROM photo_sets WHERE dir = ?`, dir).
		Scan(&kindNull, &refNull, &idNull); err != nil {
		t.Fatal(err)
	}
	if !kindNull || !refNull || !idNull {
		t.Fatalf("NULL columns = %v %v %v", kindNull, refNull, idNull)
	}

	// a full set round-trips and replaces the old row
	a, crop := fakeSHA(1), fakeSHA(2)
	full := &PhotoSet{
		Dir: dir + string(filepath.Separator), Kind: "style", Reference: a,
		Decisions: map[string]string{a: "keep", crop: "intended"},
		Derived:   map[string]string{crop: a},
		Identity:  []byte(`{"set_sig":"s1","model":"sface","faces":["` + a + `:0"],"cos":"AAA="}`),
		UpdatedAt: 42,
	}
	if err := PutPhotoSet(ctx, full); err != nil {
		t.Fatal(err)
	}
	got, err = GetPhotoSet(ctx, dir)
	if err != nil || got == nil {
		t.Fatalf("get = %+v, %v", got, err)
	}
	if got.Dir != dir || got.Kind != "style" || got.Reference != a || got.Decisions[a] != "keep" ||
		got.Decisions[crop] != "intended" || got.Derived[crop] != a || string(got.Identity) != string(full.Identity) || got.UpdatedAt != 42 {
		t.Fatalf("full set = %+v", got)
	}

	// clearing a field writes NULL again
	got.Reference, got.Identity, got.UpdatedAt = "", nil, 0
	if err := PutPhotoSet(ctx, got); err != nil {
		t.Fatal(err)
	}
	again, _ := GetPhotoSet(ctx, dir)
	if again.Reference != "" || again.Identity != nil || again.Kind != "style" {
		t.Fatalf("cleared set = %+v", again)
	}

	other := filepath.Join(root, "bo")
	if err := PutPhotoSet(ctx, &PhotoSet{Dir: other}); err != nil {
		t.Fatal(err)
	}
	dirs, err := PhotoSetDirs(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(dirs) != 2 || dirs[0] != dir || dirs[1] != other {
		t.Fatalf("dirs = %v", dirs)
	}
	if err := DeletePhotoSet(ctx, dir); err != nil {
		t.Fatal(err)
	}
	if err := DeletePhotoSet(ctx, dir); err != nil {
		t.Fatalf("second delete: %v", err)
	}
	if dirs, _ := PhotoSetDirs(ctx); len(dirs) != 1 || dirs[0] != other {
		t.Fatalf("dirs after delete = %v", dirs)
	}
}

func TestPhotoFileSHA256(t *testing.T) {
	ctx := openPhotoDB(t)
	p := filepath.Join(testutil.TempDir(t), "IMG_1.jpg")
	if err := os.WriteFile(p, []byte("one"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := sha256.Sum256([]byte("one"))
	if _, ok, err := LookupFileSHA256(ctx, p, nil); err != nil || ok {
		t.Fatalf("lookup before hashing = %v, %v; want a miss", ok, err)
	}
	sum, err := FileSHA256(ctx, p, nil)
	if err != nil || sum != hex.EncodeToString(want[:]) {
		t.Fatalf("sha = %s, %v", sum, err)
	}
	st, _ := os.Stat(p)
	if got, ok, err := LookupFileSHA256(ctx, p, st); err != nil || !ok || got != sum {
		t.Fatalf("lookup after hashing = %s %v %v", got, ok, err)
	}

	// the cache is keyed by size and mtime: a cached row answers without
	// reading the file, and a changed file is hashed again
	if _, err := DB.Exec(`UPDATE file_hashes SET sha256 = 'cached' WHERE path = ?`, p); err != nil {
		t.Fatal(err)
	}
	if got, _ := FileSHA256(ctx, p, st); got != "cached" {
		t.Fatalf("cached row not used: %s", got)
	}
	if err := os.WriteFile(p, []byte("two!"), 0o644); err != nil {
		t.Fatal(err)
	}
	later := st.ModTime().Add(time.Second)
	if err := os.Chtimes(p, later, later); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := LookupFileSHA256(ctx, p, nil); ok {
		t.Fatal("lookup hit for a changed file")
	}
	want = sha256.Sum256([]byte("two!"))
	if got, err := FileSHA256(ctx, p, nil); err != nil || got != hex.EncodeToString(want[:]) {
		t.Fatalf("changed file sha = %s, %v", got, err)
	}
	var n int
	if err := DB.QueryRow(`SELECT count(*) FROM file_hashes`).Scan(&n); err != nil || n != 1 {
		t.Fatalf("file_hashes rows = %d, %v; want one per path", n, err)
	}

	if _, err := FileSHA256(ctx, filepath.Join(filepath.Dir(p), "gone.jpg"), nil); err == nil {
		t.Fatal("missing file hashed")
	}
}
