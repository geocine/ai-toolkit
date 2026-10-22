package links

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func TestSignAndVerify(t *testing.T) {
	dir := testutil.TempDir(t)
	if err := state.Open(filepath.Join(dir, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close(); state.DB = nil; Reset() })
	Reset()
	ctx := context.Background()
	signed, exp, err := Sign(ctx, "/v1/files/content?root=training&path=job/a%20b.safetensors", time.Hour)
	if err != nil || time.Until(exp) < 59*time.Minute {
		t.Fatal(err, exp)
	}
	ok := func(method, target string) bool { return Valid(httptest.NewRequest(method, target, nil)) }
	if !ok("GET", signed) || !ok("HEAD", signed) || ok("DELETE", signed) {
		t.Fatal("method handling")
	}
	if ok("GET", strings.Replace(signed, "root=training", "root=models", 1)) || ok("GET", signed+"&x=1") {
		t.Fatal("a changed query still verifies")
	}
	// the key survives a restart (kept in the fork database)
	Reset()
	if !ok("GET", signed) {
		t.Fatal("key not persisted")
	}
	// expired
	past := strings.Replace(signed, "exp="+strconv.FormatInt(exp.Unix(), 10), "exp="+strconv.FormatInt(time.Now().Add(-time.Minute).Unix(), 10), 1)
	if ok("GET", past) {
		t.Fatal("expired link verifies")
	}
	for _, bad := range []string{"/v1/jobs", "/v1/keys", "https://evil.example/v1/files/content", "relative"} {
		if _, _, err := Sign(ctx, bad, time.Hour); err == nil {
			t.Errorf("signed %s", bad)
		}
	}
	if _, _, err := Sign(ctx, "/v1/jobs/abc/bundle?checkpoints=all", time.Hour); err != nil {
		t.Fatal(err)
	}
	if !ok("GET", must(Sign(ctx, "/api/img/%2Ftmp%2Fa.png", time.Hour))) {
		t.Fatal("media route link")
	}
	r := httptest.NewRequest("GET", "/x", nil)
	r.Host = "inner:8675"
	r.Header.Set("X-Forwarded-Proto", "https")
	r.Header.Set("X-Forwarded-Host", "pod.proxy.example")
	if o := Origin(r, ""); o != "https://pod.proxy.example" {
		t.Fatal(o)
	}
	if o := Origin(r, "https://public.example/"); o != "https://public.example" {
		t.Fatal(o)
	}
	_ = http.MethodGet
}

func must(s string, _ time.Time, err error) string {
	if err != nil {
		panic(err)
	}
	return s
}
