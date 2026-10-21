package remote

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/minio/minio-go/v7"
)

func TestS3ConfigValidation(t *testing.T) {
	sec := json.RawMessage(`{"access_key_id": "a", "secret_access_key": "b"}`)
	for _, bad := range []struct{ cfg, sec string }{
		{`{"bucket": "b"}`, string(sec)},
		{`{"endpoint": "https://x"}`, string(sec)},
		{`{"endpoint": "https://x", "bucket": "b"}`, `{}`},
		{`{"endpoint": "https://x/some/path", "bucket": "b"}`, string(sec)},
	} {
		if _, err := newS3(json.RawMessage(bad.cfg), json.RawMessage(bad.sec)); err == nil {
			t.Errorf("config %s / %s should be rejected", bad.cfg, bad.sec)
		}
	}
	c, err := newS3(json.RawMessage(`{"endpoint": "https://acct.r2.cloudflarestorage.com", "bucket": "loras", "prefix": "/runs/"}`), sec)
	if err != nil {
		t.Fatal(err)
	}
	sc := c.(*s3Client)
	if k, _ := sc.key("jobs/a/b.safetensors"); k != "runs/jobs/a/b.safetensors" {
		t.Fatalf("key = %q", k)
	}
	if sc.rel("runs/jobs/a") != "jobs/a" || sc.base != "https://acct.r2.cloudflarestorage.com/loras" {
		t.Fatalf("rel/base wrong: %q %q", sc.rel("runs/jobs/a"), sc.base)
	}
	if host, secure, _ := parseEndpoint("http://127.0.0.1:9000"); host != "127.0.0.1:9000" || secure {
		t.Fatalf("http endpoint parsed as %q secure=%v", host, secure)
	}
}

// minio starts a throwaway MinIO container and returns its endpoint.
func minio_(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("docker"); err != nil {
		t.Skip("docker not available")
	}
	if os.Getenv("AITK_SKIP_DOCKER_TESTS") != "" {
		t.Skip("AITK_SKIP_DOCKER_TESTS set")
	}
	name := fmt.Sprintf("aitk-minio-test-%d", time.Now().UnixNano())
	// minio/minio is no longer published; Chainguard's build runs the same
	// server. AITK_TEST_MINIO_IMAGE overrides.
	images := []string{"cgr.dev/chainguard/minio:latest", "minio/minio:latest"}
	if v := os.Getenv("AITK_TEST_MINIO_IMAGE"); v != "" {
		images = []string{v}
	}
	var started bool
	var lastOut []byte
	for _, img := range images {
		out, err := exec.Command("docker", "run", "-d", "--rm", "--name", name, "-p", "127.0.0.1::9000",
			"-e", "MINIO_ROOT_USER=aitkadmin", "-e", "MINIO_ROOT_PASSWORD=aitksecret123",
			img, "server", "/tmp/minio-data").CombinedOutput()
		if err == nil {
			started = true
			break
		}
		lastOut = out
	}
	if !started {
		t.Skipf("cannot start a minio container: %s", lastOut)
	}
	t.Cleanup(func() { exec.Command("docker", "rm", "-f", name).Run() })
	port, err := exec.Command("docker", "port", name, "9000/tcp").Output()
	if err != nil {
		t.Fatalf("docker port: %v", err)
	}
	hostPort := strings.TrimSpace(strings.Split(string(port), "\n")[0])
	endpoint := "http://" + hostPort
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		if resp, err := http.Get(endpoint + "/minio/health/ready"); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return endpoint
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatal("minio did not become ready")
	return ""
}

func TestS3AgainstMinIO(t *testing.T) {
	endpoint := minio_(t)
	ctx := context.Background()
	cfg, _ := json.Marshal(map[string]any{"endpoint": endpoint, "bucket": "aitk", "region": "us-east-1",
		"prefix": "runs", "path_style": true})
	sec := json.RawMessage(`{"access_key_id": "aitkadmin", "secret_access_key": "aitksecret123"}`)
	c, err := newS3(cfg, sec)
	if err != nil {
		t.Fatal(err)
	}
	sc := c.(*s3Client)

	if err := c.Check(ctx); err == nil || !strings.Contains(err.Error(), "does not exist") {
		t.Fatalf("Check without bucket = %v", err)
	}
	if err := sc.mc.MakeBucket(ctx, "aitk", minio.MakeBucketOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := c.Check(ctx); err != nil {
		t.Fatal(err)
	}

	// > 5 MiB forces a multipart upload (part size 16 MiB -> 2 parts)
	big := make([]byte, 21<<20)
	rand.Read(big)
	src := filepath.Join(t.TempDir(), "big.safetensors")
	os.WriteFile(src, big, 0o644)
	var progress atomic.Int64
	url, err := c.Put(ctx, "jobs/my job/big.safetensors", src, func(n int64) { progress.Store(n) })
	if err != nil {
		t.Fatal(err)
	}
	if progress.Load() != int64(len(big)) {
		t.Fatalf("progress ended at %d of %d", progress.Load(), len(big))
	}
	if !strings.HasSuffix(url, "/aitk/runs/jobs/my job/big.safetensors") {
		t.Fatalf("url = %q", url)
	}
	info, err := sc.mc.StatObject(ctx, "aitk", "runs/jobs/my job/big.safetensors", minio.StatObjectOptions{})
	if err != nil || !strings.Contains(info.ETag, "-") {
		t.Fatalf("expected a multipart ETag, got %q (%v)", info.ETag, err)
	}

	obj, err := c.Stat(ctx, "jobs/my job/big.safetensors")
	if err != nil || obj.Size != int64(len(big)) {
		t.Fatalf("Stat = %+v, %v", obj, err)
	}
	if _, err := c.Stat(ctx, "nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Stat missing = %v", err)
	}

	dst := filepath.Join(t.TempDir(), "d", "big.safetensors")
	var got atomic.Int64
	if err := c.Get(ctx, "jobs/my job/big.safetensors", dst, func(n int64) { got.Store(n) }); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(dst); !bytes.Equal(b, big) || got.Load() != int64(len(big)) {
		t.Fatalf("Get content mismatch (progress %d)", got.Load())
	}
	if err := c.Get(ctx, "missing", filepath.Join(t.TempDir(), "m"), nil); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get missing = %v", err)
	}

	if err := c.PutBytes(ctx, "jobs/my job/manifest.json", []byte(`{"step":20}`)); err != nil {
		t.Fatal(err)
	}
	if b, err := c.GetBytes(ctx, "jobs/my job/manifest.json"); err != nil || string(b) != `{"step":20}` {
		t.Fatalf("GetBytes = %q, %v", b, err)
	}
	if _, err := c.GetBytes(ctx, "missing.json"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetBytes missing = %v", err)
	}

	list, err := c.List(ctx, "jobs/")
	if err != nil || len(list) != 2 || list[0].Key != "jobs/my job/big.safetensors" || list[1].Key != "jobs/my job/manifest.json" {
		t.Fatalf("List = %+v, %v", list, err)
	}
	if all, _ := c.List(ctx, ""); len(all) != 2 {
		t.Fatalf("List all = %+v", all)
	}

	if err := c.Delete(ctx, "jobs/my job/big.safetensors"); err != nil {
		t.Fatal(err)
	}
	if err := c.Delete(ctx, "jobs/my job/big.safetensors"); err != nil {
		t.Fatalf("delete missing: %v", err)
	}
	if list, _ := c.List(ctx, ""); len(list) != 1 {
		t.Fatalf("after delete: %+v", list)
	}

	bad, _ := newS3(cfg, json.RawMessage(`{"access_key_id": "aitkadmin", "secret_access_key": "wrong-secret"}`))
	if err := bad.Check(ctx); err == nil {
		t.Fatal("Check with bad credentials should fail")
	}
}
