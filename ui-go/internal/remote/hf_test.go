package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeHub is a stand-in huggingface_hub package backed by a folder, so the
// real helper script and the Go plumbing run end to end without the network.
const fakeHub = `
import datetime, os, shutil

class EntryNotFoundError(Exception): pass
class RepositoryNotFoundError(Exception): pass

class _Commit:
    def __init__(self, date): self.date = date

class _File:
    def __init__(self, path, size, mtime):
        self.path, self.size = path, size
        self.last_commit = _Commit(datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc))

class _Folder:
    def __init__(self, path): self.path = path

ROOT = os.environ["FAKE_HF_ROOT"]

class HfApi:
    endpoint = "https://huggingface.co"
    def __init__(self, token=None):
        with open(os.path.join(ROOT, "seen_token"), "w") as f: f.write(token or "")
    def _repo(self, repo_id, repo_type):
        return os.path.join(ROOT, repo_type, repo_id.replace("/", "__"))
    def create_repo(self, repo_id, repo_type, private, exist_ok):
        p = self._repo(repo_id, repo_type)
        os.makedirs(p, exist_ok=True)
        with open(p + ".private", "w") as f: f.write(str(private))
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, revision, commit_message):
        dst = os.path.join(self._repo(repo_id, repo_type), path_in_repo)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if isinstance(path_or_fileobj, bytes):
            with open(dst, "wb") as f: f.write(path_or_fileobj)
        else:
            shutil.copyfile(path_or_fileobj, dst)
    def delete_file(self, path_in_repo, repo_id, repo_type, revision, commit_message):
        p = os.path.join(self._repo(repo_id, repo_type), path_in_repo)
        if not os.path.exists(p): raise EntryNotFoundError(path_in_repo)
        os.remove(p)
    def hf_hub_download(self, repo_id, filename, repo_type, revision, local_dir):
        src = os.path.join(self._repo(repo_id, repo_type), filename)
        if not os.path.exists(src): raise EntryNotFoundError(filename)
        dst = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return dst
    def get_paths_info(self, repo_id, paths, repo_type, revision, expand):
        out = []
        for p in paths:
            full = os.path.join(self._repo(repo_id, repo_type), p)
            if os.path.isfile(full): out.append(_File(p, os.path.getsize(full), os.path.getmtime(full)))
        return out
    def list_repo_tree(self, repo_id, path_in_repo, recursive, repo_type, revision, expand):
        base = self._repo(repo_id, repo_type)
        start = os.path.join(base, path_in_repo) if path_in_repo else base
        if not os.path.isdir(start): raise EntryNotFoundError(path_in_repo)
        for d, dirs, files in os.walk(start):
            for name in dirs: yield _Folder(os.path.relpath(os.path.join(d, name), base).replace(os.sep, "/"))
            for name in files:
                full = os.path.join(d, name)
                yield _File(os.path.relpath(full, base).replace(os.sep, "/"), os.path.getsize(full), os.path.getmtime(full))
`

func pythonOrSkip(t *testing.T) string {
	t.Helper()
	py := hfPython()
	if _, err := exec.LookPath(py); err != nil {
		t.Skipf("python not available (%s)", py)
	}
	return py
}

func fakeHubEnv(t *testing.T) (repoDir string) {
	t.Helper()
	pythonOrSkip(t)
	mod := t.TempDir()
	pkg := filepath.Join(mod, "huggingface_hub")
	os.MkdirAll(pkg, 0o755)
	if err := os.WriteFile(filepath.Join(pkg, "__init__.py"), []byte(fakeHub), 0o644); err != nil {
		t.Fatal(err)
	}
	data := t.TempDir()
	hfExtraEnv = []string{"PYTHONPATH=" + mod, "FAKE_HF_ROOT=" + data}
	t.Cleanup(func() { hfExtraEnv = nil })
	return filepath.Join(data, "model", "me__loras")
}

func newTestHF(t *testing.T, config string) Client {
	t.Helper()
	c, err := newHF(json.RawMessage(config), json.RawMessage(`{"token": "hf_secret_test"}`))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestHFConfigValidation(t *testing.T) {
	for _, bad := range []string{`{}`, `{"repo_id": "noslash"}`, `{"repo_id": "a/b", "repo_type": "space"}`} {
		if _, err := newHF(json.RawMessage(bad), nil); err == nil {
			t.Errorf("config %s should be rejected", bad)
		}
	}
	c, err := newHF(json.RawMessage(`{"repo_id": "me/x", "path_in_repo": "/runs/a/"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	hc := c.(*hfClient)
	req := hc.request("put")
	if req.RepoType != "model" || req.Revision != "main" || !req.Private {
		t.Fatalf("defaults wrong: %+v", req)
	}
	if k, _ := hc.key("jobs/x/a.safetensors"); k != "runs/a/jobs/x/a.safetensors" {
		t.Fatalf("key = %q", k)
	}
	if _, err := hc.key("../.."); err == nil {
		t.Fatal("escaping key should fail")
	}
	if k, _ := hc.key("../../etc/passwd"); k != "runs/a/etc/passwd" {
		t.Fatalf("traversal not contained: %q", k)
	}
	pub := false
	hc.cfg.Private = &pub
	if hc.request("check").Private {
		t.Fatal("private=false not passed")
	}
}

func TestHFHelperEndToEndWithFakeHub(t *testing.T) {
	repo := fakeHubEnv(t)
	ctx := context.Background()
	c := newTestHF(t, `{"repo_id": "me/loras", "path_in_repo": "runs"}`)

	if err := c.Check(ctx); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(repo + ".private"); string(b) != "True" {
		t.Fatalf("repo should be created private, got %q", b)
	}
	if tok, _ := os.ReadFile(filepath.Join(filepath.Dir(filepath.Dir(repo)), "seen_token")); string(tok) != "hf_secret_test" {
		t.Fatalf("token not passed via env: %q", tok)
	}

	src := filepath.Join(t.TempDir(), "w.safetensors")
	os.WriteFile(src, bytes.Repeat([]byte("x"), 1234), 0o644)
	var got int64
	url, err := c.Put(ctx, "jobs/my job/w.safetensors", src, func(n int64) { got = n })
	if err != nil {
		t.Fatal(err)
	}
	if url != "https://huggingface.co/me/loras/resolve/main/runs/jobs/my job/w.safetensors" || got != 1234 {
		t.Fatalf("url %q progress %d", url, got)
	}
	if err := c.PutBytes(ctx, "jobs/my job/manifest.json", []byte(`{"a":1}`)); err != nil {
		t.Fatal(err)
	}
	if b, err := c.GetBytes(ctx, "jobs/my job/manifest.json"); err != nil || string(b) != `{"a":1}` {
		t.Fatalf("GetBytes = %q, %v", b, err)
	}
	obj, err := c.Stat(ctx, "jobs/my job/w.safetensors")
	if err != nil || obj.Size != 1234 || obj.Modified.IsZero() {
		t.Fatalf("Stat = %+v, %v", obj, err)
	}
	if _, err := c.Stat(ctx, "nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Stat missing = %v", err)
	}
	list, err := c.List(ctx, "jobs/")
	if err != nil || len(list) != 2 || list[0].Key != "jobs/my job/manifest.json" || list[1].Key != "jobs/my job/w.safetensors" {
		t.Fatalf("List = %+v, %v", list, err)
	}
	dst := filepath.Join(t.TempDir(), "sub", "out.safetensors")
	if err := c.Get(ctx, "jobs/my job/w.safetensors", dst, nil); err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(dst); err != nil || st.Size() != 1234 {
		t.Fatalf("Get wrote %v, %v", st, err)
	}
	if err := c.Get(ctx, "missing.bin", dst, nil); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get missing = %v", err)
	}
	if err := c.Delete(ctx, "jobs/my job/w.safetensors"); err != nil {
		t.Fatal(err)
	}
	if err := c.Delete(ctx, "jobs/my job/w.safetensors"); err != nil {
		t.Fatalf("deleting a missing key should be fine: %v", err)
	}
	if list, _ := c.List(ctx, ""); len(list) != 1 {
		t.Fatalf("after delete: %+v", list)
	}
}

func TestHFHelperReportsMissingLibrary(t *testing.T) {
	pythonOrSkip(t)
	// an empty PYTHONPATH dir with a broken package shadows any real install
	mod := t.TempDir()
	os.MkdirAll(filepath.Join(mod, "huggingface_hub"), 0o755)
	os.WriteFile(filepath.Join(mod, "huggingface_hub", "__init__.py"), []byte("raise ImportError('no hub here')\n"), 0o644)
	hfExtraEnv = []string{"PYTHONPATH=" + mod}
	defer func() { hfExtraEnv = nil }()
	err := newTestHF(t, `{"repo_id": "me/x"}`).Check(context.Background())
	if err == nil || !strings.Contains(err.Error(), "huggingface_hub is not installed") {
		t.Fatalf("err = %v", err)
	}
}

// Real Hugging Face round trip; runs only with AITK_TEST_HF_TOKEN and
// AITK_TEST_HF_REPO (a repo the token may write; it is created private).
func TestHFIntegration(t *testing.T) {
	token, repo := os.Getenv("AITK_TEST_HF_TOKEN"), os.Getenv("AITK_TEST_HF_REPO")
	if token == "" || repo == "" {
		t.Skip("set AITK_TEST_HF_TOKEN and AITK_TEST_HF_REPO to run against the real hub")
	}
	cfg, _ := json.Marshal(map[string]any{"repo_id": repo, "path_in_repo": "aitk-test"})
	sec, _ := json.Marshal(map[string]string{"token": token})
	c, err := newHF(cfg, sec)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	if err := c.Check(ctx); err != nil {
		t.Fatal(err)
	}
	key := "probe-" + time.Now().Format("150405") + ".txt"
	if err := c.PutBytes(ctx, key, []byte("hello")); err != nil {
		t.Fatal(err)
	}
	if b, err := c.GetBytes(ctx, key); err != nil || string(b) != "hello" {
		t.Fatalf("GetBytes = %q, %v", b, err)
	}
	if err := c.Delete(ctx, key); err != nil {
		t.Fatal(err)
	}
}
