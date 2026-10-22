package v1

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/links"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/store"
)

func stFile(step int) []byte {
	info, _ := json.Marshal(map[string]int{"step": step})
	head, _ := json.Marshal(map[string]any{"__metadata__": map[string]string{"training_info": string(info)}})
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(head)))
	b.Write(head)
	return b.Bytes()
}

func put(t *testing.T, p string, data []byte) {
	t.Helper()
	os.MkdirAll(filepath.Dir(p), 0o755)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// waitTask polls an import or push until it settles.
func (e *env) waitTask(t *testing.T, kind, id string) map[string]any {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for {
		m := e.call(t, "GET", "/v1/bundles/"+kind+"/"+id, nil).json(t)
		if m["state"] == "done" || m["state"] == "failed" {
			tasksWG.Wait() // its goroutine may still emit the final event
			return m
		}
		if time.Now().After(deadline) {
			t.Fatalf("task %s still %v", id, m["state"])
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func jobByName(t *testing.T, name string) *store.Job {
	t.Helper()
	j, err := store.FindFirstJob(context.Background(), store.JobQuery{Where: `"name" = ?`, Args: []any{name}})
	if err != nil || j == nil {
		t.Fatalf("job %s: %v", name, err)
	}
	return j
}

// a job "run" training on dataset "set", with two saves and optimizer state
func bundleFixture(t *testing.T, e *env) string {
	t.Helper()
	ds := filepath.Join(e.datasets, "set")
	put(t, filepath.Join(ds, "a.png"), pngBytes(4, 3))
	put(t, filepath.Join(ds, "a.txt"), []byte("a caption"))
	put(t, filepath.Join(ds, "_latent_cache", "a_123.safetensors"), []byte("latent"))
	cfg := map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
		"type": "sd_trainer", "training_folder": e.training,
		"datasets": []any{map[string]any{"folder_path": ds, "caption_ext": "txt"}},
		"train":    map[string]any{"steps": 1000},
		"model":    map[string]any{"name_or_path": "org/base-model", "arch": "sd15"},
	}}}, "meta": map[string]any{"name": "run"}}
	r := e.call(t, "POST", "/v1/jobs", map[string]any{"name": "run", "gpu_ids": "0", "config": cfg})
	want(t, r, 201, "")
	folder := filepath.Join(e.training, "run")
	put(t, filepath.Join(folder, "run_000000100.safetensors"), stFile(100))
	time.Sleep(20 * time.Millisecond)
	put(t, filepath.Join(folder, "run_000000200.safetensors"), stFile(200))
	put(t, filepath.Join(folder, "optimizer.pt"), []byte("optimizer state"))
	put(t, filepath.Join(folder, "samples", "1__000000200_0.jpg"), []byte("sample"))
	put(t, filepath.Join(folder, "log.txt"), []byte("log line\n"))
	put(t, filepath.Join(folder, "config.yaml"), []byte("job: extension\n"))
	put(t, filepath.Join(folder, ".job_config.json"), []byte(`{"old":true}`))
	return r.json(t)["id"].(string)
}

func (e *env) upload(t *testing.T, data []byte, chunk int) string {
	t.Helper()
	r := e.call(t, "POST", "/v1/bundles/uploads", map[string]any{"size": len(data), "filename": "b.zip"})
	want(t, r, 201, "")
	id := r.json(t)["id"].(string)
	for off := 0; off < len(data); off += chunk {
		end := min(off+chunk, len(data))
		hdr := map[string]string{"Content-Range": fmt.Sprintf("bytes %d-%d/%d", off, end-1, len(data))}
		want(t, e.do(t, "PUT", "/v1/bundles/uploads/"+id, bytes.NewReader(data[off:end]), hdr), 200, "")
		if off == 0 {
			// a retried chunk is harmless
			want(t, e.do(t, "PUT", "/v1/bundles/uploads/"+id, bytes.NewReader(data[off:end]), hdr), 200, "")
		}
	}
	m := e.call(t, "GET", "/v1/bundles/uploads/"+id, nil).json(t)
	if m["complete"] != true || m["received"] != float64(len(data)) {
		t.Fatalf("upload: %v", m)
	}
	return id
}

func TestBundleExportImport(t *testing.T) {
	e := setup(t)
	id := bundleFixture(t, e)

	r := e.do(t, "GET", "/v1/jobs/"+id+"/bundle", nil, nil)
	want(t, r, 200, "")
	bundle := r.body
	if r.hdr.Get("Content-Length") != strconv.Itoa(len(bundle)) || !strings.Contains(r.hdr.Get("Content-Disposition"), "run_step200.zip") {
		t.Fatalf("headers: %v (%d bytes)", r.hdr, len(bundle))
	}
	if h := e.do(t, "HEAD", "/v1/jobs/"+id+"/bundle", nil, nil); h.code != 200 || h.hdr.Get("Content-Length") != strconv.Itoa(len(bundle)) {
		t.Fatalf("HEAD: %d %v", h.code, h.hdr)
	}
	zr, err := zip.NewReader(bytes.NewReader(bundle), int64(len(bundle)))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range zr.File {
		names = append(names, f.Name)
	}
	all := strings.Join(names, ",")
	if names[0] != "aitk_bundle.json" || !strings.HasSuffix(all, "output/run/run_000000200.safetensors") ||
		strings.Contains(all, "run_000000100") || !strings.Contains(all, "data/_latent_cache/") || strings.Contains(all, ".job_config") {
		t.Fatalf("entries: %s", all)
	}
	want(t, e.do(t, "GET", "/v1/jobs/"+id+"/bundle?checkpoints=some", nil, nil), 400, "invalid_request")

	// chunked upload, imported under another name; the identical dataset is reused
	up := e.upload(t, bundle, len(bundle)/3+1)
	want(t, e.do(t, "PUT", "/v1/bundles/uploads/"+up, strings.NewReader("x"), map[string]string{"Content-Range": "bytes 0-0/1"}), 400, "invalid_request")
	r = e.call(t, "POST", "/v1/bundles/imports", map[string]any{"upload_id": up, "name": "run2", "gpu_ids": "1"})
	want(t, r, 202, "")
	task := e.waitTask(t, "imports", r.json(t)["id"].(string))
	if task["state"] != "done" || task["dataset"] != "set" || task["job_name"] != "run2" || task["upload_id"] != nil {
		t.Fatalf("import: %v", task)
	}
	j := jobByName(t, "run2")
	if j.Step != 200 || j.GpuIDs != "1" || j.TotalSteps == nil || *j.TotalSteps != 1000 {
		t.Fatalf("job: %+v", j)
	}
	cfg, _ := jsonx.ParseObject([]byte(j.JobConfig))
	p0 := cfg.Object("config").Array("process")[0].(*jsonx.Object)
	fp, _ := p0.Array("datasets")[0].(*jsonx.Object).Get("folder_path")
	tf, _ := p0.Get("training_folder")
	cname, _ := cfg.Object("config").Get("name")
	if fp != filepath.Join(e.datasets, "set") || tf != e.training || cname != "run2" {
		t.Fatalf("config: %s", j.JobConfig)
	}
	folder := filepath.Join(e.training, "run2")
	for _, f := range []string{"run2_000000200.safetensors", "optimizer.pt", "samples/1__000000200_0.jpg", "log.txt", ".job_config.json"} {
		if _, err := os.Stat(filepath.Join(folder, filepath.FromSlash(f))); err != nil {
			t.Errorf("restored folder lacks %s", f)
		}
	}
	if jc, _ := os.ReadFile(filepath.Join(folder, ".job_config.json")); !strings.Contains(string(jc), `"run2"`) {
		t.Errorf(".job_config.json not regenerated: %s", jc)
	}
	if r := e.call(t, "GET", "/v1/bundles/uploads/"+up, nil); r.code != 404 {
		t.Fatalf("upload kept after a successful import: %d", r.code)
	}

	// the same bundle again under its own name: the job exists (409 at once)
	r = e.do(t, "POST", "/v1/bundles/imports", bytes.NewReader(bundle), map[string]string{"Content-Type": "application/zip"})
	want(t, r, 409, "conflict")

	// a dataset of that name with other files: dataset_exists, then another name works
	put(t, filepath.Join(e.datasets, "set", "a.txt"), []byte("other caption"))
	r = e.do(t, "POST", "/v1/bundles/imports?name=run3", bytes.NewReader(bundle), map[string]string{"Content-Type": "application/zip"})
	want(t, r, 409, "dataset_exists")
	r = e.do(t, "POST", "/v1/bundles/imports?name=run3&dataset=set_b&start=true", bytes.NewReader(bundle), map[string]string{"Content-Type": "application/zip"})
	want(t, r, 202, "")
	if task := e.waitTask(t, "imports", r.json(t)["id"].(string)); task["state"] != "done" || task["dataset"] != "set_b" {
		t.Fatalf("import 3: %v", task)
	}
	if b, _ := os.ReadFile(filepath.Join(e.datasets, "set_b", "a.txt")); string(b) != "a caption" {
		t.Fatalf("set_b caption %q", b)
	}
	if _, err := os.Stat(filepath.Join(e.datasets, "set_b", "_latent_cache", "a_123.safetensors")); err != nil {
		t.Fatal("dataset caches not restored")
	}
	if j := jobByName(t, "run3"); j.Status != "queued" {
		t.Fatalf("start: status %s", j.Status)
	}

	// by URL
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(bundle) }))
	defer srv.Close()
	r = e.call(t, "POST", "/v1/bundles/imports", map[string]any{"url": srv.URL + "/b.zip?sig=secret", "name": "run4", "dataset": "set_c"})
	want(t, r, 202, "")
	task = e.waitTask(t, "imports", r.json(t)["id"].(string))
	if task["state"] != "done" || strings.Contains(fmt.Sprint(task["url"]), "secret") {
		t.Fatalf("url import: %v", task)
	}
	// a failing import keeps the staged file for a retry
	r = e.call(t, "POST", "/v1/bundles/imports", map[string]any{"url": srv.URL + "/b.zip", "name": "run4"})
	task = e.waitTask(t, "imports", r.json(t)["id"].(string))
	if task["state"] != "failed" || task["error"].(map[string]any)["code"] != "conflict" || task["upload_id"] == nil {
		t.Fatalf("failed import: %v", task)
	}
	r = e.call(t, "POST", "/v1/bundles/imports", map[string]any{"upload_id": task["upload_id"], "name": "run5", "dataset": "set_d"})
	want(t, r, 202, "")
	e.waitTask(t, "imports", r.json(t)["id"].(string))
	jobByName(t, "run5")

	if items := e.call(t, "GET", "/v1/bundles/imports", nil).json(t)["items"].([]any); len(items) < 5 {
		t.Fatalf("imports list: %d", len(items))
	}
	want(t, e.call(t, "POST", "/v1/bundles/imports", map[string]any{"url": "ftp://x"}), 400, "invalid_request")
	want(t, e.call(t, "POST", "/v1/bundles/imports", map[string]any{}), 400, "invalid_request")
}

const legacyConfig = "job: extension\nconfig:\n  name: lora\n  process:\n  - type: sd_trainer\n    training_folder: /workspace/output\n" +
	"    datasets:\n    - folder_path: /workspace/data\n      caption_ext: txt\n    train:\n      steps: 5000\n" +
	"    model:\n      name_or_path: black-forest-labs/FLUX.1-dev\n  meta:\n    name: lora\n"

func TestLegacyBundleImport(t *testing.T) {
	e := setup(t)
	var b bytes.Buffer
	zw := zip.NewWriter(&b)
	for name, data := range map[string][]byte{
		"data/x.jpg": []byte("jpg"), "data/x.txt": []byte("cap"), "data/.ipynb_checkpoints/x-checkpoint.txt": []byte("j"),
		"output/lora/lora_000000200.safetensors": stFile(200), "output/lora/lora_000000400.safetensors": stFile(400),
		"output/lora/optimizer.pt": []byte("opt"), "output/lora/config.yaml": []byte(legacyConfig),
		"output/generations/": nil, "config.yaml": []byte("steps: 5000"),
	} {
		w, _ := zw.Create(name)
		w.Write(data)
	}
	zw.Close()
	var mb bytes.Buffer
	mw := newMultipart(&mb, map[string]string{"name": "bali"}, "file", "bali.zip", b.Bytes())
	r := e.do(t, "POST", "/v1/bundles/imports", &mb, map[string]string{"Content-Type": mw})
	want(t, r, 202, "")
	task := e.waitTask(t, "imports", r.json(t)["id"].(string))
	if task["state"] != "done" || task["dataset"] != "bali" {
		t.Fatalf("legacy import: %v", task)
	}
	j := jobByName(t, "bali")
	if j.Step != 400 || !strings.Contains(j.JobConfig, `"folder_path":`+jsonStr(filepath.Join(e.datasets, "bali"))) ||
		!strings.Contains(j.JobConfig, `"training_folder":`+jsonStr(e.training)) || !strings.Contains(j.JobConfig, `"name":"bali"`) {
		t.Fatalf("legacy job: step %d %s", j.Step, j.JobConfig)
	}
	for _, f := range []string{"bali_000000400.safetensors", "bali_000000200.safetensors", "optimizer.pt"} {
		if _, err := os.Stat(filepath.Join(e.training, "bali", f)); err != nil {
			t.Errorf("missing %s", f)
		}
	}
	if _, err := os.Stat(filepath.Join(e.datasets, "bali", ".ipynb_checkpoints")); err == nil {
		t.Error("junk restored")
	}
}

func jsonStr(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestBundleLibraryAndLinks(t *testing.T) {
	e := setup(t)
	links.Reset()
	t.Cleanup(links.Reset)
	id := bundleFixture(t, e)
	lib := filepath.Join(e.root, "library")
	if _, err := remote.Create(context.Background(), "lib", "fs", json.RawMessage(`{"path":`+jsonStr(lib)+`}`), nil); err != nil {
		t.Fatal(err)
	}
	r := e.call(t, "POST", "/v1/jobs/"+id+"/bundle/push", map[string]any{"remote": "lib", "checkpoints": "all"})
	want(t, r, 202, "")
	task := e.waitTask(t, "pushes", r.json(t)["id"].(string))
	if task["state"] != "done" || task["key"] != "bundles/run_step200.zip" || task["bytes_done"] != task["bytes"] {
		t.Fatalf("push: %v", task)
	}
	want(t, e.call(t, "POST", "/v1/jobs/"+id+"/bundle/push", map[string]any{"remote": "lib", "bundle": "run_step200"}), 409, "conflict")
	items := e.call(t, "GET", "/v1/bundles?remote=lib", nil).json(t)["items"].([]any)
	if len(items) != 1 {
		t.Fatalf("library: %v", items)
	}
	it := items[0].(map[string]any)
	s := it["summary"].(map[string]any)
	if it["id"] != "run_step200" || s["step"] != float64(200) || s["images"] != float64(1) || s["base_model"] != "org/base-model" ||
		len(s["checkpoints"].([]any)) != 2 || s["sha256"] == "" {
		t.Fatalf("library item: %v", it)
	}
	r = e.call(t, "POST", "/v1/bundles/imports", map[string]any{"remote": "lib", "key": it["key"], "name": "from_lib"})
	want(t, r, 202, "")
	if task := e.waitTask(t, "imports", r.json(t)["id"].(string)); task["state"] != "done" {
		t.Fatalf("library import: %v", task)
	}
	want(t, e.call(t, "POST", "/v1/bundles/imports", map[string]any{"remote": "lib", "key": "bundles/nope.zip"}), 404, "not_found")
	dl := e.call(t, "GET", "/v1/bundles/content?remote=lib&id=run_step200", nil)
	want(t, dl, 200, "")
	if zst, _ := os.Stat(filepath.Join(lib, "bundles", "run_step200.zip")); int64(len(dl.body)) != zst.Size() {
		t.Fatalf("library download: %d bytes, want %d", len(dl.body), zst.Size())
	}
	want(t, e.call(t, "GET", "/v1/bundles/content?remote=lib&id=nope", nil), 404, "not_found")
	want(t, e.call(t, "DELETE", "/v1/bundles/run_step200?remote=lib", nil), 204, "")
	if items := e.call(t, "GET", "/v1/bundles?remote=lib", nil).json(t)["items"].([]any); len(items) != 0 {
		t.Fatalf("after delete: %v", items)
	}
	want(t, e.call(t, "GET", "/v1/bundles?remote=nope", nil), 404, "not_found")

	// signed links: the exact GET works without a token, nothing else does
	t.Setenv("AI_TOOLKIT_AUTH", "secret")
	h := httpx.Auth(e.h)
	call := func(method, target string, body any, token string) *httptest.ResponseRecorder {
		var b bytes.Buffer
		if body != nil {
			json.NewEncoder(&b).Encode(body)
		}
		req := httptest.NewRequest(method, target, &b)
		req.Host = "node.example:8675"
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}
	rec := call("POST", "/v1/links", map[string]any{"path": "/v1/jobs/" + id + "/bundle?checkpoints=none", "ttl_s": 120}, "secret")
	var link struct{ URL, Path, ExpiresAt string }
	json.Unmarshal(rec.Body.Bytes(), &link)
	if rec.Code != 200 || !strings.HasPrefix(link.URL, "http://node.example:8675/v1/jobs/") || !strings.Contains(link.Path, "sig=") {
		t.Fatalf("link: %d %s", rec.Code, rec.Body)
	}
	if rec := call("GET", link.Path, nil, ""); rec.Code != 200 || rec.Header().Get("Content-Type") != "application/zip" {
		t.Fatalf("signed GET: %d %s", rec.Code, rec.Body)
	}
	u, _ := url.Parse(link.Path)
	q := u.Query()
	q.Set("checkpoints", "all")
	if rec := call("GET", u.Path+"?"+q.Encode(), nil, ""); rec.Code != 401 {
		t.Fatalf("tampered link: %d", rec.Code)
	}
	if rec := call("DELETE", link.Path, nil, ""); rec.Code != 401 {
		t.Fatalf("signed link used for DELETE: %d", rec.Code)
	}
	if rec := call("POST", "/v1/links", map[string]any{"path": "/v1/jobs"}, "secret"); rec.Code != 400 {
		t.Fatalf("link to a non-download: %d", rec.Code)
	}
	if rec := call("POST", "/v1/links", map[string]any{"path": "https://evil/v1/files/content"}, "secret"); rec.Code != 400 {
		t.Fatalf("link to another host: %d", rec.Code)
	}
	rec = call("POST", "/v1/links", map[string]any{"path": "/v1/files/content?root=training&path=run/optimizer.pt&download=1"}, "secret")
	json.Unmarshal(rec.Body.Bytes(), &link)
	if rec := call("GET", link.Path, nil, ""); rec.Code != 200 || rec.Body.String() != "optimizer state" ||
		!strings.Contains(rec.Header().Get("Content-Disposition"), "optimizer.pt") {
		t.Fatalf("file link: %d %v", rec.Code, rec.Header())
	}
}

func TestDatasetPutAndImportURL(t *testing.T) {
	e := setup(t)
	want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": "set"}), 201, "")
	r := e.do(t, "PUT", "/v1/datasets/set/files/sub/a.png", bytes.NewReader(pngBytes(5, 2)), nil)
	want(t, r, 200, "")
	if m := r.json(t); m["file"] != "sub/a.png" || m["width"] != float64(5) {
		t.Fatalf("put: %v", m)
	}
	want(t, e.do(t, "PUT", "/v1/datasets/set/files/.hidden", strings.NewReader("x"), nil), 400, "invalid_request")

	var zb bytes.Buffer
	zw := zip.NewWriter(&zb)
	w, _ := zw.Create("z/b.png")
	w.Write(pngBytes(1, 1))
	zw.Close()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/img":
			w.Write(pngBytes(3, 3)) // no extension: named from its bytes
		case "/set.zip":
			w.Write(zb.Bytes())
		case "/text":
			w.Write([]byte("hello"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	r = e.call(t, "POST", "/v1/datasets/set/import", map[string]any{"url": srv.URL + "/img", "caption": "three"})
	want(t, r, 200, "")
	files := r.json(t)["files"].([]any)
	if len(files) != 1 || !strings.HasSuffix(files[0].(string), ".png") {
		t.Fatalf("image import: %v", files)
	}
	if c, _ := os.ReadFile(filepath.Join(e.datasets, "set", strings.TrimSuffix(files[0].(string), ".png")+".txt")); string(c) != "three" {
		t.Fatalf("caption %q", c)
	}
	r = e.call(t, "POST", "/v1/datasets/set/import", map[string]any{"url": srv.URL + "/set.zip"})
	want(t, r, 200, "")
	if r.json(t)["added"] != float64(1) {
		t.Fatalf("zip import: %s", r.body)
	}
	want(t, e.call(t, "POST", "/v1/datasets/set/import", map[string]any{"url": srv.URL + "/text"}), 400, "invalid_request")
	want(t, e.call(t, "POST", "/v1/datasets/set/import", map[string]any{"url": srv.URL + "/missing"}), 502, "remote_failed")
	want(t, e.call(t, "POST", "/v1/datasets/set/import", map[string]any{"url": "file:///etc/passwd"}), 400, "invalid_request")
}

// newMultipart writes fields and one file part into b; returns the content type.
func newMultipart(b *bytes.Buffer, fields map[string]string, field, filename string, data []byte) string {
	mw := multipart.NewWriter(b)
	for k, v := range fields {
		mw.WriteField(k, v)
	}
	w, _ := mw.CreateFormFile(field, filename)
	w.Write(data)
	mw.Close()
	return mw.FormDataContentType()
}

func TestBundleSaver(t *testing.T) {
	e := setup(t)
	id := bundleFixture(t, e)
	ctx := context.Background()
	j, _ := store.GetJob(ctx, id)
	newest := artifacts.Checkpoint{Name: "run_000000200.safetensors"}

	t.Setenv("AITK_WHEN_DONE_BUNDLE", "")
	if _, covered, _ := (bundleSaver{}).Save(ctx, j, newest); covered {
		t.Fatal("covered without AITK_WHEN_DONE_BUNDLE")
	}

	lib := filepath.Join(e.root, "library")
	if _, err := remote.Create(ctx, "lib", "fs", json.RawMessage(`{"path":`+jsonStr(lib)+`}`), nil); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AITK_WHEN_DONE_BUNDLE", "lib")
	p, covered, err := (bundleSaver{}).Save(ctx, j, newest)
	if !covered || err != nil || p.Via != "bundle:lib" {
		t.Fatalf("save: %+v %v %v", p, covered, err)
	}
	zip := filepath.Join(lib, "bundles", "run_step200.zip")
	st, err := os.Stat(zip)
	if err != nil {
		t.Fatal(err)
	}

	// already in the library: verified, not pushed again
	if _, _, err := (bundleSaver{}).Save(ctx, j, newest); err != nil {
		t.Fatal(err)
	}
	if st2, _ := os.Stat(zip); !st2.ModTime().Equal(st.ModTime()) {
		t.Fatal("bundle pushed again")
	}

	// a truncated zip fails verification and is pushed again
	os.WriteFile(zip, []byte("short"), 0o644)
	if _, _, err := (bundleSaver{}).Save(ctx, j, newest); err != nil {
		t.Fatal(err)
	}
	if st3, _ := os.Stat(zip); st3.Size() != st.Size() {
		t.Fatalf("not re-pushed: %d bytes", st3.Size())
	}
}
