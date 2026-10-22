package media

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/settings"
)

type fixture struct {
	root, ds, out, data string
	srv                 *httptest.Server
}

func writePNG(t *testing.T, p string, w, h int, alpha bool) {
	t.Helper()
	var img image.Image
	if alpha {
		m := image.NewNRGBA(image.Rect(0, 0, w, h))
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				m.Set(x, y, color.NRGBA{R: 200, A: uint8(x * 255 / w)})
			}
		}
		img = m
	} else {
		m := image.NewRGBA(image.Rect(0, 0, w, h))
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				m.Set(x, y, color.RGBA{G: 180, A: 255})
			}
		}
		img = m
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, p, buf.Bytes())
}

func writeJPEG(t *testing.T, p string, w, h int) {
	t.Helper()
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, m, nil); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, p, buf.Bytes())
}

func mustWrite(t *testing.T, p string, b []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func setup(t *testing.T) *fixture {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	f := &fixture{
		root: root,
		ds:   filepath.Join(root, "datasets"),
		out:  filepath.Join(root, "output"),
		data: filepath.Join(root, "data"),
	}
	set := filepath.Join(f.ds, "set 1")
	writePNG(t, filepath.Join(set, "plain.png"), 40, 20, true)
	writePNG(t, filepath.Join(set, "100% & more.png"), 8, 8, false)
	writePNG(t, filepath.Join(set, "a..b.png"), 8, 8, false)
	mustWrite(t, filepath.Join(set, "plain.txt"), []byte("a caption"))
	writeJPEG(t, filepath.Join(f.out, "job", "samples", "s_0001.jpg"), 50, 30)
	mustWrite(t, filepath.Join(f.out, "job", "model.safetensors"), []byte("fake-weights"))
	writePNG(t, filepath.Join(f.data, "images", "d.png"), 4, 4, false)
	writePNG(t, filepath.Join(root, "outside-secret.png"), 4, 4, false)

	prev := roots
	roots = func(context.Context) (settings.AllowedRoots, error) {
		return settings.AllowedRoots{Datasets: f.ds, Training: f.out, Data: f.data}, nil
	}
	mux := http.NewServeMux()
	Register(mux)
	f.srv = httptest.NewServer(Intercept(mux))
	t.Cleanup(func() {
		f.srv.Close()
		roots = prev
	})
	return f
}

// enc is ui/src/utils/basic.ts encodeFilePathForUrl: <encoded folder>/<encoded name>.
func enc(p string) string {
	idx := max(strings.LastIndex(p, "/"), strings.LastIndex(p, `\`))
	folder := p[:idx]
	if idx == 0 {
		folder = p[:1]
	}
	return encodeURIComponent(folder) + "/" + encodeURIComponent(p[idx+1:])
}

// proxyDecoded is what a proxy that decodes %2F/%5C forwards.
func proxyDecoded(p string) string {
	segs := strings.Split(filepath.ToSlash(p), "/")
	for i, s := range segs {
		segs[i] = encodeURIComponent(s)
	}
	return strings.Join(segs, "/")
}

func (f *fixture) do(t *testing.T, method, path string, hdr map[string]string) *http.Response {
	t.Helper()
	u, err := url.Parse(f.srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(method, f.srv.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	// raw path, so neither client nor url package normalizes it
	req.URL = &url.URL{Scheme: "http", Host: u.Host, Opaque: "//" + u.Host + strings.SplitN(path, "?", 2)[0]}
	if q := strings.SplitN(path, "?", 2); len(q) == 2 {
		req.URL.RawQuery = q[1]
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestServePaths(t *testing.T) {
	f := setup(t)
	set := filepath.Join(f.ds, "set 1")
	plainPath := filepath.Join(set, "plain.png")
	vol := filepath.VolumeName(plainPath)
	rootLost := strings.TrimPrefix(proxyDecoded(strings.TrimPrefix(plainPath, vol)), "/")

	cases := []struct {
		name, path string
		want       int
	}{
		{"UI form", "/api/img/" + enc(plainPath), 200},
		{"legacy single segment", "/api/img/" + encodeURIComponent(plainPath), 200},
		{"percent, ampersand, space", "/api/img/" + enc(filepath.Join(set, "100% & more.png")), 200},
		{"dots inside name", "/api/img/" + enc(filepath.Join(set, "a..b.png")), 200},
		{"training sample", "/api/img/" + enc(filepath.Join(f.out, "job", "samples", "s_0001.jpg")), 200},
		{"data root via img", "/api/img/" + enc(filepath.Join(f.data, "images", "d.png")), 200},
		{"data root via files denied", "/api/files/" + enc(filepath.Join(f.data, "images", "d.png")), 403},
		{"proxy decoded, root kept", "/api/img/" + proxyDecoded(plainPath), 200},
		{"proxy decoded, root lost", "/api/img/" + rootLost, 200},
		{"traversal ..", "/api/img/" + enc(filepath.Join(set, "..", "..", "outside-secret.png")), 403},
		{"traversal %2E%2E%2F", "/api/img/" + encodeURIComponent(set) + "/..%2F..%2F..%2Foutside-secret.png", 403},
		{"outside root", "/api/files/" + enc(filepath.Join(f.root, "outside-secret.png")), 403},
		{"missing", "/api/img/" + enc(filepath.Join(set, "nope.png")), 404},
		{"directory", "/api/img/" + encodeURIComponent(set), 400},
		{"malformed escape (rejected by net/http)", "/api/img/%E0%A4%A", 400},
		{"invalid utf-8 escape", "/api/img/%FF.png", 500},
	}
	for _, c := range cases {
		resp := f.do(t, "GET", c.path, nil)
		resp.Body.Close()
		if resp.StatusCode != c.want {
			t.Errorf("%s: GET %s = %d, want %d", c.name, c.path, resp.StatusCode, c.want)
		}
	}
}

func TestHeadersETagRange(t *testing.T) {
	f := setup(t)
	modelPath := filepath.Join(f.out, "job", "model.safetensors")
	resp := f.do(t, "GET", "/api/files/"+enc(modelPath), nil)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(body) != "fake-weights" || resp.Header.Get("Content-Disposition") != `attachment; filename="model.safetensors"` ||
		resp.Header.Get("X-Content-Type-Options") != "nosniff" || resp.Header.Get("ETag") != "" {
		t.Fatalf("files download: %q %v", body, resp.Header)
	}

	imgURL := "/api/img/" + enc(filepath.Join(f.ds, "set 1", "plain.png"))
	resp = f.do(t, "GET", imgURL, nil)
	resp.Body.Close()
	etag := resp.Header.Get("ETag")
	if !strings.HasPrefix(etag, `W/"`) || resp.Header.Get("Content-Type") != "image/png" ||
		resp.Header.Get("Cache-Control") != "public, max-age=86400, immutable" {
		t.Fatalf("img headers: %v", resp.Header)
	}
	resp = f.do(t, "GET", imgURL, map[string]string{"If-None-Match": etag})
	resp.Body.Close()
	if resp.StatusCode != 304 || resp.Header.Get("ETag") != etag {
		t.Fatalf("revalidate: %d %v", resp.StatusCode, resp.Header)
	}

	fileURL := "/api/files/" + enc(modelPath)
	resp = f.do(t, "GET", fileURL, map[string]string{"Range": "bytes=0-3"})
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 206 || string(body) != "fake" || resp.Header.Get("Content-Range") != "bytes 0-3/12" {
		t.Fatalf("range: %d %q %v", resp.StatusCode, body, resp.Header)
	}
	resp = f.do(t, "GET", fileURL, map[string]string{"Range": "bytes=5-"})
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 206 || string(body) != "weights" || resp.Header.Get("Content-Range") != "bytes 5-11/12" {
		t.Fatalf("open range: %d %q %v", resp.StatusCode, body, resp.Header)
	}
	for _, bad := range []string{"bytes=50-", "bytes=4-2", "bytes=-3", "bytes=x-1"} {
		resp = f.do(t, "GET", fileURL, map[string]string{"Range": bad})
		resp.Body.Close()
		if resp.StatusCode != 416 || resp.Header.Get("Content-Range") != "bytes */12" {
			t.Fatalf("%s: %d %v", bad, resp.StatusCode, resp.Header)
		}
	}
	resp = f.do(t, "HEAD", fileURL, nil)
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || len(body) != 0 || resp.Header.Get("Content-Length") != "12" {
		t.Fatalf("head: %d %q %v", resp.StatusCode, body, resp.Header)
	}
}

func TestThumbGeneration(t *testing.T) {
	f := setup(t)
	set := filepath.Join(f.ds, "set 1")
	check := func(src, wantThumb, wantType string) {
		t.Helper()
		resp := f.do(t, "GET", "/api/img/"+enc(src)+"?thumb=1", nil)
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != wantType {
			t.Fatalf("thumb %s: %d %v", src, resp.StatusCode, resp.Header)
		}
		if _, err := os.Stat(wantThumb); err != nil {
			t.Fatalf("thumb not written: %v", err)
		}
		cfg, _, err := image.DecodeConfig(bytes.NewReader(body))
		if err != nil || cfg.Width != 300 || cfg.Height != 300 {
			t.Fatalf("thumb size %+v %v", cfg, err)
		}
	}
	check(filepath.Join(set, "plain.png"), filepath.Join(set, ".thumbs", "plain.png.png"), "image/png")
	sample := filepath.Join(f.out, "job", "samples", "s_0001.jpg")
	check(sample, filepath.Join(f.out, "job", "samples", ".thumbs", "s_0001.jpg.jpg"), "image/jpeg")
	// a second request serves the stored thumb
	check(sample, filepath.Join(f.out, "job", "samples", ".thumbs", "s_0001.jpg.jpg"), "image/jpeg")
	// no ?thumb serves the original
	resp := f.do(t, "GET", "/api/img/"+enc(sample), nil)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if cfg, _, _ := image.DecodeConfig(bytes.NewReader(body)); cfg.Width != 50 {
		t.Fatalf("original not served: %+v", cfg)
	}
}

func id3WithAPIC(img []byte) []byte {
	frame := append([]byte{0}, []byte("image/jpeg\x00")...)
	frame = append(frame, 3)                  // picture type
	frame = append(frame, []byte("c\x00")...) // description
	frame = append(frame, img...)
	size := len(frame)
	hdr := append([]byte("APIC"), byte(size>>24), byte(size>>16), byte(size>>8), byte(size), 0, 0)
	body := append(hdr, frame...)
	n := len(body)
	tag := []byte{'I', 'D', '3', 3, 0, 0, byte(n >> 21 & 0x7f), byte(n >> 14 & 0x7f), byte(n >> 7 & 0x7f), byte(n & 0x7f)}
	return append(tag, body...)
}

func TestAlbumArtFromID3(t *testing.T) {
	f := setup(t)
	art := bytes.Repeat([]byte{0xff, 0xd8, 0x11}, 40)
	mp3 := filepath.Join(f.ds, "set 1", "song.mp3")
	mustWrite(t, mp3, append(id3WithAPIC(art), []byte("audio-frames")...))

	mime, data, ok := extractArtFromTag(id3WithAPIC(art))
	if !ok || mime != "image/jpeg" || !bytes.Equal(data, art) {
		t.Fatalf("extract: %v %q %d", ok, mime, len(data))
	}

	rec := httptest.NewRecorder()
	serveAlbumArt(rec, httptest.NewRequest("GET", "/api/audio/art/x", nil), mp3)
	if rec.Code != 200 || !bytes.Equal(rec.Body.Bytes(), art) || rec.Header().Get("Cache-Control") != "public, max-age=604800, immutable" {
		t.Fatalf("art: %d %v", rec.Code, rec.Header())
	}
	if _, err := os.Stat(filepath.Join(f.ds, "set 1", ".thumbs", "song.mp3.jpg")); err != nil {
		t.Fatalf("art thumb not cached: %v", err)
	}

	noTag := filepath.Join(f.ds, "set 1", "plain.wav")
	mustWrite(t, noTag, []byte("RIFF...."))
	rec = httptest.NewRecorder()
	serveAlbumArt(rec, httptest.NewRequest("GET", "/x", nil), noTag)
	if rec.Code != 404 || rec.Body.String() != "No ID3 tag" {
		t.Fatalf("no tag: %d %q", rec.Code, rec.Body.String())
	}
}

func postJSON(t *testing.T, f *fixture, path string, body any) (*http.Response, string) {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(f.srv.URL+path, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	out, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, strings.TrimSpace(string(out))
}

func TestCaptionRoutes(t *testing.T) {
	f := setup(t)
	img := filepath.Join(f.ds, "set 1", "plain.png")

	_, body := postJSON(t, f, "/api/caption/get", map[string]any{"imgPath": img})
	if body != "a caption" {
		t.Fatalf("get: %q", body)
	}
	_, body = postJSON(t, f, "/api/caption/get", map[string]any{"imgPath": filepath.Join(f.ds, "set 1", "a..b.png")})
	if body != "" {
		t.Fatalf("missing caption: %q", body)
	}
	resp, body := postJSON(t, f, "/api/caption/get", map[string]any{"imgPath": filepath.Join(f.root, "outside-secret.png")})
	if resp.StatusCode != 403 || body != "Access denied" {
		t.Fatalf("denied: %d %q", resp.StatusCode, body)
	}

	resp, body = postJSON(t, f, "/api/img/caption", map[string]any{"imgPath": img, "caption": "new", "ext": ".caption"})
	if resp.StatusCode != 200 || body != `{"success":true}` {
		t.Fatalf("set caption: %d %q", resp.StatusCode, body)
	}
	if b, _ := os.ReadFile(filepath.Join(f.ds, "set 1", "plain.caption")); string(b) != "new" {
		t.Fatalf("caption file: %q", b)
	}
	resp, body = postJSON(t, f, "/api/img/caption", map[string]any{"imgPath": filepath.Join(f.root, "x.png"), "caption": "x"})
	if resp.StatusCode != 400 || body != `{"error":"Invalid image path"}` {
		t.Fatalf("invalid: %d %q", resp.StatusCode, body)
	}
	// a sibling folder that only shares the datasets root's prefix, and an ext that is a path
	resp, _ = postJSON(t, f, "/api/img/caption", map[string]any{"imgPath": f.ds + "-other/x.png", "caption": "x"})
	if resp.StatusCode != 400 {
		t.Fatalf("prefix sibling: %d", resp.StatusCode)
	}
	for _, ext := range []string{"/../../escape", "x/../../y", "../x", `txt\..\x`} {
		resp, _ = postJSON(t, f, "/api/img/caption", map[string]any{"imgPath": img, "caption": "x", "ext": ext})
		if resp.StatusCode == 200 {
			t.Fatalf("ext %q accepted", ext)
		}
	}

	resp, body = postJSON(t, f, "/api/caption/getBatch", map[string]any{
		"imgPaths": []any{img, filepath.Join(f.root, "outside-secret.png"), 5}, "ext": "caption",
	})
	var got struct{ Captions map[string]string }
	json.Unmarshal([]byte(body), &got)
	if resp.StatusCode != 200 || len(got.Captions) != 1 || got.Captions[img] != "new" {
		t.Fatalf("batch: %d %q", resp.StatusCode, body)
	}
	resp, body = postJSON(t, f, "/api/caption/getBatch", map[string]any{"imgPaths": "nope"})
	if resp.StatusCode != 400 || body != `{"error":"imgPaths must be an array"}` {
		t.Fatalf("batch bad: %d %q", resp.StatusCode, body)
	}
}

func TestDeleteRoutes(t *testing.T) {
	f := setup(t)
	img := filepath.Join(f.ds, "set 1", "plain.png")
	resp, body := postJSON(t, f, "/api/img/delete", map[string]any{"imgPath": filepath.Join(f.ds, "set 1", "plain.txt")})
	if resp.StatusCode != 400 || body != `{"error":"Not an image"}` {
		t.Fatalf("not image: %d %q", resp.StatusCode, body)
	}
	resp, body = postJSON(t, f, "/api/img/delete", map[string]any{"imgPath": img})
	if resp.StatusCode != 200 || exists(img) || exists(filepath.Join(f.ds, "set 1", "plain.txt")) {
		t.Fatalf("delete: %d %q", resp.StatusCode, body)
	}

	model := filepath.Join(f.out, "job", "model.safetensors")
	resp, body = postJSON(t, f, "/api/files/delete", map[string]any{"filePath": encodeURIComponent(model)})
	if resp.StatusCode != 200 || exists(model) {
		t.Fatalf("files delete: %d %q", resp.StatusCode, body)
	}
	resp, body = postJSON(t, f, "/api/files/delete", map[string]any{"filePath": filepath.Join(f.root, "outside-secret.png")})
	if resp.StatusCode != 403 || body != "Access denied" {
		t.Fatalf("files delete denied: %d %q", resp.StatusCode, body)
	}
	resp, body = postJSON(t, f, "/api/files/delete", map[string]any{})
	if resp.StatusCode != 400 || body != "filePath is required" {
		t.Fatalf("files delete missing: %d %q", resp.StatusCode, body)
	}
}

func TestUpload(t *testing.T) {
	f := setup(t)
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, name := range []string{"one.webp", "noext"} {
		w, _ := mw.CreateFormFile("files", name)
		w.Write([]byte("data-" + name))
	}
	mw.Close()
	resp, err := http.Post(f.srv.URL+"/api/img/upload", mw.FormDataContentType(), &buf)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Message string
		Files   []string
	}
	json.NewDecoder(resp.Body).Decode(&got)
	resp.Body.Close()
	if resp.StatusCode != 200 || got.Message != "Files uploaded successfully" || len(got.Files) != 2 ||
		!strings.HasSuffix(got.Files[0], ".webp") || !strings.HasSuffix(got.Files[1], ".noext") {
		t.Fatalf("upload: %d %+v", resp.StatusCode, got)
	}
	if b, _ := os.ReadFile(got.Files[0]); string(b) != "data-one.webp" {
		t.Fatalf("saved: %q", b)
	}

	buf.Reset()
	mw = multipart.NewWriter(&buf)
	mw.WriteField("other", "x")
	mw.Close()
	resp, _ = http.Post(f.srv.URL+"/api/img/upload", mw.FormDataContentType(), &buf)
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("no files: %d", resp.StatusCode)
	}
}

func zipNames(t *testing.T, p string) []string {
	t.Helper()
	zr, err := zip.OpenReader(p)
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	var names []string
	for _, f := range zr.File {
		names = append(names, f.Name)
	}
	sort.Strings(names)
	return names
}

func TestZip(t *testing.T) {
	f := setup(t)
	resp, body := postJSON(t, f, "/api/zip", map[string]any{"zipTarget": "dataset_captions", "datasetName": "set 1"})
	var got zipResult
	json.Unmarshal([]byte(body), &got)
	if resp.StatusCode != 200 || got.FileName != "set 1_captions.zip" {
		t.Fatalf("captions zip: %d %q", resp.StatusCode, body)
	}
	if names := zipNames(t, got.ZipPath); strings.Join(names, ",") != "set 1/plain.txt" {
		t.Fatalf("captions entries: %v", names)
	}

	resp, body = postJSON(t, f, "/api/zip", map[string]any{"zipTarget": "dataset", "datasetName": "../nope"})
	if resp.StatusCode != 404 || body != `{"error":"Dataset '../nope' not found"}` {
		t.Fatalf("missing dataset: %d %q", resp.StatusCode, body)
	}

	for _, name := range []string{"..", "."} {
		resp, _ = postJSON(t, f, "/api/zip", map[string]any{"zipTarget": "dataset", "datasetName": name})
		if resp.StatusCode != 404 {
			t.Fatalf("dataset %q: %d", name, resp.StatusCode)
		}
	}

	resp, body = postJSON(t, f, "/api/zip", map[string]any{"jobName": "job"})
	json.Unmarshal([]byte(body), &got)
	if resp.StatusCode != 200 || got.FileName != "samples.zip" {
		t.Fatalf("samples zip: %d %q", resp.StatusCode, body)
	}
	if names := zipNames(t, got.ZipPath); strings.Join(names, ",") != "samples/s_0001.jpg" {
		t.Fatalf("samples entries: %v", names)
	}
	resp, body = postJSON(t, f, "/api/zip", map[string]any{})
	if resp.StatusCode != 400 || body != `{"error":"jobName is required"}` {
		t.Fatalf("no job: %d %q", resp.StatusCode, body)
	}
}

func TestJSHelpers(t *testing.T) {
	for in, want := range map[string]int64{"12": 12, " 7x": 7, "-3": -3, "1,5": 1} {
		if got, ok := jsParseInt(in); !ok || got != want {
			t.Errorf("parseInt(%q) = %d %v", in, got, ok)
		}
	}
	if _, ok := jsParseInt("x1"); ok {
		t.Error("parseInt(x1) should be NaN")
	}
	if _, err := decodeURIComponent("%FF"); err == nil {
		t.Errorf("decoding %s should fail", "%FF")
	}
	if got := stripExt("/a/b.c/file.tar.gz"); got != "/a/b.c/file.tar" {
		t.Errorf("stripExt = %q", got)
	}
	for in, want := range map[any]string{nil: "txt", "": "txt", "..json": "json", " ": "txt"} {
		if got, err := captionExt(in); err != nil || got != want {
			t.Errorf("captionExt(%v) = %q %v", in, got, err)
		}
	}
	if _, err := captionExt(true); err == nil {
		t.Error("captionExt(true) should throw")
	}
	if got := encodeURIComponent("a b/c'(é)"); got != "a%20b%2Fc'(%C3%A9)" {
		t.Errorf("encodeURIComponent = %q", got)
	}
}
