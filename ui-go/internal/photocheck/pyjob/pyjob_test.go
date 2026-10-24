package pyjob_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/imgmeta"
	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/photocheck/pyjob"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func TestWrite(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "jihan_faces_20261024_101500")
	path, err := pyjob.Write(dir)
	if err != nil {
		t.Fatal(err)
	}
	if path != filepath.Join(dir, pyjob.ScriptName) {
		t.Fatalf("path %s", path)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, pyjob.Script()) || !bytes.Contains(got, []byte("--selftest")) {
		t.Fatal("written script differs from the embedded one")
	}

	// a stale copy from an older server is replaced, and no temp file is left
	if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := pyjob.Write(dir); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); !bytes.Equal(got, pyjob.Script()) {
		t.Fatal("stale script not replaced")
	}
	ents, _ := os.ReadDir(dir)
	if len(ents) != 1 {
		t.Fatalf("job folder holds %d entries, want only the script", len(ents))
	}

	// a file where the folder should be is an error, not a panic
	blocked := filepath.Join(t.TempDir(), "file")
	os.WriteFile(blocked, nil, 0o644)
	if _, err := pyjob.Write(blocked); err == nil {
		t.Fatal("Write into a file path succeeded")
	}
}

// TestScriptMatchesGo keeps the script's constants and its copy of the model
// pins in step with Go.
func TestScriptMatchesGo(t *testing.T) {
	s := string(pyjob.Script())
	if m := regexp.MustCompile(`(?m)^ALGO = (\d+)`).FindStringSubmatch(s); m == nil || m[1] != fmt.Sprint(photocheck.FaceAlgo) {
		t.Fatalf("script ALGO %v, Go FaceAlgo %d", m, photocheck.FaceAlgo)
	}
	if !strings.Contains(s, fmt.Sprintf("MODEL_NAME = %q", photocheck.FaceModel)) {
		t.Fatalf("script MODEL_NAME differs from Go FaceModel %q", photocheck.FaceModel)
	}
	if !strings.Contains(s, `ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"`) {
		t.Fatal("script ZOO base changed")
	}
	if len(pyjob.Models) != 4 {
		t.Fatalf("%d models, want yunet4, yunet5, sface, ppocr", len(pyjob.Models))
	}
	hex64 := regexp.MustCompile(`^[0-9a-f]{64}$`)
	for key, m := range pyjob.Models {
		if !hex64.MatchString(m.SHA256) || m.Bytes <= 0 || !strings.HasPrefix(m.URL, "https://github.com/opencv/opencv_zoo/raw/main/models/") {
			t.Errorf("%s: bad pin %+v", key, m)
		}
		block := regexp.MustCompile(`(?s)"` + key + `": \{(.*?)\}`).FindStringSubmatch(s)
		if block == nil {
			t.Errorf("%s: missing from DEFAULT_MODELS", key)
			continue
		}
		path := strings.TrimPrefix(m.URL, "https://github.com/opencv/opencv_zoo/raw/main/models/")
		for _, want := range []string{`ZOO + "` + path + `"`, `"sha256": "` + m.SHA256 + `"`, fmt.Sprintf(`"bytes": %d`, m.Bytes)} {
			if !strings.Contains(block[1], want) {
				t.Errorf("%s: script lacks %s", key, want)
			}
		}
		if m.File() != filepath.Base(path) {
			t.Errorf("%s: File() = %s", key, m.File())
		}
	}
	b, _ := json.Marshal(pyjob.Models["sface"])
	if !strings.Contains(string(b), `"url":`) || !strings.Contains(string(b), `"sha256":`) || !strings.Contains(string(b), `"bytes":38696353`) {
		t.Fatalf("model JSON %s, want the section 12.2 keys", b)
	}
}

// python returns a Python 3 interpreter, or skips: Go CI has no Python.
func python(t *testing.T) string {
	t.Helper()
	for _, name := range []string{"python3", "python"} {
		exe, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		out, err := exec.Command(exe, "-I", "-c", "import sys; print(sys.version_info[0])").Output()
		if err == nil && strings.TrimSpace(string(out)) == "3" {
			return exe
		}
	}
	t.Skip("no Python 3 on PATH")
	return ""
}

// runPy imports the written script as a module (no cv2 or numpy needed at
// import) and runs code against it as m; it returns stdout.
func runPy(t *testing.T, py, code string, args ...string) string {
	t.Helper()
	dir := t.TempDir()
	if _, err := pyjob.Write(dir); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	src := "import sys\nsys.path.insert(0, sys.argv[1])\nimport photo_faces as m\n" + code
	cmd := exec.CommandContext(ctx, py, append([]string{"-I", "-B", "-c", src, dir}, args...)...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python: %v\n%s", err, stderr.String())
	}
	return strings.TrimSpace(string(out))
}

// TestMatrixDecodes checks that the script's binary16 matrix is what
// photocheck.DecodeCos reads.
func TestMatrixDecodes(t *testing.T) {
	py := python(t)
	want := []float64{0.6, 0.0, 0.48, -0.25, 0.999, 1.5}
	b64 := runPy(t, py, "import json\nprint(m.pack_f16(json.loads(sys.argv[2])))", "[0.6, 0.0, 0.48, -0.25, 0.999, 1.5]")
	got, err := photocheck.DecodeCos(photocheck.FaceMatrix{Faces: []string{"a:0", "b:0", "c:0", "d:0"}, Cos: b64})
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range want {
		w = math.Min(w, 1) // values are clamped to a cosine's range
		if math.Abs(float64(got[i])-w) > 1e-3 {
			t.Errorf("value %d: got %v, want %v", i, got[i], w)
		}
	}

	out := runPy(t, py, `print([m.parse_version(v) for v in ("4.10.0", "5.0.0-pre", "4.9", "x")] == [(4, 10), (5, 0), (4, 9), (0, 0)])`)
	if out != "True" {
		t.Fatalf("parse_version: %s", out)
	}
}

// TestFrameSharpMatchesGo checks the script's frame_sharp against Go's
// sharpness on the same luma plane (section 3, "the same definition as
// Go"). It needs numpy.
func TestFrameSharpMatchesGo(t *testing.T) {
	py := python(t)
	if exec.Command(py, "-I", "-c", "import numpy").Run() != nil {
		t.Skip("numpy not installed")
	}
	const w, h = 300, 200 // under the S edge, so S is the plane itself
	img := image.NewGray(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(7))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 128 + 60*math.Sin(float64(x)/7) + 30*math.Cos(float64(y)/5) + float64(rng.Intn(21)-10)
			if x > 200 { // a soft region, so the tiles differ
				v = 128 + 10*math.Sin(float64(x+y)/20)
			}
			img.SetGray(x, y, color.Gray{Y: uint8(math.Max(0, math.Min(255, v)))})
		}
	}
	m := photocheck.MeasureImage(img, imgmeta.Sniff{Format: "png", Width: w, Height: h}, "x.png", 1)
	if !m.Decoded || m.Sharp <= 0 {
		t.Fatalf("Go metrics %+v", m)
	}
	raw := filepath.Join(t.TempDir(), "y.raw")
	if err := os.WriteFile(raw, img.Pix, 0o644); err != nil {
		t.Fatal(err)
	}
	code := fmt.Sprintf(`import numpy
m.np = numpy
s = numpy.fromfile(sys.argv[2], dtype=numpy.uint8).reshape(%d, %d).astype(numpy.float32)
print(repr(m.frame_sharp(s, m.laplacian(s))))`, h, w)
	out := runPy(t, py, code, raw)
	var got float64
	if _, err := fmt.Sscan(out, &got); err != nil {
		t.Fatalf("python printed %q", out)
	}
	if math.Abs(got-m.Sharp) > 1e-3*math.Max(1, m.Sharp) {
		t.Fatalf("frame_sharp %v, Go sharp %v", got, m.Sharp)
	}
}

// TestGray16Reads checks that a 16-bit greyscale PNG is measured on its
// high byte, as Go reads it, and not clipped to white by convert("RGB").
// It needs numpy and Pillow.
func TestGray16Reads(t *testing.T) {
	py := python(t)
	if exec.Command(py, "-I", "-c", "import numpy, PIL").Run() != nil {
		t.Skip("numpy or Pillow not installed")
	}
	g := image.NewGray16(image.Rect(0, 0, 512, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 512; x++ {
			g.SetGray16(x, y, color.Gray16{Y: uint16(x * 65535 / 511)})
		}
	}
	path := filepath.Join(t.TempDir(), "g16.png")
	if err := os.WriteFile(path, testutil.PNG(g), 0o644); err != nil {
		t.Fatal(err)
	}
	code := `import numpy
from PIL import Image, ImageOps
m.np, m.Image = numpy, Image
im = Image.open(sys.argv[2])
a = numpy.asarray(m.to_8bit(ImageOps.exif_transpose(im)).convert("RGB")).astype(numpy.float64)
print(im.mode[:1], round(float(a.mean()), 1), round(float((a >= 255).mean()), 3))`
	out := runPy(t, py, code, path)
	if out != "I 127.5 0.004" {
		t.Fatalf("gray16 reads as %q, want mode I, mean 127.5 and 1 of 256 columns white", out)
	}
}

// jobHarness runs the script's main on a temp sqlite Job row and dataset with
// cv2, the models and measure stubbed (standard library only). mode is
// "done", "stop" (stop=1 before the start) or "queue" (return_to_queue=1 set
// while the first photo is measured). It prints the row and faces.json.
const jobHarness = `import hashlib, json, os, sqlite3, types
work, mode = sys.argv[2], sys.argv[3]
db = os.path.join(work, "aitk.db")
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE Job (id TEXT PRIMARY KEY, status TEXT, info TEXT, step INTEGER, total_steps INTEGER, stop INTEGER DEFAULT 0, return_to_queue INTEGER DEFAULT 0)")
conn.execute("INSERT INTO Job (id, status, info, stop) VALUES ('j1', 'running', 'Starting job...', ?)", (int(mode == "stop"),))
conn.commit()
conn.close()
ds = os.path.join(work, "ds")
os.makedirs(ds)
photos = {"a.png": b"A", "b.png": b"A", "c.png": b"C", "e.png": b"E", "f.png": b"BAD"}
for name, data in photos.items():
    with open(os.path.join(ds, name), "wb") as fh:
        fh.write(data)
sha = lambda b: hashlib.sha256(b).hexdigest()
files = [{"file": n, "sha256": sha(d)} for n, d in photos.items()]
files[2]["sha256"] = sha(b"C before")  # changed since the check started
files.insert(4, {"file": "d.png", "sha256": sha(b"D")})  # gone
cfg = os.path.join(work, ".job_config.json")
with open(cfg, "w") as fh:
    json.dump({"config": {"process": [{"type": "aitk_photo_faces", "sqlite_db_path": db, "dataset_dir": ds,
        "files": files, "threads": 1, "text": False, "out": "faces.json"}]}}, fh)

calls = []
m.load_libs = lambda: (4, 10)
m.cv2 = types.SimpleNamespace(__version__="4.10.0", setNumThreads=lambda n: None)
m.Image = types.SimpleNamespace(UnidentifiedImageError=type("UnidentifiedImageError", (Exception,), {}))

class FakeModels:
    def __init__(self, conf, job):
        pass
    def announce(self, keys):
        calls.append("announce")
    def detector(self, version):
        calls.append("detector")
        return "yunet4", object()
    def recognizer(self):
        return None
    def text(self):
        return None

def measure(data, det, rec, text_model):
    calls.append(data.decode())
    if mode == "queue":
        c = sqlite3.connect(db)
        c.execute("UPDATE Job SET return_to_queue = 1 WHERE id = 'j1'")
        c.commit()
        c.close()
    if data == b"BAD":
        raise ValueError("bad data")
    face = {"box": [0.1, 0.1, 0.3, 0.4], "kps": [[0.2, 0.2]] * 5, "score": 0.9, "side": 300, "align_side": 300,
        "sharp": 12.5, "mean_y": 120.0, "clip_hi": 0.0, "embedded": True}
    small = dict(face, side=20, align_side=20, embedded=False)
    return {"faces": [face, small], "frame_sharp": 40.0, "text_checked": False}, [[1.0, 0.0] if data == b"A" else [0.6, 0.8]]

def cos_upper(t):
    return m.pack_f16([sum(x * y for x, y in zip(t[i], t[j])) for i in range(len(t)) for j in range(i + 1, len(t))])

m.Models, m.measure, m.cos_upper = FakeModels, measure, cos_upper
os.environ["AITK_JOB_ID"] = "j1"
code = m.main(["photo_faces.py", cfg])
row = sqlite3.connect(db).execute("SELECT status, info, step, total_steps FROM Job WHERE id = 'j1'").fetchone()
out = os.path.join(work, "faces.json")
faces = json.load(open(out)) if os.path.exists(out) else None
print(json.dumps({"code": code, "row": row, "calls": calls, "faces": faces}))`

type jobRun struct {
	Code  int                       `json:"code"`
	Row   [4]any                    `json:"row"` // status, info, step, total_steps
	Calls []string                  `json:"calls"`
	Faces *photocheck.FaceJobResult `json:"faces"`
}

func runJob(t *testing.T, py, mode string) jobRun {
	t.Helper()
	out := runPy(t, py, jobHarness, t.TempDir(), mode)
	var r jobRun
	if err := json.Unmarshal([]byte(out[strings.LastIndex(out, "\n")+1:]), &r); err != nil {
		t.Fatalf("harness printed %q: %v", out, err)
	}
	return r
}

// TestJobProtocol checks the Job row the worker reads (PH-U16): progress,
// the final status and info, stop and return_to_queue, and a faces.json
// that decodes as photocheck.FaceJobResult. It needs only the standard
// library: cv2, the models and measure are stubs.
func TestJobProtocol(t *testing.T) {
	py := python(t)
	sha := func(s string) string {
		h := sha256.Sum256([]byte(s))
		return hex.EncodeToString(h[:])
	}

	r := runJob(t, py, "done")
	if r.Code != 0 || r.Row[0] != "completed" || r.Row[1] != "Checked 3 photos, 4 faces" || r.Row[2] != 6.0 || r.Row[3] != 6.0 {
		t.Fatalf("done: code %d row %v", r.Code, r.Row)
	}
	if strings.Join(r.Calls, ",") != "announce,detector,A,E,BAD" {
		t.Fatalf("done: calls %v (the duplicate b.png is measured once)", r.Calls)
	}
	f := r.Faces
	if f == nil || f.Algo != photocheck.FaceAlgo || f.Model != photocheck.FaceModel || f.CV2 != "4.10.0" {
		t.Fatalf("done: faces.json %+v", f)
	}
	if strings.Join(f.Skipped, ",") != "c.png,d.png" {
		t.Errorf("skipped %v, want the changed and the missing file", f.Skipped)
	}
	a, e, bad := f.Items[sha("A")], f.Items[sha("E")], f.Items[sha("BAD")]
	if len(f.Items) != 3 || len(a.Faces) != 2 || a.Algo != photocheck.FaceAlgo || !a.Faces[0].Embedded || a.Faces[1].Embedded || a.FrameSharp != 40 || len(e.Faces) != 2 {
		t.Errorf("items %+v", f.Items)
	}
	if bad.Error != "bad data" || len(bad.Faces) != 0 || bad.Algo != photocheck.FaceAlgo {
		t.Errorf("unreadable item %+v", bad)
	}
	if strings.Join(f.Matrix.Faces, ",") != sha("A")+":0,"+sha("E")+":0" {
		t.Errorf("matrix faces %v, want one row per embedded face", f.Matrix.Faces)
	}
	cos, err := photocheck.DecodeCos(f.Matrix)
	if err != nil || len(cos) != 1 || math.Abs(float64(cos[0])-0.6) > 1e-3 {
		t.Errorf("matrix cos %v, %v", cos, err)
	}

	// stop before the start: no model is loaded and progress is not set
	r = runJob(t, py, "stop")
	if r.Code != 0 || r.Row[0] != "stopped" || r.Row[1] != "Job stopped" || r.Row[2] != nil || r.Row[3] != nil || len(r.Calls) != 0 || r.Faces != nil {
		t.Fatalf("stop: %+v", r)
	}

	// return_to_queue while the first photo is measured: queued after it
	r = runJob(t, py, "queue")
	if r.Code != 0 || r.Row[0] != "queued" || r.Row[1] != "Job queued" || r.Row[2] != 1.0 || r.Row[3] != 6.0 || r.Faces != nil {
		t.Fatalf("queue: %+v", r)
	}
	if strings.Join(r.Calls, ",") != "announce,detector,A" {
		t.Fatalf("queue: calls %v", r.Calls)
	}
}

// TestDownloadStops checks that a model download polls the stop flag and
// leaves no partial file behind. The download is a local stub.
func TestDownloadStops(t *testing.T) {
	py := python(t)
	code := `import io, os, sqlite3
work = sys.argv[2]
db = os.path.join(work, "aitk.db")
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE Job (id TEXT PRIMARY KEY, status TEXT, info TEXT, step INTEGER, total_steps INTEGER, stop INTEGER DEFAULT 0, return_to_queue INTEGER DEFAULT 0)")
conn.execute("INSERT INTO Job (id, status, info, stop) VALUES ('j1', 'stopping', 'Stopping job...', 1)")
conn.commit()
conn.close()
reads = []
class Body(io.BytesIO):
    def read(self, n=-1):
        reads.append(n)
        return super().read(n)
m.urllib.request.urlopen = lambda req, timeout=None: Body(b"x" * (40 << 20))
models = m.Models({"dir": os.path.join(work, "zoo")}, m.Job(db, "j1"))
try:
    models.ensure("sface")
    print("no stop")
except m.Stop as s:
    print(s.status, len(reads), sorted(os.listdir(models.dir)))`
	out := runPy(t, py, code, t.TempDir())
	if out = strings.TrimSpace(out[strings.LastIndex(out, "\n")+1:]); out != "stopped 4 []" {
		t.Fatalf("download: %q, want a stop after 4 MB and an empty models folder", out)
	}
}

// TestModelsIgnoreConfigURLs: a url or sha256 in config.models never picks
// what the script downloads or the file it writes; only the pins count.
func TestModelsIgnoreConfigURLs(t *testing.T) {
	py := python(t)
	code := `import os, json
work = sys.argv[2]
conf = {"dir": os.path.join(work, "zoo"),
        "yunet4": {"url": "https://attacker.example/run.py", "sha256": "0" * 64},
        "extra": {"url": "https://attacker.example/x.py"}}
models = m.Models(conf, None)
print(json.dumps({"keys": sorted(models.spec), "yunet4": os.path.basename(models.path("yunet4")),
                  "url": models.spec["yunet4"]["url"]}))`
	out := runPy(t, py, code, t.TempDir())
	out = strings.TrimSpace(out[strings.LastIndex(strings.TrimSpace(out), "\n")+1:])
	var got struct {
		Keys   []string `json:"keys"`
		Yunet4 string   `json:"yunet4"`
		URL    string   `json:"url"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("output %q: %v", out, err)
	}
	pin := pyjob.Models["yunet4"]
	if got.Yunet4 != pin.File() || got.URL != pin.URL || len(got.Keys) != len(pyjob.Models) {
		t.Fatalf("models %+v, want only the pins", got)
	}
}
