package bundles

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/jsonx"
)

// safetensors is a header-only safetensors file whose metadata says step.
func safetensors(step int) []byte {
	info, _ := json.Marshal(map[string]int{"step": step})
	head, _ := json.Marshal(map[string]any{"__metadata__": map[string]string{"training_info": string(info)}})
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(head)))
	b.Write(head)
	return b.Bytes()
}

type zent struct {
	name string
	data []byte
}

func zipOf(t *testing.T, entries []zent) *zip.Reader {
	t.Helper()
	var b bytes.Buffer
	zw := zip.NewWriter(&b)
	for _, e := range entries {
		w, err := zw.Create(e.name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write(e.data)
	}
	zw.Close()
	zr, err := zip.NewReader(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if err != nil {
		t.Fatal(err)
	}
	return zr
}

const legacyYAML = `job: extension
config:
  name: lora
  process:
  - type: sd_trainer
    training_folder: /workspace/output
    device: cuda:0
    datasets:
    - folder_path: /workspace/data
      caption_ext: txt
      resolution: [512, 768]
    train:
      steps: 5000
      lr: 1e-4
      ema_config:
        use_ema: true
    model:
      name_or_path: black-forest-labs/FLUX.1-dev
      is_flux: true
  meta:
    name: lora
    version: '1.0'
`

// the layout of a hand-made run zip (like archive_name_bali.zip)
func legacyZip(t *testing.T) *zip.Reader {
	return zipOf(t, []zent{
		{"data/a.jpg", []byte("jpg-a")}, {"data/a.txt", []byte("cap a")},
		{"data/b.jpg", []byte("jpg-b")}, {"data/b.txt", []byte("cap b")},
		{"data/.aitk_size.json", []byte("{}")},
		{"data/.ipynb_checkpoints/a-checkpoint.txt", []byte("junk")},
		{"data/_latent_cache/a_x.safetensors", []byte("latent")},
		{"output/generations/", nil},
		{"output/lora/lora_000000400.safetensors", safetensors(400)},
		{"output/lora/optimizer.pt", []byte("opt")},
		{"output/lora/lora_000000200.safetensors", safetensors(200)},
		{"output/lora/samples/1__000000200_0.jpg", []byte("s")},
		{"output/lora/samples/.ipynb_checkpoints/x.jpg", []byte("junk")},
		{"output/lora/config.yaml", []byte(legacyYAML)},
		{"config.yaml", []byte("trigger: x")},
		{"__MACOSX/data/._a.jpg", []byte("junk")},
	})
}

func TestReadLegacy(t *testing.T) {
	p, err := Read(legacyZip(t))
	if err != nil {
		t.Fatal(err)
	}
	m := p.Manifest
	if !p.Legacy || m.Name != "lora" || m.Step != 400 || !m.Optimizer || m.Samples != 1 || !m.Caches ||
		m.BaseModel != "black-forest-labs/FLUX.1-dev" || m.TotalSteps == nil || *m.TotalSteps != 5000 {
		t.Fatalf("manifest: %+v", m)
	}
	if strings.Join(m.Checkpoints, ",") != "lora_000000200.safetensors,lora_000000400.safetensors" {
		t.Fatalf("checkpoints %v", m.Checkpoints)
	}
	if len(p.Datasets) != 1 || p.Datasets[0].Source != "/workspace/data" || len(p.Datasets[0].Files) != 5 ||
		p.Datasets[0].Ref.Images != 2 || p.Datasets[0].Ref.Captions != 2 {
		t.Fatalf("datasets: %+v", p.Datasets)
	}
	// saves come last, oldest first, so extraction gives the newest the newest ctime
	var rels []string
	for _, f := range p.JobFiles {
		rels = append(rels, f.Rel)
	}
	got := strings.Join(rels, ",")
	if !strings.HasSuffix(got, "lora_000000200.safetensors,lora_000000400.safetensors") || strings.Contains(got, "ipynb") {
		t.Fatalf("job files: %s", got)
	}
	skipped := strings.Join(p.Skipped, ",")
	for _, s := range []string{".aitk_size.json", ".ipynb_checkpoints", "__MACOSX", "config.yaml"} {
		if !strings.Contains(skipped, s) {
			t.Errorf("%s not skipped: %s", s, skipped)
		}
	}
	// the yaml became an ordered JSON config with numbers intact
	cfg, err := jsonx.ParseObject(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(cfg)
	if !strings.Contains(string(raw), `"lr":0.0001`) || !strings.HasPrefix(string(raw), `{"job":"extension","config":{"name":"lora"`) ||
		!strings.Contains(string(raw), `"resolution":[512,768]`) || !strings.Contains(string(raw), `"version":"1.0"`) {
		t.Fatalf("config: %s", raw)
	}
}

func TestReadRejects(t *testing.T) {
	for name, entries := range map[string][]zent{
		"traversal":        {{"data/../../x.txt", []byte("x")}, {"output/a/config.yaml", []byte(legacyYAML)}},
		"absolute":         {{"/etc/x", []byte("x")}},
		"no job":           {{"random/file.txt", []byte("x")}},
		"two jobs":         {{"output/a/config.yaml", []byte(legacyYAML)}, {"output/b/config.yaml", []byte(legacyYAML)}},
		"bad manifest dir": {{ManifestName, []byte(`{"format":1,"name":"x","config":{"config":{"process":[{}]}},"job_dir":"output/../x","datasets":[]}`)}},
		"future":           {{ManifestName, []byte(`{"format":99,"name":"x","config":{}}`)}},
	} {
		if _, err := Read(zipOf(t, entries)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestLocalizeAndRename(t *testing.T) {
	p, err := Read(legacyZip(t))
	if err != nil {
		t.Fatal(err)
	}
	ds := filepath.Join("/srv", "datasets", "bali")
	cfg, err := Localize(p.Manifest.Config, map[string]string{"/workspace/data": ds}, filepath.Join("/srv", "datasets"), filepath.Join("/srv", "output"), "bali_v2")
	if err != nil {
		t.Fatal(err)
	}
	o, _ := jsonx.ParseObject([]byte(cfg))
	p0 := o.Object("config").Array("process")[0].(*jsonx.Object)
	fp, _ := p0.Array("datasets")[0].(*jsonx.Object).Get("folder_path")
	tf, _ := p0.Get("training_folder")
	name, _ := o.Object("config").Get("name")
	meta, _ := o.Object("config").Object("meta").Get("name")
	if fp != ds || tf != filepath.Join("/srv", "output") || name != "bali_v2" || meta != "bali_v2" {
		t.Fatalf("localized: %s", cfg)
	}
	for in, want := range map[string]string{
		"lora_000000400.safetensors": "bali_v2_000000400.safetensors", "lora.safetensors": "bali_v2.safetensors",
		"lora_000000400/unet/x.bin": "bali_v2_000000400/unet/x.bin", "optimizer.pt": "optimizer.pt",
		"samples/1__000000200_0.jpg": "samples/1__000000200_0.jpg", "lora_extra.safetensors": "lora_extra.safetensors",
		"config.yaml": "config.yaml",
	} {
		if got := RenameCheckpoint(in, "lora", "bali_v2"); got != want {
			t.Errorf("RenameCheckpoint(%s) = %s, want %s", in, got, want)
		}
	}
}

func TestYAMLScalars(t *testing.T) {
	o, err := YAMLObject([]byte("a: 1e-4\nb: 0x10\nc: yes\nd: true\ne: ~\nf: '07'\ng: .inf\nh: &x {k: 1}\ni: *x\n"))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(o)
	if string(raw) != `{"a":0.0001,"b":16,"c":"yes","d":true,"e":null,"f":"07","g":".inf","h":{"k":1},"i":{"k":1}}` {
		t.Fatalf("%s", raw)
	}
}

func writeFile(t *testing.T, p string, data []byte) {
	t.Helper()
	os.MkdirAll(filepath.Dir(p), 0o755)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestExportRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ds := filepath.Join(dir, "datasets", "set")
	writeFile(t, filepath.Join(ds, "a.png"), []byte("png"))
	writeFile(t, filepath.Join(ds, "a.txt"), []byte("cap"))
	writeFile(t, filepath.Join(ds, "sub", "_t_e_cache", "a.safetensors"), []byte("te"))
	writeFile(t, filepath.Join(ds, "_latent_cache", "a.safetensors"), []byte("lat"))
	writeFile(t, filepath.Join(ds, ".thumbs", "a.png.jpg"), []byte("thumb"))
	job := filepath.Join(dir, "output", "run")
	writeFile(t, filepath.Join(job, "run_000000100.safetensors"), safetensors(100))
	writeFile(t, filepath.Join(job, "run_000000200.safetensors"), safetensors(200))
	writeFile(t, filepath.Join(job, "run_000000300", "aitk_meta.yaml"), []byte("training_info:\n  step: 300\n"))
	writeFile(t, filepath.Join(job, "optimizer.pt"), []byte("opt"))
	writeFile(t, filepath.Join(job, "config.yaml"), []byte("x: 1"))
	writeFile(t, filepath.Join(job, "samples", "1__000000100_0.jpg"), []byte("s"))
	writeFile(t, filepath.Join(job, ".job_config.json"), []byte("{}"))
	cfg := json.RawMessage(`{"job":"extension","config":{"name":"run","process":[{"datasets":[{"folder_path":"aitk://datasets/set"}]}]}}`)

	for _, tc := range []struct {
		opt   Options
		saves string
		step  int
		files int
	}{
		{DefaultOptions(), "run_000000300", 300, 0},
		{Options{Checkpoints: CheckpointsAll}, "run_000000100.safetensors,run_000000200.safetensors,run_000000300", 300, 0},
		{Options{Checkpoints: CheckpointsNone, Caches: true}, "", 0, 0},
	} {
		entries, m, err := Collect(context.Background(), Source{JobFolder: job, Datasets: []SourceDataset{{"set", ds}},
			Manifest: Manifest{Name: "run", Config: cfg}, TempDir: dir}, tc.opt)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Join(m.Checkpoints, ",") != tc.saves || m.Step != tc.step || m.Datasets[0].Images != 1 {
			t.Fatalf("%+v: manifest %+v", tc.opt, m)
		}
		size, err := Size(entries)
		if err != nil {
			t.Fatal(err)
		}
		var b bytes.Buffer
		if err := Write(context.Background(), &b, entries, nil); err != nil {
			t.Fatal(err)
		}
		if int64(b.Len()) != size {
			t.Fatalf("Size %d, wrote %d", size, b.Len())
		}
		zr, err := zip.NewReader(bytes.NewReader(b.Bytes()), int64(b.Len()))
		if err != nil {
			t.Fatal(err)
		}
		if zr.File[0].Name != ManifestName || zr.File[0].Method != zip.Store {
			t.Fatal("manifest is not the first, stored entry")
		}
		if pm, ok := PeekManifest(b.Bytes()[:min(b.Len(), 1<<20)]); !ok || pm.Name != "run" || pm.Step != tc.step {
			t.Fatalf("PeekManifest: %v %+v", ok, pm)
		}
		var names []string
		for _, f := range zr.File {
			names = append(names, f.Name)
		}
		all := strings.Join(names, ",")
		if strings.Contains(all, ".thumbs") || strings.Contains(all, ".job_config") {
			t.Fatalf("hidden files packed: %s", all)
		}
		if strings.Contains(all, "_cache") != tc.opt.Caches {
			t.Fatalf("caches %v: %s", tc.opt.Caches, all)
		}
		p, err := Read(zr)
		if err != nil {
			t.Fatal(err)
		}
		if p.Legacy || p.Manifest.Name != "run" || len(p.Datasets) != 1 || p.Datasets[0].Source != "aitk://datasets/set" {
			t.Fatalf("plan: %+v", p)
		}
		if tc.saves != "" && !strings.HasPrefix(p.JobFiles[len(p.JobFiles)-1].Rel, "run_000000300/") {
			t.Fatalf("newest save is not last: %v", p.JobFiles)
		}
		// an identical dataset folder is recognized for reuse
		same, err := SameFiles(ds, p.Datasets[0].Files)
		if err != nil || !same {
			t.Fatalf("SameFiles: %v %v", same, err)
		}
	}
	// same size, other bytes: told apart by CRC
	entries, _, _ := Collect(context.Background(), Source{JobFolder: job, Datasets: []SourceDataset{{"set", ds}},
		Manifest: Manifest{Name: "run", Config: cfg}, TempDir: dir}, DefaultOptions())
	var b bytes.Buffer
	Write(context.Background(), &b, entries, nil)
	writeFile(t, filepath.Join(ds, "a.txt"), []byte("CAP"))
	zr, _ := zip.NewReader(bytes.NewReader(b.Bytes()), int64(b.Len()))
	p, _ := Read(zr)
	if same, _ := SameFiles(ds, p.Datasets[0].Files); same {
		t.Fatal("different contents reported as the same dataset")
	}
}

func TestExtractStopsAtDeclaredSize(t *testing.T) {
	zr := zipOf(t, []zent{{"output/x/config.yaml", []byte(legacyYAML)}})
	f := zr.File[0]
	f.UncompressedSize64 = 3 // lies: the entry is longer
	dir := t.TempDir()
	if err := Extract(context.Background(), []File{{f, "c.yaml"}}, dir, nil); err == nil {
		t.Fatal("an entry larger than declared was accepted")
	}
}

func TestParseContentRange(t *testing.T) {
	if _, _, _, err := ParseContentRange("bytes 0-9/100"); err != nil {
		t.Fatal(err)
	}
	if s, e, n, err := ParseContentRange("bytes 10-19/*"); err != nil || s != 10 || e != 19 || n != -1 {
		t.Fatal(s, e, n, err)
	}
	for _, bad := range []string{"bytes 5-1/10", "items 0-1/2", ""} {
		if _, _, _, err := ParseContentRange(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}
