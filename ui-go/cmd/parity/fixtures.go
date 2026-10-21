package main

import (
	"bytes"
	"database/sql"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"mime/multipart"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/geocine/aitoolkit/internal/store"
	_ "modernc.org/sqlite"
)

// writeFixtures creates the same dataset and job folders under root on both
// sides. Image bytes are generated, so both roots get identical files.
func writeFixtures(root string) error {
	files := map[string][]byte{
		"datasets/set1/a.png":                                       pngImage(16, 12, true),
		"datasets/set1/a.txt":                                       []byte("a red square"),
		"datasets/set1/b.jpg":                                       jpegImage(20, 10),
		"datasets/set1/b.txt":                                       []byte("a blue rectangle"),
		"datasets/set1/c.png":                                       pngImage(8, 8, false),
		"datasets/set1/song.mp3":                                    []byte("not really an mp3, no ID3 tag"),
		"datasets/set 2/100% & more.png":                            pngImage(4, 4, false),
		"datasets/set 2/a..b.png":                                   pngImage(4, 4, false),
		"output/fixture_job/log.txt":                                []byte("step 1\nstep 2\nloss 0.5\n"),
		"output/fixture_job/fixture_job_000000100.safetensors":      []byte("fake-weights-100"),
		"output/fixture_job/fixture_job.safetensors":                []byte("fake-weights-final"),
		"output/fixture_job/samples/1700000000000__000000100_0.jpg": jpegImage(12, 12),
		"output/fixture_job/samples/1700000000001__000000100_1.png": pngImage(12, 12, false),
		"outside-secret.png":                                        pngImage(2, 2, false),
	}
	// sorted, with fixed distinct mtimes: listings ordered by mtime (loras,
	// samples) must come out the same on both sides
	names := make([]string, 0, len(files))
	for rel := range files {
		names = append(names, rel)
	}
	sort.Strings(names)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i, rel := range names {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(p, files[rel], 0o644); err != nil {
			return err
		}
		mt := base.Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(p, mt, mt); err != nil {
			return err
		}
	}
	return writeLossDB(filepath.Join(root, "output", "fixture_job", "loss_log.db"))
}

// writeLossDB writes a loss log in the trainer's schema (toolkit/logging_aitk.py).
func writeLossDB(path string) error {
	os.Remove(path)
	db, err := sql.Open("sqlite", store.FileURI(path))
	if err != nil {
		return err
	}
	defer db.Close()
	stmts := []string{
		`CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL)`,
		`CREATE TABLE metric_keys (key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER)`,
		`CREATE TABLE metrics (step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL, value_text TEXT,
			PRIMARY KEY (step, key), FOREIGN KEY (step) REFERENCES steps(step) ON DELETE CASCADE)`,
		`CREATE INDEX idx_metrics_key_step ON metrics (key, step)`,
		`INSERT INTO metric_keys VALUES ('loss', 1, 30), ('lr', 1, 30), ('note', 5, 5)`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}
	for step := 1; step <= 30; step++ {
		if _, err := db.Exec(`INSERT INTO steps VALUES (?, ?)`, step, 1700000000.5+float64(step)); err != nil {
			return err
		}
		if _, err := db.Exec(`INSERT INTO metrics VALUES (?, 'loss', ?, NULL), (?, 'lr', ?, NULL)`,
			step, 1.0/float64(step), step, 0.0001); err != nil {
			return err
		}
	}
	_, err = db.Exec(`INSERT INTO metrics VALUES (5, 'note', NULL, 'warmup done')`)
	return err
}

func pngImage(w, h int, alpha bool) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a := uint8(255)
			if alpha && (x+y)%3 == 0 {
				a = 64
			}
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x * 16), G: uint8(y * 16), B: 128, A: a})
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

func jpegImage(w, h int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: 30, G: uint8(x * 10), B: uint8(y * 20), A: 255})
		}
	}
	var buf bytes.Buffer
	jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90})
	return buf.Bytes()
}

func encodeMultipart(m Multipart, sd *side) (*bytes.Buffer, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for k, v := range m.Fields {
		if err := w.WriteField(k, expand(v, sd)); err != nil {
			return nil, "", err
		}
	}
	for _, f := range m.Files {
		fw, err := w.CreateFormFile(f.Field, f.Name)
		if err != nil {
			return nil, "", err
		}
		if _, err := fw.Write(f.Data); err != nil {
			return nil, "", err
		}
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return &buf, w.FormDataContentType(), nil
}
