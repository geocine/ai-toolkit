package bundles

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/portable"
)

// Checkpoint selections.
const (
	CheckpointsAll    = "all"
	CheckpointsLatest = "latest"
	CheckpointsNone   = "none"
)

// Options select what a bundle carries.
type Options struct {
	Checkpoints string `json:"checkpoints"` // all | latest (newest step + the final save) | none
	Optimizer   bool   `json:"optimizer"`
	Samples     bool   `json:"samples"`
	Caches      bool   `json:"caches"` // the datasets' _latent_cache, _t_e_cache, _clip_vision_cache folders
}

// DefaultOptions: what resuming needs, the samples and the dataset caches
// (re-caching latents and text embeddings is slow on big models).
func DefaultOptions() Options {
	return Options{Checkpoints: CheckpointsLatest, Optimizer: true, Samples: true, Caches: true}
}

// Entry is one file of a bundle being written: a local file, or Data.
type Entry struct {
	Name     string
	Path     string
	Data     []byte
	Size     int64
	Modified time.Time
}

// SourceDataset is a dataset folder the job trains on.
type SourceDataset struct {
	Name string
	Dir  string
}

// Source is the run to pack. Manifest carries the job's fields (name,
// config, policy, ...); Collect fills in the rest.
type Source struct {
	JobFolder string
	Datasets  []SourceDataset
	Manifest  Manifest
	Running   bool   // the trainer may be writing: wait for stable files, snapshot logs
	TempDir   string // where snapshots of the loss log and log go
}

var (
	mediaExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".bmp": true, ".gif": true,
		".tif": true, ".tiff": true, ".mp4": true, ".mov": true, ".webm": true, ".mkv": true, ".avi": true, ".m4v": true,
		".mp3": true, ".wav": true, ".flac": true, ".ogg": true}
)

// IsMedia reports whether a dataset file is an image, video or audio item.
func IsMedia(name string) bool { return mediaExts[strings.ToLower(filepath.Ext(name))] }

func hidden(name string) bool { return strings.HasPrefix(name, ".") }

// walkFiles lists regular files under dir (hidden entries skipped), with
// slash-separated paths relative to dir. skipDir prunes folders by name.
func walkFiles(dir string, skipDir func(string) bool) ([]string, error) {
	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p == dir {
			return nil
		}
		if hidden(d.Name()) || (d.IsDir() && skipDir != nil && skipDir(d.Name())) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Type().IsRegular() {
			rel, _ := filepath.Rel(dir, p)
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	sort.Strings(out)
	return out, err
}

func fileEntry(name, path string) (Entry, error) {
	st, err := os.Stat(path)
	if err != nil {
		return Entry{}, err
	}
	return Entry{Name: name, Path: path, Size: st.Size(), Modified: st.ModTime()}, nil
}

// Collect lists the bundle's entries and completes the manifest. Order:
// aitk_bundle.json first (a reader can parse it from the first bytes),
// then the datasets, the job folder, and the job's saves last, oldest
// first, so even a plain unzip leaves the newest save with the newest ctime
// (what ai-toolkit resumes from). Snapshots go to src.TempDir; the caller
// removes it.
func Collect(ctx context.Context, src Source, opt Options) ([]Entry, *Manifest, error) {
	m := src.Manifest
	m.Format = Format
	m.CreatedAt = time.Now().UTC().Truncate(time.Second) // fixed width: HEAD and GET agree on the size
	m.JobDir = "output/" + m.Name
	m.Caches = opt.Caches
	m.Datasets, m.Checkpoints, m.Samples, m.Optimizer, m.Step = []DatasetRef{}, []string{}, 0, false, 0
	if m.Warnings == nil {
		m.Warnings = []string{}
	}
	var entries []Entry

	for i, ds := range src.Datasets {
		ref := DatasetRef{Name: ds.Name, Dir: DatasetDir(i)}
		files, err := walkFiles(ds.Dir, func(name string) bool { return IsCacheDir(name) && !opt.Caches })
		if err != nil {
			return nil, nil, fmt.Errorf("dataset %s: %w", ds.Name, err)
		}
		for _, rel := range files {
			e, err := fileEntry(ref.Dir+"/"+rel, filepath.Join(ds.Dir, filepath.FromSlash(rel)))
			if err != nil {
				return nil, nil, err
			}
			entries = append(entries, e)
			if inCache(rel) {
				continue
			}
			if IsMedia(rel) {
				ref.Images++
			} else if strings.EqualFold(filepath.Ext(rel), ".txt") {
				ref.Captions++
			}
		}
		m.Datasets = append(m.Datasets, ref)
	}

	jobEntries, err := collectJob(ctx, src, opt, &m)
	if err != nil {
		return nil, nil, err
	}
	entries = append(entries, jobEntries...)

	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, nil, err
	}
	head := Entry{Name: ManifestName, Data: raw, Size: int64(len(raw)), Modified: m.CreatedAt}
	return append([]Entry{head}, entries...), &m, nil
}

type save struct {
	name string
	step int // FinalStep for the step-less save
	dir  bool
}

// collectJob picks the job folder's files: the selected saves (files or
// diffusers folders), the optimizer state, samples, and everything else
// that is not hidden (config.yaml, log, loss log snapshot, ...).
func collectJob(ctx context.Context, src Source, opt Options, m *Manifest) ([]Entry, error) {
	folder := src.JobFolder
	prefix := m.JobDir + "/"
	top, err := os.ReadDir(folder)
	if os.IsNotExist(err) {
		m.Warnings = append(m.Warnings, "the job has no output folder yet")
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	var saves []save
	for _, d := range top {
		if !hidden(d.Name()) && (d.IsDir() || d.Type().IsRegular()) {
			if step, ok := checkpointOf(d.Name(), m.Name, d.IsDir()); ok {
				saves = append(saves, save{d.Name(), step, d.IsDir()})
			}
		}
	}
	sort.Slice(saves, func(i, j int) bool {
		if saves[i].step != saves[j].step {
			return saves[i].step < saves[j].step
		}
		return saves[i].name < saves[j].name
	})
	isSave := map[string]bool{}
	newestStepped := -1
	for i, s := range saves {
		isSave[s.name] = true
		if s.step != FinalStep {
			newestStepped = i
		}
	}
	var kept []save
	for i, s := range saves {
		if opt.Checkpoints == CheckpointsAll ||
			(opt.Checkpoints == CheckpointsLatest && (s.step == FinalStep || i == newestStepped)) {
			kept = append(kept, s)
		}
	}

	var out, saveEntries []Entry
	add := func(dst *[]Entry, name, path string, wait time.Duration) error {
		if src.Running && wait > 0 {
			if err := portable.WaitStable(ctx, path, wait); err != nil {
				return err
			}
		}
		e, err := fileEntry(prefix+name, path)
		if err != nil {
			return err
		}
		*dst = append(*dst, e)
		return nil
	}
	for _, d := range top {
		name := d.Name()
		p := filepath.Join(folder, name)
		switch {
		case hidden(name) || isSave[name]:
		case d.IsDir():
			if name == "samples" && !opt.Samples {
				continue
			}
			files, err := walkFiles(p, nil)
			if err != nil {
				return nil, err
			}
			for _, rel := range files {
				if err := add(&out, name+"/"+rel, filepath.Join(p, filepath.FromSlash(rel)), 0); err != nil {
					return nil, err
				}
				if name == "samples" {
					m.Samples++
				}
			}
		case !d.Type().IsRegular():
		case name == "optimizer.pt":
			if opt.Optimizer {
				if err := add(&out, name, p, 60*time.Second); err != nil {
					return nil, err
				}
				m.Optimizer = true
			}
		case name == "loss_log.db":
			dst := filepath.Join(src.TempDir, "loss_log.db")
			if err := portable.SnapshotSQLite(ctx, p, dst); err != nil {
				m.Warnings = append(m.Warnings, "loss log not included: "+err.Error())
				continue
			}
			e, err := fileEntry(prefix+name, dst)
			if err != nil {
				return nil, err
			}
			out = append(out, e)
		case src.Running && strings.HasSuffix(name, ".txt"):
			// log.txt grows while we read it: a copy keeps size == content
			dst := filepath.Join(src.TempDir, name)
			if err := copyFile(p, dst); err != nil {
				return nil, err
			}
			e, err := fileEntry(prefix+name, dst)
			if err != nil {
				return nil, err
			}
			out = append(out, e)
		default:
			if err := add(&out, name, p, 0); err != nil {
				return nil, err
			}
		}
	}
	for _, s := range kept {
		p := filepath.Join(folder, s.name)
		if s.dir {
			files, err := walkFiles(p, nil)
			if err != nil {
				return nil, err
			}
			for _, rel := range files {
				if err := add(&saveEntries, s.name+"/"+rel, filepath.Join(p, filepath.FromSlash(rel)), 30*time.Second); err != nil {
					return nil, err
				}
			}
		} else if err := add(&saveEntries, s.name, p, 30*time.Second); err != nil {
			return nil, err
		}
		m.Checkpoints = append(m.Checkpoints, s.name)
	}
	// the step a resume starts at: the newest kept save's metadata
	if n := len(kept); n > 0 {
		last := kept[n-1]
		if step, ok := CheckpointStep(filepath.Join(folder, last.name)); ok {
			m.Step = step
		} else if last.step != FinalStep {
			m.Step = last.step
		} else if n > 1 {
			m.Step = kept[n-2].step
		}
	}
	return append(out, saveEntries...), nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// zeros is Size bytes of nothing, for measuring a zip without reading files.
type zeros struct{ n int64 }

var zeroBlock = make([]byte, 1<<20)

func (z *zeros) Read(p []byte) (int, error) {
	if z.n <= 0 {
		return 0, io.EOF
	}
	n := int(min(int64(len(p)), z.n))
	clear(p[:n])
	z.n -= int64(n)
	return n, nil
}

func (z *zeros) WriteTo(w io.Writer) (int64, error) {
	var total int64
	for z.n > 0 {
		n, err := w.Write(zeroBlock[:min(int64(len(zeroBlock)), z.n)])
		total += int64(n)
		z.n -= int64(n)
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

type counter struct{ n int64 }

func (c *counter) Write(p []byte) (int, error) {
	c.n += int64(len(p))
	return len(p), nil
}

// Size is the exact byte size Write produces for entries (the same zip
// written over zeros), so downloads can send Content-Length.
func Size(entries []Entry) (int64, error) {
	var c counter
	err := write(context.Background(), &c, entries, true, nil)
	return c.n, err
}

// Write streams the bundle as a zip. Everything is stored, not deflated:
// checkpoints and images do not compress, and stored sizes are what Size
// predicts. progress (optional) receives the zip bytes written so far.
func Write(ctx context.Context, w io.Writer, entries []Entry, progress func(int64)) error {
	return write(ctx, w, entries, false, progress)
}

type progressWriter struct {
	w  io.Writer
	n  int64
	fn func(int64)
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.n += int64(n)
	p.fn(p.n)
	return n, err
}

func write(ctx context.Context, w io.Writer, entries []Entry, dry bool, progress func(int64)) error {
	if progress != nil {
		w = &progressWriter{w: w, fn: progress}
	}
	zw := zip.NewWriter(w)
	for _, e := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		fw, err := zw.CreateHeader(&zip.FileHeader{Name: e.Name, Method: zip.Store, Modified: e.Modified.UTC().Truncate(time.Second)})
		if err != nil {
			return err
		}
		var src io.Reader
		switch {
		case dry:
			src = &zeros{e.Size}
		case e.Data != nil:
			src = bytes.NewReader(e.Data)
		default:
			f, err := os.Open(e.Path)
			if err != nil {
				return err
			}
			_, err = io.CopyN(fw, ctxReader{ctx, f}, e.Size)
			f.Close()
			if err != nil {
				return fmt.Errorf("%s: %w (it changed while being packed)", e.Name, err)
			}
			continue
		}
		if _, err := io.Copy(fw, src); err != nil {
			return err
		}
	}
	return zw.Close()
}

// ctxReader stops a long copy when the client goes away.
type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (c ctxReader) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}
