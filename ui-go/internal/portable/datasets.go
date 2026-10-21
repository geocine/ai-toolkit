package portable

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/remote"
)

// ErrDatasetConflict: a local dataset of that name exists with other content.
var ErrDatasetConflict = errors.New("dataset exists locally with different contents")

// scanDataset lists a dataset folder's files (hidden entries such as
// .thumbs and .zips are generated, so skipped) with their hashes.
func scanDataset(ctx context.Context, dir string) ([]DatasetFile, error) {
	var out []DatasetFile
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p != dir && strings.HasPrefix(d.Name(), ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		sha, size, err := hashFile(ctx, p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		out = append(out, DatasetFile{Path: filepath.ToSlash(rel), SHA256: sha, Size: size})
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, err
}

// pushDataset uploads the dataset's missing blobs and its manifest.
func pushDataset(ctx context.Context, c remote.Client, datasetsRoot, name string) error {
	if err := ValidName(name); err != nil {
		return err
	}
	dir := filepath.Join(datasetsRoot, name)
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		return fmt.Errorf("dataset %s not found at %s", name, dir)
	}
	files, err := scanDataset(ctx, dir)
	if err != nil {
		return err
	}
	have := map[string]bool{}
	if objs, err := c.List(ctx, "blobs/"); err == nil {
		for _, o := range objs {
			have[strings.TrimPrefix(o.Key, "blobs/")] = true
		}
	}
	for _, f := range files {
		if have[f.SHA256] {
			continue
		}
		if _, err := c.Stat(ctx, blobKey(f.SHA256)); err == nil {
			have[f.SHA256] = true
			continue
		} else if !errors.Is(err, remote.ErrNotFound) {
			return err
		}
		if _, err := c.Put(ctx, blobKey(f.SHA256), filepath.Join(dir, filepath.FromSlash(f.Path)), nil); err != nil {
			return fmt.Errorf("upload %s/%s: %w", name, f.Path, err)
		}
		have[f.SHA256] = true
	}
	if files == nil {
		files = []DatasetFile{}
	}
	b, err := json.MarshalIndent(DatasetManifest{Name: name, Files: files}, "", "  ")
	if err != nil {
		return err
	}
	if old, err := c.GetBytes(ctx, datasetKey(name)); err == nil && bytes.Equal(old, b) {
		return nil
	}
	return c.PutBytes(ctx, datasetKey(name), b)
}

// restoreDataset brings datasets/<name> to this server. An identical local
// copy is reused; a different one is a conflict. Every blob is verified.
func restoreDataset(ctx context.Context, c remote.Client, datasetsRoot, name string) error {
	if err := ValidName(name); err != nil {
		return err
	}
	raw, err := c.GetBytes(ctx, datasetKey(name))
	if err != nil {
		return fmt.Errorf("dataset %s manifest: %w", name, err)
	}
	var m DatasetManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return fmt.Errorf("dataset %s manifest: %w", name, err)
	}
	for _, f := range m.Files {
		if !safeRel(f.Path) {
			return fmt.Errorf("dataset %s lists an unsafe path %q", name, f.Path)
		}
	}
	dir := filepath.Join(datasetsRoot, name)
	if st, err := os.Stat(dir); err == nil && st.IsDir() {
		local, err := scanDataset(ctx, dir)
		if err != nil {
			return err
		}
		if sameFiles(local, m.Files) {
			return nil
		}
		return fmt.Errorf("%s: %w", name, ErrDatasetConflict)
	}
	if err := os.MkdirAll(datasetsRoot, 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(datasetsRoot, ".import-"+name+"-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	for _, f := range m.Files {
		dst := filepath.Join(tmp, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := getVerified(ctx, c, blobKey(f.SHA256), dst, f.SHA256); err != nil {
			return fmt.Errorf("dataset %s/%s: %w", name, f.Path, err)
		}
	}
	return os.Rename(tmp, dir)
}

func sameFiles(a, b []DatasetFile) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Path != b[i].Path || a[i].SHA256 != b[i].SHA256 {
			return false
		}
	}
	return true
}

// safeRel accepts slash-separated relative paths that stay inside their folder.
func safeRel(p string) bool {
	if p == "" || strings.HasPrefix(p, "/") || strings.Contains(p, `\`) || filepath.IsAbs(p) {
		return false
	}
	for _, seg := range strings.Split(p, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return false
		}
	}
	return true
}

// ErrCorrupt: a downloaded file does not match the manifest's sha256.
var ErrCorrupt = errors.New("checksum mismatch")

// getVerified downloads key to dst and checks its sha256.
func getVerified(ctx context.Context, c remote.Client, key, dst, sha string) error {
	if err := c.Get(ctx, key, dst, nil); err != nil {
		return err
	}
	got, _, err := hashFile(ctx, dst)
	if err != nil {
		return err
	}
	if got != sha {
		os.Remove(dst)
		return fmt.Errorf("%s: %w (want %s, got %s)", key, ErrCorrupt, sha[:12], got[:12])
	}
	return nil
}
