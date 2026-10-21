package remote

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// fs is a folder: a mounted network volume shared between machines, or a
// local path for tests. Config: {"path": "/mnt/shared/aitk"}.
func init() { RegisterKind("fs", newFS) }

type fsClient struct{ base string }

func newFS(config, _ json.RawMessage) (Client, error) {
	var c struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(config, &c); err != nil {
		return nil, fmt.Errorf("fs remote config: %w", err)
	}
	if c.Path == "" {
		return nil, errors.New(`fs remote needs config.path`)
	}
	abs, err := filepath.Abs(c.Path)
	if err != nil {
		return nil, err
	}
	return &fsClient{base: abs}, nil
}

// file maps a key to a path, refusing keys that would leave the base.
func (c *fsClient) file(key string) (string, error) {
	clean := path.Clean("/" + strings.ReplaceAll(key, `\`, "/"))
	if clean == "/" {
		return "", fmt.Errorf("invalid key %q", key)
	}
	return filepath.Join(c.base, filepath.FromSlash(clean[1:])), nil
}

func copyFile(dst, src string, progress Progress) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".part-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	var r io.Reader = in
	if progress != nil {
		r = &countingReader{r: in, fn: progress}
	}
	if _, err := io.Copy(tmp, r); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), dst)
}

type countingReader struct {
	r  io.Reader
	n  int64
	fn Progress
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	c.fn(c.n)
	return n, err
}

func (c *fsClient) Put(_ context.Context, key, localPath string, progress Progress) (string, error) {
	dst, err := c.file(key)
	if err != nil {
		return "", err
	}
	if err := copyFile(dst, localPath, progress); err != nil {
		return "", err
	}
	return "file://" + filepath.ToSlash(dst), nil
}

func (c *fsClient) Get(_ context.Context, key, localPath string, progress Progress) error {
	src, err := c.file(key)
	if err != nil {
		return err
	}
	if _, err := os.Stat(src); errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("%s: %w", key, ErrNotFound)
	}
	return copyFile(localPath, src, progress)
}

func (c *fsClient) PutBytes(_ context.Context, key string, data []byte) error {
	dst, err := c.file(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

func (c *fsClient) GetBytes(_ context.Context, key string) ([]byte, error) {
	src, err := c.file(key)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(src)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%s: %w", key, ErrNotFound)
	}
	return b, err
}

func (c *fsClient) Stat(_ context.Context, key string) (*Object, error) {
	p, err := c.file(key)
	if err != nil {
		return nil, err
	}
	st, err := os.Stat(p)
	if errors.Is(err, fs.ErrNotExist) || (err == nil && st.IsDir()) {
		return nil, fmt.Errorf("%s: %w", key, ErrNotFound)
	}
	if err != nil {
		return nil, err
	}
	return &Object{Key: key, Size: st.Size(), Modified: st.ModTime().UTC()}, nil
}

func (c *fsClient) List(_ context.Context, prefix string) ([]Object, error) {
	out := []Object{}
	err := filepath.WalkDir(c.base, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() || strings.HasPrefix(d.Name(), ".part-") || strings.HasSuffix(d.Name(), ".tmp") {
			return nil
		}
		rel, _ := filepath.Rel(c.base, p)
		key := filepath.ToSlash(rel)
		if !strings.HasPrefix(key, prefix) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		out = append(out, Object{Key: key, Size: info.Size(), Modified: info.ModTime().UTC()})
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out, err
}

func (c *fsClient) Delete(_ context.Context, key string) error {
	p, err := c.file(key)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return nil
}

func (c *fsClient) Check(ctx context.Context) error {
	if err := os.MkdirAll(c.base, 0o755); err != nil {
		return err
	}
	probe := ".aitk-check"
	if err := c.PutBytes(ctx, probe, []byte("ok")); err != nil {
		return err
	}
	return c.Delete(ctx, probe)
}
