package bundles

import (
	"context"
	"errors"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/safepath"
)

// limited fails instead of truncating when an entry lies about its size.
type limited struct {
	r    io.Reader
	left int64
}

var errEntryTooLarge = errors.New("bundle entry is larger than it declares")

func (l *limited) Read(p []byte) (int, error) {
	if l.left <= 0 {
		var one [1]byte
		if n, err := l.r.Read(one[:]); n > 0 {
			return 0, errEntryTooLarge
		} else if err != nil {
			return 0, err
		}
		return 0, io.EOF
	}
	if int64(len(p)) > l.left {
		p = p[:l.left]
	}
	n, err := l.r.Read(p)
	l.left -= int64(n)
	return n, err
}

// Extract writes files into dir, which must exist (an import's temp folder),
// calling progress with the bytes of each chunk written.
func Extract(ctx context.Context, files []File, dir string, progress func(int64)) error {
	for _, f := range files {
		if err := ctx.Err(); err != nil {
			return err
		}
		dst := safepath.Resolve(filepath.Join(dir, filepath.FromSlash(f.Rel)), []string{dir})
		if dst == "" || dst == filepath.Clean(dir) {
			return bad("bundle entry leaves its folder: " + f.Zip.Name)
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		rc, err := f.Zip.Open()
		if err != nil {
			return bad("corrupt bundle entry " + f.Zip.Name + ": " + err.Error())
		}
		out, err := os.Create(dst)
		if err != nil {
			rc.Close()
			return err
		}
		src := io.Reader(ctxReader{ctx, &limited{r: rc, left: int64(f.Zip.UncompressedSize64)}})
		if progress != nil {
			src = &countReader{r: src, fn: progress}
		}
		_, err = io.Copy(out, src)
		rc.Close()
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			if errors.Is(err, errEntryTooLarge) || strings.Contains(err.Error(), "zip:") {
				return bad("extracting " + f.Zip.Name + ": " + err.Error())
			}
			return err
		}
		if !f.Zip.Modified.IsZero() {
			os.Chtimes(dst, time.Now(), f.Zip.Modified)
		}
	}
	return nil
}

type countReader struct {
	r  io.Reader
	fn func(int64)
}

func (c *countReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	if n > 0 {
		c.fn(int64(n))
	}
	return n, err
}

// SameFiles reports whether the dataset folder dir already holds exactly the
// bundle's dataset files (media and captions; hidden files and ai-toolkit's
// caches aside), compared by path, size and CRC-32.
func SameFiles(dir string, files []File) (bool, error) {
	want := map[string]File{}
	for _, f := range files {
		if !inCache(f.Rel) {
			want[f.Rel] = f
		}
	}
	local, err := walkFiles(dir, IsCacheDir)
	if err != nil {
		return false, err
	}
	if len(local) != len(want) {
		return false, nil
	}
	for _, rel := range local {
		f, ok := want[rel]
		if !ok {
			return false, nil
		}
		p := filepath.Join(dir, filepath.FromSlash(rel))
		st, err := os.Stat(p)
		if err != nil || uint64(st.Size()) != f.Zip.UncompressedSize64 {
			return false, err
		}
		sum, err := crcFile(p)
		if err != nil {
			return false, err
		}
		if sum != f.Zip.CRC32 {
			return false, nil
		}
	}
	return true, nil
}

func crcFile(p string) (uint32, error) {
	f, err := os.Open(p)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	h := crc32.NewIEEE()
	if _, err := io.Copy(h, f); err != nil {
		return 0, err
	}
	return h.Sum32(), nil
}
