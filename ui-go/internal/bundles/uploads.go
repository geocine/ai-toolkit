package bundles

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
)

// Chunked uploads: a client too far behind a proxy body limit to send a
// multi-GB bundle in one request sends it in Content-Range chunks to an
// upload session; imports then read the session's file. Sessions live as
// files (<id>.part + <id>.json) in the staging folder and expire a day
// after their last write.
const (
	ChunkSize    = 32 << 20  // what clients are told to send
	MaxChunk     = 128 << 20 // the most one PUT may carry
	UploadExpiry = 24 * time.Hour
)

// Upload is one upload session.
type Upload struct {
	ID        string    `json:"id"`
	Filename  string    `json:"filename"`
	Size      int64     `json:"size"` // -1 while a server-side download does not know it
	Received  int64     `json:"received"`
	Complete  bool      `json:"complete"`
	ChunkSize int64     `json:"chunk_size"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	ExpiresAt time.Time `json:"expires_at"`
}

// Upload errors.
var (
	ErrNoUpload = errors.New("upload not found")
	ErrOffset   = errors.New("chunk does not start at or before the bytes received")
	ErrOverflow = errors.New("chunk runs past the upload's size")
)

// StagingDir is where uploads, downloads and bundles being built live: a
// hidden folder in the training root, the volume big files go to anyway
// (job names cannot start with a dot).
func StagingDir(ctx context.Context) (string, error) {
	root, err := settings.TrainingFolder(ctx)
	if err != nil {
		return "", err
	}
	return filepath.Join(root, ".aitk-bundles"), nil
}

func uploadsDir(ctx context.Context) (string, error) {
	dir, err := StagingDir(ctx)
	if err != nil {
		return "", err
	}
	dir = filepath.Join(dir, "uploads")
	return dir, os.MkdirAll(dir, 0o755)
}

func validID(id string) bool {
	if !strings.HasPrefix(id, "bupl_") || len(id) > 64 {
		return false
	}
	for _, c := range id[5:] {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}

var (
	uploadMu    sync.Mutex
	uploadLocks = map[string]*sync.Mutex{}
)

func lockUpload(id string) func() {
	uploadMu.Lock()
	l, ok := uploadLocks[id]
	if !ok {
		l = &sync.Mutex{}
		uploadLocks[id] = l
	}
	uploadMu.Unlock()
	l.Lock()
	return l.Unlock
}

type uploadMeta struct {
	ID        string    `json:"id"`
	Filename  string    `json:"filename"`
	Size      int64     `json:"size"`
	CreatedAt time.Time `json:"created_at"`
}

// CreateUpload opens a session for size bytes (-1: unknown, for downloads).
func CreateUpload(ctx context.Context, size int64, filename string) (*Upload, error) {
	dir, err := uploadsDir(ctx)
	if err != nil {
		return nil, err
	}
	m := uploadMeta{ID: state.NewID("bupl"), Filename: filepath.Base(strings.ReplaceAll(filename, `\`, "/")),
		Size: size, CreatedAt: time.Now().UTC().Truncate(time.Millisecond)}
	if m.Filename == "." || m.Filename == "/" {
		m.Filename = ""
	}
	if err := os.WriteFile(filepath.Join(dir, m.ID+".part"), nil, 0o644); err != nil {
		return nil, err
	}
	if err := writeMeta(dir, m); err != nil {
		return nil, err
	}
	return GetUpload(ctx, m.ID)
}

func writeMeta(dir string, m uploadMeta) error {
	b, _ := json.Marshal(m)
	tmp := filepath.Join(dir, m.ID+".json.tmp")
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, m.ID+".json"))
}

func readUpload(dir, id string) (*Upload, error) {
	if !validID(id) {
		return nil, ErrNoUpload
	}
	b, err := os.ReadFile(filepath.Join(dir, id+".json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNoUpload
	}
	if err != nil {
		return nil, err
	}
	var m uploadMeta
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	st, err := os.Stat(filepath.Join(dir, id+".part"))
	if err != nil {
		return nil, ErrNoUpload
	}
	u := &Upload{ID: m.ID, Filename: m.Filename, Size: m.Size, Received: st.Size(), ChunkSize: ChunkSize,
		CreatedAt: m.CreatedAt, UpdatedAt: st.ModTime().UTC().Truncate(time.Millisecond)}
	if u.UpdatedAt.Before(u.CreatedAt) {
		u.UpdatedAt = u.CreatedAt
	}
	u.ExpiresAt = u.UpdatedAt.Add(UploadExpiry)
	u.Complete = u.Size >= 0 && u.Received == u.Size
	return u, nil
}

// GetUpload returns a session (ErrNoUpload when missing or expired).
func GetUpload(ctx context.Context, id string) (*Upload, error) {
	dir, err := uploadsDir(ctx)
	if err != nil {
		return nil, err
	}
	u, err := readUpload(dir, id)
	if err == nil && time.Now().After(u.ExpiresAt) {
		removeUpload(dir, id)
		return nil, ErrNoUpload
	}
	return u, err
}

// ListUploads returns the live sessions, newest first.
func ListUploads(ctx context.Context) ([]*Upload, error) {
	dir, err := uploadsDir(ctx)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	out := []*Upload{}
	for _, e := range entries {
		id, ok := strings.CutSuffix(e.Name(), ".json")
		if !ok {
			continue
		}
		if u, err := GetUpload(ctx, id); err == nil {
			out = append(out, u)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.After(out[j].CreatedAt) })
	return out, nil
}

// UploadPath is the session's data file.
func UploadPath(ctx context.Context, id string) (string, error) {
	if _, err := GetUpload(ctx, id); err != nil {
		return "", err
	}
	dir, err := uploadsDir(ctx)
	return filepath.Join(dir, id+".part"), err
}

// WriteChunk writes r at offset. Writing at or before the bytes received is
// idempotent (a retried chunk rewrites the same bytes); a gap is ErrOffset.
func WriteChunk(ctx context.Context, id string, offset int64, r io.Reader) (*Upload, error) {
	unlock := lockUpload(id)
	defer unlock()
	u, err := GetUpload(ctx, id)
	if err != nil {
		return nil, err
	}
	if offset < 0 || offset > u.Received {
		return u, ErrOffset
	}
	dir, _ := uploadsDir(ctx)
	f, err := os.OpenFile(filepath.Join(dir, id+".part"), os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		f.Close()
		return nil, err
	}
	limit := int64(MaxChunk)
	if u.Size >= 0 {
		limit = min(limit, u.Size-offset)
	}
	_, err = io.Copy(f, ctxReader{ctx, io.LimitReader(r, limit)})
	if err == nil {
		// anything beyond the limit is an overflow, not silently dropped
		var one [1]byte
		if m, _ := r.Read(one[:]); m > 0 {
			err = ErrOverflow
		}
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	now := time.Now()
	os.Chtimes(filepath.Join(dir, id+".part"), now, now)
	got, gerr := GetUpload(ctx, id)
	if err != nil {
		return got, err
	}
	return got, gerr
}

// FinishUpload fixes the size of a session a server-side download filled.
func FinishUpload(ctx context.Context, id string) (*Upload, error) {
	dir, err := uploadsDir(ctx)
	if err != nil {
		return nil, err
	}
	u, err := readUpload(dir, id)
	if err != nil {
		return nil, err
	}
	if err := writeMeta(dir, uploadMeta{ID: u.ID, Filename: u.Filename, Size: u.Received, CreatedAt: u.CreatedAt}); err != nil {
		return nil, err
	}
	return GetUpload(ctx, id)
}

// DeleteUpload removes a session.
func DeleteUpload(ctx context.Context, id string) error {
	dir, err := uploadsDir(ctx)
	if err != nil {
		return err
	}
	if _, err := readUpload(dir, id); err != nil {
		return err
	}
	return removeUpload(dir, id)
}

func removeUpload(dir, id string) error {
	if err := os.Remove(filepath.Join(dir, id+".part")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Remove(filepath.Join(dir, id+".json"))
}

// ExpireUploads removes sessions past their expiry and stray build files.
func ExpireUploads(ctx context.Context) {
	if _, err := ListUploads(ctx); err != nil { // GetUpload removes expired ones
		return
	}
	dir, err := uploadsDir(ctx)
	if err != nil {
		return
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		id, ok := strings.CutSuffix(e.Name(), ".part")
		if !ok {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, id+".json")); errors.Is(err, os.ErrNotExist) {
			if info, err := e.Info(); err == nil && time.Since(info.ModTime()) > UploadExpiry {
				os.Remove(filepath.Join(dir, e.Name()))
			}
		}
	}
}

// ParseContentRange reads "bytes <start>-<end>/<total|*>".
func ParseContentRange(h string) (start, end, total int64, err error) {
	var t string
	if _, err = fmt.Sscanf(h, "bytes %d-%d/%s", &start, &end, &t); err != nil {
		return 0, 0, 0, fmt.Errorf("Content-Range must be \"bytes <start>-<end>/<total>\"")
	}
	total = -1
	if t != "*" {
		if _, err = fmt.Sscan(t, &total); err != nil {
			return 0, 0, 0, fmt.Errorf("bad Content-Range total %q", t)
		}
	}
	if start < 0 || end < start {
		return 0, 0, 0, fmt.Errorf("bad Content-Range %q", h)
	}
	return start, end, total, nil
}
