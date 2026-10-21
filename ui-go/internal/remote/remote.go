// Package remote is storage outside the GPU machine: where checkpoints are
// uploaded and where portable jobs live so another server (on another
// provider) can continue them. A remote is a row in the fork database; its
// kind picks the client (fs here; s3 and hf register themselves).
package remote

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
)

// Object is a stored file.
type Object struct {
	Key      string    `json:"key"`
	Size     int64     `json:"size"`
	Modified time.Time `json:"modified"`
}

// Progress receives the bytes transferred so far.
type Progress func(done int64)

// Client talks to one remote. Keys are slash-separated paths relative to the
// remote's configured prefix (bucket prefix, path in repo, base folder).
type Client interface {
	// Put uploads a local file to key and returns a URL (or URI) for it.
	Put(ctx context.Context, key, localPath string, progress Progress) (string, error)
	// Get downloads key to localPath (written atomically).
	Get(ctx context.Context, key, localPath string, progress Progress) error
	// PutBytes / GetBytes are for small documents such as manifests.
	PutBytes(ctx context.Context, key string, data []byte) error
	GetBytes(ctx context.Context, key string) ([]byte, error)
	// Stat returns the object, or nil and ErrNotFound when it is missing.
	Stat(ctx context.Context, key string) (*Object, error)
	// List returns every object under prefix ("" for all), sorted by key.
	List(ctx context.Context, prefix string) ([]Object, error)
	Delete(ctx context.Context, key string) error
	// Check verifies the remote is reachable and writable.
	Check(ctx context.Context) error
}

// ErrNotFound is returned (wrapped is fine) for a missing key.
var ErrNotFound = errors.New("remote object not found")

// Factory builds a client from a remote's config and secret (both JSON).
type Factory func(config, secret json.RawMessage) (Client, error)

var (
	factoriesMu sync.RWMutex
	factories   = map[string]Factory{}
)

// RegisterKind makes a kind available ("fs", "s3", "hf").
func RegisterKind(kind string, f Factory) {
	factoriesMu.Lock()
	factories[kind] = f
	factoriesMu.Unlock()
}

// Kinds lists the registered kinds.
func Kinds() []string {
	factoriesMu.RLock()
	defer factoriesMu.RUnlock()
	out := make([]string, 0, len(factories))
	for k := range factories {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Remote is a configured target. Secret is never serialized.
type Remote struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Kind      string          `json:"kind"`
	Config    json.RawMessage `json:"config"`
	CreatedAt time.Time       `json:"created_at"`
	Secret    json.RawMessage `json:"-"`
}

// NewClient builds the client for r.
func NewClient(r *Remote) (Client, error) {
	factoriesMu.RLock()
	f, ok := factories[r.Kind]
	factoriesMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("unknown remote kind %q", r.Kind)
	}
	return f(r.Config, r.Secret)
}

// Create stores a remote after checking its client can be built.
func Create(ctx context.Context, name, kind string, config, secret json.RawMessage) (*Remote, error) {
	if len(config) == 0 {
		config = json.RawMessage(`{}`)
	}
	if len(secret) == 0 {
		secret = json.RawMessage(`{}`)
	}
	r := &Remote{ID: state.NewID("rem"), Name: name, Kind: kind, Config: config, Secret: secret,
		CreatedAt: state.Time(state.NowMs())}
	if _, err := NewClient(r); err != nil {
		return nil, err
	}
	_, err := state.DB.ExecContext(ctx, `INSERT INTO remotes (id, name, kind, config, secret, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
		r.ID, r.Name, r.Kind, string(r.Config), string(r.Secret), r.CreatedAt.UnixMilli())
	if err != nil {
		return nil, err
	}
	return r, nil
}

func scan(row interface{ Scan(...any) error }) (*Remote, error) {
	var r Remote
	var config, secret string
	var created int64
	if err := row.Scan(&r.ID, &r.Name, &r.Kind, &config, &secret, &created); err != nil {
		return nil, err
	}
	r.Config, r.Secret, r.CreatedAt = json.RawMessage(config), json.RawMessage(secret), state.Time(created)
	return &r, nil
}

// Get finds a remote by id or name; nil when there is none.
func Get(ctx context.Context, idOrName string) (*Remote, error) {
	r, err := scan(state.DB.QueryRowContext(ctx, `SELECT id, name, kind, config, secret, created_at FROM remotes
		WHERE id = ? OR name = ? LIMIT 1`, idOrName, idOrName))
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return r, err
}

// List returns every remote by name.
func List(ctx context.Context) ([]*Remote, error) {
	rows, err := state.DB.QueryContext(ctx, `SELECT id, name, kind, config, secret, created_at FROM remotes ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Remote{}
	for rows.Next() {
		r, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// Delete removes a remote; false when there was none.
func Delete(ctx context.Context, id string) (bool, error) {
	res, err := state.DB.ExecContext(ctx, `DELETE FROM remotes WHERE id = ?`, id)
	if err != nil {
		return false, err
	}
	n, _ := res.RowsAffected()
	return n > 0, nil
}

// Open is Get + NewClient.
func Open(ctx context.Context, idOrName string) (*Remote, Client, error) {
	r, err := Get(ctx, idOrName)
	if err != nil {
		return nil, nil, err
	}
	if r == nil {
		return nil, nil, fmt.Errorf("remote %q: %w", idOrName, ErrNotFound)
	}
	c, err := NewClient(r)
	return r, c, err
}
