package state

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/hex"
	"sync"
	"time"
)

// APIKey is a named credential; the secret itself is only shown on create.
type APIKey struct {
	ID         string     `json:"id"`
	Name       string     `json:"name"`
	Prefix     string     `json:"prefix"`
	CreatedAt  time.Time  `json:"created_at"`
	LastUsedAt *time.Time `json:"last_used_at"`
}

const keyPrefix = "aitk_"

func hashKey(secret string) string {
	h := sha256.Sum256([]byte(secret))
	return hex.EncodeToString(h[:])
}

// CreateKey stores a new key and returns it with its secret (shown once).
func CreateKey(ctx context.Context, name string) (APIKey, string, error) {
	secret := keyPrefix + RandomString(32)
	k := APIKey{ID: NewID("key"), Name: name, Prefix: secret[:len(keyPrefix)+6], CreatedAt: Time(NowMs())}
	_, err := DB.ExecContext(ctx, `INSERT INTO api_keys (id, name, prefix, hash, created_at) VALUES (?, ?, ?, ?, ?)`,
		k.ID, k.Name, k.Prefix, hashKey(secret), k.CreatedAt.UnixMilli())
	if err != nil {
		return APIKey{}, "", err
	}
	invalidateKeyCount()
	return k, secret, nil
}

// ListKeys returns the active keys, oldest first.
func ListKeys(ctx context.Context) ([]APIKey, error) {
	rows, err := DB.QueryContext(ctx, `SELECT id, name, prefix, created_at, last_used_at FROM api_keys
		WHERE revoked_at IS NULL ORDER BY created_at`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []APIKey{}
	for rows.Next() {
		var k APIKey
		var created int64
		var used sql.NullInt64
		if err := rows.Scan(&k.ID, &k.Name, &k.Prefix, &created, &used); err != nil {
			return nil, err
		}
		k.CreatedAt = Time(created)
		if used.Valid {
			t := Time(used.Int64)
			k.LastUsedAt = &t
		}
		out = append(out, k)
	}
	return out, rows.Err()
}

// RevokeKey disables a key; false when there was no such active key.
func RevokeKey(ctx context.Context, id string) (bool, error) {
	res, err := DB.ExecContext(ctx, `UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL`, NowMs(), id)
	if err != nil {
		return false, err
	}
	invalidateKeyCount()
	n, _ := res.RowsAffected()
	return n > 0, nil
}

// VerifyKey reports whether secret is an active key, recording its use.
func VerifyKey(ctx context.Context, secret string) bool {
	if DB == nil || len(secret) < len(keyPrefix) || secret[:len(keyPrefix)] != keyPrefix {
		return false
	}
	want := hashKey(secret)
	var id, got string
	err := DB.QueryRowContext(ctx, `SELECT id, hash FROM api_keys WHERE hash = ? AND revoked_at IS NULL`, want).Scan(&id, &got)
	if err != nil || subtle.ConstantTimeCompare([]byte(got), []byte(want)) != 1 {
		return false
	}
	touchKey(id)
	return true
}

var (
	touchMu   sync.Mutex
	lastTouch = map[string]time.Time{}
)

// touchKey updates last_used_at at most once a minute per key.
func touchKey(id string) {
	touchMu.Lock()
	defer touchMu.Unlock()
	if time.Since(lastTouch[id]) < time.Minute {
		return
	}
	lastTouch[id] = time.Now()
	go DB.Exec(`UPDATE api_keys SET last_used_at = ? WHERE id = ?`, NowMs(), id)
}

var (
	countMu    sync.Mutex
	countValid bool
	countCache int
)

func invalidateKeyCount() {
	countMu.Lock()
	countValid = false
	countMu.Unlock()
}

// HasKeys reports whether any active key exists (then auth is required).
func HasKeys() bool {
	if DB == nil {
		return false
	}
	countMu.Lock()
	defer countMu.Unlock()
	if !countValid {
		if err := DB.QueryRow(`SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL`).Scan(&countCache); err != nil {
			return false
		}
		countValid = true
	}
	return countCache > 0
}
