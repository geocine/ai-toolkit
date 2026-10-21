package store

import (
	"context"
	"database/sql"
)

// Setting mirrors model Settings in ui/prisma/schema.prisma.
type Setting struct {
	ID    int    `json:"id"`
	Key   string `json:"key"`
	Value string `json:"value"`
}

var settingsColumns = []string{"id", "key", "value"}

// GetSetting returns the value for key and whether a row exists.
func GetSetting(ctx context.Context, key string) (string, bool, error) {
	var v string
	err := DB.QueryRowContext(ctx, `SELECT "value" FROM "Settings" WHERE "key" = ? LIMIT 1`, key).Scan(&v)
	if err == sql.ErrNoRows {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return v, true, nil
}

// ListSettings returns every setting in rowid order (Prisma findMany).
func ListSettings(ctx context.Context) ([]Setting, error) {
	rows, err := DB.QueryContext(ctx, `SELECT "id", "key", "value" FROM "Settings"`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Setting{}
	for rows.Next() {
		var s Setting
		if err := rows.Scan(&s.ID, &s.Key, &s.Value); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// UpsertSetting sets key to value, creating the row when missing (Prisma upsert).
func UpsertSetting(ctx context.Context, key, value string) error {
	_, err := DB.ExecContext(ctx, `INSERT INTO "Settings" ("key", "value") VALUES (?, ?)
		ON CONFLICT("key") DO UPDATE SET "value" = excluded."value"`, key, value)
	return mapErr(err)
}
