package store

import (
	"database/sql/driver"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Time is a Prisma DateTime: stored as unix milliseconds, serialized like
// JavaScript's Date.toISOString() ("2026-01-02T03:04:05.678Z").
type Time struct{ time.Time }

const isoMillis = "2006-01-02T15:04:05.000Z"

// Now is the current time at the millisecond precision Prisma keeps.
func Now() Time { return Time{time.Now().UTC().Truncate(time.Millisecond)} }

func (t Time) MarshalJSON() ([]byte, error) {
	return []byte(`"` + t.UTC().Format(isoMillis) + `"`), nil
}

func (t *Time) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(b), `"`)
	if s == "null" || s == "" {
		t.Time = time.Time{}
		return nil
	}
	p, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return err
	}
	t.Time = p.UTC()
	return nil
}

func (t Time) Value() (driver.Value, error) { return t.UnixMilli(), nil }

// Scan accepts Prisma's integer milliseconds and, for rows written by other
// tools (e.g. sqlite's CURRENT_TIMESTAMP default), text timestamps.
func (t *Time) Scan(src any) error {
	switch v := src.(type) {
	case nil:
		t.Time = time.Time{}
	case int64:
		t.Time = time.UnixMilli(v).UTC()
	case float64:
		t.Time = time.UnixMilli(int64(v)).UTC()
	case time.Time:
		t.Time = v.UTC()
	case []byte:
		return t.scanString(string(v))
	case string:
		return t.scanString(v)
	default:
		return fmt.Errorf("store.Time: cannot scan %T", src)
	}
	return nil
}

var textLayouts = []string{
	time.RFC3339Nano,
	"2006-01-02 15:04:05.999999999-07:00",
	"2006-01-02 15:04:05.999999999",
	"2006-01-02 15:04:05",
	"2006-01-02",
}

func (t *Time) scanString(s string) error {
	s = strings.TrimSpace(s)
	if ms, err := strconv.ParseInt(s, 10, 64); err == nil {
		t.Time = time.UnixMilli(ms).UTC()
		return nil
	}
	for _, layout := range textLayouts {
		if p, err := time.Parse(layout, s); err == nil {
			t.Time = p.UTC()
			return nil
		}
	}
	return fmt.Errorf("store.Time: unrecognized timestamp %q", s)
}
