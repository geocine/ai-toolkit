package jobfiles

import (
	"context"
	"database/sql"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/store"
)

// The loss log is the trainer's own sqlite file (toolkit/logging_aitk.py):
// steps(step, wall_time), metric_keys(key, first/last_seen_step),
// metrics(step, key, value_real, value_text).

func openLossDB(path string, readOnly bool) (*sql.DB, error) {
	dsn := store.FileURI(path) + "?_pragma=busy_timeout(30000)"
	if readOnly {
		dsn += "&mode=ro"
	}
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	return db, nil
}

// bindNum binds a JS number the way node-sqlite3 does: integers as INTEGER,
// other values as REAL. NaN binds as NULL (sqlite stores NaN as NULL anyway).
func bindNum(f float64) any {
	if math.IsNaN(f) {
		return nil
	}
	if f == math.Trunc(f) && math.Abs(f) < 1<<63 {
		return int64(f)
	}
	return f
}

type lossPoint struct {
	Step     any `json:"step"`
	WallTime any `json:"wall_time"`
	Value    any `json:"value"`
}

type lossEmpty struct {
	Keys   []string    `json:"keys"`
	Key    string      `json:"key"`
	Points []lossPoint `json:"points"`
}

type lossResponse struct {
	Key    string      `json:"key"`
	Keys   []any       `json:"keys"`
	Points []lossPoint `json:"points"`
}

// sqlNum turns a scanned sqlite number into its JSON form (non-finite -> null).
func sqlNum(v any) any {
	switch x := v.(type) {
	case float64:
		return jsonNum(x)
	case []byte:
		return string(x)
	}
	return v
}

// lossGet returns one metric's points. Query: key (default "loss"), limit
// (default 2000, capped at 20000), since_step, stride (>= 1). The numbers go
// through Number() like upstream, so odd values behave the same in sqlite.
// Mirrors jobs/[jobID]/loss/route.ts.
func lossGet(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}
	logPath := filepath.Join(folder, "loss_log.db")
	if _, err := os.Stat(logPath); err != nil {
		httpx.OK(w, lossEmpty{Keys: []string{}, Key: "loss", Points: []lossPoint{}})
		return
	}

	q := r.URL.Query()
	key := "loss"
	if q.Has("key") {
		key = q.Get("key")
	}
	limit := 2000.0
	if q.Has("limit") {
		limit = jsNumber(q.Get("limit"))
	}
	limit = jsMin(limit, 20000)
	var sinceStep any
	if q.Has("since_step") {
		sinceStep = bindNum(jsNumber(q.Get("since_step")))
	}
	stride := 1.0
	if q.Has("stride") {
		stride = jsNumber(q.Get("stride"))
	}
	stride = jsMax(1, stride)

	db, err := openLossDB(logPath, true)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	defer db.Close()
	ctx := r.Context()

	keys, err := lossKeys(ctx, db)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	rows, err := db.QueryContext(ctx, `
      SELECT
        m.step AS step,
        s.wall_time AS wall_time,
        m.value_real AS value,
        m.value_text AS value_text
      FROM metrics m
      JOIN steps s ON s.step = m.step
      WHERE m.key = ?
        AND (? IS NULL OR m.step > ?)
        AND (m.step % ?) = 0
      ORDER BY m.step ASC
      LIMIT ?
      `, key, sinceStep, sinceStep, bindNum(stride), bindNum(limit))
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	defer rows.Close()
	points := []lossPoint{}
	for rows.Next() {
		var step, wallTime, value any
		var valueText sql.NullString
		if err := rows.Scan(&step, &wallTime, &value, &valueText); err != nil {
			httpx.Internal(w, r, err)
			return
		}
		// value ?? (value_text ? Number(value_text) : null)
		var v any
		switch {
		case value != nil:
			v = sqlNum(toFloatIfNumber(value))
		case valueText.Valid && valueText.String != "":
			v = jsonNum(jsNumber(valueText.String))
		}
		points = append(points, lossPoint{Step: sqlNum(step), WallTime: sqlNum(wallTime), Value: v})
	}
	if err := rows.Err(); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	httpx.OK(w, lossResponse{Key: key, Keys: keys, Points: points})
}

func toFloatIfNumber(v any) any {
	if i, ok := v.(int64); ok {
		return float64(i)
	}
	return v
}

func lossKeys(ctx context.Context, db *sql.DB) ([]any, error) {
	rows, err := db.QueryContext(ctx, `SELECT key FROM metric_keys ORDER BY key ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	keys := []any{}
	for rows.Next() {
		var k any
		if err := rows.Scan(&k); err != nil {
			return nil, err
		}
		if b, ok := k.([]byte); ok {
			k = string(b)
		}
		keys = append(keys, k)
	}
	return keys, rows.Err()
}

// jsMin / jsMax are Math.min / Math.max for two numbers (NaN wins).
func jsMin(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	return math.Min(a, b)
}

func jsMax(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	return math.Max(a, b)
}

// lossDelete deletes every logged step in [min_step, max_step] across all
// metric keys (the loss graph's "Delete Selected Range").
func lossDelete(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}

	// an unparsable body falls through to validation as {}
	var body any = map[string]any{}
	if parsed, err := parseBody(r); err == nil {
		body = parsed
	}
	if body == nil {
		// JSON null: upstream reads null.min_step and throws
		httpx.Internal(w, r, errTypeError)
		return
	}
	var minRaw, maxRaw any = undefined, undefined
	if m, ok := body.(map[string]any); ok {
		if v, ok := m["min_step"]; ok {
			minRaw = v
		}
		if v, ok := m["max_step"]; ok {
			maxRaw = v
		}
	}
	minStep, maxStep := numberOf(minRaw), numberOf(maxRaw)
	if !isFinite(minStep) || !isFinite(maxStep) || minStep > maxStep {
		httpx.Error(w, http.StatusBadRequest, "min_step and max_step must be numbers with min_step <= max_step")
		return
	}

	logPath := filepath.Join(folder, "loss_log.db")
	if _, err := os.Stat(logPath); err != nil {
		httpx.Error(w, http.StatusNotFound, "No loss log for this job")
		return
	}

	if err := deleteRange(r.Context(), logPath, minStep, maxStep); err != nil {
		log.Printf("Error deleting loss range: %v", err)
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}
	// +0 turns -0 into 0, which is how JSON.stringify writes it
	httpx.OK(w, map[string]any{"ok": true, "min_step": minStep + 0, "max_step": maxStep + 0})
}

type undefinedT struct{}

// undefined marks a property missing from the body (Number(undefined) is NaN).
var undefined = undefinedT{}

var errTypeError = &typeError{"Cannot read properties of null (reading 'min_step')"}

type typeError struct{ msg string }

func (e *typeError) Error() string { return e.msg }

func numberOf(v any) float64 {
	if _, ok := v.(undefinedT); ok {
		return math.NaN()
	}
	return toNumber(v)
}

func isFinite(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

func deleteRange(ctx context.Context, logPath string, minStep, maxStep float64) error {
	db, err := openLossDB(logPath, false)
	if err != nil {
		return err
	}
	defer db.Close()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	lo, hi := bindNum(minStep), bindNum(maxStep)
	stmts := []struct {
		sql  string
		args []any
	}{
		// the FK cascade on metrics only fires with PRAGMA foreign_keys=ON,
		// so delete metrics explicitly rather than relying on it
		{`DELETE FROM metrics WHERE step >= ? AND step <= ?;`, []any{lo, hi}},
		{`DELETE FROM steps WHERE step >= ? AND step <= ?;`, []any{lo, hi}},
		{`DELETE FROM metric_keys WHERE NOT EXISTS (SELECT 1 FROM metrics WHERE metrics.key = metric_keys.key);`, nil},
		{`UPDATE metric_keys SET
          first_seen_step = (SELECT MIN(step) FROM metrics WHERE metrics.key = metric_keys.key),
          last_seen_step = (SELECT MAX(step) FROM metrics WHERE metrics.key = metric_keys.key);`, nil},
	}
	for _, s := range stmts {
		if _, err := tx.ExecContext(ctx, s.sql, s.args...); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}
