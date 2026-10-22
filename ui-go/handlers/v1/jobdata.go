package v1

import (
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/v1api"
)

const (
	logDefaultLimit = 64 << 10
	logMaxLimit     = 1 << 20
)

// jobLog returns a window of log.txt by byte offset. A negative offset counts
// from the end. The window is trimmed to whole UTF-8 characters so clients can
// append chunks; next_offset is where the next read starts.
func jobLog(w http.ResponseWriter, r *http.Request) {
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	offset, limit := int64(0), int64(logDefaultLimit)
	if s := q.Get("offset"); s != "" {
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			v1api.Fail(w, v1api.CodeInvalidRequest, "offset must be an integer")
			return
		}
		offset = n
	}
	if s := q.Get("limit"); s != "" {
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil || n <= 0 {
			v1api.Fail(w, v1api.CodeInvalidRequest, "limit must be a positive integer")
			return
		}
		limit = min(n, logMaxLimit)
	}
	folder, err := jobFolder(r.Context(), j.Name)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	f, err := os.Open(filepath.Join(folder, "log.txt"))
	if os.IsNotExist(err) {
		v1api.OK(w, map[string]any{"offset": 0, "next_offset": 0, "size": 0, "text": ""})
		return
	}
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	size := st.Size()
	if offset < 0 {
		offset = max(size+offset, 0)
	}
	offset = min(offset, size)
	n := min(limit, size-offset)
	// read a few bytes either side to land on character boundaries
	buf := make([]byte, n+utf8.UTFMax)
	read, err := f.ReadAt(buf, offset)
	if err != nil && err != io.EOF {
		v1api.Internal(w, r, err)
		return
	}
	buf = buf[:read]
	start := 0
	for start < len(buf) && start < utf8.UTFMax && !utf8.RuneStart(buf[start]) {
		start++
	}
	end := min(int(n), len(buf))
	if offset+int64(end) < size {
		// drop a trailing character the window cut in half
		for i := end - 1; i >= max(start, end-utf8.UTFMax); i-- {
			if utf8.RuneStart(buf[i]) {
				if !utf8.FullRune(buf[i:end]) {
					end = i
				}
				break
			}
		}
	}
	if end < start {
		end = start
	}
	v1api.OK(w, map[string]any{
		"offset":      offset + int64(start),
		"next_offset": offset + int64(end),
		"size":        size,
		"text":        strings.ToValidUTF8(string(buf[start:end]), "�"),
	})
}

type lossPoint struct {
	Step     int      `json:"step"`
	Value    float64  `json:"value"`
	WallTime *float64 `json:"wall_time"`
}

// jobLoss reads the trainer's loss log (toolkit/logging_aitk.py schema).
func jobLoss(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	afterStep, limit, stride := -1, 2000, 1
	for name, dst := range map[string]*int{"after_step": &afterStep, "limit": &limit, "stride": &stride} {
		if s := q.Get(name); s != "" {
			n, err := strconv.Atoi(s)
			if err != nil {
				v1api.Fail(w, v1api.CodeInvalidRequest, name+" must be an integer")
				return
			}
			*dst = n
		}
	}
	limit = max(1, min(limit, 20000))
	stride = max(1, stride)
	folder, err := jobFolder(ctx, j.Name)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	empty := map[string]any{"keys": []string{}, "key": nil, "points": []lossPoint{}}
	path := filepath.Join(folder, "loss_log.db")
	if _, err := os.Stat(path); err != nil {
		v1api.OK(w, empty)
		return
	}
	db, err := sql.Open("sqlite", store.FileURI(path)+"?mode=ro&_pragma=busy_timeout(5000)")
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	defer db.Close()
	keys, err := lossKeys(ctx, db)
	if err != nil {
		v1api.OK(w, empty) // a log the trainer has not initialized yet
		return
	}
	key := q.Get("key")
	if key == "" {
		key = defaultLossKey(keys)
	}
	if key == "" {
		v1api.OK(w, map[string]any{"keys": keys, "key": nil, "points": []lossPoint{}})
		return
	}
	rows, err := db.QueryContext(ctx, `SELECT m.step, m.value_real, s.wall_time FROM metrics m
		LEFT JOIN steps s ON s.step = m.step
		WHERE m.key = ? AND m.step > ? AND m.value_real IS NOT NULL AND (m.step % ?) = 0
		ORDER BY m.step LIMIT ?`, key, afterStep, stride, limit)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	defer rows.Close()
	points := []lossPoint{}
	for rows.Next() {
		var p lossPoint
		var wall sql.NullFloat64
		if err := rows.Scan(&p.Step, &p.Value, &wall); err != nil {
			v1api.Internal(w, r, err)
			return
		}
		if wall.Valid {
			p.WallTime = &wall.Float64
		}
		points = append(points, p)
	}
	v1api.OK(w, map[string]any{"keys": keys, "key": key, "points": points})
}

func lossKeys(ctx context.Context, db *sql.DB) ([]string, error) {
	rows, err := db.QueryContext(ctx, `SELECT key FROM metric_keys ORDER BY key`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	keys := []string{}
	for rows.Next() {
		var k string
		if err := rows.Scan(&k); err != nil {
			return nil, err
		}
		keys = append(keys, k)
	}
	return keys, rows.Err()
}

// defaultLossKey prefers "loss", then the first key naming a loss
// (ai-toolkit logs "loss/loss"), then the first key.
func defaultLossKey(keys []string) string {
	for _, k := range keys {
		if k == "loss" {
			return k
		}
	}
	for _, k := range keys {
		if strings.Contains(strings.ToLower(k), "loss") {
			return k
		}
	}
	if len(keys) > 0 {
		return keys[0]
	}
	return ""
}

func jobCheckpoints(w http.ResponseWriter, r *http.Request) {
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	folder, err := jobFolder(r.Context(), j.Name)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	cps, err := artifacts.Checkpoints(folder)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if r.URL.Query().Get("hash") == "true" {
		for i := range cps {
			h, err := cachedHash(r.Context(), cps[i].Path)
			if err != nil {
				v1api.Internal(w, r, err)
				return
			}
			cps[i].SHA256 = h
		}
	}
	v1api.Items(w, cps)
}

func jobSamples(w http.ResponseWriter, r *http.Request) {
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	folder, err := jobFolder(r.Context(), j.Name)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	samples, err := artifacts.Samples(folder)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	samplePrompts(j.JobConfig, samples)
	v1api.Items(w, samples)
}

// samplePrompts fills each sample's prompt and seed from the job config's
// sample section, the way ai-toolkit renders them: [trigger] becomes the
// trigger word, and with walk_seed each prompt's seed is seed + its index.
func samplePrompts(config string, samples []artifacts.Sample) {
	var cfg struct {
		Config struct {
			Process []struct {
				TriggerWord string `json:"trigger_word"`
				Sample      struct {
					Samples []struct {
						Prompt string `json:"prompt"`
						Seed   *int   `json:"seed"`
					} `json:"samples"`
					Prompts  []string `json:"prompts"`
					Seed     *int     `json:"seed"`
					WalkSeed bool     `json:"walk_seed"`
				} `json:"sample"`
			} `json:"process"`
		} `json:"config"`
	}
	if json.Unmarshal([]byte(config), &cfg) != nil || len(cfg.Config.Process) == 0 {
		return
	}
	p := cfg.Config.Process[0]
	prompts, seeds := []string{}, []*int{}
	for _, s := range p.Sample.Samples {
		prompts, seeds = append(prompts, s.Prompt), append(seeds, s.Seed)
	}
	if len(prompts) == 0 {
		for _, s := range p.Sample.Prompts {
			prompts, seeds = append(prompts, s), append(seeds, nil)
		}
	}
	for i := range samples {
		idx := samples[i].Index
		if idx == nil || *idx < 0 || *idx >= len(prompts) {
			continue
		}
		text := prompts[*idx]
		if p.TriggerWord != "" {
			text = strings.ReplaceAll(strings.ReplaceAll(text, "[trigger]", p.TriggerWord), "[name]", p.TriggerWord)
		}
		samples[i].Prompt = &text
		switch {
		case seeds[*idx] != nil:
			samples[i].Seed = seeds[*idx]
		case p.Sample.Seed != nil:
			s := *p.Sample.Seed
			if p.Sample.WalkSeed {
				s += *idx
			}
			samples[i].Seed = &s
		}
	}
}

// chooseCheckpoint is PUT /v1/jobs/{id}/chosen {step | file, strength?,
// note?}: marks the checkpoint picked as the run's result; DELETE clears it.
func chooseCheckpoint(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	j, ok := loadJob(w, r)
	if !ok {
		return
	}
	if r.Method == http.MethodDelete {
		if err := state.SetChosen(ctx, j.ID, nil); err != nil {
			v1api.Internal(w, r, err)
			return
		}
		respondJob(w, r, http.StatusOK, j.ID)
		return
	}
	var b struct {
		Step     *int    `json:"step"`
		File     string  `json:"file"`
		Strength float64 `json:"strength"`
		Note     string  `json:"note"`
	}
	if !v1api.Decode(w, r, &b) {
		return
	}
	folder, err := jobFolder(ctx, j.Name)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	cps, err := artifacts.Checkpoints(folder)
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	var pick *artifacts.Checkpoint
	for i := range cps {
		c := &cps[i]
		if (b.File != "" && c.Name == b.File) || (b.File == "" && b.Step != nil && c.Step != nil && *c.Step == *b.Step) {
			pick = c
		}
	}
	if b.File == "" && b.Step == nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "step or file is required")
		return
	}
	// a step whose file is not on this server (kept only on a remote) is
	// still a valid pick
	c := &state.Chosen{Step: b.Step, File: b.File, Strength: b.Strength, Note: b.Note, At: state.NowMs()}
	if pick != nil {
		c.Step, c.File = pick.Step, pick.Name
	}
	if err := state.SetChosen(ctx, j.ID, c); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	respondJob(w, r, http.StatusOK, j.ID)
}

// cachedHash is the file's sha256, cached in file_hashes by path, size and
// mtime so a checkpoint is hashed once.
func cachedHash(ctx context.Context, path string) (string, error) {
	st, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	var sum string
	err = state.DB.QueryRowContext(ctx, `SELECT sha256 FROM file_hashes WHERE path = ? AND size = ? AND mtime_ns = ?`,
		path, st.Size(), st.ModTime().UnixNano()).Scan(&sum)
	if err == nil {
		return sum, nil
	}
	if err != sql.ErrNoRows {
		return "", err
	}
	sum, err = artifacts.SHA256File(path)
	if err != nil {
		return "", err
	}
	_, err = state.DB.ExecContext(ctx, `INSERT INTO file_hashes (path, size, mtime_ns, sha256) VALUES (?, ?, ?, ?)
		ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime_ns = excluded.mtime_ns, sha256 = excluded.sha256`,
		path, st.Size(), st.ModTime().UnixNano(), sum)
	return sum, err
}
