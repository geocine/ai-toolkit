package v1

import (
	"encoding/json"
	"net/http"
	"regexp"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/templates"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Studio training presets kept on the server (/v1/presets), so MCP agents can
// list and use the presets the user tuned in the studio, and presets an agent
// writes reach the studio. The studio keeps its own copy (it outlives a pod)
// and syncs it with every server it connects to: the newer updated_at wins,
// and a deletion is kept as a tombstone so it syncs too. A preset is the
// studio's object (id, name, description, builtIn, base: the full ai-toolkit
// job config, defaults, promptSets, captionTemplates, notes, edited); the
// server checks id, name and base and keeps the rest as is.

const presetPrefix = "preset."

var presetID = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,80}$`)

func presetKey(id string) string { return presetPrefix + id }

// readPreset returns the stored preset object (tombstones included), or nil.
func readPreset(r *http.Request, id string) (map[string]any, error) {
	v, ok, err := state.Setting(r.Context(), presetKey(id))
	if err != nil || !ok {
		return nil, err
	}
	var p map[string]any
	if err := json.Unmarshal([]byte(v), &p); err != nil {
		return nil, err
	}
	return p, nil
}

func deleted(p map[string]any) bool { b, _ := p["deleted"].(bool); return b }

// listPresets is GET /v1/presets?deleted=1 (tombstones too, for syncing).
func listPresets(w http.ResponseWriter, r *http.Request) {
	rows, err := state.DB.QueryContext(r.Context(), `SELECT value FROM settings WHERE key LIKE ?`, presetPrefix+"%")
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	defer rows.Close()
	all := r.URL.Query().Get("deleted") == "1" || r.URL.Query().Get("deleted") == "true"
	items := []map[string]any{}
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			v1api.Internal(w, r, err)
			return
		}
		var p map[string]any
		if json.Unmarshal([]byte(v), &p) == nil && (all || !deleted(p)) {
			items = append(items, p)
		}
	}
	sort.Slice(items, func(i, j int) bool {
		a, _ := items[i]["name"].(string)
		b, _ := items[j]["name"].(string)
		return strings.ToLower(a) < strings.ToLower(b)
	})
	v1api.Items(w, items)
}

func getPreset(w http.ResponseWriter, r *http.Request) {
	p, err := readPreset(r, r.PathValue("id"))
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if p == nil || deleted(p) {
		v1api.Fail(w, v1api.CodeNotFound, "no preset "+r.PathValue("id"))
		return
	}
	v1api.OK(w, p)
}

// putPreset is PUT /v1/presets/{id}: create or replace. updated_at (ms) is
// kept when given (the studio's own clock, so syncs compare equal), else now.
// The answer carries the config check of base as problems.
func putPreset(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !presetID.MatchString(id) {
		v1api.Fail(w, v1api.CodeInvalidRequest, "preset id must be 1-80 letters, digits, '.', '_' or '-'")
		return
	}
	var p map[string]any
	if !v1api.Decode(w, r, &p) {
		return
	}
	if pid, ok := p["id"]; ok && pid != id {
		v1api.Fail(w, v1api.CodeInvalidRequest, "the body's id does not match the path")
		return
	}
	p["id"] = id
	if name, _ := p["name"].(string); strings.TrimSpace(name) == "" {
		v1api.Fail(w, v1api.CodeInvalidRequest, "name is required")
		return
	}
	base, _ := p["base"].(map[string]any)
	if base == nil {
		v1api.Fail(w, v1api.CodeInvalidConfig, "base (the ai-toolkit job config) is required")
		return
	}
	raw, _ := json.Marshal(base)
	if _, err := checkConfig(raw); err != nil {
		v1api.Respond(w, r, err)
		return
	}
	if ts, ok := p["updated_at"].(float64); !ok || ts <= 0 {
		p["updated_at"] = state.NowMs()
	}
	delete(p, "deleted")
	b, _ := json.Marshal(p)
	if err := state.SetSetting(r.Context(), presetKey(id), string(b)); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	problems := []string{}
	for _, pr := range templates.Check(base).Problems {
		problems = append(problems, pr.Text())
	}
	v1api.OK(w, map[string]any{"preset": p, "problems": problems})
}

// deletePreset is DELETE /v1/presets/{id}?updated_at=: leaves a tombstone
// so the deletion reaches the studio and other servers' syncs.
func deletePreset(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !presetID.MatchString(id) {
		v1api.Fail(w, v1api.CodeInvalidRequest, "invalid preset id")
		return
	}
	at := state.NowMs()
	if v := r.URL.Query().Get("updated_at"); v != "" {
		var n json.Number = json.Number(v)
		if i, err := n.Int64(); err == nil && i > 0 {
			at = i
		}
	}
	b, _ := json.Marshal(map[string]any{"id": id, "deleted": true, "updated_at": at})
	if err := state.SetSetting(r.Context(), presetKey(id), string(b)); err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.NoContent(w)
}
