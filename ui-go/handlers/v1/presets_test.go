package v1

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/paths"
)

func TestPresets(t *testing.T) {
	e := setup(t)
	repo, _ := filepath.Abs(filepath.Join("..", "..", ".."))
	paths.ToolkitRoot = repo // the config check reads the toolkit's Python
	base := map[string]any{"job": "extension", "config": map[string]any{"name": "x", "process": []any{map[string]any{
		"type": "diffusion_trainer", "loss_type": "mse", "train": map[string]any{"steps": 10}}}}}

	want(t, e.call(t, "PUT", "/v1/presets/bad!id", map[string]any{"name": "x", "base": base}), 400, "invalid_request")
	want(t, e.call(t, "PUT", "/v1/presets/p1", map[string]any{"name": "", "base": base}), 400, "invalid_request")
	want(t, e.call(t, "PUT", "/v1/presets/p1", map[string]any{"name": "x"}), 422, "invalid_config")
	want(t, e.call(t, "PUT", "/v1/presets/p1", map[string]any{"id": "p2", "name": "x", "base": base}), 400, "invalid_request")

	r := e.call(t, "PUT", "/v1/presets/p1", map[string]any{"name": "Krea LoKr", "base": base, "builtIn": false, "updated_at": 1700000000000})
	want(t, r, 200, "")
	m := r.json(t)
	if p := m["preset"].(map[string]any); p["updated_at"] != float64(1700000000000) || p["id"] != "p1" {
		t.Fatalf("put: %s", r.body)
	}
	// loss_type at the process level is a key the trainer ignores
	if pr := m["problems"].([]any); len(pr) != 1 || !strings.Contains(pr[0].(string), "loss_type") {
		t.Fatalf("problems: %v", m["problems"])
	}
	e.call(t, "PUT", "/v1/presets/p0", map[string]any{"name": "Another", "base": base})
	if items := e.call(t, "GET", "/v1/presets", nil).json(t)["items"].([]any); len(items) != 2 || items[0].(map[string]any)["name"] != "Another" {
		t.Fatalf("list: %v", items)
	}
	if g := e.call(t, "GET", "/v1/presets/p1", nil).json(t); g["name"] != "Krea LoKr" {
		t.Fatalf("get: %v", g)
	}

	want(t, e.call(t, "DELETE", "/v1/presets/p1?updated_at=1800000000000", nil), 204, "")
	want(t, e.call(t, "GET", "/v1/presets/p1", nil), 404, "not_found")
	if items := e.call(t, "GET", "/v1/presets", nil).json(t)["items"].([]any); len(items) != 1 {
		t.Fatalf("list after delete: %v", items)
	}
	all := e.call(t, "GET", "/v1/presets?deleted=1", nil).json(t)["items"].([]any)
	var tomb map[string]any
	for _, it := range all {
		if it.(map[string]any)["id"] == "p1" {
			tomb = it.(map[string]any)
		}
	}
	if tomb == nil || tomb["deleted"] != true || tomb["updated_at"] != float64(1800000000000) {
		t.Fatalf("tombstone: %v", all)
	}
}
