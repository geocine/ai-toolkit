package v1

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/geocine/aitoolkit/internal/paths"
)

func TestCaptionDataset(t *testing.T) {
	e := setup(t)
	repo, _ := filepath.Abs(filepath.Join("..", "..", ".."))
	if _, err := os.Stat(filepath.Join(repo, "ui", "src", "helpers", "captionOptions.ts")); err != nil {
		t.Skipf("upstream ui not found: %v", err)
	}
	paths.ToolkitRoot = repo // upstream's captioners; the db and folders stay in the temp root

	want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": "set"}), 201, "")
	want(t, e.call(t, "POST", "/v1/datasets/set/caption", map[string]any{"captioner": "nope"}), 400, "invalid_request")
	want(t, e.call(t, "POST", "/v1/datasets/missing/caption", map[string]any{}), 404, "not_found")

	r := e.call(t, "POST", "/v1/datasets/set/caption", map[string]any{"captioner": "Qwen3VLCaptioner",
		"model": "Qwen/Qwen3-VL-2B-Instruct", "overrides": map[string]any{"caption.max_res": 768}})
	want(t, r, 201, "")
	m := r.json(t)
	dir := filepath.Join(e.datasets, "set")
	if m["type"] != "caption" || m["ref"] != dir || m["status"] != "queued" {
		t.Fatalf("job: %s", r.body)
	}
	p0 := m["config"].(map[string]any)["config"].(map[string]any)["process"].([]any)[0].(map[string]any)
	c := p0["caption"].(map[string]any)
	if p0["type"] != "Qwen3VLCaptioner" || c["path_to_caption"] != dir || c["model_name_or_path"] != "Qwen/Qwen3-VL-2B-Instruct" ||
		c["max_res"] != float64(768) || c["recaption"] != false {
		t.Fatalf("config: %v", p0)
	}
	// the captioner's own defaults: images only, its image prompt
	if exts := c["extensions"].([]any); len(exts) != 5 || exts[0] != "jpg" {
		t.Errorf("extensions: %v", exts)
	}

	want(t, e.call(t, "POST", "/v1/datasets/set/caption", map[string]any{}), 409, "conflict")
	d := e.call(t, "GET", "/v1/datasets/set", nil).json(t)
	if cj, _ := d["caption_job"].(map[string]any); cj == nil || cj["id"] != m["id"] {
		t.Fatalf("caption_job: %v", d["caption_job"])
	}
}
