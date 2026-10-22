package v1

import (
	"os"
	"path/filepath"
	"testing"
)

func TestChosenAndSamplePrompts(t *testing.T) {
	e := setup(t)
	cfg := map[string]any{"job": "extension", "config": map[string]any{"name": "lora1", "process": []any{map[string]any{
		"type": "diffusion_trainer", "trigger_word": "jihan",
		"sample": map[string]any{"seed": 42, "walk_seed": true, "samples": []any{
			map[string]any{"prompt": "[trigger] at the beach"}, map[string]any{"prompt": "[trigger] in a cafe"}}},
	}}}}
	r := e.call(t, "POST", "/v1/jobs", map[string]any{"name": "lora1", "config": cfg})
	want(t, r, 201, "")
	id := r.json(t)["id"].(string)
	dir := filepath.Join(e.training, "lora1")
	os.MkdirAll(filepath.Join(dir, "samples"), 0o755)
	os.WriteFile(filepath.Join(dir, "lora1_000000100.safetensors"), []byte("w"), 0o644)
	os.WriteFile(filepath.Join(dir, "samples", "1700000000000__000000100_1.jpg"), []byte("j"), 0o644)

	items := e.call(t, "GET", "/v1/jobs/"+id+"/samples", nil).json(t)["items"].([]any)
	s := items[0].(map[string]any)
	if s["prompt"] != "jihan in a cafe" || s["seed"] != float64(43) {
		t.Fatalf("sample: %v", s)
	}

	r = e.call(t, "PUT", "/v1/jobs/"+id+"/chosen", map[string]any{"step": 100, "strength": 0.9, "note": "best likeness"})
	want(t, r, 200, "")
	c := r.json(t)["chosen"].(map[string]any)
	if c["file"] != "lora1_000000100.safetensors" || c["step"] != float64(100) || c["strength"] != 0.9 {
		t.Fatalf("chosen: %v", c)
	}
	want(t, e.call(t, "PUT", "/v1/jobs/"+id+"/chosen", map[string]any{}), 400, "invalid_request")
	if got := e.call(t, "DELETE", "/v1/jobs/"+id+"/chosen", nil).json(t)["chosen"]; got != nil {
		t.Fatalf("cleared: %v", got)
	}
}
