package v1

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPreflight(t *testing.T) {
	e := setup(t)
	dir := filepath.Join(e.datasets, "faces")
	os.MkdirAll(dir, 0o755)
	for i, ext := range []string{".png", ".png", ".jpg", ".bmp", ".gif"} {
		os.WriteFile(filepath.Join(dir, "f"+string(rune('a'+i))+ext), pngBytes(2, 2), 0o644)
	}
	os.WriteFile(filepath.Join(dir, "fa.txt"), []byte("[trigger] smiling."), 0o644)
	os.WriteFile(filepath.Join(dir, "fb.txt"), []byte("Jihan by a window, wearing a"), 0o644)
	cfg := func(folder string) map[string]any {
		return map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
			"type": "diffusion_trainer", "trigger_word": "jihan",
			"train":    map[string]any{"steps": 3000},
			"save":     map[string]any{"save_every": 150, "max_step_saves_to_keep": 4},
			"model":    map[string]any{"name_or_path": "/nowhere/model"},
			"datasets": []any{map[string]any{"folder_path": folder}},
		}}}}
	}
	m := e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg(dir)}).json(t)
	var all []string
	for _, c := range m["checks"].([]any) {
		c := c.(map[string]any)
		all = append(all, c["level"].(string)+": "+c["message"].(string))
	}
	text := strings.Join(all, "\n")
	for _, want := range []string{
		"warn: datasets[0]: the trainer skips 1 .bmp, 1 .gif",
		"warn: datasets[0]: only 3 images",
		"warn: datasets[0]: 1 of 3 images have no caption",
		"warn: datasets[0]: 1 captions lack the trigger",
		"warn: datasets[0]: 1 captions end mid-sentence",
		"block: model.name_or_path /nowhere/model does not exist",
		"info: only the last 4 checkpoints stay on disk (from about step 2400 of 3000)",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in\n%s", want, text)
		}
	}
	if m["ok"] != false {
		t.Errorf("ok with a block: %v", m["ok"])
	}

	m = e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"config": cfg(filepath.Join(e.datasets, "nope"))}).json(t)
	if !strings.Contains(m["checks"].([]any)[0].(map[string]any)["message"].(string)+" "+jsonOf(m), "does not exist on this server") {
		t.Errorf("missing dataset: %v", m)
	}
	want(t, e.call(t, "POST", "/v1/jobs/preflight", map[string]any{}), 400, "invalid_request")
	want(t, e.call(t, "POST", "/v1/jobs/preflight", map[string]any{"job_id": "nope"}), 404, "not_found")
}

func jsonOf(v any) string {
	var b strings.Builder
	for _, c := range v.(map[string]any)["checks"].([]any) {
		b.WriteString(c.(map[string]any)["message"].(string) + "\n")
	}
	return b.String()
}
