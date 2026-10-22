package v1

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCaptionCheckAndFix(t *testing.T) {
	e := setup(t)
	want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": "set"}), 201, "")
	dir := filepath.Join(e.datasets, "set")
	files := map[string]string{
		"a.png": "[trigger] as a young woman in a red coat.",
		"b.png": "Jihan as a young woman on a beach, wearing a",
		"c.png": "",
		"d.png": "A woman who appears to be smiling at the camera.",
		"e.png": "[Trigger] in a garden.",
		"f.png": "A woman who appears to be smiling at the camera.",
	}
	for f, c := range files {
		os.WriteFile(filepath.Join(dir, f), pngBytes(2, 2), 0o644)
		if c != "" {
			os.WriteFile(filepath.Join(dir, strings.TrimSuffix(f, ".png")+".txt"), []byte(c), 0o644)
		}
	}
	m := e.call(t, "GET", "/v1/datasets/set/captions/check?trigger=jihan", nil).json(t)
	got := map[string]string{}
	for _, it := range m["items"].([]any) {
		it := it.(map[string]any)
		var fl []string
		for _, f := range it["flags"].([]any) {
			fl = append(fl, f.(string))
		}
		got[it["file"].(string)] = strings.Join(fl, ",")
	}
	expect := map[string]string{"a.png": "", "b.png": "trigger_case,truncated", "c.png": "missing",
		"d.png": "no_trigger,hedging,duplicate", "e.png": "trigger_case", "f.png": "no_trigger,hedging,duplicate"}
	for f, w := range expect {
		if got[f] != w {
			t.Errorf("%s: flags %q, want %q", f, got[f], w)
		}
	}
	if m["flagged"] != float64(5) {
		t.Errorf("flagged %v", m["flagged"])
	}

	ops := []any{map[string]any{"op": "fix_trigger_case"}, map[string]any{"op": "ensure_trigger"},
		map[string]any{"op": "fill_missing", "caption": "a photo of [trigger]"}}
	r := e.call(t, "POST", "/v1/datasets/set/captions/fix", map[string]any{"ops": ops, "trigger": "jihan"})
	if n := len(r.json(t)["changes"].([]any)); n != 5 {
		t.Fatalf("preview changes %d: %s", n, r.body)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "e.txt")); string(b) != "[Trigger] in a garden." {
		t.Fatal("preview wrote")
	}
	e.call(t, "POST", "/v1/datasets/set/captions/fix", map[string]any{"ops": ops, "trigger": "jihan", "apply": true})
	for f, w := range map[string]string{"b.txt": "jihan as a young woman on a beach, wearing a", "c.txt": "a photo of [trigger]",
		"d.txt": "[trigger] A woman who appears to be smiling at the camera.", "e.txt": "[trigger] in a garden."} {
		if b, _ := os.ReadFile(filepath.Join(dir, f)); string(b) != w {
			t.Errorf("%s = %q, want %q", f, b, w)
		}
	}
	want(t, e.call(t, "POST", "/v1/datasets/set/captions/fix", map[string]any{"ops": []any{map[string]any{"op": "nope"}}}), 400, "invalid_request")
}
