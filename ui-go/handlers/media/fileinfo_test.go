package media

import (
	"encoding/binary"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func safetensors(header string) []byte {
	b := make([]byte, 8, 8+len(header)+16)
	binary.LittleEndian.PutUint64(b, uint64(len(header)))
	return append(append(b, header...), make([]byte, 16)...)
}

func TestFilesInfo(t *testing.T) {
	f := setup(t)
	p := filepath.Join(f.out, "job", "lora.safetensors")
	mustWrite(t, p, safetensors(`{"__metadata__":{"ss_rank":"16","format":"pt"},`+
		`"b.up":{"dtype":"BF16","shape":[4,2],"data_offsets":[0,16]},`+
		`"a.down":{"dtype":"F32","shape":[2,1]},`+
		`"B.alpha":{"dtype":"F32","shape":[]}}`))

	resp, body := postJSON(t, f, "/api/files/info", map[string]any{"filePath": encodeURIComponent(p)})
	if resp.StatusCode != 200 {
		t.Fatalf("info: %d %q", resp.StatusCode, body)
	}
	// keys in upstream's order, metadata as written
	for _, k := range []string{`{"filePath":`, `"fileSize":`, `"headerSize":`, `"metadata":{"ss_rank":"16","format":"pt"}`, `"tensors":`, `"tensorCount":3`, `"totalParams":11`, `"totalBytes":28`, `"dtypeCounts":{"BF16":`} {
		if !strings.Contains(body, k) {
			t.Errorf("body lacks %s: %s", k, body)
		}
	}
	var got struct {
		Tensors []struct {
			Name  string
			Numel float64
			Bytes float64
		}
		DtypeCounts map[string]struct{ Count, Numel, Bytes float64 }
	}
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatal(err)
	}
	// localeCompare order: a < b < B
	if len(got.Tensors) != 3 || got.Tensors[0].Name != "a.down" || got.Tensors[1].Name != "B.alpha" && got.Tensors[1].Name != "b.up" {
		t.Errorf("tensors: %+v", got.Tensors)
	}
	if c := got.DtypeCounts["F32"]; c.Count != 2 || c.Numel != 3 || c.Bytes != 12 {
		t.Errorf("F32 counts: %+v", c)
	}

	for _, c := range []struct {
		path string
		code int
		msg  string
	}{
		{filepath.Join(f.out, "job", "model.safetensors"), 400, "Not a safetensors file (bad header length)"},
		{filepath.Join(f.out, "job", "missing.safetensors"), 404, "File not found"},
		{filepath.Join(f.out, "job"), 400, "Not a file"},
		{filepath.Join(f.root, "outside-secret.png"), 403, "Access denied"},
	} {
		resp, body := postJSON(t, f, "/api/files/info", map[string]any{"filePath": c.path})
		if resp.StatusCode != c.code || body != c.msg {
			t.Errorf("%s: %d %q, want %d %q", c.path, resp.StatusCode, body, c.code, c.msg)
		}
	}
	mustWrite(t, p, safetensors(`{not json`))
	if resp, body := postJSON(t, f, "/api/files/info", map[string]any{"filePath": p}); resp.StatusCode != 400 || body != "Not a safetensors file (unparseable header)" {
		t.Errorf("bad json: %d %q", resp.StatusCode, body)
	}
	if resp, body := postJSON(t, f, "/api/files/info", map[string]any{}); resp.StatusCode != 400 || body != "filePath is required" {
		t.Errorf("missing: %d %q", resp.StatusCode, body)
	}
}
