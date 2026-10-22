package bundles

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/portable"
)

// Localize turns a bundle's job config into this server's: every path under
// a dataset's Source (datasets maps Source -> local folder) points at the
// restored folder, aitk:// paths resolve to this server's roots, every
// process trains into trainingRoot, and the job is named name.
func Localize(config json.RawMessage, datasets map[string]string, datasetsRoot, trainingRoot, name string) (string, error) {
	v, err := jsonx.Parse(config)
	if err != nil {
		return "", err
	}
	sources := make([]string, 0, len(datasets))
	for s := range datasets {
		if s != "" {
			sources = append(sources, s)
		}
	}
	// longest first, so a/b wins over a
	sort.Slice(sources, func(i, j int) bool { return len(sources[i]) > len(sources[j]) })
	v = portable.RewriteStrings(v, func(s string) (string, bool) {
		norm := strings.TrimRight(strings.ReplaceAll(s, `\`, "/"), "/")
		for _, src := range sources {
			key := strings.TrimRight(strings.ReplaceAll(src, `\`, "/"), "/")
			if norm == key {
				return datasets[src], true
			}
			if rest, ok := strings.CutPrefix(norm, key+"/"); ok {
				return filepath.Join(datasets[src], filepath.FromSlash(rest)), true
			}
		}
		return "", false
	})
	raw, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	local, err := portable.FromPortable(raw, datasetsRoot, trainingRoot)
	if err != nil {
		return "", err
	}
	cfg, err := jsonx.ParseObject([]byte(local))
	if err != nil {
		return "", err
	}
	if inner := cfg.Object("config"); inner != nil {
		old, _ := inner.Get("name")
		inner.Set("name", name)
		for _, p := range inner.Array("process") {
			if po, ok := p.(*jsonx.Object); ok {
				po.Set("training_folder", trainingRoot)
			}
		}
		// meta sits beside config (the UI's JSON) or inside it (older yaml)
		for _, meta := range []*jsonx.Object{cfg.Object("meta"), inner.Object("meta")} {
			if meta == nil {
				continue
			}
			if mn, ok := meta.Get("name"); ok && mn == old {
				meta.Set("name", name)
			}
		}
	}
	out, err := json.Marshal(cfg)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, out); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// RenameCheckpoint maps a job folder path of the old job name to the new
// one: saves <old>_<step>.safetensors, <old>.safetensors and diffusers save
// folders <old>_<step>/... take the new name, so ai-toolkit's
// get_latest_save_path (which globs <name>*) finds them. Other paths come
// back unchanged.
func RenameCheckpoint(rel, oldName, newName string) string {
	if oldName == newName {
		return rel
	}
	top, rest, nested := strings.Cut(rel, "/")
	if _, ok := checkpointOf(top, oldName, nested); !ok {
		return rel
	}
	top = newName + strings.TrimPrefix(top, oldName)
	if nested {
		return top + "/" + rest
	}
	return top
}
