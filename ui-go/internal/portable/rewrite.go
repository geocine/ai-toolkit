package portable

import (
	"bytes"
	"encoding/json"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/jsonx"
)

const (
	datasetsScheme = "aitk://datasets"
	trainingScheme = "aitk://training"
)

// slashed is a path with forward slashes, cleaned.
func slashed(p string) string { return path.Clean(strings.ReplaceAll(p, `\`, "/")) }

// samePrefix compares path text, case-insensitively on Windows.
func samePrefix(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	if runtime.GOOS == "windows" {
		return strings.EqualFold(s[:len(prefix)], prefix)
	}
	return s[:len(prefix)] == prefix
}

// under returns s's slash-separated path relative to root when s is root or
// inside it.
func under(root, s string) (string, bool) {
	if root == "" || !(filepath.IsAbs(s) || strings.HasPrefix(s, "/")) {
		return "", false
	}
	r, p := strings.TrimSuffix(slashed(root), "/"), slashed(s)
	if len(p) == len(r) && samePrefix(p, r) {
		return "", true
	}
	if !samePrefix(p, r+"/") {
		return "", false
	}
	return p[len(r)+1:], true
}

type rewriter func(string) (string, bool)

func walk(v any, fn rewriter) any {
	switch x := v.(type) {
	case *jsonx.Object:
		for _, k := range x.Keys() {
			val, _ := x.Get(k)
			x.Set(k, walk(val, fn))
		}
		return x
	case []any:
		for i := range x {
			x[i] = walk(x[i], fn)
		}
		return x
	case string:
		if out, ok := fn(x); ok {
			return out
		}
	}
	return v
}

func compact(v any) (json.RawMessage, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, b); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ToPortable rewrites a job config's machine-specific paths: anything under
// the datasets root becomes aitk://datasets/<name>/<rel>, anything under the
// training root aitk://training/<rel>. It returns the config and the dataset
// names it references.
func ToPortable(config string, datasetsRoot, trainingRoot string) (json.RawMessage, []string, error) {
	v, err := jsonx.Parse([]byte(config))
	if err != nil {
		return nil, nil, err
	}
	names := map[string]bool{}
	v = walk(v, func(s string) (string, bool) {
		if rel, ok := under(datasetsRoot, s); ok && rel != "" {
			names[strings.SplitN(rel, "/", 2)[0]] = true
			return datasetsScheme + "/" + rel, true
		}
		if rel, ok := under(trainingRoot, s); ok {
			if rel == "" {
				return trainingScheme, true
			}
			return trainingScheme + "/" + rel, true
		}
		return "", false
	})
	out, err := compact(v)
	if err != nil {
		return nil, nil, err
	}
	list := make([]string, 0, len(names))
	for n := range names {
		list = append(list, n)
	}
	sort.Strings(list)
	return out, list, nil
}

// FromPortable maps aitk:// paths onto this server's roots and points every
// process's training_folder at the local training root.
func FromPortable(config json.RawMessage, datasetsRoot, trainingRoot string) (string, error) {
	v, err := jsonx.Parse(config)
	if err != nil {
		return "", err
	}
	v = walk(v, func(s string) (string, bool) { return localPath(s, datasetsRoot, trainingRoot) })
	if o, ok := v.(*jsonx.Object); ok {
		if cfg := o.Object("config"); cfg != nil {
			for _, p := range cfg.Array("process") {
				if po, ok := p.(*jsonx.Object); ok {
					if _, has := po.Get("training_folder"); has {
						po.Set("training_folder", trainingRoot)
					}
				}
			}
		}
	}
	out, err := compact(v)
	return string(out), err
}

// localPath resolves one aitk:// path on this server.
func localPath(s, datasetsRoot, trainingRoot string) (string, bool) {
	for _, m := range []struct{ scheme, root string }{{datasetsScheme, datasetsRoot}, {trainingScheme, trainingRoot}} {
		if s == m.scheme {
			return m.root, true
		}
		if rel, ok := strings.CutPrefix(s, m.scheme+"/"); ok {
			return filepath.Join(m.root, filepath.FromSlash(rel)), true
		}
	}
	return "", false
}
