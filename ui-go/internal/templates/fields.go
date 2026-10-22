package templates

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
)

// The field reference: every key the trainer reads from a process config,
// per section, read statically from the toolkit's Python (importing it needs
// torch). Config classes in toolkit/config_modules.py read their section with
// kwargs.get('key', default); the process itself reads its own keys with
// get_conf('key', default). See docs/headless.md, Templates.

// Field is one key a section's class reads.
type Field struct {
	Key     string `json:"key"`
	Default string `json:"default"`        // the Python default expression
	Type    string `json:"type,omitempty"` // the annotation, when there is one
	Doc     string `json:"doc,omitempty"`  // the comments at the read
}

// Section is a place in config.process[i] and the class that reads it.
type Section struct {
	Path   string  `json:"path"` // "process", "train", "datasets[]", "train.ema_config", ...
	Class  string  `json:"class"`
	Fields []Field `json:"fields"`
}

// Reference is GET /v1/templates/fields.
type Reference struct {
	Sections []Section `json:"sections"`
	Errors   []string  `json:"errors"`
}

// sectionSpecs are the sections jobs/process/BaseSDTrainProcess.py builds a
// config class from (the nested ones are read by the outer class).
var sectionSpecs = []struct{ path, class string }{
	{"train", "TrainConfig"},
	{"train.ema_config", "EMAConfig"},
	{"train.validation_config", "ValidationConfig"},
	{"model", "ModelConfig"},
	{"network", "NetworkConfig"},
	{"network.lorm", "LoRMConfig"},
	{"save", "SaveConfig"},
	{"sample", "SampleConfig"},
	{"sample.samples[]", "SampleItem"},
	{"first_sample", "SampleConfig"},
	{"logging", "LoggingConfig"},
	{"datasets[]", "DatasetConfig"},
	{"guidance", "GuidanceConfig"},
	{"embedding", "EmbeddingConfig"},
	{"decorator", "DecoratorConfig"},
	{"adapter", "AdapterConfig"},
}

// processScanDirs hold the process classes whose get_conf reads are the
// process-level keys.
var processScanDirs = []string{filepath.Join("jobs", "process"), "extensions_built_in"}

var (
	classLine = regexp.MustCompile(`^class (\w+)`)
	kwargsGet = regexp.MustCompile(`kwargs\.get\(\s*`)
	// a process reads its own keys with get_conf or straight off self.config
	processGet = regexp.MustCompile(`\b(?:get_conf|self\.config\.get)\(\s*`)
	kwDefault  = regexp.MustCompile(`^(\w+)\s*=\s*`)
	annotation = regexp.MustCompile(`self\.\w+\s*:\s*(.+?)\s*=\s*kwargs\.get\($`)
)

var (
	refMu    sync.Mutex
	refCache *Reference
	refFiles map[string]time.Time
	refRoot  string
)

// Fields returns the reference, rebuilt when a scanned file changed.
func Fields() Reference {
	refMu.Lock()
	defer refMu.Unlock()
	// a different toolkit (or one that was missing) reads afresh
	if refCache != nil && refRoot == paths.ToolkitRoot && filesFresh(refFiles) {
		return *refCache
	}
	ref, files := buildReference()
	refCache, refFiles, refRoot = &ref, files, paths.ToolkitRoot
	return ref
}

func filesFresh(files map[string]time.Time) bool {
	for p, t := range files {
		st, err := os.Stat(p)
		if err != nil || !st.ModTime().Equal(t) {
			return false
		}
	}
	return true
}

func buildReference() (Reference, map[string]time.Time) {
	ref := Reference{Sections: []Section{}, Errors: []string{}}
	files := map[string]time.Time{}
	read := func(p string) ([]byte, bool) {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, false
		}
		if st, err := os.Stat(p); err == nil {
			files[p] = st.ModTime()
		}
		return b, true
	}

	modules := filepath.Join(paths.ToolkitRoot, "toolkit", "config_modules.py")
	classes := map[string][]Field{}
	if b, ok := read(modules); ok {
		classes = pythonClasses(b)
	} else {
		ref.Errors = append(ref.Errors, "toolkit/config_modules.py not found under "+paths.ToolkitRoot)
	}

	// the process level: get_conf reads, minus the sections
	sectionKeys := map[string]bool{}
	for _, s := range sectionSpecs {
		sectionKeys[strings.TrimSuffix(strings.SplitN(s.path, ".", 2)[0], "[]")] = true
	}
	proc := map[string]Field{"type": {Key: "type", Default: "None", Doc: "the process class (sd_trainer, diffusion_trainer, ui_trainer, ...)"}}
	for _, dir := range processScanDirs {
		filepath.WalkDir(filepath.Join(paths.ToolkitRoot, dir), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if n := d.Name(); n == "__pycache__" || strings.HasPrefix(n, ".") || n == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(p, ".py") {
				return nil
			}
			b, ok := read(p)
			if !ok {
				return nil
			}
			for _, f := range pythonReads(b, processGet) {
				if !sectionKeys[f.Key] {
					if _, seen := proc[f.Key]; !seen {
						proc[f.Key] = f
					}
				}
			}
			return nil
		})
	}
	ref.Sections = append(ref.Sections, Section{Path: "process", Class: "", Fields: sortedFields(proc)})

	for _, s := range sectionSpecs {
		fields, ok := classes[s.class]
		if !ok {
			if len(classes) > 0 {
				ref.Errors = append(ref.Errors, fmt.Sprintf("class %s (for %s) not found in toolkit/config_modules.py", s.class, s.path))
			}
			continue
		}
		ref.Sections = append(ref.Sections, Section{Path: s.path, Class: s.class, Fields: fields})
	}
	return ref, files
}

func sortedFields(m map[string]Field) []Field {
	out := make([]Field, 0, len(m))
	for _, f := range m {
		out = append(out, f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

// pythonClasses maps each top-level class to the kwargs.get reads in its
// body, in source order (a key read twice is listed once).
func pythonClasses(b []byte) map[string][]Field {
	out := map[string][]Field{}
	lines := bytes.Split(b, []byte("\n"))
	offset := 0
	name, start := "", 0
	flush := func(end int) {
		if name == "" {
			return
		}
		seen := map[string]bool{}
		fields := []Field{}
		for _, f := range pythonReads(b[start:end], kwargsGet) {
			if !seen[f.Key] {
				seen[f.Key] = true
				fields = append(fields, f)
			}
		}
		out[name] = fields
		name = ""
	}
	for _, l := range lines {
		if m := classLine.FindSubmatch(l); m != nil {
			flush(offset)
			name, start = string(m[1]), offset
		} else if len(l) > 0 && l[0] != ' ' && l[0] != '\t' && l[0] != '#' && l[0] != '\r' && !bytes.ContainsAny(l[:1], ")]}") {
			flush(offset) // anything else at the top level ends the class
		}
		offset += len(l) + 1
	}
	flush(len(b))
	return out
}

// pythonReads finds call(<'key'>[, default]) reads in b: the key literal, the
// default expression, the annotation of the assignment and the comments.
func pythonReads(b []byte, call *regexp.Regexp) []Field {
	var out []Field
	for _, loc := range call.FindAllIndex(b, -1) {
		i := loc[1]
		if i >= len(b) || (b[i] != '\'' && b[i] != '"') {
			continue
		}
		q := b[i]
		j := bytes.IndexByte(b[i+1:], q)
		if j < 0 {
			continue
		}
		key := string(b[i+1 : i+1+j])
		if !isIdent(key) {
			continue
		}
		k := skipSpace(b, i+j+2)
		def := "None"
		var inner []string
		if k < len(b) && b[k] == ',' {
			var end int
			def, end, inner = pyExpr(b, skipSpace(b, k+1))
			k = end
			// get_conf('key', default=x) or get_conf('key', required=True)
			if m := kwDefault.FindStringSubmatch(def); m != nil {
				if m[1] == "default" {
					def = def[len(m[0]):]
				} else {
					def = "None"
				}
			}
		}
		// the comments: the lines above the statement, inside the call, after it
		lineStart := bytes.LastIndexByte(b[:loc[0]], '\n') + 1
		docs := commentsAbove(b, lineStart)
		docs = append(docs, inner...)
		if k < len(b) && b[k] == ')' {
			k++
		}
		lineEnd := len(b)
		if e := bytes.IndexByte(b[k:], '\n'); e >= 0 {
			lineEnd = k + e
		}
		if c := bytes.IndexByte(b[k:lineEnd], '#'); c >= 0 {
			docs = append(docs, strings.TrimSpace(string(b[k+c+1:lineEnd])))
		}
		f := Field{Key: key, Default: def, Doc: strings.Join(nonEmpty(docs), " ")}
		if m := annotation.FindSubmatch(b[lineStart:loc[1]]); m != nil {
			f.Type = string(m[1])
		}
		out = append(out, f)
	}
	return out
}

// pyExpr reads one argument expression from b[i]: up to a top-level ',' or
// the ')' closing the call. Comments inside it are returned apart.
func pyExpr(b []byte, i int) (expr string, end int, comments []string) {
	depth := 0
	var sb strings.Builder
	for i < len(b) {
		c := b[i]
		switch {
		case c == '\'' || c == '"':
			j := i + 1
			for j < len(b) && b[j] != c && b[j] != '\n' {
				if b[j] == '\\' {
					j++
				}
				j++
			}
			if j >= len(b) {
				j = len(b) - 1
			}
			sb.Write(b[i : j+1])
			i = j + 1
			continue
		case c == '#':
			e := bytes.IndexByte(b[i:], '\n')
			if e < 0 {
				e = len(b) - i
			}
			comments = append(comments, strings.TrimSpace(string(b[i+1:i+e])))
			i += e
			continue
		case c == '(' || c == '[' || c == '{':
			depth++
		case c == ')' || c == ']' || c == '}':
			if depth == 0 {
				return strings.Join(strings.Fields(sb.String()), " "), i, comments
			}
			depth--
		case c == ',' && depth == 0:
			return strings.Join(strings.Fields(sb.String()), " "), i, comments
		}
		sb.WriteByte(c)
		i++
	}
	return strings.Join(strings.Fields(sb.String()), " "), i, comments
}

func commentsAbove(b []byte, lineStart int) []string {
	var docs []string
	for lineStart > 0 {
		prevStart := bytes.LastIndexByte(b[:lineStart-1], '\n') + 1
		line := strings.TrimSpace(string(b[prevStart : lineStart-1]))
		if !strings.HasPrefix(line, "#") {
			break
		}
		docs = append([]string{strings.TrimSpace(strings.TrimPrefix(line, "#"))}, docs...)
		lineStart = prevStart
	}
	return docs
}

func skipSpace(b []byte, i int) int {
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	return i
}

func isIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if !(r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || i > 0 && r >= '0' && r <= '9') {
			return false
		}
	}
	return true
}

func nonEmpty(ss []string) []string {
	out := ss[:0]
	for _, s := range ss {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}
