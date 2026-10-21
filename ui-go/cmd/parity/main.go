// Command parity sends the same requests to the upstream Next server (A) and
// the Go server (B) and reports where their responses differ.
//
// Each server must run on its own toolkit root with its own fresh database
// (see ui-go/scripts/parity.sh). Fixtures are written into both roots first.
// Values that legitimately differ are normalized before comparing: each
// side's root path becomes <ROOT>, UUIDs become <uuid#n> (numbered in order of
// appearance per side) and ISO timestamps become <ts>.
//
//	go run ./cmd/parity -a http://127.0.0.1:18801 -root-a /tmp/p/next \
//	                    -b http://127.0.0.1:18802 -root-b /tmp/p/go
package main

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"
)

// Mode says how much of a response must match.
type Mode int

const (
	JSON   Mode = iota // status + normalized JSON body
	Shape              // status + JSON keys and value types (volatile values: cpu, gpu)
	Status             // status only
	Bytes              // status + content-type + body hash (files)
	Media              // status + content-type (re-encoded output such as thumbnails)
	Zip                // status + content-type + entry names and sizes
)

// Step is one request. Path and Body may contain {{ROOT}} (that side's root,
// forward slashes), {{ROOT_NATIVE}} and {{name}} for a value captured earlier.
type Step struct {
	Name    string
	Method  string
	Path    string
	Body    any               // JSON body, or Multipart for uploads
	Capture map[string]string // name -> dotted JSON path in the response
	Mode    Mode
	Headers []string // response headers that must match (after normalization)

	ReqHeaders map[string]string
	// Ignore drops these object keys (at any depth) before comparing.
	Ignore []string
	// PathFn / BodyFn build the path or body per side (native paths, URL encoding).
	PathFn func(root string) string
	BodyFn func(root string) any
}

// Raw is an application/octet-stream body.
type Raw []byte

// Multipart is a multipart/form-data body.
type Multipart struct {
	Fields map[string]string
	Files  []MultipartFile
}

type MultipartFile struct {
	Field, Name string
	Data        []byte
}

type side struct {
	name, base, root string
	captures         map[string]string
	uuids            map[string]string
}

var (
	uuidRe = regexp.MustCompile(`[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`)
	tsRe   = regexp.MustCompile(`\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z`)
	tmplRe = regexp.MustCompile(`\{\{([A-Za-z0-9_]+)\}\}`)
)

func main() {
	a := flag.String("a", "", "reference server base URL (upstream Next)")
	b := flag.String("b", "", "candidate server base URL (Go)")
	rootA := flag.String("root-a", "", "toolkit root of server A")
	rootB := flag.String("root-b", "", "toolkit root of server B")
	only := flag.String("only", "", "run only steps whose name contains this")
	token := flag.String("token", os.Getenv("AI_TOOLKIT_AUTH"), "bearer token for both servers")
	verbose := flag.Bool("v", false, "print every step")
	flag.Parse()
	if *a == "" || *b == "" || *rootA == "" || *rootB == "" {
		flag.Usage()
		os.Exit(2)
	}
	sides := []*side{
		{name: "A", base: strings.TrimRight(*a, "/"), root: abs(*rootA)},
		{name: "B", base: strings.TrimRight(*b, "/"), root: abs(*rootB)},
	}
	for _, s := range sides {
		s.captures, s.uuids = map[string]string{}, map[string]string{}
		if err := writeFixtures(s.root); err != nil {
			fmt.Fprintf(os.Stderr, "fixtures for %s: %v\n", s.name, err)
			os.Exit(1)
		}
	}

	client := &http.Client{Timeout: 60 * time.Second}
	failed, ran := 0, 0
	for _, st := range scenario() {
		if *only != "" && !strings.Contains(st.Name, *only) {
			continue
		}
		ran++
		var res [2]result
		for i, s := range sides {
			res[i] = do(client, s, st, *token)
			if st.Capture != nil && res[i].raw != nil {
				for name, path := range st.Capture {
					if v, ok := lookup(res[i].raw, path); ok {
						s.captures[name] = fmt.Sprint(v)
					}
				}
			}
		}
		diffs := compare(st, sides, res)
		if len(diffs) > 0 {
			failed++
			fmt.Printf("FAIL  %s  %s %s\n", st.Name, st.Method, st.Path)
			for _, d := range diffs {
				fmt.Printf("        %s\n", d)
			}
		} else if *verbose {
			fmt.Printf("ok    %s\n", st.Name)
		}
	}
	fmt.Printf("\n%d/%d steps match\n", ran-failed, ran)
	if failed > 0 {
		os.Exit(1)
	}
}

func abs(p string) string {
	a, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return a
}

type result struct {
	status int
	header http.Header
	body   []byte
	raw    any // parsed body as sent
	json   any // parsed body, normalized for comparing
	err    error
}

func expand(s string, sd *side) string {
	return tmplRe.ReplaceAllStringFunc(s, func(m string) string {
		key := m[2 : len(m)-2]
		switch key {
		case "ROOT":
			return filepath.ToSlash(sd.root)
		case "ROOT_NATIVE":
			return sd.root
		}
		if v, ok := sd.captures[key]; ok {
			return v
		}
		return m
	})
}

func expandAny(v any, sd *side) any {
	switch x := v.(type) {
	case string:
		return expand(x, sd)
	case map[string]any:
		out := map[string]any{}
		for k, val := range x {
			out[k] = expandAny(val, sd)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = expandAny(val, sd)
		}
		return out
	}
	return v
}

func do(client *http.Client, sd *side, st Step, token string) result {
	var body io.Reader
	contentType := ""
	reqBody := st.Body
	if st.BodyFn != nil {
		reqBody = st.BodyFn(sd.root)
	}
	path := st.Path
	if st.PathFn != nil {
		path = st.PathFn(sd.root)
	}
	switch b := reqBody.(type) {
	case nil:
	case Multipart:
		buf, ct, err := encodeMultipart(b, sd)
		if err != nil {
			return result{err: err}
		}
		body, contentType = buf, ct
	case Raw:
		body, contentType = bytes.NewReader(b), "application/octet-stream"
	default:
		raw, err := json.Marshal(expandAny(b, sd))
		if err != nil {
			return result{err: err}
		}
		body, contentType = bytes.NewReader(raw), "application/json"
	}
	req, err := http.NewRequest(st.Method, sd.base+expand(path, sd), body)
	if err != nil {
		return result{err: err}
	}
	// send the path exactly as written (no client-side cleaning of `//` or `%2F`)
	req.URL.Opaque = strings.SplitN(expand(path, sd), "?", 2)[0]
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	for k, v := range st.ReqHeaders {
		req.Header.Set(k, v)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return result{err: err}
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	r := result{status: resp.StatusCode, header: resp.Header, body: data, err: err}
	var parsed any
	if json.Unmarshal(data, &parsed) == nil {
		r.raw = parsed
		r.json = normalize(parsed, sd)
	}
	return r
}

// normalize rewrites values that legitimately differ between the two sides.
func normalize(v any, sd *side) any {
	switch x := v.(type) {
	case string:
		return normString(x, sd)
	case map[string]any:
		out := map[string]any{}
		for k, val := range x {
			out[normString(k, sd)] = normalize(val, sd)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = normalize(val, sd)
		}
		return out
	}
	return v
}

func normString(s string, sd *side) string {
	native, slashed := sd.root, filepath.ToSlash(sd.root)
	escaped := strings.ReplaceAll(native, `\`, `\\`)
	for _, r := range []string{escaped, native, slashed} {
		s = strings.ReplaceAll(s, r, "<ROOT>")
	}
	s = uuidRe.ReplaceAllStringFunc(s, func(u string) string {
		if n, ok := sd.uuids[u]; ok {
			return n
		}
		n := fmt.Sprintf("<uuid#%d>", len(sd.uuids)+1)
		sd.uuids[u] = n
		return n
	})
	return tsRe.ReplaceAllString(s, "<ts>")
}

func compare(st Step, sides []*side, res [2]result) []string {
	var diffs []string
	for i, r := range res {
		if r.err != nil {
			diffs = append(diffs, fmt.Sprintf("%s: request failed: %v", sides[i].name, r.err))
		}
	}
	if len(diffs) > 0 {
		return diffs
	}
	a, b := res[0], res[1]
	if a.status != b.status {
		diffs = append(diffs, fmt.Sprintf("status A=%d B=%d", a.status, b.status))
	}
	for _, h := range st.Headers {
		ha, hb := normString(a.header.Get(h), sides[0]), normString(b.header.Get(h), sides[1])
		if ha != hb {
			diffs = append(diffs, fmt.Sprintf("header %s: A=%q B=%q", h, ha, hb))
		}
	}
	switch st.Mode {
	case JSON:
		a.json, b.json = drop(a.json, st.Ignore), drop(b.json, st.Ignore)
		if (a.json == nil) != (b.json == nil) {
			diffs = append(diffs, fmt.Sprintf("body JSON A=%v B=%v  (A=%s | B=%s)", a.json != nil, b.json != nil, snippet(a.body), snippet(b.body)))
			break
		}
		if a.json == nil {
			if !bytes.Equal(bytes.TrimSpace(a.body), bytes.TrimSpace(b.body)) && a.status < 500 {
				diffs = append(diffs, fmt.Sprintf("body A=%s B=%s", snippet(a.body), snippet(b.body)))
			}
			break
		}
		diffs = append(diffs, deepDiff("$", a.json, b.json, false)...)
	case Shape:
		diffs = append(diffs, deepDiff("$", a.json, b.json, true)...)
	case Bytes:
		if ct(a) != ct(b) {
			diffs = append(diffs, fmt.Sprintf("content-type A=%q B=%q", ct(a), ct(b)))
		}
		if hash(a.body) != hash(b.body) {
			diffs = append(diffs, fmt.Sprintf("body hash differs (A %d bytes, B %d bytes)", len(a.body), len(b.body)))
		}
	case Media:
		if ct(a) != ct(b) {
			diffs = append(diffs, fmt.Sprintf("content-type A=%q B=%q", ct(a), ct(b)))
		}
	case Zip:
		if ct(a) != ct(b) {
			diffs = append(diffs, fmt.Sprintf("content-type A=%q B=%q", ct(a), ct(b)))
		}
		ea, eb := zipEntries(a.body), zipEntries(b.body)
		if ea != eb {
			diffs = append(diffs, fmt.Sprintf("zip entries A=[%s] B=[%s]", ea, eb))
		}
	}
	return diffs
}

// drop removes the named keys from objects at any depth.
func drop(v any, keys []string) any {
	if len(keys) == 0 {
		return v
	}
	switch x := v.(type) {
	case map[string]any:
		out := map[string]any{}
		for k, val := range x {
			if !slices.Contains(keys, k) {
				out[k] = drop(val, keys)
			}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = drop(val, keys)
		}
		return out
	}
	return v
}

// zipEntries lists "name:size" for each file in a zip, sorted ("" if not a zip).
func zipEntries(data []byte) string {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return "not a zip: " + snippet(data)
	}
	var names []string
	for _, f := range zr.File {
		if strings.HasSuffix(f.Name, "/") {
			continue // directory entries are optional
		}
		names = append(names, fmt.Sprintf("%s:%d", f.Name, f.UncompressedSize64))
	}
	sort.Strings(names)
	return strings.Join(names, " ")
}

func ct(r result) string {
	return strings.ToLower(strings.ReplaceAll(r.header.Get("Content-Type"), " ", ""))
}

func hash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:8])
}

func snippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 160 {
		s = s[:160] + "..."
	}
	return fmt.Sprintf("%q", s)
}

// deepDiff compares normalized JSON; shapeOnly ignores scalar values.
func deepDiff(path string, a, b any, shapeOnly bool) []string {
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok {
			return []string{fmt.Sprintf("%s: A is object, B is %s", path, kind(b))}
		}
		var out []string
		keys := map[string]bool{}
		for k := range av {
			keys[k] = true
		}
		for k := range bv {
			keys[k] = true
		}
		sorted := make([]string, 0, len(keys))
		for k := range keys {
			sorted = append(sorted, k)
		}
		sort.Strings(sorted)
		for _, k := range sorted {
			va, inA := av[k]
			vb, inB := bv[k]
			switch {
			case !inA:
				out = append(out, fmt.Sprintf("%s.%s: only in B (%s)", path, k, short(vb)))
			case !inB:
				out = append(out, fmt.Sprintf("%s.%s: only in A (%s)", path, k, short(va)))
			default:
				out = append(out, deepDiff(path+"."+k, va, vb, shapeOnly)...)
			}
		}
		return out
	case []any:
		bv, ok := b.([]any)
		if !ok {
			return []string{fmt.Sprintf("%s: A is array, B is %s", path, kind(b))}
		}
		if shapeOnly {
			if len(av) > 0 && len(bv) > 0 {
				return deepDiff(path+"[0]", av[0], bv[0], true)
			}
			return nil
		}
		if len(av) != len(bv) {
			return []string{fmt.Sprintf("%s: length A=%d B=%d (A=%s | B=%s)", path, len(av), len(bv), short(av), short(bv))}
		}
		var out []string
		for i := range av {
			out = append(out, deepDiff(fmt.Sprintf("%s[%d]", path, i), av[i], bv[i], shapeOnly)...)
		}
		return out
	default:
		if shapeOnly {
			if kind(a) != kind(b) && a != nil && b != nil {
				return []string{fmt.Sprintf("%s: A is %s, B is %s", path, kind(a), kind(b))}
			}
			return nil
		}
		if fmt.Sprint(a) != fmt.Sprint(b) || kind(a) != kind(b) {
			return []string{fmt.Sprintf("%s: A=%s B=%s", path, short(a), short(b))}
		}
		return nil
	}
}

func kind(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case float64:
		return "number"
	case bool:
		return "bool"
	}
	return fmt.Sprintf("%T", v)
}

func short(v any) string {
	b, _ := json.Marshal(v)
	s := string(b)
	if len(s) > 120 {
		s = s[:120] + "..."
	}
	return s
}

func lookup(v any, path string) (any, bool) {
	cur := v
	for _, part := range strings.Split(path, ".") {
		switch c := cur.(type) {
		case map[string]any:
			next, ok := c[part]
			if !ok {
				return nil, false
			}
			cur = next
		case []any:
			var i int
			if _, err := fmt.Sscan(part, &i); err != nil || i < 0 || i >= len(c) {
				return nil, false
			}
			cur = c[i]
		default:
			return nil, false
		}
	}
	return cur, true
}
