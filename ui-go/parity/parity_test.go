// Package parity checks the Go API against the upstream Next.js UI it ports.
// Run after every rebase onto upstream:
//
//	go test ./parity/
//
// A failure lists exactly what upstream added, removed or changed that the Go
// side does not cover yet. The tests read upstream's own files (ui/src/app/api
// and ui/prisma/schema.prisma), so nothing here needs updating by hand.
package parity

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/server"
	"github.com/geocine/aitoolkit/internal/store"
)

var upstreamUI = filepath.Join("..", "..", "ui")

// skipped lists upstream routes the Go server deliberately does not serve,
// with the reason. Keep it empty unless there is a good one.
var skipped = map[string]string{}

var (
	routeFile   = regexp.MustCompile(`^route\.(ts|tsx|js|jsx|mjs)$`)
	exportFn    = regexp.MustCompile(`export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b`)
	exportConst = regexp.MustCompile(`export\s+const\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b`)
	exportAlias = regexp.MustCompile(`export\s*\{([^}]*)\}`)
	aliasName   = regexp.MustCompile(`\bas\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b`)
	catchAll    = regexp.MustCompile(`^\[\.\.\.[^\]]+\]$`)
	dynamic     = regexp.MustCompile(`^\[[^\]]+\]$`)
	goWild      = regexp.MustCompile(`\{[^}]*\.\.\.\}`)
	goParam     = regexp.MustCompile(`\{[^}]*\}`)
)

// upstreamRoutes returns "METHOD /api/path" for every exported handler in
// ui/src/app/api/**/route.ts, with params normalized to {} and {...}.
func upstreamRoutes(t *testing.T) []string {
	t.Helper()
	apiDir := filepath.Join(upstreamUI, "src", "app", "api")
	if _, err := os.Stat(apiDir); err != nil {
		t.Skipf("upstream UI not found at %s: %v", apiDir, err)
	}
	var out []string
	err := filepath.WalkDir(apiDir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !routeFile.MatchString(d.Name()) {
			return err
		}
		rel, _ := filepath.Rel(filepath.Join(upstreamUI, "src", "app"), filepath.Dir(p))
		segs := strings.Split(filepath.ToSlash(rel), "/")
		for i, s := range segs {
			switch {
			case catchAll.MatchString(s):
				segs[i] = "{...}"
			case dynamic.MatchString(s):
				segs[i] = "{}"
			}
		}
		route := "/" + strings.Join(segs, "/")
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		methods := map[string]bool{}
		for _, m := range exportFn.FindAllStringSubmatch(string(src), -1) {
			methods[m[1]] = true
		}
		for _, m := range exportConst.FindAllStringSubmatch(string(src), -1) {
			methods[m[1]] = true
		}
		for _, block := range exportAlias.FindAllStringSubmatch(string(src), -1) {
			for _, m := range aliasName.FindAllStringSubmatch(block[1], -1) {
				methods[m[1]] = true
			}
		}
		if len(methods) == 0 {
			t.Errorf("%s exports no recognizable handlers; update the parser in parity_test.go", p)
		}
		for m := range methods {
			out = append(out, m+" "+route)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(out)
	return out
}

// normalizeGo turns "GET /api/img/{imagePath...}" into "GET /api/img/{...}"
// and named params into {}, matching upstreamRoutes.
func normalizeGo(pattern string) string {
	const wild = "\x00"
	p := goWild.ReplaceAllString(pattern, wild)
	p = goParam.ReplaceAllString(p, "{}")
	return strings.ReplaceAll(p, wild, "{...}")
}

// TestRouteCoverage fails when upstream has a route the Go server lacks, or
// the Go server still serves one upstream removed.
func TestRouteCoverage(t *testing.T) {
	server.NewMux()
	have := map[string]bool{}
	for _, p := range httpx.Routes() {
		have[normalizeGo(p)] = true
	}
	want := map[string]bool{}
	for _, r := range upstreamRoutes(t) {
		want[r] = true
		if have[r] {
			continue
		}
		if why, ok := skipped[r]; ok {
			t.Logf("skipped %s: %s", r, why)
			continue
		}
		t.Errorf("upstream route not ported: %s", r)
	}
	for r := range have {
		if !want[r] {
			t.Errorf("Go serves %s but upstream no longer has it", r)
		}
	}
}

// prismaModels parses model blocks of schema.prisma into model -> fields.
func prismaModels(t *testing.T) map[string][]string {
	t.Helper()
	f, err := os.Open(filepath.Join(upstreamUI, "prisma", "schema.prisma"))
	if err != nil {
		t.Skipf("upstream schema not found: %v", err)
	}
	defer f.Close()
	models := map[string][]string{}
	var current string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if i := strings.Index(line, "//"); i >= 0 {
			line = strings.TrimSpace(line[:i])
		}
		switch {
		case line == "":
		case strings.HasPrefix(line, "model ") && strings.HasSuffix(line, "{"):
			current = strings.Fields(line)[1]
			models[current] = nil
		case line == "}":
			current = ""
		case current != "" && !strings.HasPrefix(line, "@@"):
			models[current] = append(models[current], strings.Fields(line)[0])
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return models
}

// TestSchemaDrift fails when upstream's Prisma schema and the columns the
// store reads and writes disagree.
func TestSchemaDrift(t *testing.T) {
	upstream := prismaModels(t)
	for model, fields := range upstream {
		goCols, ok := store.Tables[model]
		if !ok {
			t.Errorf("upstream model %s is not mapped in internal/store", model)
			continue
		}
		have := map[string]bool{}
		for _, c := range goCols {
			have[c] = true
		}
		for _, f := range fields {
			if !have[f] {
				t.Errorf("upstream added %s.%s; add it to internal/store (struct, columns, scan)", model, f)
			}
			delete(have, f)
		}
		for c := range have {
			t.Errorf("internal/store uses %s.%s, which upstream no longer has", model, c)
		}
	}
	for model := range store.Tables {
		if _, ok := upstream[model]; !ok {
			t.Errorf("internal/store maps %s, which upstream no longer has", model)
		}
	}
}
