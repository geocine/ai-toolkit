package templates

import (
	"fmt"
	"sort"
	"strings"
)

// Problem is a key in a config that the trainer does not read where it is.
type Problem struct {
	Path    string `json:"path"` // e.g. config.process[0].loss_type
	Message string `json:"message"`
	Fix     string `json:"fix,omitempty"`
}

// Text is the problem as one line.
func (p Problem) Text() string {
	if p.Fix == "" {
		return p.Path + ": " + p.Message
	}
	return p.Path + ": " + p.Message + " (" + p.Fix + ")"
}

// CheckResult is POST /v1/templates/check.
type CheckResult struct {
	Problems []Problem `json:"problems"`
	Errors   []string  `json:"errors"` // the reference could not be fully read
}

// Check reports keys the trainer ignores: at the process level or in a
// section whose class does not read them (a key read in another section is
// named as the fix, a near miss as a "did you mean"). Processes without a
// train or model section (generate, extract, ...) are not checked.
func Check(cfg any) CheckResult {
	ref := Fields()
	res := CheckResult{Problems: []Problem{}, Errors: ref.Errors}
	byPath := map[string]map[string]bool{}
	for _, s := range ref.Sections {
		keys := map[string]bool{}
		for _, f := range s.Fields {
			keys[f.Key] = true
		}
		byPath[s.Path] = keys
	}
	if byPath["process"] == nil || len(ref.Sections) < 2 {
		return res // no reference to check against
	}
	top, _ := cfg.(map[string]any)
	if c, ok := top["job_config"].(map[string]any); ok {
		top = c
	}
	conf, _ := top["config"].(map[string]any)
	procs, ok := conf["process"].([]any)
	if !ok {
		res.Problems = append(res.Problems, Problem{Path: "config.process", Message: "missing: a job config is {job, config: {name, process: [...]}}"})
		return res
	}
	for i, raw := range procs {
		p, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if _, train := p["train"]; !train {
			if _, model := p["model"]; !model {
				continue
			}
		}
		c := checker{byPath: byPath, problems: &res.Problems}
		c.walk(fmt.Sprintf("config.process[%d]", i), "process", p)
	}
	return res
}

type checker struct {
	byPath   map[string]map[string]bool
	problems *[]Problem
}

// walk checks the keys of one object read at refPath ("process", "train",
// "datasets[]", ...), and the nested sections the reference knows.
func (c checker) walk(at, refPath string, obj map[string]any) {
	keys := c.byPath[refPath]
	names := make([]string, 0, len(obj))
	for k := range obj {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		v := obj[k]
		sub := k
		if refPath != "process" {
			sub = refPath + "." + k
		}
		if _, nested := c.byPath[sub]; nested {
			if m, ok := v.(map[string]any); ok {
				c.walk(at+"."+k, sub, m)
			}
			continue
		}
		if _, nested := c.byPath[sub+"[]"]; nested {
			if items, ok := v.([]any); ok {
				for j, it := range items {
					if m, ok := it.(map[string]any); ok {
						c.walk(fmt.Sprintf("%s.%s[%d]", at, k, j), sub+"[]", m)
					}
				}
			}
			continue
		}
		if keys[k] {
			continue
		}
		p := Problem{Path: at + "." + k}
		where := "at the process level"
		if refPath != "process" {
			where = "in " + refPath
		}
		p.Message = fmt.Sprintf("%s is not read %s; the trainer ignores it", k, where)
		if elsewhere := c.readIn(k, refPath); len(elsewhere) > 0 {
			p.Fix = "it is read in " + strings.Join(elsewhere, ", ")
		} else if near := c.near(k, refPath); near != "" {
			p.Fix = "did you mean " + near + "?"
		}
		*c.problems = append(*c.problems, p)
	}
}

// readIn lists the sections other than skip that read key.
func (c checker) readIn(key, skip string) []string {
	var out []string
	for path, keys := range c.byPath {
		if path != skip && keys[key] {
			out = append(out, path)
		}
	}
	sort.Strings(out)
	return out
}

// near is the closest key: in the same section within 2 edits, else in any
// section within 1 (as section.key).
func (c checker) near(key, refPath string) string {
	best, bestD := "", 3
	for k := range c.byPath[refPath] {
		if d := editDistance(key, k); d < bestD || d == bestD && k < best {
			best, bestD = k, d
		}
	}
	if best != "" {
		return best
	}
	paths := make([]string, 0, len(c.byPath))
	for p := range c.byPath {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, p := range paths {
		for k := range c.byPath[p] {
			if editDistance(key, k) == 1 {
				return p + "." + k
			}
		}
	}
	return ""
}
