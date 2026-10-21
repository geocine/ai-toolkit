package system

import (
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/jsonx"
)

func TestNormalizeArgs(t *testing.T) {
	cases := []struct{ args, want, err string }{
		{`null`, ``, ``},
		{`["a", 1, 1.50, true, null, 1e-7, 1e21]`, `a 1 1.5 true 1e-7 1e+21`, ``},
		// Object.entries order: integer keys first, ascending
		{`{"b": "x", "10": 1, "2": 2, "flag": true, "off": false, "none": null}`, `--2 2 --10 1 --b x --flag`, ``},
		{`[{"a": 1}]`, ``, `args entries must be string|number|boolean`},
		{`{"a b": 1}`, ``, `invalid arg key: a b`},
		{`{"a": [1]}`, ``, `args.a must be string|number|boolean`},
		{`"str"`, ``, `args must be an array or object`},
	}
	for _, c := range cases {
		v, err := jsonx.Parse([]byte(c.args))
		if err != nil {
			t.Fatal(err)
		}
		got, errMsg := normalizeArgs(v)
		if errMsg != c.err || strings.Join(got, " ") != c.want {
			t.Errorf("%s: got %q / %q, want %q / %q", c.args, got, errMsg, c.want, c.err)
		}
	}
}

func TestParseResult(t *testing.T) {
	if r := parseResult("log line\n{\"a\": 1}\n\n"); r == nil {
		t.Error("last JSON line not parsed")
	}
	if r := parseResult("{\"a\": 1}\nplain last line\n"); r != nil {
		t.Error("non-JSON last line should give null")
	}
	if r := parseResult("{broken\n"); r != nil {
		t.Error("broken JSON should give null")
	}
}

func TestResolveScriptPathRejects(t *testing.T) {
	for _, name := range []any{"../x.py", "a/b.py", ".hidden.py", "x.sh", 5, nil} {
		if p := resolveScriptPath(name); p != "" {
			t.Errorf("%v resolved to %s", name, p)
		}
	}
}
