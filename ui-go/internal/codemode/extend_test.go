package codemode

import (
	"context"
	"maps"
	"slices"
	"strings"
	"testing"
)

// keepRegistries restores the init-time registries after a test that adds to them.
func keepRegistries(t *testing.T) {
	t.Helper()
	types, hs, pre, ops := slices.Clone(extraTypes), maps.Clone(helpers), slices.Clone(preludes), slices.Clone(Ops)
	t.Cleanup(func() { extraTypes, helpers, preludes, Ops = types, hs, pre, ops })
}

func TestRegisterTypesAfterTypesTS(t *testing.T) {
	keepRegistries(t)
	RegisterTypes("type ExtendProbe = { id: string; n: number };\n")
	decl := Declaration("")
	at := strings.Index(decl, "type ExtendProbe = ")
	if at < 0 {
		t.Fatalf("registered types missing:\n%s", decl)
	}
	if end := strings.Index(decl, typesTS) + len(typesTS); at < end || at > strings.Index(decl, "declare const aitk") {
		t.Errorf("registered types at %d, want after typesTS (ends %d) and before aitk", at, end)
	}
	// types are written whatever the search
	if !strings.Contains(Declaration("checkpoint"), "type ExtendProbe = ") {
		t.Error("search dropped registered types")
	}
}

func TestRegisterHelperInsideItsGroup(t *testing.T) {
	keepRegistries(t)
	Ops = append(Ops, Op{Group: "extendprobe", Name: "get", Method: "GET", Path: "/v1/extendprobe/{id}", Result: "ExtendProbe",
		Params: []Param{p("id", "string", "path", "")}, Doc: "Probe for the extension seams."})
	RegisterTypes("type ExtendProbe = { id: string; done: boolean };\n")
	RegisterHelper("extendprobe", "    /** Poll get until done. */\n    wait(id: string): Promise<ExtendProbe>;\n")
	decl := Declaration("")
	block := decl[strings.Index(decl, "  extendprobe: {\n"):]
	block = block[:strings.Index(block, "  };\n")]
	if !strings.Contains(block, "    get(id: string)") || !strings.Contains(block, "    wait(id: string): Promise<ExtendProbe>;") {
		t.Fatalf("helper not inside its group:\n%s", block)
	}
	if strings.Count(decl, "wait(id: string): Promise<ExtendProbe>") != 1 {
		t.Error("helper written outside its group")
	}
	if _, err := Transpile(decl + "\nreturn 1"); err != nil {
		t.Fatalf("declaration with extensions does not compile: %v", err)
	}
	// a group the search filters out takes its helper with it
	if strings.Contains(Declaration("checkpoint"), "Promise<ExtendProbe>;") {
		t.Error("helper of a filtered-out group written")
	}
}

func TestBuiltinWaitHelpersDeclared(t *testing.T) {
	decl := Declaration("")
	for g, want := range map[string]string{"jobs": "    wait(id: string, options?: { until?: JobState[]; ", "bundles": "    wait(id: string, options?: { timeout_ms?: number; poll_ms?: number }): Promise<BundleTask>;"} {
		block := decl[strings.Index(decl, "  "+g+": {\n"):]
		block = block[:strings.Index(block, "  };\n")]
		if !strings.Contains(block, want) {
			t.Errorf("%s.wait not declared in its group:\n%s", g, block)
		}
	}
	if _, err := Transpile(decl + "\nreturn 1"); err != nil {
		t.Fatalf("declaration does not compile: %v", err)
	}
}

func TestRegisterPrelude(t *testing.T) {
	keepRegistries(t)
	RegisterPrelude(`globalThis.extendProbe = n => n * 2;
aitk.jobs.probe = async function (id) { return (await aitk.jobs.get(id)).state + '!'; };`)
	var calls []recorded
	res := Run(context.Background(), `return [extendProbe(21), await aitk.jobs.probe("j1")]`, Options{Call: fakeAPI(t, &calls)})
	if res.Error != nil || string(res.Result) != `[42,"completed!"]` {
		t.Fatalf("%s %+v", res.Result, res.Error)
	}

	// a broken prelude fails the run instead of running the script
	RegisterPrelude(`throw new Error('prelude broke');`)
	calls = nil
	res = Run(context.Background(), `return await aitk.jobs.get("j1")`, Options{Call: fakeAPI(t, &calls)})
	if res.Error == nil || res.Error.Code != "runtime_error" || !strings.Contains(res.Error.Message, "prelude broke") || len(calls) != 0 {
		t.Fatalf("%s %+v calls=%d", res.Result, res.Error, len(calls))
	}
}
