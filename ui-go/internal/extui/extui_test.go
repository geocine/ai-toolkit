package extui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestListDiscoversAndTranspiles(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "extensions_built_in", "zeta", "ui.tsx"), `
import type { ModelArch } from "@/app/jobs/new/options";
import Link from "next/link";
export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  { name: "zeta", label: "Zeta", group: "image", extras: <Link href="/x">docs</Link> },
];`)
	write(t, filepath.Join(root, "extensions_built_in", "alpha", "ui.ts"), `export const AI_TOOLKIT_UI_MODELS: any[] = [];`)
	// .js is parsed with JSX enabled, like SWC's ecmascript+jsx parser
	write(t, filepath.Join(root, "extensions", "custom", "ui.js"), `export const AI_TOOLKIT_UI_MODELS = [<b key="1" />];`)
	// ui.tsx wins over ui.js
	write(t, filepath.Join(root, "extensions", "both", "ui.tsx"), `export const AI_TOOLKIT_UI_MODELS = [];`)
	write(t, filepath.Join(root, "extensions", "both", "ui.js"), `syntax error here (`)
	write(t, filepath.Join(root, "extensions", "broken", "ui.tsx"), `export const = ;`)
	write(t, filepath.Join(root, "extensions", ".hidden", "ui.tsx"), `export const AI_TOOLKIT_UI_MODELS = [];`)
	write(t, filepath.Join(root, "extensions", "_private", "ui.tsx"), `export const AI_TOOLKIT_UI_MODELS = [];`)
	write(t, filepath.Join(root, "extensions", "noui", "other.tsx"), `x`)

	res := List(root)
	var ids []string
	for _, m := range res.Modules {
		ids = append(ids, m.ID)
	}
	want := "extensions_built_in/alpha,extensions_built_in/zeta,extensions/both,extensions/custom"
	if got := strings.Join(ids, ","); got != want {
		t.Fatalf("modules = %s, want %s", got, want)
	}
	if len(res.Errors) != 1 || res.Errors[0].ID != "extensions/broken" || res.Errors[0].Error == "" {
		t.Fatalf("errors = %+v", res.Errors)
	}
	zeta := res.Modules[1]
	for _, frag := range []string{`require("react/jsx-runtime")`, `require("next/link")`, "AI_TOOLKIT_UI_MODELS", "module.exports"} {
		if !strings.Contains(zeta.Code, frag) {
			t.Errorf("zeta code missing %s:\n%s", frag, zeta.Code)
		}
	}
	if strings.Contains(zeta.Code, "@/app/jobs/new/options") {
		t.Errorf("type-only import was not erased:\n%s", zeta.Code)
	}
	if zeta.File != filepath.Join(root, "extensions_built_in", "zeta", "ui.tsx") || zeta.MtimeMs <= 0 {
		t.Errorf("zeta file/mtime = %s %v", zeta.File, zeta.MtimeMs)
	}

	// unchanged file: served from cache; touched file: re-transpiled
	again := List(root)
	if again.Modules[1].Code != zeta.Code {
		t.Error("cached module changed")
	}
	future := time.Now().Add(time.Hour)
	f := filepath.Join(root, "extensions_built_in", "zeta", "ui.tsx")
	write(t, f, `export const AI_TOOLKIT_UI_MODELS = [{ name: "changed" }];`)
	os.Chtimes(f, future, future)
	if got := List(root).Modules[1].Code; !strings.Contains(got, "changed") {
		t.Errorf("edited module not re-read:\n%s", got)
	}
}

// TestRepoExtensions transpiles this checkout's real ui modules.
func TestRepoExtensions(t *testing.T) {
	root, _ := filepath.Abs(filepath.Join("..", "..", ".."))
	if _, err := os.Stat(filepath.Join(root, "run.py")); err != nil {
		t.Skip("not inside an ai-toolkit checkout")
	}
	res := List(root)
	for _, e := range res.Errors {
		t.Errorf("%s: %s", e.ID, e.Error)
	}
	if len(res.Modules) == 0 {
		t.Fatal("no extension ui modules found")
	}
	for _, m := range res.Modules {
		if !strings.Contains(m.Code, "AI_TOOLKIT_UI_MODELS") {
			t.Errorf("%s: transpiled code does not export AI_TOOLKIT_UI_MODELS", m.ID)
		}
		t.Logf("%s: %d bytes", m.ID, len(m.Code))
	}
}
