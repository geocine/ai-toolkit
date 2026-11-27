package main

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/paths"
)

// The built-in snapshot must match this checkout's upstream model list, so
// a rebase that adds or renames a model fails here until it is regenerated.
// Parsed data is compared, not bytes, so a CRLF checkout still passes.
func TestSnapshotMatchesCheckout(t *testing.T) {
	if !useCheckout() {
		t.Skip("no ai-toolkit checkout above the working directory")
	}
	if _, err := os.Stat(filepath.Join(paths.ToolkitRoot, "ui", "src", "extensions", "modelArchs.ts")); err != nil {
		t.Skipf("no upstream checkout at %s", paths.ToolkitRoot)
	}
	live, err := Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got := hostfit.Catalog(); !reflect.DeepEqual(got, live) {
		t.Fatalf("internal/hostfit/upstream_models.json is stale (%d models built in, %d upstream): run `go generate ./internal/hostfit` in ui-go", len(got), len(live))
	}
}
