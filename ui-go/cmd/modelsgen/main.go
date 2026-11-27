// Command modelsgen writes internal/hostfit/upstream_models.json: the model
// archs upstream's new-job form offers, read from this checkout the same way
// GET /v1/templates reads them (internal/templates). Run it after rebasing
// onto upstream:
//
//	go generate ./internal/hostfit
//
// The output is deterministic (upstream's order, two-space JSON, LF), so an
// unchanged upstream regenerates the same bytes.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/templates"
)

// useCheckout points paths at the ai-toolkit checkout holding the working
// directory (the nearest parent with run.py), since go generate and go test
// run from deep inside ui-go. AI_TOOLKIT_ROOT still wins. false when there
// is none.
func useCheckout() bool {
	if os.Getenv("AI_TOOLKIT_ROOT") != "" {
		return true
	}
	dir, err := os.Getwd()
	if err != nil {
		return false
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "run.py")); err == nil {
			paths.SetRoot(dir)
			return true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return false
		}
		dir = parent
	}
}

func main() {
	out := flag.String("o", "", "output file (default stdout)")
	flag.Parse()
	if !useCheckout() {
		fmt.Fprintln(os.Stderr, "modelsgen: no ai-toolkit checkout (run.py) above the working directory; set AI_TOOLKIT_ROOT")
		os.Exit(1)
	}
	b, err := Generate(context.Background())
	if err != nil {
		fmt.Fprintln(os.Stderr, "modelsgen:", err)
		os.Exit(1)
	}
	if *out == "" {
		os.Stdout.Write(b)
		return
	}
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "modelsgen:", err)
		os.Exit(1)
	}
}

// Generate renders the snapshot from the checkout's templates catalog.
func Generate(ctx context.Context) ([]byte, error) {
	models, err := Read(ctx)
	if err != nil {
		return nil, err
	}
	return hostfit.EncodeCatalog(models)
}

// Read is the checkout's model list as snapshot entries. A module that
// failed to load is an error: a snapshot missing its models would look
// complete.
func Read(ctx context.Context) ([]hostfit.CatalogModel, error) {
	c, err := templates.List(ctx)
	if err != nil {
		return nil, err
	}
	if len(c.Errors) > 0 {
		return nil, fmt.Errorf("upstream ui modules failed to load: %v", c.Errors)
	}
	if len(c.Archs) == 0 {
		return nil, fmt.Errorf("upstream defines no model archs (is AI_TOOLKIT_ROOT the checkout?)")
	}
	out := make([]hostfit.CatalogModel, 0, len(c.Archs))
	for _, a := range c.Archs {
		out = append(out, hostfit.CatalogModel{
			Arch:        a.Name,
			Name:        a.Label,
			Group:       a.Group,
			Video:       a.IsVideoModel,
			DefaultRepo: defaultRepo(a.Defaults),
		})
	}
	return out, nil
}

// defaultRepo is the name_or_path the form fills in when the arch is
// selected: defaults["config.process[0].model.name_or_path"] is
// [selected, unselected].
func defaultRepo(defaults map[string]any) string {
	v, ok := defaults["config.process[0].model.name_or_path"].([]any)
	if !ok || len(v) == 0 {
		return ""
	}
	s, _ := v[0].(string)
	return s
}
