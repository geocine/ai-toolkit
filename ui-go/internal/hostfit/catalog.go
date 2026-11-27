package hostfit

import (
	_ "embed"
	"encoding/json"
	"strings"
)

// The model list is upstream's: every arch the upstream new-job form offers
// (extensions_built_in/*/ui.tsx through ui/src/extensions/modelArchs.ts).
// It is a snapshot built into the binary, because the studio API runs with
// no upstream checkout to read it from. Regenerate it after rebasing onto
// upstream:
//
//	go generate ./internal/hostfit
//
// cmd/modelsgen's test fails while the snapshot differs from the checkout.
//
//go:generate go run ../../cmd/modelsgen -o upstream_models.json

//go:embed upstream_models.json
var upstreamModelsJSON []byte

// CatalogModel is one upstream model arch as the snapshot stores it.
type CatalogModel struct {
	Arch        string `json:"arch"`         // upstream's name: "wan22_14b:t2v"
	Name        string `json:"name"`         // upstream's label: "Wan 2.2 (14B)"
	Group       string `json:"group"`        // upstream's group: "image", "video", ...
	Video       bool   `json:"video"`        // upstream's isVideoModel
	DefaultRepo string `json:"default_repo"` // the name_or_path the form fills in ("" when none)
}

// ModelInfo is one row of GET /v1/hosts/models.
type ModelInfo struct {
	CatalogModel
	// VRAM says the VRAM table has a row for this arch, so fit verdicts and
	// the GPU search's start tier are real; without it they are unknown.
	VRAM bool `json:"vram"`
}

var catalog = mustCatalog(upstreamModelsJSON)

// Catalog is the built-in snapshot (a copy).
func Catalog() []CatalogModel { return append([]CatalogModel(nil), catalog...) }

// EncodeCatalog is the snapshot file for models: two-space JSON, LF, one
// trailing newline, so regenerating an unchanged upstream is a no-op.
func EncodeCatalog(models []CatalogModel) ([]byte, error) {
	b, err := json.MarshalIndent(models, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func mustCatalog(b []byte) []CatalogModel {
	var c []CatalogModel
	if err := json.Unmarshal(b, &c); err != nil {
		panic("hostfit: upstream_models.json: " + err.Error())
	}
	return c
}

// catalogIndex finds a snapshot entry by arch.
var catalogIndex = func() map[string]int {
	idx := make(map[string]int, len(catalog))
	for i, m := range catalog {
		idx[m.Arch] = i
	}
	return idx
}()

// archAliases maps upstream arch names to the VRAM/download table row for
// the same weights. Only exact equivalents: a ":tag" is never stripped in
// general, since wan21:1b must not inherit the 14B row.
var archAliases = map[string]string{
	"wan21:14b":                 "wan21",
	"wan22_14b:t2v":             "wan22_14b",
	"wan22_14b_i2v":             "wan22_14b",
	"qwen_image:2512":           "qwen_image",
	"qwen_image_edit_plus:2511": "qwen_image_edit_plus",
	"zimage:turbo":              "zimage",
	"zimage:deturbo":            "zimage",
	"krea2:turbo":               "krea2",
}

// tableArch is the VRAM/download table key for an arch: itself, or the row
// its alias names.
func tableArch(arch string) string {
	if a, ok := archAliases[arch]; ok {
		return a
	}
	return arch
}

// catalogEntry is the snapshot entry for arch, if upstream has one.
func catalogEntry(arch string) (CatalogModel, bool) {
	i, ok := catalogIndex[arch]
	if !ok {
		return CatalogModel{}, false
	}
	return catalog[i], true
}

// Models is GET /v1/hosts/models: upstream's archs in the snapshot's order
// (by label, as the upstream form lists them). The table's older short
// names ("wan21") are not listed but still resolve in a ModelRef.
func Models() []ModelInfo {
	out := make([]ModelInfo, 0, len(catalog))
	for _, m := range catalog {
		out = append(out, ModelInfo{CatalogModel: m, VRAM: KnownArch(m.Arch)})
	}
	return out
}

// catalogName is upstream's label for arch, trimmed; "" when it has none.
func catalogName(arch string) string {
	if m, ok := catalogEntry(arch); ok {
		return strings.TrimSpace(m.Name)
	}
	return ""
}
