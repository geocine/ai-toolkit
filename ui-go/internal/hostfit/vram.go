package hostfit

import (
	"fmt"
	"strings"
)

// VRAM verdicts.
const (
	VRAMFits    = "fits"
	VRAMTight   = "tight"
	VRAMNo      = "no"
	VRAMUnknown = "unknown"
)

// TierSizes are the GPU sizes (GB) the VRAM table has a cell for.
var TierSizes = []float64{16, 24, 32, 48, 80}

// tierSlack places a card in a tier: a "24 GB" card reports 22.49 GiB (L4,
// A10) to 23.99 GiB (4090), so a card is in tier T when it has at least
// T x 0.9. The same value as gpupick.TierSlack.
const tierSlack = 0.9

// Tier is one GPU size and the verdict there. The JSON is flat: the
// /v1 HostFit tier is { vram_gb } & VramFit.
type Tier struct {
	VRAMGB float64 `json:"vram_gb"`
	VRAMFit
}

// recipe is the job setting a tight cell needs to fit.
type recipe struct {
	qtypes   []string // any of these qtype prefixes (lower bits are fine): "uint3", "uint4"
	lowVRAM  bool
	quantize bool
}

// met reports a job (model section) that already uses the recipe.
func (r recipe) met(m ModelRef) bool {
	if len(r.qtypes) == 0 && !r.lowVRAM && !r.quantize {
		return false
	}
	if r.lowVRAM && !m.LowVRAM {
		return false
	}
	if r.quantize && !m.Quantize && m.QType == "" {
		return false
	}
	if len(r.qtypes) > 0 {
		ok := false
		for _, q := range r.qtypes {
			if strings.HasPrefix(strings.ToLower(m.QType), q) {
				ok = true
			}
		}
		return ok
	}
	return true
}

// cell is one VRAM table entry. needs names a setting a fits cell relies on:
// a job section that turns it off downgrades the cell to tight.
type cell struct {
	verdict string
	source  string // repo | vendor | estimate
	note    string
	recipe  string // shown with a tight verdict (what makes it fit)
	upgrade recipe // a job already using this makes tight fit
	needs   recipe // a fits cell that relies on this (quantize)
}

// archNames are display names for notes.
var archNames = map[string]string{
	"flux": "FLUX.1", "flux_kontext": "FLUX.1 Kontext", "flux2": "FLUX.2", "qwen_image": "Qwen-Image",
	"qwen_image_edit": "Qwen-Image-Edit", "qwen_image_edit_plus": "Qwen-Image-Edit-2509", "wan22_14b": "Wan 2.2 14B",
	"wan22_5b": "Wan 2.2 5B", "wan21": "Wan 2.1 14B", "hidream": "HiDream", "ltx2": "LTX-2", "zimage": "Z-Image", "krea2": "Krea 2",
}

// ArchName is a model arch's display name ("flux" -> "FLUX.1"); the arch
// itself when unknown.
func ArchName(arch string) string {
	if n, ok := archNames[arch]; ok {
		return n
	}
	return arch
}

func est(verdict string) cell { return cell{verdict: verdict, source: "estimate"} }

// vramTable is IMAGE-PLAN §5, cells for 16, 24, 32, 48 and 80 GB. A
// "tight/no" cell is stored as no (MASTER §3.10.5). Repo cells come from
// ai-toolkit config/examples/*.yaml; vendor cells from Lightricks' own LTX-2
// trainer; every other cell is an estimate.
var vramTable = map[string][5]cell{
	"flux": {
		est(VRAMNo),
		{verdict: VRAMFits, source: "repo", note: "with 8-bit quantization and gradient checkpointing (train_lora_flux_24gb.yaml)",
			recipe: "quantize: true", needs: recipe{quantize: true}},
		est(VRAMFits), est(VRAMFits), est(VRAMFits),
	},
	"flux_kontext": {
		est(VRAMNo),
		{verdict: VRAMTight, source: "repo", note: "may run out of memory at 1024; 768 is fine (train_lora_flux_kontext_24gb.yaml)",
			recipe: "train at 768 with quantize: true"},
		est(VRAMFits), est(VRAMFits), est(VRAMFits),
	},
	"flux2": {
		est(VRAMNo), est(VRAMNo), est(VRAMNo),
		{verdict: VRAMTight, source: "estimate", recipe: "8-bit quantization and low VRAM mode", upgrade: recipe{quantize: true, lowVRAM: true}},
		est(VRAMFits),
	},
	"qwen_image": {
		est(VRAMNo),
		{verdict: VRAMTight, source: "repo", note: "the 24 GB example uses uint3 with an accuracy recovery adapter (train_lora_qwen_image_24gb.yaml)",
			recipe: "3-bit quantization (uint3 + ARA), cache_text_embeddings and low VRAM mode", upgrade: recipe{qtypes: []string{"uint2", "uint3"}, lowVRAM: true}},
		{verdict: VRAMFits, source: "estimate", note: "at uint3-uint4"},
		{verdict: VRAMFits, source: "estimate", note: "with qfloat8"},
		est(VRAMFits),
	},
	"qwen_image_edit": {
		est(VRAMNo), est(VRAMNo),
		{verdict: VRAMFits, source: "repo", note: "with uint3 (train_lora_qwen_image_edit_32gb.yaml)", recipe: "uint3"},
		est(VRAMFits), est(VRAMFits),
	},
	"qwen_image_edit_plus": {
		est(VRAMNo), est(VRAMNo),
		{verdict: VRAMFits, source: "repo", note: "with uint3 (train_lora_qwen_image_edit_2509_32gb.yaml)", recipe: "uint3"},
		est(VRAMFits), est(VRAMFits),
	},
	"wan22_14b": {
		est(VRAMNo),
		{verdict: VRAMTight, source: "repo", note: "trains one expert at a time (train_lora_wan22_14b_24gb.yaml)",
			recipe:  "4-bit quantization (uint4 + ARA), cache_text_embeddings or unload_text_encoder, and low VRAM mode",
			upgrade: recipe{qtypes: []string{"uint2", "uint3", "uint4"}, lowVRAM: true}},
		{verdict: VRAMFits, source: "estimate", note: "for short clips"},
		est(VRAMFits), est(VRAMFits),
	},
	"wan22_5b": {
		{verdict: VRAMTight, source: "estimate", recipe: "low VRAM mode", upgrade: recipe{lowVRAM: true}},
		est(VRAMFits), est(VRAMFits), est(VRAMFits), est(VRAMFits),
	},
	"hidream": {
		est(VRAMNo),
		{verdict: VRAMNo, source: "repo", note: "not possible on a single 24 GB card (train_lora_hidream_48.yaml)"},
		{verdict: VRAMNo, source: "estimate", note: "needs about 35.2 GB"},
		{verdict: VRAMFits, source: "repo", note: "train_lora_hidream_48.yaml"},
		est(VRAMFits),
	},
	"ltx2": {
		est(VRAMNo), est(VRAMNo),
		{verdict: VRAMTight, source: "vendor", note: "Lightricks' own trainer, not ai-toolkit's", recipe: "the low-VRAM INT8 config",
			upgrade: recipe{qtypes: []string{"int8", "qint8", "uint8", "uint4", "uint3"}, lowVRAM: true}},
		est(VRAMFits),
		{verdict: VRAMFits, source: "vendor", note: "the standard config; Lightricks' own trainer, not ai-toolkit's"},
	},
	"zimage": {
		{verdict: VRAMFits, source: "estimate", note: "at 768 or below; tight at 1024"},
		est(VRAMFits), est(VRAMFits), est(VRAMFits), est(VRAMFits),
	},
	"krea2": {
		{verdict: VRAMTight, source: "estimate", recipe: "low VRAM mode", upgrade: recipe{lowVRAM: true}},
		est(VRAMFits), est(VRAMFits), est(VRAMFits), est(VRAMFits),
	},
}

// tierIndex is the index into TierSizes for a card of vramGB, or -1 below
// the smallest tier.
func tierIndex(vramGB float64) int {
	i := -1
	for j, t := range TierSizes {
		if vramGB >= t*tierSlack {
			i = j
		}
	}
	return i
}

// VRAMFor is the verdict for training model m on one GPU of vramGB. The
// job's own settings shift it: a tight cell becomes fits when the job
// already uses the cell's recipe, and a fits cell that relies on
// quantization becomes tight when the job section (it names its model)
// turns quantization off. Below the smallest tier nothing fits.
func VRAMFor(m ModelRef, vramGB float64) VRAMFit {
	cells, ok := vramTable[m.Arch]
	if !ok || vramGB <= 0 {
		return VRAMFit{Verdict: VRAMUnknown, Note: "No VRAM data for this model."}
	}
	i := tierIndex(vramGB)
	if i < 0 {
		return VRAMFit{Verdict: VRAMNo, Source: "estimate", Note: fmt.Sprintf("%s needs more than %.0f GB.", ArchName(m.Arch), vramGB)}
	}
	c := cells[i]
	fit := VRAMFit{Verdict: c.verdict, Source: c.source, Recipe: c.recipe}
	switch {
	case c.verdict == VRAMTight && c.upgrade.met(m):
		fit.Verdict = VRAMFits
	case c.verdict == VRAMFits && c.needs.quantize && m.NameOrPath != "" && !m.Quantize && m.QType == "":
		fit.Verdict = VRAMTight
	case m.Arch == "zimage" && i == 0 && m.Resolution > 768:
		fit.Verdict, fit.Recipe = VRAMTight, "train at 768 or below"
	}
	fit.Note = vramNote(m.Arch, TierSizes[i], fit, c.note)
	return fit
}

// vramNote is the one-line note: "FLUX.1 fits in 24 GB with ...".
func vramNote(arch string, tier float64, fit VRAMFit, detail string) string {
	name := ArchName(arch)
	var s string
	switch fit.Verdict {
	case VRAMFits:
		s = fmt.Sprintf("%s fits in %.0f GB", name, tier)
	case VRAMTight:
		s = fmt.Sprintf("%s fits in %.0f GB only with %s", name, tier, fit.Recipe)
	default:
		s = fmt.Sprintf("%s doesn't fit in %.0f GB", name, tier)
	}
	if detail != "" {
		s += "; " + detail
	}
	if fit.Source == "estimate" {
		s += " (estimate)"
	}
	return s + "."
}

// Tiers is the verdict at each table size (16, 24, 32, 48, 80 GB).
func Tiers(m ModelRef) []Tier {
	out := make([]Tier, len(TierSizes))
	for i, t := range TierSizes {
		out[i] = Tier{VRAMGB: t, VRAMFit: VRAMFor(m, t)}
	}
	return out
}

// KnownArch reports a model arch the VRAM table has cells for.
func KnownArch(arch string) bool {
	_, ok := vramTable[arch]
	return ok
}

// MinVRAMGB is the VRAM floor offers are filtered to for model m: the
// smallest tier whose verdict is fits or tight, less the tier slack, so
// every card of that class passes. 0 for an unknown arch, or when nothing
// up to 80 GB fits (the floor is then the 80 GB tier).
func MinVRAMGB(m *ModelRef) float64 {
	if m == nil || !KnownArch(m.Arch) {
		return 0
	}
	for _, t := range Tiers(*m) {
		if t.Verdict == VRAMFits || t.Verdict == VRAMTight {
			return t.VRAMGB * tierSlack
		}
	}
	return TierSizes[len(TierSizes)-1] * tierSlack
}
