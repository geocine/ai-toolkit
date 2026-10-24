package hostfit

import (
	"slices"
	"strings"
)

// GPU is one row of the GPU table: what a provider's GPU name means for an
// image (IMAGE-PLAN §3.3) and how fast it trains (MASTER §3.5).
type GPU struct {
	Name     string   `json:"name"`     // display: "RTX 4090"
	SM       int      `json:"sm"`       // compute capability x 10: 89
	VRAMGB   float64  `json:"vram_gb"`  // nominal per GPU
	Class    string   `json:"class"`    // geforce | pro | datacenter
	Vendor   string   `json:"vendor"`   // nvidia | amd
	Platform string   `json:"platform"` // linux/amd64 | linux/arm64
	Aliases  []string `json:"aliases"`  // exact provider names, matched after normalisation

	// Speed is the relative LoRA training throughput of ONE GPU, RTX 4090 =
	// 1.00; 0 = unknown (never recommended). Sm_75 and blocked GPUs keep a
	// speed for display only.
	Speed       float64 `json:"speed"`
	SpeedSource string  `json:"speed_source"` // key into SpeedSources; "" when Speed is 0

	// Unverified: runs through a same-major build but has not been tested
	// (B300 runs the sm_100 build).
	Unverified bool `json:"-"`
}

// SpeedSources documents where each Speed value comes from. Values are
// UNCONFIRMED seeds: replace a row with "measured" once ai-toolkit jobs on that
// GPU report s/it (the Job speed_string) for a reference config.
var SpeedSources = map[string]string{
	"est-public": "Estimate (UNCONFIRMED). Relative 16-bit training throughput from public GPU comparisons (T. Dettmers, \"Which GPU(s) to Get for Deep Learning\", 2023 update; Lambda GPU benchmarks, lambda.ai/gpu-benchmarks), rounded to 0.05.",
	"est-scaled": "Estimate (UNCONFIRMED). Newer than the public comparisons: scaled from the nearest listed GPU of the same architecture by NVIDIA's memory-bandwidth and dense BF16 tensor-throughput ratios, then rounded to 0.05.",
}

const (
	classGeForce    = "geforce"
	classPro        = "pro"
	classDatacenter = "datacenter"
	vendorNVIDIA    = "nvidia"
	vendorAMD       = "amd"
	platformAMD64   = "linux/amd64"
	platformARM64   = "linux/arm64"
	pub             = "est-public"
	scaled          = "est-scaled"
)

// g is a table row on linux/amd64 from NVIDIA.
func g(name string, sm int, vram float64, class string, speed float64, src string, aliases ...string) GPU {
	return GPU{Name: name, SM: sm, VRAMGB: vram, Class: class, Vendor: vendorNVIDIA, Platform: platformAMD64,
		Aliases: aliases, Speed: speed, SpeedSource: src}
}

// gpuTable is the seed table. CC values from developer.nvidia.com/cuda-gpus;
// aliases are the provider names seen in the RunPod gpuTypeIds enum, QuickPod
// /availablegputypes and Vast offers. Speeds are MASTER §3.5's seeds.
var gpuTable = func() []GPU {
	t := []GPU{
		// Pascal and Maxwell: no kernel image in the cu130 build (blocked).
		g("GTX 1060", 61, 6, classGeForce, 0, ""),
		g("GTX 1070", 61, 8, classGeForce, 0, "", "GTX 1070 Ti"),
		g("GTX 1080", 61, 8, classGeForce, 0, ""),
		g("GTX 1080 Ti", 61, 11, classGeForce, 0, ""),
		g("Titan Xp", 61, 12, classGeForce, 0, "", "TITAN Xp", "Titan X Pascal"),
		g("GTX Titan X", 52, 12, classGeForce, 0, "", "GTX TITAN X"),
		g("GTX 980 Ti", 52, 6, classGeForce, 0, ""),
		g("Quadro P4000", 61, 8, classPro, 0, ""),
		g("Quadro P5000", 61, 16, classPro, 0, ""),
		g("Tesla P100", 60, 16, classDatacenter, 0, "", "P100", "Tesla P100-PCIE-16GB"),
		g("Tesla P40", 61, 24, classDatacenter, 0, "", "P40"),
		// Volta: blocked.
		g("Tesla V100", 70, 16, classDatacenter, 0, "", "V100", "Tesla V100-PCIE-16GB", "Tesla V100-SXM2-16GB",
			"Tesla V100-SXM2-32GB", "Tesla V100-FHHL-16GB", "V100-SXM2", "V100 PCIE", "V100 SXM2"),
		g("Tesla V100S", 70, 32, classDatacenter, 0, "", "V100S", "Tesla V100S-PCIE-32GB"),
		g("Titan V", 70, 12, classGeForce, 0, "", "TITAN V"),
		// Turing (sm_75): torch runs, flash-attn does not; hidden by default.
		g("Tesla T4", 75, 16, classDatacenter, 0.12, pub, "T4"),
		g("RTX 2060", 75, 6, classGeForce, 0.12, pub, "RTX 2060 SUPER"),
		g("RTX 2070", 75, 8, classGeForce, 0.15, pub, "RTX 2070 SUPER"),
		g("RTX 2080", 75, 8, classGeForce, 0.20, pub, "RTX 2080 SUPER"),
		g("RTX 2080 Ti", 75, 11, classGeForce, 0.25, pub),
		g("Titan RTX", 75, 24, classGeForce, 0.25, pub, "TITAN RTX"),
		g("GTX 1660", 75, 6, classGeForce, 0.12, pub, "GTX 1660 SUPER", "GTX 1660 Ti", "GTX 1660 S"),
		g("Quadro RTX 4000", 75, 8, classPro, 0.15, pub, "Q RTX 4000"),
		g("Quadro RTX 5000", 75, 16, classPro, 0.20, pub, "Q RTX 5000"),
		g("Quadro RTX 6000", 75, 24, classPro, 0.25, pub, "Q RTX 6000"),
		g("Quadro RTX 8000", 75, 48, classPro, 0.25, pub, "Q RTX 8000"),
		// Ampere.
		g("A100 PCIe 40GB", 80, 40, classDatacenter, 0.95, pub, "A100 PCIE", "NVIDIA A100-PCIE-40GB", "A100 40GB PCIe"),
		g("A100 PCIe 80GB", 80, 80, classDatacenter, 0.95, pub, "NVIDIA A100 80GB PCIe", "A100 80GB PCIe", "A100X"),
		g("A100 SXM4 40GB", 80, 40, classDatacenter, 1.05, pub, "NVIDIA A100-SXM4-40GB", "A100-SXM4-40GB"),
		g("A100 SXM4 80GB", 80, 80, classDatacenter, 1.05, pub, "NVIDIA A100-SXM4-80GB", "A100-SXM4-80GB", "A100 SXM4", "A100 SXM"),
		g("A30", 80, 24, classDatacenter, 0.55, pub, "NVIDIA A30"),
		g("RTX 3060", 86, 12, classGeForce, 0.20, pub, "RTX 3060 12GB"),
		g("RTX 3070", 86, 8, classGeForce, 0.30, pub, "RTX 3070 Ti"),
		g("RTX 3080", 86, 10, classGeForce, 0.45, pub),
		g("RTX 3080 Ti", 86, 12, classGeForce, 0.50, pub),
		g("RTX 3090", 86, 24, classGeForce, 0.55, pub),
		g("RTX 3090 Ti", 86, 24, classGeForce, 0.60, pub),
		g("RTX 3060 Ti", 86, 8, classGeForce, 0, ""),
		g("RTX 3060 Laptop GPU", 86, 6, classGeForce, 0, "", "RTX 3060 Laptop"),
		g("RTX A2000", 86, 6, classPro, 0.20, pub, "RTX A2000 12GB", "A2000"),
		g("RTX A4000", 86, 16, classPro, 0.30, pub, "A4000"),
		g("RTX A4500", 86, 20, classPro, 0.38, pub, "A4500"),
		g("RTX A5000", 86, 24, classPro, 0.42, pub, "A5000"),
		g("RTX A6000", 86, 48, classPro, 0.55, pub, "A6000"),
		g("A10", 86, 24, classDatacenter, 0.35, pub, "NVIDIA A10"),
		g("A10G", 86, 24, classDatacenter, 0.33, pub, "NVIDIA A10G"),
		g("A40", 86, 48, classDatacenter, 0.50, pub, "NVIDIA A40"),
		// Ada.
		g("RTX 4060", 89, 8, classGeForce, 0, ""),
		g("RTX 4060 Ti", 89, 16, classGeForce, 0.30, pub),
		g("RTX 4070", 89, 12, classGeForce, 0.40, pub),
		g("RTX 4070 SUPER", 89, 12, classGeForce, 0, ""),
		g("RTX 4070 Ti", 89, 12, classGeForce, 0.50, pub),
		g("RTX 4070 Ti SUPER", 89, 16, classGeForce, 0, ""),
		g("RTX 4080", 89, 16, classGeForce, 0.70, pub, "RTX 4080 SUPER"),
		g("RTX 4090", 89, 24, classGeForce, 1.00, pub, "RTX 4090D"),
		g("L4", 89, 24, classDatacenter, 0.30, pub, "NVIDIA L4"),
		g("L40", 89, 48, classDatacenter, 0.85, pub, "NVIDIA L40"),
		g("L40S", 89, 48, classDatacenter, 0.95, pub, "NVIDIA L40S"),
		g("RTX 2000 Ada", 89, 16, classPro, 0.20, pub, "RTX 2000 Ada Generation", "RTX 2000Ada"),
		g("RTX 4000 Ada", 89, 20, classPro, 0.33, pub, "RTX 4000 Ada Generation", "RTX 4000Ada"),
		// The SFF card is the 70 W RTX 4000 Ada: scaled down by its clocks.
		g("RTX 4000 SFF Ada", 89, 20, classPro, 0.25, scaled, "RTX 4000 SFF Ada Generation"),
		g("RTX 4500 Ada", 89, 24, classPro, 0.45, pub, "RTX 4500 Ada Generation", "RTX 4500Ada"),
		g("RTX 5000 Ada", 89, 32, classPro, 0.65, pub, "RTX 5000 Ada Generation", "RTX 5000Ada"),
		g("RTX 6000 Ada", 89, 48, classPro, 0.95, pub, "RTX 6000 Ada Generation", "RTX 6000Ada"),
		// Hopper.
		g("H100 PCIe", 90, 80, classDatacenter, 1.55, pub, "NVIDIA H100 PCIe", "H100 PCIE"),
		g("H100 NVL", 90, 94, classDatacenter, 1.80, pub, "NVIDIA H100 NVL"),
		g("H100 SXM", 90, 80, classDatacenter, 2.00, pub, "NVIDIA H100 80GB HBM3", "H100 80GB HBM3", "H100 SXM5"),
		g("H200", 90, 141, classDatacenter, 2.20, scaled, "NVIDIA H200", "H200 SXM"),
		g("H200 NVL", 90, 141, classDatacenter, 2.10, scaled, "NVIDIA H200 NVL"),
		// Blackwell.
		g("B200", 100, 180, classDatacenter, 3.20, scaled, "NVIDIA B200"),
		g("RTX 5060 Ti", 120, 16, classGeForce, 0.35, scaled),
		g("RTX 5070", 120, 12, classGeForce, 0.45, scaled),
		g("RTX 5070 Ti", 120, 16, classGeForce, 0.60, scaled),
		g("RTX 5080", 120, 16, classGeForce, 0.80, scaled),
		g("RTX 5090", 120, 32, classGeForce, 1.35, scaled),
		// The laptop 5090 has half the desktop card's memory bandwidth.
		g("RTX 5090 Laptop GPU", 120, 24, classGeForce, 0.60, scaled, "RTX 5090 Laptop"),
		g("RTX PRO 4000 Blackwell", 120, 24, classPro, 0.45, scaled, "RTX PRO 4000"),
		g("RTX PRO 4500 Blackwell", 120, 32, classPro, 0.55, scaled, "RTX PRO 4500"),
		g("RTX PRO 5000 Blackwell", 120, 48, classPro, 0.85, scaled, "RTX PRO 5000"),
		g("RTX PRO 6000 Blackwell", 120, 96, classPro, 1.40, scaled,
			"RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000 Blackwell Workstation Edition",
			"RTX PRO 6000 S", "RTX PRO 6000 WS", "RTX PRO 6000"),
		g("RTX PRO 6000 Blackwell Max-Q", 120, 96, classPro, 1.15, scaled,
			"RTX PRO 6000 Blackwell Max-Q Workstation Edition", "RTX PRO 6000 Max-Q"),
	}
	b300 := g("B300", 103, 288, classDatacenter, 3.40, scaled, "NVIDIA B300 SXM6 AC", "B300 SXM6 AC")
	b300.Unverified = true
	// One MIG slice of a B300: about an eighth of the card.
	mig := g("B300 MIG 1g.34gb", 103, 34, classDatacenter, 0.40, scaled, "NVIDIA B300 SXM6 AC MIG 1g.34gb")
	mig.Unverified = true
	gh200 := g("GH200", 90, 96, classDatacenter, 0, "", "NVIDIA GH200 480GB", "GH200 480GB", "GH200 SXM")
	gh200.Platform = platformARM64
	gb10 := g("GB10", 121, 128, classPro, 0, "", "NVIDIA GB10", "DGX Spark")
	gb10.Platform = platformARM64
	mi300x := GPU{Name: "MI300X", SM: 0, VRAMGB: 192, Class: classDatacenter, Vendor: vendorAMD, Platform: platformAMD64,
		Aliases: []string{"AMD Instinct MI300X OAM", "Instinct MI300X", "AMD Instinct MI300X"}}
	return append(t, b300, mig, gh200, gb10, mi300x)
}()

// gpuIndex maps every normalised name and alias to its row.
var gpuIndex = func() map[string]int {
	idx := map[string]int{}
	for i, row := range gpuTable {
		for _, n := range append([]string{row.Name}, row.Aliases...) {
			idx[normalise(n)] = i
		}
	}
	return idx
}()

// normalise lowers a GPU name, treats "-" and "_" as spaces, collapses
// spaces and strips the vendor words providers add ("NVIDIA GeForce RTX
// 4090" -> "rtx 4090"). A laptop suffix is kept.
func normalise(name string) string {
	s := strings.ToLower(strings.TrimSpace(name))
	s = strings.NewReplacer("-", " ", "_", " ").Replace(s)
	s = strings.Join(strings.Fields(s), " ")
	for _, p := range []string{"nvidia ", "geforce "} {
		s = strings.TrimPrefix(s, p)
	}
	return s
}

// Lookup finds a provider GPU name in the table.
func Lookup(name string) (GPU, bool) {
	i, ok := gpuIndex[normalise(name)]
	if !ok {
		return GPU{}, false
	}
	return cloneGPU(gpuTable[i]), true
}

// Speed returns the table speed for a provider GPU name (Lookup's
// normalisation). ok is false for a GPU the table does not know; a known GPU
// with no rating has speed 0.
func Speed(name string) (speed float64, source string, ok bool) {
	row, ok := Lookup(name)
	if !ok {
		return 0, "", false
	}
	return row.Speed, row.SpeedSource, true
}

// GPUs is the whole table, in table order (a copy).
func GPUs() []GPU {
	out := make([]GPU, len(gpuTable))
	for i, row := range gpuTable {
		out[i] = cloneGPU(row)
	}
	return out
}

func cloneGPU(row GPU) GPU {
	row.Aliases = slices.Clone(row.Aliases)
	if row.Aliases == nil {
		row.Aliases = []string{}
	}
	return row
}

// knownSM is every compute capability the table holds on an NVIDIA amd64
// GPU, ascending: the set a Vast compute_cap must land on to be trusted, and
// the set a Filter's SMAllowed is drawn from.
func knownSM(platform string) []int {
	var out []int
	for _, row := range gpuTable {
		if row.Vendor == vendorNVIDIA && row.SM > 0 && (platform == "" || row.Platform == platform) && !slices.Contains(out, row.SM) {
			out = append(out, row.SM)
		}
	}
	slices.Sort(out)
	return out
}

// isKnownSM reports a compute capability some table GPU has (any platform).
func isKnownSM(sm int) bool { return slices.Contains(knownSM(""), sm) }
