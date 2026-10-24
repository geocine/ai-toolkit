package codemode

// The GPU recommender (handlers/v1recommend, docs/headless.md "GPU
// recommendation"). GpuRecommendation is in the provision types
// (ops_provision.go).

func init() {
	Ops = append(Ops, recommendOps...)
}

var recommendOps = []Op{
	{Group: "providers", Name: "recommend", Method: "GET", Path: "/v1/providers/{kind}/recommend", Result: "GpuRecommendation",
		Doc: "The default GPU pick on one provider: value (most speed per $/h) and speed (fastest), from the smallest VRAM tier the model fits (24, then 32, 48, ...), among offers with fast internet, enough fast disk and a driver for the image's CUDA. reason is the one line to show the user. It is only a default: show both picks, let the user choose, then quote price_per_hour and get their OK before machines.launch.",
		Params: []Param{p("kind", "ProviderKind", "path", ""), opt("model", "string", "query", "model arch (flux, qwen_image, wan22_14b, ...): the ladder starts at the smallest tier it fits"),
			opt("image", "string", "query", "default ghcr.io/geocine/aitoolkit-go:latest"), opt("disk_gb", "number", "query", "default: what the image and model need"),
			opt("cloud", `"secure" | "community"`, "query", "RunPod only"), opt("max_price", "number", "query", "$/h ceiling")}},
	{Group: "hosts", Name: "recommend", Method: "GET", Path: "/v1/hosts/recommend", Result: "GpuRecommendation",
		Doc: "providers.recommend across every provider with a key (or the comma list in providers). notes says which providers could not be searched and why. The same rules apply: it only suggests; quote the price and ask before machines.launch.",
		Params: []Param{opt("providers", "string", "query", "comma list, e.g. runpod,vast"), opt("model", "string", "query", ""), opt("image", "string", "query", ""),
			opt("disk_gb", "number", "query", ""), opt("cloud", `"secure" | "community"`, "query", ""), opt("max_price", "number", "query", "")}},
}
