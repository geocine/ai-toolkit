package codemode

// Image checks and host fit (handlers/v1images, docs/headless.md "Images and
// hosts"). The types are ImageTypesTS (image_types.go); system.cuda, the
// machine-side check, is in ops_system_cuda.go.

func init() {
	RegisterTypes(ImageTypesTS)
	Ops = append(Ops, imageOps...)
}

const offerRefTS = "{ provider: ProviderKind; offer_id: string }"

var imageOps = []Op{
	{Group: "images", Name: "check", Method: "POST", Path: "/v1/images/check", Result: "ImageCheck",
		Doc: "Check an image before a GPU server runs it: does it exist and pull anonymously, is it linux/amd64, is it the ai-toolkit Go server, and (with host or offer) will this host's driver, GPU, disk and bandwidth run it. ok is false when anything blocks; machines.launch refuses the same blocks. launch_ref is the immutable reference a launch would use. Registry trouble is a warn, never a block.",
		Params: []Param{p("image", "string", "body", "e.g. ghcr.io/geocine/aitoolkit-go:latest"), opt("platform", "string", "body", "default linux/amd64"),
			opt("host", "HostFacts", "body", "host facts to check against (gpu_name alone is enough for the GPU checks)"),
			opt("offer", offerRefTS, "body", "an offer from providers.offers; its host is read again (wins over host)"),
			opt("model", "ModelRef", "body", "the job's model section, for disk and VRAM"), opt("disk_gb", "number", "body", "the disk the launch will ask for"),
			opt("refresh", "boolean", "body", "re-resolve the tag now")}},
	{Group: "images", Name: "tags", Method: "GET", Path: "/v1/images/tags", Result: "Items<ImageTag>",
		Doc:    "Tags of an image with digest, build date, flavor and CUDA floor (newest first): latest, flavor channels, versions and recent builds.",
		Params: []Param{opt("image", "string", "query", "repository, default ghcr.io/geocine/aitoolkit-go"), opt("limit", "number", "query", "default 8, max 20")}},
	{Group: "images", Name: "recommend", Method: "POST", Path: "/v1/images/recommend", Result: "ImageRecommendation",
		Doc: "Pick the image for a host and say why: the build this server runs if it is published and fits, else the newest that fits. choice is null with action filter_hosts when nothing fits; prefer filtering hosts over a different trainer version. Pass choice.image to machines.launch.",
		Params: []Param{opt("offer", offerRefTS, "body", "an offer from providers.offers"), opt("host", "HostFacts", "body", ""),
			opt("model", "ModelRef", "body", "the job's model section, for disk and VRAM"),
			opt("prefer", `"same" | "latest"`, "body", "default same"), opt("image", "string", "body", "a custom image to check instead"),
			opt("template_id", "string", "body", "QuickPod: the template whose image runs"), opt("disk_gb", "number", "body", "the disk the launch will ask for")}},
	{Group: "hosts", Name: "gpus", Method: "GET", Path: "/v1/hosts/gpus", Result: "Items<GpuInfo>",
		Doc: "Known GPUs with compute capability, VRAM, class and whether the default image runs on them (verdict and note; the driver is judged per host by images.check)."},
	{Group: "hosts", Name: "fit", Method: "POST", Path: "/v1/hosts/fit", Result: "HostFit",
		Doc: "VRAM verdict per GPU size and the disk a model needs (image + model download + working room). Verdicts are fits, tight (only with the named recipe), no, or unknown; source says repo example or estimate. Without gpus or host it answers the five sizes 16/24/32/48/80 GB.",
		Params: []Param{p("model", "ModelRef", "body", "the job's model section (arch is required)"), opt("image", "string", "body", "default ghcr.io/geocine/aitoolkit-go:latest"),
			opt("gpus", "{ name?: string; vram_gb?: number; count?: number }[]", "body", "GPUs to judge; a known name gives its VRAM"),
			opt("host", "HostFacts", "body", "a host to judge, as from an offer")}},
}
