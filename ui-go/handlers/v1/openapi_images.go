package v1

import "github.com/geocine/aitoolkit/handlers/v1images"

// The image check and host fit endpoints (handlers/v1images) in the OpenAPI
// document. Their shapes are in docs/headless.md, "Images and hosts".
func init() {
	ops = append(ops,
		op{"POST", "/v1/images/check", "images", "Check an image (and with host or offer, that host) before a GPU server runs it; 200 even when it blocks", nil, "object", 200, "object"},
		op{"GET", "/v1/images/tags", "images", "Tags of an image with digest, build date, flavor and CUDA floor", []string{"image:string", "limit:integer"}, "", 200, "object"},
		op{"POST", "/v1/images/recommend", "images", "Pick the image for a host and say why", nil, "object", 200, "object"},
		op{"GET", "/v1/hosts/gpus", "hosts", "Known GPUs and whether the default image runs on them", nil, "", 200, "object"},
		op{"POST", "/v1/hosts/fit", "hosts", "VRAM verdict per GPU size and the disk a model needs", nil, "object", 200, "object"},
	)
	// images.recommend prefers the build this server runs: the one /v1/health reports.
	v1images.ServerBuild = func() (revision, upstream string) {
		if v := upstreamVersion(); v != nil {
			upstream = *v
		}
		return version(), upstream
	}
}
