package v1

// The GPU recommender (handlers/v1recommend) in the OpenAPI document. Its
// shape is in docs/headless.md, "GPU recommendation".
func init() {
	q := []string{"model:string", "image:string", "disk_gb:integer", "cloud:string", "max_price:number"}
	ops = append(ops,
		op{"GET", "/v1/providers/{kind}/recommend", "Providers", "The default GPU pick on one provider: best value and best speed in the smallest VRAM tier the model fits (only a suggestion)", q, "", 200, "object"},
		op{"GET", "/v1/hosts/recommend", "hosts", "The default GPU pick across every provider with a key; provider failures are notes", append([]string{"providers:string"}, q...), "", 200, "object"},
	)
}
