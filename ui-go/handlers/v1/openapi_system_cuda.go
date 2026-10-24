package v1

// The machine-side CUDA self-check (system_cuda.go) in the OpenAPI document.
// Its shape is CudaCheck in docs/headless.md, "Images and hosts".
func init() {
	ops = append(ops,
		op{"GET", "/v1/system/cuda", "system", "Run a tiny CUDA kernel through the trainer's torch on every GPU; ok false names the cause (refresh=1 runs it again)", []string{"refresh:boolean"}, "", 200, "object"},
	)
}
