package codemode

// The machine-side CUDA self-check (handlers/v1/system_cuda.go). Its result
// type, CudaCheck, is in ImageTypesTS.

func init() {
	Ops = append(Ops, Op{Group: "system", Name: "cuda", Method: "GET", Path: "/v1/system/cuda", Result: "CudaCheck",
		Doc:    "Run a tiny CUDA kernel through the trainer's torch and report the arch list, device capability and driver; failure names the cause (no_kernel_image, driver_too_old, ...). Run it on a new GPU server before downloading models. A success is cached; refresh runs it again.",
		Params: []Param{opt("refresh", "boolean", "query", "run the check again instead of returning the cached success")}})
}
