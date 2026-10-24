package v1

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// GET /v1/system/cuda is the machine-side CUDA smoke check (IMAGE-PLAN §4.6):
// it runs a tiny kernel through the trainer's torch on every GPU, so a new
// GPU server whose image cannot run on its GPU says so before models
// download. The provisioning service calls it once a machine is ready.
func init() {
	addRoutes(func(mux *http.ServeMux) {
		httpx.Handle(mux, "GET /v1/system/cuda", systemCUDA)
	})
}

// cudaCheck is the CudaCheck of internal/codemode/image_types.go.
type cudaCheck struct {
	OK        bool         `json:"ok"`
	CheckedAt string       `json:"checked_at"`
	Torch     string       `json:"torch"`
	CUDA      string       `json:"cuda"`
	ArchList  []string     `json:"arch_list"`
	Devices   []cudaDevice `json:"devices"`
	Error     *string      `json:"error"`
	Failure   *string      `json:"failure"`
}

type cudaDevice struct {
	Index      int    `json:"index"`
	Name       string `json:"name"`
	Capability string `json:"capability"`
	Driver     string `json:"driver"`
}

// Named causes of a failed check (CudaCheck.failure).
const (
	cudaNoKernel     = "no_kernel_image"
	cudaOldDriver    = "driver_too_old"
	cudaNoGPU        = "no_gpu"
	cudaTorchMissing = "torch_missing"
	cudaTimedOut     = "timeout"
)

// cudaScript reports torch's CUDA build and arch list, then runs one kernel
// per device. It always prints one JSON line; stage says how far it got.
const cudaScript = `import json
r = {"stage": "import", "torch": "", "cuda": "", "arch_list": [], "devices": [], "error": None}
try:
    import torch
    r["torch"], r["cuda"], r["stage"] = torch.__version__, torch.version.cuda or "", "init"
    flags = getattr(torch._C, "_cuda_getArchFlags", lambda: None)()
    r["arch_list"] = flags.split() if flags else []
    torch.cuda.init()
    r["stage"] = "kernel"
    for i in range(torch.cuda.device_count()):
        r["devices"].append({"index": i, "name": torch.cuda.get_device_name(i), "capability": "%d.%d" % torch.cuda.get_device_capability(i)})
        v = torch.ones(8, device="cuda:%d" % i).mul(2).sum().item()
        if v != 16:
            raise RuntimeError("cuda:%d computed %r, want 16" % (i, v))
    r["stage"] = "done"
except Exception as e:
    r["error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(r))
`

// cudaQueryArgs is the device query of this check, a separate nvidia-smi
// call: sysstats.NvQueryArgs mirrors upstream's /api/gpu and stays as it is.
var cudaQueryArgs = []string{"--query-gpu=index,name,compute_cap,driver_version", "--format=csv,noheader"}

var (
	// cudaTimeout bounds one check: importing torch is slow on a cold disk.
	cudaTimeout = 90 * time.Second
	// cudaWaitDelay is how long Run waits for the pipes after a kill: a
	// grandchild that keeps them open must not hold the check past its bound.
	cudaWaitDelay = 5 * time.Second
	// cudaExec runs one command of the check (tests swap it for a fake).
	cudaExec = execCUDA
)

func execCUDA(ctx context.Context, name string, args ...string) (stdout, stderr []byte, err error) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = paths.ToolkitRoot
	// torch numbers devices like nvidia-smi only in PCI bus order
	cmd.Env = append(os.Environ(), "CUDA_DEVICE_ORDER=PCI_BUS_ID")
	var out, errOut bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errOut
	cmd.WaitDelay = cudaWaitDelay
	err = cmd.Run()
	return out.Bytes(), errOut.Bytes(), err
}

// cudaState runs one check at a time: callers that arrive while one runs
// wait for its result. A successful result is kept until refresh asks again.
var cudaState struct {
	mu     sync.Mutex
	cached *cudaCheck
	run    *cudaRun
}

type cudaRun struct {
	done   chan struct{}
	result cudaCheck
}

func systemCUDA(w http.ResponseWriter, r *http.Request) {
	f := r.URL.Query().Get("refresh")
	res, err := cudaResult(r.Context(), f == "true" || f == "1")
	if err != nil {
		return // the caller went away; the check finishes for the next one
	}
	v1api.OK(w, res)
}

// cudaResult is the cached success, else the result of the running check,
// else of a new one. The check is not tied to ctx, so a caller that leaves
// does not cancel it for the others.
func cudaResult(ctx context.Context, refresh bool) (cudaCheck, error) {
	s := &cudaState
	s.mu.Lock()
	if s.cached != nil && !refresh {
		res := *s.cached
		s.mu.Unlock()
		return res, nil
	}
	run := s.run
	if run == nil {
		run = &cudaRun{done: make(chan struct{})}
		s.run = run
		go func() {
			res := checkCUDA(context.Background())
			s.mu.Lock()
			s.run, s.cached = nil, nil
			if res.OK {
				s.cached = &res
			}
			run.result = res
			s.mu.Unlock()
			close(run.done)
		}()
	}
	s.mu.Unlock()
	select {
	case <-run.done:
		return run.result, nil
	case <-ctx.Done():
		return cudaCheck{}, ctx.Err()
	}
}

// pyCUDA is what cudaScript prints.
type pyCUDA struct {
	Stage    string       `json:"stage"`
	Torch    string       `json:"torch"`
	CUDA     string       `json:"cuda"`
	ArchList []string     `json:"arch_list"`
	Devices  []cudaDevice `json:"devices"`
	Error    *string      `json:"error"`
}

// checkCUDA runs the device query and the torch script once.
func checkCUDA(ctx context.Context) cudaCheck {
	ctx, cancel := context.WithTimeout(ctx, cudaTimeout)
	defer cancel()
	res := cudaCheck{CheckedAt: v1api.Time(time.Now()), ArchList: []string{}, Devices: []cudaDevice{}}
	smi := queryCUDADevices(ctx)

	stdout, stderr, err := cudaExec(ctx, paths.ResolvePython(), "-c", cudaScript)
	var py pyCUDA
	parsed := false
	if line := lastJSONLine(stdout); line != nil {
		parsed = json.Unmarshal(line, &py) == nil
	}
	if parsed {
		res.Torch, res.CUDA = py.Torch, py.CUDA
		if py.ArchList != nil {
			res.ArchList = py.ArchList
		}
	}
	res.Devices = mergeDevices(py.Devices, smi)

	fail := func(failure, msg string) cudaCheck {
		msg = cudaTail(strings.TrimSpace(msg), 1000)
		res.Error = &msg
		if failure != "" {
			res.Failure = &failure
		}
		return res
	}
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		return fail(cudaTimedOut, fmt.Sprintf("the CUDA check did not finish in %s", cudaTimeout))
	case errors.Is(err, exec.ErrNotFound) || errors.Is(err, os.ErrNotExist):
		return fail(cudaTorchMissing, "the trainer's Python was not found: "+err.Error())
	case !parsed:
		text := string(stderr)
		if strings.TrimSpace(text) == "" && err != nil {
			text = err.Error()
		}
		return fail(classifyCUDA("", text), text)
	case py.Error != nil:
		text := *py.Error
		if s := strings.TrimSpace(string(stderr)); s != "" {
			text += "\n" + s
		}
		return fail(classifyCUDA(py.Stage, text), text)
	case len(py.Devices) == 0:
		return fail(cudaNoGPU, "torch sees no CUDA device")
	}
	res.OK = true
	return res
}

// classifyCUDA names the cause of a failed check from the script's stage
// and the error text. The CUDA patterns are imagecheck.ClassifyFailure's,
// matched line by line: its image-pull rules come first, so a stray
// "Permission denied" elsewhere in stderr must not hide a CUDA line.
func classifyCUDA(stage, text string) string {
	for _, line := range strings.Split(text, "\n") {
		if f, ok := imagecheck.ClassifyFailure(line); ok && (f.Code == cudaNoKernel || f.Code == cudaOldDriver) {
			return f.Code
		}
	}
	t := strings.ToLower(text)
	switch {
	case stage == "import" && (strings.Contains(t, "no module named") || strings.Contains(t, "importerror") || strings.Contains(t, "modulenotfounderror")):
		return cudaTorchMissing
	case strings.Contains(t, "not compiled with cuda"):
		return cudaTorchMissing
	case strings.Contains(t, "found no nvidia driver"), strings.Contains(t, "no cuda gpus are available"),
		strings.Contains(t, "no cuda-capable device"):
		return cudaNoGPU
	}
	return ""
}

// queryCUDADevices is nvidia-smi's index, name, compute capability and
// driver per GPU; nil when nvidia-smi is missing or fails.
func queryCUDADevices(ctx context.Context) []cudaDevice {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	out, _, err := cudaExec(ctx, "nvidia-smi", cudaQueryArgs...)
	if err != nil {
		return nil
	}
	return parseCUDADevices(string(out))
}

// parseCUDADevices reads "0, NVIDIA GeForce RTX 4090, 8.9, 580.95.05" lines.
func parseCUDADevices(out string) []cudaDevice {
	var devs []cudaDevice
	for _, line := range strings.Split(out, "\n") {
		parts := strings.Split(strings.TrimSpace(line), ",")
		if len(parts) < 4 {
			continue
		}
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		n := len(parts)
		idx, err := strconv.Atoi(parts[0])
		if err != nil {
			continue
		}
		// a name with a comma in it keeps its parts
		name := strings.Join(parts[1:n-2], ", ")
		capability := parts[n-2]
		if strings.HasPrefix(capability, "[") { // [N/A] on drivers that do not report it
			capability = ""
		}
		devs = append(devs, cudaDevice{Index: idx, Name: name, Capability: capability, Driver: parts[n-1]})
	}
	return devs
}

// mergeDevices prefers torch's view (what the trainer sees) and fills the
// driver, and a capability torch could not read, from nvidia-smi by index.
// With no torch devices, nvidia-smi's list says which GPUs the host has.
func mergeDevices(torch, smi []cudaDevice) []cudaDevice {
	if len(torch) == 0 {
		if smi == nil {
			return []cudaDevice{}
		}
		return smi
	}
	byIndex := map[int]cudaDevice{}
	for _, d := range smi {
		byIndex[d.Index] = d
	}
	out := make([]cudaDevice, len(torch))
	for i, d := range torch {
		if s, ok := byIndex[d.Index]; ok {
			d.Driver = s.Driver
			if d.Capability == "" {
				d.Capability = s.Capability
			}
			if d.Name == "" {
				d.Name = s.Name
			}
		}
		out[i] = d
	}
	return out
}

// lastJSONLine is the last stdout line that looks like a JSON object (a
// library may print warnings before it).
func lastJSONLine(out []byte) []byte {
	lines := bytes.Split(bytes.TrimSpace(out), []byte("\n"))
	for i := len(lines) - 1; i >= 0; i-- {
		if l := bytes.TrimSpace(lines[i]); len(l) > 0 && l[0] == '{' {
			return l
		}
	}
	return nil
}

// cudaTail keeps the last n bytes of s: the end of a traceback names the error.
func cudaTail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return "..." + s[len(s)-n:]
}
