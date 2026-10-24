package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/sysstats"
)

const smi4090 = "0, NVIDIA GeForce RTX 4090, 8.9, 580.95.05\n"

// fakeCUDA swaps the check's exec for py (the torch script) and an
// nvidia-smi that prints smi, with a fresh cache; it counts script runs.
func fakeCUDA(t *testing.T, smi string, py func(ctx context.Context) ([]byte, []byte, error)) *atomic.Int32 {
	t.Helper()
	runs := &atomic.Int32{}
	savedExec, savedTimeout := cudaExec, cudaTimeout
	cudaState.mu.Lock()
	cudaState.cached, cudaState.run = nil, nil
	cudaState.mu.Unlock()
	t.Cleanup(func() {
		cudaExec, cudaTimeout = savedExec, savedTimeout
		cudaState.mu.Lock()
		cudaState.cached, cudaState.run = nil, nil
		cudaState.mu.Unlock()
	})
	cudaExec = func(ctx context.Context, name string, args ...string) ([]byte, []byte, error) {
		if name == "nvidia-smi" {
			if !slices.Equal(args, []string{"--query-gpu=index,name,compute_cap,driver_version", "--format=csv,noheader"}) {
				t.Errorf("nvidia-smi args %q", args)
			}
			if smi == "" {
				return nil, nil, exec.ErrNotFound
			}
			return []byte(smi), nil, nil
		}
		if len(args) != 2 || args[0] != "-c" || !strings.Contains(args[1], "torch.ones(8") {
			t.Errorf("python args %q", args)
		}
		runs.Add(1)
		return py(ctx)
	}
	return runs
}

// pyAnswer answers like cudaScript with the given fields over the defaults.
func pyAnswer(fields map[string]any) func(context.Context) ([]byte, []byte, error) {
	r := map[string]any{"stage": "done", "torch": "2.13.0+cu130", "cuda": "13.0",
		"arch_list": []string{"sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"},
		"devices":   []map[string]any{{"index": 0, "name": "NVIDIA GeForce RTX 4090", "capability": "8.9"}}, "error": nil}
	for k, v := range fields {
		r[k] = v
	}
	b, _ := json.Marshal(r)
	return func(context.Context) ([]byte, []byte, error) {
		return append([]byte("UserWarning: something torch prints first\n"), b...), nil, nil
	}
}

func runCUDACheck(t *testing.T, refresh bool) cudaCheck {
	t.Helper()
	res, err := cudaResult(context.Background(), refresh)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func cudaFailure(c cudaCheck) string {
	if c.Failure == nil {
		return ""
	}
	return *c.Failure
}

func TestCudaOK(t *testing.T) {
	fakeCUDA(t, smi4090, pyAnswer(nil))
	c := runCUDACheck(t, false)
	if !c.OK || c.Error != nil || c.Failure != nil || c.Torch != "2.13.0+cu130" || c.CUDA != "13.0" || len(c.ArchList) != 6 || c.CheckedAt == "" {
		t.Fatalf("%+v", c)
	}
	if len(c.Devices) != 1 || c.Devices[0] != (cudaDevice{0, "NVIDIA GeForce RTX 4090", "8.9", "580.95.05"}) {
		t.Fatalf("devices %+v (driver comes from nvidia-smi)", c.Devices)
	}
	// the wire shape is CudaCheck: error and failure are null, not absent
	b, _ := json.Marshal(c)
	for _, k := range []string{`"error":null`, `"failure":null`, `"arch_list":[`, `"checked_at":"`} {
		if !strings.Contains(string(b), k) {
			t.Fatalf("%s lacks %s", b, k)
		}
	}
}

func TestCudaFailures(t *testing.T) {
	for _, tc := range []struct {
		name, smi, want, msg string
		py                   func(context.Context) ([]byte, []byte, error)
		devices              int
	}{
		{"no kernel image", smi4090, cudaNoKernel, "no kernel image is available", pyAnswer(map[string]any{"stage": "kernel",
			"error": "AcceleratorError: CUDA error: no kernel image is available for execution on the device"}), 1},
		{"no kernel image on stderr only", smi4090, cudaNoKernel, "not compatible with the current PyTorch installation",
			func(context.Context) ([]byte, []byte, error) {
				return nil, []byte("UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.\nSegmentation fault"), &exec.ExitError{}
			}, 1},
		{"no kernel image next to a denied warning", smi4090, cudaNoKernel, "no kernel image is available", pyAnswer(map[string]any{"stage": "kernel",
			"error": "OSError: [Errno 13] Permission denied: '/root/.cache/torch'\nRuntimeError: CUDA error: no kernel image is available for execution on the device"}), 1},
		{"driver too old after an unauthorized line", smi4090, cudaOldDriver, "too old",
			func(context.Context) ([]byte, []byte, error) {
				return nil, []byte("hub: 401 unauthorized, using the local cache\nRuntimeError: The NVIDIA driver on your system is too old (found version 12020)."), &exec.ExitError{}
			}, 1},
		{"driver too old", smi4090, cudaOldDriver, "too old", pyAnswer(map[string]any{"stage": "init", "devices": []any{},
			"error": "RuntimeError: The NVIDIA driver on your system is too old (found version 12020). Please update your GPU driver"}), 1},
		{"torch missing", smi4090, cudaTorchMissing, "No module named 'torch'", pyAnswer(map[string]any{"stage": "import", "torch": "", "cuda": "",
			"arch_list": []any{}, "devices": []any{}, "error": "ModuleNotFoundError: No module named 'torch'"}), 1},
		{"cpu-only torch", smi4090, cudaTorchMissing, "not compiled", pyAnswer(map[string]any{"stage": "init", "devices": []any{},
			"error": "AssertionError: Torch not compiled with CUDA enabled"}), 1},
		{"python missing", smi4090, cudaTorchMissing, "Python was not found",
			func(context.Context) ([]byte, []byte, error) { return nil, nil, exec.ErrNotFound }, 1},
		{"no gpu", "", cudaNoGPU, "no NVIDIA driver", pyAnswer(map[string]any{"stage": "init", "devices": []any{},
			"error": "RuntimeError: Found no NVIDIA driver on your system."}), 0},
		{"no devices", smi4090, cudaNoGPU, "no CUDA device", pyAnswer(map[string]any{"devices": []any{}}), 1},
		{"unknown", smi4090, "", "illegal memory access", pyAnswer(map[string]any{"stage": "kernel",
			"error": "RuntimeError: CUDA error: an illegal memory access was encountered"}), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fakeCUDA(t, tc.smi, tc.py)
			c := runCUDACheck(t, false)
			if c.OK || cudaFailure(c) != tc.want || c.Error == nil || !strings.Contains(*c.Error, tc.msg) {
				t.Fatalf("failure %q error %v, want %q with %q", cudaFailure(c), c.Error, tc.want, tc.msg)
			}
			if len(c.Devices) != tc.devices || (tc.devices > 0 && c.Devices[0].Driver != "580.95.05") {
				t.Fatalf("devices %+v (nvidia-smi still names the GPUs)", c.Devices)
			}
			if c.ArchList == nil || c.Devices == nil {
				t.Fatal("arch_list and devices must be [] not null")
			}
		})
	}
}

func TestCudaTimeout(t *testing.T) {
	fakeCUDA(t, smi4090, func(ctx context.Context) ([]byte, []byte, error) {
		<-ctx.Done() // torch import hangs on a cold disk
		return nil, nil, ctx.Err()
	})
	cudaTimeout = 20 * time.Millisecond
	c := runCUDACheck(t, false)
	if c.OK || cudaFailure(c) != cudaTimedOut || !strings.Contains(*c.Error, "did not finish") {
		t.Fatalf("%+v", c)
	}
	if cudaState.cached != nil {
		t.Fatal("a failure was cached")
	}
}

// Callers that arrive while a check runs share it; a success is cached until
// refresh, a failure is not.
func TestCudaOneRunAtATimeAndCached(t *testing.T) {
	release := make(chan struct{})
	ok := pyAnswer(nil)
	runs := fakeCUDA(t, smi4090, func(ctx context.Context) ([]byte, []byte, error) {
		<-release
		return ok(ctx)
	})
	var wg sync.WaitGroup
	results := make([]cudaCheck, 4)
	for i := range results {
		wg.Add(1)
		go func() { defer wg.Done(); results[i], _ = cudaResult(context.Background(), i == 3) }()
	}
	deadline := time.Now().Add(5 * time.Second)
	for runs.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	time.Sleep(20 * time.Millisecond) // let the others join
	close(release)
	wg.Wait()
	if n := runs.Load(); n != 1 {
		t.Fatalf("%d runs for 4 concurrent callers", n)
	}
	for _, r := range results {
		if !r.OK || r.CheckedAt != results[0].CheckedAt {
			t.Fatalf("callers got different results: %+v", results)
		}
	}
	runCUDACheck(t, false)
	if n := runs.Load(); n != 1 {
		t.Fatalf("a cached success ran again (%d runs)", n)
	}
	runCUDACheck(t, true)
	if n := runs.Load(); n != 2 {
		t.Fatalf("refresh did not run the check (%d runs)", n)
	}

	// a caller that leaves gets ctx's error; the check still finishes for the next one
	fail := pyAnswer(map[string]any{"stage": "kernel", "error": "RuntimeError: CUDA error: no kernel image is available"})
	gate := make(chan struct{})
	runs = fakeCUDA(t, smi4090, func(ctx context.Context) ([]byte, []byte, error) { <-gate; return fail(ctx) })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudaResult(ctx, false); err == nil {
		t.Fatal("a cancelled caller got a result")
	}
	close(gate)
	if c := runCUDACheck(t, false); c.OK || runs.Load() != 1 {
		t.Fatalf("joined run: %+v after %d runs", c, runs.Load())
	}
	if c := runCUDACheck(t, false); c.OK || runs.Load() != 2 {
		t.Fatalf("a failure was cached (%d runs)", runs.Load())
	}
}

// A killed check whose grandchild still holds stdout returns soon after its
// deadline (cmd.WaitDelay), so it frees cudaState.run for the next caller.
func TestCudaExecReturnsWhenAGrandchildHoldsThePipes(t *testing.T) {
	saved := cudaWaitDelay
	cudaWaitDelay = 200 * time.Millisecond
	t.Cleanup(func() { cudaWaitDelay = saved })
	t.Setenv("CUDA_HELPER", "child")
	// long enough for the child to start the grandchild before the kill
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	start := time.Now()
	out, _, err := execCUDA(ctx, os.Args[0], "-test.run=^TestCudaHelperProcess$")
	if d := time.Since(start); d > 4*time.Second {
		t.Fatalf("execCUDA took %s for a 1.5s deadline", d)
	}
	_, pid, found := strings.Cut(string(out), "grandchild started ")
	if n, perr := strconv.Atoi(strings.TrimSpace(pid)); found && perr == nil {
		// the grandchild would keep the test binary locked on Windows
		if p, ferr := os.FindProcess(n); ferr == nil {
			_ = p.Kill()
		}
	}
	if err == nil || !found {
		t.Fatalf("err %v, out %q: the child must start a grandchild and be killed", err, out)
	}
}

// TestCudaHelperProcess is the child of the test above: it starts a
// grandchild that shares its stdout, then hangs until it is killed.
func TestCudaHelperProcess(t *testing.T) {
	switch os.Getenv("CUDA_HELPER") {
	case "child":
		gc := exec.Command(os.Args[0], "-test.run=^TestCudaHelperProcess$")
		gc.Env = append(os.Environ(), "CUDA_HELPER=grandchild")
		gc.Stdout, gc.Stderr = os.Stdout, os.Stderr
		if err := gc.Start(); err != nil {
			os.Exit(2)
		}
		fmt.Fprintf(os.Stdout, "grandchild started %d\n", gc.Process.Pid)
		time.Sleep(20 * time.Second)
		os.Exit(0)
	case "grandchild":
		time.Sleep(5 * time.Second)
		os.Exit(0)
	}
}

func TestCudaParseDevices(t *testing.T) {
	got := parseCUDADevices("0, NVIDIA H100 80GB HBM3, 9.0, 580.95.05\n1, NVIDIA H100 80GB HBM3, 9.0, 580.95.05\n\n2, Tesla V100, [N/A], 535.54.03\nnot a row\n")
	want := []cudaDevice{{0, "NVIDIA H100 80GB HBM3", "9.0", "580.95.05"}, {1, "NVIDIA H100 80GB HBM3", "9.0", "580.95.05"}, {2, "Tesla V100", "", "535.54.03"}}
	if !slices.Equal(got, want) {
		t.Fatalf("%+v", got)
	}
}

// The check's device query is its own nvidia-smi call: the shared query that
// mirrors upstream's /api/gpu does not change.
func TestCudaLeavesNvQueryArgs(t *testing.T) {
	want := []string{
		"--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed",
		"--format=csv,noheader,nounits",
	}
	if !slices.Equal(sysstats.NvQueryArgs, want) {
		t.Fatalf("sysstats.NvQueryArgs changed: %q", sysstats.NvQueryArgs)
	}
	if slices.Equal(cudaQueryArgs, sysstats.NvQueryArgs) {
		t.Fatal("the CUDA check must query compute_cap in its own call")
	}
}

func TestCudaRoute(t *testing.T) {
	fakeCUDA(t, smi4090, pyAnswer(nil))
	e := setup(t)
	r := e.call(t, "GET", "/v1/system/cuda?refresh=1", nil)
	want(t, r, 200, "")
	if m := r.json(t); m["ok"] != true || m["failure"] != nil || len(m["devices"].([]any)) != 1 {
		t.Fatalf("%v", m)
	}
}
