package sysstats

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// GPUInfo mirrors GpuInfo in ui/src/types.ts. Name and DriverVersion are
// pointers because a short nvidia-smi line leaves them undefined upstream,
// which drops the key from the JSON.
type GPUInfo struct {
	Index         Num     `json:"index"`
	Name          *string `json:"name,omitempty"`
	DriverVersion *string `json:"driverVersion,omitempty"`
	Temperature   Num     `json:"temperature"`
	Utilization   struct {
		GPU    Num `json:"gpu"`
		Memory Num `json:"memory"`
	} `json:"utilization"`
	Memory GPUMemory `json:"memory"`
	Power  struct {
		Draw  Num `json:"draw"`
		Limit Num `json:"limit"`
	} `json:"power"`
	Clocks struct {
		Graphics Num `json:"graphics"`
		Memory   Num `json:"memory"`
	} `json:"clocks"`
	Fan struct {
		Speed Num `json:"speed"`
	} `json:"fan"`
}

type GPUMemory struct {
	Total  Num  `json:"total"`
	Free   Num  `json:"free"`
	Used   Num  `json:"used"`
	Shared bool `json:"shared,omitempty"`
}

func (g GPUInfo) name() string {
	if g.Name == nil {
		return ""
	}
	return *g.Name
}

// GPUResponse mirrors GPUApiResponse. IsMac is omitted where upstream's
// object literal has no isMac key (the /api/gpu NVIDIA success case).
type GPUResponse struct {
	HasNvidiaSmi bool      `json:"hasNvidiaSmi"`
	IsMac        *bool     `json:"isMac,omitempty"`
	GPUs         []GPUInfo `json:"gpus"`
	Error        string    `json:"error,omitempty"`
}

func boolPtr(b bool) *bool { return &b }

func strPtr(s string) *string { return &s }

// NvQueryArgs is the nvidia-smi query both /api/gpu and the monitor run.
var NvQueryArgs = []string{
	"--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed",
	"--format=csv,noheader,nounits",
}

// Power limits for GPUs whose driver reports none (nvidia-smi gives [N/A]).
// GB10 (DGX Spark) is a unified SoC with no settable GPU limit; 140 W is
// NVIDIA's rated TDP for the GB10 superchip.
var knownPowerLimitsW = map[string]float64{"NVIDIA GB10": 140}

func splitGPULine(line string) func(i int) (string, bool) {
	parts := strings.Split(line, ", ")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return func(i int) (string, bool) {
		if i < len(parts) {
			return parts[i], true
		}
		return "", false
	}
}

func fillGPU(field func(int) (string, bool)) GPUInfo {
	num := func(i int, parse func(string) Num) Num {
		s, ok := field(i)
		if !ok {
			return NaN
		}
		return parse(s)
	}
	var g GPUInfo
	g.Index = num(0, ParseInt)
	if s, ok := field(1); ok {
		g.Name = strPtr(s)
	}
	if s, ok := field(2); ok {
		g.DriverVersion = strPtr(s)
	}
	g.Temperature = num(3, ParseInt)
	g.Utilization.GPU = num(4, ParseInt)
	g.Utilization.Memory = num(5, ParseInt)
	g.Memory.Total = num(6, ParseInt)
	g.Memory.Free = num(7, ParseInt)
	g.Memory.Used = num(8, ParseInt)
	g.Power.Draw = num(9, ParseFloat)
	g.Power.Limit = num(10, ParseFloat)
	g.Clocks.Graphics = num(11, ParseInt)
	g.Clocks.Memory = num(12, ParseInt)
	// Some GPUs might not report fan speed, default to 0
	g.Fan.Speed = orZero(num(13, ParseInt))
	return g
}

// ParseRouteGPUs parses nvidia-smi output like api/gpu/route.ts: every line
// becomes a GPU, unsorted, with no fallbacks.
func ParseRouteGPUs(stdout string) []GPUInfo {
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	gpus := make([]GPUInfo, 0, len(lines))
	for _, line := range lines {
		gpus = append(gpus, fillGPU(splitGPULine(line)))
	}
	return gpus
}

// ParseMonitorGPULine parses one line like monitor.ts parseGpuLine: lines
// whose index is not a number are dropped, and a missing power limit falls
// back to the known TDP for that GPU.
func ParseMonitorGPULine(line string) (GPUInfo, bool) {
	g := fillGPU(splitGPULine(line))
	if g.Index.IsNaN() {
		return g, false
	}
	if g.Power.Limit.IsNaN() {
		if w, ok := knownPowerLimitsW[g.name()]; ok {
			g.Power.Limit = Num(w)
		}
	}
	return g, true
}

func sortByIndex(gpus []GPUInfo) {
	sort.SliceStable(gpus, func(i, j int) bool { return gpus[i].Index < gpus[j].Index })
}

// WithSharedMemory reports the system RAM pool for unified-memory GPUs (GB10
// / DGX Spark), whose VRAM nvidia-smi gives as [N/A].
func WithSharedMemory(gpu GPUResponse, cpu *CPUInfo) GPUResponse {
	if cpu == nil {
		return gpu
	}
	any := false
	for _, g := range gpu.GPUs {
		if g.Memory.Total.IsNaN() {
			any = true
		}
	}
	if !any {
		return gpu
	}
	total := Round(float64(cpu.TotalMemory))
	free := Round(float64(cpu.AvailableMemory))
	out := gpu
	out.GPUs = make([]GPUInfo, len(gpu.GPUs))
	for i, g := range gpu.GPUs {
		if g.Memory.Total.IsNaN() {
			g.Memory = GPUMemory{Total: Num(total), Free: Num(free), Used: Num(total - free), Shared: true}
		}
		out.GPUs[i] = g
	}
	return out
}

func nvEnv() []string { return append(os.Environ(), "CUDA_DEVICE_ORDER=PCI_BUS_ID") }

// HasNvidiaSmi mirrors checkNvidiaSmi: `nvidia-smi -L` on Windows, `which
// nvidia-smi` elsewhere.
func HasNvidiaSmi(ctx context.Context) bool {
	if runtime.GOOS == "windows" {
		cmd := exec.CommandContext(ctx, "nvidia-smi", "-L")
		hideWindow(cmd)
		return cmd.Run() == nil
	}
	_, err := exec.LookPath("nvidia-smi")
	return err == nil
}

// QueryNvidiaSmi runs the one-shot GPU query. Errors read like Node's exec
// ("Command failed: <cmd>\n<stderr>").
func QueryNvidiaSmi(ctx context.Context) (string, error) {
	cmd := exec.CommandContext(ctx, "nvidia-smi", NvQueryArgs...)
	cmd.Env = nvEnv()
	hideWindow(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return stdout.String(), fmt.Errorf("Command failed: nvidia-smi %s\n%s", strings.Join(NvQueryArgs, " "), stderr.String())
	}
	return stdout.String(), nil
}

var (
	chipsetRe = regexp.MustCompile(`Chipset Model:\s*(.+)`)
	coresRe   = regexp.MustCompile(`Total Number of Cores:\s*(\d+)`)
)

// MacGPUName reads the GPU name from system_profiler ("Apple GPU" fallback).
func MacGPUName(ctx context.Context) string {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "sh", "-c",
		`system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"`).Output()
	name := "Apple GPU"
	if err != nil {
		return name
	}
	if m := chipsetRe.FindStringSubmatch(string(out)); m != nil {
		name = strings.TrimSpace(m[1])
		if c := coresRe.FindStringSubmatch(string(out)); c != nil {
			name += fmt.Sprintf(" GPU (%s cores)", c[1])
		}
	}
	return name
}

// MacGPU is the Apple GPU entry. Upstream fills temperature, load, fan and
// power from the native macstats module; there is no Go equivalent, so they
// read 0 exactly as upstream does when macstats is not installed.
func MacGPU(name string) GPUInfo {
	memTotal := float64(TotalMemoryBytes()) / (1024 * 1024)
	memUsed := 0.0
	var g GPUInfo
	g.Index = 0
	g.Name = strPtr(name)
	g.DriverVersion = strPtr("macOS")
	g.Temperature = 0
	g.Utilization.GPU = 0
	if memTotal > 0 {
		g.Utilization.Memory = Num(Round(memUsed / memTotal * 100))
	}
	g.Memory = GPUMemory{Total: Num(Round(memTotal)), Free: Num(Round(memTotal - memUsed)), Used: Num(Round(memUsed))}
	g.Power.Draw, g.Power.Limit = 0, 0
	g.Clocks.Graphics, g.Clocks.Memory = 0, 0
	g.Fan.Speed = 0
	return g
}

// RouteGPUInfo is api/gpu/route.ts getGpuInfo().
func RouteGPUInfo(ctx context.Context) (GPUResponse, error) {
	if runtime.GOOS == "darwin" {
		return GPUResponse{HasNvidiaSmi: false, IsMac: boolPtr(true), GPUs: []GPUInfo{MacGPU(MacGPUName(ctx))}}, nil
	}
	if !HasNvidiaSmi(ctx) {
		return GPUResponse{HasNvidiaSmi: false, IsMac: boolPtr(false), GPUs: []GPUInfo{},
			Error: "nvidia-smi not found or not accessible"}, nil
	}
	out, err := QueryNvidiaSmi(ctx)
	if err != nil {
		return GPUResponse{}, err
	}
	return GPUResponse{HasNvidiaSmi: true, GPUs: ParseRouteGPUs(out)}, nil
}
