//go:build darwin

package sysstats

import (
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

func cpuModelName() string {
	v, _ := unix.Sysctl("machdep.cpu.brand_string")
	return v
}

func logicalCores() int {
	if n, err := unix.SysctlUint32("hw.logicalcpu"); err == nil && n > 0 {
		return int(n)
	}
	return runtime.NumCPU()
}

// cpuTimes: per-CPU ticks need mach host_processor_info (cgo), so macOS uses
// the load-average fallback below.
func cpuTimes() (busy, total float64, ok bool) { return 0, 0, false }

// loadFallback approximates CPU load from the 1-minute load average.
func loadFallback() float64 {
	raw, err := unix.SysctlRaw("vm.loadavg")
	if err != nil || len(raw) < 16 {
		return 0
	}
	// struct loadavg { fixpt_t ldavg[3]; long fscale; }
	ld := float64(uint32(raw[0]) | uint32(raw[1])<<8 | uint32(raw[2])<<16 | uint32(raw[3])<<24)
	scaleOff := 16
	if len(raw) < scaleOff+8 {
		return 0
	}
	var fscale float64
	for i := 7; i >= 0; i-- {
		fscale = fscale*256 + float64(raw[scaleOff+i])
	}
	if fscale == 0 {
		return 0
	}
	_, cores := StaticCPU()
	if cores == 0 {
		return 0
	}
	return min(100, ld/fscale/float64(cores)*100)
}

func TotalMemoryBytes() uint64 {
	v, _ := unix.SysctlUint64("hw.memsize")
	return v
}

var vmStatRe = regexp.MustCompile(`(?m)^(Pages [a-z ]+):\s+(\d+)\.`)

// ReadMemory uses vm_stat like systeminformation (macOS process creation is
// posix_spawn, so this is cheap): available counts inactive and speculative
// pages, which the OS reclaims on demand.
func ReadMemory() (MemoryStats, error) {
	total := float64(TotalMemoryBytes())
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return MemoryStats{Total: total}, nil
	}
	pageSize := 4096.0
	if m := regexp.MustCompile(`page size of (\d+) bytes`).FindStringSubmatch(string(out)); m != nil {
		pageSize, _ = strconv.ParseFloat(m[1], 64)
	}
	pages := map[string]float64{}
	for _, m := range vmStatRe.FindAllStringSubmatch(string(out), -1) {
		v, _ := strconv.ParseFloat(m[2], 64)
		pages[strings.TrimSpace(m[1])] = v
	}
	free := pages["Pages free"] * pageSize
	available := (pages["Pages free"] + pages["Pages inactive"] + pages["Pages speculative"]) * pageSize
	return MemoryStats{Total: total, Free: free, Available: available}, nil
}

// ReadCPUTemperature: upstream reads SMC sensors through a native module;
// not ported, reads as unknown.
func ReadCPUTemperature() (float64, bool) { return 0, false }

func hideWindow(*exec.Cmd) {}
