//go:build windows

package sysstats

import (
	"os/exec"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

var (
	kernel32                 = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")
	procGetSystemTimes       = kernel32.NewProc("GetSystemTimes")
)

func cpuModelName() string {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `HARDWARE\DESCRIPTION\System\CentralProcessor\0`, registry.QUERY_VALUE)
	if err != nil {
		return ""
	}
	defer k.Close()
	v, _, err := k.GetStringValue("ProcessorNameString")
	if err != nil {
		return ""
	}
	return v
}

func logicalCores() int {
	if n := windows.GetActiveProcessorCount(windows.ALL_PROCESSOR_GROUPS); n > 0 {
		return int(n)
	}
	return runtime.NumCPU()
}

func filetime(ft windows.Filetime) float64 {
	return float64(uint64(ft.HighDateTime)<<32 | uint64(ft.LowDateTime))
}

// cpuTimes uses GetSystemTimes; kernel time includes idle, as libuv's
// per-CPU sys = kernel - idle.
func cpuTimes() (busy, total float64, ok bool) {
	var idle, kernel, user windows.Filetime
	r, _, _ := procGetSystemTimes.Call(uintptr(unsafe.Pointer(&idle)), uintptr(unsafe.Pointer(&kernel)), uintptr(unsafe.Pointer(&user)))
	if r == 0 {
		return 0, 0, false
	}
	i, k, u := filetime(idle), filetime(kernel), filetime(user)
	total = u + k
	return total - i, total, true
}

func loadFallback() float64 { return 0 }

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

func globalMemory() (memoryStatusEx, error) {
	var m memoryStatusEx
	m.Length = uint32(unsafe.Sizeof(m))
	r, _, err := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&m)))
	if r == 0 {
		return m, err
	}
	return m, nil
}

// ReadMemory: free and available are both ullAvailPhys (os.freemem()), which
// is also what systeminformation reports on Windows.
func ReadMemory() (MemoryStats, error) {
	m, err := globalMemory()
	if err != nil {
		return MemoryStats{}, err
	}
	return MemoryStats{Total: float64(m.TotalPhys), Free: float64(m.AvailPhys), Available: float64(m.AvailPhys)}, nil
}

func TotalMemoryBytes() uint64 {
	m, err := globalMemory()
	if err != nil {
		return 0
	}
	return m.TotalPhys
}

// ReadCPUTemperature: Windows has no file-based sensor. Upstream asks WMI
// (MSAcpi_ThermalZoneTemperature) through systeminformation, which needs
// admin rights and reports nothing otherwise; not ported, reads as unknown.
func ReadCPUTemperature() (float64, bool) { return 0, false }

func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
}
