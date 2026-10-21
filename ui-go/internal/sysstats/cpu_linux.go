//go:build linux

package sysstats

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

func cpuModelName() string {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	defer f.Close()
	fallback := ""
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(k) {
		case "model name":
			return strings.TrimSpace(v)
		case "Model", "Hardware":
			if fallback == "" {
				fallback = strings.TrimSpace(v)
			}
		}
	}
	return fallback
}

// cpuLines are the per-CPU lines of /proc/stat (what os.cpus() reads).
func cpuLines() [][]string {
	raw, err := os.ReadFile("/proc/stat")
	if err != nil {
		return nil
	}
	var out [][]string
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) > 1 && strings.HasPrefix(fields[0], "cpu") && fields[0] != "cpu" {
			out = append(out, fields)
		}
	}
	return out
}

func logicalCores() int {
	if n := len(cpuLines()); n > 0 {
		return n
	}
	return runtime.NumCPU()
}

// cpuTimes sums libuv's per-CPU times (user, nice, sys, idle, irq; iowait is
// not counted) like os.cpus().
func cpuTimes() (busy, total float64, ok bool) {
	lines := cpuLines()
	if len(lines) == 0 {
		return 0, 0, false
	}
	for _, f := range lines {
		val := func(i int) float64 {
			if i >= len(f) {
				return 0
			}
			v, _ := strconv.ParseFloat(f[i], 64)
			return v
		}
		user, nice, sys, idle, irq := val(1), val(2), val(3), val(4), val(6)
		all := user + nice + sys + idle + irq
		busy += all - idle
		total += all
	}
	return busy, total, true
}

func loadFallback() float64 { return 0 }

func memValue(lines []string, key string) float64 {
	for _, l := range lines {
		if strings.HasPrefix(l, key+":") {
			v, err := strconv.ParseInt(strings.Fields(strings.TrimSpace(l[len(key)+1:]) + " x")[0], 10, 64)
			if err != nil {
				return math.NaN()
			}
			return float64(v) * 1024 // /proc/meminfo is in kB
		}
	}
	return math.NaN()
}

// ReadMemory reads /proc/meminfo for both MemFree and MemAvailable.
func ReadMemory() (MemoryStats, error) {
	if raw, err := os.ReadFile("/proc/meminfo"); err == nil {
		lines := strings.Split(string(raw), "\n")
		total, free, available := memValue(lines, "MemTotal"), memValue(lines, "MemFree"), memValue(lines, "MemAvailable")
		if math.IsNaN(available) {
			// Pre-3.14 kernels: same estimate systeminformation uses
			buffers, cached := memValue(lines, "Buffers"), memValue(lines, "Cached")
			if math.IsNaN(buffers) {
				buffers = 0
			}
			if math.IsNaN(cached) {
				cached = 0
			}
			available = free + buffers + cached
		}
		if !math.IsNaN(total) && !math.IsNaN(free) {
			if math.IsNaN(available) {
				available = free
			}
			return MemoryStats{Total: total, Free: free, Available: available}, nil
		}
	}
	return MemoryStats{}, os.ErrNotExist
}

// TotalMemoryBytes is os.totalmem().
func TotalMemoryBytes() uint64 {
	m, err := ReadMemory()
	if err != nil {
		return 0
	}
	return uint64(m.Total)
}

// ---------------------------------------------------------------------------
// Temperature: sysfs, resolved once (see cpuStats.ts scanLinuxTempSource)
// ---------------------------------------------------------------------------

type tempSource struct {
	files []string
	mean  bool
}

var (
	tempMu      sync.Mutex
	tempScanned bool
	tempSrc     *tempSource
	labelFileRe = regexp.MustCompile(`^temp\d+_label$`)
)

func millideg(raw string) (float64, bool) {
	v, err := strconv.ParseInt(strings.Fields(strings.TrimSpace(raw) + " x")[0], 10, 64)
	if err != nil {
		return 0, false
	}
	return Round(float64(v)/100) / 10, true
}

func scanLinuxTempSource() *tempSource {
	type labeled struct{ label, file string }
	var all []labeled
	hwmonRoot := "/sys/class/hwmon"
	monitors, _ := os.ReadDir(hwmonRoot)
	for _, mon := range monitors {
		dir := filepath.Join(hwmonRoot, mon.Name())
		files, _ := os.ReadDir(dir)
		for _, f := range files {
			if !labelFileRe.MatchString(f.Name()) {
				continue
			}
			raw, _ := os.ReadFile(filepath.Join(dir, f.Name()))
			label := strings.TrimSpace(string(raw))
			if label == "" {
				continue
			}
			all = append(all, labeled{label, filepath.Join(dir, strings.TrimSuffix(f.Name(), "_label")+"_input")})
		}
	}
	by := func(pred func(string) bool) []labeled {
		var out []labeled
		for _, e := range all {
			if pred(strings.ToLower(e.label)) {
				out = append(out, e)
			}
		}
		return out
	}
	if m := by(func(l string) bool { return l == "tctl" }); len(m) > 0 {
		return &tempSource{files: []string{m[0].file}}
	}
	if m := by(func(l string) bool { return l == "tdie" }); len(m) > 0 {
		return &tempSource{files: []string{m[0].file}}
	}
	if m := by(func(l string) bool {
		return strings.Contains(l, "package") || strings.Contains(l, "physical") || l == "tccd1"
	}); len(m) > 0 {
		return &tempSource{files: []string{m[0].file}}
	}
	if m := by(func(l string) bool { return strings.HasPrefix(l, "core") }); len(m) > 0 {
		src := &tempSource{mean: true}
		for _, c := range m {
			src.files = append(src.files, c.file)
		}
		return src
	}
	// No labeled hwmon sensor: try the generic thermal zones.
	thermalRoot := "/sys/class/thermal"
	zones, _ := os.ReadDir(thermalRoot)
	for _, z := range zones {
		if !strings.HasPrefix(z.Name(), "thermal_zone") {
			continue
		}
		raw, _ := os.ReadFile(filepath.Join(thermalRoot, z.Name(), "type"))
		t := strings.ToLower(strings.TrimSpace(string(raw)))
		if t == "x86_pkg_temp" || strings.Contains(t, "cpu") || strings.Contains(t, "soc") {
			return &tempSource{files: []string{filepath.Join(thermalRoot, z.Name(), "temp")}}
		}
	}
	return nil
}

// ReadCPUTemperature returns °C, or false when unknown. Upstream falls back
// to `sensors` when sysfs has nothing; that shell-out is not ported.
func ReadCPUTemperature() (float64, bool) {
	tempMu.Lock()
	defer tempMu.Unlock()
	if !tempScanned {
		tempSrc, tempScanned = scanLinuxTempSource(), true
	}
	if tempSrc == nil {
		return 0, false
	}
	var values []float64
	for _, f := range tempSrc.files {
		raw, err := os.ReadFile(f)
		if err != nil {
			// sensor went away (module unloaded / renumbered): rescan next time
			tempScanned = false
			return 0, false
		}
		if v, ok := millideg(string(raw)); ok {
			values = append(values, v)
		}
	}
	if len(values) == 0 {
		return 0, false
	}
	if tempSrc.mean {
		sum := 0.0
		for _, v := range values {
			sum += v
		}
		return Round(sum / float64(len(values))), true
	}
	return values[0], true
}
