package sysstats

import (
	"regexp"
	"strings"
	"sync"
)

// CPUInfo mirrors CpuInfo in ui/src/types.ts (memory in MiB). Field order
// matches the object literal in api/cpu/route.ts.
type CPUInfo struct {
	Name            string `json:"name"`
	Cores           int    `json:"cores"`
	Temperature     Num    `json:"temperature"`
	TotalMemory     Num    `json:"totalMemory"`
	AvailableMemory Num    `json:"availableMemory"`
	FreeMemory      Num    `json:"freeMemory"`
	CurrentLoad     Num    `json:"currentLoad"`
}

// MemoryStats is in bytes. Available is what the OS would hand to a new
// allocation without swapping; Free is untouched pages.
type MemoryStats struct {
	Total, Free, Available float64
}

var (
	registeredRe = regexp.MustCompile(`\(R\)+`)
	trademarkRe  = regexp.MustCompile(`\(TM\)+`)
	copyrightRe  = regexp.MustCompile(`\(C\)+`)
	cpuWordRe    = regexp.MustCompile(`CPU+`)
	spacesRe     = regexp.MustCompile(`\s+`)
)

// cpuDisplayName is `${manufacturer} ${brand}` from systeminformation's
// si.cpu(): the model string up to "@", with (R)/(TM)/(C) turned into
// symbols and "CPU" dropped; manufacturer is its first word.
func cpuDisplayName(model string) string {
	b := strings.TrimSpace(strings.SplitN(model, "@", 2)[0])
	clean := func(re *regexp.Regexp, with string) {
		b = strings.TrimSpace(spacesRe.ReplaceAllString(re.ReplaceAllString(b, with), " "))
	}
	clean(registeredRe, "®")
	clean(trademarkRe, "™")
	clean(copyrightRe, "©")
	clean(cpuWordRe, "")
	parts := strings.Split(b, " ")
	return parts[0] + " " + strings.Join(parts[1:], " ")
}

var (
	staticOnce  sync.Once
	staticName  string
	staticCores int
)

// StaticCPU is the CPU name and logical core count, resolved once: they never
// change, and on some platforms reading them is slow.
func StaticCPU() (string, int) {
	staticOnce.Do(func() {
		staticName = cpuDisplayName(cpuModelName())
		staticCores = logicalCores()
	})
	return staticName, staticCores
}

// LoadSampler yields the CPU load (0-100) accumulated since its previous
// call; the first call reports the since-boot average. Mirrors
// createLoadSampler (and systeminformation's currentLoad, which works the
// same way).
type LoadSampler struct {
	mu        sync.Mutex
	have      bool
	lastBusy  float64
	lastTotal float64
	lastLoad  float64
}

func (s *LoadSampler) Sample() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	busy, total, ok := cpuTimes()
	if !ok {
		s.lastLoad = loadFallback()
		return s.lastLoad
	}
	if s.have {
		dTotal, dBusy := total-s.lastTotal, busy-s.lastBusy
		// dTotal can be 0 if called twice within a jiffy; keep the last value.
		if dTotal > 0 {
			s.lastLoad = min(100, max(0, dBusy/dTotal*100))
		}
	} else if total > 0 {
		s.lastLoad = busy / total * 100
	}
	s.have, s.lastBusy, s.lastTotal = true, busy, total
	return s.lastLoad
}

// BuildCPUInfo assembles a CpuInfo from the shared readers.
func BuildCPUInfo(load float64, temperature float64) (CPUInfo, error) {
	name, cores := StaticCPU()
	mem, err := ReadMemory()
	if err != nil {
		return CPUInfo{}, err
	}
	return CPUInfo{
		Name:            name,
		Cores:           cores,
		Temperature:     Num(temperature),
		TotalMemory:     Num(mem.Total / (1024 * 1024)),
		AvailableMemory: Num(mem.Available / (1024 * 1024)),
		FreeMemory:      Num(mem.Free / (1024 * 1024)),
		CurrentLoad:     Num(load),
	}, nil
}
