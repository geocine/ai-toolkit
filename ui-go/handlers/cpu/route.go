package cpu

import (
	"encoding/json"
	"net/http"
	"runtime"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/mem"
)

type CpuInfo struct {
	Name            string  `json:"name"`
	Cores           int     `json:"cores"`
	Temperature     float64 `json:"temperature"`
	TotalMemory     float64 `json:"totalMemory"`
	FreeMemory      float64 `json:"freeMemory"`
	AvailableMemory float64 `json:"availableMemory"`
	CurrentLoad     float64 `json:"currentLoad"`
}

// RouteHandler handles GET /api/cpu
func RouteHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "Method not allowed"})
		return
	}

	// Get CPU info
	cpuInfoRaw, err := cpu.Info()
	cpuName := "Unknown CPU"
	if err == nil && len(cpuInfoRaw) > 0 {
		cpuName = cpuInfoRaw[0].ModelName
	}

	// Get number of cores
	cores := runtime.NumCPU()

	// Get CPU temperature (may not work on all systems)
	temperature := 0.0
	temps, err := host.SensorsTemperatures()
	if err == nil {
		for _, temp := range temps {
			// Look for CPU temperature sensor
			if temp.Temperature > 0 {
				temperature = temp.Temperature
				break
			}
		}
	}

	// Get memory info
	memInfo, err := mem.VirtualMemory()
	totalMemory := 0.0
	freeMemory := 0.0
	availableMemory := 0.0
	if err == nil {
		totalMemory = float64(memInfo.Total) / (1024 * 1024)     // Convert to MB
		freeMemory = float64(memInfo.Free) / (1024 * 1024)       // Convert to MB
		availableMemory = float64(memInfo.Available) / (1024 * 1024) // Convert to MB
	}

	// Get CPU load
	currentLoad := 0.0
	loadPercent, err := cpu.Percent(0, false)
	if err == nil && len(loadPercent) > 0 {
		currentLoad = loadPercent[0]
	}

	cpuInfo := CpuInfo{
		Name:            cpuName,
		Cores:           cores,
		Temperature:     temperature,
		TotalMemory:     totalMemory,
		FreeMemory:      freeMemory,
		AvailableMemory: availableMemory,
		CurrentLoad:     currentLoad,
	}

	json.NewEncoder(w).Encode(cpuInfo)
}

