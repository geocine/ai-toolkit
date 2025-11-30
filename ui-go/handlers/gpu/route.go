package gpu

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"net/http"
	"os/exec"
	"strconv"
)

func RouteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed", "--format=csv,noheader,nounits")
	output, err := cmd.CombinedOutput()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"hasNvidiaSmi": false,
			"gpus":         []interface{}{},
			"error":        "nvidia-smi not found or not accessible",
		})
		return
	}
	reader := csv.NewReader(bytes.NewReader(output))
	reader.TrimLeadingSpace = true
	lines, err := reader.ReadAll()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"hasNvidiaSmi": false,
			"gpus":         []interface{}{},
			"error":        "Failed to parse nvidia-smi output",
		})
		return
	}
	gpus := make([]map[string]interface{}, 0, len(lines))
	for _, fields := range lines {
		if len(fields) < 14 {
			continue
		}
		index, _ := strconv.Atoi(fields[0])
		temperature, _ := strconv.Atoi(fields[3])
		gpuUtil, _ := strconv.Atoi(fields[4])
		memUtil, _ := strconv.Atoi(fields[5])
		memTotal, _ := strconv.Atoi(fields[6])
		memFree, _ := strconv.Atoi(fields[7])
		memUsed, _ := strconv.Atoi(fields[8])
		powerDraw, _ := strconv.ParseFloat(fields[9], 64)
		powerLimit, _ := strconv.ParseFloat(fields[10], 64)
		clockGraphics, _ := strconv.Atoi(fields[11])
		clockMemory, _ := strconv.Atoi(fields[12])
		fanSpeed, _ := strconv.Atoi(fields[13])
		gpu := map[string]interface{}{
			"index":         index,
			"name":          fields[1],
			"driverVersion": fields[2],
			"temperature":   temperature,
			"utilization": map[string]interface{}{
				"gpu":    gpuUtil,
				"memory": memUtil,
			},
			"memory": map[string]interface{}{
				"total": memTotal,
				"free":  memFree,
				"used":  memUsed,
			},
			"power": map[string]interface{}{
				"draw":  powerDraw,
				"limit": powerLimit,
			},
			"clocks": map[string]interface{}{
				"graphics": clockGraphics,
				"memory":   clockMemory,
			},
			"fan": map[string]interface{}{
				"speed": fanSpeed,
			},
		}
		gpus = append(gpus, gpu)
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"hasNvidiaSmi": true,
		"gpus":         gpus,
	})
}
