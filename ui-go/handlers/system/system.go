// Package system ports the machine-level routes: api/cpu, api/gpu,
// api/monitor (SSE), api/model_archs, api/ostris_cloud and api/scripts.
package system

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/geocine/aitoolkit/internal/apicache"
	"github.com/geocine/aitoolkit/internal/extui"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/sysstats"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /api/cpu", cpu)
	httpx.Handle(mux, "GET /api/gpu", gpu)
	httpx.Handle(mux, "GET /api/monitor", monitor)
	httpx.Handle(mux, "GET /api/model_archs", modelArchs)
	httpx.Handle(mux, "GET /api/ostris_cloud", ostrisCloud)
	httpx.Handle(mux, "POST /api/scripts", scripts)
}

// /api/cpu keeps its own load baseline, like systeminformation's
// module-level currentLoad state.
var cpuLoad sysstats.LoadSampler

func cpu(w http.ResponseWriter, r *http.Request) {
	info, err := apicache.Cached("cpu-info", apicache.DefaultStale, nil, func() (sysstats.CPUInfo, error) {
		temp, _ := sysstats.ReadCPUTemperature() // `.main || 0`
		return sysstats.BuildCPUInfo(cpuLoad.Sample(), temp)
	})
	if err != nil {
		log.Printf("Error fetching CPU stats: %v", err)
		httpx.Error(w, http.StatusInternalServerError, "Failed to fetch CPU stats: "+err.Error())
		return
	}
	httpx.OK(w, info)
}

func gpu(w http.ResponseWriter, r *http.Request) {
	info, err := apicache.Cached("gpu-info", apicache.DefaultStale, nil, func() (sysstats.GPUResponse, error) {
		return sysstats.RouteGPUInfo(r.Context())
	})
	if err != nil {
		log.Printf("Error fetching NVIDIA GPU stats: %v", err)
		f := false
		httpx.JSON(w, http.StatusInternalServerError, sysstats.GPUResponse{
			HasNvidiaSmi: false, IsMac: &f, GPUs: []sysstats.GPUInfo{},
			Error: "Failed to fetch GPU stats: " + err.Error(),
		})
		return
	}
	httpx.OK(w, info)
}

// Dead subscribers whose queue backs up this far are dropped (upstream's
// desiredSize < -120 backstop).
const monitorBacklog = 120

// monitor streams system stats as SSE: an `init` event with the rolling
// history and the latest sample, then a `sample` event every tick.
func monitor(w http.ResponseWriter, r *http.Request) {
	mon := sysstats.StartMonitor()
	flusher, _ := w.(http.Flusher)
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("Connection", "keep-alive")
	// belt-and-braces against any buffering proxy in front of us
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	init, err := json.Marshal(mon.GetInit())
	if err != nil {
		return
	}
	if _, err := fmt.Fprintf(w, "event: init\ndata: %s\n\n", init); err != nil {
		return
	}
	if flusher != nil {
		flusher.Flush()
	}

	queue := make(chan []byte, monitorBacklog)
	var once sync.Once
	gone := make(chan struct{})
	drop := func() { once.Do(func() { close(gone) }) }
	unsubscribe := mon.Subscribe(func(serialized []byte) {
		select {
		case queue <- serialized:
		default:
			drop()
		}
	})
	defer unsubscribe()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-gone:
			return
		case s := <-queue:
			if _, err := fmt.Fprintf(w, "event: sample\ndata: %s\n\n", s); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
	}
}

func modelArchs(w http.ResponseWriter, r *http.Request) {
	result := extui.List(paths.ToolkitRoot)
	w.Header().Set("Cache-Control", "no-store")
	httpx.OK(w, result)
}

type cloudStatus struct {
	Enabled bool             `json:"enabled"`
	AppURL  string           `json:"appUrl,omitempty"`
	Balance *json.RawMessage `json:"balance,omitempty"`
	Error   string           `json:"error,omitempty"`
}

func ostrisCloud(w http.ResponseWriter, r *http.Request) {
	appURL := os.Getenv("OSTRIS_CLOUD_APP_URL")
	apiKey := os.Getenv("OSTRIS_CLOUD_API_KEY")
	if appURL == "" || apiKey == "" {
		httpx.OK(w, cloudStatus{Enabled: false})
		return
	}
	fail := func(err error) {
		httpx.OK(w, cloudStatus{Enabled: true, AppURL: appURL, Error: "Failed to fetch Ostris Cloud balance: " + err.Error()})
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, appURL+"/api/machine/me", nil)
	if err != nil {
		fail(err)
		return
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		fail(err)
		return
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode > 299 {
		httpx.OK(w, cloudStatus{Enabled: true, AppURL: appURL, Error: fmt.Sprintf("Ostris Cloud responded with %d", res.StatusCode)})
		return
	}
	body, err := io.ReadAll(res.Body)
	if err != nil {
		fail(err)
		return
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(body, &data); err != nil {
		// a JSON body that is not an object has no balance
		var anyJSON any
		if err2 := json.Unmarshal(body, &anyJSON); err2 != nil {
			fail(err)
			return
		}
	}
	balance := json.RawMessage("null")
	if b, ok := data["balance"]; ok && strings.TrimSpace(string(b)) != "" {
		balance = b // `data.balance ?? null`
	}
	httpx.OK(w, cloudStatus{Enabled: true, AppURL: appURL, Balance: &balance})
}
