package sysstats

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Mirrors ui/src/server/monitor.ts and ui/src/utils/monitorSample.ts: an
// always-on sampler (started by the first /api/monitor request) that keeps a
// 2-minute rolling history and pushes every sample to subscribers.

const (
	MonitorTick          = 500 * time.Millisecond
	MonitorHistoryLength = int(120 * time.Second / MonitorTick) // 2 minutes of samples

	// No stdout line for this long means the loop child is buffering or hung.
	nvWatchdog = 15 * time.Second
	// A line-less gap this long after the last line closes out a batch.
	nvBatchFlush = 100 * time.Millisecond
	// Temperature refresh is decoupled from the tick (see refreshCPUTemp)
	cpuTempRefresh = 5 * time.Second
	nvRestartDelay = 5 * time.Second
)

type HistoryPoint struct {
	T   int64 `json:"t"`
	CPU struct {
		Load      Num `json:"load"`
		MemUsedMb Num `json:"memUsedMb"`
	} `json:"cpu"`
	GPUs []GPUPoint `json:"gpus"`
}

type GPUPoint struct {
	Load      Num `json:"load"`
	MemUsedMb Num `json:"memUsedMb"`
}

type Sample struct {
	T   int64       `json:"t"`
	CPU *CPUInfo    `json:"cpu"`
	GPU GPUResponse `json:"gpu"`
}

type Init struct {
	T       int64          `json:"t"`
	CPU     *CPUInfo       `json:"cpu"`
	GPU     GPUResponse    `json:"gpu"`
	History []HistoryPoint `json:"history"`
}

func historyPointFromSample(s Sample) HistoryPoint {
	var p HistoryPoint
	p.T = s.T
	if s.CPU != nil {
		p.CPU.Load = s.CPU.CurrentLoad
		p.CPU.MemUsedMb = s.CPU.TotalMemory - s.CPU.AvailableMemory
	}
	p.GPUs = make([]GPUPoint, 0, len(s.GPU.GPUs))
	for _, g := range s.GPU.GPUs {
		p.GPUs = append(p.GPUs, GPUPoint{Load: g.Utilization.GPU, MemUsedMb: g.Memory.Used})
	}
	return p
}

// Subscriber receives each sample already serialized (once per tick, shared).
type Subscriber func(serialized []byte)

type Monitor struct {
	isMac bool

	mu          sync.Mutex
	history     []HistoryPoint
	subscribers map[int]Subscriber
	nextSubID   int
	latestCPU   *CPUInfo
	latestGPU   GPUResponse
	macGPUName  string

	nvChild        *exec.Cmd
	nvBatch        []GPUInfo
	nvFlushTimer   *time.Timer
	nvUnavailable  bool
	nvEverGotLine  bool
	nvOneShotMode  bool
	nvOneShotBusy  bool
	lastNvLineAt   time.Time
	lastCPUTemp    float64
	cpuTempBusy    bool
	lastCPUTempAt  time.Time
	loadSampler    LoadSampler
	tickInProgress bool
}

var (
	monitorOnce sync.Once
	monitor     *Monitor
)

// StartMonitor returns the process-wide monitor, starting it on first use.
// With AI_TOOLKIT_DISABLE_MONITOR=1 it exists but never samples.
func StartMonitor() *Monitor {
	monitorOnce.Do(func() {
		isMac := runtime.GOOS == "darwin"
		monitor = &Monitor{
			isMac:       isMac,
			subscribers: map[int]Subscriber{},
			latestGPU:   GPUResponse{HasNvidiaSmi: false, IsMac: boolPtr(isMac), GPUs: []GPUInfo{}},
			macGPUName:  "Apple GPU",
			history:     []HistoryPoint{},
		}
		if os.Getenv("AI_TOOLKIT_DISABLE_MONITOR") != "1" {
			monitor.start()
		}
	})
	return monitor
}

func (m *Monitor) start() {
	if m.isMac {
		go func() {
			name := MacGPUName(context.Background())
			m.mu.Lock()
			m.macGPUName = name
			m.mu.Unlock()
		}()
	} else {
		go m.startNvLoop(true)
	}
	go func() {
		// fixed cadence; a tick that overruns just skips beats
		m.tick()
		t := time.NewTicker(MonitorTick)
		defer t.Stop()
		for range t.C {
			m.tick()
		}
	}()
}

// Subscribe registers fn; the returned func unsubscribes.
func (m *Monitor) Subscribe(fn Subscriber) func() {
	m.mu.Lock()
	id := m.nextSubID
	m.nextSubID++
	m.subscribers[id] = fn
	m.mu.Unlock()
	return func() {
		m.mu.Lock()
		delete(m.subscribers, id)
		m.mu.Unlock()
	}
}

func (m *Monitor) GetInit() Init {
	m.mu.Lock()
	defer m.mu.Unlock()
	return Init{
		T:       time.Now().UnixMilli(),
		CPU:     m.latestCPU,
		GPU:     WithSharedMemory(m.latestGPU, m.latestCPU),
		History: append([]HistoryPoint{}, m.history...),
	}
}

func (m *Monitor) tick() {
	m.mu.Lock()
	if m.tickInProgress {
		m.mu.Unlock()
		return
	}
	m.tickInProgress = true
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		m.tickInProgress = false
		m.mu.Unlock()
	}()

	t := time.Now().UnixMilli()
	if cpu, err := m.sampleCPU(); err != nil {
		log.Printf("Monitor: CPU sample failed: %v", err)
	} else {
		m.mu.Lock()
		m.latestCPU = &cpu
		m.mu.Unlock()
	}
	if m.isMac {
		m.mu.Lock()
		m.latestGPU = GPUResponse{HasNvidiaSmi: false, IsMac: boolPtr(true), GPUs: []GPUInfo{MacGPU(m.macGPUName)}}
		m.mu.Unlock()
	} else {
		m.mu.Lock()
		oneShot := m.nvOneShotMode
		m.mu.Unlock()
		if oneShot {
			m.sampleNvOneShot()
		} else {
			m.nvWatchdogCheck()
		}
	}

	m.mu.Lock()
	sample := Sample{T: t, CPU: m.latestCPU, GPU: WithSharedMemory(m.latestGPU, m.latestCPU)}
	m.history = append(m.history, historyPointFromSample(sample))
	if len(m.history) > MonitorHistoryLength {
		m.history = append([]HistoryPoint{}, m.history[len(m.history)-MonitorHistoryLength:]...)
	}
	subs := make([]Subscriber, 0, len(m.subscribers))
	for _, s := range m.subscribers {
		subs = append(subs, s)
	}
	m.mu.Unlock()

	serialized, err := json.Marshal(sample)
	if err != nil {
		log.Printf("Monitor: encode sample: %v", err)
		return
	}
	for _, s := range subs {
		s(serialized)
	}
}

func (m *Monitor) sampleCPU() (CPUInfo, error) {
	// the temperature read can be slow, so it refreshes concurrently and the
	// tick uses the cached value (at most a tick or two old)
	m.refreshCPUTemp()
	m.mu.Lock()
	temp := m.lastCPUTemp
	m.mu.Unlock()
	return BuildCPUInfo(m.loadSampler.Sample(), temp)
}

func (m *Monitor) refreshCPUTemp() {
	m.mu.Lock()
	if m.cpuTempBusy || time.Since(m.lastCPUTempAt) < cpuTempRefresh {
		m.mu.Unlock()
		return
	}
	m.lastCPUTempAt = time.Now()
	m.cpuTempBusy = true
	m.mu.Unlock()
	go func() {
		t, ok := ReadCPUTemperature()
		m.mu.Lock()
		if ok {
			m.lastCPUTemp = t
		} else {
			m.lastCPUTemp = 0
		}
		m.cpuTempBusy = false
		m.mu.Unlock()
	}()
}

// ---------------------------------------------------------------------------
// NVIDIA loop-mode child
// ---------------------------------------------------------------------------

func (m *Monitor) startNvLoop(reap bool) {
	m.mu.Lock()
	if m.nvUnavailable || m.nvOneShotMode || m.nvChild != nil {
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()

	// A hard-killed server can orphan the resident loop child. Our exact query
	// string only ever appears in children we spawned, so reap any stray once
	// at boot (no loop flag in the pattern so strays from any cadence match).
	if reap && runtime.GOOS != "windows" {
		_ = exec.Command("pkill", "-9", "-f", "nvidia-smi "+strings.Join(NvQueryArgs, " ")).Run()
	}

	args := append(append([]string{}, NvQueryArgs...), "-lms", strconv.Itoa(int(MonitorTick.Milliseconds())))
	cmd := exec.Command("nvidia-smi", args...)
	cmd.Env = nvEnv()
	hideWindow(cmd)
	dieWithParent(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		m.markNvUnavailable()
		return
	}
	cmd.Stderr = io.Discard // nvidia-smi warnings are not actionable here
	if err := cmd.Start(); err != nil {
		if errors.Is(err, exec.ErrNotFound) || errors.Is(err, os.ErrNotExist) {
			m.markNvUnavailable()
		}
		return
	}
	// never leave a resident nvidia-smi behind (upstream kills it on exit)
	adoptChild(cmd)
	m.mu.Lock()
	m.nvChild = cmd
	m.nvBatch = nil
	m.lastNvLineAt = time.Now()
	m.mu.Unlock()

	go func() {
		r := bufio.NewReader(stdout)
		for {
			line, err := r.ReadString('\n')
			if strings.TrimSpace(line) != "" && (err == nil || strings.HasSuffix(line, "\n")) {
				m.onNvLine(line)
			}
			if err != nil {
				break
			}
		}
		cmd.Wait()
		m.mu.Lock()
		if m.nvChild != cmd {
			m.mu.Unlock()
			return
		}
		m.nvChild = nil
		if m.nvFlushTimer != nil {
			m.nvFlushTimer.Stop()
			m.nvFlushTimer = nil
		}
		restart := !m.nvUnavailable && !m.nvOneShotMode
		m.mu.Unlock()
		if restart {
			time.AfterFunc(nvRestartDelay, func() { m.startNvLoop(false) })
		}
	}()
}

func (m *Monitor) onNvLine(line string) {
	gpu, ok := ParseMonitorGPULine(line)
	m.mu.Lock()
	defer m.mu.Unlock()
	if ok {
		m.nvEverGotLine = true
		m.lastNvLineAt = time.Now()
		m.nvBatch = append(m.nvBatch, gpu)
	}
	if m.nvFlushTimer != nil {
		m.nvFlushTimer.Stop()
	}
	m.nvFlushTimer = time.AfterFunc(nvBatchFlush, m.flushNvBatch)
}

func (m *Monitor) flushNvBatch() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nvFlushTimer = nil
	if len(m.nvBatch) == 0 {
		return
	}
	batch := m.nvBatch
	m.nvBatch = nil
	sortByIndex(batch)
	m.latestGPU = GPUResponse{HasNvidiaSmi: true, IsMac: boolPtr(false), GPUs: batch}
}

func (m *Monitor) nvWatchdogCheck() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.nvUnavailable || m.nvChild == nil || time.Since(m.lastNvLineAt) <= nvWatchdog {
		return
	}
	log.Printf("Monitor: no output from nvidia-smi loop, restarting it")
	// loop mode that never produced a single line isn't going to start:
	// switch to a one-shot spawn per tick instead of kill/respawn forever
	if !m.nvEverGotLine {
		m.nvOneShotMode = true
	}
	if m.nvChild.Process != nil {
		m.nvChild.Process.Kill()
	}
}

func (m *Monitor) sampleNvOneShot() {
	m.mu.Lock()
	if m.nvUnavailable || m.nvOneShotBusy {
		m.mu.Unlock()
		return
	}
	m.nvOneShotBusy = true
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		m.nvOneShotBusy = false
		m.mu.Unlock()
	}()

	cmd := exec.Command("nvidia-smi", NvQueryArgs...)
	cmd.Env = nvEnv()
	hideWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) || errors.Is(err, os.ErrNotExist) {
			m.markNvUnavailable()
		} else {
			log.Printf("Monitor: one-shot nvidia-smi failed: %v", err)
		}
		return
	}
	gpus := []GPUInfo{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if g, ok := ParseMonitorGPULine(line); ok {
			gpus = append(gpus, g)
		}
	}
	sortByIndex(gpus)
	m.mu.Lock()
	m.latestGPU = GPUResponse{HasNvidiaSmi: true, IsMac: boolPtr(false), GPUs: gpus}
	m.mu.Unlock()
}

func (m *Monitor) markNvUnavailable() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nvUnavailable = true
	m.latestGPU = GPUResponse{HasNvidiaSmi: false, IsMac: boolPtr(false), GPUs: []GPUInfo{},
		Error: "nvidia-smi not found or not accessible"}
}
