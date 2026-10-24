// Package mockkit is shared by the three provider mocks (tests and simulate).
// Mock state is computed lazily from timestamps on each request, with no
// goroutines, so a FakeClock makes every transition deterministic.
package mockkit

import (
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
)

type Clock interface{ Now() time.Time }

type RealClock struct{}

func (RealClock) Now() time.Time { return time.Now() }

type FakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func NewFakeClock(t time.Time) *FakeClock { return &FakeClock{t: t} }
func (c *FakeClock) Now() time.Time       { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *FakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// Phase is how many of the consecutive steps have elapsed since `since`.
func Phase(now, since time.Time, steps ...time.Duration) int {
	n, at := 0, since
	for _, s := range steps {
		at = at.Add(s)
		if now.Before(at) {
			return n
		}
		n++
	}
	return n
}

type Delays struct {
	Provision time.Duration // creating -> starting (default 3s; 0 with FakeClock tests that advance explicitly)
	Start     time.Duration // starting -> running (default 6s)
	App       time.Duration // node answers 503 this long after running (default 4s)
}

type PodEnvMode string

const (
	PodEnvAppend  PodEnvMode = ""        // QuickPod per-pod docker_options appended (spec assumption)
	PodEnvReplace PodEnvMode = "replace" // replaces the template's options (template -p lost)
	PodEnvIgnore  PodEnvMode = "ignore"  // ignored (template env wins)
)

// What the simulated node's GET /v1/system/cuda reports (Faults.CUDA).
const (
	CUDAOK        = ""          // torch sees the GPU and a kernel runs
	CUDANoKernel  = "nokernel"  // no kernel image for this GPU (the image's PyTorch lacks its SM)
	CUDAOldDriver = "olddriver" // the host driver is too old for the image's CUDA
)

type Faults struct {
	Unauthorized      bool       // every authenticated call -> provider 401
	NoCredit          bool       // create/start -> provider credit error; the balance reads 0
	NoCapacity        bool       // create -> provider capacity error
	RateLimitEvery    int        // every Nth authenticated call -> 429 (0 = off)
	CrashAfterRunning bool       // provider error state after running
	NoPortMapping     bool       // Vast/QuickPod: running without 8675
	OpenNode          bool       // the simulated node does not enforce auth
	Slow              bool       // all delays x10
	PodEnv            PodEnvMode // QuickPod per-pod docker_options semantics
	GoneAsStatus      bool       // RunPod: terminated pods answer 200 TERMINATED instead of 404
	CUDA              string     // the node's /v1/system/cuda: CUDAOK, CUDANoKernel or CUDAOldDriver
	StaleImage        bool       // the node's /v1/health reports an older build than the launch ref implies
}

// faultNames are the suffixes a user can type as a key in simulate mode
// ("sim_full" etc.); docs/providers.md lists them.
var faultNames = map[string]func(*Faults){
	"sim_bad":       func(f *Faults) { f.Unauthorized = true },
	"sim_broke":     func(f *Faults) { f.NoCredit = true },
	"sim_full":      func(f *Faults) { f.NoCapacity = true },
	"sim_429":       func(f *Faults) { f.RateLimitEvery = 3 },
	"sim_crash":     func(f *Faults) { f.CrashAfterRunning = true },
	"sim_noport":    func(f *Faults) { f.NoPortMapping = true },
	"sim_open":      func(f *Faults) { f.OpenNode = true },
	"sim_slow":      func(f *Faults) { f.Slow = true },
	"sim_replace":   func(f *Faults) { f.PodEnv = PodEnvReplace },
	"sim_ignore":    func(f *Faults) { f.PodEnv = PodEnvIgnore },
	"sim_gone":      func(f *Faults) { f.GoneAsStatus = true },
	"sim_nokernel":  func(f *Faults) { f.CUDA = CUDANoKernel },
	"sim_olddriver": func(f *Faults) { f.CUDA = CUDAOldDriver },
	"sim_stale":     func(f *Faults) { f.StaleImage = true },
}

// FaultNames lists the fault keys, sorted (docs and tests).
func FaultNames() []string {
	out := make([]string, 0, len(faultNames))
	for k := range faultNames {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// NodeSpec describes the simulated aitk server of one running instance.
// The provider mocks fill OpenNode, CUDA and StaleImage from the faults
// Authorize returned (WithFaults), and Image from the create body, when they
// start a node; sim/node.go reads only this struct.
type NodeSpec struct {
	Kind       string
	InstanceID string
	Env        map[string]string // what the container would receive
	GPU        string
	GPUCount   int
	VRAMGB     int
	OpenNode   bool
	AppDelay   time.Duration
	CUDA       string // Faults.CUDA: what /v1/system/cuda reports (CUDAOK, CUDANoKernel, CUDAOldDriver)
	StaleImage bool   // Faults.StaleImage: /v1/health reports an older build than Image implies
	Image      string // LaunchSpec.Image as the mock received it (the launch_ref); /v1/health version derives from it
}

// WithFaults copies the node-side faults into n (OpenNode, CUDA, StaleImage).
func (n NodeSpec) WithFaults(f Faults) NodeSpec {
	n.OpenNode, n.CUDA, n.StaleImage = f.OpenNode, f.CUDA, f.StaleImage
	return n
}

// NodeFactory starts a node and returns its base URL (http://127.0.0.1:<port>).
// nil means no node (unit tests): mocks then report 203.0.113.10 addresses.
type NodeFactory func(n NodeSpec) (baseURL string, stop func())

type Options struct {
	Clock     Clock
	Delays    Delays
	Keys      []string // accepted keys; nil = any non-empty key (unit tests)
	SimSecret string   // simulate: key must be SimSecret or SimSecret+"."+<fault name>
	Credit    float64  // default 25
	Faults    Faults   // applied to every key (tests)
	Node      NodeFactory
}

func (o *Options) Defaults() {
	if o.Clock == nil {
		o.Clock = RealClock{}
	}
	if o.Credit == 0 {
		o.Credit = 25
	}
}

// Authorize checks a presented key and returns the faults for it.
func (o Options) Authorize(key string) (Faults, bool) {
	f := o.Faults
	if key == "" {
		return f, false
	}
	if o.SimSecret != "" {
		if key == o.SimSecret {
			return f, !f.Unauthorized
		}
		suffix, ok := strings.CutPrefix(key, o.SimSecret+".")
		if !ok {
			return f, false
		}
		if apply, known := faultNames[suffix]; known {
			apply(&f)
		}
		return f, !f.Unauthorized
	}
	if o.Keys != nil && !slices.Contains(o.Keys, key) {
		return f, false
	}
	return f, !f.Unauthorized
}

// Balance is the credit a mock reports to a caller with faults f: a sim_broke
// key (NoCredit) reads as an empty account, as its creates and starts fail.
func (f Faults) Balance(credit float64) float64 {
	if f.NoCredit {
		return 0
	}
	return credit
}

func (f Faults) Scale(d time.Duration) time.Duration {
	if f.Slow {
		return 10 * d
	}
	return d
}

// Recorded is one request a mock saw (tests assert exact shapes).
type Recorded struct {
	Method, Path, RawQuery string
	Header                 http.Header
	Body                   []byte
}

type Recorder struct {
	mu   sync.Mutex
	reqs []Recorded
}

// Wrap records every request, then serves it.
func (r *Recorder) Wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		b, _ := io.ReadAll(io.LimitReader(req.Body, 1<<20))
		req.Body = io.NopCloser(strings.NewReader(string(b)))
		r.mu.Lock()
		r.reqs = append(r.reqs, Recorded{req.Method, req.URL.Path, req.URL.RawQuery, req.Header.Clone(), b})
		r.mu.Unlock()
		next.ServeHTTP(w, req)
	})
}

func (r *Recorder) Requests() []Recorded {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.reqs)
}

func WriteJSON(w http.ResponseWriter, status int, v any, contentType string) {
	if contentType == "" {
		contentType = "application/json"
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
