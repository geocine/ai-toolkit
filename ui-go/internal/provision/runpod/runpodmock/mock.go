// Package runpodmock is an httptest stand-in for RunPod's REST API v2, the
// GraphQL balance query and the two v1 routes the lifecycle falls back to.
// It is written from the published OpenAPI (spec: runpod.md) and nothing
// else; where it has to guess, the line says UNCONFIRMED. Tests and simulate
// mode run against it, so no call here ever reaches RunPod.
//
// Pod state is computed lazily from timestamps (mockkit.Phase) on every
// request: a FakeClock makes each transition deterministic.
package runpodmock

import (
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/geocine/aitoolkit/internal/provision/mockkit"
)

// gpusJSON seeds the catalog. spec: runpod.md §5.1.
//
//go:embed testdata/gpus.json
var gpusJSON []byte

// Pod statuses (spec: runpod.md §4.1 PodStatus).
const (
	Provisioning = "PROVISIONING"
	Starting     = "STARTING"
	Running      = "RUNNING"
	Exited       = "EXITED"
	Errored      = "ERROR"
	Terminated   = "TERMINATED"
)

// createBodyMax is createPod's body limit (spec: runpod.md §3.1, 413).
const createBodyMax = 102400

// GPUType is one catalog row (spec: runpod.md §5.1 GpuType).
type GPUType struct {
	ID           string        `json:"id"`
	Name         string        `json:"name"`
	Pool         *string       `json:"pool"`
	Manufacturer string        `json:"manufacturer"`
	Memory       int           `json:"memory"`
	Secure       bool          `json:"secure"`
	Community    bool          `json:"community"`
	Price        Price         `json:"price"`
	MaxCount     MaxCount      `json:"maxCount"`
	Availability string        `json:"availability,omitempty"`
	DataCenters  []DataCenter  `json:"dataCenters,omitempty"`
	CudaVersions []CudaVersion `json:"cudaVersions,omitempty"`
}

type Price struct {
	Secure     float64  `json:"secure"`
	Community  float64  `json:"community"`
	Serverless *float64 `json:"serverless,omitempty"`
}

type MaxCount struct {
	Secure    int `json:"secure"`
	Community int `json:"community"`
}

type DataCenter struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Availability string `json:"availability"`
}

type CudaVersion struct {
	Version   string `json:"version"`
	Available bool   `json:"available"`
}

// Template is a v2 pod template (spec: runpod.md §9.1).
type Template struct {
	ID                  string            `json:"id"`
	Name                string            `json:"name"`
	Image               string            `json:"image"`
	Args                string            `json:"args"`
	Disk                int               `json:"disk"`
	Mounts              map[string]any    `json:"mounts"`
	Ports               []string          `json:"ports"`
	Env                 map[string]string `json:"env"`
	Registry            *string           `json:"registry"`
	Serverless          bool              `json:"serverless"`
	Public              bool              `json:"public"`
	Category            string            `json:"category"`
	StartSsh            bool              `json:"startSsh"`
	StartJupyter        bool              `json:"startJupyter"`
	AllowedCudaVersions []string          `json:"allowedCudaVersions"`
}

// Pod is a test's view of one pod (Instances).
type Pod struct {
	ID, Name, Image, GPUID, Cloud, Status string
	GPUCount, DiskGB                      int
	Ports                                 []string
	Env                                   map[string]string
	MinCudaVersion                        string   // gpu.minCudaVersion as sent
	AllowedCudaVersions                   []string // gpu.allowedCudaVersions as sent
	NodeURL                               string
}

// Injected is a scripted failure: the next authenticated request whose method
// and path match answers Status with a problem body (and Header) instead.
type Injected struct {
	Method string // "" = any
	Path   string // exact path, e.g. "/v2/pods" or "/graphql"
	Status int
	Detail string
	Errors []string
	Header http.Header
}

type mode int

const (
	modeCreate mode = iota // PROVISIONING -> STARTING -> RUNNING
	modeStart              // start / restart: STARTING -> RUNNING
	modeStop               // the status before the stop, then EXITED
)

type pod struct {
	id, name, image, gpuID, cloud, dc string
	count, disk, seq                  int
	ports                             []string
	env                               map[string]string
	minCuda                           string
	allowedCuda                       []string
	price                             float64 // $/h while it holds a GPU
	faults                            mockkit.Faults
	created                           time.Time
	mode                              mode
	base                              time.Time // start of the current mode
	stopFrom                          string    // modeStop: the status shown until the stop settles
	startedAt                         *time.Time
	deleted                           bool
	nodeURL                           string
	nodeStop                          func()
}

type ctxKey struct{}

type caller struct {
	key    string
	faults mockkit.Faults
}

// Mock is one RunPod account.
type Mock struct {
	Rec *mockkit.Recorder
	URL string // set by Start

	o       mockkit.Options
	handler http.Handler
	srv     *httptest.Server
	hold    atomic.Bool

	mu        sync.Mutex
	gpus      []GPUType
	templates []Template
	pods      map[string]*pod
	seq       int
	credit    float64
	calls     map[string]int // authenticated calls per key (RateLimitEvery)
	inject    []Injected
	reqID     int
}

// New builds a mock seeded with testdata/gpus.json and one aitk template.
func New(o mockkit.Options) *Mock {
	o.Defaults()
	var seed struct {
		GPUs []GPUType `json:"gpus"`
	}
	if err := json.Unmarshal(gpusJSON, &seed); err != nil {
		panic("runpodmock: testdata/gpus.json: " + err.Error())
	}
	m := &Mock{
		Rec:    &mockkit.Recorder{},
		o:      o,
		gpus:   seed.GPUs,
		pods:   map[string]*pod{},
		credit: o.Credit,
		calls:  map[string]int{},
		templates: []Template{{
			ID: "tpl_aitk0001", Name: "aitk-go headless", Image: "ghcr.io/geocine/aitoolkit-go:latest", Disk: 60,
			Mounts: map[string]any{"persistent": map[string]any{"size": 50, "path": "/workspace"}},
			Ports:  []string{"8675/http", "22/tcp"},
			Env:    map[string]string{"AI_TOOLKIT_AUTH": "{{ RUNPOD_SECRET_aitk_token }}", "AITK_CORS_ORIGINS": "*"},
			// spec: runpod.md §9.1 example template
			Category: "NVIDIA", StartSsh: true, AllowedCudaVersions: []string{},
		}, {
			ID: "9x4m2p7v", Name: "PyTorch GPU Template", Image: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", Disk: 50,
			Mounts: map[string]any{"persistent": map[string]any{"size": 20, "path": "/workspace"}},
			Ports:  []string{"8888/http"}, Env: map[string]string{"JUPYTER_ENABLE_LAB": "yes"},
			Category: "NVIDIA", StartSsh: true, AllowedCudaVersions: []string{},
		}},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v2/catalog/gpus", m.listGPUs)
	mux.HandleFunc("GET /v2/catalog/gpus/{id}", m.getGPU)
	mux.HandleFunc("GET /v2/pods", m.listPods)
	mux.HandleFunc("POST /v2/pods", m.createPod)
	mux.HandleFunc("GET /v2/pods/{id}", m.getPod)
	mux.HandleFunc("DELETE /v2/pods/{id}", m.deletePod)
	mux.HandleFunc("POST /v2/pods/{id}/action", m.podAction)
	mux.HandleFunc("GET /v2/pods/{id}/logs", m.podLogs)
	mux.HandleFunc("GET /v2/templates", m.listTemplates)
	mux.HandleFunc("POST /graphql", m.graphql)
	// v1 (retires 2026-11-15) only for the lifecycle's fallback test.
	mux.HandleFunc("POST /v1/pods/{id}/stop", m.v1Stop)
	mux.HandleFunc("DELETE /v1/pods/{id}", m.v1Delete)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		problem(w, http.StatusNotFound, "resource not found")
	})
	m.handler = m.Rec.Wrap(m.auth(mux))
	return m
}

func (m *Mock) Handler() http.Handler { return m.handler }

// Start serves the mock on 127.0.0.1:0 and returns its URL.
func (m *Mock) Start() string {
	m.srv = httptest.NewServer(m.handler)
	m.URL = m.srv.URL
	return m.URL
}

// Close stops every node and the listener.
func (m *Mock) Close() {
	m.mu.Lock()
	for _, p := range m.pods {
		stopNode(p)
	}
	m.mu.Unlock()
	if m.srv != nil {
		m.srv.Close()
	}
}

// SetCredit sets the balance GraphQL reports and createPod checks.
func (m *Mock) SetCredit(usd float64) { m.mu.Lock(); m.credit = usd; m.mu.Unlock() }

// SetAvailability changes one GPU type's stock (HIGH, MEDIUM, LOW or NONE).
func (m *Mock) SetAvailability(gpuID, level string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i := range m.gpus {
		if m.gpus[i].ID == gpuID {
			m.gpus[i].Availability = level
		}
	}
}

// FailNext scripts the next matching request's answer.
func (m *Mock) FailNext(f Injected) { m.mu.Lock(); m.inject = append(m.inject, f); m.mu.Unlock() }

// HoldLogs keeps log streams open after the backfill until the client hangs
// up, like the live API does.
func (m *Mock) HoldLogs(on bool) { m.hold.Store(on) }

// NodeURL is the simulated node of a running pod ("" when none). Simulate
// assigns it to runpod.ProxyURL: the one documented non-faithful substitution
// for https://<id>-8675.proxy.runpod.net.
func (m *Mock) NodeURL(id string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.pods[id]
	if !ok {
		return ""
	}
	m.settle(p, m.o.Clock.Now())
	return p.nodeURL
}

// Instances lists every pod ever created, newest first (deleted ones read TERMINATED).
func (m *Mock) Instances() []Pod {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.o.Clock.Now()
	var out []Pod
	for _, p := range m.sortedPods() {
		out = append(out, Pod{ID: p.id, Name: p.name, Image: p.image, GPUID: p.gpuID, Cloud: p.cloud, Status: m.settle(p, now),
			GPUCount: p.count, DiskGB: p.disk, Ports: slices.Clone(p.ports), Env: cloneEnv(p.env),
			MinCudaVersion: p.minCuda, AllowedCudaVersions: slices.Clone(p.allowedCuda), NodeURL: p.nodeURL})
	}
	return out
}

// ---- auth, rate limits, scripted failures ----

func (m *Mock) auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		m.reqID++
		w.Header().Set("X-Request-Id", fmt.Sprintf("req_%08d", m.reqID)) // observed on every v2 response
		m.mu.Unlock()
		v1 := strings.HasPrefix(r.URL.Path, "/v1/")
		// spec: runpod.md §2. The preflight is refused before CORS: 401, no
		// Access-Control-Allow-* headers (v2 has no browser access).
		if r.Method == http.MethodOptions && !v1 {
			w.Header().Set("Allow", "GET, POST")
			problem(w, http.StatusUnauthorized, "missing bearer token")
			return
		}
		key, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok && r.URL.Path == "/graphql" {
			key, ok = r.URL.Query().Get("api_key"), true // GraphQL also takes ?api_key= (spec: runpod.md §1.2)
		}
		key = strings.TrimSpace(key)
		f, allowed := m.o.Authorize(key)
		if !allowed {
			if v1 {
				w.WriteHeader(http.StatusUnauthorized) // spec: runpod.md §1.3, v1 401 has no body
				return
			}
			detail := "missing bearer token"
			if key != "" {
				detail = "invalid api key" // UNCONFIRMED: only the missing-token text was observed
			}
			problem(w, http.StatusUnauthorized, detail)
			return
		}
		m.mu.Lock()
		m.calls[key]++
		n := m.calls[key]
		var inj *Injected
		for i, next := range m.inject {
			if (next.Method == "" || next.Method == r.Method) && next.Path == r.URL.Path {
				inj = &next
				m.inject = slices.Delete(m.inject, i, i+1)
				break
			}
		}
		m.mu.Unlock()
		// spec: runpod.md §3.3; the quotas are the docs' illustrative ones.
		w.Header().Set("RateLimit-Policy", `"minute";q=60;w=60, "hour";q=3000;w=3600, "day";q=50000;w=86400`)
		if f.RateLimitEvery > 0 && n%f.RateLimitEvery == 0 {
			w.Header().Set("RateLimit", `"minute";r=0;t=2, "hour";r=2800;t=1812, "day";r=49500;t=45012`)
			w.Header().Set("Retry-After", "2")
			problem(w, http.StatusTooManyRequests, "rate limit exceeded for the minute window")
			return
		}
		w.Header().Set("RateLimit", fmt.Sprintf(`"minute";r=%d;t=60, "hour";r=%d;t=3600, "day";r=%d;t=86400`, 60-n%60, 3000-n%3000, 50000-n))
		if inj != nil {
			for k, vs := range inj.Header {
				w.Header()[k] = vs
			}
			problem(w, inj.Status, inj.Detail, inj.Errors...)
			return
		}
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxKey{}, caller{key, f})))
	})
}

func callerOf(r *http.Request) caller { c, _ := r.Context().Value(ctxKey{}).(caller); return c }

// problem writes RFC 9457 problem details (spec: runpod.md §3.1).
func problem(w http.ResponseWriter, status int, detail string, errs ...string) {
	body := map[string]any{"title": http.StatusText(status), "status": status, "detail": detail}
	if len(errs) > 0 {
		body["errors"] = errs
	}
	mockkit.WriteJSON(w, status, body, "application/problem+json")
}

// ---- catalog (spec: runpod.md §5.1) ----

var (
	catalogParams = []string{"include", "product", "count", "cloud", "countryCodes", "cudaVersions", "minCudaVersion"}
	countryRE     = regexp.MustCompile(`^[A-Z]{2}$`)
	cudaRE        = regexp.MustCompile(`^\d+\.\d+$`)
)

// catalogQuery validates the catalog params. With include=AVAILABILITY it
// returns the cloud and count to price availability for.
func catalogQuery(w http.ResponseWriter, r *http.Request) (include bool, cloud string, count int, ok bool) {
	q := r.URL.Query()
	for k := range q {
		if !slices.Contains(catalogParams, k) {
			problem(w, http.StatusUnprocessableEntity, "Request validation failed.", "query: unknown parameter '"+k+"'")
			return
		}
	}
	inc, prod := q.Get("include"), q.Get("product")
	if (inc == "") != (prod == "") {
		problem(w, http.StatusBadRequest, "include=AVAILABILITY requires product, and product is valid only with it")
		return
	}
	if inc == "" {
		for _, k := range catalogParams[2:] {
			if q.Has(k) {
				problem(w, http.StatusBadRequest, k+" is valid only with include=AVAILABILITY")
				return
			}
		}
		return false, "SECURE", 1, true
	}
	var errs []string
	if inc != "AVAILABILITY" {
		errs = append(errs, "include: must be AVAILABILITY")
	}
	for _, p := range strings.Split(prod, ",") {
		if !slices.Contains([]string{"POD", "CLUSTER", "SERVERLESS"}, p) {
			errs = append(errs, "product: must be one of POD, CLUSTER, SERVERLESS")
		}
	}
	count = 1
	if v := q.Get("count"); q.Has("count") {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			errs = append(errs, "count: must be an integer >= 1")
		}
		count = n
	}
	cloud = "SECURE"
	if q.Has("cloud") {
		cloud = q.Get("cloud")
		if cloud != "SECURE" && cloud != "COMMUNITY" {
			errs = append(errs, "cloud: must be one of SECURE, COMMUNITY")
		}
	}
	if q.Has("countryCodes") {
		for _, c := range strings.Split(q.Get("countryCodes"), ",") {
			if !countryRE.MatchString(c) {
				errs = append(errs, "countryCodes: must be ISO alpha-2 uppercase")
			}
		}
	}
	for _, v := range append(strings.Split(q.Get("cudaVersions"), ","), q.Get("minCudaVersion")) {
		if v != "" && !cudaRE.MatchString(v) {
			errs = append(errs, "cuda: must be major.minor")
		}
	}
	if len(errs) > 0 {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", errs...)
		return
	}
	if q.Get("cudaVersions") != "" && q.Get("minCudaVersion") != "" {
		problem(w, http.StatusBadRequest, "cudaVersions and minCudaVersion are mutually exclusive")
		return
	}
	// countryCodes and the CUDA filters are accepted but do not narrow the
	// seeded list (UNCONFIRMED how the live API narrows it).
	return true, cloud, count, true
}

// availability is a type's stock for one cloud and count: NONE when that
// cloud does not offer it or count is above maxCount.
func availability(g GPUType, cloud string, count int) string {
	offered, maxN := g.Secure, g.MaxCount.Secure
	if cloud == "COMMUNITY" {
		offered, maxN = g.Community, g.MaxCount.Community
	}
	if !offered || count > maxN || g.Availability == "" {
		return "NONE"
	}
	return g.Availability
}

func catalogRow(g GPUType, include bool, cloud string, count int) GPUType {
	if !include {
		g.Availability, g.DataCenters, g.CudaVersions = "", nil, nil
		return g
	}
	g.Availability = availability(g, cloud, count)
	return g
}

func (m *Mock) listGPUs(w http.ResponseWriter, r *http.Request) {
	include, cloud, count, ok := catalogQuery(w, r)
	if !ok {
		return
	}
	m.mu.Lock()
	out := make([]GPUType, 0, len(m.gpus))
	for _, g := range m.gpus {
		out = append(out, catalogRow(g, include, cloud, count))
	}
	m.mu.Unlock()
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"gpus": out}, "")
}

// getGPU: an unavailable type is 200 with NONE; 404 means it does not exist.
func (m *Mock) getGPU(w http.ResponseWriter, r *http.Request) {
	include, cloud, count, ok := catalogQuery(w, r)
	if !ok {
		return
	}
	m.mu.Lock()
	g, found := m.gpu(r.PathValue("id"))
	m.mu.Unlock()
	if !found {
		problem(w, http.StatusNotFound, "gpu type not found")
		return
	}
	mockkit.WriteJSON(w, http.StatusOK, catalogRow(g, include, cloud, count), "")
}

func (m *Mock) gpu(id string) (GPUType, bool) {
	for _, g := range m.gpus {
		if g.ID == id {
			return g, true
		}
	}
	return GPUType{}, false
}

// ---- pods (spec: runpod.md §4) ----

// createFields are CreatePodRequest's properties (unevaluatedProperties: false).
var (
	createFields = []string{"name", "image", "args", "entrypoint", "cmd", "disk", "env", "ports", "registry", "cloud", "gpu", "cpu",
		"dataCenterIds", "mounts", "templateId", "startSsh", "startJupyter", "globalNetworking"}
	gpuFields = []string{"id", "count", "allowedCudaVersions", "minCudaVersion", "minRamPerGpu", "minVcpuCountPerGpu"}
	portRE    = regexp.MustCompile(`^[0-9]{1,5}/(http|tcp)$`)
)

type createGPU struct {
	ID                  string   `json:"id"`
	Count               *int     `json:"count"`
	AllowedCudaVersions []string `json:"allowedCudaVersions"`
	MinCudaVersion      string   `json:"minCudaVersion"`
}

type createReq struct {
	Name       *string           `json:"name"`
	Image      string            `json:"image"`
	Disk       *int              `json:"disk"`
	Env        map[string]string `json:"env"`
	Ports      []string          `json:"ports"`
	Cloud      *string           `json:"cloud"`
	TemplateID string            `json:"templateId"`
	Mounts     map[string]any    `json:"mounts"`
}

func unknownKeys(raw map[string]json.RawMessage, allowed []string, at string) []string {
	var errs []string
	for _, k := range sortedKeys(raw) {
		if !slices.Contains(allowed, k) {
			errs = append(errs, at+": additional properties '"+k+"' not allowed")
		}
	}
	return errs
}

func (m *Mock) createPod(w http.ResponseWriter, r *http.Request) {
	c := callerOf(r)
	body, _ := io.ReadAll(io.LimitReader(r.Body, createBodyMax+1))
	if len(body) > createBodyMax {
		problem(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("Request body exceeds the %d byte limit.", createBodyMax))
		return
	}
	var raw map[string]json.RawMessage
	var req createReq
	if json.Unmarshal(body, &raw) != nil || json.Unmarshal(body, &req) != nil {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", "$: body must be a JSON object matching CreatePodRequest")
		return
	}
	errs := unknownKeys(raw, createFields, "$")
	if req.Name == nil {
		errs = append(errs, "$.name: required")
	} else if *req.Name == "" {
		errs = append(errs, "$.name: minLength 1")
	}
	_, hasGPU := raw["gpu"]
	_, hasCPU := raw["cpu"]
	if hasGPU == hasCPU {
		errs = append(errs, "$: exactly one of gpu or cpu is required")
	}
	var g createGPU
	if hasGPU {
		var graw map[string]json.RawMessage
		if json.Unmarshal(raw["gpu"], &graw) != nil || json.Unmarshal(raw["gpu"], &g) != nil {
			errs = append(errs, "$.gpu: must be a CreateGpuConfig object")
		} else {
			errs = append(errs, unknownKeys(graw, gpuFields, "$.gpu")...)
			if g.ID == "" {
				errs = append(errs, "$.gpu.id: required")
			}
			if g.Count != nil && *g.Count < 1 {
				errs = append(errs, "$.gpu.count: minimum 1")
			}
		}
	}
	if req.Disk != nil && *req.Disk < 1 {
		errs = append(errs, "$.disk: minimum 1")
	}
	if req.Cloud != nil && *req.Cloud != "SECURE" && *req.Cloud != "COMMUNITY" {
		errs = append(errs, "$.cloud: must be one of SECURE, COMMUNITY")
	}
	for i, p := range req.Ports {
		if !portRE.MatchString(p) {
			errs = append(errs, fmt.Sprintf("$.ports[%d]: must look like 8675/http or 22/tcp", i))
		}
	}
	if req.Image == "" && req.TemplateID == "" {
		errs = append(errs, "$.image: required unless templateId is set")
	}
	if len(errs) > 0 {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", errs...)
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if req.TemplateID != "" {
		i := slices.IndexFunc(m.templates, func(t Template) bool { return t.ID == req.TemplateID })
		if i < 0 {
			problem(w, http.StatusNotFound, "template not found")
			return
		}
		// A one-time copy: body fields win, env merges per key (spec: runpod.md §4.3).
		t := m.templates[i]
		if req.Image == "" {
			req.Image = t.Image
		}
		if req.Disk == nil {
			req.Disk = &t.Disk
		}
		if req.Ports == nil {
			req.Ports = t.Ports
		}
		env := cloneEnv(t.Env)
		for k, v := range req.Env {
			env[k] = v
		}
		req.Env = env
	}
	// 400: the body is valid but breaks a cross-field rule or cannot be placed.
	if len(g.AllowedCudaVersions) > 0 && g.MinCudaVersion != "" {
		problem(w, http.StatusBadRequest, "gpu.allowedCudaVersions and gpu.minCudaVersion are mutually exclusive")
		return
	}
	if _, ok := req.Mounts["persistent"]; ok && (req.Mounts["network"] != nil || req.Mounts["global"] != nil) {
		problem(w, http.StatusBadRequest, "mounts.persistent cannot be combined with network or global mounts")
		return
	}
	if len(req.Env) > 50 {
		// UNCONFIRMED: the docs give the 50-variable limit, not the status it fails with.
		problem(w, http.StatusBadRequest, "a pod takes at most 50 environment variables")
		return
	}
	if hasCPU {
		problem(w, http.StatusBadRequest, "CPU pods are not part of this mock")
		return
	}
	cloud := "SECURE"
	if req.Cloud != nil {
		cloud = *req.Cloud
	}
	count := 1
	if g.Count != nil {
		count = *g.Count
	}
	disk := 50 // UNCONFIRMED: the v2 default container disk (v1's containerDiskInGb default is 50)
	if req.Disk != nil {
		disk = *req.Disk
	}
	gt, known := m.gpu(g.ID)
	if c.faults.NoCapacity || !known || availability(gt, cloud, count) == "NONE" {
		// UNCONFIRMED: invented text; the live capacity detail is undocumented.
		problem(w, http.StatusBadRequest, "no available machines for the requested GPU type")
		return
	}
	price := round4(perGPU(gt, cloud)*float64(count) + float64(disk)*0.10/730)
	if c.faults.NoCredit || m.credit < price {
		// spec: runpod.md §11, on-demand needs an hour of credit (402, in text only).
		problem(w, http.StatusPaymentRequired, "insufficient balance: deploying needs at least one hour of credit")
		return
	}
	now := m.o.Clock.Now()
	m.seq++
	dc := ""
	if len(gt.DataCenters) > 0 {
		dc = gt.DataCenters[0].ID
	}
	p := &pod{id: newPodID(), name: *req.Name, image: req.Image, gpuID: g.ID, cloud: cloud, dc: dc, count: count, disk: disk, seq: m.seq,
		ports: slices.Clone(req.Ports), env: cloneEnv(req.Env), minCuda: g.MinCudaVersion, allowedCuda: slices.Clone(g.AllowedCudaVersions),
		price: price, faults: c.faults, created: now, mode: modeCreate, base: now}
	if p.ports == nil {
		p.ports = []string{}
	}
	m.pods[p.id] = p
	st := m.settle(p, now)
	mockkit.WriteJSON(w, http.StatusCreated, m.podJSON(p, st, now, false), "")
}

func (m *Mock) getPod(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.pods[r.PathValue("id")]
	if !ok {
		problem(w, http.StatusNotFound, "pod not found")
		return
	}
	now := m.o.Clock.Now()
	st := m.settle(p, now)
	if p.deleted && !(p.faults.GoneAsStatus || callerOf(r).faults.GoneAsStatus) {
		// UNCONFIRMED whether a deleted pod is 404 or 200 TERMINATED; both are served.
		problem(w, http.StatusNotFound, "pod not found")
		return
	}
	mockkit.WriteJSON(w, http.StatusOK, m.podJSON(p, st, now, false), "")
}

func (m *Mock) listPods(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	var errs []string
	for k := range q {
		if !slices.Contains([]string{"includeClusterPods", "cursor", "limit"}, k) {
			errs = append(errs, "query: unknown parameter '"+k+"'")
		}
	}
	limit := 1000
	if q.Has("limit") {
		n, err := strconv.Atoi(q.Get("limit"))
		if err != nil || n < 1 || n > 1000 {
			errs = append(errs, "limit: must be 1-1000")
		}
		limit = n
	}
	offset, ok := decodeCursor(q.Get("cursor"))
	if !ok {
		errs = append(errs, "cursor: malformed or foreign cursor")
	}
	if len(errs) > 0 {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", errs...)
		return
	}
	m.mu.Lock()
	now := m.o.Clock.Now()
	var live []map[string]any
	for _, p := range m.sortedPods() {
		if st := m.settle(p, now); !p.deleted {
			live = append(live, m.podJSON(p, st, now, true))
		}
	}
	m.mu.Unlock()
	page, next := paginate(live, offset, limit)
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"pods": page, "pagination": next}, "")
}

// actionsFor is the actions table (spec: runpod.md §4.1).
func actionsFor(status string) []string {
	switch status {
	case Running:
		return []string{"stop", "restart", "terminate"}
	case Exited, Errored:
		return []string{"start", "terminate"}
	case Provisioning, Starting:
		return []string{"stop", "terminate"}
	}
	return []string{}
}

func (m *Mock) podAction(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Action string `json:"action"`
	}
	dec := json.NewDecoder(io.LimitReader(r.Body, 1<<16))
	dec.DisallowUnknownFields() // additionalProperties: false
	if err := dec.Decode(&req); err != nil || req.Action == "" {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", "$.action: required, one of start, stop, restart, terminate")
		return
	}
	if !slices.Contains([]string{"start", "stop", "restart", "terminate"}, req.Action) {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", "$.action: must be one of start, stop, restart, terminate")
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.pods[r.PathValue("id")]
	if !ok || p.deleted {
		problem(w, http.StatusNotFound, "pod not found")
		return
	}
	now := m.o.Clock.Now()
	st := m.settle(p, now)
	if !slices.Contains(actionsFor(st), req.Action) {
		problem(w, http.StatusConflict, fmt.Sprintf("pod is %s; %s is not a valid action now", st, req.Action))
		return
	}
	switch req.Action {
	case "terminate":
		p.deleted = true
		stopNode(p)
		w.WriteHeader(http.StatusNoContent)
		return
	case "stop":
		// UNCONFIRMED what a stop returns at once; the mock keeps the old
		// status until Delays.Provision has passed, then EXITED.
		p.mode, p.base, p.stopFrom = modeStop, now, st
	case "start", "restart":
		if req.Action == "start" && (callerOf(r).faults.NoCredit || m.credit < p.price) {
			problem(w, http.StatusPaymentRequired, "insufficient balance to start this pod")
			return
		}
		p.mode, p.base, p.startedAt = modeStart, now, nil
		stopNode(p) // the container restarts on a new node
	}
	st = m.settle(p, now)
	mockkit.WriteJSON(w, http.StatusOK, m.podJSON(p, st, now, false), "")
}

func (m *Mock) deletePod(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.pods[r.PathValue("id")]
	if !ok || p.deleted {
		problem(w, http.StatusNotFound, "pod not found")
		return
	}
	p.deleted = true
	stopNode(p)
	w.WriteHeader(http.StatusNoContent)
}

// podLogs streams Server-Sent Events (spec: runpod.md §4.7, the example's
// id format with the /sequence suffix).
func (m *Mock) podLogs(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	tail := 100
	var errs []string
	if q.Has("tail") {
		n, err := strconv.Atoi(q.Get("tail"))
		if err != nil || n < 0 || n > 5000 {
			errs = append(errs, "tail: must be 0-5000")
		}
		tail = n
	}
	src := q.Get("source")
	if src != "" && src != "container" && src != "system" {
		errs = append(errs, "source: must be container or system")
	}
	if len(errs) > 0 {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", errs...)
		return
	}
	m.mu.Lock()
	p, ok := m.pods[r.PathValue("id")]
	if !ok || p.deleted {
		m.mu.Unlock()
		problem(w, http.StatusNotFound, "pod not found")
		return
	}
	now := m.o.Clock.Now()
	lines := logLines(p, m.settle(p, now))
	m.mu.Unlock()
	if src != "" {
		lines = slices.DeleteFunc(lines, func(l logLine) bool { return l.source != src })
	}
	lines = lines[max(0, len(lines)-tail):]
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	for i, l := range lines {
		ts := l.at.UTC().Format(time.RFC3339)
		data, _ := json.Marshal(map[string]string{"ts": ts, "source": l.source, "line": l.text})
		fmt.Fprintf(w, "id: %s/%012d\ndata: %s\n\n", ts, i+1, data)
	}
	if fl, ok := w.(http.Flusher); ok {
		fl.Flush()
	}
	if m.hold.Load() {
		<-r.Context().Done()
	}
}

type logLine struct {
	at           time.Time
	source, text string
}

func logLines(p *pod, st string) []logLine {
	t := p.created
	out := []logLine{{t, "system", "Pulling image " + p.image}}
	if st == Provisioning {
		return out
	}
	out = append(out, logLine{t, "system", "Image pulled"}, logLine{t, "system", "Container started"})
	if p.startedAt != nil {
		out = append(out, logLine{*p.startedAt, "container", "aitk server listening on :8675"})
	}
	switch st {
	case Errored:
		out = append(out, logLine{p.base, "system", "Container exited with an error"})
	case Exited:
		out = append(out, logLine{p.base, "system", "Container stopped"})
	}
	return out
}

func (m *Mock) listTemplates(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit := 1000
	if q.Has("limit") {
		n, err := strconv.Atoi(q.Get("limit"))
		if err != nil || n < 1 || n > 1000 {
			problem(w, http.StatusUnprocessableEntity, "Request validation failed.", "limit: must be 1-1000")
			return
		}
		limit = n
	}
	offset, ok := decodeCursor(q.Get("cursor"))
	if !ok {
		problem(w, http.StatusUnprocessableEntity, "Request validation failed.", "cursor: malformed or foreign cursor")
		return
	}
	m.mu.Lock()
	all := slices.Clone(m.templates)
	m.mu.Unlock()
	page, next := paginate(all, offset, limit)
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"templates": page, "pagination": next}, "")
}

// graphql answers the balance query only (spec: runpod.md §11). Errors come
// as HTTP 200 with errors[] (spec: runpod.md §3.1).
func (m *Mock) graphql(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query string `json:"query"`
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req)
	if !strings.Contains(req.Query, "myself") {
		mockkit.WriteJSON(w, http.StatusOK, map[string]any{"data": nil, "errors": []map[string]string{{"message": "this mock answers only myself { clientBalance }"}}}, "")
		return
	}
	m.mu.Lock()
	now := m.o.Clock.Now()
	spend := 0.0
	for _, p := range m.pods {
		if st := m.settle(p, now); st != Exited && st != Terminated {
			spend += p.price
		}
	}
	credit := m.credit
	m.mu.Unlock()
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"data": map[string]any{"myself": map[string]any{
		"id": "user_mock", "clientBalance": credit, "currentSpendPerHr": round4(spend), "spendLimit": 80}}}, "")
}

func (m *Mock) v1Stop(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.pods[r.PathValue("id")]
	if !ok || p.deleted {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	now := m.o.Clock.Now()
	p.mode, p.base, p.stopFrom = modeStop, now, m.settle(p, now)
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"id": p.id, "desiredStatus": Exited}, "")
}

func (m *Mock) v1Delete(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.pods[r.PathValue("id")]
	if !ok || p.deleted {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	p.deleted = true
	stopNode(p)
	w.WriteHeader(http.StatusNoContent)
}

// ---- lazy state ----

// settle computes p's status at now and starts or stops its node to match:
// a node runs exactly while the pod reads RUNNING. Callers hold m.mu.
func (m *Mock) settle(p *pod, now time.Time) string {
	if p.deleted {
		stopNode(p)
		return Terminated
	}
	f, d := p.faults, m.o.Delays
	var st string
	var runAt time.Time
	switch p.mode {
	case modeCreate:
		st = []string{Provisioning, Starting, Running}[mockkit.Phase(now, p.base, f.Scale(d.Provision), f.Scale(d.Start))]
		runAt = p.base.Add(f.Scale(d.Provision) + f.Scale(d.Start))
	case modeStart:
		st = []string{Starting, Running}[mockkit.Phase(now, p.base, f.Scale(d.Start))]
		runAt = p.base.Add(f.Scale(d.Start))
	case modeStop:
		st = []string{p.stopFrom, Exited}[mockkit.Phase(now, p.base, f.Scale(d.Provision))]
	}
	if p.mode != modeStop && st == Running {
		if p.startedAt == nil {
			t := runAt
			p.startedAt = &t
		}
		if f.CrashAfterRunning && !now.Before(runAt.Add(f.Scale(d.App))) {
			st = Errored
		}
	}
	if st == Running {
		if p.nodeURL == "" && m.o.Node != nil {
			g, _ := m.gpu(p.gpuID)
			spec := mockkit.NodeSpec{Kind: "runpod", InstanceID: p.id, Env: cloneEnv(p.env), GPU: g.Name, GPUCount: p.count,
				VRAMGB: g.Memory, AppDelay: f.Scale(d.App), Image: p.image}.WithFaults(f)
			p.nodeURL, p.nodeStop = m.o.Node(spec)
		}
	} else {
		stopNode(p)
	}
	return st
}

func stopNode(p *pod) {
	if p.nodeStop != nil {
		p.nodeStop()
	}
	p.nodeURL, p.nodeStop = "", nil
}

// podJSON is the v2 Pod object (spec: runpod.md §4.4). In lists runtime is
// always null.
func (m *Mock) podJSON(p *pod, st string, now time.Time, list bool) map[string]any {
	g, _ := m.gpu(p.gpuID)
	cost := p.price
	if st == Exited || st == Terminated {
		cost = 0
	}
	actions := actionsFor(st)
	var dc, runtime, started any
	if st != Provisioning && p.dc != "" {
		dc = p.dc
	}
	ssh := map[string]any{"proxy": nil, "direct": nil} // direct needs 22/tcp, never exposed here
	if st != Provisioning && st != Terminated {
		user := p.id + "-64411eb2"
		ssh["proxy"] = map[string]any{"host": "ssh.runpod.io", "port": 22, "username": user, "command": "ssh " + user + "@ssh.runpod.io"}
	}
	if p.startedAt != nil {
		started = p.startedAt.UTC().Format(time.RFC3339)
	}
	if st == Running && !list {
		gpus := make([]map[string]float64, p.count)
		for i := range gpus {
			gpus[i] = map[string]float64{"util": 0, "memoryUtil": 0}
		}
		var uptime int64
		if p.startedAt != nil {
			uptime = int64(now.Sub(*p.startedAt).Seconds())
		}
		var ports []map[string]any
		for _, pt := range p.ports {
			num, typ, _ := strings.Cut(pt, "/")
			n, _ := strconv.Atoi(num)
			// UNCONFIRMED whether public/ip are null for http ports.
			port := map[string]any{"private": n, "public": nil, "type": typ, "ip": nil}
			if typ == "tcp" {
				port["public"], port["ip"] = 30000+n, "203.0.113.10"
			}
			ports = append(ports, port)
		}
		runtime = map[string]any{"uptime": uptime, "gpus": gpus, "cpu": map[string]float64{"util": 0}, "memory": map[string]float64{"util": 0}, "ports": ports}
	}
	cuda := "12.8"
	for _, v := range g.CudaVersions {
		if v.Available {
			cuda = v.Version
		}
	}
	return map[string]any{
		"id": p.id, "name": p.name, "image": p.image, "args": "", "entrypoint": []string{}, "cmd": []string{},
		"disk": p.disk, "ports": p.ports, "env": p.env, "registry": nil,
		"status": st, "actions": actions, "mounts": map[string]any{},
		"gpu":   map[string]any{"id": p.gpuID, "count": p.count, "vcpuCount": 8 * p.count, "memory": 31 * p.count},
		"cloud": p.cloud, "dataCenterId": dc, "cudaVersion": cuda, "ssh": ssh, "template": nil,
		"cost": cost, "locked": false, "globalNetworking": map[string]bool{"enabled": false},
		"runtime": runtime, "createdAt": p.created.UTC().Format(time.RFC3339), "startedAt": started,
	}
}

// ---- helpers ----

func (m *Mock) sortedPods() []*pod {
	out := make([]*pod, 0, len(m.pods))
	for _, p := range m.pods {
		out = append(out, p)
	}
	slices.SortFunc(out, func(a, b *pod) int { return b.seq - a.seq }) // newest first
	return out
}

// paginate returns one page and its Pagination (spec: runpod.md §3.2).
func paginate[T any](all []T, offset, limit int) ([]T, map[string]any) {
	if all == nil {
		all = []T{}
	}
	offset = min(offset, len(all))
	end := min(offset+limit, len(all))
	next := map[string]any{"nextCursor": nil, "hasNextPage": false}
	if end < len(all) {
		next = map[string]any{"nextCursor": base64.RawURLEncoding.EncodeToString([]byte("offset=" + strconv.Itoa(end))), "hasNextPage": true}
	}
	return all[offset:end], next
}

func decodeCursor(c string) (int, bool) {
	if c == "" {
		return 0, true
	}
	b, err := base64.RawURLEncoding.DecodeString(c)
	if err != nil {
		return 0, false
	}
	v, ok := strings.CutPrefix(string(b), "offset=")
	n, err := strconv.Atoi(v)
	return n, ok && err == nil && n >= 0
}

func perGPU(g GPUType, cloud string) float64 {
	if cloud == "COMMUNITY" {
		return g.Price.Community
	}
	return g.Price.Secure
}

func round4(x float64) float64 { return math.Round(x*1e4) / 1e4 }

// newPodID is 10 lowercase letters and digits, like "7h9k2m4n6p".
func newPodID() string {
	const abc = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 10)
	_, _ = rand.Read(b)
	for i := range b {
		b[i] = abc[int(b[i])%len(abc)]
	}
	return string(b)
}

func cloneEnv(env map[string]string) map[string]string {
	out := make(map[string]string, len(env))
	for k, v := range env {
		out[k] = v
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}
