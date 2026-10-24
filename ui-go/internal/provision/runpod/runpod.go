// Package runpod is the RunPod adapter. It is written only from RunPod's
// published REST API v2 OpenAPI (scratchpad spec runpod.md) and runs on the
// server that holds the provider keys: v2 sends no CORS headers, so the studio never calls it.
// runpodmock implements the same surface for tests and simulate mode.
package runpod

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/provision"
)

// Bases are package vars only tests and the simulate wiring change (PLAN D7).
var (
	API     = "https://api.runpod.io"
	GraphQL = API + "/graphql"
	// ProxyURL is a pod's public https address for port 8675 (spec: runpod.md
	// §7). Simulate points it at the mock's node; "" means no endpoint yet.
	ProxyURL = func(id string) string { return "https://" + id + "-8675.proxy.runpod.net" }
	// LogsWait bounds a log read: RunPod keeps the SSE stream open.
	LogsWait = 3 * time.Second
)

// diskPerGBHour is container disk at $0.10/GB/month while running (spec:
// runpod.md §11), spread over 730 hours.
const diskPerGBHour = 0.10 / 730

func init() { provision.Register(New()) }

// Adapter is the RunPod provision.Cloud. Its only state is the catalog's
// display names, cached from Verify, Offers and Quote so Get can show
// "RTX 4090" for the pod's "NVIDIA GeForce RTX 4090". Get and List read the
// catalog themselves when a pod's type is not cached (after a restart).
type Adapter struct {
	mu     sync.RWMutex
	types  map[string]gpuInfo
	filled time.Time // last catalog read for a cache miss
}

// refillEvery spaces the catalog reads a cache miss triggers, so a type the
// catalog does not list does not cost a call on every watcher refresh.
const refillEvery = 5 * time.Minute

type gpuInfo struct {
	name   string
	vramGB int
}

func New() *Adapter { return &Adapter{types: map[string]gpuInfo{}} }

func (a *Adapter) Kind() provision.Kind { return provision.RunPod }

func (a *Adapter) Info() provision.KindInfo {
	return provision.KindInfo{
		Kind:  provision.RunPod,
		Label: provision.Label(provision.RunPod),
		Stage: "stable",
		Caps: provision.Caps{
			CloudChoice: true, CountChoice: true, StopStart: true, StopKeepsDisk: false,
			HTTPS: "yes", SelfDestroy: "key", Credit: true, EnvCountMax: 50, EnvTotalMax: 100000,
			ImageDigestRef: "unconfirmed", ReportsInet: false, ReportsDiskBW: false,
		},
		Unconfirmed: []string{
			"Whether a pod's own RUNPOD_API_KEY works on API v2",
			"The exact text RunPod returns when a GPU is out of stock",
			"Whether GET returns 404 or TERMINATED after a pod is deleted",
		},
		ConsoleURL: "https://console.runpod.io",
		KeysURL:    "https://console.runpod.io/user/settings",
		KeyScopes:  []string{"pods:read", "pods:write", "catalog:read", "templates:read"},
	}
}

// ---- wire shapes (spec: runpod.md §4.4, §5.1, §9.1) ----

type gpuType struct {
	ID           string  `json:"id"`
	Name         string  `json:"name"`
	Manufacturer string  `json:"manufacturer"`
	Memory       float64 `json:"memory"` // VRAM GB
	Secure       bool    `json:"secure"`
	Community    bool    `json:"community"`
	Availability string  `json:"availability"` // NONE | LOW | MEDIUM | HIGH
	Price        struct {
		Secure    float64 `json:"secure"`
		Community float64 `json:"community"`
	} `json:"price"`
	MaxCount struct {
		Secure    int `json:"secure"`
		Community int `json:"community"`
	} `json:"maxCount"`
	CUDAVersions []hostfit.RunPodCUDAVersion `json:"cudaVersions"` // with include=AVAILABILITY
}

// host is the type's host facts for q: RunPod places the pod on a host at
// the image's CUDA floor (gpu.minCudaVersion), so the floor is a placement
// rule, not a host filter.
func (g gpuType) host(q provision.OfferQuery) hostfit.Host {
	cudaMin := ""
	if q.Filter != nil {
		cudaMin = hostfit.RunPodGPUConfig(*q.Filter)
	}
	return hostfit.FromRunPodGPU(hostfit.RunPodGPUType{ID: g.ID, Name: g.Name, Manufacturer: g.Manufacturer,
		Memory: g.Memory, CUDAVersions: g.CUDAVersions}, cudaMin)
}

// allowed applies q's image filter to a catalog type: a GPU the image's
// build cannot run (or the policy hides) and, with a model, one below its
// VRAM floor are dropped unless ShowAll, where the fit badges them.
func (g gpuType) allowed(q provision.OfferQuery, h hostfit.Host) bool {
	f := q.Filter
	if f == nil || f.ShowAll {
		return true
	}
	if ok, _ := hostfit.RunPodAllowedType(*f, g.ID); !ok {
		return false
	}
	return f.MinVRAMGB <= 0 || h.VRAMGB <= 0 || h.VRAMGB >= f.MinVRAMGB
}

type podJSON struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Status       string            `json:"status"`
	Actions      []string          `json:"actions"`
	DataCenterID *string           `json:"dataCenterId"`
	Cost         *float64          `json:"cost"`
	CreatedAt    string            `json:"createdAt"`
	Env          map[string]string `json:"env"` // for adopt only (Instance.Env)
	GPU          *struct {
		ID    string `json:"id"`
		Count int    `json:"count"`
	} `json:"gpu"`
}

type pagination struct {
	NextCursor  *string `json:"nextCursor"`
	HasNextPage bool    `json:"hasNextPage"`
}

// ---- calls ----

// req is one v2 call. scope names the key scope a 403 means is missing;
// create marks POST /v2/pods, where 400 and 403 mean capacity.
type req struct {
	method, path string
	query        url.Values
	body         any
	scope        string
	create       bool
	secrets      []string // redacted from errors besides the key (env values)
}

func (a *Adapter) do(ctx context.Context, key string, r req) (provision.Response, error) {
	u := strings.TrimRight(API, "/") + r.path
	if len(r.query) > 0 {
		u += "?" + r.query.Encode()
	}
	res, err := provision.Do(ctx, provision.Call{Kind: provision.RunPod, Method: r.method, URL: u,
		Header: map[string]string{"Authorization": "Bearer " + key}, Body: r.body, Secret: key})
	if err != nil {
		return res, err
	}
	if res.Status >= 400 {
		return res, apiError(res, r, append([]string{key}, r.secrets...))
	}
	return res, nil
}

func decode(res provision.Response, v any, what string) error {
	if err := json.Unmarshal(res.Body, v); err != nil {
		return provision.Errorf(provision.RunPod, provision.ClassUnavailable, res.Status, "unreadable %s", what)
	}
	return nil
}

// Verify reads the catalog (no query: it has no limit) and one pod, then
// asks GraphQL for the balance, best effort.
func (a *Adapter) Verify(ctx context.Context, key string) (provision.Account, error) {
	res, err := a.do(ctx, key, req{method: http.MethodGet, path: "/v2/catalog/gpus", scope: "catalog:read"})
	if err != nil {
		return provision.Account{}, err
	}
	a.rememberCatalog(res)
	if _, err := a.do(ctx, key, req{method: http.MethodGet, path: "/v2/pods", query: url.Values{"limit": {"1"}}, scope: "pods:read"}); err != nil {
		return provision.Account{}, err
	}
	return provision.Account{Credit: a.balance(ctx, key)}, nil
}

// balance is GraphQL myself.clientBalance (v2 has no balance). Any failure
// or a missing field gives nil.
func (a *Adapter) balance(ctx context.Context, key string) *float64 {
	res, err := provision.Do(ctx, provision.Call{Kind: provision.RunPod, Method: http.MethodPost, URL: GraphQL,
		Header: map[string]string{"Authorization": "Bearer " + key}, Body: map[string]string{"query": "query { myself { clientBalance } }"}, Secret: key})
	if err != nil || res.Status != http.StatusOK {
		return nil
	}
	var out struct {
		Data *struct {
			Myself *struct {
				ClientBalance *float64 `json:"clientBalance"`
			} `json:"myself"`
		} `json:"data"`
	}
	if json.Unmarshal(res.Body, &out) != nil || out.Data == nil || out.Data.Myself == nil {
		return nil
	}
	return out.Data.Myself.ClientBalance
}

// ---- offers ----

// cloudParam maps OfferQuery.Cloud to the API's enum.
func cloudParam(c string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(c)) {
	case "", "secure":
		return "SECURE", nil
	case "community":
		return "COMMUNITY", nil
	}
	return "", provision.Errorf(provision.RunPod, provision.ClassInvalid, 0, "cloud must be secure or community")
}

// catalogQuery is the exact availability query: include, product, cloud and
// count, nothing else.
func catalogQuery(cloud string, count int) url.Values {
	return url.Values{"include": {"AVAILABILITY"}, "product": {"POD"}, "cloud": {cloud}, "count": {strconv.Itoa(count)}}
}

func (a *Adapter) remember(gs ...gpuType) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, g := range gs {
		if g.ID != "" && g.Name != "" {
			a.types[g.ID] = gpuInfo{g.Name, int(math.Round(g.Memory))}
		}
	}
}

// rememberCatalog caches the rows of a GET /v2/catalog/gpus answer; an
// unreadable body is ignored.
func (a *Adapter) rememberCatalog(res provision.Response) {
	var body struct {
		GPUs []gpuType `json:"gpus"`
	}
	if json.Unmarshal(res.Body, &body) == nil {
		a.remember(body.GPUs...)
	}
}

// learn reads the whole catalog (no query, the same cheap call as Verify)
// when one of pods has a GPU type that is not cached, at most once per
// refillEvery. Any error keeps the id-and-0 fallback.
func (a *Adapter) learn(ctx context.Context, key string, pods ...podJSON) {
	a.mu.Lock()
	miss := false
	for _, p := range pods {
		if p.GPU != nil && p.GPU.ID != "" {
			if _, ok := a.types[p.GPU.ID]; !ok {
				miss = true
				break
			}
		}
	}
	if !miss || (!a.filled.IsZero() && time.Since(a.filled) < refillEvery) {
		a.mu.Unlock()
		return
	}
	a.filled = time.Now()
	a.mu.Unlock()
	if res, err := a.do(ctx, key, req{method: http.MethodGet, path: "/v2/catalog/gpus", scope: "catalog:read"}); err == nil {
		a.rememberCatalog(res)
	}
}

// display is the catalog name and VRAM of a GPU type id (the id itself and 0
// when the catalog does not list it).
func (a *Adapter) display(id string) (string, int) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if t, ok := a.types[id]; ok {
		return t.name, t.vramGB
	}
	return id, 0
}

// OfferID is "<gpu type id>|<SECURE|COMMUNITY>|<count>".
func OfferID(gpuID, cloud string, count int) string {
	return gpuID + "|" + cloud + "|" + strconv.Itoa(count)
}

func parseOfferID(id string) (gpuID, cloud string, count int, err error) {
	bad := provision.Errorf(provision.RunPod, provision.ClassInvalid, 0, "unknown offer id %q", provision.Clip(id, 80))
	rest, n, ok := cutLast(id)
	if !ok {
		return "", "", 0, bad
	}
	gpuID, cloud, ok = cutLast(rest)
	count, convErr := strconv.Atoi(n)
	if !ok || gpuID == "" || (cloud != "SECURE" && cloud != "COMMUNITY") || convErr != nil || count < 1 {
		return "", "", 0, bad
	}
	return gpuID, cloud, count, nil
}

func cutLast(s string) (before, after string, ok bool) {
	i := strings.LastIndex(s, "|")
	if i < 0 {
		return "", "", false
	}
	return s[:i], s[i+1:], true
}

// offerFor prices one catalog row for cloud and count. ok is false when the
// row cannot be rented that way: no stock, that cloud does not offer it, the
// count is above maxCount, or no price.
func offerFor(g gpuType, cloud string, count, diskGB int) (provision.Offer, bool) {
	perGPU, offered, maxN := g.Price.Secure, g.Secure, g.MaxCount.Secure
	if cloud == "COMMUNITY" {
		perGPU, offered, maxN = g.Price.Community, g.Community, g.MaxCount.Community
	}
	if strings.EqualFold(g.Availability, "NONE") || !offered || count > maxN || perGPU <= 0 {
		return provision.Offer{}, false
	}
	zero := 0.0 // the container disk is wiped on stop, so a stopped pod bills nothing
	return provision.Offer{
		ID: OfferID(g.ID, cloud, count), Provider: provision.RunPod, GPU: g.Name, GPUCount: count,
		VRAMGB:              int(math.Round(g.Memory)),
		PricePerHour:        round4(perGPU*float64(count) + float64(diskGB)*diskPerGBHour),
		StoppedPricePerHour: &zero,
		Location:            "", // RunPod picks the data center; dataCenterIds is never sent
		Availability:        strings.ToLower(g.Availability),
		Cloud:               strings.ToLower(cloud),
		HTTPS:               true,
	}, true
}

func diskOf(q provision.OfferQuery) int {
	if q.DiskGB > 0 {
		return q.DiskGB
	}
	return provision.DefaultDiskGB
}

// Offers lists one offer per GPU type with stock, filtered on VRAM (min and
// the exclusive max), the GPU substring and the price ceiling, cheapest
// first. RunPod reports no bandwidth, so q.Fast changes nothing (D-15).
func (a *Adapter) Offers(ctx context.Context, key string, q provision.OfferQuery) ([]provision.Offer, error) {
	cloud, err := cloudParam(q.Cloud)
	if err != nil {
		return nil, err
	}
	count := max(q.GPUCount, 1)
	res, err := a.do(ctx, key, req{method: http.MethodGet, path: "/v2/catalog/gpus", query: catalogQuery(cloud, count), scope: "catalog:read"})
	if err != nil {
		return nil, err
	}
	var body struct {
		GPUs []gpuType `json:"gpus"`
	}
	if err := decode(res, &body, "GPU catalog"); err != nil {
		return nil, err
	}
	a.remember(body.GPUs...)
	needle := strings.ToLower(strings.TrimSpace(q.GPU))
	out := []provision.Offer{}
	for _, g := range body.GPUs {
		o, ok := offerFor(g, cloud, count, diskOf(q))
		switch {
		case !ok,
			o.VRAMGB < q.MinVRAMGB,
			q.MaxVRAMGB > 0 && o.VRAMGB >= q.MaxVRAMGB,
			needle != "" && !strings.Contains(strings.ToLower(g.Name), needle) && !strings.Contains(strings.ToLower(g.ID), needle),
			q.MaxPrice > 0 && o.PricePerHour > q.MaxPrice:
			continue
		}
		h := g.host(q)
		if !g.allowed(q, h) {
			continue
		}
		provision.DecorateOffer(&o, h, q)
		out = append(out, o)
	}
	slices.SortStableFunc(out, func(x, y provision.Offer) int {
		if x.PricePerHour != y.PricePerHour {
			if x.PricePerHour < y.PricePerHour {
				return -1
			}
			return 1
		}
		return strings.Compare(x.GPU, y.GPU)
	})
	limit := q.Limit
	if limit <= 0 {
		limit = 50
	}
	return out[:min(len(out), limit, 200)], nil
}

// Quote re-reads the offer's GPU type. The offer id fixes the cloud and the
// GPU count (what the user picked is what is rented); q.DiskGB prices the
// disk. A type that is gone or has no stock is a capacity error.
func (a *Adapter) Quote(ctx context.Context, key, offerID string, q provision.OfferQuery) (provision.Offer, error) {
	gpuID, cloud, count, err := parseOfferID(offerID)
	if err != nil {
		return provision.Offer{}, err
	}
	res, err := a.do(ctx, key, req{method: http.MethodGet, path: "/v2/catalog/gpus/" + url.PathEscape(gpuID), query: catalogQuery(cloud, count), scope: "catalog:read"})
	if provision.ClassOf(err) == provision.ClassNotFound {
		return provision.Offer{}, provision.Errorf(provision.RunPod, provision.ClassCapacity, http.StatusNotFound, "%s is no longer offered", gpuID)
	}
	if err != nil {
		return provision.Offer{}, err
	}
	var g gpuType
	if err := decode(res, &g, "GPU type"); err != nil {
		return provision.Offer{}, err
	}
	a.remember(g)
	o, ok := offerFor(g, cloud, count, diskOf(q))
	if !ok {
		name := g.Name
		if name == "" {
			name = gpuID
		}
		return provision.Offer{}, provision.Errorf(provision.RunPod, provision.ClassCapacity, res.Status,
			"no %d× %s free on %s cloud right now", count, name, strings.ToLower(cloud))
	}
	provision.DecorateOffer(&o, g.host(q), q)
	return o, nil
}

// ---- templates ----

// maxPages stops a cursor loop that never ends.
const maxPages = 50

func (a *Adapter) Templates(ctx context.Context, key string) ([]provision.Template, error) {
	out := []provision.Template{}
	cursor := ""
	for range maxPages {
		q := url.Values{}
		if cursor != "" {
			q.Set("cursor", cursor)
		}
		res, err := a.do(ctx, key, req{method: http.MethodGet, path: "/v2/templates", query: q, scope: "templates:read"})
		if err != nil {
			return nil, err
		}
		var body struct {
			Templates []struct {
				ID    string                     `json:"id"`
				Name  string                     `json:"name"`
				Image string                     `json:"image"`
				Ports []string                   `json:"ports"`
				Env   map[string]json.RawMessage `json:"env"` // only the keys are read
			} `json:"templates"`
			Pagination pagination `json:"pagination"`
		}
		if err := decode(res, &body, "template list"); err != nil {
			return nil, err
		}
		for _, t := range body.Templates {
			_, token := t.Env["AI_TOOLKIT_AUTH"]
			out = append(out, provision.Template{ID: t.ID, Name: t.Name, Image: t.Image,
				Ours:          strings.Contains(t.Image, "aitoolkit-go"),
				PublishesPort: slices.Contains(t.Ports, httpPort()),
				HasToken:      token})
		}
		if !body.Pagination.HasNextPage || body.Pagination.NextCursor == nil || *body.Pagination.NextCursor == "" {
			break
		}
		cursor = *body.Pagination.NextCursor
	}
	return out, nil
}

func httpPort() string { return strconv.Itoa(provision.Port) + "/http" }

// ---- pods ----

// gpuConfig is CreateGpuConfig. minCudaVersion is the image's CUDA floor
// and allowedCudaVersions is never sent: RunPod answers 400 when both are.
type gpuConfig struct {
	ID             string `json:"id"`
	Count          int    `json:"count"`
	MinCudaVersion string `json:"minCudaVersion,omitempty"`
}

// createBody is the whole POST /v2/pods body: the API rejects any field it
// does not know (unevaluatedProperties: false), and nothing else is needed.
type createBody struct {
	Name  string            `json:"name"`
	Image string            `json:"image"`
	GPU   gpuConfig         `json:"gpu"`
	Cloud string            `json:"cloud"`
	Disk  int               `json:"disk,omitempty"`
	Ports []string          `json:"ports"`
	Env   map[string]string `json:"env,omitempty"`
}

func (a *Adapter) Launch(ctx context.Context, key string, s provision.LaunchSpec) (provision.Ref, error) {
	gpuID, cloud, count, err := parseOfferID(s.OfferID)
	if err != nil {
		return provision.Ref{}, err
	}
	var secrets []string
	for _, v := range s.Env {
		if len(v) >= 8 { // short values ("1", "true") would garble the message
			secrets = append(secrets, v)
		}
	}
	res, err := a.do(ctx, key, req{method: http.MethodPost, path: "/v2/pods", scope: "pods:write", create: true, secrets: secrets,
		body: createBody{Name: s.Label, Image: s.Image, GPU: gpuConfig{gpuID, count, provision.LaunchCUDAMin(ctx)}, Cloud: cloud, Disk: s.DiskGB,
			Ports: []string{httpPort()}, Env: s.Env}})
	if err != nil {
		return provision.Ref{}, err
	}
	var p podJSON
	if err := decode(res, &p, "created pod"); err != nil {
		return provision.Ref{}, err
	}
	if p.ID == "" {
		return provision.Ref{}, provision.Errorf(provision.RunPod, provision.ClassUnavailable, res.Status, "RunPod answered without a pod id")
	}
	return provision.Ref{ID: p.ID, OfferID: s.OfferID, Label: s.Label}, nil
}

func podPath(id string) string { return "/v2/pods/" + url.PathEscape(id) }

// Get reads one pod. A deleted pod is ClassNotFound whether RunPod answers
// 404 or 200 TERMINATED (which one is UNCONFIRMED).
func (a *Adapter) Get(ctx context.Context, key string, ref provision.Ref) (provision.Instance, error) {
	if ref.ID == "" {
		return provision.Instance{}, provision.Errorf(provision.RunPod, provision.ClassInvalid, 0, "no pod id")
	}
	res, err := a.do(ctx, key, req{method: http.MethodGet, path: podPath(ref.ID), scope: "pods:read"})
	if err != nil {
		return provision.Instance{}, err
	}
	var p podJSON
	if err := decode(res, &p, "pod"); err != nil {
		return provision.Instance{}, err
	}
	if p.Status == "TERMINATED" {
		return provision.Instance{}, provision.Errorf(provision.RunPod, provision.ClassNotFound, res.Status, "pod %s is terminated", ref.ID)
	}
	a.learn(ctx, key, p)
	return a.instance(p), nil
}

// List pages through GET /v2/pods and keeps the pods whose name starts with
// labelPrefix (runtime is null in lists, which nothing here needs).
func (a *Adapter) List(ctx context.Context, key, labelPrefix string) ([]provision.Instance, error) {
	out := []provision.Instance{}
	cursor := ""
	for range maxPages {
		q := url.Values{"limit": {"1000"}}
		if cursor != "" {
			q.Set("cursor", cursor)
		}
		res, err := a.do(ctx, key, req{method: http.MethodGet, path: "/v2/pods", query: q, scope: "pods:read"})
		if err != nil {
			return nil, err
		}
		var body struct {
			Pods       []podJSON  `json:"pods"`
			Pagination pagination `json:"pagination"`
		}
		if err := decode(res, &body, "pod list"); err != nil {
			return nil, err
		}
		var mine []podJSON
		for _, p := range body.Pods {
			if strings.HasPrefix(p.Name, labelPrefix) {
				mine = append(mine, p)
			}
		}
		a.learn(ctx, key, mine...)
		for _, p := range mine {
			out = append(out, a.instance(p))
		}
		if !body.Pagination.HasNextPage || body.Pagination.NextCursor == nil || *body.Pagination.NextCursor == "" {
			break
		}
		cursor = *body.Pagination.NextCursor
	}
	return out, nil
}

// instance maps a v2 pod. The endpoint is the proxy URL built from the id
// (it is not a pod field); its scheme decides Secure, so a simulated node on
// http://127.0.0.1 is not marked https.
func (a *Adapter) instance(p podJSON) provision.Instance {
	st, detail := mapStatus(p.Status)
	inst := provision.Instance{ProviderID: p.ID, Label: p.Name, Status: st, ProviderStatus: p.Status, Detail: detail}
	if p.GPU != nil {
		inst.GPU, inst.VRAMGB = a.display(p.GPU.ID)
		inst.GPUCount = p.GPU.Count
	}
	if p.DataCenterID != nil {
		inst.Location = *p.DataCenterID
	}
	if u := strings.TrimRight(ProxyURL(p.ID), "/"); u != "" && st != provision.StatusTerminated {
		inst.Endpoints = []provision.Endpoint{{URL: u, Secure: strings.HasPrefix(u, "https://"), Via: "proxy"}}
	}
	if p.Cost != nil {
		cost := *p.Cost
		inst.PricePerHour = &cost
	}
	zero := 0.0
	inst.StoppedPricePerHour = &zero
	inst.Actions = []string{}
	for _, act := range p.Actions {
		if act == "stop" || act == "start" || act == "terminate" {
			inst.Actions = append(inst.Actions, act)
		}
	}
	if t, err := time.Parse(time.RFC3339, p.CreatedAt); err == nil {
		inst.CreatedAt = t
	}
	inst.Env = p.Env
	return inst
}

func (a *Adapter) action(ctx context.Context, key, id, action string) error {
	_, err := a.do(ctx, key, req{method: http.MethodPost, path: podPath(id) + "/action", body: map[string]string{"action": action}, scope: "pods:write"})
	return err
}

func (a *Adapter) Stop(ctx context.Context, key, id string) error {
	return a.action(ctx, key, id, "stop")
}

func (a *Adapter) Start(ctx context.Context, key, id string) error {
	return a.action(ctx, key, id, "start")
}

// Terminate deletes the pod; one that is already gone is not an error.
func (a *Adapter) Terminate(ctx context.Context, key, id string) error {
	_, err := a.do(ctx, key, req{method: http.MethodDelete, path: podPath(id), scope: "pods:write"})
	if provision.ClassOf(err) == provision.ClassNotFound {
		return nil
	}
	return err
}

// Logs reads the SSE log stream for at most LogsWait or tail lines and joins
// each event's line. The stream stays open on the live API, so running out
// of time is the normal end, not an error.
func (a *Adapter) Logs(ctx context.Context, key, id string, tail int) (string, error) {
	if tail <= 0 {
		tail = 100
	}
	tail = min(tail, 5000)
	ctx, cancel := context.WithTimeout(ctx, LogsWait)
	defer cancel()
	u := strings.TrimRight(API, "/") + podPath(id) + "/logs?tail=" + strconv.Itoa(tail)
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", provision.Errorf(provision.RunPod, provision.ClassInvalid, 0, "bad pod id")
	}
	hreq.Header.Set("Authorization", "Bearer "+key)
	hreq.Header.Set("Accept", "text/event-stream")
	res, err := provision.HTTPClient.Do(hreq)
	if err != nil {
		return "", provision.Errorf(provision.RunPod, provision.ClassUnavailable, 0, "%s", provision.Clip(provision.Redact(err.Error(), key), 300))
	}
	defer res.Body.Close()
	if res.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(res.Body, 64<<10))
		return "", apiError(provision.Response{Status: res.StatusCode, Header: res.Header, Body: body}, req{scope: "pods:read"}, []string{key})
	}
	sc := bufio.NewScanner(io.LimitReader(res.Body, 4<<20))
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	var lines []string
	for len(lines) < tail && sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		var ev struct {
			Line *string `json:"line"`
		}
		if json.Unmarshal([]byte(data), &ev) == nil && ev.Line != nil {
			data = *ev.Line
		}
		lines = append(lines, data)
	}
	return strings.Join(lines, "\n"), nil
}

func round4(x float64) float64 { return math.Round(x*1e4) / 1e4 }

var _ provision.Cloud = (*Adapter)(nil)
