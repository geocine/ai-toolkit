// Package quickpodmock is an httptest QuickPod built from quickpod.md (§8,
// the mock checklist) and checked against QuickPod's own source: the update
// API's answers and error shapes, the host agent's port_mappings and log
// upload, and the billing rules. Tests and simulate mode run the
// quickpod adapter against it. State is computed lazily from timestamps on
// each request, so a mockkit.FakeClock makes every transition deterministic.
package quickpodmock

import (
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/provision/mockkit"
)

// rentableJSON is the live /rentable sample (quickpod-fixture-rentable.json,
// public IPs scrubbed to 203.0.113.x) plus two synthetic rows: an RTX A6000
// on a fast host and an offline RTX 4090 that reports no bandwidth.
//
//go:embed testdata/rentable.json
var rentableJSON []byte

// liveJSON is the image-filter fixture (IMAGE-PLAN §7.3): the rest of the
// live 18-offer /rentable shape (max_cuda "13.1\n", "" and "13.0";
// driver_vers float-encoded or "0"; IPs scrubbed) plus a GTX 1660 SUPER row,
// a 30 GB max_disk_size row and a CUDA 12.8 (R570) row the image blocks, all
// made online and free.
//
//go:embed testdata/rentable_live.json
var liveJSON []byte

// LiveOffers are the rows of the image-filter fixture, for AddOffers.
func LiveOffers() []map[string]any {
	var rows []map[string]any
	if err := json.Unmarshal(liveJSON, &rows); err != nil {
		panic("quickpodmock: bad live fixture: " + err.Error())
	}
	return rows
}

// AddOffers lists more /rentable rows (LiveOffers) after the fixture's.
func (m *Mock) AddOffers(rows ...map[string]any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.offers = append(m.offers, rows...)
}

// Seeded templates (spec: PLAN §6.3 mock). The aitk template publishes 8675
// and carries a fallback token, as the setup guide tells users to do.
// /templates also lists other users' public templates, as QuickPod does.
const (
	// MeID is the account's user id (/update/api/me) and its templates' user_id.
	MeID = 1234
	// OtherUserID owns the public templates that are not the account's.
	OtherUserID       = 77
	AitkTemplateUUID  = "5e1f9f29-aitk-4c1e-9d6a-000000008675"
	OtherTemplateUUID = "17dc5ed1-69b8-4239-a678-31e8fa14d858"
	TemplateToken     = "template-token-0000"
	// AitkTemplateImage is the aitk template's image: the tag lives in
	// image_path (createpod never reads version_tag), so it follows :latest.
	AitkTemplateImage = "ghcr.io/geocine/aitoolkit-go:latest"
	// PinnedTemplateUUID is PinnedTemplate's id.
	PinnedTemplateUUID = "5e1f9f29-aitk-4c1e-9d6a-00000000f53a"
)

// PinnedTemplate is an aitk template pinned to build in its image path, as
// the studio's "change the tag in the Docker Image Path" fix leaves it
// (simulate seeds it with registrymock's revision).
func PinnedTemplate(build string) Template {
	return Template{UUID: PinnedTemplateUUID, Name: "aitk headless (pinned)", ImagePath: "ghcr.io/geocine/aitoolkit-go:" + build,
		DockerOptions: "-p 8675:8675 -e AI_TOOLKIT_AUTH=" + TemplateToken}
}

// minRuntimeMinutes is the credit rule of a create or start: 30 minutes of
// every running pod of the account plus the new one.
const minRuntimeMinutes = 30

// minDiskGB: createpod raises a smaller disk to 30 GB and bills it.
const minDiskGB = 30

type Template struct {
	UserID        int64 // 0 = the account's (MeID)
	UUID          string
	Name          string
	ImagePath     string
	VersionTag    string
	DockerOptions string
	Public        bool
}

// Pod is a test's view of one pod.
type Pod struct {
	UUID          string
	ID            int64
	Altname       string
	OfferID       int64
	TemplateUUID  string
	DiskSize      string // as the create body sent it (billed at least 30 GB)
	DockerOptions string // the per-pod docker_options as received
	Env           map[string]string
	Ports         bool // container port 8675 is published
	Image         string
	Intended      string
	State         string
	Destroyed     bool
	NodeURL       string
}

type pod struct {
	Pod
	owner  string
	faults mockkit.Faults
	offer  map[string]any
	// One pending Docker state change: from -> to, delay after at.
	from, to string
	at       time.Time
	delay    time.Duration
	seen     bool   // read as running once (CrashAfterRunning dies after it)
	logs     string // the tail the host agent uploaded after podlogs
	created  time.Time
	nodeStop func()
}

type Mock struct {
	Rec *mockkit.Recorder
	URL string
	// CreditStatus is the HTTP status of an insufficient_credit answer: 402,
	// as QuickPod sends it; tests also try 400.
	CreditStatus int

	o         mockkit.Options
	mu        sync.Mutex
	offers    []map[string]any
	occupied  map[int64]string // offer id -> pod uuid (or "other" for Occupy)
	pods      []*pod
	templates []Template
	credit    float64
	calls     int
	nextID    int64
	srv       *httptest.Server
}

func New(o mockkit.Options) *Mock {
	o.Defaults()
	var offers []map[string]any
	if err := json.Unmarshal(rentableJSON, &offers); err != nil {
		panic("quickpodmock: bad fixture: " + err.Error())
	}
	return &Mock{
		Rec:          &mockkit.Recorder{},
		CreditStatus: http.StatusPaymentRequired,
		o:            o,
		offers:       offers,
		occupied:     map[int64]string{},
		credit:       o.Credit,
		nextID:       41000,
		templates: []Template{
			{UUID: AitkTemplateUUID, Name: "aitk headless", ImagePath: AitkTemplateImage,
				DockerOptions: "-p 8675:8675 -e AI_TOOLKIT_AUTH=" + TemplateToken},
			{UUID: OtherTemplateUUID, Name: "FluxGym", ImagePath: "quickpod/fluxgym", VersionTag: "latest",
				DockerOptions: "-p 7860:7860", Public: true, UserID: OtherUserID},
		},
	}
}

// Start serves the mock on 127.0.0.1 and returns its URL.
func (m *Mock) Start() string {
	m.srv = httptest.NewServer(m.Handler())
	m.URL = m.srv.URL
	return m.URL
}

// Close stops the server and every node.
func (m *Mock) Close() {
	if m.srv != nil {
		m.srv.Close()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.pods {
		m.stopNode(p)
	}
}

func (m *Mock) SetCredit(c float64) { m.mu.Lock(); m.credit = c; m.mu.Unlock() }

// Occupy marks an offer as rented by someone else; Free releases it.
func (m *Mock) Occupy(offerID int64) { m.mu.Lock(); m.occupied[offerID] = "other"; m.mu.Unlock() }
func (m *Mock) Free(offerID int64)   { m.mu.Lock(); delete(m.occupied, offerID); m.mu.Unlock() }

// AddTemplate seeds another template for the account.
func (m *Mock) AddTemplate(t Template) {
	m.mu.Lock()
	m.templates = append(m.templates, t)
	m.mu.Unlock()
}

// Instances is every pod, with its Docker state as of now.
func (m *Mock) Instances() []Pod {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.o.Clock.Now()
	out := make([]Pod, 0, len(m.pods))
	for _, p := range m.pods {
		v := p.Pod
		v.State = m.state(p, now)
		v.Env = maps.Clone(p.Env)
		out = append(out, v)
	}
	return out
}

// ---- HTTP ----

func (m *Mock) Handler() http.Handler {
	mux := http.NewServeMux()
	// public (spec: quickpod.md §4)
	mux.HandleFunc("GET /rentable", m.rentable)
	mux.HandleFunc("GET /availablegputypes", m.gpuTypes)
	mux.HandleFunc("GET /update/offer_is_busy", m.offerBusy)
	mux.HandleFunc("GET /update/health_check", func(w http.ResponseWriter, _ *http.Request) {
		mockkit.WriteJSON(w, 200, map[string]string{"scope": "service_availability", "service": "quickpod-update-api", "status": "healthy"}, "")
	})
	mux.HandleFunc("GET /public_templates", m.publicTemplates)
	// templates (spec: quickpod.md §5.2)
	mux.HandleFunc("GET /templates", m.myTemplates)
	mux.HandleFunc("POST /update/templates", m.saveTemplate)
	mux.HandleFunc("GET /update/deleteTemplate/{uuid}", m.deleteTemplate)
	// pods (spec: quickpod.md §6)
	mux.HandleFunc("POST /update/api/createpod", m.createPod)
	mux.HandleFunc("GET /update/api/gpu_pods", m.listPods(false))
	mux.HandleFunc("GET /mypods", m.listPods(true))
	mux.HandleFunc("GET /update/api/me", m.me)
	for _, a := range []string{"stoppod", "startpod", "restartpod", "destroypod", "podlogs"} {
		mux.HandleFunc("GET /update/api/"+a, m.action(a))
	}
	return m.Rec.Wrap(cors(mux))
}

// cors mirrors the measured CORS (spec: quickpod.md §2): /update/* echoes
// the Origin and the requested headers with credentials; the read-only
// routes answer "*".
func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		if strings.HasPrefix(r.URL.Path, "/update/") {
			if o := r.Header.Get("Origin"); o != "" {
				h.Set("Access-Control-Allow-Origin", o)
				h.Set("Access-Control-Allow-Credentials", "true")
				h.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
				if rh := r.Header.Get("Access-Control-Request-Headers"); rh != "" {
					h.Set("Access-Control-Allow-Headers", rh)
				}
				h.Set("Access-Control-Max-Age", "86400")
			}
		} else {
			h.Set("Access-Control-Allow-Origin", "*")
			h.Set("Access-Control-Allow-Headers", "*")
			h.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// fail writes QuickPod's flat error object.
func fail(w http.ResponseWriter, status int, key, msg string) {
	mockkit.WriteJSON(w, status, map[string]string{key: msg}, "")
}

// credential reads the secure API key (X-API-Key or Authorization: ApiKey)
// and, where a route also takes it, a Bearer token.
func credential(r *http.Request) (key string, bearer bool) {
	if k := strings.TrimSpace(r.Header.Get("X-API-Key")); k != "" {
		return k, false
	}
	a := r.Header.Get("Authorization")
	if k, ok := strings.CutPrefix(a, "ApiKey "); ok {
		return strings.TrimSpace(k), false
	}
	if k, ok := strings.CutPrefix(a, "Bearer "); ok {
		return strings.TrimSpace(k), true
	}
	return "", false
}

// authed checks the credential (bearerOK: the route also takes Bearer) and
// the rate-limit fault. It answers the request itself when ok is false.
func (m *Mock) authed(w http.ResponseWriter, r *http.Request, bearerOK bool) (owner string, f mockkit.Faults, ok bool) {
	key, bearer := credential(r)
	if key == "" || (bearer && !bearerOK) {
		fail(w, http.StatusUnauthorized, "error", "Unauthorized")
		return "", f, false
	}
	f, ok = m.o.Authorize(key)
	if !ok {
		fail(w, http.StatusUnauthorized, "error", "Unauthorized")
		return "", f, false
	}
	m.mu.Lock()
	m.calls++
	n := m.calls
	m.mu.Unlock()
	if f.RateLimitEvery > 0 && n%f.RateLimitEvery == 0 {
		// QuickPod's API has no limiter of its own (one may sit at the edge);
		// the shape is ours.
		w.Header().Set("Retry-After", "2")
		fail(w, http.StatusTooManyRequests, "error", "too many requests")
		return "", f, false
	}
	// In simulate every key is the per-process secret plus a fault suffix,
	// so the pods belong to the secret and survive a fault-key change.
	if m.o.SimSecret != "" {
		return m.o.SimSecret, f, true
	}
	return key, f, true
}

// ---- offers ----

func num(m map[string]any, k string) float64 { f, _ := m[k].(float64); return f }
func str(m map[string]any, k string) string  { s, _ := m[k].(string); return s }
func machine(o map[string]any) map[string]any {
	mm, _ := o["_machines"].(map[string]any)
	return mm
}

func (m *Mock) offer(id int64) map[string]any {
	for _, o := range m.offers {
		if int64(num(o, "id")) == id {
			return o
		}
	}
	return nil
}

func (m *Mock) rentable(w http.ResponseWriter, _ *http.Request) {
	m.mu.Lock()
	out := make([]map[string]any, 0, len(m.offers))
	for _, o := range m.offers {
		if _, taken := m.occupied[int64(num(o, "id"))]; !taken {
			out = append(out, o) // occupied offers move to /notrentable
		}
	}
	m.mu.Unlock()
	mockkit.WriteJSON(w, 200, out, "")
}

func (m *Mock) gpuTypes(w http.ResponseWriter, _ *http.Request) {
	out := []map[string]string{{"gpu_type": "All GPUs"}}
	seen := map[string]bool{}
	for _, o := range m.offers {
		if t := str(o, "gpu_type"); !seen[t] {
			seen[t] = true
			out = append(out, map[string]string{"gpu_type": t})
		}
	}
	mockkit.WriteJSON(w, 200, out, "")
}

func (m *Mock) offerBusy(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.URL.Query().Get("offer_id"), 10, 64)
	if err != nil {
		fail(w, http.StatusBadRequest, "error", "offer_id is required")
		return
	}
	m.mu.Lock()
	_, taken := m.occupied[id]
	busy := taken || m.offer(id) == nil
	m.mu.Unlock()
	mockkit.WriteJSON(w, 200, busy, "")
}

// ---- templates ----

func templateJSON(t Template, withSecrets bool) map[string]any {
	uid := t.UserID
	if uid == 0 {
		uid = MeID
	}
	out := map[string]any{
		"user_id":       uid,
		"template_uuid": t.UUID, "template_name": t.Name, "template_type": "gpu", "launch_mode": "docker",
		"image_path": t.ImagePath, "version_tag": t.VersionTag, "docker_options": t.DockerOptions,
		"docker_run_options": "", "cli_command": "", "on_start_script": "", "launch_port": 0, "disk_space": 100,
		"docker_repo_server": "", "extra_filters": "", "env_tag": "", "template_image_url": "",
		"is_public": t.Public, "is_community": false,
	}
	if withSecrets {
		out["docker_repo_username"], out["docker_repo_password"] = "", ""
	}
	return out
}

func (m *Mock) publicTemplates(w http.ResponseWriter, _ *http.Request) {
	m.mu.Lock()
	out := []map[string]any{}
	for _, t := range m.templates {
		if t.Public {
			out = append(out, templateJSON(t, false))
		}
	}
	m.mu.Unlock()
	mockkit.WriteJSON(w, 200, out, "")
}

func (m *Mock) myTemplates(w http.ResponseWriter, r *http.Request) {
	if _, _, ok := m.authed(w, r, true); !ok {
		return
	}
	m.mu.Lock()
	out := []map[string]any{}
	for _, t := range m.templates {
		out = append(out, templateJSON(t, true))
	}
	m.mu.Unlock()
	mockkit.WriteJSON(w, 200, out, "")
}

// saveTemplate is Bearer-only (spec: quickpod.md §5.2): a secure API key
// gets 401, which is why the adapter never creates templates.
func (m *Mock) saveTemplate(w http.ResponseWriter, r *http.Request) {
	if _, bearer := credential(r); !bearer {
		fail(w, http.StatusUnauthorized, "error", "Unauthorized")
		return
	}
	if _, _, ok := m.authed(w, r, true); !ok {
		return
	}
	var in struct {
		UUID          string `json:"template_uuid"`
		Name          string `json:"template_name"`
		ImagePath     string `json:"image_path"`
		VersionTag    string `json:"version_tag"`
		DockerOptions string `json:"docker_options"`
		Public        bool   `json:"is_public"`
	}
	if json.NewDecoder(r.Body).Decode(&in) != nil {
		fail(w, http.StatusBadRequest, "error", "invalid request body")
		return
	}
	t := Template{UUID: in.UUID, Name: in.Name, ImagePath: in.ImagePath, VersionTag: in.VersionTag, DockerOptions: in.DockerOptions, Public: in.Public}
	m.mu.Lock()
	defer m.mu.Unlock()
	if t.UUID == "" {
		t.UUID = newUUID()
		m.templates = append(m.templates, t)
	} else {
		i := -1
		for j := range m.templates {
			if m.templates[j].UUID == t.UUID {
				i = j
			}
		}
		if i < 0 {
			fail(w, http.StatusNotFound, "error", "template not found")
			return
		}
		m.templates[i] = t
	}
	mockkit.WriteJSON(w, 200, templateJSON(t, false), "")
}

// deleteTemplate is a GET and Bearer-only, like saveTemplate (spec:
// quickpod.md §5.2).
func (m *Mock) deleteTemplate(w http.ResponseWriter, r *http.Request) {
	if _, bearer := credential(r); !bearer {
		fail(w, http.StatusUnauthorized, "error", "Unauthorized")
		return
	}
	if _, _, ok := m.authed(w, r, true); !ok {
		return
	}
	id := r.PathValue("uuid")
	m.mu.Lock()
	defer m.mu.Unlock()
	i := slices.IndexFunc(m.templates, func(t Template) bool { return t.UUID == id })
	if i < 0 {
		fail(w, http.StatusNotFound, "error", "template not found")
		return
	}
	m.templates = slices.Delete(m.templates, i, i+1)
	mockkit.WriteJSON(w, 200, map[string]string{"message": "Template deleted"}, "")
}

// ---- docker options ----

// dockerOpts reads "-e K=V" and "-p H:C" flags from a docker_options string
// as the host agent does: only the short "-e" and "-p" forms survive
// ("--env=", "--publish" and the rest are dropped).
func dockerOpts(s string) (env map[string]string, port8675 bool) {
	env = map[string]string{}
	f := strings.Fields(s)
	for i := 0; i+1 < len(f); i++ {
		switch f[i] {
		case "-e":
			i++
			if k, v, ok := strings.Cut(f[i], "="); ok {
				env[k] = v
			}
		case "-p":
			i++
			if parts := strings.Split(f[i], ":"); len(parts) >= 2 && strings.TrimSuffix(parts[1], "/tcp") == "8675" {
				port8675 = true
			}
		}
	}
	return env, port8675
}

// podEnv applies the per-pod docker_options to the template's. QuickPod
// appends them after the template's (later -e wins); the PodEnv fault's
// replace and ignore modes stay for the probe's no_port / token_mismatch
// tests.
func podEnv(mode mockkit.PodEnvMode, template, perPod string) (map[string]string, bool) {
	tEnv, tPort := dockerOpts(template)
	pEnv, pPort := dockerOpts(perPod)
	switch mode {
	case mockkit.PodEnvReplace:
		return pEnv, pPort
	case mockkit.PodEnvIgnore:
		return tEnv, tPort
	}
	maps.Copy(tEnv, pEnv) // append: later flags win
	return tEnv, tPort || pPort
}

// ---- pods ----

func newUUID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	h := hex.EncodeToString(b)
	return h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

func (m *Mock) template(uuid string) (Template, bool) {
	for _, t := range m.templates {
		if t.UUID == uuid {
			return t, true
		}
	}
	return Template{}, false
}

// storageCost is QuickPod's rule: listed_storage_cost (USD per TB-hour) x
// the billed disk / 1024.
func storageCost(o map[string]any, disk int) float64 {
	return num(machine(o), "listed_storage_cost") * float64(max(disk, minDiskGB)) / 1024
}

// runningCost is the hourly cost of the owner's running pods (but skip).
func (m *Mock) runningCost(owner string, skip *pod) float64 {
	sum := 0.0
	for _, p := range m.pods {
		if p.owner == owner && p != skip && !p.Destroyed && p.Intended == "running" {
			sum += p.hourly()
		}
	}
	return sum
}

// creditFail answers insufficient_credit (402 with numbers, as QuickPod
// sends it): the credit must cover 30 minutes of every running pod plus this
// one.
func (m *Mock) creditFail(w http.ResponseWriter, owner string, f mockkit.Faults, hourly float64, skip *pod, action string) bool {
	cur := f.Balance(m.credit)
	need := (m.runningCost(owner, skip) + hourly) * minRuntimeMinutes / 60
	if !f.NoCredit && cur >= need {
		return false
	}
	mockkit.WriteJSON(w, m.CreditStatus, map[string]any{
		"error":          "insufficient_credit",
		"message":        "Insufficient credit to " + action + " this pod",
		"current_credit": cur, "required_credit": need, "hourly_cost": hourly,
		"minimum_runtime_minutes": minRuntimeMinutes, "action": "Add credit to your account",
	}, "")
	return true
}

// dbFail is createpod's answer to any failure of its database step.
func dbFail(w http.ResponseWriter, msg string) {
	mockkit.WriteJSON(w, http.StatusNotAcceptable, map[string]string{"error": "Database query failed", "message": msg}, "")
}

func (m *Mock) createPod(w http.ResponseWriter, r *http.Request) {
	owner, f, ok := m.authed(w, r, false)
	if !ok {
		return
	}
	var in map[string]any
	if json.NewDecoder(r.Body).Decode(&in) != nil {
		fail(w, http.StatusBadRequest, "error", "invalid request body")
		return
	}
	tid := str(in, "template_uuid")
	if tid == "" {
		fail(w, http.StatusBadRequest, "error", "template_uuid is required")
		return
	}
	ds, isStr := in["disk_size"].(string)
	disk, err := strconv.Atoi(ds)
	if !isStr || err != nil || disk <= 0 {
		fail(w, http.StatusBadRequest, "error", "disk_size must be a string of GB")
		return
	}
	opts, _ := in["docker_options"].(string)
	m.mu.Lock()
	defer m.mu.Unlock()
	t, found := m.template(tid)
	if !found {
		fail(w, http.StatusBadRequest, "error", "template not found")
		return
	}
	oid := int64(num(in, "offers_id"))
	o := m.offer(oid)
	_, taken := m.occupied[oid]
	if o == nil {
		dbFail(w, "offer or machine not found")
		return
	}
	if taken || f.NoCapacity {
		dbFail(w, "offer is busy")
		return
	}
	if maxDisk := int(num(o, "max_disk_size")); maxDisk > 0 && disk > maxDisk {
		dbFail(w, fmt.Sprintf("requested disk size %dGB exceeds the maximum disk size %dGB", disk, maxDisk))
		return
	}
	if m.creditFail(w, owner, f, num(o, "hourly_cost")+storageCost(o, disk), nil, "create") {
		return
	}
	env, port := podEnv(f.PodEnv, t.DockerOptions, opts)
	if f.NoPortMapping {
		port = false
	}
	image := t.ImagePath // createpod runs image_path alone; version_tag is never read
	now := m.o.Clock.Now()
	m.nextID++
	p := &pod{
		Pod: Pod{
			UUID: newUUID(), ID: m.nextID, Altname: str(in, "altname"), OfferID: oid, TemplateUUID: tid,
			DiskSize: ds, DockerOptions: opts, Env: env, Ports: port, Image: image, Intended: "running",
		},
		owner: owner, faults: f, offer: o, created: now,
		from: "", to: "running", at: now, delay: f.Scale(m.o.Delays.Provision) + f.Scale(m.o.Delays.Start),
	}
	m.pods = append(m.pods, p)
	m.occupied[oid] = p.UUID
	start := int(num(o, "open_port_start"))
	mockkit.WriteJSON(w, 200, map[string]any{
		"status": "success", "pod_uuid": p.UUID, "public_ipaddress": str(machine(o), "public_ipaddr"),
		"open_port_start": start, "open_port_end": int(num(o, "open_port_end")),
	}, "")
}

// hourly is what the pod bills while running: compute plus storage
// (QuickPod's hourly_cost already includes storage_cost).
func (p *pod) hourly() float64 {
	disk, _ := strconv.Atoi(p.DiskSize)
	return num(p.offer, "hourly_cost") + storageCost(p.offer, disk)
}

// state is the pod's Docker State now (lazy, spec: PLAN §6 common rules).
func (m *Mock) state(p *pod, now time.Time) string {
	if p.Destroyed {
		return "exited"
	}
	s := p.from
	if mockkit.Phase(now, p.at, p.delay) >= 1 {
		s = p.to
	}
	if s == "running" && p.faults.CrashAfterRunning && p.seen {
		return "dead"
	}
	return s
}

// observe advances a pod for a read: it starts the node the first time the
// pod reads as running with 8675 published, and stops it otherwise.
func (m *Mock) observe(p *pod, now time.Time) string {
	s := m.state(p, now)
	if s == "running" {
		if p.nodeStop == nil && p.Ports && m.o.Node != nil {
			mib := num(p.offer, "gpu_vram")
			spec := mockkit.NodeSpec{
				Kind: "quickpod", InstanceID: p.UUID, Env: maps.Clone(p.Env),
				GPU: str(p.offer, "gpu_type"), GPUCount: int(num(p.offer, "num_gpus")), VRAMGB: int(mib/1024 + 0.5),
				AppDelay: p.faults.Scale(m.o.Delays.App), Image: p.Image,
			}.WithFaults(p.faults)
			p.NodeURL, p.nodeStop = m.o.Node(spec)
		}
		p.seen = true
	} else {
		m.stopNode(p)
	}
	return s
}

func (m *Mock) stopNode(p *pod) {
	if p.nodeStop != nil {
		p.nodeStop()
	}
	p.nodeStop, p.NodeURL = nil, ""
}

// portMappings is the field the host agent writes: one entry per -p option,
// the address as a link, host ports from open_port_start+1.
func portMappings(p *pod, state string) string {
	if state != "running" || !p.Ports {
		return ""
	}
	ip, port := str(machine(p.offer), "public_ipaddr"), int(num(p.offer, "open_port_start"))+1
	if p.NodeURL != "" {
		if h, ps, err := net.SplitHostPort(strings.TrimPrefix(p.NodeURL, "http://")); err == nil {
			ip, port = h, atoi(ps)
		}
	}
	return mapping(8675, ip, port)
}

func mapping(container int, ip string, port int) string {
	return fmt.Sprintf(`%d -> <a target="_new" href="http://%s:%d">%s:%d</a> |`, container, ip, port, ip, port)
}

func atoi(s string) int { n, _ := strconv.Atoi(s); return n }

func podJSON(p *pod, state string, now time.Time) map[string]any {
	mm := machine(p.offer)
	disk, _ := strconv.Atoi(p.DiskSize)
	storage, hourly := storageCost(p.offer, disk), p.hourly()
	status := ""
	switch state {
	case "running":
		status = "Up"
	case "exited", "dead":
		status = "Exited (0)"
	}
	return map[string]any{
		"id": p.ID, "Names": p.UUID, "altname": p.Altname, "State": state, "intended_state": p.Intended,
		"Status": status, "destroyed": p.Destroyed, "public_ipaddr": str(mm, "public_ipaddr"),
		"open_port_start": num(p.offer, "open_port_start"), "open_port_end": num(p.offer, "open_port_end"),
		"port_mappings": portMappings(p, state), "hourly_cost": hourly, "storage_cost": storage,
		"inet_cost": 0, "pod_cost": hourly * now.Sub(p.created).Hours(),
		"disk_size": max(disk, minDiskGB), "gpu_count": num(p.offer, "num_gpus"), "offers_id": p.OfferID, "machines_id": num(p.offer, "machines_id"),
		"created_at": p.created.UnixMilli(), "last_seen": now.UnixMilli(), "Image": p.Image,
		"launch_mode": "docker", "pod_type": "GPU", "docker_sshd": false, "logs": p.logs,
		"_machines": map[string]any{
			"id": num(mm, "id"), "online": mm["online"], "geoinfo": str(mm, "geoinfo"),
			"geolocation": str(mm, "geolocation"), "public_ipaddr": str(mm, "public_ipaddr"),
		},
		"_offers": map[string]any{
			"id": p.OfferID, "gpu_type": str(p.offer, "gpu_type"), "gpu_vram": num(p.offer, "gpu_vram"),
			"num_gpus": num(p.offer, "num_gpus"), "hourly_cost": num(p.offer, "hourly_cost"),
		},
	}
}

func (m *Mock) listPods(bearerOK bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		owner, _, ok := m.authed(w, r, bearerOK)
		if !ok {
			return
		}
		m.mu.Lock()
		now := m.o.Clock.Now()
		out := []map[string]any{}
		for _, p := range m.pods {
			if p.owner == owner && !p.Destroyed { // terminated pods are not listed
				out = append(out, podJSON(p, m.observe(p, now), now))
			}
		}
		m.mu.Unlock()
		mockkit.WriteJSON(w, 200, out, "")
	}
}

func (m *Mock) me(w http.ResponseWriter, r *http.Request) {
	_, f, ok := m.authed(w, r, false)
	if !ok {
		return
	}
	m.mu.Lock()
	c := f.Balance(m.credit)
	m.mu.Unlock()
	mockkit.WriteJSON(w, 200, map[string]any{
		"id": MeID, "name": "Sim", "email": "sim@example.com", "user_type": "user", "role": "user",
		"can_pay": true, "credit": c, "credit_limit": 0, "total_billed": 0, "email_verified": true,
	}, "")
}

// move starts a Docker state change from the state now.
func (m *Mock) move(p *pod, now time.Time, to string, delay time.Duration) {
	p.from, p.to, p.at, p.delay = m.state(p, now), to, now, delay
}

func (m *Mock) action(a string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		owner, f, ok := m.authed(w, r, false)
		if !ok {
			return
		}
		id := r.URL.Query().Get("pod_uuid")
		if id == "" {
			fail(w, http.StatusBadRequest, "error", "Missing pod_uuid parameter")
			return
		}
		m.mu.Lock()
		defer m.mu.Unlock()
		var p *pod
		for _, q := range m.pods {
			if q.UUID == id {
				p = q
			}
		}
		// startpod checks the offer before the pod's owner, as QuickPod does.
		if a == "startpod" && p != nil {
			if holder, taken := m.occupied[p.OfferID]; taken && holder != p.UUID {
				fail(w, http.StatusConflict, "error", "Offer is busy")
				return
			}
		}
		// A missing, destroyed or foreign pod reads as user 0: a 403 in
		// QuickPod's own words ("start" on every call but stop).
		if p == nil || p.Destroyed || p.owner != owner {
			verb := "start"
			if a == "stoppod" {
				verb = "stop"
			}
			fail(w, http.StatusForbidden, "error", "You do not have permission to "+verb+" this pod")
			return
		}
		now := m.o.Clock.Now()
		ok200 := func(msg string) {
			mockkit.WriteJSON(w, 200, map[string]string{"message": msg, "pod_uuid": p.UUID}, "")
		}
		switch a {
		case "stoppod":
			p.Intended = "exited"
			m.move(p, now, "exited", p.faults.Scale(m.o.Delays.Provision))
			if m.occupied[p.OfferID] == p.UUID {
				delete(m.occupied, p.OfferID) // the GPU is released; the disk stays
			}
			ok200("Pod stopped")
		case "startpod", "restartpod":
			if m.creditFail(w, owner, f, p.hourly(), p, "start") {
				return
			}
			p.Intended = "running"
			if a == "restartpod" {
				m.stopNode(p)
				p.from = "exited"
			} else {
				p.from = m.state(p, now)
			}
			p.to, p.at, p.delay, p.seen = "running", now, p.faults.Scale(m.o.Delays.Start), false
			m.occupied[p.OfferID] = p.UUID
			if a == "restartpod" {
				ok200("Pod restarted")
			} else {
				ok200("Pod started")
			}
		case "destroypod":
			p.Destroyed, p.Intended = true, "terminated"
			m.stopNode(p)
			if m.occupied[p.OfferID] == p.UUID {
				delete(m.occupied, p.OfferID)
			}
			ok200("Pod destroyed")
		case "podlogs":
			// podlogs only queues "docker logs"; the host agent uploads the
			// last lines joined with <br> (here at once) into the pod's logs.
			lines := []string{"Pulling image " + p.Image}
			if s := m.state(p, now); s == "running" || s == "exited" || s == "dead" {
				lines = append(lines, "Status: Downloaded newer image for "+p.Image, "aitk: listening on :8675")
			}
			p.logs = strings.Join(lines, "<br>")
			ok200("Pod logs fetched")
		}
	}
}
