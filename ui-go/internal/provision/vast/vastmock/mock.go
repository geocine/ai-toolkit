// Package vastmock is an httptest double of console.vast.ai for the Vast
// adapter's tests and for simulate mode (spec: vast.md §9). State changes are
// computed lazily from the mock clock on each request.
package vastmock

import (
	"crypto/rand"
	_ "embed"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/provision/mockkit"
)

// bundlesJSON: vastsrc/bundles_get.json (2 live RTX 4090 offers, IPs
// scrubbed to 203.0.113.x) plus synthetic rows: RTX 3090, RTX A6000,
// 2x RTX 4090, one with direct_port_count 0, one already rented, and one
// slow host (73 Mbps, 150 disk_bw).
//
//go:embed testdata/bundles.json
var bundlesJSON []byte

const (
	userID        = 123456
	keylessCap    = 64  // keyless searches return at most 64 rows (IMG §0.2)
	keyedCap      = 200 // MASTER §3.9.2
	firstContract = 30000001
)

// Mock is one in-memory Vast account.
type Mock struct {
	URL string // set by Start
	Rec *mockkit.Recorder
	// PageSize is the GET /api/v1/instances/ page cap (Vast: 25). Tests lower
	// it to exercise next_token.
	PageSize int

	o   mockkit.Options
	srv *httptest.Server

	mu        sync.Mutex
	offers    []map[string]any
	insts     map[int64]*inst
	next      int64
	credit    float64
	calls     int
	logs      map[string]*logFile
	templates []map[string]any
}

type inst struct {
	id        int64
	offer     map[string]any // the rented offer row, as it was
	label     string
	image     string
	runtype   string
	env       [][2]string // in arrival order, docker flags included
	disk      float64
	apiKey    string
	created   time.Time
	since     time.Time // when the current start began
	restarted bool      // started again after a stop: goes straight to loading
	intended  string    // "running" | "stopped"
	destroyed bool
	faults    mockkit.Faults

	// materialized when the instance first reads as running
	ip       string
	port     string
	lastPort string // a restart never reuses it
	nodeStop func()
}

type logFile struct {
	text string
	gets int
}

// Instance is a test snapshot of one instance.
type Instance struct {
	ID        int64
	Label     string
	Image     string
	RunType   string
	Env       map[string]string // docker flags included, as sent
	Disk      float64
	Intended  string
	Actual    string // "" while provisioning
	HostPort  string
	Destroyed bool
}

// New builds a mock over the fixture offers.
func New(o mockkit.Options) *Mock {
	o.Defaults()
	m := &Mock{
		Rec:      &mockkit.Recorder{},
		PageSize: 25,
		o:        o,
		insts:    map[int64]*inst{},
		next:     firstContract,
		credit:   o.Credit,
		logs:     map[string]*logFile{},
	}
	var fx struct {
		Offers []map[string]any `json:"offers"`
	}
	if err := json.Unmarshal(bundlesJSON, &fx); err != nil {
		panic("vastmock: bad fixture: " + err.Error())
	}
	m.offers = fx.Offers
	m.templates = []map[string]any{
		{
			"id": 334501, "hash_id": "5915f1dc1ce881defb572015eb9d8178", "creator_id": userID,
			"name": "aitk headless", "image": "ghcr.io/geocine/aitoolkit-go", "tag": "latest",
			"env":     "-p 8675:8675 -e OPEN_BUTTON_PORT=8675 -e AI_TOOLKIT_AUTH=template-token-0000",
			"runtype": "args", "args_str": "/start.sh", "private": true, "recommended_disk_space": 100,
		},
		{
			"id": 100200, "hash_id": "0c1d2e3f405162738495a6b7c8d9e0f1", "creator_id": 1,
			"name": "Vast base image", "image": "vastai/base-image", "tag": "@vastai-automatic-tag",
			"env": "-p 1111:1111 -p 8080:8080", "runtype": "ssh", "private": false,
		},
	}
	return m
}

// Handler serves the mock API (recorded).
func (m *Mock) Handler() http.Handler { return m.Rec.Wrap(http.HandlerFunc(m.serve)) }

// Start listens on 127.0.0.1:0 and returns the base URL.
func (m *Mock) Start() string {
	m.srv = httptest.NewServer(m.Handler())
	m.URL = m.srv.URL
	return m.URL
}

// Close stops the server and every node.
func (m *Mock) Close() {
	m.mu.Lock()
	for _, in := range m.insts {
		in.stopNode()
	}
	m.mu.Unlock()
	if m.srv != nil {
		m.srv.Close()
	}
}

// SetCredit sets the account credit (<= 0 makes creates fail with missing_billing).
func (m *Mock) SetCredit(c float64) {
	m.mu.Lock()
	m.credit = c
	m.mu.Unlock()
}

// Instances returns every instance created so far, destroyed ones included.
func (m *Mock) Instances() []Instance {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.o.Clock.Now()
	ids := slices.Sorted(maps.Keys(m.insts))
	out := make([]Instance, 0, len(ids))
	for _, id := range ids {
		in := m.insts[id]
		actual := m.advance(in, now)
		env := map[string]string{}
		for _, p := range in.env {
			env[p[0]] = p[1]
		}
		out = append(out, Instance{ID: in.id, Label: in.label, Image: in.image, RunType: in.runtype, Env: env,
			Disk: in.disk, Intended: in.intended, Actual: actual, HostPort: in.port, Destroyed: in.destroyed})
	}
	return out
}

// Rented reports whether a create has taken the offer.
func (m *Mock) Rented(offerID int64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if row := m.offer(offerID); row != nil {
		return row["rented"] == true
	}
	return false
}

// ---- routing ----

var (
	reAsk      = regexp.MustCompile(`^/api/v0/asks/(\d+)$`)
	reInstance = regexp.MustCompile(`^/api/v0/instances/([^/]+)$`)
	reLogs     = regexp.MustCompile(`^/api/v0/instances/request_logs/([^/]+)$`)
	reLogFile  = regexp.MustCompile(`^/logs/([0-9a-f]+)\.log$`)
)

func (m *Mock) serve(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	if len(path) > 1 {
		path = strings.TrimSuffix(path, "/") // both slash forms are accepted (vast.md §1.1)
	}
	if r.Method == http.MethodOptions {
		// spec: vast.md §1.4: 204 with allow lists and no Access-Control-Allow-Origin.
		h := w.Header()
		h.Set("Access-Control-Allow-Methods", "GET,HEAD,POST,PUT,DELETE,OPTIONS,PATCH")
		h.Set("Access-Control-Allow-Headers", "authorization,content-type")
		h.Set("Access-Control-Max-Age", "86400")
		m.rateHeaders(h, 500)
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if mm := reLogFile.FindStringSubmatch(path); mm != nil && r.Method == http.MethodGet {
		m.logFile(w, mm[1]) // the S3 link needs no key
		return
	}
	keyless := path == "/api/v0/bundles" && r.Method == http.MethodGet && presentedKey(r) == ""
	var f mockkit.Faults
	if !keyless {
		var ok bool
		if f, ok = m.authorize(w, r); !ok {
			return
		}
	}
	switch {
	case path == "/api/v0/users/current" && r.Method == http.MethodGet:
		m.user(w, r)
	case path == "/api/v0/bundles" && (r.Method == http.MethodPost || r.Method == http.MethodGet):
		m.bundles(w, r, keyless)
	case reAsk.MatchString(path) && r.Method == http.MethodPut:
		id, _ := strconv.ParseInt(reAsk.FindStringSubmatch(path)[1], 10, 64)
		m.create(w, r, id, f)
	case path == "/api/v1/instances" && r.Method == http.MethodGet:
		m.listV1(w, r)
	case path == "/api/v0/instances" && r.Method == http.MethodGet:
		m.listV0(w)
	case reLogs.MatchString(path) && r.Method == http.MethodPut:
		m.requestLogs(w, r, reLogs.FindStringSubmatch(path)[1])
	case reInstance.MatchString(path):
		m.instance(w, r, reInstance.FindStringSubmatch(path)[1], f)
	case path == "/api/v0/template" && r.Method == http.MethodGet:
		m.searchTemplates(w, r)
	case path == "/api/v0/template" && r.Method == http.MethodPut:
		fail(w, http.StatusGone, "gone", "This endpoint is deprecated; use PUT /api/v1/template/")
	default:
		fail(w, http.StatusNotFound, "not_found", "no such endpoint: "+r.Method+" "+r.URL.Path)
	}
}

func presentedKey(r *http.Request) string {
	if k, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer "); ok {
		return strings.TrimSpace(k)
	}
	return r.URL.Query().Get("api_key")
}

// authorize checks the key (Bearer or ?api_key=) and applies RateLimitEvery.
func (m *Mock) authorize(w http.ResponseWriter, r *http.Request) (mockkit.Faults, bool) {
	f, ok := m.o.Authorize(presentedKey(r))
	if !ok {
		// spec: vast.md §1.3
		fail(w, http.StatusUnauthorized, "auth_error", "Invalid user key")
		return f, false
	}
	m.mu.Lock()
	m.calls++
	n := m.calls
	m.mu.Unlock()
	if f.RateLimitEvery > 0 && n%f.RateLimitEvery == 0 {
		// spec: vast.md §1.5: often plain text, with X-RateLimit-* headers.
		m.rateHeaders(w.Header(), 30)
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, "API requests too frequent endpoint threshold=1.0")
		return f, false
	}
	return f, true
}

func (m *Mock) rateHeaders(h http.Header, limit int) {
	h.Set("X-RateLimit-Limit", strconv.Itoa(limit))
	h.Set("X-RateLimit-Remaining", "0")
	h.Set("X-RateLimit-Reset", strconv.FormatInt(m.o.Clock.Now().Add(2*time.Second).Unix(), 10))
}

func fail(w http.ResponseWriter, status int, code, msg string) {
	mockkit.WriteJSON(w, status, map[string]any{"success": false, "error": code, "msg": msg}, "")
}

func ok(w http.ResponseWriter, v map[string]any) {
	v["success"] = true
	mockkit.WriteJSON(w, http.StatusOK, v, "")
}

func readJSON(r *http.Request, v any) error {
	b, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// ---- account ----

func (m *Mock) user(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	credit := m.credit
	m.mu.Unlock()
	// spec: vast.md §2. The raw answer carries api_key; the adapter must drop it.
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{
		"id": userID, "key_id": 789, "email": "you@example.com", "credit": credit, "balance": 0.0,
		"ssh_key": "", "sid": "mock-session", "api_key": presentedKey(r), "can_pay": credit > 0, "has_billing": true,
	}, "")
}

// ---- offer search ----

var searchTypes = []string{"on-demand", "ondemand", "bid", "reserved", "interruptible"}

func (m *Mock) bundles(w http.ResponseWriter, r *http.Request, keyless bool) {
	q := map[string]any{}
	var err error
	if r.Method == http.MethodGet {
		if s := r.URL.Query().Get("q"); s != "" {
			err = json.Unmarshal([]byte(s), &q)
		}
	} else {
		err = readJSON(r, &q)
	}
	if err != nil || q == nil {
		fail(w, http.StatusBadRequest, "invalid_request", "Invalid json body")
		return
	}
	if t, has := q["type"]; has {
		if s, _ := t.(string); !slices.Contains(searchTypes, s) {
			fail(w, http.StatusBadRequest, "invalid_request", fmt.Sprintf("invalid type %v", t))
			return
		}
	}
	storage := 8.0 // what Vast prices when allocated_storage is absent (vast.md §3.3)
	if v, has := q["allocated_storage"]; has {
		if n, isNum := v.(float64); isNum && n > 0 {
			storage = n
		}
	}
	limit := keylessCap
	if v, isNum := q["limit"].(float64); isNum && v > 0 {
		limit = int(v)
	}
	if keyless {
		limit = min(limit, keylessCap)
	} else {
		limit = min(limit, keyedCap)
	}

	m.mu.Lock()
	rows := make([]map[string]any, 0, len(m.offers))
	for _, o := range m.offers {
		rows = append(rows, priced(o, storage))
	}
	m.mu.Unlock()

	var out []map[string]any
	for _, row := range rows {
		match, ferr := matches(row, q)
		if ferr != nil {
			fail(w, http.StatusBadRequest, "invalid_request", ferr.Error())
			return
		}
		if match {
			out = append(out, row)
		}
	}
	if err := sortRows(out, q["order"]); err != nil {
		fail(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if len(out) > limit {
		out = out[:limit]
	}
	if out == nil {
		out = []map[string]any{}
	}
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"offers": out, "truncated": false}, "")
}

// priced copies an offer row with dph_total recomputed for the storage size:
// dph_base + storage_cost * GB / 720 (vast.md §3.3).
func priced(o map[string]any, storage float64) map[string]any {
	row := maps.Clone(o)
	base, _ := row["dph_base"].(float64)
	cost, _ := row["storage_cost"].(float64)
	disk := cost * storage / 720
	row["storage_total_cost"] = disk
	row["dph_total"] = base + disk
	row["search"] = map[string]any{"gpuCostPerHour": base, "diskHour": disk, "totalHour": base + disk, "discountTotalHour": 0, "discountedTotalPerHour": base + disk}
	row["instance"] = map[string]any{"gpuCostPerHour": 0, "diskHour": disk, "totalHour": disk, "discountTotalHour": 0, "discountedTotalPerHour": disk}
	return row
}

var notFilters = []string{"order", "type", "limit", "allocated_storage"}

// matches applies every {"field": {"op": value}} filter in q to row.
func matches(row, q map[string]any) (bool, error) {
	for field, cond := range q {
		if slices.Contains(notFilters, field) {
			continue
		}
		ops, isMap := cond.(map[string]any)
		if !isMap {
			return false, fmt.Errorf("%s: a filter must be an object of operators", field)
		}
		val, has := row[field]
		if !has && field == "verified" { // a search key, not a row column
			val = row["verification"] == "verified"
		}
		for op, arg := range ops {
			hit, err := compare(val, op, arg)
			if err != nil {
				return false, fmt.Errorf("%s: %w", field, err)
			}
			if !hit {
				return false, nil
			}
		}
	}
	return true, nil
}

// compare implements the 8 documented operators. A null column equals false
// (live rows carry external:null and still match external {eq:false}).
func compare(val any, op string, arg any) (bool, error) {
	switch op {
	case "eq":
		return equal(val, arg), nil
	case "neq":
		return !equal(val, arg), nil
	case "in", "notin":
		list, isList := arg.([]any)
		if !isList {
			return false, fmt.Errorf("%s needs a list", op)
		}
		hit := slices.ContainsFunc(list, func(a any) bool { return equal(val, a) })
		return hit == (op == "in"), nil
	case "gt", "lt", "gte", "lte":
		a, aok := number(val)
		b, bok := number(arg)
		if !aok || !bok {
			return false, nil
		}
		switch op {
		case "gt":
			return a > b, nil
		case "lt":
			return a < b, nil
		case "gte":
			return a >= b, nil
		default:
			return a <= b, nil
		}
	}
	return false, fmt.Errorf("unknown operator %q", op)
}

func equal(a, b any) bool {
	if a == nil {
		a = false
	}
	if b == nil {
		b = false
	}
	if x, ok := number(a); ok {
		y, ok := number(b)
		return ok && x == y
	}
	return a == b
}

func number(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}

func sortRows(rows []map[string]any, order any) error {
	if order == nil {
		return nil
	}
	list, isList := order.([]any)
	if !isList {
		return fmt.Errorf("order must be a list of [field, direction]")
	}
	type key struct {
		field string
		desc  bool
	}
	var keys []key
	for _, item := range list {
		pair, _ := item.([]any)
		if len(pair) == 0 {
			return fmt.Errorf("order must be a list of [field, direction]")
		}
		f, _ := pair[0].(string)
		dir := "asc"
		if len(pair) > 1 {
			dir, _ = pair[1].(string)
		}
		if f == "" || (dir != "asc" && dir != "desc") {
			return fmt.Errorf("invalid order %v", item)
		}
		keys = append(keys, key{f, dir == "desc"})
	}
	sort.SliceStable(rows, func(i, j int) bool {
		for _, k := range keys {
			a, _ := number(rows[i][k.field])
			b, _ := number(rows[j][k.field])
			if a != b {
				return (a < b) != k.desc
			}
		}
		return false
	})
	return nil
}

func (m *Mock) offer(id int64) map[string]any {
	for _, o := range m.offers {
		if n, _ := number(o["id"]); int64(n) == id {
			return o
		}
	}
	return nil
}

// ---- create ----

var (
	runTypes = []string{"ssh", "jupyter", "args", "ssh_proxy", "ssh_direct", "jupyter_proxy", "jupyter_direct",
		"ssh_direc", "jupyter_direc"} // the last two are vast-cli's legacy spellings
	rePortFlag = regexp.MustCompile(`^-p (\d+):(\d+)(/(tcp|udp))?$`)
)

func (m *Mock) create(w http.ResponseWriter, r *http.Request, offerID int64, f mockkit.Faults) {
	// spec: vast.md §1.3 (asks rows) and §4.1.
	var body map[string]any
	if err := readJSON(r, &body); err != nil || body == nil {
		fail(w, http.StatusBadRequest, "invalid_args", "error 400/1782: Invalid args: invalid json")
		return
	}
	notAvail := func(status int, code, msg string) {
		mockkit.WriteJSON(w, status, map[string]any{"success": false, "error": code, "msg": msg, "ask_id": offerID}, "")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	row := m.offer(offerID)
	if row == nil {
		notAvail(http.StatusNotFound, "invalid_args", fmt.Sprintf("error 404/3603: no_such_ask Instance type by id %d is not available.", offerID))
		return
	}
	cancel, _ := body["cancel_unavail"].(bool)
	if row["rented"] == true || f.NoCapacity {
		if cancel || f.NoCapacity {
			notAvail(http.StatusGone, "no_such_ask", fmt.Sprintf("error 410/3907: no_such_ask Instance type %d is no longer available.", offerID))
		} else {
			notAvail(http.StatusNotFound, "invalid_args", fmt.Sprintf("error 404/3603: no_such_ask Instance type by id %d is not available.", offerID))
		}
		return
	}
	image, _ := body["image"].(string)
	if image == "" && body["template_hash_id"] == nil {
		notAvail(http.StatusBadRequest, "invalid_args", "error 400/3467: Invalid args: image is required")
		return
	}
	if p, has := body["price"]; has {
		if n, isNum := number(p); !isNum || n < 0.001 || n > 128 {
			notAvail(http.StatusBadRequest, "invalid_price", "error 400: invalid bid price")
			return
		}
	}
	label, _ := body["label"].(string)
	if len(label) > 256 {
		notAvail(http.StatusBadRequest, "invalid_args", "error 400: label is longer than 256 characters")
		return
	}
	runtype, _ := body["runtype"].(string)
	if runtype == "" {
		runtype = "ssh"
		if body["args"] != nil || body["args_str"] != nil {
			runtype = "args"
		}
	}
	for _, part := range strings.Fields(runtype) {
		if !slices.Contains(runTypes, part) {
			notAvail(http.StatusBadRequest, "invalid_args", "error 400: invalid runtype "+runtype)
			return
		}
	}
	disk := 8.0
	if v, has := body["disk"]; has {
		n, isNum := number(v)
		if !isNum || n < 8 {
			notAvail(http.StatusBadRequest, "invalid_args", "error 400: disk must be at least 8 GB")
			return
		}
		disk = n
	}
	if space, _ := number(row["disk_space"]); space > 0 && disk > space {
		notAvail(http.StatusBadRequest, "invalid_args", fmt.Sprintf("error 400: disk %.0f GB exceeds the %.0f GB this host offers", disk, space))
		return
	}
	env, msg := parseEnv(body["env"])
	if msg != "" {
		notAvail(http.StatusBadRequest, "invalid_args", "error 400: "+msg)
		return
	}
	if f.NoCredit || m.credit <= 0 {
		// UNCONFIRMED: the error code for a create without credit.
		notAvail(http.StatusBadRequest, "missing_billing", "error 400: missing_billing Your account has no credit; add some at cloud.vast.ai/billing")
		return
	}
	now := m.o.Clock.Now()
	id := m.next
	m.next++
	in := &inst{id: id, offer: maps.Clone(row), label: label, image: image, runtype: runtype, env: env, disk: disk,
		apiKey: randHex(32), created: now, since: now, intended: "running", faults: f}
	if state, _ := body["target_state"].(string); state == "stopped" {
		in.intended = "stopped"
	}
	m.insts[id] = in
	row["rented"], row["rentable"] = true, false
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"success": true, "new_contract": id, "instance_api_key": in.apiKey}, "")
}

// parseEnv accepts the dict form (vast-cli, the guide) and tolerates the
// docker-flag string the OpenAPI text describes. Values are capped at 256
// and the whole at 32768 characters (vast.md §4.1).
func parseEnv(v any) ([][2]string, string) {
	var out [][2]string
	switch e := v.(type) {
	case nil:
	case map[string]any:
		for _, k := range slices.Sorted(maps.Keys(e)) {
			s, isStr := e[k].(string)
			if !isStr {
				b, _ := json.Marshal(e[k])
				s = string(b)
			}
			out = append(out, [2]string{k, s})
		}
	case string:
		f := strings.Fields(e)
		for i := 0; i < len(f); i++ {
			switch {
			case f[i] == "-e" && i+1 < len(f):
				k, val, _ := strings.Cut(f[i+1], "=")
				out = append(out, [2]string{k, val})
				i++
			case f[i] == "-p" && i+1 < len(f):
				out = append(out, [2]string{"-p " + f[i+1], "1"})
				i++
			}
		}
	default:
		return nil, "env must be an object"
	}
	total := 0
	for _, p := range out {
		if len(p[1]) > 256 {
			return nil, "env value of " + p[0] + " is longer than 256 characters"
		}
		total += len(p[0]) + len(p[1])
	}
	if total > 32768 {
		return nil, "env is larger than 32768 characters"
	}
	return out, ""
}

func randHex(n int) string {
	b := make([]byte, n/2)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// ---- lifecycle ----

// advance brings in up to now and returns actual_status ("" = null). The
// first start goes null -> loading -> running; a restart goes loading ->
// running on a new port. Caller holds m.mu.
func (m *Mock) advance(in *inst, now time.Time) string {
	if in.destroyed {
		return ""
	}
	if in.intended == "stopped" {
		in.stopNode()
		return "exited"
	}
	f := in.faults
	steps := []time.Duration{f.Scale(m.o.Delays.Provision), f.Scale(m.o.Delays.Start)}
	phase := 0
	if in.restarted {
		phase = 1 + mockkit.Phase(now, in.since, steps[1])
	} else {
		phase = mockkit.Phase(now, in.since, steps...)
	}
	switch phase {
	case 0:
		return ""
	case 1:
		return "loading"
	}
	if f.CrashAfterRunning {
		ran := in.since.Add(steps[1])
		if !in.restarted {
			ran = ran.Add(steps[0])
		}
		if !now.Before(ran.Add(f.Scale(max(m.o.Delays.Start, time.Second)))) {
			in.stopNode()
			return "exited"
		}
	}
	if in.ip == "" {
		m.materialize(in)
	}
	return "running"
}

// materialize gives a running instance its address: the simulated node's
// when a NodeFactory is set, else a documentation address and a random port.
func (m *Mock) materialize(in *inst) {
	in.ip, _ = in.offer["public_ipaddr"].(string)
	if in.ip == "" {
		in.ip = "203.0.113.10"
	}
	for in.port == "" || in.port == in.lastPort {
		in.port = strconv.Itoa(20000 + int(randUint(45001)))
	}
	if m.o.Node == nil {
		return
	}
	gpus, _ := number(in.offer["num_gpus"])
	ram, _ := number(in.offer["gpu_ram"])
	gpu, _ := in.offer["gpu_name"].(string)
	env := map[string]string{
		"CONTAINER_ID":        strconv.FormatInt(in.id, 10),
		"CONTAINER_API_KEY":   in.apiKey,
		"VAST_CONTAINERLABEL": "C." + strconv.FormatInt(in.id, 10),
		"GPU_COUNT":           strconv.Itoa(int(gpus)),
	}
	for _, p := range in.env {
		if !strings.HasPrefix(p[0], "-") {
			env[p[0]] = p[1]
		}
	}
	spec := mockkit.NodeSpec{Kind: "vast", InstanceID: strconv.FormatInt(in.id, 10), Env: env, GPU: gpu,
		GPUCount: int(gpus), VRAMGB: int(ram/1024 + 0.5), AppDelay: in.faults.Scale(m.o.Delays.App), Image: in.image}
	base, stop := m.o.Node(spec.WithFaults(in.faults))
	u, err := url.Parse(base)
	if err != nil {
		stop()
		return
	}
	in.ip, in.port, in.nodeStop = u.Hostname(), u.Port(), stop
	in.lastPort = in.port
}

func (in *inst) stopNode() {
	if in.nodeStop != nil {
		in.nodeStop()
		in.nodeStop = nil
	}
	if in.port != "" {
		in.lastPort = in.port
	}
	in.ip, in.port = "", ""
}

func randUint(n uint64) uint64 {
	var b [8]byte
	_, _ = rand.Read(b[:])
	var v uint64
	for _, x := range b {
		v = v<<8 | uint64(x)
	}
	return v % n
}

// view renders the instance object of vast.md §5.1. Caller holds m.mu.
func (m *Mock) view(in *inst, now time.Time) map[string]any {
	actual := m.advance(in, now)
	base, _ := number(in.offer["dph_base"])
	cost, _ := number(in.offer["storage_cost"])
	disk := cost * in.disk / 720
	v := map[string]any{
		"id": in.id, "label": in.label, "intended_status": in.intended, "cur_state": in.intended, "next_state": in.intended,
		"actual_status": nil, "status_msg": "", "template_id": nil, "template_hash_id": nil, "template_name": nil,
		"image_uuid": in.image, "image_runtype": in.runtype, "image_args": []string{}, "onstart": nil,
		"public_ipaddr": in.offer["public_ipaddr"], "direct_port_count": in.offer["direct_port_count"],
		"static_ip": in.offer["static_ip"], "machine_id": in.offer["machine_id"], "host_id": in.offer["host_id"],
		"bundle_id": in.offer["bundle_id"], "start_date": float64(in.created.UnixMilli()) / 1000,
		"gpu_name": in.offer["gpu_name"], "num_gpus": in.offer["num_gpus"], "gpu_ram": in.offer["gpu_ram"],
		"gpu_totalram": in.offer["gpu_total_ram"], "cpu_name": in.offer["cpu_name"], "cpu_cores": in.offer["cpu_cores"],
		"disk_space": in.disk, "geolocation": in.offer["geolocation"], "verification": in.offer["verification"],
		"reliability2": in.offer["reliability2"], "is_bid": false, "dph_base": base, "dph_total": base + disk,
		"storage_cost": cost, "storage_total_cost": disk,
		"search":      map[string]any{"gpuCostPerHour": base, "diskHour": disk, "totalHour": base + disk, "discountTotalHour": 0, "discountedTotalPerHour": base + disk},
		"instance":    map[string]any{"gpuCostPerHour": 0, "diskHour": disk, "totalHour": disk, "discountTotalHour": 0, "discountedTotalPerHour": disk},
		"volume_info": []any{},
	}
	if strings.Contains(in.runtype, "args") {
		v["image_args"] = []string{"/start.sh"}
	}
	pairs := make([][]string, 0, len(in.env))
	for _, p := range in.env {
		pairs = append(pairs, []string{p[0], p[1]})
	}
	v["extra_env"] = pairs
	if actual != "" {
		v["actual_status"] = actual
	}
	switch actual {
	case "loading":
		v["status_msg"] = "Pulling image " + in.image
		v["ports"] = nil
	case "running":
		v["status_msg"] = "success, running " + strings.ReplaceAll(in.image, ":", "_") + "/" + in.runtype
		v["public_ipaddr"] = in.ip
		ports := map[string]any{}
		if !in.faults.NoPortMapping {
			for _, p := range in.env {
				if mm := rePortFlag.FindStringSubmatch(p[0]); mm != nil {
					hostPort := in.port
					if mm[2] != "8675" {
						hostPort = strconv.Itoa(20000 + int(randUint(45001)))
					}
					proto := mm[4]
					if proto == "" {
						proto = "tcp"
					}
					ports[mm[2]+"/"+proto] = []map[string]string{{"HostIp": "0.0.0.0", "HostPort": hostPort}}
				}
			}
		}
		v["ports"] = ports
	case "exited":
		if in.intended == "running" {
			v["status_msg"] = "Error: container exited with code 1"
		}
	}
	return v
}

func (m *Mock) live(idText string) (*inst, bool) {
	id, err := strconv.ParseInt(idText, 10, 64)
	if err != nil {
		return nil, false
	}
	in, has := m.insts[id]
	return in, has && !in.destroyed
}

func (m *Mock) instance(w http.ResponseWriter, r *http.Request, idText string, f mockkit.Faults) {
	if _, err := strconv.ParseInt(idText, 10, 64); err != nil {
		fail(w, http.StatusBadRequest, "invalid_args", "error 400/1782: Invalid args: invalid instance_id")
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.o.Clock.Now()
	in, live := m.live(idText)
	switch r.Method {
	case http.MethodGet:
		// spec: vast.md §5.1: one object under "instances", null when gone.
		if !live {
			mockkit.WriteJSON(w, http.StatusOK, map[string]any{"instances": nil}, "")
			return
		}
		mockkit.WriteJSON(w, http.StatusOK, map[string]any{"instances": m.view(in, now)}, "")
	case http.MethodPut:
		var body map[string]any
		if err := readJSON(r, &body); err != nil || body == nil {
			fail(w, http.StatusBadRequest, "invalid_args", "error 400/1782: Invalid args: invalid json")
			return
		}
		if !live {
			fail(w, http.StatusNotFound, "no_such_instance", "Instance "+idText+" not found")
			return
		}
		if l, has := body["label"]; has {
			s, _ := l.(string)
			if len(s) > 1024 {
				fail(w, http.StatusBadRequest, "label_too_long", "label is longer than 1024 characters")
				return
			}
			in.label = s
		}
		if st, has := body["state"]; has {
			switch st {
			case "stopped":
				in.intended = "stopped"
				in.stopNode()
			case "running":
				if f.NoCredit || m.credit <= 0 {
					fail(w, http.StatusBadRequest, "missing_billing", "Your account has no credit; add some at cloud.vast.ai/billing")
					return
				}
				if in.intended == "stopped" {
					in.intended, in.since, in.restarted = "running", now, true
				}
			default:
				fail(w, http.StatusBadRequest, "invalid_args", fmt.Sprintf("invalid state %v", st))
				return
			}
		}
		ok(w, map[string]any{})
	case http.MethodDelete:
		if !live {
			fail(w, http.StatusNotFound, "not_found", "Instance not found")
			return
		}
		in.stopNode()
		in.destroyed = true
		ok(w, map[string]any{})
	default:
		fail(w, http.StatusMethodNotAllowed, "invalid_request", "method not allowed")
	}
}

// listV1 pages by id with a base64 next_token (vast.md §5.3).
func (m *Mock) listV1(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit := m.PageSize
	if s := q.Get("limit"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil {
			fail(w, http.StatusBadRequest, "invalid_request", "Error in query parameters: limit")
			return
		}
		if n <= 0 {
			n = 5
		}
		limit = min(n, m.PageSize)
	}
	after := int64(0)
	if tok := q.Get("after_token"); tok != "" {
		var t struct {
			Values struct {
				ID int64 `json:"id"`
			} `json:"values"`
		}
		b, err := base64.StdEncoding.DecodeString(tok)
		if err != nil || json.Unmarshal(b, &t) != nil {
			fail(w, http.StatusBadRequest, "invalid_token", "Error in query parameters: after_token")
			return
		}
		after = t.Values.ID
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.o.Clock.Now()
	var ids []int64
	for id, in := range m.insts {
		if !in.destroyed {
			ids = append(ids, id)
		}
	}
	slices.Sort(ids)
	total := len(ids)
	page := []map[string]any{}
	var next any
	for i, id := range ids {
		if id <= after {
			continue
		}
		if len(page) == limit {
			b, _ := json.Marshal(map[string]any{"values": map[string]any{"id": ids[i-1]}})
			next = base64.StdEncoding.EncodeToString(b)
			break
		}
		page = append(page, m.view(m.insts[id], now))
	}
	ok(w, map[string]any{"instances_found": len(page), "total_instances": total, "label_counts": map[string]int{},
		"next_token": next, "instances": page})
}

func (m *Mock) listV0(w http.ResponseWriter) {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.o.Clock.Now()
	page := []map[string]any{}
	for _, id := range slices.Sorted(maps.Keys(m.insts)) {
		if in := m.insts[id]; !in.destroyed {
			page = append(page, m.view(in, now))
		}
	}
	mockkit.WriteJSON(w, http.StatusOK, map[string]any{"instances_found": len(page), "instances": page}, "")
}

// ---- logs ----

func (m *Mock) requestLogs(w http.ResponseWriter, r *http.Request, idText string) {
	var body struct {
		Tail string `json:"tail"`
	}
	_ = readJSON(r, &body)
	m.mu.Lock()
	defer m.mu.Unlock()
	in, live := m.live(idText)
	if !live {
		mockkit.WriteJSON(w, http.StatusNotFound, map[string]any{"error": "invalid_id", "msg": "Invalid instance id."}, "")
		return
	}
	lines := []string{
		"Pulling image " + in.image,
		"Status: Downloaded newer image for " + in.image,
		"Starting /start.sh",
		"aitk server listening on :8675",
	}
	if n, err := strconv.Atoi(body.Tail); err == nil && n > 0 && n < len(lines) {
		lines = lines[len(lines)-n:]
	}
	hash := randHex(32)
	m.logs[hash] = &logFile{text: strings.Join(lines, "\n") + "\n"}
	// spec: vast.md §5.6: the real link is on s3.amazonaws.com; the mock serves it itself.
	ok(w, map[string]any{"result_url": "http://" + r.Host + "/logs/" + hash + ".log", "msg": "Logs are being prepared"})
}

func (m *Mock) logFile(w http.ResponseWriter, hash string) {
	m.mu.Lock()
	lf := m.logs[hash]
	if lf != nil {
		lf.gets++
	}
	m.mu.Unlock()
	if lf == nil || lf.gets == 1 { // the upload is asynchronous: the first GET misses
		w.Header().Set("Content-Type", "application/xml")
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, "<Error><Code>NoSuchKey</Code></Error>")
		return
	}
	w.Header().Set("Content-Type", "text/plain")
	_, _ = io.WriteString(w, lf.text)
}

// ---- templates ----

func (m *Mock) searchTemplates(w http.ResponseWriter, r *http.Request) {
	filters := map[string]any{}
	if s := r.URL.Query().Get("select_filters"); s != "" {
		if err := json.Unmarshal([]byte(s), &filters); err != nil {
			fail(w, http.StatusBadRequest, "invalid_request", "select_filters is not valid JSON")
			return
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	out := []map[string]any{}
	for _, t := range m.templates {
		row := map[string]any{}
		for k, v := range t {
			if n, isNum := number(v); isNum {
				v = n
			}
			row[k] = v
		}
		match, err := matches(row, filters)
		if err != nil {
			fail(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
		if match {
			out = append(out, t)
		}
	}
	ok(w, map[string]any{"templates_found": len(out), "templates": out})
}
