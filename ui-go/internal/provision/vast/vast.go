// Package vast is the Vast.ai adapter, written only from Vast's published API
// (spec: scratchpad provision/vast.md; vast-cli is the ground truth for
// request shapes). vastmock is its httptest double.
package vast

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/provision"
)

var (
	// API is the Vast base. Only tests and the in-process simulate wiring change it.
	API = "https://console.vast.ai"
	// LogsHostAllowed admits the host (URL.Host, with a port when the URL
	// names one) of a request_logs result_url before it is fetched without
	// auth. Simulate points it at the mock.
	LogsHostAllowed = func(host string) bool {
		return host == "s3.amazonaws.com" || strings.HasSuffix(host, ".amazonaws.com")
	}
	// Pace spaces calls per endpoint (limiter.go). Tests swap it for one on a
	// fake clock.
	Pace = NewLimiter(nil)
	// searchMaxWait is the longest an offer search queues for a search slot
	// before it answers rate_limited with the wait: below the recommender's
	// per-provider budget (gpupick.PerProviderTimeout, 20 s), so a full
	// window reads "busy, try again in N s" instead of a timeout.
	searchMaxWait = 15 * time.Second
)

func init() { provision.Register(adapter{}) }

type adapter struct{}

func (adapter) Kind() provision.Kind { return provision.Vast }

func (adapter) Info() provision.KindInfo {
	return provision.KindInfo{
		Kind:  provision.Vast,
		Label: provision.Label(provision.Vast),
		Stage: "stable",
		Caps: provision.Caps{
			StopStart:      true,
			StopKeepsDisk:  true,
			HTTPS:          "no",
			SelfDestroy:    "builtin",
			Credit:         true,
			EnvValueMax:    256,
			EnvTotalMax:    32768,
			ImageDigestRef: "unconfirmed",
			ReportsInet:    true,
			ReportsDiskBW:  true,
		},
		Unconfirmed: []string{
			"Whether actual_status reports scheduling or stopped",
			"Exact permission names for scoped keys",
			"Whether an account environment variable or the instance's own env wins for AI_TOOLKIT_AUTH",
		},
		ConsoleURL: "https://cloud.vast.ai/instances/",
		KeysURL:    "https://cloud.vast.ai/manage-keys/",
		KeyScopes:  []string{"user_read", "instance_read", "instance_write"},
	}
}

// ---- units ----

// vramFilterMB is the gpu_ram floor sent for a per-GPU VRAM in GB. Vast
// reports gpu_ram in MiB labelled "MB" (a 24 GB RTX 4090 is 24564), and
// vast-cli multiplies GB by 1000, so 24 GB asks for >= 24000 and the 24564
// card matches. Lenient on purpose.
func vramFilterMB(gb int) int { return gb * 1000 }

// vramGB is the display GB of a gpu_ram value: 24564 MiB -> 24.
func vramGB(mib float64) int { return int(math.Round(mib / 1024)) }

// location tidies Vast's geolocation, which leaves the region empty on some
// hosts (", CN").
func location(geo string) string { return strings.Trim(geo, ", ") }

// ---- transport ----

// ops change how a status maps to a class (PLAN §6.2 errors).
const (
	opAsks   = "asks"   // PUT /asks/{id}/: 403 is "not your own", 404/410 is capacity
	opSearch = "search" // POST /bundles/: a bad or missing key answers 404 (vast.md §1.3)
	opAny    = ""
)

type envelope struct {
	Success *bool  `json:"success"`
	Error   string `json:"error"`
	Msg     string `json:"msg"`
	Message string `json:"message"`
	Detail  string `json:"detail"`
}

// call paces, sends and checks one Vast request, decoding a 2xx body into
// out. Every 2xx is checked for success:false.
func call(ctx context.Context, ep Endpoint, op, key, method, path string, query url.Values, body, out any) error {
	if ep == EpSearch {
		busy, err := Pace.WaitAtMost(ctx, ep, searchMaxWait)
		if err != nil {
			return &provision.Error{Provider: provision.Vast, Class: provision.ClassUnavailable, Message: err.Error()}
		}
		if busy > 0 {
			return &provision.Error{Provider: provision.Vast, Class: provision.ClassRateLimited, RetryAfter: busy.Round(time.Second) + time.Second,
				Message: "Vast allows 10 offer searches a minute and this server has used them", SearchOnly: true}
		}
	} else if err := Pace.Wait(ctx, ep); err != nil {
		return &provision.Error{Provider: provision.Vast, Class: provision.ClassUnavailable, Message: err.Error()}
	}
	u := strings.TrimRight(API, "/") + path
	if len(query) > 0 {
		u += "?" + query.Encode()
	}
	res, err := provision.Do(ctx, provision.Call{
		Kind: provision.Vast, Method: method, URL: u, Body: body, Secret: key,
		Header: map[string]string{"Authorization": "Bearer " + key},
	})
	if err != nil {
		return err
	}
	if res.Status < 200 || res.Status > 299 {
		return classify(op, res, key)
	}
	var env envelope
	if json.Unmarshal(res.Body, &env) == nil && env.Success != nil && !*env.Success {
		return classify(op, res, key)
	}
	if out != nil {
		if err := json.Unmarshal(res.Body, out); err != nil {
			return provision.Errorf(provision.Vast, provision.ClassUnavailable, res.Status, "unreadable answer from %s", path)
		}
	}
	return nil
}

// classify maps a Vast error answer (a non-2xx, or a 2xx with success:false)
// onto an adapter error. The message is msg || message || error || detail,
// or the plain-text body, redacted and clipped.
func classify(op string, res provision.Response, key string) error {
	var env envelope
	msg := ""
	if json.Unmarshal(res.Body, &env) == nil {
		msg = firstNonEmpty(env.Msg, env.Message, env.Error, env.Detail)
	}
	if msg == "" {
		msg = strings.TrimSpace(string(res.Body))
	}
	if msg == "" {
		msg = http.StatusText(res.Status)
	}
	e := &provision.Error{Provider: provision.Vast, Status: res.Status, Message: provision.Clip(provision.Redact(msg, key), 300)}
	code := env.Error
	switch {
	case code == "search_quota_exceeded":
		e.Class = provision.ClassRateLimited
		e.Message = "Vast's daily search quota is used up until UTC midnight"
		e.RetryAfter = untilUTCMidnight(time.Now())
		e.SearchOnly = true // the instance endpoints keep working: the watcher goes on
	case res.Status == http.StatusTooManyRequests || code == "rate_limit_exceeded":
		e.Class = provision.ClassRateLimited
		e.RetryAfter = provision.RetryAfter(res.Header, time.Now())
	case res.Status == http.StatusUnauthorized || code == "auth_error",
		res.Status == http.StatusNotFound && (op == opSearch || strings.Contains(strings.ToLower(msg), "invalid user key")):
		e.Class = provision.ClassAuth
	case code == "missing_billing":
		e.Class = provision.ClassCredit
	case op == opAsks && res.Status == http.StatusForbidden:
		e.Class = provision.ClassInvalid // "Offer N is not your own"
	case op == opAsks && (res.Status == http.StatusNotFound || res.Status == http.StatusGone ||
		code == "no_such_ask" || strings.Contains(msg, "no_such_ask")):
		e.Class = provision.ClassCapacity
	case res.Status == http.StatusForbidden:
		e.Class = provision.ClassAuth
		e.Message = provision.Clip("the key lacks a permission (user_read, instance_read, instance_write): "+e.Message, 300)
	case code == "no_such_instance" || code == "not_found" || res.Status == http.StatusNotFound || res.Status == http.StatusGone:
		e.Class = provision.ClassNotFound
	case code == "invalid_args" || code == "invalid_price" || code == "invalid_request":
		e.Class = provision.ClassInvalid
	case res.Status >= 500:
		e.Class = provision.ClassUnavailable
	default:
		e.Class = provision.ClassInvalid
	}
	return e
}

func untilUTCMidnight(now time.Time) time.Duration {
	now = now.UTC()
	next := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
	return next.Sub(now)
}

func firstNonEmpty(s ...string) string {
	for _, v := range s {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func parseID(kind, id string) (int64, error) {
	n, err := strconv.ParseInt(strings.TrimSpace(id), 10, 64)
	if err != nil || n <= 0 {
		return 0, provision.Errorf(provision.Vast, provision.ClassInvalid, 0, "%q is not a Vast %s id", provision.Clip(id, 40), kind)
	}
	return n, nil
}

// ---- account ----

type user struct {
	ID     int64    `json:"id"`
	Credit *float64 `json:"credit"`
	// api_key is in the raw answer and deliberately not decoded.
}

func currentUser(ctx context.Context, key string) (user, error) {
	var u user
	err := call(ctx, EpUser, opAny, key, http.MethodGet, "/api/v0/users/current/", nil, nil, &u)
	return u, err
}

func (adapter) Verify(ctx context.Context, key string) (provision.Account, error) {
	u, err := currentUser(ctx, key)
	if err != nil {
		return provision.Account{}, err
	}
	acc := provision.Account{Credit: u.Credit}
	if u.Credit != nil {
		acc.Detail = fmt.Sprintf("$%.2f credit", *u.Credit)
	}
	return acc, nil
}

// ---- offers ----

type bundle struct {
	ID          int64    `json:"id"`
	GPUName     string   `json:"gpu_name"`
	NumGPUs     int      `json:"num_gpus"`
	GPURAM      float64  `json:"gpu_ram"`
	DPHTotal    float64  `json:"dph_total"`
	Reliability *float64 `json:"reliability"`
	Geolocation string   `json:"geolocation"`
	DiskSpace   *float64 `json:"disk_space"`
	Rented      *bool    `json:"rented"`
	Instance    struct {
		TotalHour *float64 `json:"totalHour"`
	} `json:"instance"`
}

func (b bundle) offer() provision.Offer {
	o := provision.Offer{
		ID:                  strconv.FormatInt(b.ID, 10),
		Provider:            provision.Vast,
		GPU:                 b.GPUName,
		GPUCount:            max(b.NumGPUs, 1),
		VRAMGB:              vramGB(b.GPURAM),
		PricePerHour:        b.DPHTotal,
		StoppedPricePerHour: b.Instance.TotalHour,
		Location:            location(b.Geolocation),
		Reliability:         b.Reliability,
		Availability:        "available",
		HTTPS:               false,
	}
	if b.DiskSpace != nil {
		d := int(math.Floor(*b.DiskSpace))
		o.MaxDiskGB = &d
	}
	return o
}

const (
	defaultLimit = 50
	maxLimit     = 200
)

func limitOf(q provision.OfferQuery) int {
	if q.Limit <= 0 {
		return defaultLimit
	}
	return min(q.Limit, maxLimit)
}

func diskOf(q provision.OfferQuery) int {
	if q.DiskGB <= 0 {
		return provision.DefaultDiskGB
	}
	return q.DiskGB
}

// baseFilters are the host conditions every search and quote sends: a
// verified, rentable, free host with at least one direct port (8675 needs a
// mapped external port).
func baseFilters(q provision.OfferQuery) map[string]any {
	return map[string]any{
		"verified":          map[string]any{"eq": true},
		"rentable":          map[string]any{"eq": true},
		"rented":            map[string]any{"eq": false},
		"external":          map[string]any{"eq": false},
		"direct_port_count": map[string]any{"gte": 1},
		"order":             [][]string{{"dph_total", "asc"}},
		"type":              "on-demand",
		"allocated_storage": diskOf(q), // dph_total then prices the disk the user will create
	}
}

// searchBody is the POST /bundles/ body for q (PLAN §6.2 with MASTER §3.9.2).
// A GPU text search widens the page to 200, because Vast has no substring
// operator on gpu_name and the match runs client-side.
func searchBody(q provision.OfferQuery) map[string]any {
	b := baseFilters(q)
	b["num_gpus"] = map[string]any{"eq": max(q.GPUCount, 1)}
	ram := map[string]any{"gte": vramFilterMB(max(q.MinVRAMGB, 0))}
	if q.MaxVRAMGB > 0 {
		ram["lt"] = q.MaxVRAMGB * 1024 // exclusive cap, in MiB
	}
	b["gpu_ram"] = ram
	if q.MaxPrice > 0 {
		b["dph_total"] = map[string]any{"lte": q.MaxPrice}
	}
	if q.Fast {
		// Server-side here, so a host that reports 0 is left out too (D-15
		// applies to the client-side filters of the other providers).
		b["inet_down"] = map[string]any{"gte": hostfit.FastInetDownMbps}
		b["disk_bw"] = map[string]any{"gte": hostfit.FastDiskBWMBps}
	}
	if strings.TrimSpace(q.GPU) != "" {
		b["limit"] = maxLimit
	} else {
		b["limit"] = limitOf(q)
	}
	return b
}

// row is one search result: the fields the offer needs, and the raw row
// hostfit reads the host facts from.
type row struct {
	bundle
	raw map[string]any
}

// offer is the row's offer, decorated with its host and fit for q.
func (r row) offer(q provision.OfferQuery) provision.Offer {
	o := r.bundle.offer()
	provision.DecorateOffer(&o, hostfit.FromVast(r.raw), q)
	return o
}

func search(ctx context.Context, key string, body map[string]any) ([]row, error) {
	var out struct {
		Offers []json.RawMessage `json:"offers"`
	}
	if err := call(ctx, EpSearch, opSearch, key, http.MethodPost, "/api/v0/bundles/", nil, body, &out); err != nil {
		return nil, err
	}
	rows := make([]row, 0, len(out.Offers))
	for _, b := range out.Offers {
		var r row
		if json.Unmarshal(b, &r.bundle) != nil || json.Unmarshal(b, &r.raw) != nil {
			return nil, provision.Errorf(provision.Vast, provision.ClassUnavailable, 200, "unreadable offer in the search answer")
		}
		rows = append(rows, r)
	}
	return rows, nil
}

// queryBody is searchBody with the service's image filter merged in
// (hostfit.VastQuery: cuda_max_good, compute_cap, cpu_arch, gpu_arch,
// disk_space and, with a model or Fast, gpu_ram and the bandwidth floors;
// never driver_vers).
func queryBody(q provision.OfferQuery) map[string]any {
	b := searchBody(q)
	if q.Filter != nil {
		b = hostfit.VastQuery(*q.Filter, b)
	}
	return b
}

// Offers searches with the image filter server-side, then applies the GPU
// substring, the filter again client-side (Vast's answer is trusted for
// nothing the image needs) and the limit.
func (adapter) Offers(ctx context.Context, key string, q provision.OfferQuery) ([]provision.Offer, error) {
	rows, err := search(ctx, key, queryBody(q))
	if err != nil {
		return nil, err
	}
	gpu := strings.ToLower(strings.TrimSpace(q.GPU))
	limit := limitOf(q)
	out := make([]provision.Offer, 0, min(len(rows), limit))
	for _, r := range rows {
		if r.Rented != nil && *r.Rented { // vast-cli filters rented rows client-side too
			continue
		}
		if gpu != "" && !strings.Contains(strings.ToLower(r.GPUName), gpu) {
			continue
		}
		if !provision.KeepOffer(hostfit.FromVast(r.raw), q) {
			continue
		}
		out = append(out, r.offer(q))
		if len(out) == limit {
			break
		}
	}
	return out, nil
}

// Quote re-reads one offer with the base filters plus its id, priced for
// q.DiskGB. The query fields the offer id already fixes (GPU count, VRAM,
// price ceiling) are not sent, so a 2x offer quotes as itself.
func (adapter) Quote(ctx context.Context, key, offerID string, q provision.OfferQuery) (provision.Offer, error) {
	id, err := parseID("offer", offerID)
	if err != nil {
		return provision.Offer{}, err
	}
	body := baseFilters(q)
	body["id"] = map[string]any{"eq": id}
	body["limit"] = 1
	rows, err := search(ctx, key, body)
	if err != nil {
		return provision.Offer{}, err
	}
	for _, r := range rows {
		if r.ID == id && (r.Rented == nil || !*r.Rented) {
			return r.offer(q), nil
		}
	}
	return provision.Offer{}, provision.Errorf(provision.Vast, provision.ClassCapacity, 0, "offer %d is no longer available", id)
}

// ---- templates ----

type template struct {
	ID     int64           `json:"id"`
	HashID string          `json:"hash_id"`
	Name   string          `json:"name"`
	Image  string          `json:"image"`
	Tag    string          `json:"tag"`
	Env    json.RawMessage `json:"env"` // a docker-flag string in templates ("-p 8675:8675 -e K=V")
}

func (adapter) Templates(ctx context.Context, key string) ([]provision.Template, error) {
	u, err := currentUser(ctx, key)
	if err != nil {
		return nil, err
	}
	filters, _ := json.Marshal(map[string]any{"creator_id": map[string]any{"eq": u.ID}})
	var out struct {
		Templates []template `json:"templates"`
	}
	q := url.Values{"select_filters": {string(filters)}}
	if err := call(ctx, EpTemplates, opAny, key, http.MethodGet, "/api/v0/template/", q, nil, &out); err != nil {
		return nil, err
	}
	res := make([]provision.Template, 0, len(out.Templates))
	for _, t := range out.Templates {
		var env string
		if json.Unmarshal(t.Env, &env) != nil {
			env = string(t.Env)
		}
		image := t.Image
		if t.Tag != "" && !strings.Contains(image[strings.LastIndex(image, "/")+1:], ":") {
			image += ":" + t.Tag
		}
		id := t.HashID
		if id == "" {
			id = strconv.FormatInt(t.ID, 10)
		}
		res = append(res, provision.Template{
			ID:            id,
			Name:          t.Name,
			Image:         image,
			Ours:          strings.Contains(image, "aitoolkit-go"),
			PublishesPort: strings.Contains(env, strconv.Itoa(provision.Port)),
			HasToken:      strings.Contains(env, "AI_TOOLKIT_AUTH="),
		})
	}
	return res, nil
}

// ---- lifecycle ----

// portFlag asks Vast for a mapped external port for 8675 (vast.md §4.2).
var portFlag = fmt.Sprintf("-p %d:%d", provision.Port, provision.Port)

func (adapter) Launch(ctx context.Context, key string, s provision.LaunchSpec) (provision.Ref, error) {
	offer, err := parseID("offer", s.OfferID)
	if err != nil {
		return provision.Ref{}, err
	}
	env := map[string]string{portFlag: "1"}
	for k, v := range s.Env {
		env[k] = v
	}
	body := map[string]any{
		"client_id":      "me",
		"image":          s.Image,
		"label":          s.Label,
		"disk":           s.DiskGB,
		"runtype":        "args",
		"args_str":       "/start.sh", // the image has CMD /start.sh and no ENTRYPOINT
		"env":            env,
		"cancel_unavail": true, // fail with 410 instead of creating a stuck "scheduling" instance
	}
	var out struct {
		NewContract int64 `json:"new_contract"` // the instance id; instance_api_key is not kept
	}
	path := "/api/v0/asks/" + strconv.FormatInt(offer, 10) + "/"
	if err := call(ctx, EpCreate, opAsks, key, http.MethodPut, path, nil, body, &out); err != nil {
		return provision.Ref{}, err
	}
	if out.NewContract <= 0 {
		return provision.Ref{}, provision.Errorf(provision.Vast, provision.ClassUnavailable, 200, "the create answer had no new_contract")
	}
	return provision.Ref{ID: strconv.FormatInt(out.NewContract, 10), OfferID: s.OfferID, Label: s.Label}, nil
}

func instancePath(id int64) string { return "/api/v0/instances/" + strconv.FormatInt(id, 10) + "/" }

func (adapter) Get(ctx context.Context, key string, ref provision.Ref) (provision.Instance, error) {
	id, err := parseID("instance", ref.ID)
	if err != nil {
		return provision.Instance{}, err
	}
	var out struct {
		Instances *instance `json:"instances"` // one object despite the name; null when gone
	}
	if err := call(ctx, EpShow, opAny, key, http.MethodGet, instancePath(id), url.Values{"owner": {"me"}}, nil, &out); err != nil {
		return provision.Instance{}, err
	}
	if out.Instances == nil {
		return provision.Instance{}, provision.Errorf(provision.Vast, provision.ClassNotFound, 200, "instance %d not found or no longer exists", id)
	}
	return out.Instances.toInstance(), nil
}

const maxListPages = 400 // 10 000 instances; a loop guard, not a product limit

func (adapter) List(ctx context.Context, key, labelPrefix string) ([]provision.Instance, error) {
	var res []provision.Instance
	after := ""
	for range maxListPages {
		q := url.Values{"limit": {"25"}}
		if after != "" {
			q.Set("after_token", after)
		}
		var page struct {
			Instances []instance `json:"instances"`
			NextToken *string    `json:"next_token"`
		}
		if err := call(ctx, EpList, opAny, key, http.MethodGet, "/api/v1/instances/", q, nil, &page); err != nil {
			return nil, err
		}
		for _, in := range page.Instances {
			if strings.HasPrefix(in.Label, labelPrefix) {
				res = append(res, in.toInstance())
			}
		}
		if page.NextToken == nil || *page.NextToken == "" || *page.NextToken == after {
			return res, nil
		}
		after = *page.NextToken
	}
	return res, nil
}

func setState(ctx context.Context, key, id, state string) error {
	n, err := parseID("instance", id)
	if err != nil {
		return err
	}
	return call(ctx, EpManage, opAny, key, http.MethodPut, instancePath(n), nil, map[string]string{"state": state}, nil)
}

func (adapter) Stop(ctx context.Context, key, id string) error {
	return setState(ctx, key, id, "stopped")
}
func (adapter) Start(ctx context.Context, key, id string) error {
	return setState(ctx, key, id, "running")
}

func (adapter) Terminate(ctx context.Context, key, id string) error {
	n, err := parseID("instance", id)
	if err != nil {
		return err
	}
	err = call(ctx, EpDestroy, opAny, key, http.MethodDelete, instancePath(n), nil, map[string]any{}, nil)
	if provision.ClassOf(err) == provision.ClassNotFound {
		return nil
	}
	return err
}

// ---- logs ----

const (
	logsMax        = 4 << 20
	logsTries      = 10
	logsRetryDelay = 500 * time.Millisecond
)

// logsURLAllowed: https on an allowed host, or plain http on an allowed
// loopback host (only the in-process mock, which LogsHostAllowed admits in
// simulate).
func logsURLAllowed(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil || u.User != nil || u.Host == "" || !LogsHostAllowed(u.Host) {
		return false
	}
	if u.Scheme == "https" {
		return true
	}
	ip := net.ParseIP(u.Hostname())
	return u.Scheme == "http" && ip != nil && ip.IsLoopback()
}

func (adapter) Logs(ctx context.Context, key, id string, tail int) (string, error) {
	n, err := parseID("instance", id)
	if err != nil {
		return "", err
	}
	var out struct {
		ResultURL string `json:"result_url"`
	}
	path := "/api/v0/instances/request_logs/" + strconv.FormatInt(n, 10) + "/"
	if err := call(ctx, EpManage, opAny, key, http.MethodPut, path, nil, map[string]string{"tail": strconv.Itoa(max(tail, 1))}, &out); err != nil {
		return "", err
	}
	if !logsURLAllowed(out.ResultURL) {
		return "", provision.Errorf(provision.Vast, provision.ClassUnavailable, 200, "the logs link points at a host this server does not fetch from")
	}
	// The upload is asynchronous: the link answers 404 for a few seconds.
	for try := 0; ; try++ {
		res, err := provision.Do(ctx, provision.Call{Kind: provision.Vast, Method: http.MethodGet, URL: out.ResultURL, Max: logsMax, Secret: key})
		if err != nil {
			return "", err
		}
		switch {
		case res.Status == http.StatusOK:
			return string(res.Body), nil
		case res.Status == http.StatusNotFound && try+1 < logsTries:
			if err := Pace.clock.Sleep(ctx, logsRetryDelay); err != nil {
				return "", provision.Errorf(provision.Vast, provision.ClassUnavailable, 0, "%v", err)
			}
		default:
			return "", provision.Errorf(provision.Vast, provision.ClassUnavailable, res.Status, "the logs were not ready (HTTP %d)", res.Status)
		}
	}
}
