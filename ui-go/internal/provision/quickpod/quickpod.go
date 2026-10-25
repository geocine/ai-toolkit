// Package quickpod is the QuickPod adapter. It was written from QuickPod's
// published Swagger, CLIs and console bundle (quickpod.md), checked against
// QuickPod's own source (the update API, host agent and proxy), and is
// tested against quickpodmock.
//
// Pods always come from a template (the image, ports and env live there),
// create answers with the pod_uuid (a pod is also found again by offers_id +
// altname when that answer was lost), and every lifecycle call is a GET with
// ?pod_uuid=<Names>.
package quickpod

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/provision"
)

// Package vars: tests and the in-process simulate wiring change them.
var (
	API = "https://api.quickpod.org"
	// ProxyURL is QuickPod's https proxy for port 8675 of a pod: the first
	// segment of Names before "-". The proxy serves any published port of a
	// running pod and forwards Authorization unchanged.
	ProxyURL = func(names string) string {
		seg, _, _ := strings.Cut(names, "-")
		return "https://" + seg + "-8675.proxy.quickpod.io"
	}
	// UseProxy offers the proxy endpoint before the direct one; the probe
	// falls back to the direct http endpoint when it does not answer.
	UseProxy = true
	// PodEnv sends the machine env as per-pod "-e K=V" docker_options.
	// QuickPod appends them after the template's options, so a per-pod value
	// wins over the template's (docker keeps the last -e).
	PodEnv = true
	// PodPortFlag also sends "-p 8675:8675" per pod. Off: the template
	// publishes 8675 already, and a second -p would give it a second host
	// port.
	PodPortFlag = false
)

var gpuPrefixes = []string{"NVIDIA GeForce ", "NVIDIA "}

// now is the adapter's clock (the creating window of statusOf).
var now = time.Now

const (
	listMax    = 8 << 20 // /rentable is ~3 KB an offer, /gpu_pods larger
	maxLimit   = 200
	defLimit   = 50
	errClipLen = 300
	// minDiskGB: createpod raises a smaller disk_size to 30 GB, and bills it.
	minDiskGB = 30
)

type adapter struct{}

func init() { provision.Register(adapter{}) }

func (adapter) Kind() provision.Kind { return provision.QuickPod }

func (adapter) Info() provision.KindInfo {
	return provision.KindInfo{
		Kind:  provision.QuickPod,
		Label: provision.Label(provision.QuickPod),
		Stage: "stable",
		Caps: provision.Caps{
			TemplateRequired: true,
			StopStart:        true,
			StopKeepsDisk:    true,
			HTTPS:            "yes",
			SelfDestroy:      "key",
			Credit:           true,
			EnvFlagString:    true,
			ImageDigestRef:   "yes",
			ReportsInet:      true,
			ReportsDiskBW:    true,
		},
		ConsoleURL: "https://console.quickpod.io",
		KeysURL:    "https://console.quickpod.io/settings",
		KeyScopes:  []string{"pods:read", "pods:write", "templates:read", "profile:read"},
	}
}

// ---- wire shapes (spec: quickpod.md §4.1, §5.1, §6.2, §7) ----

// rawOffer is one /rentable row. The live shape nests the host under
// _machines (the Swagger's models.Machine[] is wrong).
type rawOffer struct {
	ID          int64   `json:"id"`
	GPUType     string  `json:"gpu_type"`
	NumGPUs     int     `json:"num_gpus"`
	GPUVRAM     float64 `json:"gpu_vram"` // MiB
	HourlyCost  float64 `json:"hourly_cost"`
	MaxDiskSize int     `json:"max_disk_size"`
	Occupied    bool    `json:"occupied"`
	Machine     struct {
		Online            bool     `json:"online"`
		Reliability       *float64 `json:"reliability"`         // 0..100
		ListedStorageCost float64  `json:"listed_storage_cost"` // USD per TB per hour
		GeoInfo           string   `json:"geoinfo"`             // "US|Kentucky|Lexington"
		InetDown          float64  `json:"inet_down"`           // Mbps (speedtest); 0 = not reported
		DiskBW            float64  `json:"disk_bw"`             // MB/s write test; 0 = not reported
		AvailDiskSpace    float64  `json:"avail_disk_space"`    // GB free on the machine; 0 = not reported
	} `json:"_machines"`

	raw map[string]any // the row as listed, for hostfit.FromQuickPod
}

// maxDisk is the most disk this offer can be rented with: max_disk_size,
// or the machine's free space when that is less (as hostfit.FromQuickPod
// reads it). 0 = no limit reported.
func (r rawOffer) maxDisk() int {
	md := r.MaxDiskSize
	if a := int(math.Floor(r.Machine.AvailDiskSpace)); a > 0 && (md <= 0 || a < md) {
		md = a
	}
	return md
}

// host is the row's host facts (driver, CUDA, disk, bandwidth).
func (r rawOffer) host() hostfit.Host { return hostfit.FromQuickPod(r.raw) }

type rawTemplate struct {
	UserID        int64  `json:"user_id"`
	IsPublic      bool   `json:"is_public"`
	UUID          string `json:"template_uuid"`
	Name          string `json:"template_name"`
	ImagePath     string `json:"image_path"`
	VersionTag    string `json:"version_tag"`
	DockerOptions string `json:"docker_options"` // read for the flags only, never returned
}

// rawPod is one models.UserPod. Names is the pod UUID (pod_uuid is read too,
// as the Go CLI does). Destroyed pods are not listed. Logs is the tail the
// host agent uploads after a podlogs call, lines joined with "<br>".
type rawPod struct {
	ID           int64   `json:"id"`
	Names        string  `json:"Names"`
	PodUUID      string  `json:"pod_uuid"`
	Altname      string  `json:"altname"`
	State        string  `json:"State"`
	Intended     string  `json:"intended_state"`
	Destroyed    bool    `json:"destroyed"`
	PortMappings string  `json:"port_mappings"`
	Logs         string  `json:"logs"`
	HourlyCost   float64 `json:"hourly_cost"`
	StorageCost  float64 `json:"storage_cost"`
	GPUCount     int     `json:"gpu_count"`
	OffersID     int64   `json:"offers_id"`
	CreatedAt    int64   `json:"created_at"` // epoch ms
	Machine      struct {
		Online  *bool  `json:"online"`
		GeoInfo string `json:"geoinfo"`
	} `json:"_machines"`
	Offer struct {
		GPUType string  `json:"gpu_type"`
		GPUVRAM float64 `json:"gpu_vram"`
		NumGPUs int     `json:"num_gpus"`
	} `json:"_offers"`
}

func (p rawPod) uuid() string {
	if p.Names != "" {
		return p.Names
	}
	return p.PodUUID
}

// ---- calls ----

func base() string { return strings.TrimRight(API, "/") }

// call sends one request. key == "" sends no credentials (the public routes);
// otherwise the key goes in both X-API-Key and Authorization: ApiKey, as the
// official CLI sends a qpk_ key. A non-2xx answer becomes a *provision.Error
// with the key and every extra secret (the create body's env values)
// redacted.
func call(ctx context.Context, op, method, path, key string, body any, secrets ...string) ([]byte, error) {
	c := provision.Call{Kind: provision.QuickPod, Method: method, URL: base() + path, Body: body, Max: listMax, Secret: key}
	if key != "" {
		c.Header = map[string]string{"X-API-Key": key, "Authorization": "ApiKey " + key}
	}
	res, err := provision.Do(ctx, c)
	if err != nil {
		return nil, err
	}
	if res.Status < 200 || res.Status > 299 {
		return nil, apiError(op, res, append([]string{key}, secrets...)...)
	}
	return res.Body, nil
}

func decode(b []byte, v any, key string) error {
	if err := json.Unmarshal(b, v); err != nil {
		return provision.Errorf(provision.QuickPod, provision.ClassUnavailable, http.StatusOK,
			"unreadable answer: %s", provision.Clip(provision.Redact(err.Error(), key), errClipLen))
	}
	return nil
}

// errorBody reads QuickPod's flat error object. Values are mostly strings,
// but insufficient_credit carries numbers, so any scalar is accepted.
func errorBody(b []byte) map[string]string {
	var raw map[string]any
	if json.Unmarshal(b, &raw) != nil {
		return nil
	}
	out := make(map[string]string, len(raw))
	for k, v := range raw {
		switch t := v.(type) {
		case string:
			out[k] = t
		case float64:
			out[k] = strconv.FormatFloat(t, 'f', -1, 64)
		case bool:
			out[k] = strconv.FormatBool(t)
		}
	}
	return out
}

// message is the first non-empty of error, details and message (the
// official CLI's rule). createpod answers every failure of its database step
// as 406 {"error":"Database query failed","message":<the reason>}, so there
// the message comes first.
func message(m map[string]string) string {
	keys := []string{"error", "details", "message"}
	if m["error"] == dbFailed {
		keys = []string{"message", "details", "error"}
	}
	for _, k := range keys {
		if v := strings.TrimSpace(m[k]); v != "" {
			return v
		}
	}
	return ""
}

const dbFailed = "Database query failed"

// capacityText is what a createpod 406 says when the offer can't be had now
// (busy, occupied, offline, gone, no free port range or GPU slots).
var capacityText = regexp.MustCompile(`(?i)offer is busy|occupied|offline|not found|port range|gpu`)

// apiError classifies a failed call. op is "create", "start" or anything
// else; secrets are redacted from the message.
func apiError(op string, res provision.Response, secrets ...string) *provision.Error {
	m := errorBody(res.Body)
	msg := message(m)
	if msg == "" {
		msg = provision.Clip(string(res.Body), errClipLen)
	}
	if msg == "" {
		msg = http.StatusText(res.Status)
	}
	e := &provision.Error{Provider: provision.QuickPod, Status: res.Status}
	switch {
	case m["error"] == "insufficient_credit":
		e.Class = provision.ClassCredit
		msg = creditMessage(m)
	case res.Status == http.StatusConflict && op == "start":
		e.Class = provision.ClassConflict
		if !strings.Contains(strings.ToLower(msg), "in use") {
			msg = "This machine is in use at the moment (" + msg + ")"
		}
	case res.Status == http.StatusNotAcceptable && strings.Contains(strings.ToLower(msg), "disk"):
		e.Class = provision.ClassInvalid
	case res.Status == http.StatusNotAcceptable && m["error"] == dbFailed && !capacityText.MatchString(msg):
		e.Class = provision.ClassUnavailable
	case res.Status == http.StatusNotAcceptable || res.Status == http.StatusConflict ||
		strings.Contains(strings.ToLower(msg), "offer is busy"):
		e.Class = provision.ClassCapacity
	case res.Status == http.StatusForbidden && strings.HasPrefix(msg, "You do not have permission"):
		// a lifecycle call on a pod that is gone (or not this account's)
		e.Class = provision.ClassNotFound
	case res.Status == http.StatusUnauthorized || res.Status == http.StatusForbidden:
		e.Class = provision.ClassAuth
	case res.Status == http.StatusPaymentRequired:
		e.Class = provision.ClassCredit
	case res.Status == http.StatusTooManyRequests:
		e.Class = provision.ClassRateLimited
		e.RetryAfter = provision.RetryAfter(res.Header, time.Now())
	case res.Status == http.StatusNotFound:
		e.Class = provision.ClassNotFound
	case res.Status == http.StatusBadRequest:
		e.Class = provision.ClassInvalid
	default:
		e.Class = provision.ClassUnavailable
	}
	e.Message = provision.Clip(provision.Redact(msg, secrets...), errClipLen)
	return e
}

func creditMessage(m map[string]string) string {
	s := "insufficient credit"
	if c := m["current_credit"]; c != "" {
		s += ": $" + c + " left"
		if r := m["required_credit"]; r != "" {
			s += ", $" + r + " needed"
		}
	}
	return s
}

// ---- account ----

func (adapter) Verify(ctx context.Context, key string) (provision.Account, error) {
	b, err := call(ctx, "verify", http.MethodGet, "/update/api/me", key, nil)
	if err != nil {
		return provision.Account{}, err
	}
	var me struct {
		Credit *float64 `json:"credit"`
	}
	if err := decode(b, &me, key); err != nil {
		return provision.Account{}, err
	}
	acc := provision.Account{Credit: me.Credit}
	if me.Credit != nil {
		acc.Detail = fmt.Sprintf("$%.2f credit", *me.Credit)
	}
	return acc, nil
}

// ---- offers ----

func rentable(ctx context.Context) ([]rawOffer, error) {
	b, err := call(ctx, "offers", http.MethodGet, "/rentable", "", nil)
	if err != nil {
		return nil, err
	}
	var rows []rawOffer
	if err := decode(b, &rows, ""); err != nil {
		return nil, err
	}
	var raws []map[string]any
	if err := decode(b, &raws, ""); err != nil {
		return nil, err
	}
	for i := range rows {
		if i < len(raws) {
			rows[i].raw = raws[i]
		}
	}
	return rows, nil
}

func vramGB(mib float64) int { return int(math.Round(mib / 1024)) }

// displayGPU strips the vendor prefix QuickPod puts on every name
// ("NVIDIA GeForce RTX 3090" -> "RTX 3090", "NVIDIA RTX A6000" -> "RTX A6000").
func displayGPU(s string) string {
	s = strings.TrimSpace(s)
	for _, p := range gpuPrefixes {
		if t, ok := strings.CutPrefix(s, p); ok {
			return t
		}
	}
	return s
}

// location turns geoinfo "US|Kentucky|Lexington" into "US, Kentucky, Lexington".
func location(geo string) string {
	return strings.Join(strings.FieldsFunc(geo, func(c rune) bool { return c == '|' }), ", ")
}

func round6(f float64) float64 { return math.Round(f*1e6) / 1e6 }

func diskOf(q provision.OfferQuery) int {
	if q.DiskGB > 0 {
		return q.DiskGB
	}
	return provision.DefaultDiskGB
}

// billedDisk is the disk QuickPod creates and bills: at least 30 GB.
func billedDisk(disk int) int { return max(disk, minDiskGB) }

// storageCost is QuickPod's own rule: listed_storage_cost (USD per TB-hour)
// x disk / 1024, on the billed disk.
func storageCost(listed float64, disk int) float64 {
	return round6(listed * float64(billedDisk(disk)) / 1024)
}

// offerOf maps one row for a disk size: compute + storage, the storage part
// billing on while stopped.
func offerOf(r rawOffer, disk int) provision.Offer {
	storage := storageCost(r.Machine.ListedStorageCost, disk)
	o := provision.Offer{
		ID:                  strconv.FormatInt(r.ID, 10),
		Provider:            provision.QuickPod,
		GPU:                 displayGPU(r.GPUType),
		GPUCount:            r.NumGPUs,
		VRAMGB:              vramGB(r.GPUVRAM),
		PricePerHour:        round6(r.HourlyCost + storage),
		StoppedPricePerHour: &storage,
		Location:            location(r.Machine.GeoInfo),
		Availability:        "available",
		HTTPS:               UseProxy,
	}
	if r.Machine.Reliability != nil {
		rel := round6(*r.Machine.Reliability / 100)
		o.Reliability = &rel
	}
	if md := r.maxDisk(); md > 0 {
		o.MaxDiskGB = &md
	}
	return o
}

// keep applies the query client-side (/rentable has no server-side filters).
// Fast drops only hosts that report a bandwidth below the threshold: a 0 or
// missing value passes (D-15).
func keep(r rawOffer, q provision.OfferQuery, disk, count int) bool {
	if r.Occupied || !r.Machine.Online || r.NumGPUs != count {
		return false
	}
	if vramGB(r.GPUVRAM) < q.MinVRAMGB {
		return false
	}
	if q.MaxVRAMGB > 0 && r.GPUVRAM >= float64(q.MaxVRAMGB)*1024 {
		return false
	}
	if q.GPU != "" && !strings.Contains(strings.ToLower(r.GPUType), strings.ToLower(strings.TrimSpace(q.GPU))) {
		return false
	}
	if md := r.maxDisk(); md > 0 && md < disk {
		return false
	}
	if q.Fast {
		if v := r.Machine.InetDown; v > 0 && v < hostfit.FastInetDownMbps {
			return false
		}
		if v := r.Machine.DiskBW; v > 0 && v < hostfit.FastDiskBWMBps {
			return false
		}
	}
	return true
}

func (adapter) Offers(ctx context.Context, key string, q provision.OfferQuery) ([]provision.Offer, error) {
	rows, err := rentable(ctx)
	if err != nil {
		return nil, err
	}
	disk, count := diskOf(q), max(q.GPUCount, 1)
	limit := q.Limit
	if limit <= 0 {
		limit = defLimit
	}
	limit = min(limit, maxLimit)
	out := []provision.Offer{}
	for _, r := range rows {
		if !keep(r, q, disk, count) {
			continue
		}
		h := r.host()
		if !provision.KeepOffer(h, q) { // /rentable has no server-side filters
			continue
		}
		o := offerOf(r, disk)
		if q.MaxPrice > 0 && o.PricePerHour > q.MaxPrice {
			continue
		}
		provision.DecorateOffer(&o, h, q)
		out = append(out, o)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].PricePerHour != out[j].PricePerHour {
			return out[i].PricePerHour < out[j].PricePerHour
		}
		return out[i].ID < out[j].ID
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// Quote re-reads the offer from /rentable and asks offer_is_busy; an offer
// that is gone or busy is capacity.
func (adapter) Quote(ctx context.Context, key, offerID string, q provision.OfferQuery) (provision.Offer, error) {
	rows, err := rentable(ctx)
	if err != nil {
		return provision.Offer{}, err
	}
	i := slices.IndexFunc(rows, func(r rawOffer) bool { return strconv.FormatInt(r.ID, 10) == offerID })
	if i < 0 || rows[i].Occupied || !rows[i].Machine.Online {
		return provision.Offer{}, provision.Errorf(provision.QuickPod, provision.ClassCapacity, 0, "offer %s is no longer listed", offerID)
	}
	b, err := call(ctx, "quote", http.MethodGet, "/update/offer_is_busy?offer_id="+url.QueryEscape(offerID), "", nil)
	if err != nil {
		return provision.Offer{}, err
	}
	var busy bool
	if err := decode(b, &busy, ""); err != nil {
		return provision.Offer{}, err
	}
	if busy {
		return provision.Offer{}, provision.Errorf(provision.QuickPod, provision.ClassCapacity, 0, "offer is busy")
	}
	disk := diskOf(q)
	if md := rows[i].maxDisk(); md > 0 && disk > md {
		return provision.Offer{}, provision.Errorf(provision.QuickPod, provision.ClassInvalid, 0,
			"%d GB of disk is more than this offer's %d GB", disk, md)
	}
	o := offerOf(rows[i], disk)
	provision.DecorateOffer(&o, rows[i].host(), q)
	return o, nil
}

// ---- templates ----

var (
	portFlag  = regexp.MustCompile(`-p\s+8675:8675\b`)
	tokenFlag = regexp.MustCompile(`AI_TOOLKIT_AUTH=`)
)

// imageOf is the image a template runs: image_path as typed. createpod pulls
// and runs image_path alone and never reads version_tag, so a tag or digest
// counts only inside the path.
func imageOf(t rawTemplate) string { return strings.TrimSpace(t.ImagePath) }

// Templates lists the account's own GPU templates. /templates also returns
// every public template; those are left out, because the studio launches only
// a template the account controls. docker_options is read only to set the
// flags: it may carry a token and is never returned.
func (adapter) Templates(ctx context.Context, key string) ([]provision.Template, error) {
	b, err := call(ctx, "templates", http.MethodGet, "/templates", key, nil)
	if err != nil {
		return nil, err
	}
	var rows []rawTemplate
	if err := decode(b, &rows, key); err != nil {
		return nil, err
	}
	me := int64(-1)
	if slices.ContainsFunc(rows, func(t rawTemplate) bool { return t.IsPublic }) {
		mb, err := call(ctx, "templates", http.MethodGet, "/update/api/me", key, nil)
		if err != nil {
			return nil, err
		}
		var u struct {
			ID int64 `json:"id"`
		}
		if err := decode(mb, &u, key); err != nil {
			return nil, err
		}
		me = u.ID
	}
	out := make([]provision.Template, 0, len(rows))
	for _, t := range rows {
		if t.IsPublic && t.UserID != me {
			continue
		}
		img := imageOf(t)
		out = append(out, provision.Template{
			ID:            t.UUID,
			Name:          t.Name,
			Image:         img,
			Ours:          strings.Contains(t.ImagePath, "aitoolkit-go"),
			PublishesPort: portFlag.MatchString(t.DockerOptions),
			HasToken:      tokenFlag.MatchString(t.DockerOptions),
		})
	}
	return out, nil
}

// ---- launch ----

var envKey = regexp.MustCompile(`^[A-Z_][A-Z0-9_]{0,63}$`)

// envValueOK keeps a value to QuickPod's docker-options charset: the host
// agent runs the options through sh -c unquoted, so a space, quote, glob or
// shell character could split the string or change the command.
func envValueOK(v string) bool { return provision.QuickPodValue.MatchString(v) }

// dockerOptions is the per-pod docker_options for an env: "-e K=V" pairs in
// key order (plus "-p 8675:8675" first with PodPortFlag). The service has
// already applied QuickPod's strict charset; it is checked again here so no
// value can ever smuggle another docker flag into the string.
func dockerOptions(env map[string]string) (string, error) {
	var parts []string
	if PodPortFlag {
		parts = append(parts, "-p 8675:8675")
	}
	if !PodEnv {
		return strings.Join(parts, " "), nil
	}
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		v := env[k]
		if !envKey.MatchString(k) {
			return "", provision.Errorf(provision.QuickPod, provision.ClassInvalid, 0, "env name %q is not allowed on QuickPod", k)
		}
		if !envValueOK(v) {
			return "", provision.Errorf(provision.QuickPod, provision.ClassInvalid, 0, "the value of %s may hold only letters, digits and . _ / : @ %% + = , - on QuickPod", k)
		}
		parts = append(parts, "-e "+k+"="+v)
	}
	return strings.Join(parts, " "), nil
}

type createBody struct {
	OffersID      int64  `json:"offers_id"`
	TemplateUUID  string `json:"template_uuid"`
	DiskSize      string `json:"disk_size"` // a string of GB on create (spec: quickpod.md §5.3)
	Altname       string `json:"altname"`
	DockerOptions string `json:"docker_options,omitempty"`
}

// Launch creates a pod from the template. The image is the template's
// (createpod has no image field), so s.Image is not sent. The answer carries
// the pod_uuid; when it does not, Get resolves the pod by offers_id +
// altname.
func (adapter) Launch(ctx context.Context, key string, s provision.LaunchSpec) (provision.Ref, error) {
	offer, err := strconv.ParseInt(s.OfferID, 10, 64)
	if err != nil {
		return provision.Ref{}, provision.Errorf(provision.QuickPod, provision.ClassInvalid, 0, "offer id %q is not a QuickPod offer", s.OfferID)
	}
	if strings.TrimSpace(s.TemplateID) == "" {
		return provision.Ref{}, provision.Errorf(provision.QuickPod, provision.ClassInvalid, 0, "a QuickPod pod needs a template")
	}
	opts, err := dockerOptions(s.Env)
	if err != nil {
		return provision.Ref{}, err
	}
	disk := s.DiskGB
	if disk <= 0 {
		disk = provision.DefaultDiskGB
	}
	// The body carries every env value (the machine token, S3 and HF
	// secrets): an error that echoes it must not carry them on.
	var secrets []string
	for _, v := range s.Env {
		if len(v) >= 8 { // short values ("1", "true") would garble the message
			secrets = append(secrets, v)
		}
	}
	body := createBody{OffersID: offer, TemplateUUID: s.TemplateID, DiskSize: strconv.Itoa(disk), Altname: s.Label, DockerOptions: opts}
	b, err := call(ctx, "create", http.MethodPost, "/update/api/createpod", key, body, secrets...)
	if err != nil {
		return provision.Ref{}, err
	}
	m := errorBody(b)
	if m["status"] != "success" {
		// 200 without success: classify by the text like an error answer.
		return provision.Ref{}, apiError("create", provision.Response{Status: http.StatusBadRequest, Body: b}, append([]string{key}, secrets...)...)
	}
	return provision.Ref{ID: m["pod_uuid"], OfferID: s.OfferID, Label: s.Label}, nil
}

// ---- instances ----

func pods(ctx context.Context, key string) ([]rawPod, error) {
	b, err := call(ctx, "pods", http.MethodGet, "/update/api/gpu_pods", key, nil)
	if err != nil {
		return nil, err
	}
	var rows []rawPod
	if err := decode(b, &rows, key); err != nil {
		return nil, err
	}
	return rows, nil
}

// find matches a pod by uuid, or by offers_id + altname while the id is not
// known yet (the newest live match wins).
func find(rows []rawPod, ref provision.Ref) (rawPod, bool) {
	if ref.ID != "" {
		for _, p := range rows {
			if p.uuid() == ref.ID {
				return p, true
			}
		}
		return rawPod{}, false
	}
	var best rawPod
	found := false
	for _, p := range rows {
		if strconv.FormatInt(p.OffersID, 10) != ref.OfferID || p.Altname != ref.Label || p.uuid() == "" {
			continue
		}
		if !found || (best.Destroyed && !p.Destroyed) || (best.Destroyed == p.Destroyed && p.CreatedAt > best.CreatedAt) {
			best, found = p, true
		}
	}
	return best, found
}

func instanceOf(p rawPod, t time.Time) provision.Instance {
	st, detail := statusOf(p, t)
	in := provision.Instance{
		ProviderID:     p.uuid(),
		Label:          p.Altname,
		Status:         st,
		ProviderStatus: providerStatus(p),
		Detail:         detail,
		GPU:            displayGPU(p.Offer.GPUType),
		GPUCount:       p.GPUCount,
		VRAMGB:         vramGB(p.Offer.GPUVRAM),
		Location:       location(p.Machine.GeoInfo),
		Actions:        actionsFor(st),
	}
	if in.GPUCount == 0 {
		in.GPUCount = p.Offer.NumGPUs
	}
	if p.CreatedAt > 0 {
		in.CreatedAt = time.UnixMilli(p.CreatedAt)
	}
	if p.HourlyCost > 0 || p.StorageCost > 0 {
		// hourly_cost already includes storage_cost (createpod adds it).
		run, stopped := round6(p.HourlyCost), round6(p.StorageCost)
		in.PricePerHour, in.StoppedPricePerHour = &run, &stopped
	}
	if st == provision.StatusTerminated {
		return in
	}
	if UseProxy && in.ProviderID != "" {
		in.Endpoints = append(in.Endpoints, provision.Endpoint{URL: ProxyURL(in.ProviderID), Secure: true, Via: "proxy"})
	}
	ip, port, ok := Parse8675(p.PortMappings)
	if ok {
		in.Endpoints = append(in.Endpoints, provision.Endpoint{URL: "http://" + hostPort(ip, port), Via: "direct"})
	}
	in.PortMissing = st == provision.StatusRunning && !ok
	if p.Machine.Online != nil && !*p.Machine.Online && in.Detail == "" {
		in.Detail = "QuickPod reports the host offline."
	}
	return in
}

func hostPort(ip string, port int) string {
	if strings.Contains(ip, ":") {
		return "[" + ip + "]:" + strconv.Itoa(port)
	}
	return ip + ":" + strconv.Itoa(port)
}

func (adapter) Get(ctx context.Context, key string, ref provision.Ref) (provision.Instance, error) {
	rows, err := pods(ctx, key)
	if err != nil {
		return provision.Instance{}, err
	}
	p, ok := find(rows, ref)
	if !ok {
		return provision.Instance{}, provision.Errorf(provision.QuickPod, provision.ClassNotFound, 0, "pod not found")
	}
	return instanceOf(p, now()), nil
}

func (adapter) List(ctx context.Context, key, labelPrefix string) ([]provision.Instance, error) {
	rows, err := pods(ctx, key)
	if err != nil {
		return nil, err
	}
	t := now()
	out := []provision.Instance{}
	for _, p := range rows {
		if strings.HasPrefix(p.Altname, labelPrefix) && p.uuid() != "" {
			out = append(out, instanceOf(p, t))
		}
	}
	return out, nil
}

// ---- actions ----

func podAction(ctx context.Context, key, action, id string) ([]byte, error) {
	if id == "" {
		return nil, provision.Errorf(provision.QuickPod, provision.ClassInvalid, 0, "the pod is not listed yet; try again in a moment")
	}
	op := action
	if action == "startpod" {
		op = "start"
	}
	return call(ctx, op, http.MethodGet, "/update/api/"+action+"?pod_uuid="+url.QueryEscape(id), key, nil)
}

func (adapter) Stop(ctx context.Context, key, id string) error {
	_, err := podAction(ctx, key, "stoppod", id)
	return err
}

func (adapter) Start(ctx context.Context, key, id string) error {
	_, err := podAction(ctx, key, "startpod", id)
	return err
}

// Terminate destroys the pod and its disk. A pod that is already destroyed
// or no longer listed counts as done (gpu_pods drops destroyed pods, and a
// lifecycle call on one answers 403 "You do not have permission").
func (adapter) Terminate(ctx context.Context, key, id string) error {
	_, err := podAction(ctx, key, "destroypod", id)
	if err == nil || id == "" {
		return err
	}
	switch provision.ClassOf(err) {
	case provision.ClassNotFound:
		return nil
	case provision.ClassAuth, provision.ClassRateLimited:
		return err
	}
	rows, lerr := pods(ctx, key)
	if lerr != nil {
		return err
	}
	if p, ok := find(rows, provision.Ref{ID: id}); !ok || p.Destroyed {
		return nil
	}
	return err
}

// brTag is how the host agent joins the log lines it uploads.
var brTag = regexp.MustCompile(`(?i)<br\s*/?>`)

// Logs asks the host agent for a fresh tail (podlogs only queues "docker
// logs"; the agent uploads the last 25 lines on its next beat) and returns
// the tail gpu_pods holds now, which is the previous upload until then.
func (adapter) Logs(ctx context.Context, key, id string, tail int) (string, error) {
	if _, err := podAction(ctx, key, "podlogs", id); err != nil {
		return "", err
	}
	rows, err := pods(ctx, key)
	if err != nil {
		return "", err
	}
	p, ok := find(rows, provision.Ref{ID: id})
	if !ok {
		return "", provision.Errorf(provision.QuickPod, provision.ClassNotFound, 0, "pod not found")
	}
	text := provision.Redact(brTag.ReplaceAllString(p.Logs, "\n"), key)
	if tail > 0 {
		lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
		if len(lines) > tail {
			text = strings.Join(lines[len(lines)-tail:], "\n")
		}
	}
	return text, nil
}
