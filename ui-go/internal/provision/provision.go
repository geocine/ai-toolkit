// Package provision launches and tracks GPU servers ("machines") that run the
// aitk headless image on RunPod, Vast.ai and QuickPod. Each adapter
// (internal/provision/<kind>) is written only from the provider's published
// API definition and has an httptest mock (<kind>/<kind>mock) that tests and
// simulate mode run against. No test and no simulate run may reach a real
// provider: call.go refuses the real hosts in both cases.
//
// provision imports hostfit and imagecheck (the host facts and image checks
// offers carry) but never gpupick, which imports it.
package provision

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// ---- kinds ----

type Kind string

const (
	RunPod   Kind = "runpod"
	Vast     Kind = "vast"
	QuickPod Kind = "quickpod"
)

var allKinds = []Kind{RunPod, Vast, QuickPod}

// ParseKind accepts exactly "runpod", "vast" or "quickpod".
func ParseKind(s string) (Kind, bool) {
	k := Kind(s)
	return k, slices.Contains(allKinds, k)
}

// Defaults shared by the service, the guides and the studio.
const (
	DefaultImage      = "ghcr.io/geocine/aitoolkit-go:latest"
	DefaultDiskGB     = 100
	Port              = 8675
	DefaultMaxRunning = 2
	LabelPrefix       = "aitk-"
	MachineIDPrefix   = "mch"
)

// LabelFor is the provider-side name (RunPod name, Vast label, QuickPod
// altname) of a machine: "aitk-" + the id without its "mch_" prefix.
func LabelFor(machineID string) string {
	return LabelPrefix + strings.TrimPrefix(machineID, MachineIDPrefix+"_")
}

// EnvKeyName is the env fallback for a provider's launcher key. Deliberately
// not AITK_RUNPOD_API_KEY / AITK_VAST_API_KEY: those let a pod end itself and
// must not turn every GPU node into a launcher.
func EnvKeyName(k Kind) string { return "AITK_PROVISION_" + strings.ToUpper(string(k)) + "_KEY" }

// SettingKey / MaxRunningSetting are fork settings (state.Setting).
func SettingKey(k Kind) string        { return "provision.key." + string(k) }
func MaxRunningSetting(k Kind) string { return "provision.max_running." + string(k) }

// ---- statuses ----

type Status string

const (
	StatusCreating    Status = "creating" // accepted; provider allocating (QuickPod: until listed)
	StatusStarting    Status = "starting" // host assigned; image pulling / container starting
	StatusRunning     Status = "running"  // provider says running; aitk not (yet) answering with our token
	StatusReady       Status = "ready"    // service only: /v1/system answered 200 with the machine token
	StatusStopping    Status = "stopping"
	StatusStopped     Status = "stopped"     // compute released; disk may still bill
	StatusTerminating Status = "terminating" // service only: terminate sent, provider not gone yet
	StatusTerminated  Status = "terminated"  // gone at the provider
	StatusFailed      Status = "failed"      // launch refused, provider error state, or not ready in time
	StatusUnknown     Status = "unknown"     // provider state we cannot map (raw value in provider_status)
)

// Transitional statuses are refreshed every 10 s by the watcher.
func (s Status) Transitional() bool {
	switch s {
	case StatusCreating, StatusStarting, StatusRunning, StatusStopping, StatusTerminating:
		return true
	}
	return false
}

// Problem qualifies a status; the human text is Machine.Detail.
type Problem string

const (
	ProblemTokenMismatch   Problem = "token_mismatch"    // health 200, /v1/system 401 with our token (template token won)
	ProblemOpenServer      Problem = "open_server"       // unauthenticated /v1/system answered 200
	ProblemNoPort          Problem = "no_port"           // running > 3 min without a mapping for 8675
	ProblemNotReady        Problem = "not_ready_timeout" // not ready in time
	ProblemUnreachable     Problem = "unreachable"       // was ready, probe failing > 5 min
	ProblemNeverListed     Problem = "never_listed"      // QuickPod accepted the create but no pod appeared in 5 min
	ProblemProvider        Problem = "provider_error"    // provider error state; detail has its text
	ProblemNotRefreshed    Problem = "not_refreshed"     // real machine while simulate is on (or the reverse)
	ProblemCUDAFailed      Problem = "cuda_failed"       // GET /v1/system/cuda on the machine answered ok:false
	ProblemPullSlow        Problem = "pull_slow"         // still pulling long after the host's bandwidth predicts
	ProblemImagePullFailed Problem = "image_pull_failed" // the provider could not pull the image (imagecheck.ClassifyFailure)
)

type WhenDone string

const (
	WhenKeep    WhenDone = "keep"
	WhenStop    WhenDone = "stop"
	WhenDestroy WhenDone = "destroy"
)

// ---- provider description ----

type Caps struct {
	TemplateRequired bool   `json:"template_required"` // QuickPod: launch needs a template_id
	CloudChoice      bool   `json:"cloud_choice"`      // RunPod: secure | community
	CountChoice      bool   `json:"count_choice"`      // RunPod: gpu_count is a query; others fixed per offer
	StopStart        bool   `json:"stop_start"`
	StopKeepsDisk    bool   `json:"stop_keeps_disk"` // RunPod false: container disk is wiped on stop
	HTTPS            string `json:"https"`           // "yes" (RunPod) | "no" (Vast) | "unconfirmed" (QuickPod proxy)
	SelfDestroy      string `json:"self_destroy"`    // "builtin" (Vast CONTAINER_API_KEY) | "key" (needs self_destroy_key)
	Credit           bool   `json:"credit"`          // Verify reports credit (RunPod: best effort)
	EnvCountMax      int    `json:"env_count_max"`   // RunPod 50; 0 = none documented
	EnvValueMax      int    `json:"env_value_max"`   // Vast 256
	EnvTotalMax      int    `json:"env_total_max"`   // Vast 32768, RunPod 100000 (body)
	EnvFlagString    bool   `json:"env_flag_string"` // QuickPod: env travels as "-e K=V" in docker_options
	// ImageDigestRef says whether a launch may send repo@sha256:...:
	// "yes" | "no" | "unconfirmed" (all three providers are "unconfirmed" today).
	ImageDigestRef string `json:"image_digest_ref"`
	// ReportsInet / ReportsDiskBW: offers MAY carry download / disk bandwidth
	// (Vast and QuickPod true, RunPod false). Per offer, 0 or missing means
	// not reported, which passes the fast-host rules.
	ReportsInet   bool `json:"reports_inet"`
	ReportsDiskBW bool `json:"reports_disk_bw"`
}

type KindInfo struct {
	Kind        Kind     `json:"kind"`
	Label       string   `json:"label"` // "RunPod", "Vast.ai", "QuickPod"
	Stage       string   `json:"stage"` // "stable" | "beta"
	Caps        Caps     `json:"capabilities"`
	Unconfirmed []string `json:"unconfirmed"` // sentences shown under "Not verified against the live API"
	ConsoleURL  string   `json:"console_url"`
	KeysURL     string   `json:"keys_url"`
	KeyScopes   []string `json:"key_scopes"` // scopes to tick when creating the key
}

// ---- adapter-level types ----

type OfferQuery struct {
	GPU       string            // case-insensitive substring of the GPU name
	MinVRAMGB int               // per GPU
	MaxPrice  float64           // $/h total, 0 = any
	GPUCount  int               // default 1
	Cloud     string            // RunPod: "secure" (default) | "community"
	DiskGB    int               // prices storage; default DefaultDiskGB
	Limit     int               // default 50, max 200
	Image     string            // "" = DefaultImage; the image the host filter is derived from
	Model     *hostfit.ModelRef // VRAM badges and disk need; query ?model=<arch>
	ShowAll   bool              // ?show_all=1: keep incompatible offers, badged
	// Fast keeps hosts with inet_down >= hostfit.FastInetDownMbps and
	// disk_bw >= hostfit.FastDiskBWMBps: server-side on Vast, client-side
	// elsewhere, and there only for offers that report the value.
	Fast bool
	// MaxVRAMGB caps VRAM per GPU, exclusive (keeps offers with VRAM <
	// MaxVRAMGB GiB); 0 = no cap. Vast sends gpu_ram {lt: MaxVRAMGB*1024}
	// (MiB); RunPod and QuickPod filter client-side.
	MaxVRAMGB int

	// Set by the service, never by callers or JSON (once per offers call,
	// from a cached Inspect).
	Filter    *hostfit.Filter   `json:"-"` // nil = no image filter (adapters filter only on their own fields)
	ImageInfo *imagecheck.Image `json:"-"` // for hostfit.OfferFitFor; nil when the image could not be read
}

type Offer struct {
	ID                  string            `json:"id"` // opaque; round-trips to LaunchRequest.offer_id
	Provider            Kind              `json:"provider"`
	GPU                 string            `json:"gpu"` // display name, e.g. "RTX 4090"
	GPUCount            int               `json:"gpu_count"`
	VRAMGB              int               `json:"vram_gb"`                // per GPU
	PricePerHour        float64           `json:"price_per_hour"`         // while running, incl. the queried disk
	StoppedPricePerHour *float64          `json:"stopped_price_per_hour"` // storage only; null = unknown
	Location            string            `json:"location"`               // "" = provider picks (RunPod)
	Reliability         *float64          `json:"reliability"`            // 0..1; null when not reported
	Availability        string            `json:"availability"`           // "high" | "medium" | "low" | "available"
	MaxDiskGB           *int              `json:"max_disk_gb"`
	Cloud               string            `json:"cloud,omitempty"` // RunPod
	HTTPS               bool              `json:"https"`           // the machine URL will be https
	Host                *hostfit.Host     `json:"host"`            // null until the service decorates offers
	Fit                 *hostfit.OfferFit `json:"fit"`             // null until the service decorates offers
}

type Template struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Image         string `json:"image"`
	Ours          bool   `json:"ours"`           // image contains "aitoolkit-go"
	PublishesPort bool   `json:"publishes_port"` // docker options / ports map 8675
	HasToken      bool   `json:"has_token"`      // sets AI_TOOLKIT_AUTH (the value is never returned)
}

// LaunchSpec is what the service hands an adapter: validated, env final.
type LaunchSpec struct {
	Label      string // LabelFor(machine id)
	OfferID    string
	Image      string // the image check's launch_ref (a sha tag when the image has one), not the tag as typed
	DiskGB     int
	TemplateID string // QuickPod (required)
	Env        map[string]string
}

// Ref locates an instance; QuickPod resolves ID by OfferID+Label while ID == "".
type Ref struct{ ID, OfferID, Label string }

type Endpoint struct {
	URL    string `json:"url"`    // base for 8675, no trailing slash
	Secure bool   `json:"secure"` // https
	Via    string `json:"via"`    // "proxy" | "direct"
}

// Instance is the provider's view of one machine, normalized. Adapters never
// return StatusReady or StatusTerminating; the service adds those.
type Instance struct {
	ProviderID          string
	Label               string
	Status              Status
	ProviderStatus      string // raw, e.g. "RUNNING", "loading/running", "exited/exited"
	Detail              string
	Endpoints           []Endpoint // preferred first; empty until mapped
	PortMissing         bool       // provider running but no mapping for 8675 (Vast/QuickPod)
	GPU                 string
	GPUCount            int
	VRAMGB              int
	Location            string
	PricePerHour        *float64
	StoppedPricePerHour *float64
	Actions             []string // subset of stop|start|terminate valid now
	CreatedAt           time.Time
	Env                 map[string]string // only for adopt; never serialized or logged
}

type Account struct {
	Credit *float64 // USD; nil when the provider does not say
	Detail string
}

// Cloud is one provider adapter. Adapters are stateless singletons; the key
// is passed per call (one key per provider on this server).
type Cloud interface {
	Kind() Kind
	Info() KindInfo
	Verify(ctx context.Context, key string) (Account, error)
	Offers(ctx context.Context, key string, q OfferQuery) ([]Offer, error)
	// Quote re-reads one offer's current price for q.DiskGB/q.GPUCount.
	// An offer that is gone or taken is a ClassCapacity error.
	Quote(ctx context.Context, key, offerID string, q OfferQuery) (Offer, error)
	Templates(ctx context.Context, key string) ([]Template, error)
	Launch(ctx context.Context, key string, s LaunchSpec) (Ref, error)
	// Get returns ClassNotFound when the instance is gone.
	Get(ctx context.Context, key string, ref Ref) (Instance, error)
	// List returns every instance whose label starts with labelPrefix.
	List(ctx context.Context, key, labelPrefix string) ([]Instance, error)
	Stop(ctx context.Context, key, id string) error
	Start(ctx context.Context, key, id string) error
	Terminate(ctx context.Context, key, id string) error // already gone = nil
	Logs(ctx context.Context, key, id string, tail int) (string, error)
}

// ---- registry ----

var (
	regMu    sync.RWMutex
	registry = map[Kind]Cloud{}
)

// Register is called from each adapter's init(); a duplicate kind panics.
func Register(c Cloud) {
	regMu.Lock()
	defer regMu.Unlock()
	if _, dup := registry[c.Kind()]; dup {
		panic("provision: duplicate adapter " + string(c.Kind()))
	}
	registry[c.Kind()] = c
}

// SetForTest swaps the adapter of c.Kind() and returns a restore func.
func SetForTest(c Cloud) (restore func()) {
	regMu.Lock()
	old, had := registry[c.Kind()]
	registry[c.Kind()] = c
	regMu.Unlock()
	return func() {
		regMu.Lock()
		defer regMu.Unlock()
		if had {
			registry[c.Kind()] = old
		} else {
			delete(registry, c.Kind())
		}
	}
}

// enabledByEnv reads AITK_PROVISION_PROVIDERS (comma list; unset = all).
func enabledByEnv(k Kind) bool {
	v := strings.TrimSpace(os.Getenv("AITK_PROVISION_PROVIDERS"))
	if v == "" {
		return true
	}
	for _, p := range strings.Split(v, ",") {
		if strings.EqualFold(strings.TrimSpace(p), string(k)) {
			return true
		}
	}
	return false
}

// Lookup returns the adapter when it is registered and enabled.
func Lookup(k Kind) (Cloud, bool) {
	regMu.RLock()
	defer regMu.RUnlock()
	c, ok := registry[k]
	return c, ok && enabledByEnv(k)
}

// Enabled lists the registered, enabled kinds in a fixed order.
func Enabled() []Kind {
	var out []Kind
	for _, k := range allKinds {
		if _, ok := Lookup(k); ok {
			out = append(out, k)
		}
	}
	return out
}

// ---- errors ----

type Class string

const (
	ClassAuth        Class = "auth"         // key rejected or missing a scope
	ClassCredit      Class = "credit"       // not enough balance
	ClassCapacity    Class = "capacity"     // offer taken / pool unavailable
	ClassConflict    Class = "conflict"     // action invalid now, price rose, cap reached
	ClassInvalid     Class = "invalid"      // provider validation, or ours before sending
	ClassNotFound    Class = "not_found"    // instance or offer gone
	ClassRateLimited Class = "rate_limited" // 429
	ClassUnsupported Class = "unsupported"
	ClassUnavailable Class = "unavailable" // 5xx, network, unparseable
)

// Error is the one error type adapters return. Message is the provider's own
// text, already passed through Redact and Clip.
type Error struct {
	Provider   Kind
	Class      Class
	Status     int // provider HTTP status; 0 = network / ours
	Message    string
	RetryAfter time.Duration
	// SearchOnly marks a limit on offer search alone (Vast's daily search
	// quota, or its 10-a-minute window): the instance endpoints still answer,
	// so it never pauses the watcher.
	SearchOnly bool
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s %s (HTTP %d): %s", e.Provider, e.Class, e.Status, e.Message)
}

func Errorf(k Kind, c Class, status int, format string, a ...any) *Error {
	return &Error{Provider: k, Class: c, Status: status, Message: fmt.Sprintf(format, a...)}
}

// ClassOf returns the class of a *Error in err's chain, or "".
func ClassOf(err error) Class {
	var e *Error
	if errors.As(err, &e) {
		return e.Class
	}
	return ""
}

var labels = map[Kind]string{RunPod: "RunPod", Vast: "Vast.ai", QuickPod: "QuickPod"}

// Label is the display name of a kind.
func Label(k Kind) string { return labels[k] }

// New /v1 codes (also in internal/v1api).
const (
	CodeProviderAuth       = v1api.CodeProviderAuth
	CodeInsufficientCredit = v1api.CodeInsufficientCredit
	CodeNoCapacity         = v1api.CodeNoCapacity
	CodeRateLimited        = v1api.CodeRateLimited
)

// V1 maps an adapter error onto a /v1 code and message. ok is false for
// errors that are not *Error (the caller answers 500 or a v1api.Problem).
func V1(err error) (code, message string, retryAfter time.Duration, ok bool) {
	var e *Error
	if !errors.As(err, &e) {
		return "", "", 0, false
	}
	p := Label(e.Provider)
	switch e.Class {
	case ClassAuth:
		return v1api.CodeProviderAuth, authMessage(p, e.Message), 0, true
	case ClassCredit:
		return v1api.CodeInsufficientCredit, p + " says the balance is too low: " + e.Message, 0, true
	case ClassCapacity:
		return v1api.CodeNoCapacity, p + ": that offer can't be rented right now (" + e.Message + "). Pick another.", 0, true
	case ClassConflict:
		return v1api.CodeConflict, e.Message, 0, true
	case ClassInvalid, ClassUnsupported:
		return v1api.CodeInvalidRequest, p + ": " + e.Message, 0, true
	case ClassNotFound:
		return v1api.CodeNotFound, p + ": " + e.Message, 0, true
	case ClassRateLimited:
		ra := e.RetryAfter
		if ra <= 0 {
			ra = 10 * time.Second
		}
		return v1api.CodeRateLimited, p + " is rate limiting this server; try again in " + ra.Round(time.Second).String(), ra, true
	default:
		return v1api.CodeRemoteFailed, p + " did not answer: " + e.Message, 0, true
	}
}

// bareRejections are provider texts that only restate a rejected key.
var bareRejections = map[string]bool{
	"unauthorized": true, "forbidden": true, "invalid user key": true, "invalid api key": true, "invalid key": true,
}

// authMessage says once that p rejected the key, with the provider's reason
// when it adds one ("Vast.ai rejected the API key." for "Invalid user key").
func authMessage(p, msg string) string {
	reason := strings.TrimSpace(msg)
	if bareRejections[strings.ToLower(strings.TrimRight(reason, ". "))] || reason == "" {
		return p + " rejected the API key."
	}
	return p + " rejected the API key: " + reason
}

// ---- secrets and tokens ----

// NewToken is a machine's AI_TOOLKIT_AUTH: "aitk_" + 43 base64url chars (256 bits).
func NewToken() string {
	b := make([]byte, 32)
	_, _ = rand.Read(b) // crypto/rand.Read does not fail (Go 1.24+ aborts instead)
	return "aitk_" + base64.RawURLEncoding.EncodeToString(b)
}

// Redact replaces every secret (len >= 4) in s with "[redacted]".
func Redact(s string, secrets ...string) string {
	for _, sec := range secrets {
		if len(sec) >= 4 {
			s = strings.ReplaceAll(s, sec, "[redacted]")
		}
	}
	return s
}

// Clip shortens provider text for messages and logs.
func Clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// KeyHint is "…" + the last 4 characters.
func KeyHint(key string) string {
	if len(key) < 8 {
		return "…"
	}
	return "…" + key[len(key)-4:]
}

// ---- simulate globals (set by internal/provision/sim) ----

// Simulated is true while AITK_PROVISION_SIMULATE is active. call.go then
// refuses the real provider hosts, adapters' bases point at in-process mocks
// and new machines are flagged simulated.
var Simulated atomic.Bool

var simSecret atomic.Value // string

func SetSimSecret(s string) { simSecret.Store(s) }

// WireKey is the key actually sent to a provider. Outside simulate it is the
// stored key. In simulate the mocks accept only the per-process secret, with
// the user's typed key (e.g. "sim_full") as a fault suffix.
func WireKey(stored string) string {
	if !Simulated.Load() {
		return stored
	}
	sec, _ := simSecret.Load().(string)
	if stored == "" {
		return sec
	}
	return sec + "." + stored
}

// ---- /v1 shapes (JSON is the contract; studio/src/lib/provisionTypes.ts mirrors it) ----

type ProviderView struct {
	KindInfo
	Configured bool    `json:"configured"`
	KeySource  string  `json:"key_source"` // "api" | "env" | "sim" | ""
	KeyHint    *string `json:"key_hint"`
	MaxRunning int     `json:"max_running"`
	Running    int     `json:"running"` // live machines of this kind
	Simulated  bool    `json:"simulated"`
}

type ProviderUpdate struct {
	APIKey     *string `json:"api_key"`
	MaxRunning *int    `json:"max_running"` // 1..20
}

type ProviderTest struct {
	OK     bool     `json:"ok"`
	Detail string   `json:"detail"`
	Credit *float64 `json:"credit"`
}

type LaunchRequest struct {
	Provider        Kind     `json:"provider"`
	OfferID         string   `json:"offer_id"`
	MaxPricePerHour float64  `json:"max_price_per_hour"` // required, > 0
	RequestID       string   `json:"request_id"`         // optional idempotency key, 8..64 [A-Za-z0-9_-]
	Name            string   `json:"name"`               // display; default "<gpu> on <provider>"
	Image           string   `json:"image"`              // default DefaultImage
	DiskGB          int      `json:"disk_gb"`            // default 100, <= offer max
	GPUCount        int      `json:"gpu_count"`          // RunPod only; default 1
	Cloud           string   `json:"cloud"`              // RunPod only; "secure" | "community"
	TemplateID      string   `json:"template_id"`        // QuickPod: required
	CORSOrigins     []string `json:"cors_origins"`       // default: the request's Origin when present
	StudioURL       string   `json:"studio_url"`         // AITK_STUDIO_URL on the machine
	CopyRemote      string   `json:"copy_remote"`        // one s3 or hf remote on this server
	CopyHFToken     bool     `json:"copy_hf_token"`
	CopyNotify      bool     `json:"copy_notify"`
	// WhenDone defaults to destroy when copy_remote is set and destroy is
	// allowed (on QuickPod only with self_destroy_key), else keep.
	WhenDone       WhenDone          `json:"when_done"`
	SelfDestroyKey bool              `json:"self_destroy_key"` // put the account key in the machine env
	Env            map[string]string `json:"env"`              // allow-listed passthrough
	// SkipImageCheck is honoured only for custom (non-Go) images or
	// registries that cannot be read, never to bypass a host-side block.
	SkipImageCheck bool `json:"skip_image_check"`
}

type Machine struct {
	ID                  string     `json:"id"`
	Name                string     `json:"name"`
	Label               string     `json:"label"`
	Provider            Kind       `json:"provider"`
	ProviderID          string     `json:"provider_id"`
	Status              Status     `json:"status"`
	Problem             *Problem   `json:"problem"`
	Detail              string     `json:"detail"`
	ProviderStatus      string     `json:"provider_status"`
	GPU                 string     `json:"gpu"`
	GPUCount            int        `json:"gpu_count"`
	VRAMGB              int        `json:"vram_gb"`
	Location            string     `json:"location"`
	Image               string     `json:"image"`            // as the user chose it
	ImageDigest         *string    `json:"image_digest"`     // what the checked tag pointed at; null when unchecked
	ImageLaunchRef      *string    `json:"image_launch_ref"` // the reference sent to the provider; null when unchecked
	ImageWarnings       []string   `json:"image_warnings"`
	DiskGB              int        `json:"disk_gb"`
	BaseURL             *string    `json:"base_url"`
	Secure              bool       `json:"secure"`
	RelayPath           string     `json:"relay_path"` // "/v1/machines/<id>/relay"
	PricePerHour        *float64   `json:"price_per_hour"`
	StoppedPricePerHour *float64   `json:"stopped_price_per_hour"`
	WhenDone            WhenDone   `json:"when_done"`
	SelfDestroyKey      bool       `json:"self_destroy_key"`
	CopiedRemote        string     `json:"copied_remote"`
	Actions             []string   `json:"actions"` // "stop" | "start" | "terminate" | "forget"
	Self                bool       `json:"self"`    // this is the server answering
	Simulated           bool       `json:"simulated"`
	Warnings            []string   `json:"warnings"`
	CreatedAt           time.Time  `json:"created_at"`
	UpdatedAt           time.Time  `json:"updated_at"`
	ReadyAt             *time.Time `json:"ready_at"`
	EndedAt             *time.Time `json:"ended_at"`
}

type MachinePatch struct {
	Name  *string `json:"name"`
	Token *string `json:"token"` // the template's token when it won (QuickPod)
}

type MachineLink struct {
	Name      string `json:"name"`
	BaseURL   string `json:"base_url"`
	Token     string `json:"token"`
	Secure    bool   `json:"secure"`
	RelayPath string `json:"relay_path"`
}

type MachineLogs struct {
	Text      string `json:"text"`
	Source    string `json:"source"` // "runpod" | "vast" | "quickpod" | "simulated"
	Truncated bool   `json:"truncated"`
}

type GuideEnv struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Secret bool   `json:"secret"`
	Note   string `json:"note"`
}

type GuideField struct {
	Label string `json:"label"` // console field label, verbatim
	Value string `json:"value"`
	Note  string `json:"note"`
}

type SetupGuide struct {
	Kind          Kind         `json:"kind"`
	Image         string       `json:"image"`
	Port          int          `json:"port"`
	DiskGB        int          `json:"disk_gb"`
	Env           []GuideEnv   `json:"env"`
	EnvBlock      string       `json:"env_block"`      // RunPod "K=V" lines
	DockerOptions string       `json:"docker_options"` // Vast/QuickPod "-p 8675:8675 -e K=V ..."
	Fields        []GuideField `json:"fields"`
	Steps         []string     `json:"steps"`
	URLPattern    string       `json:"url_pattern"`
	ConsoleURL    string       `json:"console_url"`
	KeysURL       string       `json:"keys_url"`
	KeyScopes     []string     `json:"key_scopes"`
	Token         string       `json:"token,omitempty"` // only with ?generate_token=1
}

// RelayTarget is what the relay needs about one machine.
type RelayTarget struct {
	MachineID string
	BaseURL   string
	Token     string
	Simulated bool
}

// Manager is the service behind /v1/providers and /v1/machines (the service
// implements it; handler tests use a fake). Errors are *Error (map with V1)
// or *v1api.Problem (answer with v1api.Respond).
type Manager interface {
	Providers(ctx context.Context) ([]ProviderView, error)
	SetKey(ctx context.Context, k Kind, in ProviderUpdate) (ProviderView, error)
	RemoveKey(ctx context.Context, k Kind) error
	Test(ctx context.Context, k Kind) (ProviderTest, error)
	Offers(ctx context.Context, k Kind, q OfferQuery) ([]Offer, error)
	// Quote re-reads one offer through the 30 s quote cache (the launch
	// re-quote); images.check {offer} and gpupick use it too.
	Quote(ctx context.Context, k Kind, offerID string, q OfferQuery) (Offer, error)
	Templates(ctx context.Context, k Kind) ([]Template, error)
	// Launch returns created=false for a repeated request_id.
	Launch(ctx context.Context, req LaunchRequest, origin string) (m Machine, created bool, err error)
	List(ctx context.Context, activeOnly bool) ([]Machine, error)
	Get(ctx context.Context, id string, refresh bool) (Machine, error)
	Update(ctx context.Context, id string, p MachinePatch) (Machine, error)
	Stop(ctx context.Context, id string) (Machine, error)
	Start(ctx context.Context, id string) (Machine, error)
	Terminate(ctx context.Context, id string, confirmSelf bool) (Machine, error)
	Forget(ctx context.Context, id string, force bool) error
	Link(ctx context.Context, id string) (MachineLink, error)
	Logs(ctx context.Context, id string, tail int) (MachineLogs, error)
	RelayTarget(ctx context.Context, id string) (RelayTarget, error)
	Run(ctx context.Context) // the watcher loop (bg.Go)
}

// Default is set by the service's init() (service.go).
var Default Manager

// ---- readiness probe (probe.go implements it; the service calls it) ----

type ProbeResult struct {
	BaseURL       string // the endpoint that answered /v1/health
	Health        bool
	TokenOK       bool  // GET /v1/system with the token: 200
	TokenRejected bool  // ... 401
	AuthEnforced  *bool // GET /v1/system without a token: 401 -> true, 200 -> false
	Version       string
	Err           string
}

type Prober interface {
	Probe(ctx context.Context, endpoints []Endpoint, token string, simulated bool) ProbeResult
}

var DefaultProber Prober
