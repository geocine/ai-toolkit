package provision

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/url"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Service is the Manager behind /v1/providers and /v1/machines: keys in fork
// settings, machines in the fork machines table, provider calls through the
// registered adapters. It never terminates a machine on its own.
type Service struct {
	// Now is the clock; nil means time.Now. Tests and the simulate e2e set a
	// FakeClock's Now so every timeout below is deterministic.
	Now func() time.Time

	cache    caches
	launchMu sync.Mutex // launches validate (request_id, names, max_running) and insert one at a time

	mu        sync.Mutex
	machineMu map[string]*sync.Mutex // one row read-modify-write at a time
	kindMu    map[Kind]*sync.Mutex   // provider calls serialized per provider
	refreshed map[string]time.Time   // last refresh per machine: watcher due times and the 5 s Get cache
	seen      map[string]bool        // listed or read at its provider at least once
	pausedTil map[Kind]time.Time     // a Retry-After pauses the watcher for that provider
}

func init() { Default = &Service{} }

var _ Manager = (*Service)(nil)

// Timings (PLAN §5).
const (
	refreshBusy      = 10 * time.Second // transitional, or with a problem
	refreshReady     = 60 * time.Second
	refreshStopped   = 5 * time.Minute
	getCacheTTL      = 5 * time.Second
	noPortGrace      = 3 * time.Minute
	unreachableAfter = 5 * time.Minute
	neverListedAfter = 5 * time.Minute
	// launchCallTimeout bounds the create and its bookkeeping, which run
	// detached from the caller's request.
	launchCallTimeout = 2 * time.Minute
	defaultReady      = 20 * time.Minute
	logsMax           = 256 << 10
	logsTailMax       = 2000
)

// Copy shown on machines (problem details and warnings).
const (
	detailNotReady    = "It is still billed: terminate it or open the provider console."
	detailNoPort      = "This host gave no public port for 8675. Terminate it and pick an offer with direct ports."
	detailMismatch    = "The template set its own AI_TOOLKIT_AUTH. Paste that token to link it."
	detailOpen        = "This server answers without a token. Terminate it, or fix the template."
	detailNeverListed = "QuickPod accepted the pod but it never appeared. Check console.quickpod.io/pods; it may be billing."
	detailSelfStopped = "Stopped by itself (when done) or the container exited."
	detailSimEnded    = "Simulated machine ended when the server restarted"
	detailRealInSim   = "This is a real machine; simulate mode doesn't refresh it or act on it."
	warnQuickPodKeep  = "When done stays Keep running: QuickPod can stop the pod only with your key (self_destroy_key)."
)

func (s *Service) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

// readyTimeout is AITK_PROVISION_READY_TIMEOUT (a Go duration), default 20 min.
func readyTimeout() time.Duration {
	if d, err := time.ParseDuration(strings.TrimSpace(getenv("AITK_PROVISION_READY_TIMEOUT"))); err == nil && d > 0 {
		return d
	}
	return defaultReady
}

func (s *Service) lockMachine(id string) func() {
	s.mu.Lock()
	if s.machineMu == nil {
		s.machineMu = map[string]*sync.Mutex{}
	}
	m := s.machineMu[id]
	if m == nil {
		m = &sync.Mutex{}
		s.machineMu[id] = m
	}
	s.mu.Unlock()
	m.Lock()
	return m.Unlock
}

// lockKind serializes calls to one provider. Lock order: a machine, then its
// provider; the watcher never holds a provider while locking a machine.
func (s *Service) lockKind(k Kind) func() {
	s.mu.Lock()
	if s.kindMu == nil {
		s.kindMu = map[Kind]*sync.Mutex{}
	}
	m := s.kindMu[k]
	if m == nil {
		m = &sync.Mutex{}
		s.kindMu[k] = m
	}
	s.mu.Unlock()
	m.Lock()
	return m.Unlock
}

func (s *Service) markRefreshed(id string, t time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.refreshed == nil {
		s.refreshed = map[string]time.Time{}
	}
	if t.IsZero() {
		delete(s.refreshed, id)
	} else {
		s.refreshed[id] = t
	}
}

func (s *Service) lastRefresh(id string) time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.refreshed[id]
}

func (s *Service) markSeen(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.seen == nil {
		s.seen = map[string]bool{}
	}
	s.seen[id] = true
}

func (s *Service) wasSeen(r *row) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seen[r.ID] || r.Status != StatusCreating
}

// noteRetry pauses the watcher for k after a rate limit.
func (s *Service) noteRetry(k Kind, err error) {
	var e *Error
	if !errors.As(err, &e) || (e.Class != ClassRateLimited && e.RetryAfter <= 0) {
		return
	}
	d := e.RetryAfter
	if d <= 0 {
		d = 10 * time.Second
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pausedTil == nil {
		s.pausedTil = map[Kind]time.Time{}
	}
	s.pausedTil[k] = s.now().Add(d)
}

func (s *Service) paused(k Kind, now time.Time) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return now.Before(s.pausedTil[k])
}

func (s *Service) forgetMachine(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.refreshed, id)
	delete(s.seen, id)
	delete(s.machineMu, id)
}

func bad(msg string) error      { return v1api.Errorf(v1api.CodeInvalidRequest, msg) }
func conflict(msg string) error { return v1api.Errorf(v1api.CodeConflict, msg) }

// row reads a machine by id or live name; 404 when there is none.
func (s *Service) row(ctx context.Context, idOrName string) (*row, error) {
	r, err := getRow(ctx, idOrName)
	if errors.Is(err, errNoRow) {
		return nil, v1api.Errorf(v1api.CodeNotFound, "no machine "+strconv.Quote(idOrName))
	}
	return r, err
}

func (s *Service) emit(ctx context.Context, r *row) {
	events.Emit(ctx, events.MachineState, "", s.machine(r))
}

// save persists r when anything changed since before, and emits
// machine.state when its status, problem, base_url, price or name did.
func (s *Service) save(ctx context.Context, r *row, before row) error {
	if reflect.DeepEqual(before, *r) {
		return nil
	}
	r.UpdatedAt = s.now().UnixMilli()
	if err := updateRow(ctx, r); err != nil {
		return err
	}
	if before.Status != r.Status || before.Problem != r.Problem || before.BaseURL != r.BaseURL || before.Name != r.Name ||
		!samePrice(before.Price, r.Price) || !samePrice(before.StoppedPrice, r.StoppedPrice) {
		s.emit(ctx, r)
	}
	return nil
}

func samePrice(a, b *float64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func capsOf(k Kind) Caps {
	if c, ok := Lookup(k); ok {
		return c.Info().Caps
	}
	return Caps{}
}

// actions are what the machine row offers now (Machine.actions).
func (s *Service) actions(r *row) []string {
	out := []string{}
	if r.EndedAt == nil && r.ProviderID != "" && r.Simulated == Simulated.Load() {
		caps := capsOf(r.Provider)
		switch r.Status {
		case StatusRunning, StatusReady:
			if caps.StopStart && r.Pending != pendingStop {
				out = append(out, "stop")
			}
		case StatusStopped:
			if caps.StopStart {
				out = append(out, "start")
			}
		}
		if r.Status != StatusTerminating && r.Status != StatusTerminated {
			out = append(out, "terminate")
		}
	}
	if canForget(r) {
		out = append(out, "forget")
	}
	return out
}

// canForget: nothing can be billing (ended, or never created at the provider).
func canForget(r *row) bool {
	return r.EndedAt != nil || (r.ProviderID == "" && r.Status == StatusFailed)
}

// ---- providers ----

// Templates lists k's templates (QuickPod launches need one).
func (s *Service) Templates(ctx context.Context, k Kind) ([]Template, error) {
	c, key, err := requireKey(ctx, k)
	if err != nil {
		return nil, err
	}
	t, err := c.Templates(ctx, WireKey(key))
	if err != nil {
		s.noteRetry(k, err)
		return nil, scrub(err, key)
	}
	if t == nil {
		t = []Template{}
	}
	return t, nil
}

// ---- launch ----

var (
	requestIDRe = regexp.MustCompile(`^[A-Za-z0-9_-]{8,64}$`)
	originRe    = regexp.MustCompile(`^https?://(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(:[0-9]{1,5})?$`)
)

func validName(n string) error {
	if n == "" || utf8.RuneCountInString(n) > 64 {
		return bad("name must be 1 to 64 characters")
	}
	if strings.IndexFunc(n, unicode.IsControl) >= 0 {
		return bad("name can't contain control characters")
	}
	return nil
}

func validURL(raw string) bool {
	if strings.ContainsAny(raw, " \t\r\n") || len(raw) > 2048 {
		return false
	}
	u, err := url.Parse(raw)
	return err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host != ""
}

// whenDone resolves LaunchRequest.when_done (MASTER §3.9.4). Omitted, it is
// destroy when a remote keeps the results and destroy is possible (on
// QuickPod only with the account key), else keep, with a warning when only
// the missing QuickPod key stood in the way.
func whenDone(k Kind, req LaunchRequest) (WhenDone, string, error) {
	switch req.WhenDone {
	case "":
		if req.CopyRemote == "" {
			return WhenKeep, "", nil
		}
		if k == QuickPod && !req.SelfDestroyKey {
			return WhenKeep, warnQuickPodKeep, nil
		}
		return WhenDestroy, "", nil
	case WhenKeep:
		return WhenKeep, "", nil
	case WhenStop, WhenDestroy:
		if req.CopyRemote == "" {
			return "", "", v1api.Errorf(v1api.CodeInvalidConfig, "When done needs a remote so results are kept")
		}
		if k == QuickPod && !req.SelfDestroyKey {
			return "", "", v1api.Errorf(v1api.CodeInvalidConfig, "QuickPod can stop the pod only with your key; tick self_destroy_key or pick keep")
		}
		return req.WhenDone, "", nil
	}
	return "", "", bad("when_done must be keep, stop or destroy")
}

// defaultName is "<gpu> on <provider>", numbered when a live machine has it.
func defaultName(ctx context.Context, k Kind, o Offer, count int) (string, error) {
	gpu := o.GPU
	if gpu == "" {
		gpu = "GPU"
	}
	if count > 1 {
		gpu = fmt.Sprintf("%d× %s", count, gpu)
	}
	base := gpu + " on " + Label(k)
	if r := []rune(base); len(r) > 60 {
		base = string(r[:60])
	}
	for n := 1; n < 1000; n++ {
		name := base
		if n > 1 {
			name = base + " " + strconv.Itoa(n)
		}
		taken, err := nameTaken(ctx, name, "")
		if err != nil || !taken {
			return name, err
		}
	}
	return "", conflict("too many servers named " + strconv.Quote(base))
}

// Launch rents the offer and starts the image (PLAN §5 Launch, in order).
// It returns as soon as the provider accepted the create; the watcher takes
// the machine from creating to ready.
func (s *Service) Launch(ctx context.Context, req LaunchRequest, origin string) (Machine, bool, error) {
	k := req.Provider
	if _, ok := ParseKind(string(k)); !ok {
		return Machine{}, false, bad("provider must be runpod, vast or quickpod")
	}
	c, key, err := requireKey(ctx, k)
	if err != nil {
		return Machine{}, false, err
	}

	s.launchMu.Lock()
	locked := true
	defer func() {
		if locked {
			s.launchMu.Unlock()
		}
	}()

	if req.RequestID != "" {
		if !requestIDRe.MatchString(req.RequestID) {
			return Machine{}, false, bad("request_id must be 8 to 64 letters, digits, _ or -")
		}
		r, err := rowByRequestID(ctx, req.RequestID)
		if err == nil {
			return s.machine(r), false, nil
		}
		if !errors.Is(err, errNoRow) {
			return Machine{}, false, err
		}
	}

	// validation
	if !(req.MaxPricePerHour > 0) {
		return Machine{}, false, bad("max_price_per_hour is required: the most you agree to pay per hour")
	}
	if strings.TrimSpace(req.OfferID) == "" {
		return Machine{}, false, bad("offer_id is required (GET /v1/providers/" + string(k) + "/offers)")
	}
	req.Name = strings.TrimSpace(req.Name)
	if req.Name != "" {
		if err := validName(req.Name); err != nil {
			return Machine{}, false, err
		}
	}
	if req.Image == "" {
		req.Image = DefaultImage
	}
	if len(req.Image) > 1024 || strings.ContainsFunc(req.Image, unicode.IsSpace) {
		return Machine{}, false, bad("image must be one reference without spaces, at most 1024 characters")
	}
	if req.DiskGB == 0 {
		req.DiskGB = DefaultDiskGB
	}
	if req.DiskGB < 20 || req.DiskGB > 2000 {
		return Machine{}, false, bad("disk_gb must be 20 to 2000")
	}
	if req.GPUCount == 0 {
		req.GPUCount = 1
	}
	if req.GPUCount < 1 || req.GPUCount > 16 {
		return Machine{}, false, bad("gpu_count must be 1 to 16")
	}
	if k == RunPod {
		if req.Cloud == "" {
			req.Cloud = "secure"
		}
		if req.Cloud != "secure" && req.Cloud != "community" {
			return Machine{}, false, bad("cloud must be secure or community")
		}
	} else {
		req.Cloud = ""
	}
	if k == QuickPod {
		if req.TemplateID == "" {
			return Machine{}, false, bad("template_id is required on QuickPod: pick one of your templates")
		}
		ts, err := c.Templates(ctx, WireKey(key))
		if err != nil {
			s.noteRetry(k, err)
			return Machine{}, false, scrub(err, key)
		}
		if !slices.ContainsFunc(ts, func(t Template) bool { return t.ID == req.TemplateID }) {
			return Machine{}, false, bad("template_id " + strconv.Quote(req.TemplateID) + " isn't one of your QuickPod templates")
		}
	}
	if len(req.CORSOrigins) == 0 && origin != "" && originRe.MatchString(origin) {
		req.CORSOrigins = []string{origin}
	}
	for _, o := range req.CORSOrigins {
		if !originRe.MatchString(o) {
			return Machine{}, false, bad("cors_origins: " + strconv.Quote(o) + " is not scheme://host[:port]")
		}
	}
	if req.StudioURL != "" && !validURL(req.StudioURL) {
		return Machine{}, false, bad("studio_url must be an http(s) URL")
	}

	when, whenWarn, err := whenDone(k, req)
	if err != nil {
		return Machine{}, false, err
	}
	req.WhenDone = when

	// max_running
	sim := Simulated.Load()
	n, err := liveCount(ctx, k, sim)
	if err != nil {
		return Machine{}, false, err
	}
	limit, err := maxRunning(ctx, k)
	if err != nil {
		return Machine{}, false, err
	}
	if n >= limit {
		verb := "servers are"
		if n == 1 {
			verb = "server is"
		}
		return Machine{}, false, conflict(fmt.Sprintf("%d %s %s already running; terminate one or raise max_running", n, Label(k), verb))
	}

	// quote
	offer, err := s.quote(ctx, c, key, req.OfferID, OfferQuery{DiskGB: req.DiskGB, GPUCount: req.GPUCount, Cloud: req.Cloud})
	if err != nil {
		return Machine{}, false, err
	}
	if offer.PricePerHour > req.MaxPricePerHour {
		return Machine{}, false, priceRose(k, offer.PricePerHour, req.MaxPricePerHour)
	}
	if offer.MaxDiskGB != nil && req.DiskGB > *offer.MaxDiskGB {
		return Machine{}, false, bad(fmt.Sprintf("disk_gb %d is more than this offer's %d GB", req.DiskGB, *offer.MaxDiskGB))
	}
	count := offer.GPUCount
	if count <= 0 {
		count = req.GPUCount
	}
	if req.Name == "" {
		if req.Name, err = defaultName(ctx, k, offer, count); err != nil {
			return Machine{}, false, err
		}
	} else if taken, err := nameTaken(ctx, req.Name, ""); err != nil {
		return Machine{}, false, err
	} else if taken {
		return Machine{}, false, conflict("A server named " + strconv.Quote(req.Name) + " is already live; pick another name")
	}

	// env
	id := state.NewID(MachineIDPrefix)
	token := NewToken()
	env, warnings, err := BuildEnv(ctx, req, k, id, token, c.Info().Caps, key)
	if err != nil {
		return Machine{}, false, scrub(err, key)
	}
	if whenWarn != "" {
		warnings = append([]string{whenWarn}, warnings...)
	}
	envKeys := make([]string, 0, len(env))
	for name := range env {
		envKeys = append(envKeys, name)
	}
	slices.Sort(envKeys)

	// insert
	now := s.now().UnixMilli()
	price := offer.PricePerHour
	r := &row{
		ID: id, Provider: k, Label: LabelFor(id), Name: req.Name, Token: token, Status: StatusCreating,
		Spec: machineSpec{
			Offer: offer, GPUCount: count, Cloud: req.Cloud, DiskGB: req.DiskGB, Image: req.Image,
			TemplateID: req.TemplateID, WhenDone: req.WhenDone, SelfDestroyKey: req.SelfDestroyKey,
			CopiedRemote: req.CopyRemote, EnvKeys: envKeys, Warnings: warnings,
		},
		Price: &price, StoppedPrice: offer.StoppedPricePerHour, Simulated: sim, CreatedAt: now, UpdatedAt: now,
	}
	// the id the create used: QuickPod finds its pod by offer id and label
	r.Spec.Offer.ID, r.Spec.Offer.Host, r.Spec.Offer.Fit = req.OfferID, nil, nil
	if req.RequestID != "" {
		rid := req.RequestID
		r.RequestID = &rid
	}
	if err := insertRow(ctx, r); err != nil {
		return Machine{}, false, err
	}
	// Once the row exists, a client that goes away neither aborts a create
	// the provider may already have accepted nor skips recording its outcome.
	lctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), launchCallTimeout)
	defer cancel()
	s.launchMu.Unlock()
	locked = false

	// launch
	unlock := s.lockMachine(id)
	defer unlock()
	ref, err := c.Launch(lctx, WireKey(key), LaunchSpec{
		Label: r.Label, OfferID: req.OfferID, Image: req.Image, DiskGB: req.DiskGB, TemplateID: req.TemplateID, Env: env,
	})
	before := *r
	if err != nil {
		s.noteRetry(k, err)
		err = scrub(err, append([]string{key}, envValues(env)...)...)
		r.Status, r.Detail, r.EndedAt = StatusFailed, launchDetail(err), ms(s.now())
		if serr := s.save(lctx, r, before); serr != nil {
			log.Printf("provision: record failed launch %s: %v", id, serr)
		}
		return Machine{}, false, err
	}
	r.ProviderID = ref.ID
	r.UpdatedAt = s.now().UnixMilli()
	if err := updateRow(lctx, r); err != nil {
		return Machine{}, false, err
	}
	s.emit(lctx, r)
	return s.machine(r), true, nil
}

// envValues are the secrets a launch error must never echo.
func envValues(env map[string]string) []string {
	var out []string
	for _, v := range env {
		if len(v) >= 8 {
			out = append(out, v)
		}
	}
	return out
}

func launchDetail(err error) string {
	if _, msg, _, ok := V1(err); ok {
		return Clip(msg, 300)
	}
	var p *v1api.Problem
	if errors.As(err, &p) {
		return Clip(p.Message, 300)
	}
	return Clip(err.Error(), 300)
}

// ---- reads ----

// List returns machines newest first (live only with activeOnly).
func (s *Service) List(ctx context.Context, activeOnly bool) ([]Machine, error) {
	rows, err := listRows(ctx, activeOnly)
	if err != nil {
		return nil, err
	}
	out := make([]Machine, 0, len(rows))
	for _, r := range rows {
		out = append(out, s.machine(r))
	}
	return out, nil
}

// Get returns one machine; refresh re-reads it from the provider unless any
// caller did so in the last 5 s.
func (s *Service) Get(ctx context.Context, id string, refresh bool) (Machine, error) {
	r, err := s.row(ctx, id)
	if err != nil {
		return Machine{}, err
	}
	if refresh && r.EndedAt == nil {
		unlock := s.lockMachine(r.ID)
		defer unlock()
		if r, err = getRow(ctx, r.ID); err != nil {
			return Machine{}, err
		}
		if last := s.lastRefresh(r.ID); last.IsZero() || s.now().Sub(last) >= getCacheTTL {
			if err := s.refreshRow(ctx, r); err != nil {
				return Machine{}, err
			}
		}
	}
	return s.machine(r), nil
}

// RelayTarget is what the relay needs to reach a ready machine.
func (s *Service) RelayTarget(ctx context.Context, id string) (RelayTarget, error) {
	r, err := s.row(ctx, id)
	if err != nil {
		return RelayTarget{}, err
	}
	if r.Status != StatusReady || r.BaseURL == "" || CheckEndpoint(r.Provider, r.BaseURL, r.Simulated) != nil {
		return RelayTarget{}, conflict(r.Name + " isn't ready yet")
	}
	return RelayTarget{MachineID: r.ID, BaseURL: r.BaseURL, Token: r.Token, Simulated: r.Simulated}, nil
}

// Link hands out what the studio saves as a Server: only for a ready machine
// that enforces its token.
func (s *Service) Link(ctx context.Context, id string) (MachineLink, error) {
	r, err := s.row(ctx, id)
	if err != nil {
		return MachineLink{}, err
	}
	if r.Status != StatusReady || r.BaseURL == "" {
		return MachineLink{}, conflict(r.Name + " isn't ready yet")
	}
	if r.Problem == ProblemOpenServer {
		return MachineLink{}, conflict(detailOpen)
	}
	return MachineLink{Name: r.Name, BaseURL: r.BaseURL, Token: r.Token, Secure: strings.HasPrefix(r.BaseURL, "https://"),
		RelayPath: "/v1/machines/" + r.ID + "/relay"}, nil
}

// Logs reads the container log tail, clipped to its last 256 KiB.
func (s *Service) Logs(ctx context.Context, id string, tail int) (MachineLogs, error) {
	r, err := s.row(ctx, id)
	if err != nil {
		return MachineLogs{}, err
	}
	if r.Simulated != Simulated.Load() {
		return MachineLogs{}, conflict(splitMessage(r))
	}
	if r.ProviderID == "" {
		return MachineLogs{}, conflict(Label(r.Provider) + " hasn't listed this machine yet")
	}
	c, key, err := requireKey(ctx, r.Provider)
	if err != nil {
		return MachineLogs{}, err
	}
	if tail <= 0 {
		tail = 200
	}
	tail = min(tail, logsTailMax)
	unlock := s.lockKind(r.Provider)
	text, err := c.Logs(ctx, WireKey(key), r.ProviderID, tail)
	unlock()
	if err != nil {
		s.noteRetry(r.Provider, err)
		return MachineLogs{}, scrub(err, key)
	}
	out := MachineLogs{Text: text, Source: string(r.Provider)}
	if r.Simulated {
		out.Source = "simulated"
	}
	if len(out.Text) > logsMax {
		cut := len(out.Text) - logsMax
		for cut < len(out.Text) && !utf8.RuneStart(out.Text[cut]) {
			cut++
		}
		out.Text, out.Truncated = out.Text[cut:], true
	}
	return out, nil
}

// ---- changes ----

// Update renames a machine (unique among live machines) or stores the token
// the template set, which clears token_mismatch and re-probes at once.
func (s *Service) Update(ctx context.Context, id string, p MachinePatch) (Machine, error) {
	r, err := s.row(ctx, id)
	if err != nil {
		return Machine{}, err
	}
	unlock := s.lockMachine(r.ID)
	defer unlock()
	if r, err = getRow(ctx, r.ID); err != nil {
		return Machine{}, err
	}
	before := *r
	if p.Name != nil {
		name := strings.TrimSpace(*p.Name)
		if err := validName(name); err != nil {
			return Machine{}, err
		}
		if r.EndedAt == nil {
			if taken, err := nameTaken(ctx, name, r.ID); err != nil {
				return Machine{}, err
			} else if taken {
				return Machine{}, conflict("A server named " + strconv.Quote(name) + " is already live; pick another name")
			}
		}
		r.Name = name
	}
	reprobe := false
	if p.Token != nil {
		tok := strings.TrimSpace(*p.Token)
		if tok == "" || len(tok) > 512 || strings.ContainsFunc(tok, func(c rune) bool { return unicode.IsSpace(c) || unicode.IsControl(c) }) {
			return Machine{}, bad("token must be one line of at most 512 characters")
		}
		r.Token = tok
		if r.Problem == ProblemTokenMismatch {
			r.Problem, r.Detail = "", ""
		}
		reprobe = true
	}
	if err := s.save(ctx, r, before); err != nil {
		return Machine{}, err
	}
	if reprobe && (r.Status == StatusRunning || r.Status == StatusReady) && len(r.Endpoints) > 0 && r.Simulated == Simulated.Load() {
		before := *r
		r.Status = StatusRunning
		s.applyProbe(ctx, r, before, s.now())
		if err := s.save(ctx, r, before); err != nil {
			return Machine{}, err
		}
	}
	return s.machine(r), nil
}

func (s *Service) Stop(ctx context.Context, id string) (Machine, error) {
	return s.act(ctx, id, "stop", false)
}

func (s *Service) Start(ctx context.Context, id string) (Machine, error) {
	return s.act(ctx, id, "start", false)
}

// Terminate deletes the machine at its provider. The server answering this
// call is refused unless confirmSelf.
func (s *Service) Terminate(ctx context.Context, id string, confirmSelf bool) (Machine, error) {
	return s.act(ctx, id, "terminate", confirmSelf)
}

func splitMessage(r *row) string {
	if r.Simulated {
		return "This machine was simulated; it can't be used while simulate mode is off."
	}
	return detailRealInSim
}

func (s *Service) act(ctx context.Context, id, action string, confirmSelf bool) (Machine, error) {
	r, err := s.row(ctx, id)
	if err != nil {
		return Machine{}, err
	}
	unlock := s.lockMachine(r.ID)
	defer unlock()
	if r, err = getRow(ctx, r.ID); err != nil {
		return Machine{}, err
	}
	if r.Simulated != Simulated.Load() {
		return Machine{}, conflict(splitMessage(r))
	}
	if action == "terminate" {
		if r.isSelf() && !confirmSelf {
			return Machine{}, bad("This is the server you are using. Send confirm_self.")
		}
		if r.Status == StatusTerminating || r.EndedAt != nil {
			return s.machine(r), nil // already asked, or gone
		}
		if r.ProviderID == "" {
			return Machine{}, conflict(Label(r.Provider) + " hasn't listed this machine yet; try again in a minute, or delete it in the " + Label(r.Provider) + " console")
		}
	}
	if !slices.Contains(s.actions(r), action) {
		return Machine{}, conflict(fmt.Sprintf("%s isn't possible while %s is %s", action, r.Name, r.Status))
	}
	c, key, err := requireKey(ctx, r.Provider)
	if err != nil {
		return Machine{}, err
	}
	unlockKind := s.lockKind(r.Provider)
	switch action {
	case "stop":
		err = c.Stop(ctx, WireKey(key), r.ProviderID)
	case "start":
		err = c.Start(ctx, WireKey(key), r.ProviderID)
	case "terminate":
		err = c.Terminate(ctx, WireKey(key), r.ProviderID)
	}
	unlockKind()
	if err != nil {
		s.noteRetry(r.Provider, err)
		return Machine{}, scrub(err, key)
	}
	before := *r
	switch action {
	case "stop":
		r.Pending, r.Status = pendingStop, StatusStopping
	case "start":
		r.Pending, r.Status, r.RunningAt = pendingStart, StatusStarting, nil
		if ownedProblem(r.Problem) {
			r.Problem, r.Detail = "", ""
		}
	case "terminate":
		r.Pending, r.Status = pendingTerminate, StatusTerminating
	}
	if r.Detail == detailSelfStopped {
		r.Detail = ""
	}
	if err := s.save(ctx, r, before); err != nil {
		return Machine{}, err
	}
	s.markRefreshed(r.ID, time.Time{}) // due on the next tick
	return s.machine(r), nil
}

// Forget deletes the row. While the machine may still bill it is refused
// unless force, which leaves it running at the provider.
func (s *Service) Forget(ctx context.Context, id string, force bool) error {
	r, err := s.row(ctx, id)
	if err != nil {
		return err
	}
	unlock := s.lockMachine(r.ID)
	defer unlock()
	if r, err = getRow(ctx, r.ID); err != nil {
		return err
	}
	if !canForget(r) {
		if !force {
			return conflict(r.Name + " may still be billing at " + Label(r.Provider) + ": terminate it first, or forget it with force")
		}
		log.Printf("provision: %s (%s) forgotten here; it keeps running and billing at %s", r.Name, r.ID, Label(r.Provider))
	}
	if err := deleteRow(ctx, r.ID); err != nil {
		return err
	}
	s.forgetMachine(r.ID)
	events.Emit(ctx, events.MachineDeleted, "", map[string]string{"id": r.ID})
	return nil
}

// ---- refresh: the provider's view plus this server's overlays ----

// ownedProblem: problems refresh computes (and therefore clears). Others
// (cuda_failed, pull_slow, image_pull_failed) belong to the image checks and
// survive a refresh.
func ownedProblem(p Problem) bool {
	switch p {
	case ProblemTokenMismatch, ProblemOpenServer, ProblemNoPort, ProblemNotReady, ProblemUnreachable,
		ProblemNeverListed, ProblemProvider, ProblemNotRefreshed:
		return true
	}
	return false
}

// splitRow handles a row of the other mode (a real machine while simulating,
// or a simulated one after a restart without simulate) with no provider
// call; true when it did.
func (s *Service) splitRow(ctx context.Context, r *row) (bool, error) {
	if r.Simulated == Simulated.Load() {
		return false, nil
	}
	before := *r
	if r.Simulated {
		r.Status, r.EndedAt, r.Pending, r.BaseURL, r.Endpoints = StatusTerminated, ms(s.now()), "", "", []Endpoint{}
		r.Problem, r.Detail = "", detailSimEnded
	} else {
		r.Problem, r.Detail = ProblemNotRefreshed, detailRealInSim
	}
	return true, s.save(ctx, r, before)
}

// refreshRow reads one machine at its provider and applies it. The caller
// holds the machine lock. Provider outages leave the row as it was.
func (s *Service) refreshRow(ctx context.Context, r *row) error {
	s.markRefreshed(r.ID, s.now())
	if done, err := s.splitRow(ctx, r); done {
		return err
	}
	c, ok := Lookup(r.Provider)
	if !ok {
		return nil
	}
	key, source, err := storedKey(ctx, r.Provider)
	if err != nil || source == "" {
		return err
	}
	unlock := s.lockKind(r.Provider)
	inst, err := c.Get(ctx, WireKey(key), Ref{ID: r.ProviderID, OfferID: r.Spec.Offer.ID, Label: r.Label})
	unlock()
	switch {
	case err == nil:
		return s.apply(ctx, r, &inst, false)
	case ClassOf(err) == ClassNotFound:
		return s.apply(ctx, r, nil, true)
	default:
		s.noteRetry(r.Provider, err)
		return s.apply(ctx, r, nil, false)
	}
}

// apply maps the provider's view (inst, or gone) onto r with the overlays
// and timeouts, persists it and emits machine.state on a change. With
// neither, only the timeouts run.
func (s *Service) apply(ctx context.Context, r *row, inst *Instance, gone bool) error {
	before := *r
	now := s.now()
	switch {
	case gone:
		s.applyGone(r, now)
	case inst != nil:
		s.applyInstance(ctx, r, *inst, before, now)
	}
	s.applyTimeouts(r, now)
	return s.save(ctx, r, before)
}

// applyGone: the provider no longer has it. A create that was never listed
// (a QuickPod create returns no id) is not gone: it may still appear
// (never_listed after 5 min).
func (s *Service) applyGone(r *row, now time.Time) {
	if r.ProviderID == "" {
		if r.Status == StatusCreating && now.Sub(time.UnixMilli(r.CreatedAt)) > neverListedAfter {
			r.Status, r.Problem, r.Detail = StatusFailed, ProblemNeverListed, neverListedDetail(r.Provider)
		}
		return
	}
	s.markEnded(r, now)
}

// neverListedDetail names the provider; only QuickPod has the console link.
func neverListedDetail(k Kind) string {
	if k == QuickPod {
		return detailNeverListed
	}
	return "The create at " + Label(k) + " never reported a server and none appeared. Check the " + Label(k) + " console; it may be billing."
}

func (s *Service) markEnded(r *row, now time.Time) {
	if r.Pending == pendingTerminate || r.Status == StatusTerminating {
		r.Detail = ""
	} else {
		r.Detail = "It's gone at " + Label(r.Provider) + ": it was deleted outside this server."
	}
	r.Status, r.EndedAt, r.Pending, r.BaseURL, r.Endpoints = StatusTerminated, ms(now), "", "", []Endpoint{}
	if ownedProblem(r.Problem) {
		r.Problem = ""
	}
}

func (s *Service) applyInstance(ctx context.Context, r *row, inst Instance, before row, now time.Time) {
	s.markSeen(r.ID)
	if r.ProviderID == "" && inst.ProviderID != "" {
		r.ProviderID = inst.ProviderID
	}
	r.ProviderStatus = inst.ProviderStatus
	if inst.Location != "" {
		r.Spec.Offer.Location = inst.Location
	}
	if r.Spec.Offer.GPU == "" && inst.GPU != "" {
		r.Spec.Offer.GPU = inst.GPU
	}
	if inst.PricePerHour != nil {
		p := *inst.PricePerHour
		r.Price = &p
	}
	if inst.StoppedPricePerHour != nil {
		p := *inst.StoppedPricePerHour
		r.StoppedPrice = &p
	}
	r.Endpoints = append([]Endpoint{}, inst.Endpoints...)
	if inst.Status == StatusTerminated {
		s.markEnded(r, now)
		return
	}

	foreign := r.Problem != "" && !ownedProblem(r.Problem)
	problem, detail := Problem(""), ""
	status := inst.Status
	switch status {
	case StatusStopped:
		r.RunningAt = nil
		switch {
		case r.Pending == pendingStart:
			status = StatusStarting // just asked; the provider has not moved yet
		case r.Pending == pendingStop:
			r.Pending = ""
		case before.Status != StatusStopped:
			detail = detailSelfStopped
		default:
			detail = r.Detail
		}
		r.BaseURL = ""
	case StatusRunning:
		if r.Pending == pendingStart {
			r.Pending = ""
		}
		if r.Pending == pendingStop {
			status = StatusStopping
		} else if r.RunningAt == nil {
			r.RunningAt = ms(now)
		}
	case StatusFailed:
		problem, detail = ProblemProvider, Clip(inst.Detail, 300)
		if detail == "" {
			detail = Label(r.Provider) + " reports " + strconv.Quote(inst.ProviderStatus)
		}
	case StatusUnknown:
		detail = Label(r.Provider) + " reports " + strconv.Quote(inst.ProviderStatus)
	default: // creating, starting, stopping
		r.RunningAt, r.BaseURL = nil, ""
	}
	if r.Pending == pendingTerminate {
		status = StatusTerminating
	}
	r.Status, r.Problem, r.Detail = status, problem, detail

	if status == StatusRunning {
		if inst.PortMissing {
			if r.RunningAt != nil && now.Sub(time.UnixMilli(*r.RunningAt)) > noPortGrace {
				r.Problem, r.Detail = ProblemNoPort, detailNoPort
			}
		} else if len(r.Endpoints) > 0 {
			s.applyProbe(ctx, r, before, now)
		}
	}
	if r.Problem == "" && foreign {
		r.Problem, r.Detail = before.Problem, before.Detail
	}
}

// applyProbe asks the machine whether it is ready (r.Status is running).
func (s *Service) applyProbe(ctx context.Context, r *row, before row, now time.Time) {
	// only the endpoints the provider's rules allow ever get the token
	eps := FilterEndpoints(r.Provider, r.Endpoints, r.Simulated)
	var res ProbeResult
	if DefaultProber != nil {
		res = DefaultProber.Probe(ctx, eps, r.Token, r.Simulated)
	}
	wasReady := before.Status == StatusReady || before.Problem == ProblemUnreachable
	switch {
	case res.TokenOK:
		r.Status, r.LastOKAt, r.BaseURL = StatusReady, ms(now), res.BaseURL
		if r.ReadyAt == nil {
			r.ReadyAt = ms(now)
		}
		if r.BaseURL == "" && len(eps) > 0 {
			r.BaseURL = eps[0].URL
		}
		if res.AuthEnforced != nil && !*res.AuthEnforced {
			r.Problem, r.Detail = ProblemOpenServer, detailOpen
		} else {
			r.Problem, r.Detail = "", ""
		}
	case res.TokenRejected:
		r.Problem, r.Detail = ProblemTokenMismatch, detailMismatch
	case wasReady:
		if r.LastOKAt != nil && now.Sub(time.UnixMilli(*r.LastOKAt)) > unreachableAfter {
			r.Problem = ProblemUnreachable
			r.Detail = "It stopped answering at " + before.BaseURL + " " + now.Sub(time.UnixMilli(*r.LastOKAt)).Round(time.Minute).String() + " ago."
		} else {
			// a missed probe or two: still ready until 5 min without an answer
			r.Status, r.Problem, r.Detail = StatusReady, before.Problem, before.Detail
		}
	}
}

// applyTimeouts: never ready 20 min after the create is failed/not_ready_timeout
// (still refreshed, and it recovers if it answers later).
func (s *Service) applyTimeouts(r *row, now time.Time) {
	if r.EndedAt != nil || r.ReadyAt != nil {
		return
	}
	switch r.Status {
	case StatusCreating, StatusStarting, StatusRunning, StatusFailed:
	default:
		return
	}
	if r.Problem != "" && r.Problem != ProblemNotReady {
		return // a more specific problem (token_mismatch, no_port, provider_error, never_listed) says more
	}
	if r.Status == StatusFailed && r.Problem == "" {
		return
	}
	if now.Sub(time.UnixMilli(r.CreatedAt)) > readyTimeout() {
		r.Status, r.Problem, r.Detail = StatusFailed, ProblemNotReady, detailNotReady
	}
}
