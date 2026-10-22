// Package lifecycle is "when done" for rented GPU servers: once every queue
// is idle and the last job reached the trigger state, it counts down
// (AITK_WHEN_DONE_DELAY) and then stops or destroys the server through its
// provider (RunPod, Vast.ai, a custom command, or a process exit), but only
// after proving the finished jobs' results are on a remote. When they are
// not, it stays up and says so. The countdown is visible and controllable
// over /v1/lifecycle and as lifecycle.state events (which notify turns into
// "Destroying in 10 min" messages). See docs/headless.md, "When done".
package lifecycle

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// Event is the event type carrying Status on every change.
const Event = "lifecycle.state"

// Modes (AITK_WHEN_DONE).
const (
	ModeKeep    = "keep"
	ModeStop    = "stop"
	ModeDestroy = "destroy"
)

// Triggers (AITK_WHEN_DONE_ON).
const (
	OnCompleted = "completed"
	OnAny       = "any" // completed or failed
)

// States.
const (
	StateOff       = "off"       // mode keep
	StateWaiting   = "waiting"   // work is running, or the last job did not end in the trigger state
	StateCountdown = "countdown" // deadline set
	StateVerifying = "verifying" // checking (and pushing) results to remotes
	StateActing    = "acting"    // calling the provider
	StateHeld      = "held"      // results not safe off the server: staying up
	StateCancelled = "cancelled" // the user cancelled this countdown; re-arms after the next job
	StateDone      = "done"      // the provider accepted (or dry run)
	StateFailed    = "failed"    // the provider call failed: staying up
)

// Config is the when-done setting (fork setting lifecycle.config; env wins at boot).
type Config struct {
	Mode   string `json:"mode"`
	On     string `json:"on"`
	DelayS int    `json:"delay_s"`
	DryRun bool   `json:"dry_run"`
}

// DefaultDelay is AITK_WHEN_DONE_DELAY's default.
const DefaultDelay = 10 * time.Minute

// Status is GET /v1/lifecycle and the data of lifecycle.state.
type Status struct {
	Mode     string     `json:"mode"`
	On       string     `json:"on"`
	DelayS   int        `json:"delay_s"`
	DryRun   bool       `json:"dry_run"`
	State    string     `json:"state"`
	Deadline *time.Time `json:"deadline"`
	Reason   string     `json:"reason"`
	Provider string     `json:"provider"`
	// the last job to end (what armed the countdown)
	JobID     *string     `json:"job_id"`
	JobName   *string     `json:"job_name"`
	Results   []JobResult `json:"results"`
	UpdatedAt time.Time   `json:"updated_at"`
}

const configKey = "lifecycle.config"

// cancelledKey remembers a cancelled ending across restarts.
const cancelledKey = "lifecycle.cancelled"

// restoreCancel keeps a countdown cancelled before a restart cancelled.
func (m *Manager) restoreCancel() {
	if state.DB == nil || m.cfg.Mode == ModeKeep {
		return
	}
	if v, ok, _ := state.Setting(m.ctx, cancelledKey); ok && v != "" {
		m.stopKey = v
		m.st.State, m.st.Reason = StateCancelled, "cancelled (before the server restarted): it stays up until the next job finishes"
	}
}

// rememberCancel persists (or, with "", forgets) the cancelled ending.
func (m *Manager) rememberCancel(key string) {
	if state.DB == nil {
		return
	}
	if err := state.SetSetting(context.WithoutCancel(m.ctx), cancelledKey, key); err != nil {
		log.Printf("lifecycle: %v", err)
	}
}

// verifyTimeout bounds how long the results check may push and wait.
var verifyTimeout = 15 * time.Minute

// beforeAct hooks run (bounded) right before the provider is called:
// notify flushes its queue so "destroying now" reaches the phone.
var (
	hooksMu   sync.Mutex
	beforeAct []func(context.Context)
)

// OnBeforeAct registers a hook run before the server is stopped or destroyed.
func OnBeforeAct(fn func(context.Context)) {
	hooksMu.Lock()
	beforeAct = append(beforeAct, fn)
	hooksMu.Unlock()
}

// Manager holds the state machine; one per process (see Start).
type Manager struct {
	mu       sync.Mutex
	cfg      Config
	st       Status
	provider Provider
	armedKey string // the ended attempt the countdown is for
	stopKey  string // an attempt the user cancelled for, or that ended held/failed/done
	busy     bool   // verify/act running
	ctx      context.Context
	// lastLogged is the state last written to the log
	lastLogged string
	now        func() time.Time
}

var (
	mgrMu sync.Mutex
	mgr   *Manager
)

func current() *Manager {
	mgrMu.Lock()
	defer mgrMu.Unlock()
	return mgr
}

// ErrConflict marks a request that does not fit the current state (409).
var ErrConflict = errors.New("conflict")

// ErrInvalid marks a bad request (400).
var ErrInvalid = errors.New("invalid")

// reqError carries a message for the API and its kind (ErrConflict, ErrInvalid).
type reqError struct {
	kind error
	msg  string
}

func (e *reqError) Error() string { return e.msg }
func (e *reqError) Unwrap() error { return e.kind }

func failf(kind error, format string, args ...any) error {
	return &reqError{kind: kind, msg: fmt.Sprintf(format, args...)}
}

func normalize(c Config) (Config, error) {
	c.Mode = strings.ToLower(strings.TrimSpace(c.Mode))
	c.On = strings.ToLower(strings.TrimSpace(c.On))
	if c.Mode == "" {
		c.Mode = ModeKeep
	}
	if c.On == "" {
		c.On = OnCompleted
	}
	switch c.Mode {
	case ModeKeep, ModeStop, ModeDestroy:
	default:
		return c, failf(ErrInvalid, "mode must be keep, stop or destroy")
	}
	switch c.On {
	case OnCompleted, OnAny:
	default:
		return c, failf(ErrInvalid, "on must be completed or any")
	}
	if c.DelayS < 0 {
		return c, failf(ErrInvalid, "delay must not be negative")
	}
	return c, nil
}

func boolEnv(key string) (bool, bool) {
	switch strings.ToLower(env(key)) {
	case "1", "true", "yes", "on":
		return true, true
	case "0", "false", "no", "off":
		return false, true
	}
	return false, false
}

// configFromEnv lays AITK_WHEN_DONE, AITK_WHEN_DONE_DELAY,
// AITK_WHEN_DONE_ON and AITK_WHEN_DONE_DRY_RUN over the stored config.
func configFromEnv(c Config) Config {
	if v := env("AITK_WHEN_DONE"); v != "" {
		c.Mode = v
	}
	if v := env("AITK_WHEN_DONE_ON"); v != "" {
		c.On = v
	}
	if v := env("AITK_WHEN_DONE_DELAY"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			c.DelayS = int(d.Seconds())
		} else if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			c.DelayS = n * 60 // a bare number is minutes
		} else {
			log.Printf("lifecycle: AITK_WHEN_DONE_DELAY %q is not a duration (e.g. 10m)", v)
		}
	}
	if v, ok := boolEnv("AITK_WHEN_DONE_DRY_RUN"); ok {
		c.DryRun = v
	}
	return c
}

func loadConfig(ctx context.Context) Config {
	c := Config{DelayS: int(DefaultDelay.Seconds())}
	if state.DB != nil {
		if v, ok, err := state.Setting(ctx, configKey); err == nil && ok {
			json.Unmarshal([]byte(v), &c)
		}
	}
	c = configFromEnv(c)
	n, err := normalize(c)
	if err != nil {
		log.Printf("lifecycle: %v; keeping the server (mode keep)", err)
		n = Config{Mode: ModeKeep, On: OnCompleted, DelayS: int(DefaultDelay.Seconds())}
	}
	return n
}

func saveConfig(ctx context.Context, c Config) error {
	b, _ := json.Marshal(c)
	return state.SetSetting(ctx, configKey, string(b))
}

// tick is how often the trigger is re-evaluated (besides on job events).
var tick = 3 * time.Second

// Start creates the manager and runs it until ctx ends.
func Start(ctx context.Context) {
	m := newManager(ctx, loadConfig(ctx), detectProvider())
	m.restoreCancel()
	mgrMu.Lock()
	mgr = m
	mgrMu.Unlock()
	if m.cfg.Mode != ModeKeep {
		dry := ""
		if m.cfg.DryRun {
			dry = " (dry run)"
		}
		log.Printf("lifecycle: when done: %s via %s after %s, on %s%s", m.cfg.Mode, m.provider.Name(),
			time.Duration(m.cfg.DelayS)*time.Second, m.cfg.On, dry)
	}
	m.run(ctx)
}

func newManager(ctx context.Context, c Config, p Provider) *Manager {
	m := &Manager{cfg: c, provider: p, ctx: ctx, now: time.Now}
	m.st = Status{State: StateWaiting, Results: []JobResult{}}
	if c.Mode == ModeKeep {
		m.st.State, m.st.Reason = StateOff, "when done: keep the server"
	}
	m.fill()
	return m
}

func (m *Manager) run(ctx context.Context) {
	ch, cancel := events.Subscribe(events.Filter{Types: map[string]bool{
		events.JobState: true, events.JobAttempt: true, events.JobCreated: true, events.JobDeleted: true}})
	defer cancel()
	t := time.NewTicker(tick)
	defer t.Stop()
	m.evaluate()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ch:
			m.evaluate()
		case <-t.C:
			m.evaluate()
		}
	}
}

// fill copies the config into the status (caller holds mu or owns m).
func (m *Manager) fill() {
	m.st.Mode, m.st.On, m.st.DelayS, m.st.DryRun = m.cfg.Mode, m.cfg.On, m.cfg.DelayS, m.cfg.DryRun
	m.st.Provider = m.provider.Name()
	if m.st.Results == nil {
		m.st.Results = []JobResult{}
	}
}

// set changes the state and emits the event when anything visible changed.
// Caller holds mu.
func (m *Manager) set(st string, deadline *time.Time, reason string) {
	prev := m.st
	m.st.State, m.st.Deadline, m.st.Reason = st, deadline, reason
	m.fill()
	if prev.State == m.st.State && eqTime(prev.Deadline, m.st.Deadline) && prev.Reason == m.st.Reason &&
		prev.Mode == m.st.Mode && prev.DelayS == m.st.DelayS && prev.On == m.st.On && prev.DryRun == m.st.DryRun {
		return
	}
	m.emitLocked()
}

func (m *Manager) emitLocked() {
	m.st.UpdatedAt = m.now().UTC().Truncate(time.Millisecond)
	snap := m.snapshotLocked()
	// synchronous, so events keep their order (Emit never blocks on subscribers)
	events.Emit(context.WithoutCancel(m.ctx), Event, "", snap)
	if snap.State != m.lastLogged {
		m.lastLogged = snap.State
		log.Printf("lifecycle: %s: %s", snap.State, snap.Reason)
	}
}

func eqTime(a, b *time.Time) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return a.Equal(*b)
}

func (m *Manager) snapshotLocked() Status {
	s := m.st
	s.Results = append([]JobResult{}, m.st.Results...)
	if m.st.Deadline != nil {
		d := *m.st.Deadline
		s.Deadline = &d
	}
	return s
}

// Snapshot returns the current status.
func (m *Manager) Snapshot() Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.snapshotLocked()
}

// lastEnd is the newest ended job: what decides the trigger.
type lastEnd struct {
	job   *store.Job
	key   string // job id + attempt n: one countdown per ending
	state string // client state
	at    time.Time
}

// survey reads the jobs: whether any is queued/running/stopping, and the
// one that ended last.
func survey(ctx context.Context) (active *store.Job, last *lastEnd, err error) {
	jobs, err := store.FindJobs(ctx, store.JobQuery{OrderBy: `"created_at" ASC`})
	if err != nil {
		return nil, nil, err
	}
	for _, j := range jobs {
		switch j.Status {
		case "queued", "running", "stopping":
			if active == nil {
				active = j
			}
			continue
		}
		st := jobstate.FromStatus(j.Status, j.Step, j.Info)
		if st == jobstate.Draft {
			continue
		}
		at, n := j.UpdatedAt.Time, 0
		if a, err := state.LastAttempt(ctx, j.ID); err == nil && a != nil {
			n = a.N
			if a.EndedAt != nil {
				at = *a.EndedAt
			}
		}
		if last == nil || at.After(last.at) {
			last = &lastEnd{job: j, key: j.ID + "#" + strconv.Itoa(n), state: st, at: at}
		}
	}
	return active, last, nil
}

func (m *Manager) triggers(st string) bool {
	return st == jobstate.Completed || m.cfg.On == OnAny && st == jobstate.Failed
}

// evaluate moves the state machine on from what the jobs look like now.
func (m *Manager) evaluate() {
	ctx := m.ctx
	active, last, err := survey(ctx)
	if err != nil {
		if ctx.Err() == nil {
			log.Printf("lifecycle: %v", err)
		}
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cfg.Mode == ModeKeep {
		m.armedKey = ""
		m.set(StateOff, nil, "when done: keep the server")
		return
	}
	if m.busy {
		if active != nil && (m.st.State == StateVerifying || m.st.State == StateActing) {
			// act sees the state change and stands down before calling the provider
			m.set(StateWaiting, nil, "cancelled: job "+active.Name+" was queued")
		}
		return
	}
	if active != nil {
		reason := "job " + active.Name + " is " + jobstate.FromStatus(active.Status, active.Step, active.Info)
		if m.st.State == StateCountdown {
			reason = "countdown cancelled: job " + active.Name + " was queued"
		}
		// new work: whatever was decided for the previous ending is over
		if m.stopKey != "" {
			m.rememberCancel("")
		}
		m.armedKey, m.stopKey = "", ""
		m.st.JobID, m.st.JobName = nil, nil
		m.set(StateWaiting, nil, reason)
		return
	}
	if last == nil {
		if m.stopKey == "now" {
			return // an explicit /now with no jobs: its outcome stays
		}
		m.set(StateWaiting, nil, "no job has finished yet")
		return
	}
	id, name := last.job.ID, last.job.Name
	m.st.JobID, m.st.JobName = &id, &name
	if !m.triggers(last.state) {
		m.armedKey = ""
		m.set(StateWaiting, nil, fmt.Sprintf("last job %s ended %s (when done on %s)", name, last.state, m.cfg.On))
		return
	}
	if last.key == m.stopKey {
		return // cancelled, held, failed or done for this ending: stay as is
	}
	now := m.now()
	if m.st.State != StateCountdown || m.armedKey != last.key {
		m.armedKey = last.key
		d := now.Add(time.Duration(m.cfg.DelayS) * time.Second).UTC().Truncate(time.Second)
		m.st.Results = []JobResult{}
		m.set(StateCountdown, &d, fmt.Sprintf("job %s %s; %s in %s unless cancelled", name, last.state,
			actionWord(m.cfg.Mode), time.Duration(m.cfg.DelayS)*time.Second))
	}
	if m.st.Deadline != nil && !now.Before(*m.st.Deadline) {
		m.startActLocked()
	}
}

func actionWord(mode string) string {
	if mode == ModeStop {
		return "stopping the server"
	}
	return "destroying the server"
}

// startActLocked verifies results and acts in the background. Caller holds mu.
func (m *Manager) startActLocked() {
	m.busy = true
	key := m.armedKey
	m.st.Results = []JobResult{}
	m.set(StateVerifying, nil, "checking the finished jobs' results are on a remote")
	go m.act(key)
}

func (m *Manager) act(key string) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("lifecycle: panic: %v", r)
			m.mu.Lock()
			m.busy, m.stopKey = false, key
			m.set(StateFailed, nil, fmt.Sprintf("internal error: %v", r))
			m.mu.Unlock()
		}
	}()
	ctx, cancel := context.WithTimeout(m.ctx, verifyTimeout)
	results, safe := verifyAll(ctx)
	cancel()

	m.mu.Lock()
	m.st.Results = results
	if m.st.State != StateVerifying {
		// cancelled (or mode changed) while verifying
		m.busy = false
		m.emitLocked()
		m.mu.Unlock()
		return
	}
	mode, dry, p := m.cfg.Mode, m.cfg.DryRun, m.provider
	if !safe {
		m.busy, m.stopKey = false, key
		m.set(StateHeld, nil, "results only on this server, not "+strings.TrimSuffix(actionWord(mode), " the server")+": "+unsafeSummary(results))
		m.mu.Unlock()
		return
	}
	verb := "destroying"
	if mode == ModeStop {
		verb = "stopping"
	}
	reason := "results are safe on remotes; " + verb + " via " + p.Name()
	if dry {
		reason += " (dry run)"
	}
	m.set(StateActing, nil, reason)
	m.mu.Unlock()

	hooksMu.Lock()
	hooks := append([]func(context.Context){}, beforeAct...)
	hooksMu.Unlock()
	hctx, hcancel := context.WithTimeout(m.ctx, 15*time.Second)
	for _, h := range hooks {
		h(hctx)
	}
	hcancel()

	m.mu.Lock()
	if m.st.State != StateActing {
		m.busy = false // cancelled in the last seconds
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()

	var detail string
	var err error
	if dry {
		detail = "dry run: would " + map[string]string{ModeStop: "stop", ModeDestroy: "destroy"}[mode] + " via " + p.Name()
		log.Printf("lifecycle: %s", detail)
	} else if mode == ModeStop {
		detail, err = p.Stop(m.ctx)
	} else {
		detail, err = p.Destroy(m.ctx)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.busy, m.stopKey = false, key
	if err != nil {
		m.set(StateFailed, nil, "could not end the server, staying up: "+err.Error())
		return
	}
	m.set(StateDone, nil, detail)
}

func unsafeSummary(rs []JobResult) string {
	var parts []string
	for _, r := range rs {
		if !r.Safe {
			parts = append(parts, r.JobName+": "+r.Detail)
		}
	}
	return strings.Join(parts, "; ")
}

// verifyAll checks every job that ran and ended here.
func verifyAll(ctx context.Context) ([]JobResult, bool) {
	jobs, err := store.FindJobs(ctx, store.JobQuery{OrderBy: `"created_at" ASC`})
	if err != nil {
		return []JobResult{{JobName: "(jobs)", Detail: err.Error()}}, false
	}
	out := []JobResult{}
	safe := true
	for _, j := range jobs {
		switch jobstate.FromStatus(j.Status, j.Step, j.Info) {
		case jobstate.Completed, jobstate.Failed, jobstate.Stopped, jobstate.Interrupted:
		default:
			continue
		}
		r := verifyJob(ctx, j)
		safe = safe && r.Safe
		out = append(out, r)
	}
	return out, safe
}

// Get returns the status (zero Status when the manager is not running).
func Get() (Status, error) {
	m := current()
	if m == nil {
		return Status{}, errors.New("lifecycle is not running")
	}
	return m.Snapshot(), nil
}

// Cancel stops the countdown (or a pending verify); the server stays up until
// a new job runs and ends.
func Cancel() (Status, error) {
	m := current()
	if m == nil {
		return Status{}, errors.New("lifecycle is not running")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	switch m.st.State {
	case StateCountdown, StateVerifying, StateActing:
	default:
		return m.snapshotLocked(), failf(ErrConflict, "nothing to cancel (state %s)", m.st.State)
	}
	m.stopKey = m.armedKey
	m.rememberCancel(m.stopKey)
	m.set(StateCancelled, nil, "cancelled: the server stays up until the next job finishes")
	return m.snapshotLocked(), nil
}

// Extend pushes the deadline out by d.
func Extend(d time.Duration) (Status, error) {
	m := current()
	if m == nil {
		return Status{}, errors.New("lifecycle is not running")
	}
	if d <= 0 || d > 7*24*time.Hour {
		return Status{}, failf(ErrInvalid, "minutes must be between 1 and 10080")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.st.State != StateCountdown || m.st.Deadline == nil {
		return m.snapshotLocked(), failf(ErrConflict, "no countdown to extend (state %s)", m.st.State)
	}
	base := *m.st.Deadline
	if now := m.now(); base.Before(now) {
		base = now
	}
	next := base.Add(d).UTC().Truncate(time.Second)
	m.set(StateCountdown, &next, fmt.Sprintf("extended by %s", d))
	return m.snapshotLocked(), nil
}

// Now skips the countdown: verify the results and act at once. Refused while
// a job is queued or running, or with mode keep.
func Now() (Status, error) {
	m := current()
	if m == nil {
		return Status{}, errors.New("lifecycle is not running")
	}
	active, last, err := survey(m.ctx)
	if err != nil {
		return Status{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cfg.Mode == ModeKeep {
		return m.snapshotLocked(), failf(ErrConflict, "when done is keep; set mode stop or destroy first")
	}
	if m.busy {
		return m.snapshotLocked(), failf(ErrConflict, "already %s", m.st.State)
	}
	if active != nil {
		return m.snapshotLocked(), failf(ErrConflict, "job %s is %s", active.Name, active.Status)
	}
	m.armedKey = "now"
	if last != nil {
		m.armedKey = last.key
	}
	m.startActLocked()
	return m.snapshotLocked(), nil
}

// Patch changes the configuration at runtime (and persists it). A change
// restarts the decision: a running countdown starts over with the new delay.
type Patch struct {
	Mode   *string `json:"mode"`
	On     *string `json:"on"`
	DelayS *int    `json:"delay_s"`
	DryRun *bool   `json:"dry_run"`
}

// Update applies p.
func Update(ctx context.Context, p Patch) (Status, error) {
	m := current()
	if m == nil {
		return Status{}, errors.New("lifecycle is not running")
	}
	m.mu.Lock()
	c := m.cfg
	if p.Mode != nil {
		c.Mode = *p.Mode
	}
	if p.On != nil {
		c.On = *p.On
	}
	if p.DelayS != nil {
		c.DelayS = *p.DelayS
	}
	if p.DryRun != nil {
		c.DryRun = *p.DryRun
	}
	c, err := normalize(c)
	if err != nil {
		m.mu.Unlock()
		return Status{}, err
	}
	if err := saveConfig(ctx, c); err != nil {
		m.mu.Unlock()
		return Status{}, err
	}
	changed := c != m.cfg
	m.cfg = c
	if changed && !m.busy {
		// an explicit change decides afresh, even for an ending that was
		// cancelled or held: a countdown starts over with the new delay
		m.armedKey, m.stopKey = "", ""
		m.rememberCancel("")
		// not emitted: evaluate (below) publishes where it lands, so a
		// countdown that only changed its delay is not reported as cancelled
		m.st.State, m.st.Deadline, m.st.Reason = StateWaiting, nil, "configuration changed"
	} else if m.busy && c.Mode == ModeKeep {
		m.set(StateCancelled, nil, "when done set to keep")
	}
	m.mu.Unlock()
	m.evaluate()
	return m.Snapshot(), nil
}
