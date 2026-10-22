// Package notify pushes what happens on the server to the user's phone (or
// any webhook) so an unattended run needs no babysitting: job started,
// resumed, samples, checkpoints synced, completed, failed, interrupted, lease
// lost, and the when-done countdown (internal/lifecycle). Sinks are ntfy,
// Discord webhooks, Telegram bots and plain JSON webhooks, configured from
// AITK_NOTIFY at boot and at runtime through /v1/notify. Delivery never
// blocks the server: messages queue per sink and retry with backoff.
// See docs/headless.md, "Notifications".
package notify

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
)

// Notification kinds (what AITK_NOTIFY_EVENTS filters on).
const (
	KindStarted     = "started"
	KindResumed     = "resumed"
	KindSample      = "sample"
	KindCheckpoint  = "checkpoint"
	KindCompleted   = "completed"
	KindFailed      = "failed"
	KindInterrupted = "interrupted"
	KindLeaseLost   = "lease_lost"
	KindSyncFailed  = "sync_failed"
	KindLifecycle   = "lifecycle"
	KindTest        = "test" // POST /v1/notify/test; never filtered
)

// Kinds lists every filterable kind, in the order the docs give them.
var Kinds = []string{KindStarted, KindResumed, KindSample, KindCheckpoint, KindCompleted, KindFailed,
	KindInterrupted, KindLeaseLost, KindSyncFailed, KindLifecycle}

// Levels say how loud a message is (ntfy priority, Discord colour).
const (
	LevelInfo  = "info"
	LevelOK    = "ok"
	LevelWarn  = "warn"
	LevelError = "error"
)

// Message is one notification, before a sink formats it.
type Message struct {
	Kind    string    `json:"kind"`
	Level   string    `json:"level"`
	Title   string    `json:"title"`
	Body    string    `json:"body"`
	URL     string    `json:"url,omitempty"` // tap-through link
	JobID   string    `json:"job_id,omitempty"`
	JobName string    `json:"job_name,omitempty"`
	At      time.Time `json:"at"`
	Image   []byte    `json:"-"` // JPEG, attached where the sink can
}

// Sink is a configured destination. URL holds its secret (webhook token, bot
// token, private topic) and is never returned by the API.
type Sink struct {
	ID        string    `json:"id"`
	URL       string    `json:"url"`
	Source    string    `json:"source"` // "env" (AITK_NOTIFY, replaced at every boot) or "api"
	CreatedAt time.Time `json:"created_at"`
}

// Config is the stored configuration (fork setting notify.config).
type Config struct {
	Sinks []Sink `json:"sinks"`
	// Events limits the kinds sent; nil sends every kind.
	Events      []string `json:"events"`
	PublicURL   string   `json:"public_url"`
	StudioURL   string   `json:"studio_url"`
	SampleEvery int      `json:"sample_every_s"` // min seconds between sample notifications per job (0 = default)
}

// DefaultSampleEvery spaces sample notifications of one job.
const DefaultSampleEvery = 15 * time.Minute

const configKey = "notify.config"

var (
	cfgMu  sync.Mutex
	cfg    Config
	loaded bool // cfg was read from the fork database
	// localStudio: this server serves the studio at /studio (main sets it),
	// so links can point there when only the public URL is known
	localStudio bool
)

// SetLocalStudio records that the studio is served at <public url>/studio/.
func SetLocalStudio(on bool) {
	cfgMu.Lock()
	localStudio = on
	cfgMu.Unlock()
}

// ensureLoadedLocked reads the stored configuration once. Caller holds cfgMu.
func ensureLoadedLocked() {
	if loaded || state.DB == nil {
		return
	}
	loaded = true
	stored, err := load(context.Background())
	if err != nil {
		log.Printf("notify: stored config unreadable, starting empty: %v", err)
		return
	}
	cfg = stored
}

// Current returns a copy of the configuration.
func Current() Config {
	cfgMu.Lock()
	defer cfgMu.Unlock()
	ensureLoadedLocked()
	c := cfg
	c.Sinks = slices.Clone(cfg.Sinks)
	c.Events = slices.Clone(cfg.Events)
	return c
}

func load(ctx context.Context) (Config, error) {
	var c Config
	v, ok, err := state.Setting(ctx, configKey)
	if err != nil || !ok {
		return c, err
	}
	err = json.Unmarshal([]byte(v), &c)
	return c, err
}

// Update applies fn to the stored configuration and persists it.
func Update(ctx context.Context, fn func(*Config) error) (Config, error) {
	cfgMu.Lock()
	defer cfgMu.Unlock()
	ensureLoadedLocked()
	next := cfg
	next.Sinks = slices.Clone(cfg.Sinks)
	next.Events = slices.Clone(cfg.Events)
	if err := fn(&next); err != nil {
		return cfg, err
	}
	b, err := json.Marshal(next)
	if err != nil {
		return cfg, err
	}
	if state.DB != nil {
		if err := state.SetSetting(ctx, configKey, string(b)); err != nil {
			return cfg, err
		}
	}
	cfg = next
	return cfg, nil
}

// ErrInvalid marks a configuration the API should answer 400 to.
var ErrInvalid = errors.New("invalid notify configuration")

// ValidEvents checks a kinds filter.
func ValidEvents(kinds []string) error {
	for _, k := range kinds {
		if !slices.Contains(Kinds, k) {
			return fmt.Errorf("%w: unknown event %q (one of %s)", ErrInvalid, k, strings.Join(Kinds, ", "))
		}
	}
	return nil
}

// SinkID is stable for a URL, so env sinks keep their id across restarts.
func SinkID(rawURL string) string {
	h := sha256.Sum256([]byte(rawURL))
	return "ntf_" + hex.EncodeToString(h[:])[:12]
}

// NewSink validates a sink URL and builds the record.
func NewSink(rawURL, source string) (Sink, error) {
	rawURL = strings.TrimSpace(rawURL)
	if _, err := parseSink(rawURL); err != nil {
		return Sink{}, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	return Sink{ID: SinkID(rawURL), URL: rawURL, Source: source, CreatedAt: time.Now().UTC().Truncate(time.Millisecond)}, nil
}

func env(key string) string { return strings.TrimSpace(os.Getenv(key)) }

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// applyEnv loads the stored configuration and lays the environment over it:
// AITK_NOTIFY replaces the env sinks (API-added ones stay), and
// AITK_NOTIFY_EVENTS, AITK_PUBLIC_URL, AITK_STUDIO_URL and
// AITK_NOTIFY_SAMPLE_EVERY, when set, replace their stored values.
func applyEnv(ctx context.Context) error {
	cfgMu.Lock()
	loaded = false // (re)read the database, which may be a new one (tests)
	cfgMu.Unlock()
	var envSinks []Sink
	for _, u := range splitList(os.Getenv("AITK_NOTIFY")) {
		s, err := NewSink(u, "env")
		if err != nil {
			// a typo in one sink must not cost the others (or the boot)
			log.Printf("notify: AITK_NOTIFY entry %s ignored: %v", redact(u), err)
			continue
		}
		envSinks = append(envSinks, s)
	}
	var events []string
	if v := env("AITK_NOTIFY_EVENTS"); v != "" {
		events = splitList(v)
		if slices.Contains(events, "all") || slices.Contains(events, "*") {
			events = nil
		} else if err := ValidEvents(events); err != nil {
			log.Printf("notify: AITK_NOTIFY_EVENTS: %v (sending every event)", err)
			events = nil
		}
	}
	_, err := Update(ctx, func(c *Config) error {
		kept := c.Sinks[:0]
		for _, s := range c.Sinks {
			if s.Source != "env" {
				kept = append(kept, s)
			}
		}
		c.Sinks = kept
		for _, s := range envSinks {
			if !slices.ContainsFunc(c.Sinks, func(x Sink) bool { return x.ID == s.ID }) {
				c.Sinks = append(c.Sinks, s)
			}
		}
		if env("AITK_NOTIFY_EVENTS") != "" {
			c.Events = events
		}
		if v := env("AITK_PUBLIC_URL"); v != "" {
			c.PublicURL = v
		}
		if v := env("AITK_STUDIO_URL"); v != "" {
			c.StudioURL = v
		}
		if v := env("AITK_NOTIFY_SAMPLE_EVERY"); v != "" {
			if d, err := time.ParseDuration(v); err == nil && d >= 0 {
				c.SampleEvery = int(d.Seconds())
			} else {
				log.Printf("notify: AITK_NOTIFY_SAMPLE_EVERY %q is not a duration (e.g. 15m)", v)
			}
		}
		return nil
	})
	if n := len(Current().Sinks); n > 0 {
		log.Printf("notify: %d sink(s) configured", n)
	}
	return err
}

// wanted reports whether kind passes the configured filter.
func wanted(c Config, kind string) bool {
	return kind == KindTest || c.Events == nil || slices.Contains(c.Events, kind)
}

func sampleEvery(c Config) time.Duration {
	if c.SampleEvery > 0 {
		return time.Duration(c.SampleEvery) * time.Second
	}
	return DefaultSampleEvery
}

// Send queues m for every sink (that its kind passes the filter for). It
// never blocks: a full queue drops the message with a log line.
func Send(m Message) {
	c := Current()
	if !wanted(c, m.Kind) {
		return
	}
	if m.At.IsZero() {
		m.At = time.Now().UTC().Truncate(time.Millisecond)
	}
	for _, s := range c.Sinks {
		enqueue(s, m)
	}
}

// Result is the outcome of a synchronous send (POST /v1/notify/test).
type Result struct {
	Sink   string `json:"sink"`
	Kind   string `json:"kind"`
	Target string `json:"target"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

// SendNow delivers m to the given sinks (all when ids is empty) and waits,
// with one attempt each: the API's test reports what the sink answered.
func SendNow(ctx context.Context, m Message, ids []string) []Result {
	if m.At.IsZero() {
		m.At = time.Now().UTC().Truncate(time.Millisecond)
	}
	var out []Result
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, s := range Current().Sinks {
		if len(ids) > 0 && !slices.Contains(ids, s.ID) {
			continue
		}
		wg.Add(1)
		go func(s Sink) {
			defer wg.Done()
			r := Result{Sink: s.ID}
			if p, err := parseSink(s.URL); err == nil {
				r.Kind, r.Target = p.kind, p.target
			}
			err := deliverOnce(ctx, s, m)
			r.OK = err == nil
			r.Detail = "delivered"
			if err != nil {
				r.Detail = err.Error()
			}
			mu.Lock()
			out = append(out, r)
			mu.Unlock()
		}(s)
	}
	wg.Wait()
	slices.SortFunc(out, func(a, b Result) int { return strings.Compare(a.Sink, b.Sink) })
	return out
}
