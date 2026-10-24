package provision

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/v1api"
)

var getenv = os.Getenv

// Key sources (ProviderView.key_source).
const (
	sourceAPI = "api"
	sourceEnv = "env"
	sourceSim = "sim"
)

// settingFor is the fork setting that holds k's key. Simulate keeps its fault
// names in a setting of their own, so a simulate run in a normal data dir
// never overwrites or deletes the real key.
func settingFor(k Kind) string {
	if Simulated.Load() {
		return SettingKey(k) + ".sim"
	}
	return SettingKey(k)
}

// storedKey is the launcher key of k and where it comes from. In simulate the
// stored value is a fault name or "" (WireKey adds the sim secret) and every
// kind counts as configured. Otherwise the setting wins, then the env
// fallback; source "" means no key.
func storedKey(ctx context.Context, k Kind) (key, source string, err error) {
	v, ok, err := state.Setting(ctx, settingFor(k))
	if err != nil {
		return "", "", err
	}
	if Simulated.Load() {
		return v, sourceSim, nil
	}
	if ok && v != "" {
		return v, sourceAPI, nil
	}
	if v := strings.TrimSpace(getenv(EnvKeyName(k))); v != "" {
		return v, sourceEnv, nil
	}
	return "", "", nil
}

// keyFingerprint identifies a key in cache keys without holding it. It is
// taken of the wire key, so a new sim secret is a new key too.
func keyFingerprint(wireKey string) string {
	h := sha256.Sum256([]byte(wireKey))
	return hex.EncodeToString(h[:8])
}

// requireKey returns the key of an enabled kind, or the refusal the API shows.
func requireKey(ctx context.Context, k Kind) (Cloud, string, error) {
	c, ok := Lookup(k)
	if !ok {
		return nil, "", v1api.Errorf(v1api.CodeNotFound, "unknown provider")
	}
	key, source, err := storedKey(ctx, k)
	if err != nil {
		return nil, "", err
	}
	if source == "" {
		return nil, "", v1api.Errorf(v1api.CodeInvalidRequest, "Add a "+Label(k)+" API key first")
	}
	return c, key, nil
}

// maxRunning is the per-provider cap on live machines: the setting, else
// AITK_PROVISION_MAX_RUNNING, else DefaultMaxRunning.
func maxRunning(ctx context.Context, k Kind) (int, error) {
	v, ok, err := state.Setting(ctx, MaxRunningSetting(k))
	if err != nil {
		return 0, err
	}
	if ok {
		if n, err := strconv.Atoi(v); err == nil && validMaxRunning(n) {
			return n, nil
		}
	}
	if n, err := strconv.Atoi(strings.TrimSpace(getenv("AITK_PROVISION_MAX_RUNNING"))); err == nil && validMaxRunning(n) {
		return n, nil
	}
	return DefaultMaxRunning, nil
}

func validMaxRunning(n int) bool { return n >= 1 && n <= 20 }

func (s *Service) view(ctx context.Context, c Cloud) (ProviderView, error) {
	k := c.Kind()
	key, source, err := storedKey(ctx, k)
	if err != nil {
		return ProviderView{}, err
	}
	mr, err := maxRunning(ctx, k)
	if err != nil {
		return ProviderView{}, err
	}
	sim := Simulated.Load()
	running, err := liveCount(ctx, k, sim)
	if err != nil {
		return ProviderView{}, err
	}
	v := ProviderView{KindInfo: c.Info(), Configured: source != "", KeySource: source, MaxRunning: mr, Running: running, Simulated: sim}
	if source == sourceAPI || source == sourceEnv {
		h := KeyHint(key)
		v.KeyHint = &h
	}
	return v, nil
}

// Providers makes no provider calls.
func (s *Service) Providers(ctx context.Context) ([]ProviderView, error) {
	out := []ProviderView{}
	for _, k := range Enabled() {
		c, _ := Lookup(k)
		v, err := s.view(ctx, c)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// SetKey verifies a new key with the provider and stores it only when the
// provider accepts it; max_running is 1..20.
func (s *Service) SetKey(ctx context.Context, k Kind, in ProviderUpdate) (ProviderView, error) {
	c, ok := Lookup(k)
	if !ok {
		return ProviderView{}, v1api.Errorf(v1api.CodeNotFound, "unknown provider")
	}
	if in.MaxRunning != nil && !validMaxRunning(*in.MaxRunning) {
		return ProviderView{}, v1api.Errorf(v1api.CodeInvalidRequest, "max_running must be 1 to 20")
	}
	if in.APIKey != nil {
		key := strings.TrimSpace(*in.APIKey)
		if key == "" && !Simulated.Load() {
			return ProviderView{}, v1api.Errorf(v1api.CodeInvalidRequest, "api_key is empty (DELETE removes the key)")
		}
		if len(key) > 512 || strings.ContainsAny(key, " \t\r\n") {
			return ProviderView{}, v1api.Errorf(v1api.CodeInvalidRequest, "api_key must be one line of at most 512 characters")
		}
		if _, err := c.Verify(ctx, WireKey(key)); err != nil {
			s.noteRetry(k, err)
			return ProviderView{}, scrub(err, key)
		}
		if err := state.SetSetting(ctx, settingFor(k), key); err != nil {
			return ProviderView{}, err
		}
		s.dropOffers(k)
	}
	if in.MaxRunning != nil {
		if err := state.SetSetting(ctx, MaxRunningSetting(k), strconv.Itoa(*in.MaxRunning)); err != nil {
			return ProviderView{}, err
		}
	}
	return s.view(ctx, c)
}

// RemoveKey drops the stored key; an env key still applies.
func (s *Service) RemoveKey(ctx context.Context, k Kind) error {
	if _, ok := Lookup(k); !ok {
		return v1api.Errorf(v1api.CodeNotFound, "unknown provider")
	}
	if err := deleteSetting(ctx, settingFor(k)); err != nil {
		return err
	}
	s.dropOffers(k)
	return nil
}

// Test checks the configured key; a rejected key is ok:false, not an error.
func (s *Service) Test(ctx context.Context, k Kind) (ProviderTest, error) {
	c, ok := Lookup(k)
	if !ok {
		return ProviderTest{}, v1api.Errorf(v1api.CodeNotFound, "unknown provider")
	}
	key, source, err := storedKey(ctx, k)
	if err != nil {
		return ProviderTest{}, err
	}
	if source == "" {
		return ProviderTest{OK: false, Detail: "No " + Label(k) + " API key yet."}, nil
	}
	acct, err := c.Verify(ctx, WireKey(key))
	if err != nil {
		s.noteRetry(k, err)
		err = scrub(err, key)
		if _, msg, _, ok := V1(err); ok {
			return ProviderTest{OK: false, Detail: msg}, nil
		}
		return ProviderTest{OK: false, Detail: err.Error()}, nil
	}
	detail := acct.Detail
	if detail == "" {
		detail = "The key works."
	}
	return ProviderTest{OK: true, Detail: detail, Credit: acct.Credit}, nil
}

// scrub removes the keys (and the wire form of each) from any error text, so
// a provider that echoes a key never leaks it through this server.
func scrub(err error, keys ...string) error {
	if err == nil {
		return nil
	}
	var secrets []string
	for _, k := range keys {
		if k != "" {
			secrets = append(secrets, WireKey(k), k)
		}
	}
	if sec, _ := simSecret.Load().(string); sec != "" {
		secrets = append(secrets, sec)
	}
	// The typed errors are rebuilt (dropping any wrapper) so errors.As on the
	// result can never reach the original text.
	var e *Error
	if errors.As(err, &e) {
		c := *e
		c.Message = Redact(c.Message, secrets...)
		return &c
	}
	var p *v1api.Problem
	if errors.As(err, &p) {
		return &v1api.Problem{Code: p.Code, Message: Redact(p.Message, secrets...)}
	}
	if red := Redact(err.Error(), secrets...); red != err.Error() {
		return &redactedError{msg: red, err: err}
	}
	return err
}

type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string { return e.msg }

// Unwrap keeps errors.Is (context.Canceled and friends) working; the chain
// holds no *Error or *v1api.Problem (scrub rebuilt those).
func (e *redactedError) Unwrap() error { return e.err }
