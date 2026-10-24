package provision

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// fakeCloud is a do-nothing adapter for registry tests.
type fakeCloud struct{ kind Kind }

func (f fakeCloud) Kind() Kind                                    { return f.kind }
func (f fakeCloud) Info() KindInfo                                { return KindInfo{Kind: f.kind, Label: Label(f.kind)} }
func (fakeCloud) Verify(context.Context, string) (Account, error) { return Account{}, nil }
func (fakeCloud) Offers(context.Context, string, OfferQuery) ([]Offer, error) {
	return nil, nil
}
func (fakeCloud) Quote(context.Context, string, string, OfferQuery) (Offer, error) {
	return Offer{}, nil
}
func (fakeCloud) Templates(context.Context, string) ([]Template, error) { return nil, nil }
func (fakeCloud) Launch(context.Context, string, LaunchSpec) (Ref, error) {
	return Ref{}, nil
}
func (fakeCloud) Get(context.Context, string, Ref) (Instance, error) { return Instance{}, nil }
func (fakeCloud) List(context.Context, string, string) ([]Instance, error) {
	return nil, nil
}
func (fakeCloud) Stop(context.Context, string, string) error                { return nil }
func (fakeCloud) Start(context.Context, string, string) error               { return nil }
func (fakeCloud) Terminate(context.Context, string, string) error           { return nil }
func (fakeCloud) Logs(context.Context, string, string, int) (string, error) { return "", nil }

// emptyRegistry gives a test the registry to itself and restores it after.
func emptyRegistry(t *testing.T) {
	t.Helper()
	regMu.Lock()
	saved := registry
	registry = map[Kind]Cloud{}
	regMu.Unlock()
	t.Cleanup(func() {
		regMu.Lock()
		registry = saved
		regMu.Unlock()
	})
}

func TestNewToken(t *testing.T) {
	seen := map[string]bool{}
	for range 1000 {
		tok := NewToken()
		if len(tok) != 48 || !strings.HasPrefix(tok, "aitk_") {
			t.Fatalf("token %q: len %d", tok, len(tok))
		}
		if _, err := base64.RawURLEncoding.DecodeString(tok[5:]); err != nil {
			t.Fatalf("token %q is not base64url: %v", tok, err)
		}
		if seen[tok] {
			t.Fatalf("duplicate token %q", tok)
		}
		seen[tok] = true
	}
}

func TestParseKind(t *testing.T) {
	for _, s := range []string{"runpod", "vast", "quickpod"} {
		if k, ok := ParseKind(s); !ok || string(k) != s {
			t.Errorf("ParseKind(%q) = %q, %v", s, k, ok)
		}
	}
	for _, s := range []string{"", "RunPod", "vast.ai", "lambda"} {
		if _, ok := ParseKind(s); ok {
			t.Errorf("ParseKind(%q) accepted", s)
		}
	}
}

func TestRegisterPanicsOnDuplicate(t *testing.T) {
	emptyRegistry(t)
	Register(fakeCloud{RunPod})
	defer func() {
		if recover() == nil {
			t.Fatal("a second runpod adapter must panic")
		}
	}()
	Register(fakeCloud{RunPod})
}

func TestLookupHonoursProvidersEnv(t *testing.T) {
	emptyRegistry(t)
	Register(fakeCloud{QuickPod})
	Register(fakeCloud{RunPod})
	restore := SetForTest(fakeCloud{Vast})

	t.Setenv("AITK_PROVISION_PROVIDERS", "")
	if got := Enabled(); !slices.Equal(got, []Kind{RunPod, Vast, QuickPod}) {
		t.Fatalf("all enabled: %v", got)
	}
	t.Setenv("AITK_PROVISION_PROVIDERS", " RunPod , vast")
	if got := Enabled(); !slices.Equal(got, []Kind{RunPod, Vast}) {
		t.Fatalf("kill switch: %v", got)
	}
	if _, ok := Lookup(QuickPod); ok {
		t.Fatal("quickpod must be disabled by AITK_PROVISION_PROVIDERS")
	}
	if c, ok := Lookup(Vast); !ok || c.Kind() != Vast {
		t.Fatalf("vast lookup: %v %v", c, ok)
	}

	restore()
	t.Setenv("AITK_PROVISION_PROVIDERS", "")
	if _, ok := Lookup(Vast); ok {
		t.Fatal("SetForTest restore must remove a kind that was not registered")
	}
	restore = SetForTest(fakeCloud{RunPod})
	restore()
	if _, ok := Lookup(RunPod); !ok {
		t.Fatal("SetForTest restore must put the earlier adapter back")
	}
}

func TestV1AuthReadsOnce(t *testing.T) {
	for msg, want := range map[string]string{
		"Invalid user key":  "Vast.ai rejected the API key.",
		"Unauthorized":      "Vast.ai rejected the API key.",
		"invalid api key.":  "Vast.ai rejected the API key.",
		"":                  "Vast.ai rejected the API key.",
		"key expired today": "Vast.ai rejected the API key: key expired today",
	} {
		if _, got, _, _ := V1(Errorf(Vast, ClassAuth, 401, "%s", msg)); got != want {
			t.Errorf("%q: %q, want %q", msg, got, want)
		}
	}
}

func TestV1EveryClass(t *testing.T) {
	cases := []struct {
		class  Class
		code   string
		status int
		text   string
	}{
		{ClassAuth, v1api.CodeProviderAuth, 400, "RunPod rejected the API key: nope"},
		{ClassCredit, v1api.CodeInsufficientCredit, 402, "RunPod says the balance is too low: nope"},
		{ClassCapacity, v1api.CodeNoCapacity, 409, "RunPod: that offer can't be rented right now (nope). Pick another."},
		{ClassConflict, v1api.CodeConflict, 409, "nope"},
		{ClassInvalid, v1api.CodeInvalidRequest, 400, "RunPod: nope"},
		{ClassUnsupported, v1api.CodeInvalidRequest, 400, "RunPod: nope"},
		{ClassNotFound, v1api.CodeNotFound, 404, "RunPod: nope"},
		{ClassRateLimited, v1api.CodeRateLimited, 429, "RunPod is rate limiting this server; try again in 10s"},
		{ClassUnavailable, v1api.CodeRemoteFailed, 502, "RunPod did not answer: nope"},
	}
	for _, c := range cases {
		err := fmt.Errorf("wrapped: %w", Errorf(RunPod, c.class, 0, "nope"))
		code, msg, ra, ok := V1(err)
		if !ok || code != c.code || msg != c.text || v1api.StatusOf(code) != c.status {
			t.Errorf("%s: %q %q %v (status %d)", c.class, code, msg, ok, v1api.StatusOf(code))
		}
		if ClassOf(err) != c.class {
			t.Errorf("ClassOf(%s) = %q", c.class, ClassOf(err))
		}
		if c.class == ClassRateLimited && ra != 10*time.Second {
			t.Errorf("rate limited default Retry-After %v", ra)
		}
	}
	e := &Error{Provider: Vast, Class: ClassRateLimited, Status: 429, Message: "slow down", RetryAfter: 2500 * time.Millisecond}
	if code, msg, ra, _ := V1(e); code != CodeRateLimited || ra != e.RetryAfter || !strings.HasPrefix(msg, "Vast.ai is rate limiting") || !strings.HasSuffix(msg, "3s") {
		t.Errorf("explicit Retry-After: %q %q %v", code, msg, ra)
	}
	if _, _, _, ok := V1(errors.New("plain")); ok {
		t.Error("a plain error is not a provider error")
	}
	if ClassOf(errors.New("plain")) != "" {
		t.Error("ClassOf(plain) must be empty")
	}
	if got := e.Error(); got != "vast rate_limited (HTTP 429): slow down" {
		t.Errorf("Error() = %q", got)
	}
}

func TestRedactClipKeyHintLabel(t *testing.T) {
	if got := Redact("key abcd1234 and abcd1234 and xyz", "abcd1234", "xyz", ""); got != "key [redacted] and [redacted] and xyz" {
		t.Errorf("Redact: %q (secrets under 4 chars are left alone)", got)
	}
	if got := Clip("  hello world  ", 5); got != "hello…" {
		t.Errorf("Clip: %q", got)
	}
	if got := Clip("short", 10); got != "short" {
		t.Errorf("Clip short: %q", got)
	}
	if got := KeyHint("rpa_0123456789WXYZ"); got != "…WXYZ" {
		t.Errorf("KeyHint: %q", got)
	}
	if got := KeyHint("short"); got != "…" {
		t.Errorf("KeyHint of a short key reveals nothing: %q", got)
	}
	if got := LabelFor("mch_01HZY3"); got != "aitk-01HZY3" {
		t.Errorf("LabelFor: %q", got)
	}
	if EnvKeyName(QuickPod) != "AITK_PROVISION_QUICKPOD_KEY" || SettingKey(Vast) != "provision.key.vast" ||
		MaxRunningSetting(RunPod) != "provision.max_running.runpod" {
		t.Error("setting and env names")
	}
	if Label(Vast) != "Vast.ai" || Label(QuickPod) != "QuickPod" || Label(RunPod) != "RunPod" {
		t.Error("labels")
	}
}

func TestStatusTransitional(t *testing.T) {
	for _, s := range []Status{StatusCreating, StatusStarting, StatusRunning, StatusStopping, StatusTerminating} {
		if !s.Transitional() {
			t.Errorf("%s should be transitional", s)
		}
	}
	for _, s := range []Status{StatusReady, StatusStopped, StatusTerminated, StatusFailed, StatusUnknown} {
		if s.Transitional() {
			t.Errorf("%s should not be transitional", s)
		}
	}
}

func TestWireKey(t *testing.T) {
	SetSimSecret("s3cr3t")
	t.Cleanup(func() { Simulated.Store(false); SetSimSecret("") })
	if WireKey("rpa_x") != "rpa_x" {
		t.Error("outside simulate the stored key is sent")
	}
	Simulated.Store(true)
	if WireKey("") != "s3cr3t" || WireKey("sim_full") != "s3cr3t.sim_full" {
		t.Errorf("simulate: %q %q", WireKey(""), WireKey("sim_full"))
	}
}

// The disk floor without a model and the launch default are one number.
func TestDiskFloorMatchesDefault(t *testing.T) {
	if hostfit.MinDiskGBUnknownModel != DefaultDiskGB {
		t.Fatalf("hostfit.MinDiskGBUnknownModel %d != DefaultDiskGB %d", hostfit.MinDiskGBUnknownModel, DefaultDiskGB)
	}
}

// The JSON names the studio's provisionTypes.ts reads.
func TestContractJSON(t *testing.T) {
	keys := func(v any) map[string]json.RawMessage {
		t.Helper()
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		m := map[string]json.RawMessage{}
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	caps := keys(Caps{ImageDigestRef: "unconfirmed", ReportsInet: true})
	for _, k := range []string{"image_digest_ref", "reports_inet", "reports_disk_bw", "env_flag_string"} {
		if _, ok := caps[k]; !ok {
			t.Errorf("caps lacks %s", k)
		}
	}
	offer := keys(Offer{ID: "o1"})
	if string(offer["host"]) != "null" || string(offer["fit"]) != "null" {
		t.Errorf("undecorated offer host/fit: %s %s", offer["host"], offer["fit"])
	}
	m := keys(Machine{})
	for _, k := range []string{"image_digest", "image_launch_ref", "image_warnings"} {
		if _, ok := m[k]; !ok {
			t.Errorf("machine lacks %s", k)
		}
	}
	if _, ok := keys(LaunchRequest{})["skip_image_check"]; !ok {
		t.Error("launch request lacks skip_image_check")
	}
	q := keys(OfferQuery{Filter: &hostfit.Filter{}})
	if _, ok := q["Filter"]; ok {
		t.Error("OfferQuery.Filter is set by the service and never serialised")
	}
	if _, ok := q["ImageInfo"]; ok {
		t.Error("OfferQuery.ImageInfo is set by the service and never serialised")
	}
	if ProblemCUDAFailed != "cuda_failed" || ProblemPullSlow != "pull_slow" || ProblemImagePullFailed != "image_pull_failed" {
		t.Error("image problem codes")
	}
}
