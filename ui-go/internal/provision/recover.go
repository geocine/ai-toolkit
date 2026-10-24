package provision

import (
	"context"
	"errors"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Recovery (PLAN §12): the aitk-labelled instances on a provider account,
// and adopting one this server does not track (the server that created it was lost, or
// it was forgotten with force). The handler asserts Recoverer on Default, so
// Manager fakes need not implement it.

// InstanceView is one aitk-labelled instance at a provider
// (GET /v1/providers/{kind}/instances). It never carries env values.
type InstanceView struct {
	ProviderID     string     `json:"provider_id"`
	Label          string     `json:"label"`
	Status         Status     `json:"status"` // the provider's view: never ready or terminating
	ProviderStatus string     `json:"provider_status"`
	GPU            string     `json:"gpu"`
	GPUCount       int        `json:"gpu_count"`
	PricePerHour   *float64   `json:"price_per_hour"`
	MachineID      *string    `json:"machine_id"`  // the live machine here that tracks it; null = adoptable
	TokenKnown     bool       `json:"token_known"` // the provider echoes its AI_TOOLKIT_AUTH, so adopt needs no token
	CreatedAt      *time.Time `json:"created_at"`  // null when the provider does not say
}

// AdoptRequest is POST /v1/providers/{kind}/adopt.
type AdoptRequest struct {
	ProviderID string `json:"provider_id"`
	Name       string `json:"name"`  // default "<gpu> on <provider>"
	Token      string `json:"token"` // the instance's AI_TOOLKIT_AUTH; required when the provider does not echo it (QuickPod)
}

// Recoverer finds and adopts instances (the service implements it).
type Recoverer interface {
	Instances(ctx context.Context, k Kind) ([]InstanceView, error)
	Adopt(ctx context.Context, k Kind, in AdoptRequest) (Machine, error)
}

var _ Recoverer = (*Service)(nil)

var machineIDRe = regexp.MustCompile(`^` + MachineIDPrefix + `_[A-Za-z0-9]{8,64}$`)

// echoedToken is the AI_TOOLKIT_AUTH an instance's env echo holds, or "".
// A RunPod secret reference ({{ RUNPOD_SECRET_x }}) is not the value.
func echoedToken(env map[string]string) string {
	t := strings.TrimSpace(env["AI_TOOLKIT_AUTH"])
	if strings.Contains(t, "{{") || validToken(t) != nil {
		return ""
	}
	return t
}

func validToken(tok string) error {
	if tok == "" || len(tok) > 512 || strings.ContainsFunc(tok, func(c rune) bool { return unicode.IsSpace(c) || unicode.IsControl(c) }) {
		return bad("token must be one line of at most 512 characters")
	}
	return nil
}

// listOwn lists k's aitk-labelled instances, terminated ones left out.
func (s *Service) listOwn(ctx context.Context, k Kind) ([]Instance, error) {
	c, key, err := requireKey(ctx, k)
	if err != nil {
		return nil, err
	}
	unlock := s.lockKind(k)
	insts, err := c.List(ctx, WireKey(key), LabelPrefix)
	unlock()
	if err != nil {
		s.noteRetry(k, err)
		return nil, scrub(err, key)
	}
	return slices.DeleteFunc(insts, func(in Instance) bool {
		return in.Status == StatusTerminated || in.ProviderID == "" || !strings.HasPrefix(in.Label, LabelPrefix)
	}), nil
}

// trackedBy maps provider ids (and, for rows not resolved yet, labels) to
// the live machine of k that tracks them, in the current mode.
func trackedBy(ctx context.Context, k Kind) (byID, byLabel map[string]string, err error) {
	rows, err := listRows(ctx, true)
	if err != nil {
		return nil, nil, err
	}
	byID, byLabel = map[string]string{}, map[string]string{}
	for _, r := range rows {
		if r.Provider != k || r.Simulated != Simulated.Load() {
			continue
		}
		if r.ProviderID != "" {
			byID[r.ProviderID] = r.ID
		} else {
			byLabel[r.Label] = r.ID
		}
	}
	return byID, byLabel, nil
}

// Instances lists every aitk-labelled instance on k's account, newest first;
// machine_id is set when a live machine here tracks it.
func (s *Service) Instances(ctx context.Context, k Kind) ([]InstanceView, error) {
	insts, err := s.listOwn(ctx, k)
	if err != nil {
		return nil, err
	}
	byID, byLabel, err := trackedBy(ctx, k)
	if err != nil {
		return nil, err
	}
	out := make([]InstanceView, 0, len(insts))
	for _, in := range insts {
		v := InstanceView{ProviderID: in.ProviderID, Label: in.Label, Status: in.Status, ProviderStatus: in.ProviderStatus,
			GPU: in.GPU, GPUCount: in.GPUCount, PricePerHour: in.PricePerHour, TokenKnown: echoedToken(in.Env) != ""}
		if id, ok := byID[in.ProviderID]; ok {
			v.MachineID = &id
		} else if id, ok := byLabel[in.Label]; ok {
			v.MachineID = &id
		}
		if !in.CreatedAt.IsZero() {
			t := in.CreatedAt.UTC().Truncate(time.Millisecond) // the precision of every /v1 time
			v.CreatedAt = &t
		}
		out = append(out, v)
	}
	slices.SortStableFunc(out, func(a, b InstanceView) int {
		switch {
		case a.CreatedAt == nil && b.CreatedAt == nil:
			return strings.Compare(a.ProviderID, b.ProviderID)
		case a.CreatedAt == nil:
			return 1
		case b.CreatedAt == nil:
			return -1
		}
		return b.CreatedAt.Compare(*a.CreatedAt)
	})
	return out, nil
}

// Adopt starts tracking an aitk-labelled instance this server does not
// track: the token comes from the provider's env echo (RunPod env, Vast
// extra_env) unless in.Token is given, and QuickPod, which has no echo,
// needs it. The machine keeps the id in its AITK_MACHINE_ID when that id is
// free here, is probed at once and is watched from then on; its created_at
// is the adoption. max_running is not applied: the instance already bills.
func (s *Service) Adopt(ctx context.Context, k Kind, in AdoptRequest) (Machine, error) {
	if _, ok := ParseKind(string(k)); !ok {
		return Machine{}, bad("provider must be runpod, vast or quickpod")
	}
	pid := strings.TrimSpace(in.ProviderID)
	if pid == "" || len(pid) > 256 || strings.ContainsFunc(pid, func(c rune) bool { return unicode.IsSpace(c) || unicode.IsControl(c) }) {
		return Machine{}, bad("provider_id is required (GET /v1/providers/" + string(k) + "/instances)")
	}
	name := strings.TrimSpace(in.Name)
	if name != "" {
		if err := validName(name); err != nil {
			return Machine{}, err
		}
	}
	token := strings.TrimSpace(in.Token)
	if token != "" {
		if err := validToken(token); err != nil {
			return Machine{}, err
		}
	}

	insts, err := s.listOwn(ctx, k)
	if err != nil {
		return Machine{}, err
	}
	i := slices.IndexFunc(insts, func(x Instance) bool { return x.ProviderID == pid })
	if i < 0 {
		return Machine{}, v1api.Errorf(v1api.CodeNotFound, "no aitk server "+strconv.Quote(pid)+" on your "+Label(k)+" account")
	}
	inst := insts[i]
	if err := s.notTracked(ctx, k, inst); err != nil {
		return Machine{}, err
	}
	if token == "" {
		if token = echoedToken(inst.Env); token == "" {
			if k == QuickPod {
				return Machine{}, bad("QuickPod doesn't return a server's env: paste its AI_TOOLKIT_AUTH as token")
			}
			return Machine{}, bad(Label(k) + " didn't return this server's AI_TOOLKIT_AUTH: paste it as token")
		}
	}

	s.launchMu.Lock()
	r, err := s.adoptRow(ctx, k, inst, name, token)
	s.launchMu.Unlock()
	if err != nil {
		return Machine{}, err
	}

	unlock := s.lockMachine(r.ID)
	defer unlock()
	now := s.now()
	before := *r
	s.markRefreshed(r.ID, now)
	s.applyInstance(ctx, r, inst, before, now)
	s.applyTimeouts(r, now)
	r.UpdatedAt = now.UnixMilli()
	if err := updateRow(ctx, r); err != nil {
		return Machine{}, err
	}
	s.emit(ctx, r)
	return s.machine(r), nil
}

// notTracked refuses an instance a live machine here already tracks.
func (s *Service) notTracked(ctx context.Context, k Kind, inst Instance) error {
	byID, byLabel, err := trackedBy(ctx, k)
	if err != nil {
		return err
	}
	id, ok := byID[inst.ProviderID]
	if !ok {
		id, ok = byLabel[inst.Label]
	}
	if ok {
		return conflict("This server already tracks it as " + id)
	}
	return nil
}

// adoptRow checks the instance is not tracked yet and inserts its row. The
// caller holds launchMu (names and the tracked check stay consistent).
func (s *Service) adoptRow(ctx context.Context, k Kind, inst Instance, name, token string) (*row, error) {
	if err := s.notTracked(ctx, k, inst); err != nil {
		return nil, err
	}
	var err error
	if name == "" {
		count := max(inst.GPUCount, 1)
		if name, err = defaultName(ctx, k, Offer{GPU: inst.GPU}, count); err != nil {
			return nil, err
		}
	} else if taken, err := nameTaken(ctx, name, ""); err != nil {
		return nil, err
	} else if taken {
		return nil, conflict("A server named " + strconv.Quote(name) + " is already live; pick another name")
	}

	id, err := adoptedID(ctx, inst.Env["AITK_MACHINE_ID"], inst.Label)
	if err != nil {
		return nil, err
	}
	// created_at is the adoption: the ready timeout counts from when this
	// server started watching (the instance list keeps the provider's time)
	now := s.now()
	env := inst.Env
	envKeys := make([]string, 0, len(env))
	for name := range env {
		envKeys = append(envKeys, name)
	}
	slices.Sort(envKeys)
	when := WhenDone(env["AITK_WHEN_DONE"])
	if when != WhenStop && when != WhenDestroy {
		when = WhenKeep
	}
	_, rpKey := env["AITK_RUNPOD_API_KEY"]
	_, qpKey := env["AITK_QUICKPOD_API_KEY"]
	r := &row{
		ID: id, Provider: k, ProviderID: inst.ProviderID, Label: inst.Label, Name: name, Token: token, Status: inst.Status,
		ProviderStatus: inst.ProviderStatus,
		Spec: machineSpec{
			Offer:    Offer{Provider: k, GPU: inst.GPU, GPUCount: inst.GPUCount, VRAMGB: inst.VRAMGB, Location: inst.Location},
			GPUCount: inst.GPUCount, WhenDone: when, SelfDestroyKey: rpKey || qpKey, CopiedRemote: env["AITK_SYNC_REMOTE"],
			EnvKeys: envKeys, Warnings: []string{}, ImageWarnings: []string{},
		},
		Price: inst.PricePerHour, StoppedPrice: inst.StoppedPricePerHour, Simulated: Simulated.Load(),
		CreatedAt: now.UnixMilli(), UpdatedAt: now.UnixMilli(),
	}
	if err := insertRow(ctx, r); err != nil {
		return nil, err
	}
	return r, nil
}

// adoptedID is the id the instance was launched with (its AITK_MACHINE_ID,
// else the one its label was made from) when it is free here (no row has
// it, not even an ended one), else a new one.
func adoptedID(ctx context.Context, fromEnv, label string) (string, error) {
	for _, id := range []string{strings.TrimSpace(fromEnv), MachineIDPrefix + "_" + strings.TrimPrefix(label, LabelPrefix)} {
		if !machineIDRe.MatchString(id) {
			continue
		}
		_, err := getRow(ctx, id)
		if errors.Is(err, errNoRow) {
			return id, nil
		}
		if err != nil {
			return "", err
		}
	}
	return state.NewID(MachineIDPrefix), nil
}
