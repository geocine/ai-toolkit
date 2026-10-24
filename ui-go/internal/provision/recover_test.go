package provision

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// orphan puts an aitk instance this server does not track at k's stub
// provider, running with the endpoint running() gives it.
func (e *testEnv) orphan(k Kind, id, label string, env map[string]string, created time.Time) {
	e.t.Helper()
	c := e.clouds[k]
	price := 0.44
	c.mu.Lock()
	c.insts[id] = &Instance{ProviderID: id, Label: label, Status: StatusCreating, ProviderStatus: "CREATED", GPU: "RTX 4090",
		GPUCount: 1, VRAMGB: 24, Location: "EU", PricePerHour: &price, CreatedAt: created, Env: env}
	c.mu.Unlock()
	c.running(id)
}

func TestRecoverInstances(t *testing.T) {
	e := newEnv(t)
	tracked := e.launch(RunPod, nil)
	e.orphan(RunPod, "podorphan", "aitk-orphan1", map[string]string{"AI_TOOLKIT_AUTH": "aitk_orphan_token_0001"}, t0.Add(-time.Hour))
	e.orphan(RunPod, "podsecret", "aitk-orphan2", map[string]string{"AI_TOOLKIT_AUTH": "{{ RUNPOD_SECRET_aitk_token }}"}, t0.Add(-2*time.Hour))
	e.orphan(RunPod, "podother", "jupyter", nil, t0)
	e.orphan(RunPod, "podgone", "aitk-gone", nil, t0)
	e.clouds[RunPod].set("podgone", func(in *Instance) { in.Status = StatusTerminated })

	items, err := e.s.Instances(e.ctx, RunPod)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, v := range items {
		mid := ""
		if v.MachineID != nil {
			mid = *v.MachineID
		}
		got = append(got, v.ProviderID+"|"+mid+"|"+map[bool]string{true: "token", false: "-"}[v.TokenKnown])
	}
	// the tracked pod has no created_at at the stub, so it sorts last
	want := []string{"podorphan||token", "podsecret||-", tracked.ProviderID + "|" + tracked.ID + "|-"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("instances %v, want %v", got, want)
	}
	b, _ := json.Marshal(items)
	if strings.Contains(string(b), "aitk_orphan_token_0001") || strings.Contains(string(b), "RUNPOD_SECRET") {
		t.Fatalf("instances leak env: %s", b)
	}
	if items[0].GPU != "RTX 4090" || items[0].PricePerHour == nil || items[0].CreatedAt == nil || items[0].Status != StatusRunning {
		t.Fatalf("instance view %+v", items[0])
	}

	// no key: the usual refusal
	if _, err := e.s.Instances(e.ctx, Vast); codeOf(err) != "invalid_request" {
		t.Fatalf("no key: %v", err)
	}
	e.setKey(Vast, "key-vast-0001")
	e.clouds[Vast].listErr = Errorf(Vast, ClassRateLimited, 429, "slow down")
	if _, err := e.s.Instances(e.ctx, Vast); codeOf(err) != "rate_limited" {
		t.Fatalf("provider error: %v", err)
	}
}

func TestAdoptRecoversToken(t *testing.T) {
	e := newEnv(t)
	e.setKey(RunPod, "key-runpod-0001")
	const token = "aitk_orphan_token_0001"
	e.prober.accept = token
	e.orphan(RunPod, "podorphan", "aitk-ORPHAN0001abcd", map[string]string{
		"AI_TOOLKIT_AUTH": token, "AITK_MACHINE_ID": "mch_ORPHAN0001abcd", "AITK_WHEN_DONE": "destroy",
		"AITK_SYNC_REMOTE": "r2", "AITK_RUNPOD_API_KEY": "rpa_self_destroy_key",
	}, t0.Add(-time.Hour))

	m, err := e.s.Adopt(e.ctx, RunPod, AdoptRequest{ProviderID: "podorphan"})
	if err != nil {
		t.Fatal(err)
	}
	if m.ID != "mch_ORPHAN0001abcd" || m.Label != "aitk-ORPHAN0001abcd" || m.Name != "RTX 4090 on RunPod" || m.ProviderID != "podorphan" {
		t.Fatalf("adopted %+v", m)
	}
	if m.Status != StatusReady || m.Problem != nil || m.BaseURL == nil {
		t.Fatalf("adopted machine not ready: %s %v %s", m.Status, m.Problem, m.Detail)
	}
	if m.WhenDone != WhenDestroy || m.CopiedRemote != "r2" || !m.SelfDestroyKey || m.PricePerHour == nil || *m.PricePerHour != 0.44 ||
		!m.CreatedAt.Equal(t0) || m.VRAMGB != 24 || m.GPUCount != 1 {
		t.Fatalf("adopted fields %+v", m)
	}
	r := e.row(m.ID)
	if r.Token != token || strings.Contains(string(mustJSON(t, r.Spec)), "rpa_self_destroy_key") {
		t.Fatalf("row token %q spec %+v", r.Token, r.Spec)
	}
	l, err := e.s.Link(e.ctx, m.ID)
	if err != nil || l.Token != token {
		t.Fatalf("link %+v %v", l, err)
	}
	if !strings.Contains(strings.Join(r.Spec.EnvKeys, ","), "AITK_SYNC_REMOTE") {
		t.Fatalf("env keys %v", r.Spec.EnvKeys)
	}

	// tracked now: listed with its machine, and a second adopt is refused
	items, err := e.s.Instances(e.ctx, RunPod)
	if err != nil || len(items) != 1 || items[0].MachineID == nil || *items[0].MachineID != m.ID {
		t.Fatalf("instances after adopt %+v %v", items, err)
	}
	if _, err := e.s.Adopt(e.ctx, RunPod, AdoptRequest{ProviderID: "podorphan"}); codeOf(err) != "conflict" {
		t.Fatalf("second adopt: %v", err)
	}
	// the watcher keeps it ready
	if m := e.refresh(m.ID); m.Status != StatusReady {
		t.Fatalf("after refresh %s %v", m.Status, m.Problem)
	}
}

func TestAdoptNeedsToken(t *testing.T) {
	e := newEnv(t)
	e.setKey(QuickPod, "key-quickpod-0001")
	e.setKey(RunPod, "key-runpod-0001")
	e.orphan(QuickPod, "qpuuid1", "aitk-qp1", nil, time.Time{})
	e.orphan(RunPod, "podsecret", "aitk-rp1", map[string]string{"AI_TOOLKIT_AUTH": "{{ RUNPOD_SECRET_aitk_token }}"}, time.Time{})

	_, err := e.s.Adopt(e.ctx, QuickPod, AdoptRequest{ProviderID: "qpuuid1"})
	if codeOf(err) != "invalid_request" || !strings.Contains(err.Error(), "QuickPod doesn't return") {
		t.Fatalf("QuickPod without a token: %v", err)
	}
	// a secret reference is not the token
	if _, err := e.s.Adopt(e.ctx, RunPod, AdoptRequest{ProviderID: "podsecret"}); codeOf(err) != "invalid_request" {
		t.Fatalf("RunPod secret reference: %v", err)
	}

	e.prober.accept = "aitk_typed_token_0001"
	m, err := e.s.Adopt(e.ctx, QuickPod, AdoptRequest{ProviderID: "qpuuid1", Name: "Old QuickPod box", Token: " aitk_typed_token_0001 "})
	if err != nil {
		t.Fatal(err)
	}
	if m.Status != StatusReady || m.Name != "Old QuickPod box" || !strings.HasPrefix(m.ID, "mch_") || m.WhenDone != WhenKeep || m.SelfDestroyKey {
		t.Fatalf("QuickPod adopt %+v", m)
	}

	// a wrong token is token_mismatch, fixed with PATCH like a template token
	m, err = e.s.Adopt(e.ctx, RunPod, AdoptRequest{ProviderID: "podsecret", Token: "aitk_wrong"})
	if err != nil {
		t.Fatal(err)
	}
	if problemOf(m) != ProblemTokenMismatch {
		t.Fatalf("wrong token: %s %v", m.Status, m.Problem)
	}
}

func TestAdoptRefusals(t *testing.T) {
	e := newEnv(t)
	e.setKey(Vast, "key-vast-0001")
	e.orphan(Vast, "101", "aitk-v1", map[string]string{"AI_TOOLKIT_AUTH": "aitk_vast_token_0001", "AITK_MACHINE_ID": "mch_taken000000001"}, t0)
	e.orphan(Vast, "102", "notours", map[string]string{"AI_TOOLKIT_AUTH": "aitk_vast_token_0002"}, t0)
	existing := e.launch(Vast, func(r *LaunchRequest) { r.Name = "Taken name" })

	for _, c := range []struct {
		in   AdoptRequest
		code string
	}{
		{AdoptRequest{}, "invalid_request"},
		{AdoptRequest{ProviderID: "a b"}, "invalid_request"},
		{AdoptRequest{ProviderID: "999"}, "not_found"},
		{AdoptRequest{ProviderID: "102"}, "not_found"}, // not aitk-labelled
		{AdoptRequest{ProviderID: "101", Token: "two\nlines"}, "invalid_request"},
		{AdoptRequest{ProviderID: "101", Name: "Taken name"}, "conflict"},
		{AdoptRequest{ProviderID: existing.ProviderID}, "conflict"}, // already tracked
	} {
		if _, err := e.s.Adopt(e.ctx, Vast, c.in); codeOf(err) != c.code {
			t.Errorf("%+v: %v, want %s", c.in, err, c.code)
		}
	}
	if _, err := e.s.Adopt(e.ctx, "aws", AdoptRequest{ProviderID: "1"}); codeOf(err) != "invalid_request" {
		t.Errorf("unknown kind: %v", err)
	}

	// an id from the env that a row already has (here an ended one) is not reused
	if _, err := insertForTest(e, "mch_taken000000001"); err != nil {
		t.Fatal(err)
	}
	m, err := e.s.Adopt(e.ctx, Vast, AdoptRequest{ProviderID: "101"})
	if err != nil {
		t.Fatal(err)
	}
	if m.ID == "mch_taken000000001" || m.Label != "aitk-v1" {
		t.Fatalf("adopted %s (%s)", m.ID, m.Label)
	}
}

// insertForTest stores an ended machine row with id.
func insertForTest(e *testEnv, id string) (*row, error) {
	now := e.clock.Now().UnixMilli()
	r := &row{ID: id, Provider: Vast, Label: LabelFor(id), Name: "old", Token: "x", Status: StatusTerminated,
		CreatedAt: now, UpdatedAt: now, EndedAt: &now}
	return r, insertRow(e.ctx, r)
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
