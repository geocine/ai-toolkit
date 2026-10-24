package v1machines

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/quickpod/quickpodmock"
	"github.com/geocine/aitoolkit/internal/provision/sim"
	"github.com/geocine/aitoolkit/internal/provision/vast"
)

// fakeRecoverer is the fake manager plus the recovery methods.
type fakeRecoverer struct {
	*fakeManager
	adopt provision.AdoptRequest
}

func (f *fakeRecoverer) Instances(_ context.Context, k provision.Kind) ([]provision.InstanceView, error) {
	f.kind = k
	return []provision.InstanceView{{ProviderID: "pod1", Label: "aitk-x", Status: provision.StatusRunning}}, f.note("Instances")
}

func (f *fakeRecoverer) Adopt(_ context.Context, k provision.Kind, in provision.AdoptRequest) (provision.Machine, error) {
	f.kind, f.adopt = k, in
	return machine("mch_adopted"), f.note("Adopt")
}

func TestRecoverRoutes(t *testing.T) {
	openState(t)
	f := &fakeRecoverer{fakeManager: &fakeManager{}}
	old := provision.Default
	provision.Default = f
	t.Cleanup(func() { provision.Default = old })
	mux := http.NewServeMux()
	Register(mux)

	w := call(t, mux, "GET", "/v1/providers/vast/instances", "")
	if w.Code != 200 || f.last() != "Instances" || f.kind != provision.Vast || !strings.Contains(w.Body.String(), `"items":[{"provider_id":"pod1"`) {
		t.Fatalf("instances %d %s (%s)", w.Code, w.Body, f.last())
	}
	w = call(t, mux, "POST", "/v1/providers/quickpod/adopt", `{"provider_id":"qp1","name":"old box","token":"aitk_t"}`)
	if w.Code != 201 || f.last() != "Adopt" || f.kind != provision.QuickPod || f.adopt != (provision.AdoptRequest{ProviderID: "qp1", Name: "old box", Token: "aitk_t"}) {
		t.Fatalf("adopt %d %s %+v", w.Code, w.Body, f.adopt)
	}
	for _, c := range []struct {
		method, path, body string
		opts               []func(*http.Request)
		status             int
		code               string
	}{
		{"GET", "/v1/providers/aws/instances", "", nil, 404, "not_found"},
		{"POST", "/v1/providers/aws/adopt", `{"provider_id":"1"}`, nil, 404, "not_found"},
		{"POST", "/v1/providers/runpod/adopt", `{`, nil, 400, "invalid_request"},
		// adopt hands this server a token: the open-server guard applies, the list does not need it
		{"POST", "/v1/providers/runpod/adopt", `{"provider_id":"p"}`, []func(*http.Request){fromRemote}, 403, "forbidden"},
	} {
		if w := call(t, mux, c.method, c.path, c.body, c.opts...); w.Code != c.status || errCode(t, w) != c.code {
			t.Errorf("%s %s: %d %s, want %d %s", c.method, c.path, w.Code, w.Body, c.status, c.code)
		}
	}
	if w := call(t, mux, "GET", "/v1/providers/runpod/instances", "", fromRemote); w.Code != 200 {
		t.Errorf("instances from a remote peer: %d", w.Code)
	}
	f.err = provision.Errorf(provision.RunPod, provision.ClassRateLimited, 429, "slow down")
	if w := call(t, mux, "GET", "/v1/providers/runpod/instances", ""); w.Code != 429 || w.Header().Get("Retry-After") == "" {
		t.Errorf("provider error: %d %v", w.Code, w.Header())
	}

	// a Manager without recovery
	provision.Default = &fakeManager{}
	if w := call(t, mux, "GET", "/v1/providers/runpod/instances", ""); w.Code != 404 {
		t.Errorf("no Recoverer: %d %s", w.Code, w.Body)
	}
}

// In simulate mode, per provider: launch a machine, lose it here (forget
// with force: it keeps running at the mock), find it in instances and
// adopt it back. RunPod and Vast give the token back from their env echo,
// QuickPod needs it typed (400 without). The adopted machine probes ready
// with the same id and token.
func TestAdoptSimulated(t *testing.T) {
	openState(t)
	t.Setenv("AITK_PROVISION_SIMULATE_DELAY", "0")
	nodeInner := http.NewServeMux()
	nodeInner.HandleFunc("GET /v1/health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"ok":true,"version":"test"}`)) })
	nodeInner.HandleFunc("GET /v1/system", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"gpus":[]}`)) })
	ctx, cancel := context.WithCancel(context.Background())
	stop, err := sim.Start(ctx, nodeInner)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() { stop(); cancel() })
	oldPace := vast.Pace
	vast.Pace = vast.NewLimiter(nowClock{})
	t.Cleanup(func() { vast.Pace = oldPace })
	old := provision.Default
	provision.Default = &provision.Service{}
	t.Cleanup(func() { provision.Default = old })
	mux := http.NewServeMux()
	Register(mux)

	do := func(t *testing.T, method, path, body string, want int, v any) {
		t.Helper()
		w := call(t, mux, method, path, body)
		if w.Code != want {
			t.Fatalf("%s %s: %d %s, want %d", method, path, w.Code, w.Body, want)
		}
		if v != nil {
			if err := json.Unmarshal(w.Body.Bytes(), v); err != nil {
				t.Fatal(err)
			}
		}
	}
	waitReady := func(t *testing.T, id string) provision.Machine {
		t.Helper()
		deadline := time.Now().Add(30 * time.Second)
		for {
			var m provision.Machine
			do(t, "GET", "/v1/machines/"+id+"?refresh=1", "", 200, &m)
			if m.Status == provision.StatusReady || time.Now().After(deadline) {
				return m
			}
			time.Sleep(200 * time.Millisecond)
		}
	}

	for _, k := range []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod} {
		t.Run(string(k), func(t *testing.T) {
			var offers struct{ Items []provision.Offer }
			do(t, "GET", "/v1/providers/"+string(k)+"/offers?limit=5", "", 200, &offers)
			if len(offers.Items) == 0 {
				t.Fatal("no offers")
			}
			o := offers.Items[0]
			req := map[string]any{"provider": k, "offer_id": o.ID, "max_price_per_hour": o.PricePerHour + 1}
			if k == provision.QuickPod {
				req["template_id"] = quickpodmock.AitkTemplateUUID
			}
			body, _ := json.Marshal(req)
			var m provision.Machine
			do(t, "POST", "/v1/machines", string(body), 201, &m)
			if m = waitReady(t, m.ID); m.Status != provision.StatusReady {
				t.Fatalf("launched machine %s %v %s", m.Status, m.Problem, m.Detail)
			}
			var l provision.MachineLink
			do(t, "POST", "/v1/machines/"+m.ID+"/link", "", 200, &l)
			do(t, "DELETE", "/v1/machines/"+m.ID+"?force=1", "", 204, nil)

			var list struct{ Items []provision.InstanceView }
			do(t, "GET", "/v1/providers/"+string(k)+"/instances", "", 200, &list)
			var found *provision.InstanceView
			for i := range list.Items {
				if list.Items[i].ProviderID == m.ProviderID {
					found = &list.Items[i]
				}
			}
			if found == nil || found.MachineID != nil || found.Label != m.Label {
				t.Fatalf("instances %+v, want %s untracked", list.Items, m.ProviderID)
			}

			// RunPod echoes the pod env and Vast extra_env; QuickPod returns no env.
			wantToken := map[provision.Kind]bool{provision.Vast: true, provision.QuickPod: false, provision.RunPod: true}[k]
			if found.TokenKnown != wantToken {
				t.Fatalf("token_known %v, want %v", found.TokenKnown, wantToken)
			}
			adopt := `{"provider_id":"` + m.ProviderID + `"}`
			if !found.TokenKnown {
				if w := call(t, mux, "POST", "/v1/providers/"+string(k)+"/adopt", adopt); w.Code != 400 || errCode(t, w) != "invalid_request" {
					t.Fatalf("adopt without a token: %d %s", w.Code, w.Body)
				}
				adopt = `{"provider_id":"` + m.ProviderID + `","token":"` + l.Token + `"}`
			}
			var a provision.Machine
			do(t, "POST", "/v1/providers/"+string(k)+"/adopt", adopt, 201, &a)
			if a.ID != m.ID || a.Label != m.Label || a.Provider != k || !a.Simulated {
				t.Fatalf("adopted %+v, launched %s", a, m.ID)
			}
			if a = waitReady(t, a.ID); a.Status != provision.StatusReady || a.Problem != nil {
				t.Fatalf("adopted machine %s %v %s", a.Status, a.Problem, a.Detail)
			}
			var l2 provision.MachineLink
			do(t, "POST", "/v1/machines/"+a.ID+"/link", "", 200, &l2)
			if l2.Token != l.Token || l2.BaseURL != l.BaseURL {
				t.Fatalf("link after adopt %+v, before %+v", l2, l)
			}
			w := call(t, mux, "POST", "/v1/providers/"+string(k)+"/adopt", adopt)
			if w.Code != 409 {
				t.Fatalf("second adopt %d %s", w.Code, w.Body)
			}
			do(t, "POST", "/v1/machines/"+a.ID+"/terminate", "", 200, nil)
		})
	}
}
