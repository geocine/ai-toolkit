package codemode

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// provisionTypesTS is studio/src/lib/provisionTypes.ts without `export` and
// without its import line, so the agent and the studio see the same shapes.
func TestProvisionTypesMatchStudio(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "studio", "src", "lib", "provisionTypes.ts"))
	if err != nil {
		t.Fatal(err)
	}
	var want strings.Builder
	for _, line := range strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n") {
		if strings.HasPrefix(line, "import type ") {
			continue
		}
		want.WriteString(strings.TrimPrefix(line, "export "))
		want.WriteString("\n")
	}
	if got, w := strings.TrimSpace(provisionTypesTS), strings.TrimSpace(want.String()); got != w {
		gl, wl := strings.Split(got, "\n"), strings.Split(w, "\n")
		for i := range max(len(gl), len(wl)) {
			var g, x string
			if i < len(gl) {
				g = gl[i]
			}
			if i < len(wl) {
				x = wl[i]
			}
			if g != x {
				t.Fatalf("provisionTypesTS differs from provisionTypes.ts at line %d:\n got %q\nwant %q", i+1, g, x)
			}
		}
	}
	if strings.Contains(provisionTypesTS, "export ") || strings.Contains(provisionTypesTS, "import ") {
		t.Error("provisionTypesTS keeps export or import")
	}
}

func TestProvisionDeclaration(t *testing.T) {
	decl := Declaration("")
	for _, typ := range []string{"interface Machine {", "interface Offer {", "type ProviderKind = ", "interface SetupGuide {", "interface GpuRecommendation {"} {
		if !strings.Contains(decl, typ) {
			t.Errorf("declaration lacks %q", typ)
		}
	}
	block := decl[strings.Index(decl, "  machines: {\n"):]
	block = block[:strings.Index(block, "  };\n")]
	if !strings.Contains(block, "    wait(id: string, options?: { until?: MachineStatus[]; timeout_ms?: number; poll_ms?: number }): Promise<Machine>;") {
		t.Fatalf("machines.wait not declared in its group:\n%s", block)
	}
	if _, err := Transpile(decl + "\nreturn 1"); err != nil {
		t.Fatalf("declaration does not compile: %v", err)
	}
	// every OfferQuery param is on providers.offers, skip_image_check on launch
	params := func(name string) map[string]bool {
		out := map[string]bool{}
		for _, o := range Ops {
			if o.Group+"."+o.Name == name {
				for _, p := range o.Params {
					out[p.Name] = true
				}
			}
		}
		return out
	}
	offers := params("providers.offers")
	for _, q := range []string{"gpu", "min_vram_gb", "max_price", "gpu_count", "cloud", "disk_gb", "limit", "image", "model", "show_all", "fast", "max_vram_gb"} {
		if !offers[q] {
			t.Errorf("providers.offers lacks %s", q)
		}
	}
	if !params("machines.launch")["skip_image_check"] {
		t.Error("machines.launch lacks skip_image_check")
	}
	// the docs the money rules depend on
	for name, want := range map[string]string{
		"machines.launch":    "costs money from the moment it returns",
		"machines.terminate": "anything not on a remote is lost",
		"providers.setKey":   "stored on this server, write-only",
	} {
		found := false
		for _, o := range Ops {
			if o.Group+"."+o.Name == name && strings.Contains(o.Doc, want) {
				found = true
			}
		}
		if !found {
			t.Errorf("%s doc lacks %q", name, want)
		}
	}
}

// providers.offers sends its params as the query, launch sends a JSON body.
func TestProvisionOpsWire(t *testing.T) {
	var calls []recorded
	res := Run(context.Background(), `
await aitk.providers.offers("vast", { gpu: "4090", min_vram_gb: 24, max_vram_gb: 48, max_price: 0.8, gpu_count: 1, cloud: "community",
  disk_gb: 150, limit: 200, image: "ghcr.io/geocine/aitoolkit-go:latest", model: "flux", show_all: true, fast: true });
await aitk.machines.launch({ provider: "vast", offer_id: "o1", max_price_per_hour: 0.5, request_id: "req_12345", skip_image_check: true });
await aitk.machines.terminate("mch_1", { confirm_self: true });
return 1`, Options{Call: fakeAPI(t, &calls)})
	if res.Error != nil {
		t.Fatalf("%+v", res.Error)
	}
	q, _ := url.ParseQuery(calls[0].query)
	want := map[string]string{"gpu": "4090", "min_vram_gb": "24", "max_vram_gb": "48", "max_price": "0.8", "gpu_count": "1", "cloud": "community",
		"disk_gb": "150", "limit": "200", "image": "ghcr.io/geocine/aitoolkit-go:latest", "model": "flux", "show_all": "true", "fast": "true"}
	if calls[0].method != "GET" || calls[0].path != "/v1/providers/vast/offers" || len(q) != len(want) {
		t.Fatalf("offers call %+v", calls[0])
	}
	for k, v := range want {
		if q.Get(k) != v {
			t.Errorf("offers %s = %q, want %q", k, q.Get(k), v)
		}
	}
	var body map[string]any
	json.Unmarshal([]byte(calls[1].body), &body)
	if calls[1].method != "POST" || calls[1].path != "/v1/machines" || body["skip_image_check"] != true || body["max_price_per_hour"] != 0.5 {
		t.Fatalf("launch call %+v", calls[1])
	}
	if calls[2].path != "/v1/machines/mch_1/terminate" || calls[2].body != `{"confirm_self":true}` {
		t.Fatalf("terminate call %+v", calls[2])
	}
}

// machines.wait re-reads the machine until its status is in until, and
// returns the last one when the timeout passes.
func TestMachinesWait(t *testing.T) {
	statuses := []string{"creating", "starting", "running", "ready"}
	n := 0
	var queries []string
	call := func(_ context.Context, method, path string, q url.Values, _ []byte, _ string) (Response, error) {
		if method != "GET" || path != "/v1/machines/mch_1" {
			t.Errorf("unexpected %s %s", method, path)
		}
		queries = append(queries, q.Encode())
		s := statuses[min(n, len(statuses)-1)]
		n++
		b, _ := json.Marshal(map[string]any{"id": "mch_1", "status": s, "problem": nil})
		return Response{Status: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: b}, nil
	}
	res := Run(context.Background(), `return (await aitk.machines.wait("mch_1", { poll_ms: 1 })).status`, Options{Call: call})
	if res.Error != nil || string(res.Result) != `"ready"` || n != 4 {
		t.Fatalf("%s %+v after %d gets", res.Result, res.Error, n)
	}
	if queries[0] != "refresh=true" {
		t.Errorf("wait polls with %q, want refresh=true", queries[0])
	}

	n = 0
	res = Run(context.Background(), `return (await aitk.machines.wait("mch_1", { until: ["stopped"], timeout_ms: 0 })).status`, Options{Call: call})
	if res.Error != nil || string(res.Result) != `"creating"` || n != 1 {
		t.Fatalf("timeout: %s %+v after %d gets", res.Result, res.Error, n)
	}

	// a rate-limited refresh between starting and ready falls back to the
	// stored row and keeps waiting; other errors still throw
	statuses = []string{"starting", "429", "starting", "ready"}
	n, queries = 0, nil
	failCode := "rate_limited"
	flaky := func(ctx context.Context, method, path string, q url.Values, body []byte, ct string) (Response, error) {
		if statuses[min(n, len(statuses)-1)] == "429" && q.Get("refresh") == "true" {
			n++
			queries = append(queries, q.Encode())
			status := map[string]int{"rate_limited": 429, "remote_failed": 502, "not_found": 404}[failCode]
			b, _ := json.Marshal(map[string]any{"error": map[string]string{"code": failCode, "message": "provider said no"}})
			return Response{Status: status, Header: http.Header{"Content-Type": {"application/json"}}, Body: b}, nil
		}
		return call(ctx, method, path, q, body, ct)
	}
	res = Run(context.Background(), `return (await aitk.machines.wait("mch_1", { poll_ms: 1 })).status`, Options{Call: flaky})
	if res.Error != nil || string(res.Result) != `"ready"` {
		t.Fatalf("after a 429: %s %+v (queries %v)", res.Result, res.Error, queries)
	}
	failCode = "not_found"
	n, queries = 0, nil
	res = Run(context.Background(), `return (await aitk.machines.wait("mch_1", { poll_ms: 1 })).status`, Options{Call: flaky})
	if res.Error == nil || res.Error.Status != 404 || !strings.Contains(res.Error.Message, "not_found") {
		t.Fatalf("not_found must throw: %s %+v", res.Result, res.Error)
	}
}

// The declaration's provision block lists no type twice (provisionTypesTS is
// registered once).
func TestProvisionTypesRegisteredOnce(t *testing.T) {
	decl := Declaration("")
	for _, m := range regexp.MustCompile(`(?m)^(?:interface|type) (\w+)`).FindAllStringSubmatch(provisionTypesTS, -1) {
		re := regexp.MustCompile(`(?m)^(?:interface|type) ` + m[1] + `\b`)
		if c := len(re.FindAllString(decl, -1)); c != 1 {
			t.Errorf("%s declared %d times", m[1], c)
		}
	}
}
