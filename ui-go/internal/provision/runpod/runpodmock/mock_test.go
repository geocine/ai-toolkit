package runpodmock

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/provision/mockkit"
)

type client struct {
	t   *testing.T
	url string
	key string
}

func start(t *testing.T, o mockkit.Options) (*Mock, client, *mockkit.FakeClock) {
	t.Helper()
	clock := mockkit.NewFakeClock(time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC))
	if o.Clock == nil {
		o.Clock = clock
	}
	m := New(o)
	m.Start()
	t.Cleanup(m.Close)
	return m, client{t, m.URL, "k"}, clock
}

func (c client) do(method, path, body string, hdr ...string) (*http.Response, string) {
	c.t.Helper()
	req, _ := http.NewRequest(method, c.url+path, strings.NewReader(body))
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		c.t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	return res, string(b)
}

const goodPod = `{"name":"aitk-x","image":"ghcr.io/geocine/aitoolkit-go:latest","gpu":{"id":"NVIDIA GeForce RTX 4090","count":1},"cloud":"SECURE","disk":50,"ports":["8675/http"],"env":{"AI_TOOLKIT_AUTH":"tok"}}`

func create(t *testing.T, c client, body string) map[string]any {
	t.Helper()
	res, b := c.do("POST", "/v2/pods", body)
	if res.StatusCode != http.StatusCreated {
		t.Fatalf("create: %d %s", res.StatusCode, b)
	}
	var p map[string]any
	_ = json.Unmarshal([]byte(b), &p)
	return p
}

func TestNoCORSAndUnauthorized(t *testing.T) {
	_, c, _ := start(t, mockkit.Options{})
	res, _ := c.do("OPTIONS", "/v2/pods", "", "Origin", "https://studio.example.com", "Access-Control-Request-Method", "POST")
	if res.StatusCode != http.StatusUnauthorized || res.Header.Get("Access-Control-Allow-Origin") != "" || res.Header.Get("Allow") != "GET, POST" {
		t.Fatalf("preflight %d %v", res.StatusCode, res.Header)
	}
	res, _ = c.do("GET", "/v2/catalog/gpus", "", "Origin", "https://studio.example.com")
	if res.StatusCode != 200 || res.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("GET with Origin %d %v", res.StatusCode, res.Header)
	}
	if res.Header.Get("RateLimit-Policy") == "" || res.Header.Get("RateLimit") == "" || res.Header.Get("X-Request-Id") == "" {
		t.Errorf("rate limit headers missing: %v", res.Header)
	}
	c.key = ""
	res, b := c.do("GET", "/v2/catalog/gpus", "")
	if res.StatusCode != 401 || res.Header.Get("Content-Type") != "application/problem+json" ||
		strings.TrimSpace(b) != `{"detail":"missing bearer token","status":401,"title":"Unauthorized"}` {
		t.Fatalf("no token: %d %s", res.StatusCode, b)
	}
}

func TestSimSecretKeys(t *testing.T) {
	_, c, _ := start(t, mockkit.Options{SimSecret: "S3CRET"})
	for k, want := range map[string]int{"S3CRET": 200, "S3CRET.sim_full": 200, "S3CRET.sim_bad": 401, "sim_full": 401, "other": 401} {
		c.key = k
		if res, _ := c.do("GET", "/v2/catalog/gpus", ""); res.StatusCode != want {
			t.Errorf("key %q: %d, want %d", k, res.StatusCode, want)
		}
	}
	c.key = "S3CRET.sim_full"
	if res, b := c.do("POST", "/v2/pods", goodPod); res.StatusCode != 400 || !strings.Contains(b, "no available machines") {
		t.Errorf("sim_full create: %d %s", res.StatusCode, b)
	}
}

func TestCatalogParams(t *testing.T) {
	_, c, _ := start(t, mockkit.Options{})
	for q, want := range map[string]int{
		"":                                  200,
		"?include=AVAILABILITY&product=POD": 200,
		"?include=AVAILABILITY":             400, // product required with include
		"?product=POD":                      400, // and only valid with it
		"?count=2":                          400,
		"?limit=1":                          422, // no limit param
		"?include=AVAILABILITY&product=POD&count=0":                               422,
		"?include=AVAILABILITY&product=POD&cloud=ALL":                             422,
		"?include=AVAILABILITY&product=POD&cudaVersions=12.8&minCudaVersion=12.1": 400,
	} {
		if res, b := c.do("GET", "/v2/catalog/gpus"+q, ""); res.StatusCode != want {
			t.Errorf("%q: %d %s, want %d", q, res.StatusCode, b, want)
		}
	}
	_, b := c.do("GET", "/v2/catalog/gpus", "")
	if strings.Contains(b, "availability") {
		t.Errorf("availability without include: %s", b)
	}
	_, b = c.do("GET", "/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=COMMUNITY", "")
	var body struct{ GPUs []GPUType }
	_ = json.Unmarshal([]byte(b), &body)
	if len(body.GPUs) != 5 || body.GPUs[4].Availability != "NONE" || body.GPUs[0].Availability != "HIGH" || len(body.GPUs[0].CudaVersions) == 0 {
		t.Errorf("with include: %+v", body.GPUs)
	}
	if res, _ := c.do("GET", "/v2/catalog/gpus/NVIDIA%20H100%2080GB%20HBM3?include=AVAILABILITY&product=POD", ""); res.StatusCode != 200 {
		t.Errorf("an unavailable type is still 200: %d", res.StatusCode)
	}
	if res, _ := c.do("GET", "/v2/catalog/gpus/nope", ""); res.StatusCode != 404 {
		t.Errorf("unknown type: %d", res.StatusCode)
	}
}

func TestCreateValidation(t *testing.T) {
	_, c, _ := start(t, mockkit.Options{})
	for _, cs := range []struct {
		body   string
		status int
		text   string
	}{
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090"},"startSsh":true}`, 201, ""}, // startSsh is a known field
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090"},"bogus":1}`, 422, "additional properties 'bogus' not allowed"},
		{`{"image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090"}}`, 422, "$.name: required"},
		{`{"name":"","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090"}}`, 422, "minLength"},
		{`{"name":"a","image":"i"}`, 422, "exactly one of gpu or cpu"},
		{`{"name":"a","image":"i","gpu":{"id":"x"},"cpu":{}}`, 422, "exactly one of gpu or cpu"},
		{`{"name":"a","gpu":{"id":"NVIDIA GeForce RTX 4090"}}`, 422, "$.image"},
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090","speed":9}}`, 422, "$.gpu: additional properties 'speed'"},
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090"},"ports":["8675"]}`, 422, "$.ports[0]"},
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090","minCudaVersion":"13.0","allowedCudaVersions":["13.0"]}}`, 400, "mutually exclusive"},
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA H100 80GB HBM3"}}`, 400, "no available machines"},
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090","count":5},"cloud":"COMMUNITY"}`, 400, "no available machines"},
		{`{"name":"a","templateId":"missing","gpu":{"id":"NVIDIA GeForce RTX 4090"}}`, 404, "template"},
		{`{"name":"a","image":"i","gpu":{"id":"NVIDIA GeForce RTX 4090"},"env":{"A":"` + strings.Repeat("x", createBodyMax) + `"}}`, 413, "102400"},
	} {
		res, b := c.do("POST", "/v2/pods", cs.body)
		if res.StatusCode != cs.status || !strings.Contains(b, cs.text) {
			t.Errorf("%.80s: %d %s, want %d %q", cs.body, res.StatusCode, b, cs.status, cs.text)
		}
	}
	// the template copy merges env, the body winning
	p := create(t, c, `{"name":"t","templateId":"tpl_aitk0001","gpu":{"id":"NVIDIA GeForce RTX 4090"},"env":{"AI_TOOLKIT_AUTH":"mine"}}`)
	env := p["env"].(map[string]any)
	if p["image"] != "ghcr.io/geocine/aitoolkit-go:latest" || env["AI_TOOLKIT_AUTH"] != "mine" || env["AITK_CORS_ORIGINS"] != "*" || p["template"] != nil {
		t.Errorf("template pod %v", p)
	}
}

func TestCreditAndCapacity(t *testing.T) {
	m, c, _ := start(t, mockkit.Options{})
	m.SetCredit(0.1)
	if res, b := c.do("POST", "/v2/pods", goodPod); res.StatusCode != 402 {
		t.Fatalf("low balance: %d %s", res.StatusCode, b)
	}
	m.SetCredit(25)
	m.SetAvailability("NVIDIA GeForce RTX 4090", "NONE")
	if res, _ := c.do("POST", "/v2/pods", goodPod); res.StatusCode != 400 {
		t.Fatalf("NONE: %d", res.StatusCode)
	}
}

func TestPodLifecycleAndNodes(t *testing.T) {
	type node struct {
		spec    mockkit.NodeSpec
		stopped bool
	}
	var nodes []*node
	factory := func(n mockkit.NodeSpec) (string, func()) {
		nd := &node{spec: n}
		nodes = append(nodes, nd)
		return fmt.Sprintf("http://127.0.0.1:%d", 40000+len(nodes)), func() { nd.stopped = true }
	}
	m, c, clock := start(t, mockkit.Options{Delays: mockkit.Delays{Provision: 3 * time.Second, Start: 6 * time.Second, App: 4 * time.Second}, Node: factory})
	p := create(t, c, goodPod)
	id := p["id"].(string)
	if p["status"] != Provisioning || p["runtime"] != nil || p["startedAt"] != nil || len(id) != 10 {
		t.Fatalf("created %v", p)
	}
	status := func() map[string]any {
		_, b := c.do("GET", "/v2/pods/"+id, "")
		var out map[string]any
		_ = json.Unmarshal([]byte(b), &out)
		return out
	}
	if m.NodeURL(id) != "" || len(nodes) != 0 {
		t.Fatal("no node before RUNNING")
	}
	clock.Advance(9 * time.Second)
	got := status()
	if got["status"] != Running || got["runtime"] == nil || got["dataCenterId"] != "US-KS-2" {
		t.Fatalf("running %v", got)
	}
	if len(nodes) != 1 || m.NodeURL(id) != "http://127.0.0.1:40001" || nodes[0].spec.Env["AI_TOOLKIT_AUTH"] != "tok" ||
		nodes[0].spec.GPU != "RTX 4090" || nodes[0].spec.VRAMGB != 24 || nodes[0].spec.AppDelay != 4*time.Second {
		t.Fatalf("node %+v", nodes)
	}
	// the list never carries runtime
	_, b := c.do("GET", "/v2/pods", "")
	if !strings.Contains(b, `"runtime":null`) || !strings.Contains(b, `"pagination":{"hasNextPage":false,"nextCursor":null}`) {
		t.Errorf("list %s", b)
	}
	if res, _ := c.do("POST", "/v2/pods/"+id+"/action", `{"action":"start"}`); res.StatusCode != 409 {
		t.Errorf("start while running: %d", res.StatusCode)
	}
	if res, _ := c.do("POST", "/v2/pods/"+id+"/action", `{"action":"stop","force":true}`); res.StatusCode != 422 {
		t.Errorf("unknown action field: %d", res.StatusCode)
	}
	if res, _ := c.do("POST", "/v2/pods/"+id+"/action", `{"action":"stop"}`); res.StatusCode != 200 {
		t.Fatalf("stop: %d", res.StatusCode)
	}
	clock.Advance(3 * time.Second)
	if got := status(); got["status"] != Exited || got["cost"] != 0.0 || !nodes[0].stopped || m.NodeURL(id) != "" {
		t.Fatalf("stopped %v nodes %+v", got, nodes)
	}
	c.do("POST", "/v2/pods/"+id+"/action", `{"action":"start"}`)
	clock.Advance(6 * time.Second)
	if got := status(); got["status"] != Running || len(nodes) != 2 || m.NodeURL(id) != "http://127.0.0.1:40002" {
		t.Fatalf("restarted %v", got)
	}
	_, logs := c.do("GET", "/v2/pods/"+id+"/logs?tail=2", "")
	if !strings.Contains(logs, "/000000000001\ndata: {") || strings.Count(logs, "data: ") != 2 {
		t.Errorf("sse %q", logs)
	}
	if res, _ := c.do("DELETE", "/v2/pods/"+id, ""); res.StatusCode != 204 || !nodes[1].stopped {
		t.Fatalf("delete %d", res.StatusCode)
	}
	if res, _ := c.do("GET", "/v2/pods/"+id, ""); res.StatusCode != 404 {
		t.Errorf("get after delete %d", res.StatusCode)
	}
	if res, _ := c.do("DELETE", "/v2/pods/"+id, ""); res.StatusCode != 404 {
		t.Errorf("delete twice %d", res.StatusCode)
	}
	if pods := m.Instances(); len(pods) != 1 || pods[0].Status != Terminated {
		t.Errorf("instances %+v", pods)
	}
}

func TestV1FallbackRoutes(t *testing.T) {
	_, c, _ := start(t, mockkit.Options{})
	id := create(t, c, goodPod)["id"].(string)
	if res, _ := c.do("POST", "/v1/pods/"+id+"/stop", ""); res.StatusCode != 200 {
		t.Fatalf("v1 stop %d", res.StatusCode)
	}
	if res, _ := c.do("DELETE", "/v1/pods/"+id, ""); res.StatusCode != 204 {
		t.Fatalf("v1 delete %d", res.StatusCode)
	}
	c.key = ""
	if res, b := c.do("POST", "/v1/pods/"+id+"/stop", ""); res.StatusCode != 401 || b != "" {
		t.Fatalf("v1 unauthorized %d %q", res.StatusCode, b)
	}
}

func TestGraphQLBalance(t *testing.T) {
	m, c, _ := start(t, mockkit.Options{})
	m.SetCredit(12.34)
	_, b := c.do("POST", "/graphql", `{"query":"query { myself { clientBalance } }"}`)
	if !strings.Contains(b, `"clientBalance":12.34`) {
		t.Fatalf("balance %s", b)
	}
	res, b := c.do("POST", "/graphql", `{"query":"query { gpuTypes { id } }"}`)
	if res.StatusCode != 200 || !strings.Contains(b, `"errors"`) {
		t.Fatalf("unknown query %d %s", res.StatusCode, b)
	}
}

func TestPaginationCursor(t *testing.T) {
	_, c, _ := start(t, mockkit.Options{})
	for range 3 {
		create(t, c, goodPod)
	}
	_, b := c.do("GET", "/v2/pods?limit=2", "")
	var page struct {
		Pods       []map[string]any
		Pagination struct {
			NextCursor  *string
			HasNextPage bool
		}
	}
	_ = json.Unmarshal([]byte(b), &page)
	if len(page.Pods) != 2 || !page.Pagination.HasNextPage || page.Pagination.NextCursor == nil {
		t.Fatalf("page 1 %s", b)
	}
	_, b = c.do("GET", "/v2/pods?limit=2&cursor="+*page.Pagination.NextCursor, "")
	_ = json.Unmarshal([]byte(b), &page)
	if len(page.Pods) != 1 || page.Pagination.HasNextPage {
		t.Fatalf("page 2 %s", b)
	}
	if res, _ := c.do("GET", "/v2/pods?cursor=!!", ""); res.StatusCode != 422 {
		t.Errorf("foreign cursor %d", res.StatusCode)
	}
}

func TestHandlerWithoutListener(t *testing.T) {
	m := New(mockkit.Options{})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/v2/templates", nil)
	req.Header.Set("Authorization", "Bearer k")
	m.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "aitk-go headless") || len(m.Rec.Requests()) != 1 {
		t.Fatalf("templates %d %s", rec.Code, rec.Body)
	}
}
