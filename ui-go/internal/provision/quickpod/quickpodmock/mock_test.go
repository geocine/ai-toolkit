package quickpodmock

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/provision/mockkit"
)

const key = "qpk_mock.key"

type client struct {
	t   *testing.T
	url string
}

func start(t *testing.T, o mockkit.Options) (*Mock, client) {
	t.Helper()
	if o.Clock == nil {
		o.Clock = mockkit.NewFakeClock(time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC))
	}
	m := New(o)
	u := m.Start()
	t.Cleanup(m.Close)
	return m, client{t, u}
}

// do sends a request with header pairs and decodes the JSON answer into out.
func (c client) do(method, path, body string, out any, hdr ...string) int {
	c.t.Helper()
	req, _ := http.NewRequest(method, c.url+path, strings.NewReader(body))
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		c.t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	if out != nil && len(b) > 0 {
		if err := json.Unmarshal(b, out); err != nil {
			c.t.Fatalf("%s %s: %v in %s", method, path, err, b)
		}
	}
	return res.StatusCode
}

func (c client) authed(method, path, body string, out any) int {
	return c.do(method, path, body, out, "X-API-Key", key)
}

func (c client) rentableIDs() map[float64]bool {
	var rows []map[string]any
	c.do("GET", "/rentable", "", &rows)
	ids := map[float64]bool{}
	for _, r := range rows {
		ids[r["id"].(float64)] = true
	}
	return ids
}

func TestAuthHeaders(t *testing.T) {
	_, c := start(t, mockkit.Options{})
	var me map[string]any
	if s := c.do("GET", "/update/api/me", "", &me, "X-API-Key", key); s != 200 || me["credit"] != 25.0 {
		t.Fatalf("X-API-Key: %d %v", s, me)
	}
	if s := c.do("GET", "/update/api/me", "", nil, "Authorization", "ApiKey "+key); s != 200 {
		t.Fatalf("Authorization ApiKey: %d", s)
	}
	var e map[string]string
	if s := c.do("GET", "/update/api/me", "", &e); s != 401 || e["error"] == "" {
		t.Fatalf("no key: %d %v", s, e)
	}
	// /update/api/* is ApiKey-only; /mypods and /templates also take Bearer.
	if s := c.do("GET", "/update/api/gpu_pods", "", nil, "Authorization", "Bearer "+key); s != 401 {
		t.Fatalf("bearer on secure api: %d", s)
	}
	if s := c.do("GET", "/mypods", "", nil, "Authorization", "Bearer "+key); s != 200 {
		t.Fatalf("bearer on mypods: %d", s)
	}
}

func TestSimSecretKeys(t *testing.T) {
	m, c := start(t, mockkit.Options{SimSecret: "S3CRET"})
	if s := c.do("GET", "/update/api/me", "", nil, "X-API-Key", key); s != 401 {
		t.Fatalf("a key without the secret: %d", s)
	}
	body := `{"offers_id":6302336,"template_uuid":"` + AitkTemplateUUID + `","disk_size":"100","altname":"aitk-a"}`
	if s := c.do("POST", "/update/api/createpod", body, nil, "X-API-Key", "S3CRET"); s != 200 {
		t.Fatalf("create: %d", s)
	}
	var pods []map[string]any
	c.do("GET", "/update/api/gpu_pods", "", &pods, "X-API-Key", "S3CRET.sim_slow")
	if len(pods) != 1 || len(m.Instances()) != 1 {
		t.Fatalf("pods belong to the secret, whatever the fault suffix: %v", pods)
	}
	if s := c.do("POST", "/update/api/createpod", strings.Replace(body, "6302336", "6289141", 1), nil, "X-API-Key", "S3CRET.sim_full"); s != 406 {
		t.Fatalf("sim_full: %d", s)
	}
}

func TestTemplatesRoutes(t *testing.T) {
	_, c := start(t, mockkit.Options{})
	var mine []map[string]any
	if s := c.authed("GET", "/templates", "", &mine); s != 200 || len(mine) != 2 || mine[0]["image_path"] != "ghcr.io/geocine/aitoolkit-go:latest" {
		t.Fatalf("templates %d %v", s, mine)
	}
	var pub []map[string]any
	if c.do("GET", "/public_templates", "", &pub); len(pub) != 1 || pub[0]["template_uuid"] != OtherTemplateUUID {
		t.Fatalf("public %v", pub)
	}
	if _, has := pub[0]["docker_repo_password"]; has {
		t.Fatal("PublicTemplate has no registry credentials")
	}
	body := `{"template_uuid":"","template_name":"mine","image_path":"x/y:1"}`
	if s := c.authed("POST", "/update/templates", body, nil); s != 401 {
		t.Fatalf("ApiKey on /update/templates: %d (Bearer only)", s)
	}
	var made map[string]any
	if s := c.do("POST", "/update/templates", body, &made, "Authorization", "Bearer "+key); s != 200 || made["template_uuid"] == "" {
		t.Fatalf("bearer create %d %v", s, made)
	}
	del := "/update/deleteTemplate/" + made["template_uuid"].(string)
	if s := c.authed("GET", del, "", nil); s != 401 {
		t.Fatalf("ApiKey on deleteTemplate: %d (Bearer only)", s)
	}
	var gone map[string]string
	if s := c.do("GET", del, "", &gone, "Authorization", "Bearer "+key); s != 200 || gone["message"] != "Template deleted" {
		t.Fatalf("bearer delete %d %v", s, gone)
	}
	if s := c.do("GET", del, "", nil, "Authorization", "Bearer "+key); s != 404 {
		t.Fatalf("delete again: %d, want 404", s)
	}
	if c.authed("GET", "/templates", "", &mine); len(mine) != 2 {
		t.Fatalf("templates after delete %v", mine)
	}
}

func TestCreateValidation(t *testing.T) {
	_, c := start(t, mockkit.Options{})
	for _, b := range []string{
		`{"offers_id":6302336,"disk_size":"100"}`,
		`{"offers_id":6302336,"template_uuid":"` + AitkTemplateUUID + `","disk_size":100}`,
		`{"offers_id":6302336,"template_uuid":"` + AitkTemplateUUID + `"}`,
		`{"offers_id":6302336,"template_uuid":"nope","disk_size":"100"}`,
		`not json`,
	} {
		var e map[string]string
		if s := c.authed("POST", "/update/api/createpod", b, &e); s != 400 || e["error"] == "" {
			t.Errorf("%s: %d %v", b, s, e)
		}
	}
	// Every failure of createpod's database step is 406 "Database query failed".
	for b, msg := range map[string]string{
		`{"offers_id":1,"template_uuid":"` + AitkTemplateUUID + `","disk_size":"100"}`:        "offer or machine not found",
		`{"offers_id":6302336,"template_uuid":"` + AitkTemplateUUID + `","disk_size":"9999"}`: "requested disk size 9999GB exceeds the maximum disk size 712GB",
	} {
		var e map[string]string
		if s := c.authed("POST", "/update/api/createpod", b, &e); s != 406 || e["error"] != "Database query failed" || e["message"] != msg {
			t.Errorf("%s: %d %v", b, s, e)
		}
	}
}

func TestPodLifecycleShape(t *testing.T) {
	var starts, stops int
	node := func(n mockkit.NodeSpec) (string, func()) {
		starts++
		return "http://127.0.0.1:4500" + string(rune('0'+starts)), func() { stops++ }
	}
	m, c := start(t, mockkit.Options{Node: node})
	body := `{"offers_id":6302336,"template_uuid":"` + AitkTemplateUUID + `","disk_size":"200","altname":"aitk-a","docker_options":"-e AI_TOOLKIT_AUTH=tok"}`
	var created map[string]any
	if s := c.authed("POST", "/update/api/createpod", body, &created); s != 200 || created["status"] != "success" {
		t.Fatalf("create %d %v", s, created)
	}
	if id, _ := created["pod_uuid"].(string); id == "" {
		t.Fatalf("create answers with the pod_uuid: %v", created)
	}
	if c.rentableIDs()[6302336] {
		t.Fatal("a rented offer leaves /rentable")
	}
	var busy bool
	if c.do("GET", "/update/offer_is_busy?offer_id=6302336", "", &busy); !busy {
		t.Fatal("offer_is_busy false for a rented offer")
	}
	var pods []map[string]any
	c.authed("GET", "/update/api/gpu_pods", "", &pods)
	p := pods[0]
	if p["State"] != "running" || p["intended_state"] != "running" || p["altname"] != "aitk-a" || p["offers_id"] != 6302336.0 ||
		p["port_mappings"] != `8675 -> <a target="_new" href="http://127.0.0.1:45001">127.0.0.1:45001</a> |` ||
		math.Abs(p["storage_cost"].(float64)-0.15*200/1024) > 1e-9 || starts != 1 {
		t.Fatalf("pod %v (node starts %d)", p, starts)
	}
	uuid := p["Names"].(string)
	if s := c.authed("GET", "/update/api/stoppod", "", nil); s != 400 {
		t.Fatalf("missing pod_uuid: %d", s)
	}
	if s := c.do("GET", "/update/api/stoppod?pod_uuid="+uuid, "", nil, "X-API-Key", "qpk_other.key"); s != 403 {
		t.Fatalf("not owned: %d", s)
	}
	c.authed("GET", "/update/api/stoppod?pod_uuid="+uuid, "", nil)
	c.authed("GET", "/update/api/gpu_pods", "", &pods)
	if pods[0]["State"] != "exited" || pods[0]["port_mappings"] != "" || stops != 1 {
		t.Fatalf("stopped %v (node stops %d)", pods[0], stops)
	}
	c.authed("GET", "/update/api/startpod?pod_uuid="+uuid, "", nil)
	c.authed("GET", "/update/api/gpu_pods", "", &pods)
	if pods[0]["port_mappings"] != `8675 -> <a target="_new" href="http://127.0.0.1:45002">127.0.0.1:45002</a> |` || starts != 2 {
		t.Fatalf("restarted on a new node %v", pods[0])
	}
	c.authed("GET", "/update/api/destroypod?pod_uuid="+uuid, "", nil)
	c.authed("GET", "/update/api/gpu_pods", "", &pods)
	if len(pods) != 0 || stops != 2 || !c.rentableIDs()[6302336] {
		t.Fatalf("destroyed pods are not listed %v; the offer is free again", pods)
	}
	if s := c.authed("GET", "/update/api/destroypod?pod_uuid="+uuid, "", nil); s != 403 {
		t.Fatalf("destroy again: %d", s)
	}
	if got := m.Instances(); len(got) != 1 || !got[0].Destroyed || got[0].Env["AI_TOOLKIT_AUTH"] != "tok" {
		t.Fatalf("instances %+v", got)
	}
}

func TestNoPortWithoutNode(t *testing.T) {
	_, c := start(t, mockkit.Options{Faults: mockkit.Faults{NoPortMapping: true}})
	body := `{"offers_id":6302336,"template_uuid":"` + AitkTemplateUUID + `","disk_size":"100","altname":"aitk-a"}`
	c.authed("POST", "/update/api/createpod", body, nil)
	var pods []map[string]any
	c.authed("GET", "/update/api/gpu_pods", "", &pods)
	if pods[0]["port_mappings"] != "" {
		t.Fatalf("port_mappings %q", pods[0]["port_mappings"])
	}
}

func TestCORS(t *testing.T) {
	_, c := start(t, mockkit.Options{})
	req, _ := http.NewRequest("OPTIONS", c.url+"/update/api/createpod", nil)
	req.Header.Set("Origin", "https://studio.example.com")
	req.Header.Set("Access-Control-Request-Headers", "x-api-key,content-type")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != 204 || res.Header.Get("Access-Control-Allow-Origin") != "https://studio.example.com" ||
		res.Header.Get("Access-Control-Allow-Headers") != "x-api-key,content-type" || res.Header.Get("Access-Control-Allow-Credentials") != "true" {
		t.Fatalf("update preflight %d %v", res.StatusCode, res.Header)
	}
	res, _ = http.Get(c.url + "/rentable")
	res.Body.Close()
	if res.Header.Get("Access-Control-Allow-Origin") != "*" {
		t.Fatalf("rentable ACAO %q", res.Header.Get("Access-Control-Allow-Origin"))
	}
}
