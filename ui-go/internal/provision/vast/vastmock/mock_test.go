package vastmock

import (
	"encoding/json"
	"io"
	"maps"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/provision/mockkit"
)

func start(t *testing.T, o mockkit.Options) *Mock {
	t.Helper()
	if o.Clock == nil {
		o.Clock = mockkit.NewFakeClock(time.Unix(1_790_000_000, 0))
	}
	m := New(o)
	m.Start()
	t.Cleanup(m.Close)
	return m
}

func do(t *testing.T, method, u, key, body string) (int, http.Header, []byte) {
	t.Helper()
	var r io.Reader
	if body != "" {
		r = strings.NewReader(body)
	}
	req, _ := http.NewRequest(method, u, r)
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	return res.StatusCode, res.Header, b
}

func offerIDs(t *testing.T, b []byte) []int64 {
	t.Helper()
	var out struct {
		Offers []struct {
			ID int64 `json:"id"`
		} `json:"offers"`
	}
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("offers %s: %v", b, err)
	}
	ids := []int64{}
	for _, o := range out.Offers {
		ids = append(ids, o.ID)
	}
	return ids
}

func TestFixtureIsScrubbed(t *testing.T) {
	m := New(mockkit.Options{})
	if len(m.offers) != 8 {
		t.Fatalf("fixture rows %d, want 2 live + 6 synthetic", len(m.offers))
	}
	for _, o := range m.offers {
		if ip, _ := o["public_ipaddr"].(string); !strings.HasPrefix(ip, "203.0.113.") {
			t.Errorf("offer %v has a non-documentation address %q", o["id"], ip)
		}
	}
}

func TestAuthAndCORS(t *testing.T) {
	m := start(t, mockkit.Options{Keys: []string{"good"}})
	status, _, b := do(t, "GET", m.URL+"/api/v0/users/current/", "", "")
	if status != 401 || string(b) != `{"error":"auth_error","msg":"Invalid user key","success":false}`+"\n" {
		t.Fatalf("no key: %d %s", status, b)
	}
	if status, _, _ := do(t, "GET", m.URL+"/api/v0/users/current/", "bad", ""); status != 401 {
		t.Fatalf("bad key: %d", status)
	}
	// both slash forms, and the legacy ?api_key=
	for _, u := range []string{m.URL + "/api/v0/users/current/", m.URL + "/api/v0/users/current"} {
		if status, _, _ := do(t, "GET", u, "good", ""); status != 200 {
			t.Fatalf("%s: %d", u, status)
		}
	}
	if status, _, b := do(t, "GET", m.URL+"/api/v0/users/current/?api_key=good", "", ""); status != 200 || !strings.Contains(string(b), `"api_key":"good"`) {
		t.Fatalf("api_key param: %d %s", status, b)
	}
	// the public offer search needs no key
	if status, _, _ := do(t, "GET", m.URL+"/api/v0/bundles/?q=%7B%7D", "", ""); status != 200 {
		t.Fatalf("keyless search: %d", status)
	}
	// preflight: 204 and no Access-Control-Allow-Origin
	req, _ := http.NewRequest("OPTIONS", m.URL+"/api/v0/instances/", nil)
	req.Header.Set("Origin", "https://studio.example.com")
	req.Header.Set("Access-Control-Request-Method", "PUT")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != 204 || res.Header.Get("Access-Control-Allow-Origin") != "" || res.Header.Get("Access-Control-Allow-Methods") == "" {
		t.Fatalf("preflight %d %v", res.StatusCode, res.Header)
	}
}

func TestRateLimitIsPlainText(t *testing.T) {
	m := start(t, mockkit.Options{Faults: mockkit.Faults{RateLimitEvery: 2}})
	if status, _, _ := do(t, "GET", m.URL+"/api/v0/users/current/", "k", ""); status != 200 {
		t.Fatalf("first call %d", status)
	}
	status, h, b := do(t, "GET", m.URL+"/api/v0/users/current/", "k", "")
	if status != 429 || string(b) != "API requests too frequent endpoint threshold=1.0" || !strings.HasPrefix(h.Get("Content-Type"), "text/plain") ||
		h.Get("X-RateLimit-Limit") == "" || h.Get("X-RateLimit-Reset") == "" {
		t.Fatalf("429: %d %v %s", status, h, b)
	}
}

func TestSearchOperators(t *testing.T) {
	m := start(t, mockkit.Options{})
	search := func(body string) []int64 {
		t.Helper()
		status, _, b := do(t, "POST", m.URL+"/api/v0/bundles/", "k", body)
		if status != 200 {
			t.Fatalf("%s: %d %s", body, status, b)
		}
		ids := offerIDs(t, b)
		slices.Sort(ids)
		return ids
	}
	for body, want := range map[string][]int64{
		`{"gpu_name":{"eq":"RTX A6000"}}`:              {90000002},
		`{"gpu_name":{"in":["RTX 3090","RTX A6000"]}}`: {90000001, 90000002},
		`{"num_gpus":{"neq":1}}`:                       {90000003},
		`{"inet_down":{"lt":100}}`:                     {90000006},
		`{"disk_bw":{"gt":3000}}`:                      {90000001},
		`{"reliability":{"gte":0.995}}`:                {90000002},
		`{"direct_port_count":{"lte":0}}`:              {90000004},
		`{"geolocation":{"notin":["Germany, DE","Texas, US","Ohio, US","Türkiye, TR","California, US"]}}`: {90000001, 90000002},
		`{"rented":{"eq":true}}`: {90000005},
		`{"verified":{"eq":true},"external":{"eq":false},"rentable":{"eq":true},"rented":{"eq":false},"num_gpus":{"eq":2}}`: {90000003},
	} {
		if got := search(body); !slices.Equal(got, want) {
			t.Errorf("%s: %v, want %v", body, got, want)
		}
	}
	// order, limit and every type spelling
	status, _, b := do(t, "POST", m.URL+"/api/v0/bundles/", "k", `{"order":[["dph_total","desc"]],"limit":2,"type":"ondemand"}`)
	if got := offerIDs(t, b); status != 200 || len(got) != 2 || got[0] != 90000003 {
		t.Fatalf("order desc limit 2: %d %v", status, got)
	}
	for _, typ := range []string{"on-demand", "bid", "reserved"} {
		if status, _, _ := do(t, "POST", m.URL+"/api/v0/bundles/", "k", `{"type":"`+typ+`"}`); status != 200 {
			t.Errorf("type %s: %d", typ, status)
		}
	}
	if status, _, _ := do(t, "POST", m.URL+"/api/v0/bundles/", "k", `{"type":"spot"}`); status != 400 {
		t.Errorf("unknown type: %d", status)
	}
	if status, _, b := do(t, "POST", m.URL+"/api/v0/bundles/", "k", `{not json`); status != 400 || !strings.Contains(string(b), "invalid_request") {
		t.Errorf("bad json: %d %s", status, b)
	}
	if status, _, _ := do(t, "POST", m.URL+"/api/v0/bundles/", "k", `{"gpu_ram":{"about":1}}`); status != 400 {
		t.Errorf("unknown operator: %d", status)
	}
	// the GET ?q= form takes the same body
	q := url.QueryEscape(`{"gpu_name":{"eq":"RTX 3090"}}`)
	if status, _, b := do(t, "GET", m.URL+"/api/v0/bundles/?q="+q, "", ""); status != 200 || !slices.Equal(offerIDs(t, b), []int64{90000001}) {
		t.Errorf("GET ?q=: %d %s", status, b)
	}
}

func TestSearchRecomputesPrice(t *testing.T) {
	m := start(t, mockkit.Options{})
	price := func(body string) (float64, float64) {
		_, _, b := do(t, "POST", m.URL+"/api/v0/bundles/", "k", body)
		var out struct {
			Offers []struct {
				DPHTotal float64 `json:"dph_total"`
				Instance struct {
					TotalHour float64 `json:"totalHour"`
				} `json:"instance"`
			} `json:"offers"`
		}
		_ = json.Unmarshal(b, &out)
		return out.Offers[0].DPHTotal, out.Offers[0].Instance.TotalHour
	}
	// no allocated_storage prices 8 GB (vast.md §3.3)
	if p, s := price(`{"id":{"eq":57278070}}`); math.Abs(p-(0.44+0.2*8/720)) > 1e-9 || math.Abs(s-0.2*8/720) > 1e-9 {
		t.Errorf("8 GB: %v %v", p, s)
	}
	if p, _ := price(`{"id":{"eq":57278070},"allocated_storage":200}`); math.Abs(p-(0.44+0.2*200/720)) > 1e-9 {
		t.Errorf("200 GB: %v", p)
	}
}

func TestInstanceRoutes(t *testing.T) {
	m := start(t, mockkit.Options{})
	status, _, b := do(t, "PUT", m.URL+"/api/v0/asks/57278070", "k", `{"client_id":"me","image":"img","disk":20,"runtype":"ssh_direc ssh_proxy","env":"-e A=1 -p 8675:8675"}`)
	if status != 200 || !strings.Contains(string(b), `"new_contract":30000001`) || !strings.Contains(string(b), "instance_api_key") {
		t.Fatalf("create (legacy runtype, env string): %d %s", status, b)
	}
	in := m.Instances()[0]
	if in.Env["A"] != "1" || in.Env["-p 8675:8675"] != "1" || in.RunType != "ssh_direc ssh_proxy" {
		t.Fatalf("instance %+v", in)
	}
	if status, _, b := do(t, "PUT", m.URL+"/api/v0/asks/55830991/", "k", `{"image":"img","env":{"K":"`+strings.Repeat("x", 257)+`"}}`); status != 400 || !strings.Contains(string(b), "invalid_args") {
		t.Fatalf("257-char value: %d %s", status, b)
	}
	if status, _, _ := do(t, "PUT", m.URL+"/api/v0/asks/55830991/", "k", `{"image":"img","disk":5000}`); status != 400 {
		t.Fatalf("disk above the host: %d", status)
	}
	if status, _, b := do(t, "PUT", m.URL+"/api/v0/asks/57278070/", "k", `{"image":"img"}`); status != 404 || !strings.Contains(string(b), "no_such_ask") {
		t.Fatalf("taken without cancel_unavail: %d %s", status, b)
	}
	// both slash forms on show; v0 list; label and unknown ids
	for _, p := range []string{"/api/v0/instances/30000001/", "/api/v0/instances/30000001"} {
		if status, _, b := do(t, "GET", m.URL+p, "k", ""); status != 200 || !strings.Contains(string(b), `"id":30000001`) {
			t.Fatalf("%s: %d %s", p, status, b)
		}
	}
	if status, _, b := do(t, "GET", m.URL+"/api/v0/instances/99/", "k", ""); status != 200 || string(b) != `{"instances":null}`+"\n" {
		t.Fatalf("unknown instance: %d %s", status, b)
	}
	if status, _, b := do(t, "GET", m.URL+"/api/v0/instances/", "k", ""); status != 200 || !strings.Contains(string(b), `"instances_found":1`) {
		t.Fatalf("v0 list: %d %s", status, b)
	}
	if status, _, _ := do(t, "PUT", m.URL+"/api/v0/instances/30000001/", "k", `{"label":"aitk-renamed"}`); status != 200 || m.Instances()[0].Label != "aitk-renamed" {
		t.Fatalf("label: %d", status)
	}
	if status, _, b := do(t, "PUT", m.URL+"/api/v0/instances/99/", "k", `{"state":"stopped"}`); status != 404 || !strings.Contains(string(b), "no_such_instance") {
		t.Fatalf("manage unknown: %d %s", status, b)
	}
	if status, _, b := do(t, "DELETE", m.URL+"/api/v0/instances/99/", "k", `{}`); status != 404 || !strings.Contains(string(b), `"not_found"`) {
		t.Fatalf("destroy unknown: %d %s", status, b)
	}
	if status, _, _ := do(t, "PUT", m.URL+"/api/v0/template/", "k", `{}`); status != 410 {
		t.Fatalf("v0 template edit: %d", status)
	}
}

func TestSearchLimitCaps(t *testing.T) {
	m := start(t, mockkit.Options{})
	// enough rows to pass both caps: clones of a fixture row with unique ids
	base := maps.Clone(m.offers[0])
	m.mu.Lock()
	for i := range 250 {
		row := maps.Clone(base)
		row["id"] = float64(91000000 + i)
		m.offers = append(m.offers, row)
	}
	m.mu.Unlock()
	count := func(method, u, key, body string) int {
		t.Helper()
		status, _, b := do(t, method, u, key, body)
		if status != 200 {
			t.Fatalf("%s %s %s: %d %s", method, u, body, status, b)
		}
		return len(offerIDs(t, b))
	}
	bundles := m.URL + "/api/v0/bundles/"
	for body, want := range map[string]int{
		`{"limit":200}`: 200, // keyed searches honour limit up to 200 (MASTER §3.9.2)
		`{"limit":500}`: 200,
		`{"limit":10}`:  10,
		`{}`:            64, // no limit: the default page
	} {
		if got := count("POST", bundles, "k", body); got != want {
			t.Errorf("keyed %s: %d rows, want %d", body, got, want)
		}
	}
	// keyless searches stay capped at 64 whatever they ask for (IMG §0.2)
	if got := count("GET", bundles+"?q="+url.QueryEscape(`{"limit":200}`), "", ""); got != 64 {
		t.Errorf("keyless limit 200: %d rows, want 64", got)
	}
}
