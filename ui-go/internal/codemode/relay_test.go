package codemode

import (
	"context"
	"encoding/json"
	"net/url"
	"testing"
)

// The relay ops reach a machine's route: the path keeps its slashes, other
// params are the query (GET, DELETE) or the JSON body (POST, PUT, PATCH).
func TestRelayOps(t *testing.T) {
	type got struct{ method, path, query, body string }
	var calls []got
	call := func(_ context.Context, method, path string, q url.Values, body []byte, _ string) (Response, error) {
		calls = append(calls, got{method, path, q.Encode(), string(body)})
		return Response{Status: 200, Body: []byte(`{"ok":true}`)}, nil
	}
	res := Run(context.Background(), `
await aitk.machines.relay_get("mch_1", "v1/jobs", { limit: 5 });
await aitk.machines.relay_post("mch_1", "v1/jobs/j 1/start", { force: true });
await aitk.machines.relay_delete("mch_1", "v1/jobs/j1");
return (await aitk.machines.relay_get("mch_1", "v1/health")).ok;
`, Options{Call: call})
	if res.Error != nil {
		t.Fatalf("%+v", res.Error)
	}
	want := []got{
		{"GET", "/v1/machines/mch_1/relay/v1/jobs", "limit=5", ""},
		{"POST", "/v1/machines/mch_1/relay/v1/jobs/j%201/start", "", `{"force":true}`},
		{"DELETE", "/v1/machines/mch_1/relay/v1/jobs/j1", "", ""},
		{"GET", "/v1/machines/mch_1/relay/v1/health", "", ""},
	}
	if len(calls) != len(want) {
		t.Fatalf("calls %+v", calls)
	}
	for i, w := range want {
		c := calls[i]
		if c.body != "" && w.body != "" {
			var a, b any
			json.Unmarshal([]byte(c.body), &a)
			json.Unmarshal([]byte(w.body), &b)
			if jsonEq(a, b) {
				c.body = w.body
			}
		}
		if c != w {
			t.Errorf("call %d: %+v, want %+v", i, c, w)
		}
	}
	if v := string(res.Result); v != "true" {
		t.Fatalf("value %s", v)
	}
}

func jsonEq(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}
