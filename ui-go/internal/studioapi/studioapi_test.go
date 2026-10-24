package studioapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestAllowed(t *testing.T) {
	for _, c := range []struct {
		method, path string
		want         bool
	}{
		{"POST", "/v1/jobs", false},
		{"GET", "/v1/jobs", true},
		{"HEAD", "/v1/jobs/j1", true},
		{"DELETE", "/v1/jobs/j1", false},
		{"POST", "/v1/jobs/j1/start", false},
		{"POST", "/v1/jobsx", true}, // not under /v1/jobs: the mux answers it
		{"PUT", "/v1/queues/0", false},
		{"POST", "/v1/datasets", false},
		{"GET", "/v1/datasets", true},
		{"POST", "/v1/bundles", false},
		{"POST", "/v1/uploads", false},
		{"PUT", "/v1/lifecycle", false},
		{"GET", "/v1/lifecycle", false}, // the lifecycle loop does not run here
		{"POST", "/v1/lifecycle/cancel", false},
		{"GET", "/v1/lifecyclex", true},
		{"POST", "/v1/portable/import", false},
		{"DELETE", "/v1/files/x", false},
		{"GET", "/v1/system", true},
		{"GET", "/v1/system/cuda", false},
		{"POST", "/v1/system/anything", false},
		{"POST", "/v1/providers/vast/test", true},
		{"POST", "/v1/machines", true},
		{"POST", "/v1/images/check", true},
		{"PUT", "/v1/settings", true},
		{"POST", "/v1/keys", true},
		{"GET", "/v1/health", true},
		{"GET", "/v1/bogus", true},
		{"GET", "/api/jobs", false},
		{"POST", "/api/settings", false},
		{"GET", "/studio/x", true},
		{"POST", "/mcp", true},
		{"GET", "/", true},
	} {
		if got := Allowed(c.method, c.path); got != c.want {
			t.Errorf("Allowed(%s %s) = %v, want %v", c.method, c.path, got, c.want)
		}
	}
}

func serve(h http.Handler, method, path, host string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, nil)
	req.Host = host
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestGateHost(t *testing.T) {
	t.Cleanup(SetForTest(true))
	t.Setenv("AI_TOOLKIT_AUTH", "")
	t.Setenv("AITK_ALLOWED_HOSTS", "")
	t.Setenv("AITK_PUBLIC_URL", "")
	h := Gate(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) }))

	for _, c := range []struct {
		host string
		want int
	}{
		{"evil.com:8675", http.StatusForbidden},
		{"evil.com", http.StatusForbidden},
		{"", http.StatusForbidden},
		{"127.0.0.1:8675", http.StatusTeapot},
		{"[::1]:8675", http.StatusTeapot},
		{"localhost:8675", http.StatusTeapot},
		{"LOCALHOST", http.StatusTeapot},
		{"box.lan:8675", http.StatusForbidden},
	} {
		for _, path := range []string{"/v1/health", "/mcp", "/studio/"} {
			if rec := serve(h, "GET", path, c.host); rec.Code != c.want {
				t.Errorf("Host %q %s = %d, want %d", c.host, path, rec.Code, c.want)
			}
		}
	}
	rec := serve(h, "GET", "/v1/health", "evil.com:8675")
	var body struct {
		Error struct{ Code, Message string } `json:"error"`
	}
	json.Unmarshal(rec.Body.Bytes(), &body)
	if body.Error.Code != "forbidden" || body.Error.Message != HostRefusal {
		t.Errorf("host refusal body = %s", rec.Body)
	}

	t.Setenv("AITK_ALLOWED_HOSTS", "box.lan, Other.Example:9000")
	for _, c := range []struct {
		host string
		want int
	}{
		{"box.lan:8675", http.StatusTeapot},
		{"BOX.LAN", http.StatusTeapot},
		{"other.example:9000", http.StatusTeapot},
		{"other.example:9001", http.StatusForbidden},
		{"evil.com:8675", http.StatusForbidden},
	} {
		if rec := serve(h, "GET", "/v1/health", c.host); rec.Code != c.want {
			t.Errorf("allowed hosts: Host %q = %d, want %d", c.host, rec.Code, c.want)
		}
	}

	// the operator's public name is allowed: MCP's in-process calls carry it
	t.Setenv("AITK_ALLOWED_HOSTS", "")
	for _, c := range []struct {
		public, host string
		want         int
	}{
		{"https://notify.example", "notify.example", http.StatusTeapot},
		{"https://notify.example", "NOTIFY.example:443", http.StatusTeapot},
		{"https://notify.example", "evil.com", http.StatusForbidden},
		{"https://notify.example/", "evil.com:8675", http.StatusForbidden},
		{"http://box.lan:9000", "box.lan:9000", http.StatusTeapot},
		{"http://box.lan:9000", "box.lan:9001", http.StatusForbidden},
		{"not a url", "evil.com", http.StatusForbidden},
	} {
		t.Setenv("AITK_PUBLIC_URL", c.public)
		if rec := serve(h, "GET", "/mcp", c.host); rec.Code != c.want {
			t.Errorf("AITK_PUBLIC_URL=%s: Host %q = %d, want %d", c.public, c.host, rec.Code, c.want)
		}
	}
	t.Setenv("AITK_PUBLIC_URL", "")

	// with a token the page cannot send the bearer header: the check is off
	t.Setenv("AI_TOOLKIT_AUTH", "tok")
	if rec := serve(h, "GET", "/v1/health", "evil.com:8675"); rec.Code != http.StatusTeapot {
		t.Errorf("with a token, evil.com = %d", rec.Code)
	}
}

func TestGateRefusals(t *testing.T) {
	t.Setenv("AI_TOOLKIT_AUTH", "")
	h := Gate(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) }))

	t.Cleanup(SetForTest(false))
	if rec := serve(h, "POST", "/v1/jobs", "evil.com"); rec.Code != http.StatusTeapot {
		t.Fatalf("outside studio API mode the gate is a no-op: %d", rec.Code)
	}
	SetForTest(true)

	rec := serve(h, "POST", "/v1/jobs", "127.0.0.1:8675")
	var v1 struct {
		Error struct{ Code, Message string } `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &v1); err != nil || rec.Code != http.StatusConflict ||
		v1.Error.Code != "conflict" || v1.Error.Message != Refusal {
		t.Fatalf("/v1 refusal: %d %s", rec.Code, rec.Body)
	}
	rec = serve(h, "GET", "/api/jobs", "127.0.0.1:8675")
	var api struct{ Error string }
	if err := json.Unmarshal(rec.Body.Bytes(), &api); err != nil || rec.Code != http.StatusConflict || api.Error != Refusal {
		t.Fatalf("/api refusal: %d %s", rec.Code, rec.Body)
	}
	if rec := serve(h, "GET", "/v1/jobs", "127.0.0.1:8675"); rec.Code != http.StatusTeapot {
		t.Fatalf("GET /v1/jobs = %d", rec.Code)
	}
}

func TestCheckBind(t *testing.T) {
	for _, c := range []struct {
		host, token string
		ok          bool
	}{
		{"0.0.0.0", "", false},
		{"", "", false},
		{"192.168.1.5", "", false},
		{"::", "", false},
		{"127.0.0.1", "", true},
		{"::1", "", true},
		{"[::1]", "", true},
		{"localhost", "", true},
		{"0.0.0.0", "tok", true},
	} {
		if err := CheckBind(c.host, c.token); (err == nil) != c.ok {
			t.Errorf("CheckBind(%q, %q) = %v", c.host, c.token, err)
		} else if err != nil && err.Error() != BindRefusal {
			t.Errorf("CheckBind message = %q", err)
		}
	}
}

func TestFromEnv(t *testing.T) {
	for v, want := range map[string]bool{"1": true, "true": true, "TRUE": true, "Yes": true, " yes ": true,
		"": false, "0": false, "no": false, "false": false, "on": false} {
		t.Setenv("AITK_STUDIO_API", v)
		if got := FromEnv(); got != want {
			t.Errorf("AITK_STUDIO_API=%q: FromEnv = %v", v, got)
		}
	}
}

func TestSetForTest(t *testing.T) {
	before := Enabled()
	restore := SetForTest(!before)
	if Enabled() == before {
		t.Fatal("SetForTest did not switch")
	}
	inner := SetForTest(before)
	inner()
	if Enabled() == before {
		t.Fatal("nested restore went too far")
	}
	restore()
	if Enabled() != before {
		t.Fatal("restore did not bring the previous value back")
	}
}

func TestDataDir(t *testing.T) {
	tmp := t.TempDir()
	flagDir, envDir, cfg := filepath.Join(tmp, "flag"), filepath.Join(tmp, "env"), filepath.Join(tmp, "cfg")
	// os.UserConfigDir reads these
	t.Setenv("APPDATA", cfg)
	t.Setenv("XDG_CONFIG_HOME", cfg)
	t.Setenv("HOME", cfg)

	t.Setenv("AITK_STUDIO_API_DIR", envDir)
	if got, err := DataDir(flagDir); err != nil || got != flagDir {
		t.Fatalf("flag first: %q %v", got, err)
	}
	if got, err := DataDir(""); err != nil || got != envDir {
		t.Fatalf("env second: %q %v", got, err)
	}
	t.Setenv("AITK_STUDIO_API_DIR", "")
	base, err := os.UserConfigDir()
	if err != nil {
		t.Fatal(err)
	}
	if got, err := DataDir(""); err != nil || got != filepath.Join(base, "aitk-studio-api") {
		t.Fatalf("default: %q %v (config dir %q)", got, err, base)
	}
	for _, d := range []string{flagDir, envDir} {
		if st, err := os.Stat(d); err != nil || !st.IsDir() {
			t.Fatalf("%s not created: %v", d, err)
		}
	}

	if runtime.GOOS == "windows" {
		t.Skip("file modes are not enforced on Windows")
	}
	loose := filepath.Join(tmp, "loose")
	os.Mkdir(loose, 0o755)
	if _, err := DataDir(loose); err != nil {
		t.Fatal(err)
	}
	for _, d := range []string{flagDir, loose} {
		if st, _ := os.Stat(d); st.Mode().Perm() != 0o700 {
			t.Errorf("%s mode = %o, want 700", d, st.Mode().Perm())
		}
	}
}
