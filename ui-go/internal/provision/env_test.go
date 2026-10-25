package provision

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/bootstrap"
	"github.com/geocine/aitoolkit/internal/notify"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/testutil"
	"github.com/geocine/aitoolkit/internal/v1api"
)

func addS3Remote(t *testing.T, name string) *remote.Remote {
	t.Helper()
	r, err := remote.Create(context.Background(), name, "s3",
		json.RawMessage(`{"endpoint":"https://s3.example.com","bucket":"runs","region":"eu-west-1","prefix":"aitk/","path_style":true}`),
		json.RawMessage(`{"access_key_id":"AKIAEXAMPLE0001","secret_access_key":"s3SecretExample0001"}`))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func addHFRemote(t *testing.T, name, token string) *remote.Remote {
	t.Helper()
	secret := `{}`
	if token != "" {
		secret = `{"token":"` + token + `"}`
	}
	r, err := remote.Create(context.Background(), name, "hf",
		json.RawMessage(`{"repo_id":"geocine/runs","repo_type":"dataset","path_in_repo":"lora/x"}`), json.RawMessage(secret))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// build runs BuildEnv for k with the stub's caps.
func (e *testEnv) build(k Kind, req LaunchRequest) (map[string]string, []string, error) {
	req.Provider = k
	return BuildEnv(e.ctx, req, k, "mch_test", "aitk_token_value", e.clouds[k].caps, "acct-key-"+string(k))
}

func TestEnvRows(t *testing.T) {
	e := newEnv(t)
	addS3Remote(t, "r2")
	e.hf = "hf_server_token_01"
	e.sinks = notify.Config{Sinks: []notify.Sink{{URL: "https://ntfy.sh/topic-a"}, {URL: "https://discord.com/api/webhooks/1/x"}},
		Events: []string{"completed", "failed"}}

	env, warn, err := e.build(RunPod, LaunchRequest{})
	if err != nil || len(env) != 2 || env["AI_TOOLKIT_AUTH"] != "aitk_token_value" || env["AITK_MACHINE_ID"] != "mch_test" || len(warn) != 0 {
		t.Fatalf("minimal: %v %v %v", env, warn, err)
	}
	if env, _, _ := e.build(Vast, LaunchRequest{}); env["OPEN_BUTTON_PORT"] != "8675" {
		t.Fatalf("vast open button: %v", env)
	}

	env, warn, err = e.build(RunPod, LaunchRequest{
		CORSOrigins: []string{"https://a.example", "http://localhost:5173"}, StudioURL: "https://a.example/studio/",
		CopyRemote: "r2", CopyHFToken: true, CopyNotify: true, WhenDone: WhenDestroy, SelfDestroyKey: true,
		Env: map[string]string{"TZ": "Europe/Berlin", "HF_HOME": "/workspace/hf", "AITK_SYNC_KEEP_CHECKPOINTS": "3"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"AI_TOOLKIT_AUTH":            "aitk_token_value",
		"AITK_MACHINE_ID":            "mch_test",
		"AITK_CORS_ORIGINS":          "https://a.example,http://localhost:5173",
		"AITK_STUDIO_URL":            "https://a.example/studio/",
		"AITK_S3_NAME":               "r2",
		"AITK_S3_ENDPOINT":           "https://s3.example.com",
		"AITK_S3_BUCKET":             "runs",
		"AITK_S3_REGION":             "eu-west-1",
		"AITK_S3_PREFIX":             "aitk/",
		"AITK_S3_PATH_STYLE":         "1",
		"AITK_S3_ACCESS_KEY_ID":      "AKIAEXAMPLE0001",
		"AITK_S3_SECRET_ACCESS_KEY":  "s3SecretExample0001",
		"AITK_SYNC_REMOTE":           "r2",
		"AITK_AUTO_RESUME":           "1",
		"HF_TOKEN":                   "hf_server_token_01",
		"AITK_NOTIFY":                "https://ntfy.sh/topic-a,https://discord.com/api/webhooks/1/x",
		"AITK_NOTIFY_EVENTS":         "completed,failed",
		"AITK_WHEN_DONE":             "destroy",
		"AITK_WHEN_DONE_PROVIDER":    "runpod",
		"AITK_RUNPOD_API_KEY":        "acct-key-runpod",
		"TZ":                         "Europe/Berlin",
		"HF_HOME":                    "/workspace/hf",
		"AITK_SYNC_KEEP_CHECKPOINTS": "3",
	}
	if len(env) != len(want) {
		t.Errorf("got %d vars, want %d: %v", len(env), len(want), env)
	}
	for k, v := range want {
		if env[k] != v {
			t.Errorf("%s = %q, want %q", k, env[k], v)
		}
	}
	if len(warn) != 0 {
		t.Errorf("warnings %v", warn)
	}

	// the account key goes only where the provider needs it
	if env, _, _ := e.build(QuickPod, LaunchRequest{SelfDestroyKey: true}); env["AITK_QUICKPOD_API_KEY"] != "acct-key-quickpod" || env["AITK_RUNPOD_API_KEY"] != "" {
		t.Fatalf("quickpod key: %v", env)
	}
	if env, _, _ := e.build(Vast, LaunchRequest{SelfDestroyKey: true}); len(env) != 3 {
		t.Fatalf("vast destroys itself with its builtin key: %v", env)
	}
	if env, _, _ := e.build(RunPod, LaunchRequest{WhenDone: WhenKeep, CopyRemote: "r2"}); env["AITK_WHEN_DONE"] != "" {
		t.Fatal("keep sends no AITK_WHEN_DONE")
	}

	// nothing to copy: a warning, not a refusal
	e.hf, e.sinks = "", notify.Config{}
	env, warn, err = e.build(Vast, LaunchRequest{CopyHFToken: true, CopyNotify: true})
	if err != nil || env["HF_TOKEN"] != "" || env["AITK_NOTIFY"] != "" || len(warn) != 2 {
		t.Fatalf("empty copies: %v %v %v", env, warn, err)
	}
}

func TestEnvRunPodDestroyWarning(t *testing.T) {
	e := newEnv(t)
	addS3Remote(t, "r2")
	_, warn, err := e.build(RunPod, LaunchRequest{CopyRemote: "r2", WhenDone: WhenDestroy})
	if err != nil || len(warn) != 1 || warn[0] != warnRunPodNoKey {
		t.Fatalf("%v %v", warn, err)
	}
	if _, warn, _ := e.build(RunPod, LaunchRequest{CopyRemote: "r2", WhenDone: WhenDestroy, SelfDestroyKey: true}); len(warn) != 0 {
		t.Fatalf("with the key: %v", warn)
	}
	m := e.launch(RunPod, func(r *LaunchRequest) { r.CopyRemote = "r2" })
	if m.WhenDone != WhenDestroy || len(m.Warnings) != 1 || m.Warnings[0] != warnRunPodNoKey {
		t.Fatalf("machine warnings %v", m.Warnings)
	}
}

func TestEnvPassthroughRules(t *testing.T) {
	e := newEnv(t)
	addS3Remote(t, "r2")
	const secret = "SECRETVALUE123456"
	for _, k := range []string{"AI_TOOLKIT_AUTH", "AITK_CORS_ORIGINS", "AITK_STUDIO_URL", "AITK_WHEN_DONE", "AITK_WHEN_DONE_DELAY",
		"AITK_PROVISION_RUNPOD_KEY", "AITK_VAST_API_KEY", "AITK_RUNPOD_API_KEY", "AITK_MACHINE_ID", "PORT", "RUNPOD_POD_ID", "VAST_X",
		"CONTAINER_API_KEY", "QUICKPOD_PORT_8675", "PUBLIC_IPADDR", "PATH", "LD_PRELOAD", "aitk_lower", "HF_"} {
		_, _, err := e.build(RunPod, LaunchRequest{Env: map[string]string{k: secret}})
		if codeOf(err) != v1api.CodeInvalidConfig || !strings.Contains(err.Error(), k) || strings.Contains(err.Error(), secret) {
			t.Errorf("%s: %v", k, err)
		}
	}
	// a passthrough may not override what the launch's own options set
	_, _, err := e.build(RunPod, LaunchRequest{CopyRemote: "r2", Env: map[string]string{"AITK_S3_BUCKET": secret}})
	if codeOf(err) != v1api.CodeInvalidConfig || strings.Contains(err.Error(), secret) {
		t.Fatalf("override: %v", err)
	}
	for _, k := range []string{"AITK_FOO", "HF_HUB_ENABLE_HF_TRANSFER", "HUGGING_FACE_HUB_TOKEN", "TZ"} {
		if env, _, err := e.build(RunPod, LaunchRequest{Env: map[string]string{k: "1"}}); err != nil || env[k] != "1" {
			t.Errorf("%s refused: %v", k, err)
		}
	}
	// HF_HOME moves the model cache onto a volume (IMAGE-PLAN §3.4, RunPod)
	const cache = "/workspace/.cache/huggingface"
	for _, k := range allKinds {
		if env, _, err := e.build(k, LaunchRequest{Env: map[string]string{"HF_HOME": cache}}); err != nil || env["HF_HOME"] != cache {
			t.Errorf("%s: HF_HOME refused: %v", k, err)
		}
	}
}

func TestEnvLimits(t *testing.T) {
	e := newEnv(t)
	// Vast: 256 per value
	if _, _, err := e.build(Vast, LaunchRequest{Env: map[string]string{"AITK_X": strings.Repeat("v", 256)}}); err != nil {
		t.Fatalf("256: %v", err)
	}
	_, _, err := e.build(Vast, LaunchRequest{Env: map[string]string{"AITK_X": strings.Repeat("v", 257)}})
	if codeOf(err) != v1api.CodeInvalidConfig || !strings.Contains(err.Error(), "AITK_X") || strings.Contains(err.Error(), "vvvv") {
		t.Fatalf("257: %v", err)
	}
	// Vast: 32768 in total
	big := map[string]string{}
	for i := range 140 {
		big[fmt.Sprintf("AITK_V%03d", i)] = strings.Repeat("v", 250)
	}
	if _, _, err := e.build(Vast, LaunchRequest{Env: big}); codeOf(err) != v1api.CodeInvalidConfig || !strings.Contains(err.Error(), "32768") {
		t.Fatalf("total: %v", err)
	}
	// RunPod: 50 variables (2 come with every launch)
	vars := func(n int) map[string]string {
		m := map[string]string{}
		for i := range n {
			m[fmt.Sprintf("AITK_N%02d", i)] = "1"
		}
		return m
	}
	if _, _, err := e.build(RunPod, LaunchRequest{Env: vars(48)}); err != nil {
		t.Fatalf("50 vars: %v", err)
	}
	if _, _, err := e.build(RunPod, LaunchRequest{Env: vars(49)}); codeOf(err) != v1api.CodeInvalidConfig || !strings.Contains(err.Error(), "51") {
		t.Fatalf("51 vars: %v", err)
	}
	// QuickPod: one docker_options string, so a strict charset
	for _, v := range []string{"a b", "x;rm", "$(id)", "a|b", "q\"", "back`tick", "*", "a?b", "~x", "[ab]", "x!y"} {
		_, _, err := e.build(QuickPod, LaunchRequest{Env: map[string]string{"AITK_Q": v}})
		if codeOf(err) != v1api.CodeInvalidConfig || !strings.Contains(err.Error(), "AITK_Q") || strings.Contains(err.Error(), v) {
			t.Errorf("quickpod %q: %v", v, err)
		}
	}
	if _, _, err := e.build(QuickPod, LaunchRequest{Env: map[string]string{"AITK_Q": "https://x.example/a,b:1"}}); err != nil {
		t.Fatalf("quickpod plain value: %v", err)
	}
	// the same values are fine where env is not a flag string
	if _, _, err := e.build(RunPod, LaunchRequest{Env: map[string]string{"AITK_Q": "a b"}}); err != nil {
		t.Fatalf("runpod space: %v", err)
	}
}

// TestEnvRemoteRoundTrip feeds the built env to bootstrap (what the GPU
// server runs at boot) and checks it recreates the same remote.
func TestEnvRemoteRoundTrip(t *testing.T) {
	type s3 struct {
		Endpoint, Bucket, Region, Prefix string
		PathStyle                        bool `json:"path_style"`
	}
	type s3sec struct {
		AccessKeyID     string `json:"access_key_id"`
		SecretAccessKey string `json:"secret_access_key"`
	}
	type hf struct {
		RepoID     string `json:"repo_id"`
		RepoType   string `json:"repo_type"`
		PathInRepo string `json:"path_in_repo"`
	}
	type hfsec struct{ Token string }

	for _, kind := range []string{"s3", "hf"} {
		t.Run(kind, func(t *testing.T) {
			e := newEnv(t)
			for _, k := range []string{"AITK_REMOTES", "AITK_S3_ENDPOINT", "AITK_HF_REPO", "AITK_HF_TOKEN", "HF_TOKEN",
				"AITK_SYNC_REMOTE", "AITK_AUTO_RESUME", "AITK_AUTO_UPLOAD_REMOTE", "AITK_SYNC_KEEP_CHECKPOINTS", "AITK_MAX_AUTO_RESUMES"} {
				t.Setenv(k, "")
			}
			var orig *remote.Remote
			if kind == "s3" {
				orig = addS3Remote(t, "store")
			} else {
				orig = addHFRemote(t, "store", "hf_remote_token_01")
			}
			env, _, err := e.build(Vast, LaunchRequest{CopyRemote: "store"})
			if err != nil {
				t.Fatal(err)
			}
			if ok, err := remote.Delete(e.ctx, orig.ID); err != nil || !ok {
				t.Fatal(ok, err)
			}
			for k, v := range env {
				if strings.HasPrefix(k, "AITK_S3_") || strings.HasPrefix(k, "AITK_HF_") || k == "AITK_SYNC_REMOTE" || k == "AITK_AUTO_RESUME" {
					t.Setenv(k, v)
				}
			}
			if err := bootstrap.Apply(e.ctx); err != nil {
				t.Fatal(err)
			}
			got, err := remote.Get(e.ctx, "store")
			if err != nil || got == nil || got.Kind != kind {
				t.Fatalf("remote after boot: %+v %v", got, err)
			}
			same := func(a, b json.RawMessage, v1, v2 any) {
				t.Helper()
				if err := json.Unmarshal(a, v1); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(b, v2); err != nil {
					t.Fatal(err)
				}
				if fmt.Sprint(v1) != fmt.Sprint(v2) {
					t.Fatalf("%s: got %+v, want %+v", kind, v1, v2)
				}
			}
			if kind == "s3" {
				same(got.Config, orig.Config, &s3{}, &s3{})
				same(got.Secret, orig.Secret, &s3sec{}, &s3sec{})
			} else {
				same(got.Config, orig.Config, &hf{}, &hf{})
				same(got.Secret, orig.Secret, &hfsec{}, &hfsec{})
			}
		})
	}
}

func TestEnvRemoteKinds(t *testing.T) {
	e := newEnv(t)
	if _, err := remote.Create(e.ctx, "local", "fs", json.RawMessage(`{"path":"`+strings.ReplaceAll(testutil.TempDir(t), `\`, `/`)+`"}`), nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := e.build(RunPod, LaunchRequest{CopyRemote: "local"}); codeOf(err) != v1api.CodeInvalidRequest || !strings.Contains(err.Error(), "folder") {
		t.Fatalf("fs remote: %v", err)
	}
	if _, _, err := e.build(RunPod, LaunchRequest{CopyRemote: "missing"}); codeOf(err) != v1api.CodeInvalidRequest {
		t.Fatalf("unknown remote: %v", err)
	}
	addHFRemote(t, "hub", "")
	env, warn, err := e.build(RunPod, LaunchRequest{CopyRemote: "hub"})
	if err != nil || env["AITK_HF_TOKEN"] != "" || len(warn) != 1 || !strings.Contains(warn[0], "no token") {
		t.Fatalf("hf without a token: %v %v %v", env, warn, err)
	}
	if _, warn, _ := e.build(RunPod, LaunchRequest{CopyRemote: "hub", CopyHFToken: true}); len(warn) != 1 || !strings.Contains(warn[0], "no Hugging Face token") {
		t.Fatalf("hf token copied instead: %v", warn) // this server has none either (e.hf == "")
	}
}
