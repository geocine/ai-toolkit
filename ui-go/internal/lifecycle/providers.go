package lifecycle

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// Provider ends this server: stop (the machine halts, its disk is kept and
// usually still billed) or destroy (gone for good).
type Provider interface {
	Name() string
	// Stop and Destroy return a human detail of what happened.
	Stop(ctx context.Context) (string, error)
	Destroy(ctx context.Context) (string, error)
}

// API bases (tests point them at httptest servers).
var (
	runpodAPI   = "https://api.runpod.io"
	runpodV1API = "https://rest.runpod.io/v1"
	vastAPI     = "https://console.vast.ai/api/v0"
	quickpodAPI = "https://api.quickpod.org"
)

// v1Retire is the day RunPod retires REST v1. Until then a pod key that v2
// refuses is retried on v1.
var v1Retire = time.Date(2026, 11, 15, 0, 0, 0, 0, time.UTC)

// now is the clock the v1 cutoff reads (tests move it).
var now = time.Now

var httpClient = &http.Client{Timeout: 60 * time.Second}

func env(key string) string { return strings.TrimSpace(os.Getenv(key)) }

// detectProvider picks the provider from the environment:
// AITK_WHEN_DONE_PROVIDER (runpod | vast | quickpod | command | exit) when
// set, else AITK_WHEN_DONE_CMD, else RunPod (RUNPOD_POD_ID), else Vast.ai
// (CONTAINER_ID with CONTAINER_API_KEY), else QuickPod (CONTAINER_LABEL,
// QUICKPOD_PORT_8675 and AITK_QUICKPOD_API_KEY), else a plain process exit.
func detectProvider() Provider {
	runpod := &runPod{id: env("RUNPOD_POD_ID"), key: firstNonEmpty(env("AITK_RUNPOD_API_KEY"), env("RUNPOD_API_KEY")),
		podKey: env("AITK_RUNPOD_API_KEY") == ""}
	vast := &vastAI{id: env("CONTAINER_ID"), key: firstNonEmpty(env("AITK_VAST_API_KEY"), env("CONTAINER_API_KEY"))}
	quickpod := &quickPod{id: env("CONTAINER_LABEL"), key: env("AITK_QUICKPOD_API_KEY")}
	cmd := &command{cmd: env("AITK_WHEN_DONE_CMD")}
	switch strings.ToLower(env("AITK_WHEN_DONE_PROVIDER")) {
	case "runpod":
		return runpod
	case "vast", "vastai", "vast.ai":
		return vast
	case "quickpod":
		return quickpod
	case "command", "cmd":
		return cmd
	case "exit":
		return &exitProvider{}
	}
	switch {
	case cmd.cmd != "":
		return cmd
	case runpod.id != "":
		return runpod
	case vast.id != "" && vast.key != "":
		return vast
	case quickpod.id != "" && env("QUICKPOD_PORT_8675") != "" && quickpod.key != "":
		return quickpod
	}
	return &exitProvider{}
}

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

// apiError is a provider's non-2xx answer.
type apiError struct {
	status int
	body   string
}

func (e *apiError) Error() string { return fmt.Sprintf("HTTP %d: %s", e.status, e.body) }

func denied(err error) bool {
	var ae *apiError
	return errors.As(err, &ae) && (ae.status == http.StatusUnauthorized || ae.status == http.StatusForbidden)
}

// call sends a Bearer-authenticated JSON request.
func call(ctx context.Context, method, u, key string, body any) ([]byte, error) {
	return callWith(ctx, method, u, http.Header{"Authorization": {"Bearer " + key}}, body)
}

// callWith sends a JSON request with the given auth headers and returns the
// body of a 2xx answer, or an *apiError.
func callWith(ctx context.Context, method, u string, auth http.Header, body any) ([]byte, error) {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, u, rd)
	if err != nil {
		return nil, err
	}
	for k, vs := range auth {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if resp.StatusCode/100 != 2 {
		return out, &apiError{status: resp.StatusCode, body: strings.TrimSpace(string(clip(out, 300)))}
	}
	return out, nil
}

func clip(b []byte, n int) []byte {
	if len(b) > n {
		return b[:n]
	}
	return b
}

// runPod uses RunPod's REST API v2 (https://docs.runpod.io/api-reference-v2):
// DELETE /v2/pods/{id} terminates (204), POST /v2/pods/{id}/action
// {"action":"stop"} stops (200). The RUNPOD_API_KEY RunPod injects into a pod
// is pod-scoped: when v2 refuses it (401/403) before v1 retires (v1Retire),
// the same call is retried on v1 (POST /v1/pods/{id}/stop, DELETE
// /v1/pods/{id}). A terminate the key may not do falls back to stop.
// AITK_RUNPOD_API_KEY supplies a full key instead.
type runPod struct {
	id, key string
	podKey  bool // key is the pod-scoped RUNPOD_API_KEY
}

func (p *runPod) Name() string { return "runpod" }

func (p *runPod) ready() error {
	if p.id == "" {
		return errors.New("RUNPOD_POD_ID is not set")
	}
	if p.key == "" {
		return errors.New("no RunPod API key (RUNPOD_API_KEY or AITK_RUNPOD_API_KEY)")
	}
	return nil
}

func (p *runPod) podURL(base, suffix string) string {
	return strings.TrimRight(base, "/") + "/pods/" + url.PathEscape(p.id) + suffix
}

// do sends the v2 call and, for a pod key v2 refuses before v1 retires, the
// same call on v1. via names the API when v1 answered ("" for v2).
func (p *runPod) do(ctx context.Context, method, v2suffix string, body any, v1suffix string) (via string, err error) {
	_, err = call(ctx, method, p.podURL(strings.TrimRight(runpodAPI, "/")+"/v2", v2suffix), p.key, body)
	if err == nil || !denied(err) || !p.podKey || !now().Before(v1Retire) {
		return "", err
	}
	if _, v1err := call(ctx, method, p.podURL(runpodV1API, v1suffix), p.key, nil); v1err != nil {
		return "", fmt.Errorf("%v; API v1: %w", err, v1err)
	}
	return " through API v1 (RunPod retires it on " + v1Retire.Format("2006-01-02") + ")", nil
}

func (p *runPod) Stop(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	via, err := p.do(ctx, http.MethodPost, "/action", map[string]string{"action": "stop"}, "/stop")
	if err != nil {
		return "", fmt.Errorf("runpod stop pod %s: %w", p.id, err)
	}
	return "RunPod pod " + p.id + " stopped" + via + " (its disk is kept and still billed)", nil
}

func (p *runPod) Destroy(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	via, err := p.do(ctx, http.MethodDelete, "", nil, "")
	if err == nil {
		return "RunPod pod " + p.id + " terminated" + via, nil
	}
	if !denied(err) {
		return "", fmt.Errorf("runpod terminate pod %s: %w", p.id, err)
	}
	detail, serr := p.Stop(ctx)
	if serr != nil {
		return "", fmt.Errorf("runpod terminate pod %s: %v; stop: %w", p.id, err, serr)
	}
	return detail + "; the API key may not terminate (set AITK_RUNPOD_API_KEY to a full key to terminate)", nil
}

// vastAI uses Vast.ai's REST API (https://docs.vast.ai/api-reference):
// DELETE /api/v0/instances/{id}/ destroys, PUT /api/v0/instances/{id}/
// {"state": "stopped"} stops; both answer {"success": true}. Vast injects
// CONTAINER_ID and a per-instance CONTAINER_API_KEY that may start, stop and
// destroy that instance.
type vastAI struct{ id, key string }

func (p *vastAI) Name() string { return "vast" }

func (p *vastAI) ready() error {
	if p.id == "" || p.key == "" {
		return errors.New("CONTAINER_ID and CONTAINER_API_KEY (or AITK_VAST_API_KEY) are required")
	}
	return nil
}

func (p *vastAI) instanceURL() string {
	return strings.TrimRight(vastAPI, "/") + "/instances/" + url.PathEscape(p.id) + "/"
}

func vastOK(body []byte) error {
	var r struct {
		Success *bool  `json:"success"`
		Msg     string `json:"msg"`
		Error   string `json:"error"`
	}
	if json.Unmarshal(body, &r) == nil && r.Success != nil && !*r.Success {
		return fmt.Errorf("vast: %s", firstNonEmpty(r.Msg, r.Error, "success false"))
	}
	return nil
}

func (p *vastAI) Stop(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	body, err := call(ctx, http.MethodPut, p.instanceURL(), p.key, map[string]string{"state": "stopped"})
	if err == nil {
		err = vastOK(body)
	}
	if err != nil {
		return "", fmt.Errorf("vast stop instance %s: %w", p.id, err)
	}
	return "Vast.ai instance " + p.id + " stopped (storage is kept and still billed)", nil
}

func (p *vastAI) Destroy(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	body, err := call(ctx, http.MethodDelete, p.instanceURL(), p.key, nil)
	if err == nil {
		err = vastOK(body)
	}
	if err == nil {
		return "Vast.ai instance " + p.id + " destroyed", nil
	}
	if !denied(err) {
		return "", fmt.Errorf("vast destroy instance %s: %w", p.id, err)
	}
	detail, serr := p.Stop(ctx)
	if serr != nil {
		return "", fmt.Errorf("vast destroy instance %s: %v; stop: %w", p.id, err, serr)
	}
	return detail + "; the API key may not destroy", nil
}

// quickPod uses QuickPod's secure-key API (https://api.quickpod.org):
// GET /update/api/stoppod?pod_uuid= stops (compute billing stops, storage is
// kept) and GET /update/api/destroypod?pod_uuid= removes the pod and its
// disk. The pod UUID is CONTAINER_LABEL. QuickPod injects no API key, so
// AITK_QUICKPOD_API_KEY (a qpk_ key that may write pods) is passed in; it
// goes in both X-API-Key and Authorization: ApiKey, as QuickPod's CLI sends
// it. Any 2xx is success, and the answer's message (or result) is shown.
type quickPod struct{ id, key string }

func (p *quickPod) Name() string { return "quickpod" }

func (p *quickPod) ready() error {
	if p.id == "" || p.key == "" {
		return errors.New("CONTAINER_LABEL and AITK_QUICKPOD_API_KEY are required")
	}
	return nil
}

// act calls one pod action and returns ": <message>" from the answer, or "".
func (p *quickPod) act(ctx context.Context, action string) (string, error) {
	u := strings.TrimRight(quickpodAPI, "/") + "/update/api/" + action + "?pod_uuid=" + url.QueryEscape(p.id)
	body, err := callWith(ctx, http.MethodGet, u, http.Header{"X-API-Key": {p.key}, "Authorization": {"ApiKey " + p.key}}, nil)
	if err != nil {
		return "", err
	}
	var r struct {
		Message string `json:"message"`
		Result  string `json:"result"`
	}
	json.Unmarshal(body, &r)
	if msg := strings.TrimSpace(firstNonEmpty(r.Message, r.Result)); msg != "" {
		return ": " + string(clip([]byte(msg), 300)), nil
	}
	return "", nil
}

func (p *quickPod) Stop(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	msg, err := p.act(ctx, "stoppod")
	if err != nil {
		return "", fmt.Errorf("quickpod stop pod %s: %w", p.id, err)
	}
	return "QuickPod pod " + p.id + " stopped (storage is kept and still billed)" + msg, nil
}

func (p *quickPod) Destroy(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	msg, err := p.act(ctx, "destroypod")
	if err == nil {
		return "QuickPod pod " + p.id + " destroyed" + msg, nil
	}
	if !denied(err) {
		return "", fmt.Errorf("quickpod destroy pod %s: %w", p.id, err)
	}
	detail, serr := p.Stop(ctx)
	if serr != nil {
		return "", fmt.Errorf("quickpod destroy pod %s: %v; stop: %w", p.id, err, serr)
	}
	return detail + "; the API key may not destroy", nil
}

// command runs AITK_WHEN_DONE_CMD (sh -c, or cmd /C on Windows) with
// AITK_WHEN_DONE_ACTION=stop|destroy in its environment.
type command struct{ cmd string }

func (p *command) Name() string { return "command" }

func (p *command) run(ctx context.Context, action string) (string, error) {
	if p.cmd == "" {
		return "", errors.New("AITK_WHEN_DONE_CMD is not set")
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	var c *exec.Cmd
	if runtime.GOOS == "windows" {
		c = exec.CommandContext(ctx, "cmd", "/C", p.cmd)
	} else {
		c = exec.CommandContext(ctx, "sh", "-c", p.cmd)
	}
	c.Env = append(os.Environ(), "AITK_WHEN_DONE_ACTION="+action)
	out, err := c.CombinedOutput()
	detail := strings.TrimSpace(string(out))
	if r := []rune(detail); len(r) > 500 {
		detail = "…" + string(r[len(r)-500:])
	}
	if err != nil {
		return "", fmt.Errorf("AITK_WHEN_DONE_CMD (%s): %v: %s", action, err, detail)
	}
	if detail == "" {
		detail = "AITK_WHEN_DONE_CMD ran (" + action + ")"
	}
	return detail, nil
}

func (p *command) Stop(ctx context.Context) (string, error)    { return p.run(ctx, ModeStop) }
func (p *command) Destroy(ctx context.Context) (string, error) { return p.run(ctx, ModeDestroy) }

// exitProvider ends the server process (local and dev machines, or a
// container whose restart policy is "no"): stop and destroy are the same.
type exitProvider struct{}

func (p *exitProvider) Name() string { return "exit" }

func (p *exitProvider) Stop(ctx context.Context) (string, error) {
	go func() {
		// let the API answer and the last events go out first
		time.Sleep(exitDelay)
		exitFn()
	}()
	return "the server process exits", nil
}

func (p *exitProvider) Destroy(ctx context.Context) (string, error) { return p.Stop(ctx) }

var exitDelay = 2 * time.Second

// exitFn is what the exit provider calls; main sets it (SetExit) to cancel
// its context for a graceful shutdown.
var exitFn = func() { os.Exit(0) }

// SetExit replaces the process exit (main passes its context's cancel).
func SetExit(fn func()) {
	if fn != nil {
		exitFn = fn
	}
}
