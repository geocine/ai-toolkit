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
	runpodAPI = "https://rest.runpod.io/v1"
	vastAPI   = "https://console.vast.ai/api/v0"
)

var httpClient = &http.Client{Timeout: 60 * time.Second}

func env(key string) string { return strings.TrimSpace(os.Getenv(key)) }

// detectProvider picks the provider from the environment:
// AITK_WHEN_DONE_PROVIDER (runpod | vast | command | exit) when set, else
// AITK_WHEN_DONE_CMD, else RunPod (RUNPOD_POD_ID), else Vast.ai
// (CONTAINER_ID with CONTAINER_API_KEY), else a plain process exit.
func detectProvider() Provider {
	runpod := &runPod{id: env("RUNPOD_POD_ID"), key: firstNonEmpty(env("AITK_RUNPOD_API_KEY"), env("RUNPOD_API_KEY"))}
	vast := &vastAI{id: env("CONTAINER_ID"), key: firstNonEmpty(env("AITK_VAST_API_KEY"), env("CONTAINER_API_KEY"))}
	cmd := &command{cmd: env("AITK_WHEN_DONE_CMD")}
	switch strings.ToLower(env("AITK_WHEN_DONE_PROVIDER")) {
	case "runpod":
		return runpod
	case "vast", "vastai", "vast.ai":
		return vast
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

func call(ctx context.Context, method, u, key string, body any) ([]byte, error) {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, u, rd)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+key)
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

// runPod uses RunPod's REST API (https://docs.runpod.io/api-reference):
// DELETE /v1/pods/{podId} terminates (204), POST /v1/pods/{podId}/stop stops
// (200). The RUNPOD_API_KEY RunPod injects into a pod is pod-scoped and may
// not be allowed to terminate; a 401/403 there falls back to stop.
// AITK_RUNPOD_API_KEY supplies a full key instead.
type runPod struct{ id, key string }

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

func (p *runPod) podURL(suffix string) string {
	return strings.TrimRight(runpodAPI, "/") + "/pods/" + url.PathEscape(p.id) + suffix
}

func (p *runPod) Stop(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	if _, err := call(ctx, http.MethodPost, p.podURL("/stop"), p.key, nil); err != nil {
		return "", fmt.Errorf("runpod stop pod %s: %w", p.id, err)
	}
	return "RunPod pod " + p.id + " stopped (its disk is kept and still billed)", nil
}

func (p *runPod) Destroy(ctx context.Context) (string, error) {
	if err := p.ready(); err != nil {
		return "", err
	}
	_, err := call(ctx, http.MethodDelete, p.podURL(""), p.key, nil)
	if err == nil {
		return "RunPod pod " + p.id + " terminated", nil
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
