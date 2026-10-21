package remote

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
)

// hf is a Hugging Face Hub repo (private by default). Operations run through
// the toolkit's own Python and huggingface_hub (the same client, with xet,
// the trainer uses to download models), via an embedded helper script.
// Config {repo_id, repo_type: model|dataset, private, path_in_repo,
// revision}; secret {token}, else the HF_TOKEN setting.
func init() { RegisterKind("hf", newHF) }

//go:embed hf_helper.py
var hfHelper string

type hfConfig struct {
	RepoID     string `json:"repo_id"`
	RepoType   string `json:"repo_type"`
	Private    *bool  `json:"private"`
	PathInRepo string `json:"path_in_repo"`
	Revision   string `json:"revision"`
}

type hfClient struct {
	cfg   hfConfig
	token string
	base  string // path_in_repo, no slashes at the ends
}

// hfTimeout bounds one helper call; uploads of big checkpoints get longer.
const (
	hfTimeout       = 10 * time.Minute
	hfUploadTimeout = 6 * time.Hour
)

func newHF(config, secret json.RawMessage) (Client, error) {
	var c hfConfig
	if err := json.Unmarshal(config, &c); err != nil {
		return nil, fmt.Errorf("hf remote config: %w", err)
	}
	if c.RepoID == "" || !strings.Contains(c.RepoID, "/") {
		return nil, errors.New(`hf remote needs config.repo_id ("owner/name")`)
	}
	switch c.RepoType {
	case "":
		c.RepoType = "model"
	case "model", "dataset":
	default:
		return nil, fmt.Errorf("hf remote repo_type must be model or dataset, not %q", c.RepoType)
	}
	if c.Revision == "" {
		c.Revision = "main"
	}
	var s struct {
		Token string `json:"token"`
	}
	if len(secret) > 0 {
		if err := json.Unmarshal(secret, &s); err != nil {
			return nil, fmt.Errorf("hf remote credentials: %w", err)
		}
	}
	return &hfClient{cfg: c, token: s.Token, base: strings.Trim(c.PathInRepo, "/")}, nil
}

func (c *hfClient) key(k string) (string, error) {
	clean := path.Clean("/" + strings.ReplaceAll(k, `\`, "/"))
	if clean == "/" {
		return "", fmt.Errorf("invalid key %q", k)
	}
	if c.base == "" {
		return clean[1:], nil
	}
	return c.base + clean, nil
}

func (c *hfClient) rel(p string) string {
	if c.base == "" {
		return p
	}
	return strings.TrimPrefix(p, c.base+"/")
}

// hfRequest is what the helper reads on stdin.
type hfRequest struct {
	Op        string `json:"op"`
	RepoID    string `json:"repo_id"`
	RepoType  string `json:"repo_type"`
	Revision  string `json:"revision"`
	Private   bool   `json:"private"`
	Key       string `json:"key,omitempty"`
	LocalPath string `json:"local_path,omitempty"`
	Data      string `json:"data,omitempty"`
	Base      string `json:"base,omitempty"`
}

type hfResponse struct {
	Error    string  `json:"error"`
	NotFound bool    `json:"not_found"`
	URL      string  `json:"url"`
	Data     string  `json:"data"`
	Size     int64   `json:"size"`
	Modified *string `json:"modified"`
	Items    []struct {
		Path     string  `json:"path"`
		Size     int64   `json:"size"`
		Modified *string `json:"modified"`
	} `json:"items"`
}

func (c *hfClient) request(op string) hfRequest {
	private := true
	if c.cfg.Private != nil {
		private = *c.cfg.Private
	}
	return hfRequest{Op: op, RepoID: c.cfg.RepoID, RepoType: c.cfg.RepoType, Revision: c.cfg.Revision, Private: private}
}

// hfPython is the interpreter used for the helper; tests override it.
var hfPython = paths.ResolvePython

// hfExtraEnv is added to the helper's environment; tests use it for PYTHONPATH.
var hfExtraEnv []string

func (c *hfClient) resolveToken(ctx context.Context) string {
	if c.token != "" {
		return c.token
	}
	tok, _ := settings.HFToken(ctx)
	return tok
}

func (c *hfClient) run(ctx context.Context, req hfRequest, timeout time.Duration) (*hfResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	in, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, hfPython(), "-c", hfHelper)
	cmd.Dir = paths.ToolkitRoot
	cmd.Env = append(os.Environ(), "HF_HUB_DISABLE_PROGRESS_BARS=1", "PYTHONIOENCODING=utf-8", "PYTHONUNBUFFERED=1")
	if tok := c.resolveToken(ctx); tok != "" {
		cmd.Env = append(cmd.Env, "HF_TOKEN="+tok)
	}
	cmd.Env = append(cmd.Env, hfExtraEnv...)
	hfHideWindow(cmd)
	cmd.Stdin = bytes.NewReader(in)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	runErr := cmd.Run()

	// the result is the last non-empty stdout line; anything before it is
	// library chatter
	lines := strings.Split(strings.TrimSpace(stdout.String()), "\n")
	var resp hfResponse
	if last := strings.TrimSpace(lines[len(lines)-1]); last == "" || json.Unmarshal([]byte(last), &resp) != nil {
		msg := strings.TrimSpace(stderr.String())
		if len(msg) > 2000 {
			msg = "..." + msg[len(msg)-2000:]
		}
		if ctx.Err() != nil {
			return nil, fmt.Errorf("hf %s: timed out after %s", req.Op, timeout)
		}
		if runErr != nil {
			return nil, fmt.Errorf("hf %s: helper failed (%v): %s", req.Op, runErr, msg)
		}
		return nil, fmt.Errorf("hf %s: helper returned no result: %s", req.Op, msg)
	}
	if resp.Error != "" {
		err := fmt.Errorf("hf %s %s: %s", req.Op, c.cfg.RepoID, resp.Error)
		if resp.NotFound {
			return nil, fmt.Errorf("%w: %w", err, ErrNotFound)
		}
		return nil, err
	}
	return &resp, nil
}

func parseHFTime(s *string) time.Time {
	if s == nil {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339Nano, *s)
	if err != nil {
		return time.Time{}
	}
	return t.UTC()
}

func (c *hfClient) Put(ctx context.Context, key, localPath string, progress Progress) (string, error) {
	k, err := c.key(key)
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(localPath)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(abs)
	if err != nil {
		return "", err
	}
	req := c.request("put")
	req.Key, req.LocalPath = k, abs
	resp, err := c.run(ctx, req, hfUploadTimeout)
	if err != nil {
		return "", err
	}
	if progress != nil {
		progress(st.Size()) // the hub client reports no progress back; done at once
	}
	return resp.URL, nil
}

func (c *hfClient) Get(ctx context.Context, key, localPath string, progress Progress) error {
	k, err := c.key(key)
	if err != nil {
		return err
	}
	abs, err := filepath.Abs(localPath)
	if err != nil {
		return err
	}
	req := c.request("get")
	req.Key, req.LocalPath = k, abs
	if _, err := c.run(ctx, req, hfUploadTimeout); err != nil {
		return err
	}
	if progress != nil {
		if st, err := os.Stat(abs); err == nil {
			progress(st.Size())
		}
	}
	return nil
}

func (c *hfClient) PutBytes(ctx context.Context, key string, data []byte) error {
	k, err := c.key(key)
	if err != nil {
		return err
	}
	req := c.request("put_bytes")
	req.Key, req.Data = k, base64.StdEncoding.EncodeToString(data)
	_, err = c.run(ctx, req, hfTimeout)
	return err
}

func (c *hfClient) GetBytes(ctx context.Context, key string) ([]byte, error) {
	k, err := c.key(key)
	if err != nil {
		return nil, err
	}
	req := c.request("get_bytes")
	req.Key = k
	resp, err := c.run(ctx, req, hfTimeout)
	if err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(resp.Data)
}

func (c *hfClient) Stat(ctx context.Context, key string) (*Object, error) {
	k, err := c.key(key)
	if err != nil {
		return nil, err
	}
	req := c.request("stat")
	req.Key = k
	resp, err := c.run(ctx, req, hfTimeout)
	if err != nil {
		return nil, err
	}
	return &Object{Key: key, Size: resp.Size, Modified: parseHFTime(resp.Modified)}, nil
}

func (c *hfClient) List(ctx context.Context, prefix string) ([]Object, error) {
	req := c.request("list")
	req.Base = c.base
	resp, err := c.run(ctx, req, hfTimeout)
	if err != nil {
		return nil, err
	}
	out := []Object{}
	for _, it := range resp.Items {
		if c.base != "" && !strings.HasPrefix(it.Path, c.base+"/") {
			continue
		}
		key := c.rel(it.Path)
		if !strings.HasPrefix(key, strings.TrimLeft(prefix, "/")) {
			continue
		}
		out = append(out, Object{Key: key, Size: it.Size, Modified: parseHFTime(it.Modified)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out, nil
}

func (c *hfClient) Delete(ctx context.Context, key string) error {
	k, err := c.key(key)
	if err != nil {
		return err
	}
	req := c.request("delete")
	req.Key = k
	_, err = c.run(ctx, req, hfTimeout)
	return err
}

func (c *hfClient) Check(ctx context.Context) error {
	_, err := c.run(ctx, c.request("check"), hfTimeout)
	return err
}
