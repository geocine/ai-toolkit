package provision

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/notify"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// What BuildEnv copies from this server; tests swap them.
var (
	hfToken      = settings.HFToken
	notifyConfig = notify.Current
)

// passthroughKey is what LaunchRequest.env may set.
var passthroughKey = regexp.MustCompile(`^(AITK_[A-Z0-9_]{1,60}|HF_[A-Z0-9_]{1,60}|HUGGING_FACE_HUB_TOKEN|TZ)$`)

// QuickPod carries env inside one docker_options string ("-e K=V ..."), so
// keys and values are held to a strict charset (no quoting is attempted).
var (
	quickpodKey      = regexp.MustCompile(`^[A-Z_][A-Z0-9_]{0,63}$`)
	quickpodBadChars = " \t\r\n\v\f\"'`$\\;|&<>"
)

// reservedEnv: set by the launch itself, or the provider's own; a request
// may never pass them.
func reservedEnv(k string) bool {
	switch k {
	case "AI_TOOLKIT_AUTH", "AITK_CORS_ORIGINS", "AITK_STUDIO_URL", "AITK_MACHINE_ID", "PORT", "PUBLIC_IPADDR":
		return true
	}
	if strings.HasPrefix(k, "AITK_WHEN_DONE") || strings.HasPrefix(k, "AITK_PROVISION_") ||
		(strings.HasPrefix(k, "AITK_") && strings.HasSuffix(k, "_API_KEY")) {
		return true
	}
	for _, p := range []string{"RUNPOD_", "VAST_", "CONTAINER_", "QUICKPOD_", "PUBLIC_IPADDR"} {
		if strings.HasPrefix(k, p) {
			return true
		}
	}
	return false
}

func envInvalid(format string, a ...any) error {
	return v1api.Errorf(v1api.CodeInvalidConfig, fmt.Sprintf(format, a...))
}

// RunPod destroy without the account key: the pod can only stop itself.
const warnRunPodNoKey = "Without your key this pod can only stop itself, not delete itself. A stopped RunPod pod wipes its container disk; terminate it from here or the RunPod console."

// BuildEnv is the machine's environment (PLAN §5 Launch step 7) and the
// warnings to show on the Machine. req has its defaults applied (when_done,
// cors_origins). key is the stored launcher key (self_destroy_key copies it).
// Errors name the variable, never its value.
func BuildEnv(ctx context.Context, req LaunchRequest, kind Kind, machineID, token string, caps Caps, key string) (map[string]string, []string, error) {
	env := map[string]string{
		"AI_TOOLKIT_AUTH": token,
		"AITK_MACHINE_ID": machineID,
	}
	var warnings []string
	if len(req.CORSOrigins) > 0 {
		env["AITK_CORS_ORIGINS"] = strings.Join(req.CORSOrigins, ",")
	}
	if req.StudioURL != "" {
		env["AITK_STUDIO_URL"] = req.StudioURL
	}
	if kind == Vast {
		env["OPEN_BUTTON_PORT"] = strconv.Itoa(Port)
	}
	if req.CopyRemote != "" {
		name, w, err := remoteEnv(ctx, req.CopyRemote, req.CopyHFToken, env)
		if err != nil {
			return nil, nil, err
		}
		warnings = append(warnings, w...)
		env["AITK_SYNC_REMOTE"] = name
		env["AITK_AUTO_RESUME"] = "1"
	}
	if req.CopyHFToken {
		tok, err := hfToken(ctx)
		if err != nil {
			return nil, nil, err
		}
		if tok == "" {
			warnings = append(warnings, "This server has no Hugging Face token, so none was copied.")
		} else {
			env["HF_TOKEN"] = tok
		}
	}
	if req.CopyNotify {
		cfg := notifyConfig()
		var urls []string
		for _, s := range cfg.Sinks {
			urls = append(urls, s.URL)
		}
		if len(urls) == 0 {
			warnings = append(warnings, "This server has no notification sinks, so none were copied.")
		} else {
			env["AITK_NOTIFY"] = strings.Join(urls, ",")
			if cfg.Events != nil {
				env["AITK_NOTIFY_EVENTS"] = strings.Join(cfg.Events, ",")
			}
		}
	}
	if req.WhenDone != "" && req.WhenDone != WhenKeep {
		env["AITK_WHEN_DONE"] = string(req.WhenDone)
		env["AITK_WHEN_DONE_PROVIDER"] = string(kind)
	}
	if req.SelfDestroyKey && key != "" {
		switch kind {
		case RunPod:
			env["AITK_RUNPOD_API_KEY"] = key
		case QuickPod:
			env["AITK_QUICKPOD_API_KEY"] = key
		}
	}
	if kind == RunPod && req.WhenDone == WhenDestroy && !req.SelfDestroyKey {
		warnings = append(warnings, warnRunPodNoKey)
	}

	names := make([]string, 0, len(req.Env))
	for k := range req.Env {
		names = append(names, k)
	}
	slices.Sort(names)
	for _, k := range names {
		if reservedEnv(k) {
			return nil, nil, envInvalid("env %s is set by the launch itself; it can't be passed in env", k)
		}
		if !passthroughKey.MatchString(k) {
			return nil, nil, envInvalid("env %s isn't allowed: env passes only AITK_*, HF_*, HUGGING_FACE_HUB_TOKEN and TZ", k)
		}
		if _, set := env[k]; set {
			return nil, nil, envInvalid("env %s is already set by this launch's options", k)
		}
		if v := req.Env[k]; v != "" {
			env[k] = v
		}
	}
	if err := checkEnvLimits(kind, caps, env); err != nil {
		return nil, nil, err
	}
	return env, warnings, nil
}

// remoteEnv adds the AITK_S3_* or AITK_HF_* variables bootstrap.remotesFromEnv
// reads back into the same remote, and returns the remote's name.
func remoteEnv(ctx context.Context, name string, copyHF bool, env map[string]string) (string, []string, error) {
	r, err := remote.Get(ctx, name)
	if err != nil {
		return "", nil, err
	}
	if r == nil {
		return "", nil, v1api.Errorf(v1api.CodeInvalidRequest, "copy_remote: no remote named "+strconv.Quote(name))
	}
	set := func(k, v string) {
		if v != "" {
			env[k] = v
		}
	}
	var warnings []string
	switch r.Kind {
	case "s3":
		var c struct {
			Endpoint, Bucket, Region, Prefix string
			PathStyle                        bool `json:"path_style"`
		}
		var sec struct {
			AccessKeyID     string `json:"access_key_id"`
			SecretAccessKey string `json:"secret_access_key"`
		}
		if err := decodeRemote(r, &c, &sec); err != nil {
			return "", nil, err
		}
		set("AITK_S3_NAME", r.Name)
		set("AITK_S3_ENDPOINT", c.Endpoint)
		set("AITK_S3_BUCKET", c.Bucket)
		set("AITK_S3_REGION", c.Region)
		set("AITK_S3_PREFIX", c.Prefix)
		if c.PathStyle {
			env["AITK_S3_PATH_STYLE"] = "1"
		}
		set("AITK_S3_ACCESS_KEY_ID", sec.AccessKeyID)
		set("AITK_S3_SECRET_ACCESS_KEY", sec.SecretAccessKey)
	case "hf":
		var c struct {
			RepoID     string `json:"repo_id"`
			RepoType   string `json:"repo_type"`
			PathInRepo string `json:"path_in_repo"`
			Revision   string `json:"revision"`
		}
		var sec struct{ Token string }
		if err := decodeRemote(r, &c, &sec); err != nil {
			return "", nil, err
		}
		set("AITK_HF_NAME", r.Name)
		set("AITK_HF_REPO", c.RepoID)
		set("AITK_HF_REPO_TYPE", c.RepoType)
		set("AITK_HF_PATH", c.PathInRepo)
		set("AITK_HF_TOKEN", sec.Token)
		if c.Revision != "" {
			warnings = append(warnings, "The GPU server uses the repo's default branch; the remote's revision "+strconv.Quote(c.Revision)+" isn't copied.")
		}
		if sec.Token == "" && !copyHF {
			warnings = append(warnings, "The Hugging Face remote has no token of its own; copy the HF token too, or the GPU server can't push to it.")
		}
	case "fs":
		return "", nil, v1api.Errorf(v1api.CodeInvalidRequest, "copy_remote: "+strconv.Quote(name)+" is a folder on this server, which a GPU server can't reach; pick an S3 or Hugging Face remote")
	default:
		return "", nil, v1api.Errorf(v1api.CodeInvalidRequest, "copy_remote: "+strconv.Quote(name)+" is a "+r.Kind+" remote; only S3 and Hugging Face remotes can be copied")
	}
	return r.Name, warnings, nil
}

func decodeRemote(r *remote.Remote, config, secret any) error {
	if err := json.Unmarshal(r.Config, config); err != nil {
		return fmt.Errorf("remote %s config: %w", r.Name, err)
	}
	if len(r.Secret) > 0 {
		if err := json.Unmarshal(r.Secret, secret); err != nil {
			return fmt.Errorf("remote %s credentials: %w", r.Name, err)
		}
	}
	return nil
}

// checkEnvLimits applies the provider's documented env limits (Caps).
func checkEnvLimits(kind Kind, caps Caps, env map[string]string) error {
	p := Label(kind)
	if caps.EnvCountMax > 0 && len(env) > caps.EnvCountMax {
		return envInvalid("%s takes at most %d environment variables; this launch has %d", p, caps.EnvCountMax, len(env))
	}
	names := make([]string, 0, len(env))
	total := 0
	for k, v := range env {
		names = append(names, k)
		total += len(k) + 1 + len(v)
	}
	slices.Sort(names)
	for _, k := range names {
		v := env[k]
		if caps.EnvValueMax > 0 && len(v) > caps.EnvValueMax {
			return envInvalid("env %s is %d characters; %s takes at most %d per value", k, len(v), p, caps.EnvValueMax)
		}
		if caps.EnvFlagString {
			if !quickpodKey.MatchString(k) {
				return envInvalid("env %s: %s takes only names of capital letters, digits and _", k, p)
			}
			if strings.ContainsAny(v, quickpodBadChars) {
				return envInvalid("env %s has a space or one of \" ' ` $ \\ ; | & < >, which %s can't carry", k, p)
			}
		}
	}
	if caps.EnvTotalMax > 0 && total > caps.EnvTotalMax {
		return envInvalid("the environment is %d characters; %s takes at most %d", total, p, caps.EnvTotalMax)
	}
	return nil
}
