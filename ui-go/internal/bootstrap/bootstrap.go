// Package bootstrap configures a fresh server from its environment, so the
// single image started with env vars on any GPU provider is ready to train,
// and picks up portable jobs left by a server that went away, without any API
// calls. See docs/deploy.md for the variables.
package bootstrap

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/portable"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

func env(key string) string { return strings.TrimSpace(os.Getenv(key)) }

func envOr(key, def string) string {
	if v := env(key); v != "" {
		return v
	}
	return def
}

func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

// remoteSpec is one entry of AITK_REMOTES.
type remoteSpec struct {
	Name        string          `json:"name"`
	Kind        string          `json:"kind"`
	Config      json.RawMessage `json:"config"`
	Credentials json.RawMessage `json:"credentials"`
}

// remotesFromEnv collects AITK_REMOTES plus the S3-compatible and Hugging
// Face shorthands.
func remotesFromEnv() ([]remoteSpec, error) {
	var specs []remoteSpec
	if raw := env("AITK_REMOTES"); raw != "" {
		if err := json.Unmarshal([]byte(raw), &specs); err != nil {
			return nil, fmt.Errorf("AITK_REMOTES is not a JSON array of {name, kind, config, credentials}: %w", err)
		}
	}
	if endpoint := env("AITK_S3_ENDPOINT"); endpoint != "" {
		// an empty region lets the client pick ("auto" for R2, the bucket's own for AWS)
		cfg := map[string]any{"endpoint": endpoint, "bucket": env("AITK_S3_BUCKET"), "region": env("AITK_S3_REGION"),
			"prefix": env("AITK_S3_PREFIX")}
		if env("AITK_S3_PATH_STYLE") == "1" || env("AITK_S3_PATH_STYLE") == "true" {
			cfg["path_style"] = true
		}
		specs = append(specs, remoteSpec{Name: envOr("AITK_S3_NAME", "s3"), Kind: "s3", Config: mustJSON(cfg),
			Credentials: mustJSON(map[string]string{
				"access_key_id": env("AITK_S3_ACCESS_KEY_ID"), "secret_access_key": env("AITK_S3_SECRET_ACCESS_KEY")})})
	}
	if repo := env("AITK_HF_REPO"); repo != "" {
		cfg := map[string]any{"repo_id": repo, "repo_type": envOr("AITK_HF_REPO_TYPE", "model"), "private": true,
			"path_in_repo": env("AITK_HF_PATH")}
		secret := map[string]string{}
		if tok := envOr("AITK_HF_TOKEN", env("HF_TOKEN")); tok != "" {
			secret["token"] = tok
		}
		specs = append(specs, remoteSpec{Name: envOr("AITK_HF_NAME", "hf"), Kind: "hf", Config: mustJSON(cfg),
			Credentials: mustJSON(secret)})
	}
	return specs, nil
}

func boolEnv(key string) (bool, bool) {
	switch strings.ToLower(env(key)) {
	case "1", "true", "yes", "on":
		return true, true
	case "0", "false", "no", "off":
		return false, true
	}
	return false, false
}

// Apply writes env configuration into the fork database. It runs at boot,
// before the worker, supervisor and sync loops start.
func Apply(ctx context.Context) error {
	specs, err := remotesFromEnv()
	if err != nil {
		return err
	}
	for _, s := range specs {
		if s.Name == "" || s.Kind == "" {
			return fmt.Errorf("remote from env needs name and kind: %+v", s)
		}
		if _, err := remote.Upsert(ctx, s.Name, s.Kind, s.Config, s.Credentials); err != nil {
			return fmt.Errorf("remote %q from env: %w", s.Name, err)
		}
		log.Printf("bootstrap: remote %q (%s) configured from env", s.Name, s.Kind)
	}
	if v, ok := boolEnv("AITK_AUTO_RESUME"); ok {
		if err := state.SetSetting(ctx, state.SettingAutoResumeDefault, strconv.FormatBool(v)); err != nil {
			return err
		}
	}
	if v := env("AITK_MAX_AUTO_RESUMES"); v != "" {
		if _, err := strconv.Atoi(v); err != nil {
			return fmt.Errorf("AITK_MAX_AUTO_RESUMES must be a number: %q", v)
		}
		if err := state.SetSetting(ctx, state.SettingMaxAutoResumesDefault, v); err != nil {
			return err
		}
	}
	for envKey, setting := range map[string]string{
		"AITK_SYNC_REMOTE":        state.SettingSyncRemoteDefault,
		"AITK_AUTO_UPLOAD_REMOTE": state.SettingAutoUploadRemoteDefault,
	} {
		if v := env(envKey); v != "" {
			r, err := remote.Get(ctx, v)
			if err != nil {
				return err
			}
			if r == nil {
				return fmt.Errorf("%s names remote %q, which is not configured", envKey, v)
			}
			if err := state.SetSetting(ctx, setting, r.ID); err != nil {
				return err
			}
		}
	}
	return nil
}

const importPoll = 30 * time.Second

// ImportOnBoot (AITK_IMPORT_ON_BOOT=all or a comma list of job names) brings
// unfinished portable jobs from the import remote (AITK_IMPORT_REMOTE, else
// AITK_SYNC_REMOTE) to this server and starts them. A job whose lease another
// server still holds is retried until the lease expires (AITK_IMPORT_FORCE=1
// takes it over at once), for up to AITK_IMPORT_WAIT (default 15m).
func ImportOnBoot(ctx context.Context) {
	which := env("AITK_IMPORT_ON_BOOT")
	if which == "" {
		return
	}
	rem := envOr("AITK_IMPORT_REMOTE", env("AITK_SYNC_REMOTE"))
	if rem == "" {
		log.Printf("bootstrap: AITK_IMPORT_ON_BOOT is set but neither AITK_IMPORT_REMOTE nor AITK_SYNC_REMOTE names a remote")
		return
	}
	wait := 15 * time.Minute
	if d, err := time.ParseDuration(env("AITK_IMPORT_WAIT")); err == nil && d > 0 {
		wait = d
	}
	force, _ := boolEnv("AITK_IMPORT_FORCE")
	var names []string
	if which != "all" && which != "*" {
		for _, n := range strings.Split(which, ",") {
			if n = strings.TrimSpace(n); n != "" {
				names = append(names, n)
			}
		}
	}
	deadline := time.Now().Add(wait)
	for {
		pending := importPass(ctx, rem, names, force)
		if pending == 0 || time.Now().After(deadline) {
			if pending > 0 {
				log.Printf("bootstrap: gave up importing %d job(s) after %s", pending, wait)
			}
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(importPoll):
		}
	}
}

// importPass imports what it can and returns how many candidates are left.
func importPass(ctx context.Context, rem string, names []string, force bool) int {
	jobs, err := portable.ListRemoteJobs(ctx, rem)
	if err != nil {
		log.Printf("bootstrap: list portable jobs on %q: %v", rem, err)
		return 1 // retry: the remote may not be reachable yet
	}
	node, _ := state.NodeID(ctx)
	pending := 0
	for _, rj := range jobs {
		if names != nil && !slices.Contains(names, rj.Name) {
			continue
		}
		if names == nil && (rj.State == jobstate.Completed || rj.State == jobstate.Draft) {
			continue // "all" means unfinished work
		}
		local, err := store.FindFirstJob(ctx, store.JobQuery{Where: `"name" = ?`, Args: []any{rj.Name}})
		if err != nil {
			pending++
			continue
		}
		if local != nil {
			continue // already here (imported before this restart, or created locally)
		}
		if !force && rj.Lease != nil && rj.Lease.Held(node, time.Now()) {
			log.Printf("bootstrap: %s is leased by %s until %s; waiting", rj.Name, rj.Lease.NodeLabel, rj.Lease.ExpiresAt.Format(time.RFC3339))
			pending++
			continue
		}
		j, err := portable.Import(ctx, portable.ImportRequest{Remote: rem, Name: rj.Name, Start: true, Force: force})
		if err != nil {
			log.Printf("bootstrap: import %s: %v", rj.Name, err)
			pending++
			continue
		}
		log.Printf("bootstrap: imported %s from %q at step %d and queued it", j.Name, rem, rj.Step)
	}
	return pending
}
