package bootstrap

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func openState(t *testing.T) {
	t.Helper()
	if err := state.Open(filepath.Join(testutil.TempDir(t), "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.DB.Close() })
}

func TestApplyFromEnv(t *testing.T) {
	openState(t)
	ctx := context.Background()
	shared := testutil.TempDir(t)
	remotes, _ := json.Marshal([]map[string]any{{"name": "shared", "kind": "fs", "config": map[string]string{"path": shared}}})
	t.Setenv("AITK_REMOTES", string(remotes))
	t.Setenv("AITK_S3_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
	t.Setenv("AITK_S3_BUCKET", "training")
	t.Setenv("AITK_S3_ACCESS_KEY_ID", "AK")
	t.Setenv("AITK_S3_SECRET_ACCESS_KEY", "SK")
	t.Setenv("AITK_SYNC_REMOTE", "shared")
	t.Setenv("AITK_AUTO_RESUME", "1")
	t.Setenv("AITK_MAX_AUTO_RESUMES", "5")

	if err := Apply(ctx); err != nil {
		t.Fatal(err)
	}
	r2, err := remote.Get(ctx, "s3")
	if err != nil || r2 == nil || r2.Kind != "s3" {
		t.Fatalf("s3 remote: %+v %v", r2, err)
	}
	var cfg map[string]any
	json.Unmarshal(r2.Config, &cfg)
	if cfg["region"] != "" || cfg["bucket"] != "training" {
		t.Fatalf("s3 config %v", cfg)
	}
	sharedRemote, _ := remote.Get(ctx, "shared")
	if v, _, _ := state.Setting(ctx, state.SettingSyncRemoteDefault); v != sharedRemote.ID {
		t.Fatalf("sync default %q, want %q", v, sharedRemote.ID)
	}
	if auto, max, _, _ := state.EffectivePolicy(ctx, "any"); !auto || max != 5 {
		t.Fatalf("policy defaults: auto=%v max=%d", auto, max)
	}

	// applying again (next boot, changed secret) updates in place
	t.Setenv("AITK_S3_SECRET_ACCESS_KEY", "SK2")
	if err := Apply(ctx); err != nil {
		t.Fatal(err)
	}
	all, _ := remote.List(ctx)
	if len(all) != 2 {
		t.Fatalf("remotes after second apply: %d", len(all))
	}
	again, _ := remote.Get(ctx, "s3")
	if again.ID != r2.ID || string(again.Secret) == string(r2.Secret) {
		t.Fatal("s3 should keep its id and get the new secret")
	}

	// a job without a policy gets the default; one with a policy keeps it
	state.ApplyDefaultPolicy(ctx, "job-a")
	if p, _ := state.GetPolicy(ctx, "job-a"); p.SyncRemote == nil || *p.SyncRemote != sharedRemote.ID {
		t.Fatalf("job-a policy %+v", p)
	}
	// other explicit fields do not block the default for an unset remote
	off := false
	state.SetPolicy(ctx, "job-b", state.Policy{AutoResume: &off})
	state.ApplyDefaultPolicy(ctx, "job-b")
	if p, _ := state.GetPolicy(ctx, "job-b"); p.SyncRemote == nil || *p.SyncRemote != sharedRemote.ID || *p.AutoResume {
		t.Fatalf("job-b policy %+v", p)
	}
	// an explicit "" opts out
	none := ""
	state.SetPolicy(ctx, "job-c", state.Policy{SyncRemote: &none})
	state.ApplyDefaultPolicy(ctx, "job-c")
	if p, _ := state.GetPolicy(ctx, "job-c"); p.SyncRemote == nil || *p.SyncRemote != "" {
		t.Fatalf("explicit opt-out was overridden: %+v", p)
	}

	if n := importPass(ctx, "shared", nil, false); n != 0 {
		t.Fatalf("empty remote should leave nothing pending, got %d", n)
	}
}

func TestApplyRejectsBadEnv(t *testing.T) {
	openState(t)
	t.Setenv("AITK_SYNC_REMOTE", "nope")
	if err := Apply(context.Background()); err == nil {
		t.Fatal("a sync remote that is not configured must fail at boot")
	}
	t.Setenv("AITK_SYNC_REMOTE", "")
	t.Setenv("AITK_REMOTES", "{not json")
	if err := Apply(context.Background()); err == nil {
		t.Fatal("malformed AITK_REMOTES must fail at boot")
	}
}
