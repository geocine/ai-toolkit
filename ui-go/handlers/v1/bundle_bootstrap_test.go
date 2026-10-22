package v1

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/bundles"
	"github.com/geocine/aitoolkit/internal/store"
)

func TestBootstrap(t *testing.T) {
	e := setup(t)
	t.Setenv("AITK_BOOTSTRAP_PREFETCH", "0")
	id := bundleFixture(t, e)
	bundle := e.do(t, "GET", "/v1/jobs/"+id+"/bundle", nil, nil).body
	// a fresh server: the job and its folder are gone, the bundle is on disk
	want(t, e.call(t, "DELETE", "/v1/jobs/"+id+"?files=true", nil), 204, "")
	local := filepath.Join(e.root, "boot", "run.zip")
	put(t, local, bundle)
	ctx := context.Background()

	if err := bootstrapOne(ctx, local, true); err != nil {
		t.Fatal(err)
	}
	j := jobByName(t, "run")
	if j.Status != "queued" || j.Step != 200 {
		t.Fatalf("bootstrapped job: %s step %d", j.Status, j.Step)
	}
	tasks, _ := bundles.ListTasks(ctx, bundles.KindImport)
	if len(tasks) != 1 || tasks[0].Ref == nil || *tasks[0].Ref != local || tasks[0].State != bundles.TaskDone || tasks[0].Source != "local" {
		t.Fatalf("task: %+v", tasks)
	}
	// idempotent across restarts
	if err := bootstrapOne(ctx, local, true); err != nil {
		t.Fatal(err)
	}
	if n, _ := store.FindJobs(ctx, store.JobQuery{}); len(n) != 1 {
		t.Fatalf("%d jobs after a second boot", len(n))
	}

	// by URL: the manifest at the head of the zip shows the job is already
	// here before anything big is downloaded
	ranged := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "" {
			ranged++
		}
		http.ServeContent(w, r, "run.zip", time.Now(), bytes.NewReader(bundle))
	}))
	defer srv.Close()
	if err := bootstrapOne(ctx, srv.URL+"/run.zip", true); err != nil {
		t.Fatal(err)
	}
	if tasks, _ := bundles.ListTasks(ctx, bundles.KindImport); len(tasks) != 1 || ranged != 1 {
		t.Fatalf("url ref of a job already here: %d tasks, %d ranged GETs", len(tasks), ranged)
	}

	// refs that cannot work fail; a malformed one is not retried
	for _, ref := range []string{"nope:bundles/x.zip", "/no/such/file.zip", "garbage"} {
		if err := bootstrapOne(ctx, ref, false); err == nil || (ref == "garbage" && retryable(err)) {
			t.Errorf("%s: %v", ref, err)
		}
	}
}

func TestHubRepo(t *testing.T) {
	for base, want := range map[string]bool{
		"black-forest-labs/FLUX.1-dev": true, "stabilityai/sdxl": true, "/models/flux": false, `C:\models\x`: false,
		"a/b/c": false, "single": false, "org/model.safetensors": false, "": false, "./org/x": false,
	} {
		if hubRepo(base) != want {
			t.Errorf("hubRepo(%q) = %v", base, !want)
		}
	}
}
