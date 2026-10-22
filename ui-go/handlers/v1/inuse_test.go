package v1

import (
	"path/filepath"
	"testing"

	"github.com/geocine/aitoolkit/internal/store"
)

// A dataset a queued or running job reads is not deleted under it.
func TestDatasetInUseIsKept(t *testing.T) {
	e := setup(t)
	want(t, e.call(t, "POST", "/v1/datasets", map[string]any{"name": "faces"}), 201, "")
	cfg := map[string]any{"job": "extension", "config": map[string]any{"name": "run", "process": []any{map[string]any{
		"type": "diffusion_trainer", "datasets": []any{map[string]any{"folder_path": filepath.Join(e.datasets, "faces")}}}}}}
	r := e.call(t, "POST", "/v1/jobs", map[string]any{"name": "run", "config": cfg})
	want(t, r, 201, "")
	id := r.json(t)["id"].(string)
	if _, err := store.UpdateJob(t.Context(), id, store.Fields{"status": "running"}); err != nil {
		t.Fatal(err)
	}
	want(t, e.call(t, "DELETE", "/v1/datasets/faces", nil), 409, "conflict")
	store.UpdateJob(t.Context(), id, store.Fields{"status": "stopped"})
	want(t, e.call(t, "DELETE", "/v1/datasets/faces", nil), 204, "")
}
