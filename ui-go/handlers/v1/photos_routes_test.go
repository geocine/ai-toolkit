package v1

import (
	"net/http"
	"slices"
	"testing"

	"github.com/geocine/aitoolkit/internal/bg"
)

// The photo routes answer through the mux Register builds, and registering
// them registers the photo-cache GC loop.
func TestPhotoRoutes(t *testing.T) {
	e := setup(t)
	photoSet(t, e, "me", map[string][]byte{"a.jpg": photoScene(1), "b.jpg": photoScene(2)})

	if !slices.Contains(bg.Names(), "photocheck-gc") {
		t.Error("registering the photo routes did not register the photocheck-gc loop")
	}

	r := e.call(t, "POST", "/v1/photos/assess", map[string]any{"items": []any{}})
	want(t, r, 200, "")
	if _, ok := r.json(t)["convert"].([]any); !ok {
		t.Fatalf("assess with no items lacks convert: %s", r.body)
	}

	rv := photoReview(t, e.call(t, "GET", "/v1/datasets/me/photos/check?wait_ms=20000&resolution=512,1024", nil))
	if rv.Total != 2 || rv.Pending != 0 {
		t.Fatalf("check: total %d pending %d", rv.Total, rv.Pending)
	}
	want(t, e.call(t, "GET", "/v1/datasets/missing/photos/check", nil), 404, "not_found")

	r = e.call(t, "POST", "/v1/datasets/me/photos/fix", map[string]any{"ops": []any{map[string]any{"op": "set_aside", "files": []string{"a.jpg"}}}})
	want(t, r, 200, "")
	if m := r.json(t); m["applied"] != false || len(m["changes"].([]any)) != 1 {
		t.Fatalf("fix preview: %s", r.body)
	}
	want(t, e.call(t, "POST", "/v1/datasets/me/photos/fix", map[string]any{"ops": []any{map[string]any{"op": "nope"}}}), 400, "invalid_request")

	rv = photoReview(t, e.call(t, "PUT", "/v1/datasets/me/photos/settings", map[string]any{"kind": "style"}))
	if rv.Kind != "style" {
		t.Fatalf("settings: kind %q", rv.Kind)
	}

	r = e.call(t, "DELETE", "/v1/datasets/me/photos/analysis", nil)
	want(t, r, http.StatusNoContent, "")
	if rv := photoReview(t, e.call(t, "GET", "/v1/datasets/me/photos/check?wait_ms=20000", nil)); rv.Kind != "person" {
		t.Fatalf("forget kept the kind %q", rv.Kind)
	}
}
