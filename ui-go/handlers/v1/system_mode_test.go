package v1

import (
	"testing"

	"github.com/geocine/aitoolkit/internal/studioapi"
)

// health says whether this server trains or is the no-GPU studio API.
func TestHealthMode(t *testing.T) {
	e := setup(t)
	t.Cleanup(studioapi.SetForTest(false))
	if m := e.call(t, "GET", "/v1/health", nil).json(t); m["mode"] != "server" {
		t.Fatalf("default mode: %v", m)
	}
	studioapi.SetForTest(true)
	if m := e.call(t, "GET", "/v1/health", nil).json(t); m["mode"] != "studio-api" {
		t.Fatalf("studio API mode: %v", m)
	}
}
