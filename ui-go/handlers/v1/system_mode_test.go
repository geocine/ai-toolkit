package v1

import (
	"testing"

	"github.com/geocine/aitoolkit/internal/controller"
)

// health says whether this server trains or is a no-GPU controller.
func TestHealthMode(t *testing.T) {
	e := setup(t)
	t.Cleanup(controller.SetForTest(false))
	if m := e.call(t, "GET", "/v1/health", nil).json(t); m["mode"] != "server" {
		t.Fatalf("default mode: %v", m)
	}
	controller.SetForTest(true)
	if m := e.call(t, "GET", "/v1/health", nil).json(t); m["mode"] != "controller" {
		t.Fatalf("controller mode: %v", m)
	}
}
