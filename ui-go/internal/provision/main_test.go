package provision

import (
	"os"
	"testing"
)

// Every launch needs a URL its machine can dial (StudioAPIURL); tests that
// check the refusal set their own.
func TestMain(m *testing.M) {
	os.Setenv("AITK_PUBLIC_URL", "https://studio.example")
	os.Exit(m.Run())
}
