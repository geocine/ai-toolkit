package imagecheck

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
)

func runMockCLI(t *testing.T, c *Client, args ...string) (int, string, string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code := runCLI(context.Background(), c, args, &out, &errOut)
	return code, out.String(), errOut.String()
}

func TestCLI(t *testing.T) {
	_, c, _ := newMock(t, registrymock.Faults{})

	code, out, stderr := runMockCLI(t, c, "--require-go", goRef)
	if code != 0 || !strings.Contains(out, "CUDA 13.0 (driver 580 or newer)") || !strings.Contains(out, "No problems found") {
		t.Fatalf("labelled Go image: exit %d\n%s%s", code, out, stderr)
	}

	// Upstream is not the Go server.
	if code, out, _ := runMockCLI(t, c, "--require-go", "ostris/aitoolkit:latest"); code != 1 || !strings.Contains(out, "image_not_go") {
		t.Fatalf("ostris --require-go: exit %d\n%s", code, out)
	}
	// A Go image without labels runs, but --require-go wants the sm label.
	if code, _, _ := runMockCLI(t, c, DefaultRepo+":oldsha1"); code != 0 {
		t.Fatalf("oldsha1: exit %d", code)
	}
	if code, out, _ := runMockCLI(t, c, "--require-go", DefaultRepo+":oldsha1"); code != 1 || !strings.Contains(out, "image_labels") {
		t.Fatalf("oldsha1 --require-go: exit %d\n%s", code, out)
	}
	if code, _, stderr := runMockCLI(t, c, DefaultRepo+":nope"); code != 1 || !strings.Contains(stderr, "There's no tag nope") {
		t.Fatalf("missing tag: exit %d\n%s", code, stderr)
	}

	// --json
	code, out, _ = runMockCLI(t, c, "--json", goRef)
	var res struct {
		OK     bool    `json:"ok"`
		Level  string  `json:"level"`
		Image  *Image  `json:"image"`
		Checks []Check `json:"checks"`
	}
	if err := json.Unmarshal([]byte(out), &res); err != nil || code != 0 || !res.OK || res.Level != "ok" || res.Image == nil || res.Image.Revision != registrymock.Revision {
		t.Fatalf("--json: exit %d %v\n%s", code, err, out)
	}

	// --help and usage errors.
	if code, _, stderr := runMockCLI(t, c, "--help"); code != 0 || !strings.Contains(stderr, "usage: aitk-server image-check") {
		t.Fatalf("--help: exit %d\n%s", code, stderr)
	}
	if code, _, _ := runMockCLI(t, c); code != 1 {
		t.Fatalf("no image: exit %d", code)
	}
}

// MASTER §3.4: the unpublished default image prints the settings URL and exits 2.
func TestCLIDeniedDefaultRepo(t *testing.T) {
	_, c, _ := newMock(t, registrymock.Faults{Private: map[string]bool{registrymock.DefaultRepo: true}})
	code, _, stderr := runMockCLI(t, c, "--require-go", DefaultRepo+":latest")
	if code != 2 {
		t.Fatalf("exit %d\n%s", code, stderr)
	}
	for _, want := range []string{
		"The server image ghcr.io/geocine/aitoolkit-go isn't published yet, or it's still private.",
		"https://github.com/users/geocine/packages/container/aitoolkit-go/settings",
	} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr lacks %q:\n%s", want, stderr)
		}
	}
	// Another denied repo exits 2 too, without the settings URL.
	code, _, stderr = runMockCLI(t, c, "ghcr.io/someone/else:1")
	if code != 2 || strings.Contains(stderr, "settings") {
		t.Fatalf("other denied repo: exit %d\n%s", code, stderr)
	}
}

func TestCLIHostJSON(t *testing.T) {
	_, c, _ := newMock(t, registrymock.Faults{})
	path := filepath.Join(t.TempDir(), "host.json")
	if err := os.WriteFile(path, []byte(`{"gpu_name":"Tesla V100","sm":70}`), 0o644); err != nil {
		t.Fatal(err)
	}
	prev := HostChecks
	t.Cleanup(func() { HostChecks = prev })

	HostChecks = nil
	if code, _, stderr := runMockCLI(t, c, "--host-json", path, goRef); code != 1 || !strings.Contains(stderr, "host checks") {
		t.Fatalf("no hook: exit %d\n%s", code, stderr)
	}
	HostChecks = func(img *Image, b []byte) ([]Check, error) {
		var h struct{ SM int }
		if err := json.Unmarshal(b, &h); err != nil {
			return nil, err
		}
		if !img.Requires.Runs(h.SM) {
			return []Check{{Level: LevelBlock, Code: "gpu_sm", Message: "This GPU isn't supported."}}, nil
		}
		return nil, nil
	}
	if code, out, _ := runMockCLI(t, c, "--host-json", path, goRef); code != 1 || !strings.Contains(out, "gpu_sm") {
		t.Fatalf("V100 host: exit %d\n%s", code, out)
	}
	HostChecks = func(*Image, []byte) ([]Check, error) { return nil, errors.New("bad host facts") }
	if code, _, stderr := runMockCLI(t, c, "--host-json", path, goRef); code != 1 || !strings.Contains(stderr, "bad host facts") {
		t.Fatalf("hook error: exit %d\n%s", code, stderr)
	}
}
