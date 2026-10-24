package v1images

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"sync"
	"testing"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
)

// TestMain is the network tripwire (IMAGE-PLAN §7.4): a request through the
// default transport to a host that is not loopback fails, and so does the
// run. Loopback passes, for the httptest mocks and the simulated provider.
// The image check and the model sizer start on clients that only reach the
// tripwire, so a test that forgets imagetest.Use cannot make Register set up
// the real ones (ghcr.io, Hugging Face).
func TestMain(m *testing.M) {
	http.DefaultTransport = tripwire{next: http.DefaultTransport}
	offline := &http.Client{Transport: tripwire{}}
	imagecheck.Default = imagecheck.New(imagecheck.Options{HTTP: offline, Logf: func(string, ...any) {}})
	hostfit.DefaultSizer = hostfit.NewSizer(hostfit.SizerOptions{HTTP: offline})
	code := m.Run()
	if hosts := reached(); len(hosts) > 0 {
		fmt.Fprintf(os.Stderr, "v1images tests reached the network: %v\n", hosts)
		code = 1
	}
	os.Exit(code)
}

var tripped struct {
	mu    sync.Mutex
	hosts []string
}

func reached() []string {
	tripped.mu.Lock()
	defer tripped.mu.Unlock()
	return tripped.hosts
}

// tripwire passes loopback requests to next and refuses every other one.
// With no next (the offline clients) it refuses everything.
type tripwire struct{ next http.RoundTripper }

func (w tripwire) RoundTrip(r *http.Request) (*http.Response, error) {
	if h := r.URL.Hostname(); w.next != nil && (h == "127.0.0.1" || h == "::1" || h == "localhost") {
		return w.next.RoundTrip(r)
	}
	tripped.mu.Lock()
	tripped.hosts = append(tripped.hosts, r.URL.Host)
	tripped.mu.Unlock()
	return nil, errors.New("tripwire: v1images test reached " + r.URL.Host)
}
