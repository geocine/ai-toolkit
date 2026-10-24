package provision

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestDoSendsJSONAndReturnsAnyStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Accept") != "application/json" || r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("headers %v", r.Header)
		}
		buf := make([]byte, 64)
		n, _ := r.Body.Read(buf)
		if got := string(buf[:n]); got != `{"a":1}` {
			t.Errorf("body %q", got)
		}
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"error":"bad"}`))
	}))
	defer srv.Close()
	res, err := Do(context.Background(), Call{Kind: RunPod, Method: "POST", URL: srv.URL, Body: map[string]int{"a": 1},
		Header: map[string]string{"Authorization": "Bearer k"}})
	if err != nil || res.Status != 422 || string(res.Body) != `{"error":"bad"}` {
		t.Fatalf("Do: %+v %v", res, err)
	}
}

func TestDoDoesNotFollowRedirects(t *testing.T) {
	var followed atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/elsewhere" {
			followed.Store(true)
			return
		}
		http.Redirect(w, r, "/elsewhere", http.StatusFound)
	}))
	defer srv.Close()
	res, err := Do(context.Background(), Call{Kind: Vast, Method: "GET", URL: srv.URL + "/start"})
	if err != nil || res.Status != http.StatusFound || followed.Load() {
		t.Fatalf("redirect: %d %v followed=%v", res.Status, err, followed.Load())
	}
}

func TestDoCapsTheResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("x", 2048)))
	}))
	defer srv.Close()
	_, err := Do(context.Background(), Call{Kind: Vast, Method: "GET", URL: srv.URL, Max: 1024})
	if ClassOf(err) != ClassUnavailable || !strings.Contains(err.Error(), "response too large") {
		t.Fatalf("cap: %v", err)
	}
	res, err := Do(context.Background(), Call{Kind: Vast, Method: "GET", URL: srv.URL, Max: 2048})
	if err != nil || len(res.Body) != 2048 {
		t.Fatalf("at the cap: %d %v", len(res.Body), err)
	}
}

func TestDoNetworkErrorRedactsTheSecret(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL + "/v2/pods?api_key=topsecretkey"
	srv.Close() // nothing listens any more
	_, err := Do(context.Background(), Call{Kind: QuickPod, Method: "GET", URL: url, Secret: "topsecretkey"})
	var e *Error
	if ClassOf(err) != ClassUnavailable || strings.Contains(err.Error(), "topsecretkey") {
		t.Fatalf("network error: %v", err)
	}
	if !errors.As(err, &e) || e.Provider != QuickPod || e.Status != 0 {
		t.Fatalf("network error shape: %+v", e)
	}
}

// The real hosts are refused under go test before anything is dialed.
func TestGuardRefusesRealHosts(t *testing.T) {
	dialed := false
	g := guardTransport{next: roundTripFunc(func(*http.Request) (*http.Response, error) {
		dialed = true
		return nil, nil
	})}
	for _, u := range []string{"https://api.runpod.io/x", "https://REST.runpod.io/v1/pods", "https://console.vast.ai/api/v0/bundles/", "https://api.quickpod.org/api/rentable"} {
		req, _ := http.NewRequest("GET", u, nil)
		if _, err := g.RoundTrip(req); err == nil || !strings.Contains(err.Error(), "refusing to call") {
			t.Errorf("%s: %v", u, err)
		}
	}
	if dialed {
		t.Fatal("the guard must refuse before the next transport runs")
	}
	_, err := Do(context.Background(), Call{Kind: RunPod, Method: "GET", URL: "https://api.runpod.io/x"})
	if ClassOf(err) != ClassUnavailable || !strings.Contains(err.Error(), "refusing to call api.runpod.io") {
		t.Fatalf("Do to a real host: %v", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestRetryAfter(t *testing.T) {
	now := time.Unix(1_792_837_500, 0)
	h := func(kv ...string) http.Header {
		out := http.Header{}
		for i := 0; i < len(kv); i += 2 {
			out.Set(kv[i], kv[i+1])
		}
		return out
	}
	cases := []struct {
		name string
		h    http.Header
		want time.Duration
	}{
		{"retry-after", h("Retry-After", "7"), 7 * time.Second},
		{"retry-after wins", h("Retry-After", "3", "X-RateLimit-Reset", "1792837528"), 3 * time.Second},
		{"runpod exhausted windows", h("RateLimit", `"minute";r=0;t=12, "hour";r=2800;t=1812, "day";r=0;t=45`), 45 * time.Second},
		{"runpod nothing exhausted", h("RateLimit", `"minute";r=10;t=12, "hour";r=2800;t=1812`), 0},
		{"vast reset epoch", h("X-RateLimit-Reset", "1792837528"), 28 * time.Second},
		{"vast reset in the past", h("X-RateLimit-Reset", "1792837400"), 0},
		{"vast reset too far", h("X-RateLimit-Reset", "1792847500"), 0},
		{"junk", h("Retry-After", "soon"), 0},
		{"none", http.Header{}, 0},
	}
	for _, c := range cases {
		if got := RetryAfter(c.h, now); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}
