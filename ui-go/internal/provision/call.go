package provision

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// realHosts are the production provider API hosts. They are refused under
// `go test` and while simulating, so a forgotten mock fails loudly instead of
// calling (and billing) a real account.
var realHosts = []string{"api.runpod.io", "rest.runpod.io", "console.vast.ai", "api.quickpod.org"}

type guardTransport struct{ next http.RoundTripper }

func (g guardTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	host := strings.ToLower(r.URL.Hostname())
	if slices.Contains(realHosts, host) && (testing.Testing() || Simulated.Load()) {
		return nil, fmt.Errorf("provision: refusing to call %s (tests and simulate mode use the mocks)", host)
	}
	return g.next.RoundTrip(r)
}

// HTTPClient is used by every adapter. It never follows redirects.
var HTTPClient = &http.Client{
	Timeout:       30 * time.Second,
	Transport:     guardTransport{next: http.DefaultTransport},
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}

type Call struct {
	Kind   Kind
	Method string
	URL    string
	Header map[string]string
	Body   any    // JSON-encoded; []byte is sent as is; nil = no body
	Max    int64  // response cap; default 1 MiB
	Secret string // redacted from any error text
}

type Response struct {
	Status int
	Header http.Header
	Body   []byte
}

// Do sends c and returns any HTTP response (no status interpretation: each
// adapter decodes its provider's error shape). Network and read failures
// return *Error{Class: ClassUnavailable}.
func Do(ctx context.Context, c Call) (Response, error) {
	var body io.Reader
	switch b := c.Body.(type) {
	case nil:
	case []byte:
		body = bytes.NewReader(b)
	default:
		buf, err := json.Marshal(b)
		if err != nil {
			return Response{}, err
		}
		body = bytes.NewReader(buf)
	}
	req, err := http.NewRequestWithContext(ctx, c.Method, c.URL, body)
	if err != nil {
		return Response{}, err
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range c.Header {
		req.Header.Set(k, v)
	}
	res, err := HTTPClient.Do(req)
	if err != nil {
		return Response{}, &Error{Provider: c.Kind, Class: ClassUnavailable, Message: Clip(Redact(err.Error(), c.Secret), 300)}
	}
	defer res.Body.Close()
	limit := c.Max
	if limit <= 0 {
		limit = 1 << 20
	}
	buf, err := io.ReadAll(io.LimitReader(res.Body, limit+1))
	if err != nil {
		return Response{}, &Error{Provider: c.Kind, Class: ClassUnavailable, Status: res.StatusCode, Message: "reading the response: " + Redact(err.Error(), c.Secret)}
	}
	if int64(len(buf)) > limit {
		return Response{}, &Error{Provider: c.Kind, Class: ClassUnavailable, Status: res.StatusCode, Message: "response too large"}
	}
	return Response{Status: res.StatusCode, Header: res.Header, Body: buf}, nil
}

// RetryAfter reads Retry-After (seconds), RunPod's RateLimit ("t=<s>") and
// Vast's X-RateLimit-Reset (epoch seconds), in that order.
func RetryAfter(h http.Header, now time.Time) time.Duration {
	if v := strings.TrimSpace(h.Get("Retry-After")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return time.Duration(n) * time.Second
		}
	}
	if v := h.Get("RateLimit"); v != "" {
		// `"minute";r=0;t=12, "hour";r=2800;t=1812`: the longest reset among
		// the exhausted (r=0) windows.
		best := time.Duration(0)
		for _, item := range strings.Split(v, ",") {
			params := map[string]string{}
			for _, p := range strings.Split(item, ";") {
				if k, val, ok := strings.Cut(strings.TrimSpace(p), "="); ok {
					params[k] = strings.TrimSpace(val)
				}
			}
			if params["r"] != "0" {
				continue
			}
			if n, err := strconv.Atoi(params["t"]); err == nil && n > 0 {
				best = max(best, time.Duration(n)*time.Second)
			}
		}
		if best > 0 {
			return best
		}
	}
	if v := h.Get("X-RateLimit-Reset"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			if d := time.Unix(n, 0).Sub(now); d > 0 && d < time.Hour {
				return d
			}
		}
	}
	return 0
}
