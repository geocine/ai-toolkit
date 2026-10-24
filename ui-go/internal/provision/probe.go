package provision

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sync"

	"github.com/geocine/aitoolkit/internal/netguard"
)

// The probe asks a machine whether it is the aitk server we launched. It
// reads at most probeMax bytes of each answer and keeps only the version.
const (
	probeMax        = 64 << 10
	probeVersionMax = 64
)

func init() { DefaultProber = &prober{} }

// prober implements Prober over netguard clients: the strict guard for real
// machines, loopback only for simulated ones. It never follows redirects.
type prober struct {
	once             sync.Once
	strict, loopback *http.Client
}

func (p *prober) client(simulated bool) *http.Client {
	p.once.Do(func() {
		p.strict = netguard.NewClient(netguard.Options{})
		p.loopback = netguard.NewClient(netguard.Options{AllowLoopback: true})
	})
	if simulated {
		return p.loopback
	}
	return p.strict
}

// Probe tries each endpoint in order, after CheckEndpoint:
//   - GET /v1/health (expect 200);
//   - GET /v1/system with the machine token (200 ok, 401 rejected);
//   - GET /v1/system with no token (401 means auth is enforced).
//
// The first endpoint that accepts the token wins. A healthy endpoint that
// rejects it, or whose token check fails, is kept as a fallback while the
// rest are tried: a provider proxy may serve the public /v1/health and still
// drop Authorization (QuickPod's is UNCONFIRMED, quickpod.md §6.3), and the
// direct endpoint behind it then decides. With no token the first healthy
// endpoint wins. When none is healthy, the result carries the last error.
//
// CheckEndpoint runs with kind "": the service narrows the list to its
// provider's rules first (FilterEndpoints).
func (p *prober) Probe(ctx context.Context, endpoints []Endpoint, token string, simulated bool) ProbeResult {
	c := p.client(simulated)
	var last, fallback ProbeResult
	for _, ep := range endpoints {
		res := probeOne(ctx, c, ep, token, simulated)
		switch {
		case !res.Health:
			last = res
		case res.TokenOK || token == "":
			return res
		case !fallback.Health:
			fallback = res
		}
	}
	if fallback.Health {
		return fallback
	}
	return last
}

// probeOne runs the three checks against one endpoint.
func probeOne(ctx context.Context, c *http.Client, ep Endpoint, token string, simulated bool) ProbeResult {
	if err := CheckEndpoint("", ep.URL, simulated); err != nil {
		return ProbeResult{Err: err.Error()}
	}
	status, body, err := probeGet(ctx, c, ep.URL+"/v1/health", "")
	if err != nil {
		return ProbeResult{Err: err.Error()}
	}
	if status != http.StatusOK {
		return ProbeResult{Err: "GET /v1/health answered " + http.StatusText(status)}
	}
	res := ProbeResult{BaseURL: ep.URL, Health: true, Version: healthVersion(body)}
	if token != "" {
		status, _, err := probeGet(ctx, c, ep.URL+"/v1/system", token)
		switch {
		case err != nil:
			res.Err = err.Error()
		case status == http.StatusOK:
			res.TokenOK = true
		case status == http.StatusUnauthorized:
			res.TokenRejected = true
		default:
			res.Err = "GET /v1/system answered " + http.StatusText(status)
		}
	}
	status, _, err = probeGet(ctx, c, ep.URL+"/v1/system", "")
	if err == nil {
		switch status {
		case http.StatusUnauthorized:
			v := true
			res.AuthEnforced = &v
		case http.StatusOK:
			v := false
			res.AuthEnforced = &v
		}
	}
	return res
}

// probeGet sends one GET and reads at most probeMax bytes of the body.
func probeGet(ctx context.Context, c *http.Client, url, token string) (int, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Accept", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	res, err := c.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, probeMax))
	if err != nil {
		return res.StatusCode, nil, err
	}
	return res.StatusCode, body, nil
}

// healthVersion is the "version" of a /v1/health answer, clipped; "" when
// the body is not that shape (a 64 KiB cap may cut it, which reads as "").
func healthVersion(body []byte) string {
	var h struct {
		Version string `json:"version"`
	}
	if json.Unmarshal(body, &h) != nil {
		return ""
	}
	v := h.Version
	if len(v) > probeVersionMax {
		v = v[:probeVersionMax]
	}
	return v
}
