package vast

import (
	"encoding/json"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/provision"
)

// instance is the part of a Vast instance object the adapter reads
// (spec: vast.md §5.1). ports and extra_env are decoded leniently: ports may
// be missing, null or {} while loading, and the show_instance schema
// disagrees with the live shape.
type instance struct {
	ID             int64           `json:"id"`
	ActualStatus   *string         `json:"actual_status"`
	IntendedStatus string          `json:"intended_status"`
	StatusMsg      string          `json:"status_msg"`
	Label          string          `json:"label"`
	PublicIPAddr   string          `json:"public_ipaddr"`
	Ports          json.RawMessage `json:"ports"`
	ExtraEnv       json.RawMessage `json:"extra_env"`
	GPUName        string          `json:"gpu_name"`
	NumGPUs        int             `json:"num_gpus"`
	GPURAM         float64         `json:"gpu_ram"`
	Geolocation    string          `json:"geolocation"`
	DPHTotal       *float64        `json:"dph_total"`
	Instance       struct {
		TotalHour *float64 `json:"totalHour"`
	} `json:"instance"`
	StartDate float64 `json:"start_date"`
}

// mapStatus follows PLAN §6.2 (actual_status / intended_status).
func mapStatus(actual *string, intended string) provision.Status {
	a := ""
	if actual != nil {
		a = strings.ToLower(*actual)
	}
	intended = strings.ToLower(intended)
	switch {
	case intended == "stopped" && a == "running":
		return provision.StatusStopping
	case intended == "stopped":
		return provision.StatusStopped
	case actual == nil:
		return provision.StatusCreating
	}
	switch a {
	case "created", "loading", "scheduling":
		return provision.StatusStarting
	case "running":
		return provision.StatusRunning
	case "exited", "unknown", "offline":
		if intended == "running" {
			return provision.StatusFailed
		}
	}
	return provision.StatusUnknown
}

// providerStatus is the raw "actual/intended" pair, "null" for a missing actual.
func providerStatus(actual *string, intended string) string {
	a := "null"
	if actual != nil {
		a = *actual
	}
	return a + "/" + intended
}

// hostPort reads ports["8675/tcp"][0].HostPort, a string in the live API
// ("41234"); a number is accepted too. "" when there is no mapping.
func hostPort(raw json.RawMessage) string {
	var ports map[string][]struct {
		HostPort json.RawMessage `json:"HostPort"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &ports) != nil {
		return ""
	}
	maps := ports[strconv.Itoa(provision.Port)+"/tcp"]
	if len(maps) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(maps[0].HostPort, &s) != nil {
		var n json.Number
		if json.Unmarshal(maps[0].HostPort, &n) != nil {
			return ""
		}
		s = n.String()
	}
	if p, err := strconv.Atoi(strings.TrimSpace(s)); err != nil || p <= 0 || p > 65535 {
		return ""
	}
	return strings.TrimSpace(s)
}

// extraEnv turns extra_env ([[key, value], ...]) into a map, without the
// docker flags ("-p 8675:8675") that travel in the same list.
func extraEnv(raw json.RawMessage) map[string]string {
	var pairs [][]any
	if len(raw) == 0 || json.Unmarshal(raw, &pairs) != nil {
		return nil
	}
	out := map[string]string{}
	for _, p := range pairs {
		if len(p) != 2 {
			continue
		}
		k, ok := p[0].(string)
		if !ok || k == "" || strings.HasPrefix(k, "-") {
			continue
		}
		switch v := p[1].(type) {
		case string:
			out[k] = v
		case nil:
			out[k] = ""
		default:
			b, _ := json.Marshal(v)
			out[k] = string(b)
		}
	}
	return out
}

func (in instance) toInstance() provision.Instance {
	st := mapStatus(in.ActualStatus, in.IntendedStatus)
	out := provision.Instance{
		ProviderID:          strconv.FormatInt(in.ID, 10),
		Label:               in.Label,
		Status:              st,
		ProviderStatus:      providerStatus(in.ActualStatus, in.IntendedStatus),
		GPU:                 in.GPUName,
		GPUCount:            in.NumGPUs,
		VRAMGB:              vramGB(in.GPURAM),
		Location:            in.Geolocation,
		PricePerHour:        in.DPHTotal,
		StoppedPricePerHour: in.Instance.TotalHour,
		Env:                 extraEnv(in.ExtraEnv),
	}
	if in.StartDate > 0 {
		sec := int64(in.StartDate)
		out.CreatedAt = time.Unix(sec, int64((in.StartDate-float64(sec))*1e9)).UTC()
	}
	if st == provision.StatusFailed {
		out.Detail = in.StatusMsg
	}
	if port := hostPort(in.Ports); port != "" && net.ParseIP(in.PublicIPAddr) != nil {
		out.Endpoints = []provision.Endpoint{{URL: "http://" + net.JoinHostPort(in.PublicIPAddr, port), Secure: false, Via: "direct"}}
	}
	out.PortMissing = st == provision.StatusRunning && len(out.Endpoints) == 0
	switch st {
	case provision.StatusCreating, provision.StatusStarting, provision.StatusRunning:
		out.Actions = []string{"stop", "terminate"}
	case provision.StatusStopped:
		out.Actions = []string{"start", "terminate"}
	default:
		out.Actions = []string{"terminate"}
	}
	return out
}
