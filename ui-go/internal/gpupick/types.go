// Package gpupick suggests the default GPU offer: best value and best speed
// within the smallest VRAM tier the model fits, among offers that can run the
// image fast (internet, disk, driver, architecture). It only reads offers.
//
// gpupick imports provision, hostfit and imagecheck; none of them imports it,
// so every shared threshold lives in hostfit and is aliased here.
package gpupick

import (
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/provision"
)

// Ladder is the VRAM ladder (GB). 24, then 32, then 48, higher only if none.
var Ladder = []int{24, 32, 48, 80, 96, 141, 180}

// Thresholds. Each is a named constant so a later calibration changes one line.
// The bandwidth ones live in hostfit (provision and hostfit never import
// gpupick); their sources and UNCONFIRMED notes are there.
const (
	MinInetDownMbps = hostfit.FastInetDownMbps // 500
	MinDiskBWMBps   = hostfit.FastDiskBWMBps   // 300
	// MinReliability: ours, not a provider default. Vast reports 0..1 and QuickPod
	// 0..100 (normalised by the adapter). UNCONFIRMED tuning. Applied client-side only.
	MinReliability = 0.95
	// TierSlack: 24 GB-class cards report 22.49 GiB (L4 23034 MiB, A10 23028 MiB)
	// up to 23.99 GiB (24564 MiB). 24 x 0.9 = 21.6 keeps all of them in tier 24 and
	// keeps 20 GB cards (RTX 4000 Ada, A4500) out. 32 x 0.9 = 28.8, 48 x 0.9 = 43.2
	// (L40S 44.99 GiB), 80 x 0.9 = 72.
	TierSlack = 0.9
	// TierLimit is the offers page size per tier query (the adapters' maximum).
	TierLimit = 200
	// PerProviderTimeout bounds one provider's offers calls in a run.
	PerProviderTimeout = 20 * time.Second
)

type Query struct {
	Providers []provision.Kind // empty: every enabled provider with a key
	Model     *hostfit.ModelRef
	Image     string  // "" = provision.DefaultImage (the studio sends the Image row, or the QuickPod template's image)
	DiskGB    int     // 0 = Need.DiskGB
	Cloud     string  // RunPod: "secure" (default) | "community"
	MaxPrice  float64 // 0 = any
}

type Pick struct {
	Kind        string          `json:"kind"`         // "value" | "speed"
	Offer       provision.Offer `json:"offer"`        // as returned by providers.offers (host and fit filled)
	TierGB      int             `json:"tier_gb"`      // ladder tier the offer belongs to
	Speed       float64         `json:"speed"`        // RTX 4090 = 1.00
	SpeedSource string          `json:"speed_source"` // hostfit.SpeedSources key
	PerDollar   float64         `json:"per_dollar"`   // speed / price_per_hour
	Reason      string          `json:"reason"`       // one line: gpu · vram · price · net · cuda
}

type TierResult struct {
	TierGB    int            `json:"tier_gb"`
	Offers    int            `json:"offers"`
	Qualified int            `json:"qualified"`
	Rejected  map[string]int `json:"rejected"` // reject code -> count
}

type Note struct {
	Provider provision.Kind `json:"provider"` // "" for image and model notes
	// Code is "not_configured", a /v1 error code (rate_limited, provider_auth,
	// remote_failed, ...), "not_decorated", "model_unknown", or an imagecheck
	// kind (image_denied, image_not_found, image_unreachable).
	Code    string `json:"code"`
	Message string `json:"message"`
	Offers  int    `json:"offers"`
}

type Thresholds struct {
	MinInetDownMbps float64 `json:"min_inet_down_mbps"`
	MinDiskBWMBps   float64 `json:"min_disk_bw_mbps"`
	MinReliability  float64 `json:"min_reliability"`
	Ladder          []int   `json:"ladder"`
}

// Recommendation never carries nil slices or maps: Providers, Tiers, Notes,
// each Rejected and Thresholds.Ladder are initialised, so the JSON never has
// null where the TS declares an array or record.
type Recommendation struct {
	Providers   []provision.Kind  `json:"providers"` // searched
	Model       *hostfit.ModelRef `json:"model"`
	StartTierGB int               `json:"start_tier_gb"`
	TierGB      int               `json:"tier_gb"` // tier the picks come from; 0 = none
	Default     string            `json:"default"` // "value" | ""
	Value       *Pick             `json:"value"`
	Speed       *Pick             `json:"speed"`
	Need        hostfit.Need      `json:"need"`
	Tiers       []TierResult      `json:"tiers"` // each tier walked, in order
	Notes       []Note            `json:"notes"`
	Thresholds  Thresholds        `json:"thresholds"`
	Message     string            `json:"message"` // set when nothing qualifies, or for an image-side note
}

// DefaultThresholds is the thresholds a run applies, as the response reports them.
func DefaultThresholds() Thresholds {
	return Thresholds{MinInetDownMbps, MinDiskBWMBps, MinReliability, append([]int(nil), Ladder...)}
}
