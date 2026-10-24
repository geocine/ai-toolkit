package hostfit

import (
	"fmt"
	"math"

	"github.com/geocine/aitoolkit/internal/imagecheck"
)

// Pull model constants (IMAGE-PLAN §3.6), all UNCONFIRMED until tuned from
// observed launches: the share of the reported bandwidth a pull gets, the
// share of the disk bandwidth unpacking gets, and the single-thread gunzip
// cap.
const (
	pullEta        = 0.5
	unpackKappa    = 0.5
	gunzipCapMBps  = 150.0
	slowPullMin    = 10.0 // a longer image pull is a warn (pull_slow)
	fastPullSample = FastInetDownMbps
)

// Pull is how long a host takes to fetch the image (and the model), from
// its reported bandwidth. The /v1 PullEstimate.
type Pull struct {
	ImageMin float64  `json:"image_min"`
	ModelMin *float64 `json:"model_min"` // null without a model
	Basis    string   `json:"basis"`     // "1376 Mbps, 2842 MB/s disk"
	Estimate bool     `json:"estimate"`  // always true
}

// PullFor estimates the image pull (download, then unpack) and the model
// download on host h. nil when the host reports no bandwidth (RunPod, or a
// 0 inet_down). An image that was not read uses today's measured sizes.
func PullFor(img *imagecheck.Image, h Host, modelGB float64) *Pull {
	if h.InetDownMbps <= 0 {
		return nil
	}
	p := &Pull{ImageMin: round1(imageSeconds(img, h.InetDownMbps, h.DiskBWMBps) / 60), Estimate: true,
		Basis: fmt.Sprintf("%.0f Mbps", h.InetDownMbps)}
	if h.DiskBWMBps > 0 {
		p.Basis += fmt.Sprintf(", %.0f MB/s disk", h.DiskBWMBps)
	}
	if modelGB > 0 {
		m := round1(modelGB * 1e9 * 8 / (h.InetDownMbps * 1e6 * pullEta) / 60)
		p.ModelMin = &m
	}
	return p
}

// imageSeconds is download bytes over the effective link plus unpacked
// bytes over min(disk x kappa, gunzip cap); no disk bandwidth uses the cap.
func imageSeconds(img *imagecheck.Image, mbps, diskMBps float64) float64 {
	download, unpacked := float64(defaultDownloadBytes), float64(defaultUnpackedBytes)
	if img != nil && img.DownloadBytes > 0 {
		download = float64(img.DownloadBytes)
		unpacked = float64(img.UnpackedBytes)
		if unpacked <= 0 {
			unpacked = download * 2.1
		}
	}
	extract := gunzipCapMBps
	if diskMBps > 0 {
		extract = math.Min(diskMBps*unpackKappa, gunzipCapMBps)
	}
	return download*8/(mbps*1e6*pullEta) + unpacked/(extract*1e6)
}

// pullChecks: an image pull over 10 minutes is a warn (pull_slow, with the
// fast-hosts filter as the action), anything shorter an info (pull_eta).
func pullChecks(img *imagecheck.Image, p *Pull, model *ModelRef) []imagecheck.Check {
	if p == nil {
		return nil
	}
	gb := float64(defaultDownloadBytes) / 1e9
	if img != nil && img.DownloadBytes > 0 {
		gb = float64(img.DownloadBytes) / 1e9
	}
	if p.ImageMin > slowPullMin {
		fast := imageSeconds(img, fastPullSample, 0) / 60
		return []imagecheck.Check{{Level: imagecheck.LevelWarn, Code: "pull_slow",
			Message: fmt.Sprintf("Expect about %s to download the %.1f GB image on this host (%s).", minutes(p.ImageMin), gb, p.Basis),
			Fix:     fmt.Sprintf("Hosts above %.0f Mbps take about %s.", fastPullSample, minutes(fast)),
			Action:  &imagecheck.CheckAction{Kind: "filter_hosts", Value: "fast"}}}
	}
	msg := fmt.Sprintf("About %s to download the image", minutes(p.ImageMin))
	if p.ModelMin != nil && model != nil {
		msg += fmt.Sprintf(", then about %s for %s", minutes(*p.ModelMin), ArchName(model.Arch))
	}
	return []imagecheck.Check{{Level: imagecheck.LevelInfo, Code: "pull_eta", Message: msg + ".",
		Fix: "Estimate from the host's reported bandwidth."}}
}

// minutes is "1 minute", "3 minutes", "25 minutes" (whole minutes, at least 1).
func minutes(m float64) string {
	n := max(int(math.Round(m)), 1)
	if n == 1 {
		return "1 minute"
	}
	return fmt.Sprintf("%d minutes", n)
}

// mins is the short form for badges: "3 min".
func mins(m float64) string { return fmt.Sprintf("%d min", max(int(math.Round(m)), 1)) }
