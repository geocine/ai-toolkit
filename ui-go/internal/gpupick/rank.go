package gpupick

import (
	"fmt"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
)

// rank.go is pure: no I/O. Recommend (recommend.go) feeds it offers.

const (
	// eps absorbs float noise in tier bounds and ranking ties.
	eps = 1e-9
	// valueBand: when two offers' per-dollar scores are within 2 %, one that
	// reports its internet and disk speed ranks ahead of one that does not
	// (D-15).
	valueBand = 0.02
)

// providerOrder breaks the last ranking ties: runpod, vast, quickpod.
var providerOrder = map[provision.Kind]int{provision.RunPod: 0, provision.Vast: 1, provision.QuickPod: 2}

// Reject codes, in the order Qualify checks them. The first failure is the
// offer's code.
const (
	RejectGPUCount     = "gpu_count"
	RejectPrice        = "price"
	RejectCUDA         = "cuda"
	RejectCUDAUnknown  = "cuda_unknown"
	RejectCUDACapacity = "cuda_capacity"
	RejectGPUArch      = "gpu_arch"
	RejectDisk         = "disk"
	RejectInet         = "inet"
	RejectDiskBW       = "disk_bw"
	RejectReliability  = "reliability"
	RejectFitBlock     = "fit_block"
	RejectSpeedUnknown = "speed_unknown"
)

var rejectOrder = []string{RejectGPUCount, RejectPrice, RejectCUDA, RejectCUDAUnknown, RejectCUDACapacity, RejectGPUArch,
	RejectDisk, RejectInet, RejectDiskBW, RejectReliability, RejectFitBlock, RejectSpeedUnknown}

// RejectLabels are the words the "nothing qualifies" message uses.
var RejectLabels = map[string]string{
	RejectInet:         "slow internet",
	RejectDiskBW:       "slow disk",
	RejectDisk:         "too little disk",
	RejectCUDA:         "an old driver",
	RejectCUDAUnknown:  "an unknown driver",
	RejectGPUArch:      "an unsupported GPU",
	RejectReliability:  "low reliability",
	RejectPrice:        "price above your limit",
	RejectCUDACapacity: "no CUDA 13 hosts free",
	RejectSpeedUnknown: "an unrated GPU",
	RejectGPUCount:     "several GPUs",
	RejectFitBlock:     "not runnable",
}

// archCodes are the fit codes of a GPU the image has no kernels for, or
// that the policy hides (sm_75, D-8), or that the table does not know.
var archCodes = []string{"gpu_sm", "host_arch", "gpu_vendor", "gpu_sm_flash_attn", "gpu_unknown"}

// Candidate is one offer with its provider's capabilities.
type Candidate struct {
	Offer provision.Offer
	Caps  provision.Caps
}

// StartTier is the first ladder tier the walk tries for model m: the
// smallest VRAM table size where m fits or is tight (a tight/no cell counts
// as no), rounded up to the ladder and never below its first tier. A model
// that fits nowhere up to 80 GB starts at 80. No model, or an arch the VRAM
// table does not know, starts at the bottom.
func StartTier(m *hostfit.ModelRef) int {
	if m == nil || !hostfit.KnownArch(m.Arch) {
		return Ladder[0]
	}
	need := hostfit.TierSizes[len(hostfit.TierSizes)-1]
	for _, t := range hostfit.Tiers(*m) {
		if t.Verdict == hostfit.VRAMFits || t.Verdict == hostfit.VRAMTight {
			need = t.VRAMGB
			break
		}
	}
	for _, l := range Ladder {
		if float64(l) >= need {
			return l
		}
	}
	return Ladder[len(Ladder)-1]
}

// TierOf is the largest ladder value L with vramGB >= L x TierSlack, or 0
// below the ladder. Callers pass host.vram_gb (MiB/1024) when present, else
// the offer's integer vram_gb (offerVRAM).
func TierOf(vramGB float64) int {
	tier := 0
	for _, l := range Ladder {
		if vramGB+eps >= float64(l)*TierSlack {
			tier = l
		}
	}
	return tier
}

// offerVRAM is the per-GPU VRAM TierOf classifies an offer by.
func offerVRAM(o provision.Offer) float64 {
	if o.Host != nil && o.Host.VRAMGB > 0 {
		return o.Host.VRAMGB
	}
	return float64(o.VRAMGB)
}

// SpeedOf is the offer's relative training speed (RTX 4090 = 1.00): the
// provider's raw GPU name first (adapters rewrite display names), then the
// display name. ok is false for a GPU the table does not know or does not
// rate, which is never picked.
func SpeedOf(o provision.Offer) (speed float64, source string, ok bool) {
	if o.Host != nil && o.Host.GPUName != "" {
		if s, src, known := hostfit.Speed(o.Host.GPUName); known && s > 0 {
			return s, src, true
		}
	}
	if s, src, known := hostfit.Speed(o.GPU); known && s > 0 {
		return s, src, true
	}
	return 0, "", false
}

// diskAsk is the disk a pick is judged at: the query's, or what the image
// and model need when that is more.
func diskAsk(need hostfit.Need, q Query) int { return max(q.DiskGB, need.DiskGB) }

// inetReported / diskBWReported: a 0 or missing value, or a provider whose
// offers never carry one (RunPod), is "not reported" and passes the
// fast-host rules (D-15).
func inetReported(o provision.Offer, caps provision.Caps) bool {
	return caps.ReportsInet && o.Host != nil && o.Host.InetDownMbps > 0
}

func diskBWReported(o provision.Offer, caps provision.Caps) bool {
	return caps.ReportsDiskBW && o.Host != nil && o.Host.DiskBWMBps > 0
}

// verified reports an offer that says how fast its internet and disk are.
func verified(o provision.Offer, caps provision.Caps) bool {
	return inetReported(o, caps) && diskBWReported(o, caps)
}

// Qualify is the reject code of the first rule o fails, in rejectOrder, or
// "" when it qualifies. An offer without a fit (not decorated) fails
// fit_block. RunPod passes the CUDA rule through host_cuda_placement (an
// info: RunPod enforces minCudaVersion itself).
func Qualify(o provision.Offer, caps provision.Caps, need hostfit.Need, q Query, th Thresholds) (code string) {
	if o.GPUCount != 1 {
		return RejectGPUCount
	}
	if q.MaxPrice > 0 && o.PricePerHour > q.MaxPrice+eps {
		return RejectPrice
	}
	var codes []string
	if o.Fit != nil {
		codes = o.Fit.Codes
	}
	has := func(cs ...string) bool {
		return slices.ContainsFunc(codes, func(c string) bool { return slices.Contains(cs, c) })
	}
	switch {
	case has("host_cuda"):
		return RejectCUDA
	case has("host_cuda_unknown"):
		return RejectCUDAUnknown
	case has("runpod_no_cuda_capacity"):
		return RejectCUDACapacity
	case has(archCodes...):
		return RejectGPUArch
	case has("disk_image") || (o.MaxDiskGB != nil && *o.MaxDiskGB < diskAsk(need, q)):
		return RejectDisk
	case inetReported(o, caps) && o.Host.InetDownMbps < th.MinInetDownMbps:
		return RejectInet
	case diskBWReported(o, caps) && o.Host.DiskBWMBps < th.MinDiskBWMBps:
		return RejectDiskBW
	case o.Reliability != nil && *o.Reliability < th.MinReliability:
		return RejectReliability
	case o.Fit == nil || o.Fit.Level == imagecheck.LevelBlock:
		return RejectFitBlock
	}
	if _, _, ok := SpeedOf(o); !ok {
		return RejectSpeedUnknown
	}
	return ""
}

// scored is a qualifying offer with its pick numbers.
type scored struct {
	Pick
	caps provision.Caps
}

// Rank walks the ladder from start over offers (any tiers, any order): for
// each tier it qualifies the offers of that tier and stops at the first
// tier with a qualifying offer. value is the most speed per $/h (a verified
// offer first within 2 %), speed the fastest. tiers has one entry per tier
// walked. The picks' Reason is the caller's (ReasonLine); tier is 0 and
// both picks nil when nothing qualifies.
func Rank(offers []Candidate, start int, need hostfit.Need, q Query, th Thresholds) (value, speed *Pick, tier int, tiers []TierResult) {
	tiers = []TierResult{}
	byTier := map[int][]Candidate{}
	for _, c := range offers {
		t := TierOf(offerVRAM(c.Offer))
		byTier[t] = append(byTier[t], c)
	}
	for _, l := range th.Ladder {
		if l < start {
			continue
		}
		tr, v, s := rankTier(byTier[l], l, need, q, th)
		tiers = append(tiers, tr)
		if v != nil {
			return v, s, l, tiers
		}
	}
	return nil, nil, 0, tiers
}

// rankTier qualifies and ranks the candidates of one tier (the caller has
// classified them).
func rankTier(cands []Candidate, tier int, need hostfit.Need, q Query, th Thresholds) (tr TierResult, value, speed *Pick) {
	tr = TierResult{TierGB: tier, Rejected: map[string]int{}}
	var ok []scored
	for _, c := range cands {
		tr.Offers++
		if code := Qualify(c.Offer, c.Caps, need, q, th); code != "" {
			tr.Rejected[code]++
			continue
		}
		tr.Qualified++
		sp, src, _ := SpeedOf(c.Offer)
		pd := 0.0
		if c.Offer.PricePerHour > 0 {
			pd = sp / c.Offer.PricePerHour
		}
		ok = append(ok, scored{Pick: Pick{Offer: c.Offer, TierGB: tier, Speed: sp, SpeedSource: src, PerDollar: pd}, caps: c.Caps})
	}
	if len(ok) == 0 {
		return tr, nil, nil
	}
	v, s := bestValue(ok), bestSpeed(ok)
	v.Kind, s.Kind = "value", "speed"
	return tr, &v, &s
}

// bestValue is the highest per_dollar. When the top offer does not report
// its bandwidth, the best one that does wins if it is within valueBand.
func bestValue(ok []scored) Pick {
	s := slices.Clone(ok)
	slices.SortFunc(s, func(a, b scored) int {
		if d := b.PerDollar - a.PerDollar; d > eps || d < -eps {
			return sign(d)
		}
		return tieBreak(a.Offer, b.Offer)
	})
	top := s[0]
	if !verified(top.Offer, top.caps) {
		for _, c := range s[1:] {
			if c.PerDollar < top.PerDollar*(1-valueBand)-eps {
				break
			}
			if verified(c.Offer, c.caps) {
				return c.Pick
			}
		}
	}
	return top.Pick
}

// bestSpeed is the highest speed.
func bestSpeed(ok []scored) Pick {
	return slices.MinFunc(ok, func(a, b scored) int {
		if d := b.Speed - a.Speed; d > eps || d < -eps {
			return sign(d)
		}
		return tieBreak(a.Offer, b.Offer)
	}).Pick
}

// tieBreak orders equal scores: lower price, then higher reliability (none
// lowest), then provider order, then offer id.
func tieBreak(a, b provision.Offer) int {
	if d := a.PricePerHour - b.PricePerHour; d > eps || d < -eps {
		return sign(d)
	}
	ra, rb := -1.0, -1.0
	if a.Reliability != nil {
		ra = *a.Reliability
	}
	if b.Reliability != nil {
		rb = *b.Reliability
	}
	if d := rb - ra; d > eps || d < -eps {
		return sign(d)
	}
	if d := providerOrder[a.Provider] - providerOrder[b.Provider]; d != 0 {
		return d
	}
	return strings.Compare(a.ID, b.ID)
}

func sign(d float64) int {
	if d < 0 {
		return -1
	}
	return 1
}

// ReasonLine is the one line that explains a pick:
// "{gpu} · {vram} GB · ${price}/h · {net} · {cuda}". In a cross-provider
// result " on {Label}" follows the GPU. net is the reported download speed,
// else "net not reported"; an offer that reports internet but not disk
// speed adds "disk speed not reported". cuda is the host's CUDA when known,
// else RunPod's own placement floor (cudaMin, the image's).
func ReasonLine(p Pick, crossProvider bool, cudaMin string) string {
	o := p.Offer
	var h hostfit.Host
	if o.Host != nil {
		h = *o.Host
	}
	gpu := o.GPU
	if crossProvider {
		gpu += " on " + provision.Label(o.Provider)
	}
	parts := []string{gpu, fmt.Sprintf("%d GB", o.VRAMGB), fmt.Sprintf("$%.2f/h", o.PricePerHour)}
	if h.InetDownMbps > 0 {
		parts = append(parts, fmt.Sprintf("%.0f Mbps", h.InetDownMbps))
		if h.DiskBWMBps <= 0 {
			parts = append(parts, "disk speed not reported")
		}
	} else {
		parts = append(parts, "net not reported")
	}
	switch {
	case h.CUDAMax != "":
		parts = append(parts, "CUDA "+h.CUDAMax+" OK")
	case cudaMin != "" && (o.Provider == provision.RunPod || h.PlacementFloor):
		parts = append(parts, "CUDA "+cudaMin+"+ by RunPod")
	}
	return strings.Join(parts, " · ")
}
