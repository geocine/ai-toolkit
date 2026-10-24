package gpupick

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
)

// capsOf are the bandwidth caps the adapters' Info() reports.
func capsOf(k provision.Kind) provision.Caps {
	reports := k != provision.RunPod
	return provision.Caps{ReportsInet: reports, ReportsDiskBW: reports, ImageDigestRef: "unconfirmed"}
}

// mk is a decorated single-GPU offer that qualifies: fast internet and
// disk, a CUDA 13.2 driver, 500 GB of disk. RunPod offers report no
// bandwidth and rely on RunPod's own CUDA placement, like the adapter's.
func mk(k provision.Kind, id, gpu string, vram, price float64) provision.Offer {
	h := hostfit.Host{Provider: string(k), GPUName: gpu, GPUCount: 1, VRAMGB: vram, GPUVendor: "nvidia", CPUArch: "amd64",
		CUDAMax: "13.2", DiskGB: 500, InetDownMbps: 900, DiskBWMBps: 2000}
	rel := 0.99
	o := provision.Offer{ID: id, Provider: k, GPU: gpu, GPUCount: 1, VRAMGB: int(math.Round(vram)), PricePerHour: price,
		Reliability: &rel, Host: &h, Fit: &hostfit.OfferFit{Level: "ok", Codes: []string{}, VRAM: "unknown"}}
	if k == provision.RunPod {
		h.InetDownMbps, h.DiskBWMBps, h.DiskGB, h.CUDAMax, h.PlacementFloor = 0, 0, 0, "", true
		o.Reliability = nil
		o.Fit.Codes, o.Fit.Level = []string{"host_cuda_placement"}, "info"
	}
	return o
}

// with applies changes to a copy of o (its host and fit copied too).
func with(o provision.Offer, f func(o *provision.Offer, h *hostfit.Host)) provision.Offer {
	h := *o.Host
	o.Host = &h
	if o.Fit != nil {
		fit := *o.Fit
		fit.Codes = append([]string(nil), fit.Codes...)
		o.Fit = &fit
	}
	f(&o, &h)
	return o
}

func codes(level string, cs ...string) func(o *provision.Offer, h *hostfit.Host) {
	return func(o *provision.Offer, _ *hostfit.Host) {
		o.Fit = &hostfit.OfferFit{Level: level, Codes: cs, VRAM: "unknown"}
	}
}

func cands(offers ...provision.Offer) []Candidate {
	out := make([]Candidate, len(offers))
	for i, o := range offers {
		out[i] = Candidate{Offer: o, Caps: capsOf(o.Provider)}
	}
	return out
}

var need100 = hostfit.Need{DiskGB: 100, ImagePeakGB: 20.3, WorkGB: hostfit.WorkGB, ModelSource: "default"}

func TestStartTier(t *testing.T) {
	cases := []struct {
		arch string
		want int
	}{
		{"flux", 24}, {"wan22_14b", 24}, {"qwen_image_edit", 32}, {"hidream", 48}, {"flux2", 48},
		{"no_such_arch", 24}, {"zimage", 24},
	}
	if got := StartTier(nil); got != 24 {
		t.Errorf("StartTier(nil) = %d, want 24", got)
	}
	for _, c := range cases {
		if got := StartTier(&hostfit.ModelRef{Arch: c.arch}); got != c.want {
			t.Errorf("StartTier(%s) = %d, want %d (tiers %+v)", c.arch, got, c.want, hostfit.Tiers(hostfit.ModelRef{Arch: c.arch}))
		}
	}
}

func TestTierOf(t *testing.T) {
	cases := []struct {
		vram float64
		want int
	}{
		{24564.0 / 1024, 24}, {23034.0 / 1024, 24}, {23028.0 / 1024, 24}, {20, 0}, {32607.0 / 1024, 32},
		{40, 32}, {46068.0 / 1024, 48}, {48, 48}, {80, 80}, {141, 141}, {180, 180}, {21.6, 24},
	}
	for _, c := range cases {
		if got := TierOf(c.vram); got != c.want {
			t.Errorf("TierOf(%.2f) = %d, want %d", c.vram, got, c.want)
		}
	}
}

func TestQualifyEachRejectCode(t *testing.T) {
	th := DefaultThresholds()
	base := mk(provision.Vast, "v1", "RTX 4090", 23.99, 0.40)
	small := 50
	cases := []struct {
		name string
		o    provision.Offer
		q    Query
		want string
	}{
		{"qualifies", base, Query{}, ""},
		{"two GPUs", with(base, func(o *provision.Offer, _ *hostfit.Host) { o.GPUCount = 2 }), Query{}, RejectGPUCount},
		{"price", base, Query{MaxPrice: 0.30}, RejectPrice},
		{"price at the limit", base, Query{MaxPrice: 0.40}, ""},
		{"old driver", with(base, codes("block", "host_cuda")), Query{}, RejectCUDA},
		{"unknown driver", with(base, codes("warn", "host_cuda_unknown")), Query{}, RejectCUDAUnknown},
		{"no CUDA capacity", with(base, codes("warn", "runpod_no_cuda_capacity")), Query{}, RejectCUDACapacity},
		{"gpu_sm", with(base, codes("block", "gpu_sm")), Query{}, RejectGPUArch},
		{"host_arch", with(base, codes("block", "host_arch")), Query{}, RejectGPUArch},
		{"gpu_vendor", with(base, codes("block", "gpu_vendor")), Query{}, RejectGPUArch},
		{"sm_75", with(base, codes("warn", "gpu_sm_flash_attn")), Query{}, RejectGPUArch},
		{"gpu_unknown", with(base, codes("info", "gpu_unknown")), Query{}, RejectGPUArch},
		{"disk_image", with(base, codes("block", "disk_image")), Query{}, RejectDisk},
		{"small max disk", with(base, func(o *provision.Offer, _ *hostfit.Host) { o.MaxDiskGB = &small }), Query{}, RejectDisk},
		{"max disk below the asked disk", with(base, func(o *provision.Offer, _ *hostfit.Host) { v := 500; o.MaxDiskGB = &v }), Query{DiskGB: 600}, RejectDisk},
		{"slow internet", with(base, func(_ *provision.Offer, h *hostfit.Host) { h.InetDownMbps = 73 }), Query{}, RejectInet},
		{"slow disk", with(base, func(_ *provision.Offer, h *hostfit.Host) { h.DiskBWMBps = 120 }), Query{}, RejectDiskBW},
		{"reliability", with(base, func(o *provision.Offer, _ *hostfit.Host) { r := 0.9; o.Reliability = &r }), Query{}, RejectReliability},
		{"other block", with(base, codes("block", "image_platform")), Query{}, RejectFitBlock},
		{"not decorated", with(base, func(o *provision.Offer, _ *hostfit.Host) { o.Fit = nil }), Query{}, RejectFitBlock},
		{"unrated GPU", with(base, func(o *provision.Offer, h *hostfit.Host) { o.GPU, h.GPUName = "Mystery 9000", "Mystery 9000" }), Query{}, RejectSpeedUnknown},
		{"blocked GPU has no speed", with(base, func(o *provision.Offer, h *hostfit.Host) { o.GPU, h.GPUName = "Tesla V100", "Tesla V100" }), Query{}, RejectSpeedUnknown},
		{"cuda before inet", with(with(base, codes("block", "host_cuda")), func(_ *provision.Offer, h *hostfit.Host) { h.InetDownMbps = 73 }), Query{}, RejectCUDA},
	}
	for _, c := range cases {
		if got := Qualify(c.o, capsOf(c.o.Provider), need100, c.q, th); got != c.want {
			t.Errorf("%s: Qualify = %q, want %q", c.name, got, c.want)
		}
	}
	for code := range RejectLabels {
		if !strings.Contains(strings.Join(rejectOrder, ","), code) {
			t.Errorf("label for %s, which Qualify never returns", code)
		}
	}
	if len(RejectLabels) != len(rejectOrder) {
		t.Errorf("%d labels for %d codes", len(RejectLabels), len(rejectOrder))
	}
}

func TestUnreportedBandwidthPasses(t *testing.T) {
	th := DefaultThresholds()
	rp := mk(provision.RunPod, "rp", "RTX 4090", 24, 0.69)
	if c := Qualify(rp, capsOf(provision.RunPod), need100, Query{}, th); c != "" {
		t.Fatalf("RunPod without bandwidth: %q", c)
	}
	if r := ReasonLine(Pick{Offer: rp}, false, "13.0"); !strings.Contains(r, "net not reported") {
		t.Errorf("RunPod reason %q", r)
	}
	vast0 := with(mk(provision.Vast, "v0", "RTX 4090", 24, 0.4), func(_ *provision.Offer, h *hostfit.Host) { h.InetDownMbps = 0 })
	if c := Qualify(vast0, capsOf(provision.Vast), need100, Query{}, th); c != "" {
		t.Errorf("Vast inet 0: %q", c)
	}
	if r := ReasonLine(Pick{Offer: vast0}, false, "13.0"); !strings.Contains(r, "net not reported") {
		t.Errorf("Vast inet 0 reason %q", r)
	}
	vast73 := with(vast0, func(_ *provision.Offer, h *hostfit.Host) { h.InetDownMbps = 73 })
	if c := Qualify(vast73, capsOf(provision.Vast), need100, Query{}, th); c != RejectInet {
		t.Errorf("Vast inet 73: %q", c)
	}

	// QuickPod rows that report no disk speed never fail on it.
	img, _ := hostfit.FallbackImage(provision.DefaultImage)
	n := 0
	for _, raw := range quickPodRows(t) {
		m, _ := raw["_machines"].(map[string]any)
		if bw, _ := m["disk_bw"].(float64); bw != 0 {
			continue
		}
		n++
		o := quickPodOffer(raw, img)
		if c := Qualify(o, capsOf(provision.QuickPod), need100, Query{}, th); c == RejectDiskBW {
			t.Errorf("QuickPod %s with disk_bw 0 rejected for disk_bw", o.ID)
		}
		fast := with(o, func(o *provision.Offer, h *hostfit.Host) {
			h.InetDownMbps, h.CUDAMax, o.GPUCount, h.GPUCount = 900, "13.1", 1, 1
			fit := hostfit.OfferFitFor(img, *h, need100, hostfit.VRAMFit{Verdict: hostfit.VRAMUnknown})
			o.Fit = &fit
			r := 0.99
			o.Reliability = &r
		})
		if c := Qualify(fast, capsOf(provision.QuickPod), need100, Query{}, th); c != "" {
			t.Errorf("QuickPod %s with fast internet and disk_bw 0: %q (fit %+v)", o.ID, c, fast.Fit)
			continue
		}
		if r := ReasonLine(Pick{Offer: fast}, false, "13.0"); !strings.Contains(r, "900 Mbps · disk speed not reported") {
			t.Errorf("QuickPod reason %q", r)
		}
		slow := with(fast, func(_ *provision.Offer, h *hostfit.Host) { h.DiskBWMBps = 120 })
		if c := Qualify(slow, capsOf(provision.QuickPod), need100, Query{}, th); c != RejectDiskBW {
			t.Errorf("QuickPod with 120 MB/s disk: %q", c)
		}
	}
	if n == 0 {
		t.Fatal("no QuickPod fixture row with disk_bw 0")
	}

	// Within 2 % per dollar, the offer that reports its bandwidth wins.
	unverified := mk(provision.RunPod, "rp-4090", "RTX 4090", 24, 0.40) // 2.50 per $
	near := mk(provision.Vast, "v-4090", "RTX 4090", 24, 0.4065)        // 2.46 per $ (1.6 % less)
	far := mk(provision.Vast, "v-4090", "RTX 4090", 24, 0.4167)         // 2.40 per $ (4 % less)
	v, _, _, _ := Rank(cands(unverified, near), 24, need100, Query{}, th)
	if v == nil || v.Offer.ID != "v-4090" {
		t.Errorf("within 2 %%: value %+v, want the verified Vast offer", v)
	}
	v, _, _, _ = Rank(cands(unverified, far), 24, need100, Query{}, th)
	if v == nil || v.Offer.ID != "rp-4090" {
		t.Errorf("4 %% apart: value %+v, want the cheaper RunPod offer", v)
	}
}

func TestRankValueAndSpeed(t *testing.T) {
	th := DefaultThresholds()
	r3090 := mk(provision.Vast, "3090", "RTX 3090", 24, 0.20)
	r4090 := mk(provision.Vast, "4090", "RTX 4090", 23.99, 0.40)
	r5090 := mk(provision.Vast, "5090", "RTX 5090", 31.84, 0.90)
	v, s, tier, tiers := Rank(cands(r5090, r4090, r3090), 24, need100, Query{}, th)
	if tier != 24 || v == nil || s == nil || v.Offer.ID != "3090" || s.Offer.ID != "4090" {
		t.Fatalf("tier %d value %+v speed %+v", tier, v, s)
	}
	if math.Abs(v.PerDollar-2.75) > 1e-9 || v.Kind != "value" || s.Kind != "speed" || v.TierGB != 24 || v.SpeedSource == "" {
		t.Errorf("value pick %+v", v)
	}
	if len(tiers) != 1 || tiers[0].Offers != 2 || tiers[0].Qualified != 2 {
		t.Errorf("tiers %+v (the 5090 is tier 32 and takes no part)", tiers)
	}

	v, s, tier, tiers = Rank(cands(r5090), 24, need100, Query{}, th)
	if tier != 32 || v == nil || s == nil || v.Offer.ID != "5090" || s.Offer.ID != "5090" {
		t.Fatalf("24 empty: tier %d value %+v speed %+v", tier, v, s)
	}
	if len(tiers) != 2 || tiers[0].TierGB != 24 || tiers[0].Offers != 0 || tiers[1].TierGB != 32 {
		t.Errorf("tiers %+v", tiers)
	}

	// value == speed: both fields hold the same offer
	v, s, _, _ = Rank(cands(r4090), 24, need100, Query{}, th)
	if v == nil || s == nil || v.Offer.ID != s.Offer.ID {
		t.Errorf("one offer: value %+v speed %+v", v, s)
	}
}

func TestRankTieBreaks(t *testing.T) {
	th := DefaultThresholds()
	rel := func(r float64) func(o *provision.Offer, h *hostfit.Host) {
		return func(o *provision.Offer, _ *hostfit.Host) { o.Reliability = &r }
	}
	pick := func(offers ...provision.Offer) (string, string) {
		v, s, _, _ := Rank(cands(offers...), 24, need100, Query{}, th)
		return v.Offer.ID, s.Offer.ID
	}
	a := mk(provision.Vast, "a", "RTX 4090", 24, 0.40)
	// same per-dollar is impossible at different prices for the same GPU, so the speed pick shows the price rule
	if _, s := pick(mk(provision.Vast, "dear", "RTX 4090", 24, 0.50), a); s != "a" {
		t.Errorf("price: speed %s, want a", s)
	}
	if v, s := pick(with(a, rel(0.96)), with(mk(provision.Vast, "b", "RTX 4090", 24, 0.40), rel(0.99))); v != "b" || s != "b" {
		t.Errorf("reliability: %s %s, want b", v, s)
	}
	qp := with(mk(provision.QuickPod, "a0", "RTX 4090", 24, 0.40), rel(0.99))
	if v, s := pick(qp, a); v != "a" || s != "a" {
		t.Errorf("provider order: %s %s, want the Vast offer", v, s)
	}
	if v, s := pick(mk(provision.Vast, "b", "RTX 4090", 24, 0.40), a); v != "a" || s != "a" {
		t.Errorf("id: %s %s, want a", v, s)
	}
	// no reliability ranks below any reliability
	none := with(mk(provision.Vast, "0", "RTX 4090", 24, 0.40), func(o *provision.Offer, _ *hostfit.Host) { o.Reliability = nil })
	if v, _ := pick(none, with(a, rel(0.96))); v != "a" {
		t.Errorf("null reliability: %s, want a", v)
	}
}

func TestRankIsDeterministic(t *testing.T) {
	th := DefaultThresholds()
	var offers []provision.Offer
	gpus := []string{"RTX 3090", "RTX 4090", "RTX A5000", "L4", "RTX 3090 Ti"}
	for i := 0; i < 40; i++ {
		g := gpus[i%len(gpus)]
		k := []provision.Kind{provision.Vast, provision.QuickPod, provision.RunPod}[i%3]
		offers = append(offers, mk(k, string(rune('a'+i%26))+strings.Repeat("x", i/26), g, 23.5, 0.2+float64(i%7)*0.05))
	}
	v0, s0, _, t0 := Rank(cands(offers...), 24, need100, Query{}, th)
	r := rand.New(rand.NewPCG(1, 2))
	for i := 0; i < 20; i++ {
		r.Shuffle(len(offers), func(a, b int) { offers[a], offers[b] = offers[b], offers[a] })
		v, s, _, tiers := Rank(cands(offers...), 24, need100, Query{}, th)
		if v.Offer.ID != v0.Offer.ID || s.Offer.ID != s0.Offer.ID || tiers[0].Qualified != t0[0].Qualified {
			t.Fatalf("shuffle %d: value %s speed %s, want %s %s", i, v.Offer.ID, s.Offer.ID, v0.Offer.ID, s0.Offer.ID)
		}
	}
}

func TestReasonLine(t *testing.T) {
	rp := mk(provision.RunPod, "rp", "RTX 4090", 24, 0.69)
	rpCatalog := with(rp, func(_ *provision.Offer, h *hostfit.Host) { h.CUDAMax = "13.2" })
	vast := with(mk(provision.Vast, "v", "RTX 4090", 24, 0.34), func(_ *provision.Offer, h *hostfit.Host) { h.InetDownMbps = 912 })
	qp := with(mk(provision.QuickPod, "q", "RTX 3090", 24, 0.18), func(_ *provision.Offer, h *hostfit.Host) {
		h.InetDownMbps, h.DiskBWMBps, h.CUDAMax = 732.17, 0, "13.1"
	})
	cases := []struct {
		o     provision.Offer
		cross bool
		want  string
	}{
		{rp, false, "RTX 4090 · 24 GB · $0.69/h · net not reported · CUDA 13.0+ by RunPod"},
		{rp, true, "RTX 4090 on RunPod · 24 GB · $0.69/h · net not reported · CUDA 13.0+ by RunPod"},
		{rpCatalog, false, "RTX 4090 · 24 GB · $0.69/h · net not reported · CUDA 13.2 OK"},
		{vast, false, "RTX 4090 · 24 GB · $0.34/h · 912 Mbps · CUDA 13.2 OK"},
		{vast, true, "RTX 4090 on Vast.ai · 24 GB · $0.34/h · 912 Mbps · CUDA 13.2 OK"},
		{qp, false, "RTX 3090 · 24 GB · $0.18/h · 732 Mbps · disk speed not reported · CUDA 13.1 OK"},
		{qp, true, "RTX 3090 on QuickPod · 24 GB · $0.18/h · 732 Mbps · disk speed not reported · CUDA 13.1 OK"},
	}
	for _, c := range cases {
		if got := ReasonLine(Pick{Offer: c.o}, c.cross, "13.0"); got != c.want {
			t.Errorf("ReasonLine\n got %q\nwant %q", got, c.want)
		}
	}
}

// Every GPU in the three adapters' fixtures that the image can run (no
// gpu_arch reject) has a speed, through the raw-name-then-display lookup.
func TestFixtureGPUsHaveASpeed(t *testing.T) {
	img, _ := hostfit.FallbackImage(provision.DefaultImage)
	type gpu struct {
		where   string
		display string
		host    hostfit.Host
	}
	var all []gpu

	var rp struct {
		GPUs []hostfit.RunPodGPUType `json:"gpus"`
	}
	readJSON(t, "../provision/runpod/runpodmock/testdata/gpus.json", &rp)
	var rpNames struct {
		GPUs []struct{ Name string } `json:"gpus"`
	}
	readJSON(t, "../provision/runpod/runpodmock/testdata/gpus.json", &rpNames)
	for i, g := range rp.GPUs {
		all = append(all, gpu{"runpod " + g.ID, rpNames.GPUs[i].Name, hostfit.FromRunPodGPU(g, "13.0")})
	}
	for _, f := range []string{"bundles.json", "bundles_live.json"} {
		var v struct {
			Offers []map[string]any `json:"offers"`
		}
		readJSON(t, "../provision/vast/vastmock/testdata/"+f, &v)
		for _, raw := range v.Offers {
			h := hostfit.FromVast(raw)
			all = append(all, gpu{"vast " + f + " " + h.GPUName, h.GPUName, h})
		}
	}
	for _, raw := range quickPodRows(t) {
		h := hostfit.FromQuickPod(raw)
		all = append(all, gpu{"quickpod " + h.GPUName, displayName(h.GPUName), h})
	}
	if len(all) < 20 {
		t.Fatalf("only %d fixture GPUs read", len(all))
	}
	checked := 0
	for _, g := range all {
		blocked := false
		for _, c := range hostfit.Fit(img, g.host, hostfit.FitOptions{}) {
			for _, a := range archCodes {
				blocked = blocked || c.Code == a
			}
		}
		if blocked {
			continue
		}
		checked++
		h := g.host
		if s, _, ok := SpeedOf(provision.Offer{GPU: g.display, Host: &h}); !ok || s <= 0 {
			t.Errorf("%s (%q): no speed", g.where, g.display)
		}
	}
	if checked < 10 {
		t.Fatalf("only %d runnable fixture GPUs", checked)
	}
	t.Logf("%d of %d fixture GPUs are runnable and have a speed", checked, len(all))
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func quickPodRows(t *testing.T) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, f := range []string{"rentable.json", "rentable_live.json"} {
		var rows []map[string]any
		readJSON(t, "../provision/quickpod/quickpodmock/testdata/"+f, &rows)
		out = append(out, rows...)
	}
	return out
}

// displayName strips the vendor words QuickPod's names carry, like the adapter.
func displayName(raw string) string {
	for _, p := range []string{"NVIDIA GeForce ", "NVIDIA "} {
		raw = strings.TrimPrefix(raw, p)
	}
	return raw
}

// quickPodOffer is a fixture row as the adapter shapes it (reliability
// 0..100 normalised), decorated for img.
func quickPodOffer(raw map[string]any, img *imagecheck.Image) provision.Offer {
	h := hostfit.FromQuickPod(raw)
	o := provision.Offer{ID: jsonNum(raw["id"]), Provider: provision.QuickPod, GPU: displayName(h.GPUName), GPUCount: h.GPUCount,
		VRAMGB: int(math.Round(h.VRAMGB)), PricePerHour: raw["hourly_cost"].(float64), Host: &h}
	if m, ok := raw["_machines"].(map[string]any); ok {
		if r, ok := m["reliability"].(float64); ok {
			r /= 100
			o.Reliability = &r
		}
	}
	fit := hostfit.OfferFitFor(img, h, need100, hostfit.VRAMFit{Verdict: hostfit.VRAMUnknown})
	o.Fit = &fit
	return o
}

func jsonNum(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
