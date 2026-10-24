package hostfit

// Version is a CUDA version, major.minor. driver.go adds only the
// functions that parse and compare it.
type Version struct{ Major, Minor int }

// ModelRef is the job's model section as the studio sends it, for VRAM and
// disk hints.
type ModelRef struct {
	Arch       string `json:"arch"`
	NameOrPath string `json:"name_or_path,omitempty"`
	Quantize   bool   `json:"quantize,omitempty"`
	QType      string `json:"qtype,omitempty"`
	LowVRAM    bool   `json:"low_vram,omitempty"`
	Resolution int    `json:"resolution,omitempty"`
	Frames     int    `json:"frames,omitempty"`
}

// Need is the disk a launch should ask for: the image while it unpacks, the
// model and the run's work files.
type Need struct {
	DiskGB      int     `json:"disk_gb"`       // what to ask the provider for
	ImagePeakGB float64 `json:"image_peak_gb"` // compressed + unpacked
	ModelGB     float64 `json:"model_gb"`
	WorkGB      float64 `json:"work_gb"`      // outputs, checkpoints, optimizer state, latent/text caches
	ModelSource string  `json:"model_source"` // hf | table | none | default (no or unknown model: the MinDiskGBUnknownModel floor)
}

// VRAMFit is the VRAM verdict for one model on one GPU size.
type VRAMFit struct {
	Verdict string `json:"verdict"` // fits | tight | no | unknown
	Note    string `json:"note"`
	Source  string `json:"source"`           // repo | vendor | estimate | ""
	Recipe  string `json:"recipe,omitempty"` // what makes a tight cell fit
}

// OfferFit is the badge one offer gets for an image.
type OfferFit struct {
	Level   string   `json:"level"`   // ok | info | warn | block
	Codes   []string `json:"codes"`   // check codes, for tests and tooltips
	Summary string   `json:"summary"` // one short line, e.g. "Driver too old (CUDA 12.8)" or "≈ 3 min to pull"
	PullMin *float64 `json:"pull_min"`
	VRAM    string   `json:"vram"` // fits | tight | no | unknown
}

// Filter is what an image (plus a model and policy) needs from a host,
// provider-neutral. Adapters express it in their own query or
// apply it client-side.
type Filter struct {
	CUDAMin         Version `json:"cuda_min"`
	SMAllowed       []int   `json:"sm_allowed"`          // known CCs (from the GPU table) that RunsOn(Requires.SM); minus 75 unless AllowTuring
	SMHidden        []int   `json:"sm_hidden,omitempty"` // known CCs the image runs that the policy hides (75 unless AllowTuring): a warn when forced
	Platform        string  `json:"platform"`            // linux/amd64
	Vendor          string  `json:"vendor"`              // nvidia
	MinDiskGB       int     `json:"min_disk_gb"`         // Need.DiskGB
	ImagePeakGB     float64 `json:"image_peak_gb"`       // Need.ImagePeakGB: a disk below it plus headroom cannot even pull the image
	MinVRAMGB       float64 `json:"min_vram_gb"`         // from the model's "tight" tier when a model is given; 0 otherwise
	MinInetDownMbps float64 `json:"min_inet_down_mbps"`  // 0 unless Policy.Fast (then FastInetDownMbps)
	MinDiskBWMBps   float64 `json:"min_disk_bw_mbps"`    // 0 unless Policy.Fast (then FastDiskBWMBps)
	AllowTuring     bool    `json:"allow_turing"`
	ShowAll         bool    `json:"show_all"` // keep hosts the image cannot run and hosts below the VRAM floor, badged; the disk and fast-host floors still apply
}

// Policy is the user's choices that shape a Filter beyond what the image needs.
type Policy struct {
	AllowTuring bool // keep sm_75 hosts (hidden by default and warned when forced)
	Fast        bool // "Fast hosts only": FastInetDownMbps and FastDiskBWMBps
	ShowAll     bool // keep hosts the image cannot run, badged (offers ?show_all=1)
}

const (
	// FastInetDownMbps: the image is 6.54 GB compressed. With the pull model's
	// link efficiency eta = 0.5, 500 Mbps pulls it in about 3.5 min; the 73 Mbps
	// hosts seen on Vast and QuickPod take about 25 min. The studio labels it
	// "Fast hosts only (500 Mbps+)". UNCONFIRMED: Vast inet_down is assumed Mbps
	// (vast-cli says Mb/s, the docs say MB/s); QuickPod _machines.inet_down's unit
	// is unknown (Mbps-like values 73-908).
	FastInetDownMbps = 500.0
	// FastDiskBWMBps: derived from the pull model, where unpacking runs at
	// min(disk_bw x kappa, 150) MB/s with kappa = 0.5. At 300 MB/s the disk stops
	// being the bottleneck, so more changes nothing. Kappa and the 150 MB/s
	// gunzip cap are UNCONFIRMED. Unit UNCONFIRMED: vast.md lists disk_bw
	// without a unit (sample 2932.6), image-provider-match.md reads it as MB/s; the
	// QuickPod _machines.disk_bw unit is unknown.
	FastDiskBWMBps = 300.0
	// MinDiskGBUnknownModel: the disk floor when no model (or an unknown arch) is
	// given, so a launch can still hold a base model. Equals provision.DefaultDiskGB;
	// provision_test asserts the two are equal. UNCONFIRMED tuning.
	MinDiskGBUnknownModel = 100
)
