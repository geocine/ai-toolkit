// Package photocheck measures dataset photos and judges them for LoRA
// training: per-photo flags, duplicate groups, dataset findings, an A-F grade,
// hints and "pick the best N". One judge, in Go: the studio only measures and
// posts metrics to POST /v1/photos/assess; the face job only measures.
package photocheck

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// MetricsVersion bumps whenever the Metrics shape changes: the version a
// client posts to assess, and the one cached rows carry.
const MetricsVersion = 1

// MetricsCacheVersion is the photo_metrics version the server keeps its
// measurements under. It bumps whenever a measurement definition changes;
// cached rows under an older one are measured again lazily, while posted
// metrics of the same MetricsVersion stay accepted. 2: a pair of bars needs
// content inside that differs from them.
const MetricsCacheVersion = 2

// FaceAlgo bumps whenever the face job's measurements change.
const FaceAlgo = 1

// FaceModel names the face job's models.
const FaceModel = "yunet2023mar+sface2021dec"

// Limits.
const (
	MaxAssessItems = 2000
	MaxAssessBody  = 4 << 20
	MaxFaceJob     = 2000
	DefaultR       = 1024
)

// Kind is what the dataset teaches.
type Kind string

const (
	KindPerson Kind = "person"
	KindStyle  Kind = "style"
	KindObject Kind = "object"
)

// Level of a flag or finding.
type Level string

const (
	LevelBlock Level = "block"
	LevelWarn  Level = "warn"
	LevelInfo  Level = "info"
)

// Flag is one per-photo finding (catalogue: flags.go).
type Flag string

const (
	// file and format
	FlagUnreadable  Flag = "unreadable"
	FlagUnsupported Flag = "unsupported"
	FlagHuge        Flag = "huge"
	FlagGray16      Flag = "gray16"
	FlagTransparent Flag = "transparent"
	FlagAnimated    Flag = "animated"
	FlagCMYK        Flag = "cmyk"
	FlagSameStem    Flag = "same_stem"
	FlagRotated     Flag = "rotated"
	FlagGenerated   Flag = "generated"
	FlagDerived     Flag = "derived"
	// duplicates
	FlagDuplicate Flag = "duplicate"
	FlagSimilar   Flag = "similar"
	FlagBurst     Flag = "burst"
	// pixels
	FlagTiny        Flag = "tiny"
	FlagSmall       Flag = "small"
	FlagUpscaled    Flag = "upscaled"
	FlagBlurry      Flag = "blurry"
	FlagDark        Flag = "dark"
	FlagBright      Flag = "bright"
	FlagFlat        Flag = "flat"
	FlagGrayscale   Flag = "grayscale"
	FlagStrongColor Flag = "strong_color"
	FlagNoisy       Flag = "noisy"
	FlagCompressed  Flag = "compressed"
	FlagScreenshot  Flag = "screenshot"
	FlagFromScreen  Flag = "from_screen"
	FlagBars        Flag = "bars"
	FlagAspect      Flag = "aspect"
	// caption words
	FlagMentionsCovered Flag = "mentions_covered"
	FlagMentionsPeople  Flag = "mentions_people"
	FlagMentionsText    Flag = "mentions_text"
	FlagMentionsTone    Flag = "mentions_tone"
	FlagMentionsScreen  Flag = "mentions_screen"
	// faces (M2)
	FlagNoFace        Flag = "no_face"
	FlagOthers        Flag = "others"
	FlagSmallFace     Flag = "small_face"
	FlagFaceCut       Flag = "face_cut"
	FlagFaceBlurry    Flag = "face_blurry"
	FlagBacklit       Flag = "backlit"
	FlagNotThem       Flag = "not_them"
	FlagIdentityCheck Flag = "identity_check"
	FlagSmoothed      Flag = "smoothed"
	FlagText          Flag = "text"
	FlagWatermark     Flag = "watermark"
	// look closer (M3; info until calibrated)
	FlagEyesHidden Flag = "eyes_hidden"
	FlagCovered    Flag = "covered"
	FlagFiltered   Flag = "filtered"
	FlagCollage    Flag = "collage"
	FlagBackView   Flag = "back_view"
	FlagExpression Flag = "expression_extreme"
	FlagLookalike  Flag = "lookalike"
)

// Decision is the user's per-photo choice, stored by sha256.
type Decision string

const (
	DecisionKeep     Decision = "keep"     // keep anyway: warn flags are waived, the pick keeps it
	DecisionIntended Decision = "intended" // IntendedClears are on purpose
)

// IntendedClears are the flags "This is intended" waives.
var IntendedClears = []Flag{FlagNoFace, FlagBackView, FlagCovered, FlagEyesHidden, FlagGrayscale,
	FlagMentionsCovered, FlagMentionsTone}

// TrainerExts are what toolkit/data_loader.py reads.
var TrainerExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true}

// ConvertibleExts are images the trainer skips that a photo check lists as
// unsupported (and can convert when Convert() names the format).
var ConvertibleExts = map[string]bool{".heic": true, ".heif": true, ".avif": true, ".jxl": true,
	".bmp": true, ".gif": true, ".tif": true, ".tiff": true}

// ExifInfo is the part of EXIF the checks use.
type ExifInfo struct {
	// DateTimeOriginal as "2006-01-02T15:04:05" plus ".000" from
	// SubSecTimeOriginal when present; camera-local time, no zone.
	Time     string `json:"time,omitempty"`
	Make     string `json:"make,omitempty"`
	Model    string `json:"model,omitempty"`
	Software string `json:"software,omitempty"`
}

// Metrics is what one photo measures as (section 3); the JSON is PhotoMetrics.
type Metrics struct {
	Version      int       `json:"version"`
	File         string    `json:"file"`
	SHA256       string    `json:"sha256,omitempty"`
	Bytes        int64     `json:"bytes"`
	Format       string    `json:"format"` // jpeg png webp gif bmp tiff heic heif avif jxl other
	Width        int       `json:"width"`  // after EXIF orientation
	Height       int       `json:"height"`
	Orientation  int       `json:"orientation"` // EXIF 1..8, 1 when absent
	Decoded      bool      `json:"decoded"`
	Error        string    `json:"error,omitempty"`
	Source       string    `json:"source"`         // "server" | "browser"
	Mode         string    `json:"mode,omitempty"` // rgb gray gray16 cmyk paletted
	Alpha        float64   `json:"alpha_transparent,omitempty"`
	Animated     bool      `json:"animated,omitempty"`
	DHash        string    `json:"dhash,omitempty"` // 16 hex chars
	DHashFlip    string    `json:"dhash_flip,omitempty"`
	Thumb32      string    `json:"thumb32,omitempty"` // base64, 32x32 8-bit luma, row-major
	Sharp        float64   `json:"sharp,omitempty"`
	LumaMean     float64   `json:"luma_mean,omitempty"`
	LumaStd      float64   `json:"luma_std,omitempty"`
	ClipLo       float64   `json:"clip_lo,omitempty"`
	ClipHi       float64   `json:"clip_hi,omitempty"`
	ClipHiCenter float64   `json:"clip_hi_center,omitempty"`
	Colorful     float64   `json:"colorful,omitempty"`
	ChromaSpread float64   `json:"chroma_spread,omitempty"`
	Layout       string    `json:"layout,omitempty"` // base64, 48 bytes
	Flat         float64   `json:"flat,omitempty"`
	Bars         float64   `json:"bars,omitempty"`
	UIBand       bool      `json:"ui_band,omitempty"`
	Noise        *float64  `json:"noise,omitempty"`  // server only
	Detail       *float64  `json:"detail,omitempty"` // server only: upscale round-trip PSNR, dB
	JPEGQ        int       `json:"jpeg_q,omitempty"` // 0 unknown
	PNGColorType int       `json:"png_color_type,omitempty"`
	PNGBitDepth  int       `json:"png_bit_depth,omitempty"`
	Exif         *ExifInfo `json:"exif,omitempty"`
	Generated    bool      `json:"generated,omitempty"`
}

// Face is one detected face, measured by the face job (section 12.3).
type Face struct {
	Box       [4]float64    `json:"box"` // x, y, w, h as fractions of the oriented image
	Kps       [5][2]float64 `json:"kps"` // YuNet order: right eye, left eye, nose, right and left mouth corner; fractions
	Score     float64       `json:"score"`
	Side      int           `json:"side"`       // max(w, h) of the box in native pixels
	AlignSide int           `json:"align_side"` // the same on the alignment image (long side <= 1280)
	Sharp     float64       `json:"sharp"`      // Laplacian variance inside the box on the section-3 S scale
	MeanY     float64       `json:"mean_y"`
	ClipHi    float64       `json:"clip_hi"`
	Embedded  bool          `json:"embedded"` // has a row in the identity matrix
}

// TextBox is x, y, w, h (fractions of the oriented image) and a score.
type TextBox [5]float64

// FaceData is one photo's face measurements (photo_faces.data).
type FaceData struct {
	Algo        int       `json:"algo"`
	Faces       []Face    `json:"faces"`
	FrameSharp  float64   `json:"frame_sharp"` // tile p90 on the same copy as Face.Sharp
	TextChecked bool      `json:"text_checked"`
	Text        []TextBox `json:"text,omitempty"`
	Error       string    `json:"error,omitempty"`
}

// FaceMatrix is the face-to-face cosine matrix of one face check. It holds no
// face templates: only similarities between this dataset's faces.
type FaceMatrix struct {
	Faces []string `json:"faces"` // "<sha256>:<index into FaceData.Faces>", embedded faces only
	Cos   string   `json:"cos"`   // base64 of little-endian binary16, upper triangle i<j row-major: n(n-1)/2 values
}

// FaceJobResult is faces.json, written by photo_faces.py.
type FaceJobResult struct {
	Algo    int                 `json:"algo"`
	CV2     string              `json:"cv2"`
	Model   string              `json:"model"`
	Items   map[string]FaceData `json:"items"` // by sha256
	Matrix  FaceMatrix          `json:"identity"`
	Skipped []string            `json:"skipped"`
}

// Identity is photo_sets.identity.
type Identity struct {
	FaceMatrix
	SetSig string `json:"set_sig"`
	Model  string `json:"model"`
	JobID  string `json:"job_id"`
	At     int64  `json:"at"` // unix ms
}

// Input is one photo as the judge sees it.
type Input struct {
	Metrics  Metrics
	Face     *FaceData // nil: faces not checked for this photo
	Caption  *string   // nil: no caption file
	Decision Decision
	Parent   string // a derived face crop: the parent's file
	Existing bool   // assess with a dataset: already stored; joins duplicate groups only
	Pending  bool   // not measured yet
}

// SetState is the dataset-level state the judge needs.
type SetState struct {
	Kind      Kind
	Reference string // sha256 of the "this is them" photo
	Identity  *Identity
	Faces     FacesState
	SetAside  []SetAsideItem
	Originals []OriginalItem
	Convert   []string
}

// Options for Review, Assess and Judge.
type Options struct {
	Kind        Kind          // "" = the stored kind, then person
	Resolutions []int         // the run's resolution list; R = max; default [1024]
	CaptionExt  string        // default "txt"
	Trigger     string        // removed from captions before the word checks
	Pick        int           // > 0: also return a pick of this many
	Wait        time.Duration // how long Review waits for missing metrics
}

// R is the largest training resolution.
func (o Options) R() int {
	r := 0
	for _, v := range o.Resolutions {
		if v > r {
			r = v
		}
	}
	if r <= 0 {
		return DefaultR
	}
	return r
}

// ItemFace is the per-photo face summary.
type ItemFace struct {
	Count    int         `json:"count"`
	Px       int         `json:"px"`             // subject face side at training size
	Shot     string      `json:"shot,omitempty"` // close bust half full back
	Yaw      string      `json:"yaw,omitempty"`  // front three_quarter profile
	Cohesion *float64    `json:"cohesion,omitempty"`
	RefSim   *float64    `json:"ref_sim,omitempty"`
	Box      *[4]float64 `json:"box,omitempty"`
}

// Item is one judged photo.
type Item struct {
	File      string          `json:"file"`
	SHA256    string          `json:"sha256,omitempty"`
	Width     int             `json:"width,omitempty"`
	Height    int             `json:"height,omitempty"`
	Bytes     int64           `json:"bytes"`
	Px        int             `json:"px,omitempty"`
	Flags     []Flag          `json:"flags"`
	Waived    []Flag          `json:"waived,omitempty"`
	Level     Level           `json:"level,omitempty"`
	Notes     map[Flag]string `json:"notes"`
	Group     string          `json:"group,omitempty"`
	Keeper    bool            `json:"keeper,omitempty"`
	SameAs    string          `json:"same_as,omitempty"`
	SharpRel  float64         `json:"sharp_rel,omitempty"`
	Captured  string          `json:"capture_time,omitempty"`
	Session   int             `json:"session,omitempty"` // 1-based; 0 = no date
	Face      *ItemFace       `json:"face,omitempty"`
	Decision  Decision        `json:"decision,omitempty"`
	Reference bool            `json:"reference,omitempty"`
	Parent    string          `json:"derived_from,omitempty"`
	Pending   bool            `json:"pending,omitempty"`
	Usable    bool            `json:"usable"`
}

// FixOp is one photos/fix operation (ops: FixOps).
type FixOp struct {
	Op    string   `json:"op"`
	Files []string `json:"files,omitempty"`
	Flags []Flag   `json:"flags,omitempty"`
	All   bool     `json:"all,omitempty"`
	N     int      `json:"n,omitempty"`
	Max   int      `json:"max,omitempty"`
}

// FixOps are the valid op names, in the order the 400 lists them.
var FixOps = []string{"set_aside", "restore", "set_aside_flagged", "keep_best", "bake_orientation",
	"convert", "rename_same_stem", "face_crops", "remove_face_crops"}

// DefaultSetAsideFlags is set_aside_flagged without flags.
var DefaultSetAsideFlags = []Flag{FlagDuplicate, FlagTiny, FlagUnreadable}

// Finding is one dataset-level finding.
type Finding struct {
	ID      string   `json:"id"`
	Level   Level    `json:"level"`
	Message string   `json:"message"`
	Fix     string   `json:"fix,omitempty"`
	Files   []string `json:"files,omitempty"`
	Op      *FixOp   `json:"op,omitempty"`
}

// Group is a duplicate, similar or burst group.
type Group struct {
	ID        string   `json:"id"`
	Kind      string   `json:"kind"` // duplicate similar burst
	Files     []string `json:"files"`
	Keep      string   `json:"keep"`
	Distance  int      `json:"distance"`            // smallest dHash Hamming (direct or mirrored)
	Confirmed bool     `json:"confirmed,omitempty"` // duplicate: same sha256 or NCC-confirmed, so pre-selected
}

// Coverage is what the set covers, on the axes with data.
type Coverage struct {
	Basis          []string       `json:"basis"` // pixels exif captions faces
	Shot           map[string]int `json:"shot,omitempty"`
	Yaw            map[string]int `json:"yaw,omitempty"`
	Dated          int            `json:"dated"`
	Sessions       int            `json:"sessions"`
	LargestSession int            `json:"largest_session"`
	SpanDays       int            `json:"span_days"`
	Outfits        map[string]int `json:"outfits,omitempty"`
	Places         map[string]int `json:"places,omitempty"`
	Light          map[string]int `json:"light,omitempty"`
}

// FacesState is the face check's state for a dataset.
type FacesState struct {
	Available bool     `json:"available"`
	State     string   `json:"state"` // none queued running done failed stopped
	Checked   int      `json:"checked"`
	Stale     bool     `json:"stale"`
	Added     int      `json:"added"`
	JobID     string   `json:"job_id,omitempty"`
	Job       any      `json:"job"` // the handler fills in the Job view; null otherwise
	Skipped   []string `json:"skipped,omitempty"`
}

// LeftOut is one photo the pick left out, with a reason in plain words.
type LeftOut struct {
	File   string `json:"file"`
	Reason string `json:"reason"`
}

// Pick is "pick the best N".
type Pick struct {
	N       int       `json:"n"`
	Kept    []string  `json:"kept"`
	LeftOut []LeftOut `json:"left_out"`
	Note    string    `json:"note,omitempty"`
}

// SetAsideItem is a photo left out into .set_aside/.
type SetAsideItem struct {
	File  string `json:"file"` // path under .set_aside/
	From  string `json:"from"` // the dataset path it came from
	Flags []Flag `json:"flags"`
}

// OriginalItem is an original a convert or a bake kept in .set_aside/ as a
// backup of the photo made from it. It is not a left-out photo: restore
// with all leaves it, and restoring it by name puts it back in place of
// that photo.
type OriginalItem struct {
	File string `json:"file"` // path under .set_aside/
	From string `json:"from"` // the dataset path it came from
	Of   string `json:"of"`   // the dataset path of the photo made from it
}

// Review is the judged photo report (PhotoReview).
type Review struct {
	Kind           Kind               `json:"kind"`
	Resolution     int                `json:"resolution"`
	MetricsVersion int                `json:"metrics_version"`
	Items          []Item             `json:"items"`
	Total          int                `json:"total"`
	Flagged        int                `json:"flagged"`
	Usable         int                `json:"usable"`
	Pending        int                `json:"pending"`
	Counts         map[Flag]int       `json:"counts"`
	Groups         []Group            `json:"groups"`
	Grade          string             `json:"grade"`
	GradeLine      string             `json:"grade_line"`
	Provisional    bool               `json:"provisional"`
	Basis          []string           `json:"basis"`
	Coverage       Coverage           `json:"coverage"`
	Findings       []Finding          `json:"findings"`
	Hints          []string           `json:"hints"`
	SetAside       []SetAsideItem     `json:"set_aside"`
	Originals      []OriginalItem     `json:"originals"`
	Faces          FacesState         `json:"faces"`
	Reference      string             `json:"reference,omitempty"` // file
	Pick           *Pick              `json:"pick,omitempty"`
	Thresholds     map[string]float64 `json:"thresholds"`
	Convert        []string           `json:"convert"` // formats the server can convert
}

// Change is one photos/fix change.
type Change struct {
	File   string `json:"file"`
	Op     string `json:"op"`
	To     string `json:"to,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// Fix is the photos/fix result.
type Fix struct {
	Changes []Change `json:"changes"`
	Applied bool     `json:"applied"`
	Review  *Review  `json:"review"`
}

// FixRequest is the photos/fix body.
type FixRequest struct {
	Ops        []FixOp         `json:"ops"`
	Files      []string        `json:"files,omitempty"`
	Apply      bool            `json:"apply,omitempty"`
	Kind       Kind            `json:"kind,omitempty"`
	Resolution json.RawMessage `json:"resolution,omitempty"`
}

// SettingsPatch is the photos/settings body.
type SettingsPatch struct {
	Kind *Kind `json:"kind,omitempty"`
	// Reference: absent = unchanged; null or "" = clear; "file" = set.
	Reference json.RawMessage `json:"reference,omitempty"`
	// Decisions by file; a null value clears it.
	Decisions map[string]*Decision `json:"decisions,omitempty"`
}

// AssessRequest is the POST /v1/photos/assess body.
type AssessRequest struct {
	Items      []Metrics           `json:"items"`
	Kind       Kind                `json:"kind,omitempty"`
	Resolution json.RawMessage     `json:"resolution,omitempty"` // 1024 or [512, 768, 1024]
	Dataset    string              `json:"dataset,omitempty"`
	Pick       int                 `json:"pick,omitempty"`
	Decisions  map[string]Decision `json:"decisions,omitempty"` // by file
	Captions   map[string]string   `json:"captions,omitempty"`  // .txt picked with the photos, by image file
}

// UploadResult is POST /v1/datasets/{name}/files.
type UploadResult struct {
	Items   []string  `json:"items"`
	Renamed []Renamed `json:"renamed"`
	Skipped []string  `json:"skipped"`
}

// Renamed is one upload saved under another name.
type Renamed struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// ParseResolutions reads 1024, [512, 768, 1024] or "512,768,1024".
func ParseResolutions(raw json.RawMessage) ([]int, error) {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return nil, nil
	}
	var n int
	if json.Unmarshal(raw, &n) == nil {
		return checkRes([]int{n})
	}
	var list []int
	if json.Unmarshal(raw, &list) == nil {
		return checkRes(list)
	}
	var str string
	if json.Unmarshal(raw, &str) == nil {
		return ParseResolutionList(str)
	}
	return nil, fmt.Errorf("resolution must be a number, a list of numbers or \"512,768,1024\"")
}

// ParseResolutionList reads "1024" or "512,768,1024" (a query value).
func ParseResolutionList(s string) ([]int, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	var out []int
	for _, part := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("resolution %q is not a number", part)
		}
		out = append(out, v)
	}
	return checkRes(out)
}

func checkRes(v []int) ([]int, error) {
	for _, r := range v {
		if r < 64 || r > 8192 {
			return nil, fmt.Errorf("resolution %d is outside 64-8192", r)
		}
	}
	sort.Ints(v)
	return v, nil
}
