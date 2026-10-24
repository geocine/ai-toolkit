package photocheck

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Thresholds are the judge's rule values (section 4). Every one is a [cal]
// starting value: cmd/photocal tunes them on real sets, and a change lands
// here. Map() gives the keys of review.thresholds.
type Thresholds struct {
	// duplicates and bursts (4.2)
	DupHamming     int     // duplicate: min(direct, mirrored) Hamming at most this
	DupNCC         float64 // ... and thumb32 NCC (direct or mirrored) at least this
	SimilarHamming int     // similar: Hamming up to this (or a duplicate distance without NCC)
	BurstSeconds   float64 // burst: EXIF times at most this far apart
	BurstHamming   int     // ... and Hamming at most this
	KeeperFocus    float64 // keeper: in focus means sharp >= this x the group's best

	// file and pixels (4.1, 4.3)
	HugePixels   float64 // the trainer skips images over this many pixels
	Transparent  float64 // alpha_transparent above this
	TinyShort    int     // short side below this
	SmallRel     float64 // round(edge) below this x R
	UpscaledDB   float64 // detail above this (info)
	BlurRel      float64 // sharp below this x the set median ...
	BlurFloor    float64 // ... and below this
	SetMin       int     // set-relative rules need this many measured photos
	DarkMean     float64 // luma_mean below this
	DarkFace     float64 // subject face mean_y below this
	BrightMean   float64 // luma_mean above this
	BrightCenter float64 // clip_hi_center above this
	BrightFace   float64 // subject face mean_y above this
	BrightClip   float64 // subject face clip_hi above this
	FlatStd      float64 // luma_std below this
	GrayChroma   float64 // chroma_spread below this
	StrongColor  float64 // colorful above this
	Noisy        float64 // noise above this
	CompressedQ  int     // jpeg_q below this (and above 0)
	Bars         float64 // bars above this
	AspectInfo   float64 // long/short above this: info
	AspectWarn   float64 // ... above this: warn

	// faces (4.5, 6)
	FaceScore     float64 // a candidate face scores at least this ...
	FaceHeight    float64 // ... and its box is at least this share of the image height
	SubjectSide   float64 // subject candidates are at least this x the photo's largest face
	OthersScore   float64 // others: a second face scoring at least this ...
	OthersSide    float64 // ... at least this x the subject's side ...
	SameCos       float64 // ... with a cosine to the subject below this (OpenCV's SFace threshold)
	SmallFace     float64 // small_face: trained face side below this
	CropFace      float64 // faces up to this feed the face-crop offer only
	FaceCutBox    float64 // face_cut: the box within this of an edge ...
	FaceCutKps    float64 // ... and a landmark within this of the same edge
	FaceBlurRel   float64 // face_blurry: below this x the set's face median ...
	FaceFloor     float64 // ... and below this
	BacklitRel    float64 // backlit: face mean_y below this x luma_mean ...
	BacklitMax    float64 // ... and below this
	SmoothedSide  float64 // smoothed: alignment side at least this ...
	SmoothedRel   float64 // ... and face/frame sharpness below this x the set median
	TextScore     float64 // text boxes scoring at least this count
	WatermarkIoU  float64 // a box repeated with this IoU ...
	WatermarkRep  int     // ... in at least this many other photos
	CornerShare   float64 // or wholly inside the outer share of a corner ...
	CornerHeight  float64 // ... and lower than this share of the image
	ShotClose     float64 // fh at least this: close
	ShotBust      float64 // ... bust
	ShotHalf      float64 // ... half; below: full
	YawFront      float64 // |r - 0.5| below this: front
	YawQuarter    float64 // ... below this: three_quarter
	YawMinAlign   int     // yaw is unknown below this alignment side ...
	YawMinScore   float64 // ... or below this score
	EmbedMinAlign int     // faces below this alignment side are not embedded
	IdentityMin   int     // the set rules need this many embedded subjects
	OutlierK      float64 // outlier: c below median - k x scaled MAD ...
	MADFloor      float64 // ... with the scaled MAD at least this ...
	OutlierFloor  float64 // ... or below this
	TwoPeople     float64 // two_people: a second component this share of the faces (and >= 3)
	NotThemGradeF float64 // grade F when not_them is above this share of faced photos

	// sessions, captions and coverage (4.4, 7.2)
	SessionGapMin float64 // a session ends at a gap over this many minutes
	SessionShare  float64 // one_session: the largest session at least this share
	CoverMin      int     // coverage rules need this many dated photos or captions
	RepeatShare   float64 // same_outfit / same_place: one key above this share
	EraYears      float64 // era_span: a dated span over this many years
	GrayMixMin    int     // grayscale_mix: at least this many ...
	GrayMixShare  float64 // ... or this share ...
	GrayMixMax    float64 // ... but below this share
	LookMin       float64 // look_mismatch: covered in at least this share ...
	LookMax       float64 // ... and at most this share
	FilterShare   float64 // filter_consistent: strong colour or smoothing above this share
	LowResShare   float64 // low_res_share: small + tiny at least this share
	CountHigh     int     // count_high: usable above this
	FrontShare    float64 // angle_gap: front above this share of known yaws
	FullShare     float64 // shot_gap no_full: full + half below this share
	CloseShare    float64 // shot_gap no_close: close + bust below this share
	DayShare      float64 // light_single: day above this share of light keys
}

// DefaultThresholds are the shipped values.
func DefaultThresholds() Thresholds {
	return Thresholds{
		DupHamming: 5, DupNCC: 0.95, SimilarHamming: 10, BurstSeconds: 2, BurstHamming: 12, KeeperFocus: 0.35,

		HugePixels: 178_956_970, Transparent: 0.005, TinyShort: 512, SmallRel: 0.75, UpscaledDB: 40,
		BlurRel: 0.35, BlurFloor: 25, SetMin: 8,
		DarkMean: 40, DarkFace: 45, BrightMean: 220, BrightCenter: 0.25, BrightFace: 225, BrightClip: 0.20,
		FlatStd: 25, GrayChroma: 3.0, StrongColor: 100, Noisy: 8, CompressedQ: 70, Bars: 0.04,
		AspectInfo: 2.0, AspectWarn: 2.5,

		FaceScore: 0.6, FaceHeight: 0.03, SubjectSide: 0.4, OthersScore: 0.85, OthersSide: 0.4, SameCos: 0.363,
		SmallFace: 64, CropFace: 128, FaceCutBox: 0.02, FaceCutKps: 0.03, FaceBlurRel: 0.35, FaceFloor: 15,
		BacklitRel: 0.6, BacklitMax: 70, SmoothedSide: 96, SmoothedRel: 0.5,
		TextScore: 0.6, WatermarkIoU: 0.5, WatermarkRep: 2, CornerShare: 0.15, CornerHeight: 0.08,
		ShotClose: 0.42, ShotBust: 0.24, ShotHalf: 0.11, YawFront: 0.15, YawQuarter: 0.35,
		YawMinAlign: 40, YawMinScore: 0.7, EmbedMinAlign: 40, IdentityMin: 5,
		OutlierK: 3, MADFloor: 0.015, OutlierFloor: 0.10, TwoPeople: 0.25, NotThemGradeF: 0.30,

		SessionGapMin: 60, SessionShare: 0.5, CoverMin: 6, RepeatShare: 0.5, EraYears: 3,
		GrayMixMin: 3, GrayMixShare: 0.20, GrayMixMax: 0.90, LookMin: 0.10, LookMax: 0.50,
		FilterShare: 0.5, LowResShare: 0.25, CountHigh: 50, FrontShare: 0.75, FullShare: 0.10,
		CloseShare: 0.30, DayShare: 0.85,
	}
}

// Map is review.thresholds: every value under a snake_case key.
func (t Thresholds) Map() map[string]float64 {
	return map[string]float64{
		"dup_hamming": float64(t.DupHamming), "dup_ncc": t.DupNCC, "similar_hamming": float64(t.SimilarHamming),
		"burst_seconds": t.BurstSeconds, "burst_hamming": float64(t.BurstHamming), "keeper_focus": t.KeeperFocus,

		"huge_pixels": t.HugePixels, "transparent": t.Transparent, "tiny_short": float64(t.TinyShort),
		"small_rel": t.SmallRel, "upscaled_db": t.UpscaledDB, "blur_rel": t.BlurRel, "blur_floor": t.BlurFloor,
		"set_min": float64(t.SetMin), "dark_mean": t.DarkMean, "dark_face": t.DarkFace,
		"bright_mean": t.BrightMean, "bright_center": t.BrightCenter, "bright_face": t.BrightFace,
		"bright_clip": t.BrightClip, "flat_std": t.FlatStd, "gray_chroma": t.GrayChroma,
		"strong_color": t.StrongColor, "noisy": t.Noisy, "compressed_q": float64(t.CompressedQ), "bars": t.Bars,
		"aspect_info": t.AspectInfo, "aspect_warn": t.AspectWarn,

		"face_score": t.FaceScore, "face_height": t.FaceHeight, "subject_side": t.SubjectSide,
		"others_score": t.OthersScore, "others_side": t.OthersSide, "same_cos": t.SameCos,
		"small_face": t.SmallFace, "crop_face": t.CropFace, "face_cut_box": t.FaceCutBox,
		"face_cut_kps": t.FaceCutKps, "face_blur_rel": t.FaceBlurRel, "face_floor": t.FaceFloor,
		"backlit_rel": t.BacklitRel, "backlit_max": t.BacklitMax, "smoothed_side": t.SmoothedSide,
		"smoothed_rel": t.SmoothedRel, "text_score": t.TextScore, "watermark_iou": t.WatermarkIoU,
		"watermark_repeat": float64(t.WatermarkRep), "corner_share": t.CornerShare, "corner_height": t.CornerHeight,
		"shot_close": t.ShotClose, "shot_bust": t.ShotBust, "shot_half": t.ShotHalf,
		"yaw_front": t.YawFront, "yaw_quarter": t.YawQuarter, "yaw_min_align": float64(t.YawMinAlign),
		"yaw_min_score": t.YawMinScore, "embed_min_align": float64(t.EmbedMinAlign),
		"identity_min": float64(t.IdentityMin), "outlier_k": t.OutlierK, "mad_floor": t.MADFloor,
		"outlier_floor": t.OutlierFloor, "two_people": t.TwoPeople, "not_them_grade_f": t.NotThemGradeF,

		"session_gap_min": t.SessionGapMin, "session_share": t.SessionShare, "cover_min": float64(t.CoverMin),
		"repeat_share": t.RepeatShare, "era_years": t.EraYears, "gray_mix_min": float64(t.GrayMixMin),
		"gray_mix_share": t.GrayMixShare, "gray_mix_max": t.GrayMixMax, "look_min": t.LookMin,
		"look_max": t.LookMax, "filter_share": t.FilterShare, "low_res_share": t.LowResShare,
		"count_high": float64(t.CountHigh), "front_share": t.FrontShare, "full_share": t.FullShare,
		"close_share": t.CloseShare, "day_share": t.DayShare,
	}
}

// FlagInfo is a flag's default level, its label (it reads inside "N
// {label}" summaries) and a general line about it. The judge writes the
// finished per-photo sentence into Item.Notes.
type FlagInfo struct {
	Level Level
	Label string
	Help  string
}

// Catalog describes every flag. A few levels depend on the photo or the
// set (bars and aspect, no_face); the judge sets those per item.
var Catalog = map[Flag]FlagInfo{
	FlagUnreadable:  {LevelWarn, "unreadable", "The file does not open as an image, so the trainer skips it."},
	FlagUnsupported: {LevelWarn, "not read by the trainer", "The trainer reads JPEG, PNG and WebP only."},
	FlagHuge:        {LevelWarn, "too large", "The trainer skips images over about 179 MP."},
	FlagGray16:      {LevelWarn, "16-bit grey", "A 16-bit greyscale PNG: the trainer clips it to white."},
	FlagTransparent: {LevelWarn, "transparent", "The trainer drops transparency and trains the colour hidden underneath."},
	FlagAnimated:    {LevelInfo, "animated", "Only the first frame trains."},
	FlagCMYK:        {LevelInfo, "CMYK", "CMYK colours are converted without a profile and can shift."},
	FlagSameStem:    {LevelWarn, "sharing a caption", "Two files with the same name before the extension share one caption."},
	FlagRotated:     {LevelInfo, "stored sideways", "The trainer turns it upright, but the auto-captioner reads it sideways."},
	FlagGenerated:   {LevelInfo, "AI-generated", "Made by an image model, so it teaches that model's look."},
	FlagDerived:     {LevelInfo, "face crop", "A close-up crop added by \"Add close-up crops\"."},

	FlagDuplicate: {LevelWarn, "duplicates", "The same picture as another photo (resized, re-saved or mirrored)."},
	FlagSimilar:   {LevelInfo, "very similar", "Close to another photo, maybe the same moment."},
	FlagBurst:     {LevelInfo, "same moment", "Taken within 2 seconds of another photo; a burst counts as one photo toward variety."},

	FlagTiny:        {LevelWarn, "too small", "Under 512 px on the short side, too small to teach detail."},
	FlagSmall:       {LevelWarn, "low resolution", "Trains below the run's resolution; the trainer never scales photos up."},
	FlagUpscaled:    {LevelInfo, "maybe enlarged", "Looks enlarged from a smaller photo."},
	FlagBlurry:      {LevelWarn, "blurry", "Soft even in its sharpest part, so the LoRA learns the blur."},
	FlagDark:        {LevelWarn, "too dark", "Mostly shadow, so detail is lost."},
	FlagBright:      {LevelWarn, "blown out", "Large white areas with no detail."},
	FlagFlat:        {LevelInfo, "flat", "Very low contrast: haze, fog or a faded filter."},
	FlagGrayscale:   {LevelInfo, "black and white", "No colour, or a single tint."},
	FlagStrongColor: {LevelInfo, "strong colour", "Very saturated colour, often a filter."},
	FlagNoisy:       {LevelInfo, "grainy", "Heavy grain from low light."},
	FlagCompressed:  {LevelInfo, "compressed", "Saved at low JPEG quality, so detail is blocky."},
	FlagScreenshot:  {LevelWarn, "screenshots", "Looks like a screen capture, not a photo."},
	FlagFromScreen:  {LevelInfo, "saved from a screen", "A screen size with no camera data."},
	FlagBars:        {LevelWarn, "black bars", "Black or flat bars on the edges. The LoRA learns them."},
	FlagAspect:      {LevelInfo, "very wide or tall", "It trains in its own size group and loses detail on the short side."},

	FlagMentionsCovered: {LevelInfo, "sunglasses or hat (caption)", "The caption mentions something that hides the face."},
	FlagMentionsPeople:  {LevelInfo, "other people (caption)", "The caption mentions other people."},
	FlagMentionsText:    {LevelInfo, "text (caption)", "The caption mentions text or a logo."},
	FlagMentionsTone:    {LevelInfo, "black and white (caption)", "The caption says the photo is black and white."},
	FlagMentionsScreen:  {LevelInfo, "screen (caption)", "The caption mentions a screen."},

	FlagNoFace:        {LevelWarn, "no face", "No face found."},
	FlagOthers:        {LevelWarn, "other people", "Someone else is in the photo, so the LoRA mixes them in."},
	FlagSmallFace:     {LevelWarn, "small face", "The face trains at too few pixels to learn it from."},
	FlagFaceCut:       {LevelInfo, "face cut off", "Part of the face is outside the frame."},
	FlagFaceBlurry:    {LevelWarn, "face blurry", "The face is soft, even if the rest is sharp."},
	FlagBacklit:       {LevelInfo, "face in shadow", "The face is much darker than the rest of the photo."},
	FlagNotThem:       {LevelWarn, "someone else?", "This face looks different from the rest of the set."},
	FlagIdentityCheck: {LevelInfo, "check the face", "Hard to compare. Make sure it is them."},
	FlagSmoothed:      {LevelInfo, "smoothed skin", "The skin looks much smoother than the rest of the photo."},
	FlagText:          {LevelInfo, "text", "Text in the photo (a sign or a print)."},
	FlagWatermark:     {LevelWarn, "watermark", "A watermark or overlay text. The LoRA can learn to draw it."},

	FlagEyesHidden: {LevelInfo, "eyes hidden", "Sunglasses or something else hides the eyes."},
	FlagCovered:    {LevelInfo, "face covered", "A mask, a hand or a hat covers part of the face."},
	FlagFiltered:   {LevelInfo, "filtered", "Looks edited with a filter."},
	FlagCollage:    {LevelInfo, "collage", "Several photos in one."},
	FlagBackView:   {LevelInfo, "back view", "Seen from behind."},
	FlagExpression: {LevelInfo, "strong expression", "An exaggerated expression."},
	FlagLookalike:  {LevelInfo, "look-alike", "Looks very like another photo in the set."},
}

// flagOrder is the catalogue order (section 4), used for summaries.
var flagOrder = []Flag{
	FlagUnreadable, FlagUnsupported, FlagHuge, FlagGray16, FlagTransparent, FlagAnimated, FlagCMYK, FlagSameStem,
	FlagRotated, FlagGenerated, FlagDerived,
	FlagDuplicate, FlagSimilar, FlagBurst,
	FlagTiny, FlagSmall, FlagUpscaled, FlagBlurry, FlagDark, FlagBright, FlagFlat, FlagGrayscale, FlagStrongColor,
	FlagNoisy, FlagCompressed, FlagScreenshot, FlagFromScreen, FlagBars, FlagAspect,
	FlagMentionsCovered, FlagMentionsPeople, FlagMentionsText, FlagMentionsTone, FlagMentionsScreen,
	FlagNoFace, FlagOthers, FlagSmallFace, FlagFaceCut, FlagFaceBlurry, FlagBacklit, FlagNotThem, FlagIdentityCheck,
	FlagSmoothed, FlagText, FlagWatermark,
	FlagEyesHidden, FlagCovered, FlagFiltered, FlagCollage, FlagBackView, FlagExpression, FlagLookalike,
}

var flagRank = func() map[Flag]int {
	m := make(map[Flag]int, len(flagOrder))
	for i, f := range flagOrder {
		m[f] = i
	}
	return m
}()

// screenSizes are common phone, tablet and desktop screen sizes (portrait
// or landscape, both stored). A screen size is one signal only.
var screenSizes = func() map[[2]int]bool {
	sizes := [][2]int{
		{1170, 2532}, {1179, 2556}, {1284, 2778}, {1290, 2796}, {1206, 2622}, {1320, 2868}, {1125, 2436},
		{1242, 2688}, {828, 1792}, {750, 1334}, {1080, 2400}, {1080, 2340}, {1080, 1920}, {1440, 3200},
		{1440, 3088}, {1920, 1080}, {2560, 1440}, {3840, 2160}, {1366, 768}, {1536, 864}, {2880, 1800},
		{3024, 1964}, {3456, 2234},
	}
	m := make(map[[2]int]bool, 2*len(sizes))
	for _, s := range sizes {
		m[s] = true
		m[[2]int{s[1], s[0]}] = true
	}
	return m
}()

// isScreenSize reports a width x height from the screen-size table.
func isScreenSize(w, h int) bool { return screenSizes[[2]int{w, h}] }

// trainerUnreadable are the flags that mean the trainer cannot use the
// file at all: "Keep anyway" does not bring these back into a pick.
var trainerUnreadable = map[Flag]bool{FlagUnreadable: true, FlagUnsupported: true, FlagHuge: true}

// pickHardOut are the flags that keep a photo out of a pick unless the
// user decided to keep it (section 8).
var pickHardOut = []Flag{FlagUnreadable, FlagUnsupported, FlagHuge, FlagGray16, FlagTransparent, FlagTiny,
	FlagScreenshot, FlagNotThem, FlagOthers}

// Copy helpers. Every per-photo sentence is built here, so no UI
// hard-codes copy.

func noteUnsupported(ext string, convertible bool) string {
	s := strings.ToUpper(ext) + " files are skipped by the trainer. It reads JPEG, PNG and WebP."
	if !convertible {
		s += " Export it as JPEG, or add it in the studio in Safari."
	}
	return s
}

func noteHuge(w, h int) string {
	return fmt.Sprintf("%d MP: the trainer skips images over about 179 MP.", int(math.Round(float64(w)*float64(h)/1e6)))
}

func noteSameStem(a, b string) string {
	return fmt.Sprintf("%s and %s have the same name before the extension, so they share one caption.", a, b)
}

func noteDuplicate(same string, stored bool) string {
	s := fmt.Sprintf("Same picture as %s (resized, re-saved or mirrored). %s is the better copy", same, same)
	if stored {
		return s + ", already in the dataset."
	}
	return s + "."
}

func noteSimilar(same string) string {
	return fmt.Sprintf("Close to %s, maybe the same moment. Keep both only if the pose or expression differs.", same)
}

func noteBurst(same string) string {
	return fmt.Sprintf("Taken within 2 seconds of %s. A burst counts as one photo toward variety.", same)
}

func noteTiny(w, h int) string {
	return fmt.Sprintf("%d×%d: under 512 px on the short side, too small to teach detail.", w, h)
}

func noteSmall(w, h, px, r int) string {
	return fmt.Sprintf("%d×%d: trains at about %d px, below the run's %d px. The trainer never scales photos up.", w, h, px, r)
}

func noteBlurry(rel float64) string {
	if rel <= 0 {
		return "Soft even in its sharpest part, so the LoRA learns the blur."
	}
	return fmt.Sprintf("Soft even in its sharpest part (%s× the set's typical sharpness), so the LoRA learns the blur.", ratio(rel))
}

func noteFaceBlurry(rel float64) string {
	if rel <= 0 {
		return "The face is soft, even if the rest is sharp."
	}
	return fmt.Sprintf("The face is soft (%s× the set's typical face), even if the rest is sharp.", ratio(rel))
}

func noteCompressed(q int) string {
	return fmt.Sprintf("Saved at low JPEG quality (%d), so detail is blocky. The original from the phone is better.", q)
}

func noteFromScreen(w, h int) string {
	return fmt.Sprintf("%d×%d is a screen size and it has no camera data: maybe a video frame, a story save or a screenshot. Use the original photo if you have it.", w, h)
}

func noteAspect(r float64) string {
	return fmt.Sprintf("%s:1 is very wide or tall. It trains in its own size group and loses detail on the short side.",
		strconv.FormatFloat(math.Round(r*10)/10, 'f', -1, 64))
}

func noteNoFace() string {
	return "No face found. Fine for one or two shots from behind or far away. Mark it \"This is intended\" to keep it."
}

func noteSmallFace(fpx int) string {
	return fmt.Sprintf("The face trains at only about %d px across, too few to learn it from.", fpx)
}

func noteNotThem(ref bool) string {
	if ref {
		return "This face looks different from the rest of the set and from your reference photo."
	}
	return "This face looks different from the rest of the set."
}

func noteIdentityCheck(why string) string {
	return fmt.Sprintf("Hard to compare (%s). Make sure it is them.", why)
}

func noteWatermark(others int) string {
	if others > 0 {
		return fmt.Sprintf("A watermark or overlay text in the same spot as in %d other photos. The LoRA can learn to draw it.", others)
	}
	return "A watermark or overlay text. The LoRA can learn to draw it."
}

func noteDerived(parent string) string {
	return fmt.Sprintf("A close-up crop of %s, added by \"Add close-up crops\".", parent)
}

// fixedNotes are the flags whose sentence never changes.
var fixedNotes = map[Flag]string{
	FlagUnreadable:  "This file does not open as an image, so the trainer skips it.",
	FlagGray16:      "A 16-bit greyscale PNG: the trainer clips it to white.",
	FlagTransparent: "Transparent areas: the trainer drops the transparency and trains whatever colour is hidden underneath.",
	FlagAnimated:    "Only the first frame trains.",
	FlagCMYK:        "CMYK colours are converted without a profile and can shift.",
	FlagRotated:     "Stored sideways with a rotation tag. The trainer turns it upright, but the auto-captioner reads it sideways.",
	FlagGenerated:   "Made by an image model, so it teaches that model's look.",
	FlagUpscaled:    "Looks enlarged from a smaller photo, so it may hold less detail than its size suggests.",
	FlagFlat:        "Very low contrast: haze, fog or a faded filter.",
	FlagGrayscale:   "No colour, or a single tint (black and white, sepia). Several of these teach the LoRA to drop colour.",
	FlagStrongColor: "Very saturated colour, often a filter. It carries into results.",
	FlagNoisy:       "Heavy grain from low light.",
	FlagScreenshot:  "Looks like a screen capture (status bar or app screen), not a photo.",
	FlagBars:        "Black or flat bars on the edges. The LoRA learns them. Crop them off.",
	FlagBright:      "Large white areas with no detail.",
	FlagOthers:      "Someone else is in the photo, so the LoRA mixes them in. Crop them out or leave it out.",
	FlagFaceCut:     "Part of the face is outside the frame.",
	FlagBacklit:     "The face is much darker than the rest of the photo, often from a window behind.",
	FlagSmoothed:    "The skin looks much smoother than the rest of the photo, often a beauty filter.",
	FlagText:        "Text in the photo (a sign or a print). Fine if the caption mentions it.",
}

// ratio prints a sharpness ratio with two significant decimals ("0.12").
func ratio(v float64) string {
	return strconv.FormatFloat(math.Round(v*100)/100, 'f', -1, 64)
}

// plural is "1 photo" / "3 photos".
func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return strconv.Itoa(n) + " " + many
}
