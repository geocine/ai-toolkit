package photocheck

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// Hint copy (section 7.4), shared with findings and the pick's notes.
const (
	hintTwoPeople = "Mark one photo of the right person with This is them."
	hintNoFull    = "Add 3-4 full-length photos, or the LoRA will only do faces."
	hintNoClose   = "Add 4-6 close-ups where the face fills most of the frame."
	hintAngles    = "Add 3-4 photos with the head turned partly away."
	hintProfiles  = "Add 2 side profiles."
	hintSessions  = "Add photos from other days, places and light."
	hintOutfits   = "Add 3 or more photos in other clothes."
	hintPlaces    = "Add photos with other backgrounds."
	hintLight     = "Add a few indoor or evening photos."
	hintCountHigh = "More photos will not add variety now. Pick the best 30 instead."
	hintObject    = "Add photos from the side and back, and one detail shot."
	hintStyle     = "Add the style on other subjects."
)

// toLookAt are the warn flags the to_look_at finding sums up, in order.
var toLookAt = []Flag{FlagBlurry, FlagFaceBlurry, FlagDark, FlagBright, FlagScreenshot, FlagBars, FlagOthers,
	FlagSmallFace, FlagWatermark}

// formatProblems are the flags the format_problems finding sums up.
var formatProblems = []Flag{FlagUnreadable, FlagUnsupported, FlagHuge, FlagGray16, FlagTransparent, FlagSameStem}

var levelRank = map[Level]int{LevelBlock: 0, LevelWarn: 1, LevelInfo: 2}

// setFacts are the counts the findings, the grade and the hints share.
type setFacts struct {
	judged    int // new photos measured
	usable    int
	aim       int // the usable count to aim for
	checked   bool
	captioned int
	countNone bool
	shot      map[string]int
	shotKnown int
	noFull    bool
	noClose   bool
	angleGap  bool
	profiles  int
	oneSess   bool
	outfit    string
	outfitN   int
	sameOut   bool
	place     string
	placeN    int
	samePlace bool
	gaps      int
}

func (j *judge) facts(r *Review) setFacts {
	th := j.th
	f := setFacts{usable: r.Usable, aim: 15, checked: r.Faces.Checked > 0, shot: map[string]int{}}
	if j.kind == KindStyle {
		f.aim = 20
	}
	readable := false
	for _, p := range j.own {
		if p.Pending {
			continue
		}
		f.judged++
		if TrainerExts[p.ext] && !p.has(FlagUnreadable) && !p.has(FlagHuge) {
			readable = true
		}
		if p.cap != nil && p.dupOf == "" {
			f.captioned++
		}
	}
	f.countNone = !readable && r.Pending == 0
	cov := r.Coverage
	person := j.kind == KindPerson

	for k, n := range cov.Shot {
		if k != "back" {
			f.shot[k] = n
			f.shotKnown += n
		}
	}
	if f.checked && person && f.usable >= 10 && f.shotKnown > 0 {
		share := func(a, b string) float64 { return float64(f.shot[a]+f.shot[b]) / float64(f.shotKnown) }
		f.noFull = share("full", "half") < th.FullShare
		f.noClose = !f.noFull && share("close", "bust") < th.CloseShare
	}
	known := 0
	for _, n := range cov.Yaw {
		known += n
	}
	f.profiles = cov.Yaw["profile"]
	f.angleGap = f.checked && person && known >= th.CoverMin && float64(cov.Yaw["front"]) > th.FrontShare*float64(known)
	f.oneSess = person && cov.Dated >= th.CoverMin && float64(cov.LargestSession) >= th.SessionShare*float64(cov.Dated)
	f.outfit, f.outfitN = topKey(cov.Outfits)
	f.place, f.placeN = topKey(cov.Places)
	if person && f.captioned >= th.CoverMin {
		f.sameOut = float64(f.outfitN) > th.RepeatShare*float64(f.captioned)
		f.samePlace = float64(f.placeN) > th.RepeatShare*float64(f.captioned)
	}
	for _, g := range []bool{f.noFull || f.noClose, f.angleGap, f.oneSess, f.sameOut, f.samePlace} {
		if g {
			f.gaps++
		}
	}
	return f
}

// assess fills the findings, the grade and the hints (sections 7.2-7.4).
func (j *judge) assess(r *Review) {
	th := j.th
	f := j.facts(r)
	person := j.kind == KindPerson
	cov := r.Coverage
	var fs []Finding
	add := func(x Finding) { fs = append(fs, x) }
	files := func(flags ...Flag) []string {
		var out []string
		for _, p := range j.own {
			for _, fl := range flags {
				if p.has(fl) {
					out = append(out, p.Metrics.File)
					break
				}
			}
		}
		return out
	}
	hints := j.hints(r, f)

	if f.countNone {
		add(Finding{ID: "count_none", Level: LevelBlock, Message: "No photo the trainer can read.",
			Fix: "Add JPEG, PNG or WebP photos."})
	} else {
		u := f.usable
		switch {
		case j.kind != KindStyle && u < 10:
			add(Finding{ID: "count_very_low", Level: LevelWarn,
				Message: fmt.Sprintf("Only %s. A likeness rarely holds with fewer than 10.", plural(u, "usable photo", "usable photos")),
				Fix:     fmt.Sprintf("Add %d or more.", 15-u)})
		case u < f.aim:
			add(Finding{ID: "count_low", Level: LevelWarn,
				Message: fmt.Sprintf("%s. Aim for 15-30.", capFirst(plural(u, "usable photo", "usable photos"))),
				Fix:     fmt.Sprintf("Add %d.", f.aim-u)})
		case u > th.CountHigh:
			add(Finding{ID: "count_high", Level: LevelInfo,
				Message: fmt.Sprintf("%d photos. 25-40 is usually enough.", u), Op: &FixOp{Op: "keep_best", N: 30}})
		}
	}
	if d := files(FlagDuplicate); len(d) > 0 {
		msg := fmt.Sprintf("%d photos are copies of another one.", len(d))
		if len(d) == 1 {
			msg = "1 photo is a copy of another one."
		}
		add(Finding{ID: "duplicates", Level: LevelWarn, Message: msg, Files: d,
			Op: &FixOp{Op: "set_aside_flagged", Flags: []Flag{FlagDuplicate}}})
	}
	if fp := files(formatProblems...); len(fp) > 0 {
		var kinds []string
		onlyStem := true
		for _, fl := range formatProblems {
			if r.Counts[fl] > 0 {
				kinds = append(kinds, Catalog[fl].Label)
				onlyStem = onlyStem && fl == FlagSameStem
			}
		}
		op := &FixOp{Op: "convert"}
		if onlyStem {
			op = &FixOp{Op: "rename_same_stem"}
		}
		add(Finding{ID: "format_problems", Level: LevelWarn, Files: fp, Op: op,
			Message: fmt.Sprintf("%s the trainer reads wrongly or skips (%s).", plural(len(fp), "file", "files"), strings.Join(kinds, ", "))})
	}
	if low := files(FlagSmall, FlagTiny); f.judged > 0 && float64(len(low)) >= th.LowResShare*float64(f.judged) {
		add(Finding{ID: "low_res_share", Level: LevelWarn, Files: low,
			Message: fmt.Sprintf("%d of %d photos train below %d px.", len(low), f.judged, j.r),
			Fix:     "Use the original files, not copies from a chat app."})
	}
	if look := j.lookFiles(); len(look) > 0 {
		add(Finding{ID: "to_look_at", Level: LevelWarn, Files: look,
			Message: fmt.Sprintf("%s to look at: %s.", plural(len(look), "photo", "photos"), j.lookSummary(r))})
	}
	if j.id.twoPeople {
		add(Finding{ID: "two_people", Level: LevelWarn,
			Message: fmt.Sprintf("These photos seem to show two different people (%d and %d photos).", j.id.main, j.id.second),
			Fix:     "Mark one photo of the right person with \"This is them\"."})
	}
	if nt := files(FlagNotThem); len(nt) > 0 {
		add(Finding{ID: "identity_outliers", Level: LevelWarn, Files: nt,
			Message: fmt.Sprintf("%s may show someone else.", plural(len(nt), "photo", "photos")),
			Op:      &FixOp{Op: "set_aside", Files: nt}})
	}
	if f.oneSess {
		add(Finding{ID: "one_session", Level: LevelWarn,
			Message: fmt.Sprintf("%d of %d photos were taken within one hour.", cov.LargestSession, cov.Dated),
			Fix:     "Add photos from other days and places."})
	}
	if f.sameOut {
		add(Finding{ID: "same_outfit", Level: LevelWarn,
			Message: fmt.Sprintf("\"%s\" is in %d of %d captions.", f.outfit, f.outfitN, f.captioned),
			Fix:     "Add photos in other clothes, or keep the outfit and caption it every time."})
	}
	if f.samePlace {
		add(Finding{ID: "same_place", Level: LevelWarn,
			Message: fmt.Sprintf("%d of %d captions place them in a %s.", f.placeN, f.captioned, f.place),
			Fix:     "Add photos with other backgrounds."})
	}
	if g := files(FlagGrayscale, FlagMentionsTone); f.judged > 0 {
		share := float64(len(g)) / float64(f.judged)
		if (len(g) >= th.GrayMixMin || share >= th.GrayMixShare) && share < th.GrayMixMax && len(g) > 0 {
			add(Finding{ID: "grayscale_mix", Level: LevelWarn, Files: g,
				Message: fmt.Sprintf("%s black and white or toned.", plural(len(g), "photo is", "photos are")),
				Fix:     "Leave them out, unless the LoRA should be black and white."})
		}
	}
	if fl := files(FlagStrongColor, FlagSmoothed, FlagFiltered); f.judged >= 4 && float64(len(fl)) > th.FilterShare*float64(f.judged) {
		add(Finding{ID: "filter_consistent", Level: LevelWarn, Files: fl,
			Message: "Most photos share a strong filter, and the LoRA will learn the filter.",
			Fix:     "Use unedited originals if you have them."})
	}
	if mp := files(FlagMentionsPeople); len(mp) > 0 && !f.checked {
		add(Finding{ID: "people_in_captions", Level: LevelWarn, Files: mp,
			Message: fmt.Sprintf("%s other people.", plural(len(mp), "caption mentions", "captions mention")),
			Fix:     "Check faces, or crop them out."})
	}
	if lm := files(FlagMentionsCovered, FlagEyesHidden); f.captioned > 0 && len(lm) > 0 {
		share := float64(len(lm)) / float64(f.captioned)
		if share >= th.LookMin && share <= th.LookMax {
			add(Finding{ID: "look_mismatch", Level: LevelInfo, Files: lm,
				Message: fmt.Sprintf("Sunglasses or hats in %d of %d photos.", len(lm), f.captioned),
				Fix:     "Replace them, or mark them \"This is intended\" if they are part of the look."})
		}
	}
	if years := float64(cov.SpanDays) / 365.25; years > th.EraYears {
		// Rounded, and never at or under the threshold the rule just passed.
		span := fmt.Sprintf("%d", int(math.Round(years)))
		if math.Round(years) <= th.EraYears {
			span = fmt.Sprintf("more than %g", th.EraYears)
		}
		add(Finding{ID: "era_span", Level: LevelInfo,
			Message: fmt.Sprintf("Photos span %s years. Haircut and age changes can blur the likeness.", span),
			Fix:     "Prefer photos from the last year or two."})
	}
	if rot := files(FlagRotated); len(rot) > 0 {
		add(Finding{ID: "orientation_tags", Level: LevelInfo, Files: rot,
			Message: fmt.Sprintf("%s on a rotation tag (the auto-captioner reads them sideways).", plural(len(rot), "photo relies", "photos rely")),
			Op:      &FixOp{Op: "bake_orientation"}})
	}
	switch {
	case f.noFull:
		add(Finding{ID: "shot_gap", Level: LevelInfo, Message: "Almost all close-ups.", Fix: hintNoFull})
	case f.noClose:
		add(Finding{ID: "shot_gap", Level: LevelInfo, Message: "Few close-ups.", Fix: hintNoClose})
	}
	if f.angleGap {
		add(Finding{ID: "angle_gap", Level: LevelInfo, Message: "Almost every photo faces the camera.", Fix: hintAngles})
	}
	if k := j.faceCropOffer(f); k > 0 {
		add(Finding{ID: "face_crops_offer", Level: LevelInfo,
			Message: fmt.Sprintf("%s small faces. Add close-up crops of the face?", plural(k, "photo has", "photos have")),
			Op:      &FixOp{Op: "face_crops"}})
	}
	if person && !f.checked && r.Faces.Available {
		add(Finding{ID: "faces_unchecked", Level: LevelInfo,
			Message: "Faces not checked yet: other people, look-alikes, angles and framing are unknown.",
			Fix:     "Check faces (datasets.checkFaces)."})
	}
	if r.Faces.Stale {
		msg := "Photos changed after the face check."
		if r.Faces.Added > 0 {
			msg = fmt.Sprintf("%s added after the face check.", plural(r.Faces.Added, "photo was", "photos were"))
		}
		add(Finding{ID: "faces_stale", Level: LevelInfo, Message: msg, Fix: "Check faces again."})
	}
	if r.Pending > 0 {
		add(Finding{ID: "pending", Level: LevelInfo, Message: fmt.Sprintf("%s still being checked.", plural(r.Pending, "photo", "photos"))})
	}
	slices.SortStableFunc(fs, func(a, b Finding) int { return levelRank[a.Level] - levelRank[b.Level] })
	if fs != nil {
		r.Findings = fs
	}

	// Grade: a total cascade, top-down.
	notThem := r.Counts[FlagNotThem]
	faced := 0 // photos with a subject face: the not_them share's base
	for _, p := range j.own {
		if !p.Pending && p.fv != nil && p.fv.subj >= 0 {
			faced++
		}
	}
	r.Provisional = person && r.Faces.Checked == 0
	u := f.usable
	switch {
	case f.countNone || u < 5 || (f.checked && faced > 0 && float64(notThem) > th.NotThemGradeF*float64(faced)):
		r.Grade, r.GradeLine = "F", "Not enough to train a likeness."
	case u < 10:
		r.Grade, r.GradeLine = "D", fmt.Sprintf("Not enough yet. Add %d.", 15-u)
	case u < 15 || f.gaps >= 2 || j.id.twoPeople || notThem > 0:
		r.Grade, r.GradeLine = "C", "Thin. A likeness may not hold."
	case u < 20 || f.gaps == 1 || r.Provisional:
		r.Grade, r.GradeLine = "B", "Good. The hints below make it better."
	default:
		r.Grade, r.GradeLine = "A", "Ready to train."
	}
	r.Hints = hints
}

// hints are at most 3 lines in the section 7.4 priority order.
func (j *judge) hints(r *Review, f setFacts) []string {
	th := j.th
	cov := r.Coverage
	var out []string
	add := func(ok bool, s string) {
		if ok && len(out) < 3 && !slices.Contains(out, s) {
			out = append(out, s)
		}
	}
	if k := f.aim - f.usable; k > 0 {
		switch j.kind {
		case KindStyle:
			add(true, fmt.Sprintf("Add %d more images in this style, of other subjects.", k))
		case KindObject:
			add(true, fmt.Sprintf("Add %d more photos from other angles.", k))
		default:
			add(true, fmt.Sprintf("Add %d more photos of just them.", k))
		}
	}
	person := j.kind == KindPerson
	add(j.id.twoPeople, hintTwoPeople)
	add(f.noFull, hintNoFull)
	add(f.noClose, hintNoClose)
	add(f.angleGap, hintAngles)
	add(person && f.usable >= 12 && f.checked && f.profiles < 2, hintProfiles)
	add(f.oneSess, hintSessions)
	add(f.sameOut, hintOutfits)
	add(f.samePlace, hintPlaces)
	add(j.daytimeOnly(cov), hintLight)
	add(f.usable > th.CountHigh, hintCountHigh)
	if j.kind == KindObject && f.captioned >= th.CoverMin {
		side := false
		for _, p := range j.own {
			side = side || (p.cap != nil && p.cap.sideBack)
		}
		add(!side, hintObject)
	}
	if j.kind == KindStyle && f.captioned >= th.CoverMin {
		lim := 0.4 * float64(f.captioned)
		add(float64(f.placeN) > lim || float64(f.outfitN) > lim, hintStyle)
	}
	if out == nil {
		out = []string{}
	}
	return out
}

// daytimeOnly is hint 10: every dated photo between 08:00 and 18:00 with
// no evening or night light key, or day light in most captions.
func (j *judge) daytimeOnly(cov Coverage) bool {
	th := j.th
	if j.kind != KindPerson {
		return false
	}
	lights := 0
	for _, n := range cov.Light {
		lights += n
	}
	if lights >= th.CoverMin && float64(cov.Light["day"]) > th.DayShare*float64(lights) {
		return true
	}
	if cov.Dated < th.CoverMin || cov.Light["evening"] > 0 || cov.Light["night"] > 0 {
		return false
	}
	for _, p := range j.counted() {
		if p.dated && (p.when.Hour() < 8 || p.when.Hour() >= 18) {
			return false
		}
	}
	return true
}

// lookFiles are the photos with an active to_look_at flag.
func (j *judge) lookFiles() []string {
	var out []string
	for _, p := range j.own {
		for _, f := range toLookAt {
			if p.flags[f] == LevelWarn {
				out = append(out, p.Metrics.File)
				break
			}
		}
	}
	return out
}

// lookSummary is "2 blurry, 1 with other people", most frequent first.
func (j *judge) lookSummary(r *Review) string {
	type kc struct {
		f Flag
		n int
	}
	var ks []kc
	for _, f := range toLookAt {
		n := 0
		for _, p := range j.own {
			if p.flags[f] == LevelWarn {
				n++
			}
		}
		if n > 0 {
			ks = append(ks, kc{f, n})
		}
	}
	slices.SortStableFunc(ks, func(a, b kc) int { return b.n - a.n })
	parts := make([]string, len(ks))
	for i, k := range ks {
		label := Catalog[k.f].Label
		if k.f == FlagOthers {
			label = "with other people"
		}
		parts[i] = fmt.Sprintf("%d %s", k.n, label)
	}
	return strings.Join(parts, ", ")
}

// faceCropOffer is the section 9 offer rule: how many crops to offer
// (0 when the offer does not apply).
func (j *judge) faceCropOffer(f setFacts) int {
	if j.kind != KindPerson || !f.checked || f.usable == 0 {
		return 0
	}
	cropped := map[string]bool{}
	closeBust := 0
	for _, p := range j.own {
		if p.Parent != "" {
			cropped[p.Parent] = true
		}
		if p.usable && p.fv != nil && (p.fv.shot == "close" || p.fv.shot == "bust") {
			closeBust++
		}
	}
	if float64(closeBust) >= 0.25*float64(f.usable) {
		return 0
	}
	n := 0
	for _, p := range j.own {
		if _, ok := cropSquare(p); ok && !cropped[p.Metrics.File] {
			n++
		}
	}
	if n < 3 {
		return 0
	}
	return min(n, max(3, int(math.Floor(0.30*float64(f.usable)))))
}

// cropSquare is a face crop for a usable half- or full-length photo whose
// face is small (section 9): the square's side in source pixels, and
// whether the photo qualifies.
func cropSquare(p *photo) (int, bool) {
	if !p.usable || p.Parent != "" || p.fv == nil || (p.fv.shot != "half" && p.fv.shot != "full") {
		return 0, false
	}
	s := p.fv.subjectFace()
	if s == nil || s.Box[3] >= 0.20 {
		return 0, false
	}
	side := math.Min(2.6*float64(s.Side), float64(min(p.Metrics.Width, p.Metrics.Height)))
	if side < 512 {
		return 0, false
	}
	return int(side), true
}

// capFirst upper-cases the first letter.
func capFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
