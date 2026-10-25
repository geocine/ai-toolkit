package photocheck

import (
	"math"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/collation"
)

// errTooLarge is Metrics.Error for a photo over thumbs.MaxPixels (decode.go):
// its size is known, its pixels were never read.
const errTooLarge = "too large to analyse"

// photo is one input while it is judged.
type photo struct {
	Input
	ext   string // lower-case, with the dot
	flags map[Flag]Level
	notes map[Flag]string

	// groups (groups.go)
	hashed  bool
	h, hf   uint64
	thumb   []byte
	group   string // the item's group: duplicate, then burst, then similar
	keeper  bool
	sameAs  string
	dupOf   string // a duplicate non-keeper: the keeper
	burstOf string // a burst non-keeper: the keeper
	burstID string // any burst member: the group id

	// sessions (coverage.go)
	when    time.Time
	dated   bool
	session int

	// faces (identity.go) and captions (coverage.go)
	fv  *faceView
	cap *captionInfo

	waived []Flag
	usable bool
}

func (p *photo) set(f Flag, l Level, note string) {
	p.flags[f] = l
	if note == "" {
		note = fixedNotes[f]
	}
	p.notes[f] = note
}

func (p *photo) has(f Flag) bool { _, ok := p.flags[f]; return ok }

// measured is a photo whose pixel metrics are known.
func (p *photo) measured() bool { return !p.Pending && p.Metrics.Decoded }

// edge is sqrt(w*h) of the oriented photo.
func (p *photo) edge() float64 {
	return math.Sqrt(float64(p.Metrics.Width) * float64(p.Metrics.Height))
}

// judge holds one Judge call.
type judge struct {
	th     Thresholds
	kind   Kind
	r      int
	opts   Options
	set    SetState
	all    []*photo // every input, sorted by file; existing ones only join groups
	own    []*photo // the inputs that become items
	byFile map[string]*photo
	byHash map[string]*photo // by sha256, first wins

	groups  []Group
	id      identity
	checked int     // photos with face data
	med     float64 // the set's typical frame sharpness (0: too few photos)
}

// Judge judges a set of photos (sections 4-8): per-photo flags with their
// notes, duplicate, burst and similar groups, sessions, faces and identity,
// coverage, dataset findings, the grade and hints, and a pick when
// opts.Pick > 0. It is a pure function of its inputs.
func Judge(in []Input, set SetState, opts Options) *Review {
	return judgeWith(in, set, opts, DefaultThresholds())
}

func judgeWith(in []Input, set SetState, opts Options, th Thresholds) *Review {
	j := &judge{th: th, opts: opts, set: set, r: opts.R(), byFile: map[string]*photo{}, byHash: map[string]*photo{}}
	j.kind = opts.Kind
	if j.kind == "" {
		j.kind = set.Kind
	}
	if j.kind == "" {
		j.kind = KindPerson
	}
	for _, x := range in {
		p := &photo{Input: x, ext: strings.ToLower(path.Ext(x.Metrics.File)), flags: map[Flag]Level{}, notes: map[Flag]string{}}
		j.all = append(j.all, p)
	}
	slices.SortStableFunc(j.all, func(a, b *photo) int { return collation.Compare(a.Metrics.File, b.Metrics.File) })
	for _, p := range j.all {
		if _, dup := j.byFile[p.Metrics.File]; !dup {
			j.byFile[p.Metrics.File] = p
		}
		if s := p.Metrics.SHA256; s != "" && j.byHash[s] == nil {
			j.byHash[s] = p
		}
		if !p.Existing {
			j.own = append(j.own, p)
			if !p.Pending && p.Face != nil && p.Face.Error == "" {
				j.checked++
			}
		}
	}

	j.faces() // before the pixel flags: dark and bright read the subject's face
	for _, p := range j.own {
		if !p.Pending {
			j.fileFlags(p)
		}
	}
	j.sameStems()
	j.pixelFlags()
	j.parseTimes()
	j.groupPhotos()
	j.captions()
	j.sessions()
	j.decide()

	r := j.review()
	if opts.Pick > 0 {
		pk := PickBest(in, r, opts.Pick, opts)
		r.Pick = &pk
	}
	return r
}

// fileFlags are the section 4.1 flags.
func (j *judge) fileFlags(p *photo) {
	m := p.Metrics
	th := j.th
	if p.Parent != "" {
		p.set(FlagDerived, LevelInfo, noteDerived(p.Parent))
	}
	switch {
	case ConvertibleExts[p.ext]:
		p.set(FlagUnsupported, LevelWarn, noteUnsupported(strings.TrimPrefix(p.ext, "."), j.converts(p.ext)))
	case !m.Decoded && m.Error != errTooLarge:
		p.set(FlagUnreadable, LevelWarn, "")
	}
	if float64(m.Width)*float64(m.Height) > th.HugePixels {
		p.set(FlagHuge, LevelWarn, noteHuge(m.Width, m.Height))
	}
	if m.Mode == "gray16" || (m.Format == "png" && m.PNGColorType == 0 && m.PNGBitDepth == 16) {
		p.set(FlagGray16, LevelWarn, "")
	}
	if m.Alpha > th.Transparent {
		p.set(FlagTransparent, LevelWarn, "")
	}
	if m.Animated {
		p.set(FlagAnimated, LevelInfo, "")
	}
	if m.Mode == "cmyk" {
		p.set(FlagCMYK, LevelInfo, "")
	}
	if m.Orientation > 1 && m.Orientation <= 8 && (m.Format == "jpeg" || m.Format == "png" || m.Format == "webp") {
		p.set(FlagRotated, LevelInfo, "")
	}
	if m.Generated {
		p.set(FlagGenerated, LevelInfo, "")
	}
}

// converts reports whether the server converts this extension
// (SetState.Convert lists formats: "bmp", "gif", "tiff", "heic", ...).
func (j *judge) converts(ext string) bool {
	f := strings.TrimPrefix(ext, ".")
	switch f {
	case "tif":
		f = "tiff"
	case "heif":
		if slices.Contains(j.set.Convert, "heif") {
			return true
		}
		f = "heic"
	}
	return slices.Contains(j.set.Convert, f)
}

// sameStems flags media files that share a stem: they share one caption.
// A stored file counts too, so an upload next to it is flagged.
func (j *judge) sameStems() {
	byStem := map[string][]*photo{}
	for _, p := range j.all {
		f := p.Metrics.File
		stem := strings.TrimSuffix(f, path.Ext(f))
		byStem[stem] = append(byStem[stem], p)
	}
	for _, ps := range byStem {
		if len(ps) < 2 {
			continue
		}
		for _, p := range ps {
			if p.Existing || p.Pending {
				continue
			}
			for _, o := range ps {
				if o != p {
					p.set(FlagSameStem, LevelWarn, noteSameStem(p.Metrics.File, o.Metrics.File))
					break
				}
			}
		}
	}
}

// medianSharp is the set's typical frame sharpness, 0 below SetMin
// measured photos. Photos too dark to judge do not count.
func (j *judge) medianSharp() float64 {
	var v []float64
	for _, p := range j.own {
		if p.measured() && !j.tooDark(p) {
			v = append(v, p.Metrics.Sharp)
		}
	}
	if len(v) < j.th.SetMin {
		return 0
	}
	return median(v)
}

// tooDark reports a photo so dark overall (luma_mean under DarkMean) that
// its sharpness and detail readings say more about the exposure than the
// focus: shadows compress every edge. Its blurry and upscaled judgements
// defer to the exposure (it is flagged dark, or the subject's face is lit
// and judged on its own).
func (j *judge) tooDark(p *photo) bool { return p.Metrics.LumaMean < j.th.DarkMean }

// pixelFlags are the section 4.3 flags.
func (j *judge) pixelFlags() {
	th := j.th
	j.med = j.medianSharp()
	med := j.med
	for _, p := range j.own {
		m := p.Metrics
		if p.Pending || m.Width <= 0 || m.Height <= 0 || p.has(FlagUnreadable) {
			continue
		}
		w, h := m.Width, m.Height
		edge := p.edge()
		px := min(int(math.Round(edge)), j.r)
		if min(w, h) < th.TinyShort {
			p.set(FlagTiny, LevelWarn, noteTiny(w, h))
		} else if math.Round(edge) < th.SmallRel*float64(j.r) {
			p.set(FlagSmall, LevelWarn, noteSmall(w, h, px, j.r))
		}
		if a := float64(max(w, h)) / float64(min(w, h)); a > th.AspectWarn {
			p.set(FlagAspect, LevelWarn, noteAspect(a))
		} else if a > th.AspectInfo {
			p.set(FlagAspect, LevelInfo, noteAspect(a))
		}
		if !m.Decoded {
			continue
		}
		dim := j.tooDark(p)
		if m.Detail != nil && *m.Detail > th.UpscaledDB && !dim {
			p.set(FlagUpscaled, LevelInfo, "")
		}
		subj := p.fv.subjectFace()
		if subj == nil && !dim {
			soft := m.Sharp < th.BlurFloor
			if med > 0 {
				soft = soft && m.Sharp < th.BlurRel*med
			}
			if soft {
				rel := 0.0
				if med > 0 {
					rel = m.Sharp / med
				}
				p.set(FlagBlurry, LevelWarn, noteBlurry(rel))
			}
		}
		switch {
		case subj != nil && subj.MeanY < th.DarkFace:
			p.set(FlagDark, LevelWarn, "The face is in deep shadow.")
		case subj == nil && m.LumaMean < th.DarkMean:
			p.set(FlagDark, LevelWarn, "Mostly shadow, so detail is lost.")
		}
		switch {
		case subj != nil:
			if subj.MeanY > th.BrightFace || subj.ClipHi > th.BrightClip {
				p.set(FlagBright, LevelWarn, "")
			}
		case j.kind == KindObject:
			if m.ClipHiCenter > th.BrightCenter {
				p.set(FlagBright, LevelWarn, "")
			}
		case m.LumaMean > th.BrightMean || m.ClipHiCenter > th.BrightCenter:
			p.set(FlagBright, LevelWarn, "")
		}
		if m.LumaStd < th.FlatStd {
			p.set(FlagFlat, LevelInfo, "")
		}
		if m.ChromaSpread < th.GrayChroma {
			p.set(FlagGrayscale, LevelInfo, "")
		}
		if m.Colorful > th.StrongColor {
			p.set(FlagStrongColor, LevelInfo, "")
		}
		if m.Noise != nil && *m.Noise > th.Noisy {
			p.set(FlagNoisy, LevelInfo, "")
		}
		if m.JPEGQ > 0 && m.JPEGQ < th.CompressedQ {
			p.set(FlagCompressed, LevelInfo, noteCompressed(m.JPEGQ))
		}
		if isScreenSize(w, h) && (m.Exif == nil || m.Exif.Make == "") {
			if m.UIBand {
				p.set(FlagScreenshot, LevelWarn, "")
			} else {
				p.set(FlagFromScreen, LevelInfo, noteFromScreen(w, h))
			}
		}
		if m.Bars > th.Bars {
			l := LevelWarn
			if j.kind == KindObject {
				l = LevelInfo
			}
			p.set(FlagBars, l, "")
		}
	}
}

// decide applies "Keep anyway" and "This is intended", then works out
// which photos are usable.
func (j *judge) decide() {
	for _, p := range j.own {
		if p.Pending {
			continue
		}
		var waive []Flag
		switch p.Decision {
		case DecisionKeep:
			for f, l := range p.flags {
				if l == LevelWarn && !trainerUnreadable[f] {
					waive = append(waive, f)
				}
			}
		case DecisionIntended:
			for _, f := range IntendedClears {
				if p.has(f) {
					waive = append(waive, f)
				}
			}
		}
		for _, f := range waive {
			delete(p.flags, f)
			p.waived = append(p.waived, f)
		}
		sortFlags(p.waived)
		p.usable = !p.warned()
	}
}

// warned reports an active warn flag.
func (p *photo) warned() bool {
	for _, l := range p.flags {
		if l == LevelWarn {
			return true
		}
	}
	return false
}

// usableCount is section 7.1: a burst group counts once, a face crop half.
func (j *judge) usableCount() int {
	sum := 0.0
	bursts := map[string]float64{}
	for _, p := range j.own {
		if !p.usable {
			continue
		}
		w := 1.0
		if p.Parent != "" {
			w = 0.5
		}
		if p.burstID != "" {
			bursts[p.burstID] = max(bursts[p.burstID], w)
			continue
		}
		sum += w
	}
	for _, w := range bursts {
		sum += w
	}
	return int(math.Floor(sum + 1e-9))
}

// review assembles the Review.
func (j *judge) review() *Review {
	th := j.th
	r := &Review{
		Kind:           j.kind,
		Resolution:     j.r,
		MetricsVersion: MetricsVersion,
		Items:          []Item{},
		Counts:         map[Flag]int{},
		Groups:         j.groups,
		Findings:       []Finding{},
		Hints:          []string{},
		SetAside:       j.set.SetAside,
		Originals:      j.set.Originals,
		Faces:          j.set.Faces,
		Thresholds:     th.Map(),
		Convert:        j.set.Convert,
	}
	if r.Groups == nil {
		r.Groups = []Group{}
	}
	if r.SetAside == nil {
		r.SetAside = []SetAsideItem{}
	}
	if r.Originals == nil {
		r.Originals = []OriginalItem{}
	}
	if r.Convert == nil {
		r.Convert = []string{}
	}
	r.Faces.Checked = max(r.Faces.Checked, j.checked)
	if ref := j.byHash[j.set.Reference]; j.set.Reference != "" && ref != nil && !ref.Existing {
		r.Reference = ref.Metrics.File
	}
	for _, p := range j.own {
		it := j.item(p)
		it.Reference = r.Reference != "" && p.Metrics.File == r.Reference
		r.Items = append(r.Items, it)
		r.Total++
		if p.Pending {
			r.Pending++
			continue
		}
		if p.warned() {
			r.Flagged++
		}
		for f := range p.flags {
			r.Counts[f]++
		}
	}
	r.Usable = j.usableCount()
	cov := j.coverage()
	r.Coverage = cov
	r.Basis = slices.Clone(cov.Basis)
	j.assess(r)
	return r
}

// item is the Item for one photo.
func (j *judge) item(p *photo) Item {
	m := p.Metrics
	it := Item{
		File:     m.File,
		SHA256:   m.SHA256,
		Width:    m.Width,
		Height:   m.Height,
		Bytes:    m.Bytes,
		Flags:    []Flag{},
		Notes:    map[Flag]string{},
		Decision: p.Decision,
		Parent:   p.Parent,
		Pending:  p.Pending,
	}
	if p.Pending {
		return it
	}
	if m.Width > 0 && m.Height > 0 {
		it.Px = min(int(math.Round(p.edge())), j.r)
	}
	for f, l := range p.flags {
		it.Flags = append(it.Flags, f)
		if l == LevelWarn {
			it.Level = LevelWarn
		} else if it.Level == "" {
			it.Level = LevelInfo
		}
	}
	sortFlags(it.Flags)
	for f, n := range p.notes {
		it.Notes[f] = n
	}
	if len(p.waived) > 0 {
		it.Waived = p.waived
	}
	it.Group, it.Keeper, it.SameAs = p.group, p.keeper, p.sameAs
	if j.med > 0 && m.Decoded {
		it.SharpRel = r4(m.Sharp / j.med)
	}
	if m.Exif != nil {
		it.Captured = m.Exif.Time
	}
	it.Session = p.session
	it.Face = p.fv.itemFace()
	it.Usable = p.usable
	return it
}

func sortFlags(fs []Flag) {
	slices.SortFunc(fs, func(a, b Flag) int { return flagRank[a] - flagRank[b] })
}

// median of a copy of v; 0 when empty.
func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := slices.Clone(v)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}
