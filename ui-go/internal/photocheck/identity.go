package photocheck

import (
	"cmp"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// DecodeCos reads FaceMatrix.Cos: n(n-1)/2 little-endian binary16 values,
// the upper triangle (i < j) row-major, for n = len(m.Faces).
func DecodeCos(m FaceMatrix) ([]float32, error) {
	b, err := base64.StdEncoding.DecodeString(m.Cos)
	if err != nil {
		return nil, fmt.Errorf("face matrix: %w", err)
	}
	n := len(m.Faces)
	want := n * (n - 1) / 2
	if len(b) != 2*want {
		return nil, fmt.Errorf("face matrix: %d bytes for %d faces, want %d", len(b), n, 2*want)
	}
	out := make([]float32, want)
	for i := range out {
		out[i] = half(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out, nil
}

// half converts an IEEE 754 binary16 to float32.
func half(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal
		v := float32(frac) / 1024 / 16384
		if sign != 0 {
			return -v
		}
		return v
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | frac<<13)
}

// SetSig is the face check's set signature: sha256 over the sorted photo
// shas, one per line. A face check is stale when the set's sig changes.
func SetSig(shas []string) string {
	s := slices.Clone(shas)
	slices.Sort(s)
	sum := sha256.Sum256([]byte(strings.Join(s, "\n")))
	return hex.EncodeToString(sum[:])
}

// cosMatrix looks up face-to-face cosines.
type cosMatrix struct {
	n   int
	idx map[string]int // "<sha>:<face index>" -> row
	v   []float32
}

func newCosMatrix(id *Identity) *cosMatrix {
	if id == nil || len(id.Faces) == 0 {
		return nil
	}
	v, err := DecodeCos(id.FaceMatrix)
	if err != nil {
		return nil
	}
	m := &cosMatrix{n: len(id.Faces), idx: make(map[string]int, len(id.Faces)), v: v}
	for i, k := range id.Faces {
		m.idx[k] = i
	}
	return m
}

// row is the matrix row of one face, or -1.
func (m *cosMatrix) row(sha string, face int) int {
	if m == nil || sha == "" {
		return -1
	}
	if i, ok := m.idx[sha+":"+strconv.Itoa(face)]; ok {
		return i
	}
	return -1
}

func (m *cosMatrix) cos(a, b int) float64 {
	if a == b {
		return 1
	}
	if a > b {
		a, b = b, a
	}
	return float64(m.v[a*(m.n-1)-a*(a-1)/2+b-a-1])
}

// faceView is one photo's faces as the judge reads them.
type faceView struct {
	checked bool // the photo has face data
	data    *FaceData
	cands   []int // faces that count: score and height (6.2)
	subj    int   // the subject face, -1 for none
	row     int   // the subject's matrix row, -1 when not embedded
	fpx     int   // subject side at training size
	shot    string
	yaw     string
	cohes   *float64
	refSim  *float64
}

func (f *faceView) subjectFace() *Face {
	if f == nil || !f.checked || f.subj < 0 {
		return nil
	}
	return &f.data.Faces[f.subj]
}

func (f *faceView) itemFace() *ItemFace {
	if f == nil || !f.checked {
		return nil
	}
	it := &ItemFace{Count: len(f.cands), Shot: f.shot, Yaw: f.yaw, Cohesion: f.cohes, RefSim: f.refSim}
	if s := f.subjectFace(); s != nil {
		it.Px = f.fpx
		b := s.Box
		it.Box = &b
	}
	return it
}

// identity is the set-level face result.
type identity struct {
	n         int  // embedded subject faces
	refSet    bool // a usable reference face
	twoPeople bool
	main      int // faces in the main component
	second    int // faces in the second-largest component
}

// faces works out subjects, shot, yaw and identity (section 6), and sets
// the face flags (4.5).
func (j *judge) faces() {
	th := j.th
	mx := newCosMatrix(j.set.Identity)
	var fvs []*photo
	for _, p := range j.own {
		if p.Pending || p.Face == nil || p.Face.Error != "" {
			continue
		}
		fv := &faceView{checked: true, data: p.Face, subj: -1, row: -1}
		for i, f := range p.Face.Faces {
			if f.Score >= th.FaceScore && f.Box[3] >= th.FaceHeight {
				fv.cands = append(fv.cands, i)
			}
		}
		p.fv = fv
		fvs = append(fvs, p)
	}
	if len(fvs) == 0 {
		return
	}
	embRow := func(p *photo, i int) int {
		if !p.Face.Faces[i].Embedded || p.Face.Faces[i].AlignSide < th.EmbedMinAlign {
			return -1
		}
		return mx.row(p.Metrics.SHA256, i)
	}
	largestEmb := func(p *photo) int {
		best, side := -1, -1
		for _, i := range p.fv.cands {
			if r := embRow(p, i); r >= 0 && p.Face.Faces[i].Side > side {
				best, side = r, p.Face.Faces[i].Side
			}
		}
		return best
	}

	// Medoid: the reference photo's largest embedded face, else the
	// largest face with the highest median cosine to the others'.
	medoid := -1
	if ref := j.byHash[j.set.Reference]; j.set.Reference != "" && ref != nil && ref.fv != nil {
		medoid = largestEmb(ref)
	}
	j.id.refSet = medoid >= 0
	if medoid < 0 {
		var ls []int
		for _, p := range fvs {
			if r := largestEmb(p); r >= 0 {
				ls = append(ls, r)
			}
		}
		bestMed := math.Inf(-1)
		for _, a := range ls {
			var cs []float64
			for _, b := range ls {
				if b != a {
					cs = append(cs, mx.cos(a, b))
				}
			}
			if m := median(cs); len(cs) == 0 || m > bestMed {
				medoid, bestMed = a, m
				if len(cs) == 0 {
					break
				}
			}
		}
	}

	// Subject per photo, then fpx, shot and yaw.
	for _, p := range fvs {
		fv := p.fv
		if len(fv.cands) == 0 {
			if p.Decision == DecisionIntended {
				fv.shot = "back"
			}
			continue
		}
		largest := 0
		for _, i := range fv.cands {
			largest = max(largest, p.Face.Faces[i].Side)
		}
		bestCos, bestSide := math.Inf(-1), -1
		for _, i := range fv.cands {
			f := p.Face.Faces[i]
			if float64(f.Side) < th.SubjectSide*float64(largest) {
				continue
			}
			if r := embRow(p, i); r >= 0 && medoid >= 0 {
				if c := mx.cos(r, medoid); c > bestCos {
					fv.subj, fv.row, bestCos = i, r, c
				}
			}
		}
		if fv.subj < 0 {
			for _, i := range fv.cands {
				if s := p.Face.Faces[i].Side; s > bestSide {
					fv.subj, bestSide = i, s
				}
			}
		}
		s := p.Face.Faces[fv.subj]
		edge := p.edge()
		scale := 1.0
		if edge > 0 {
			scale = math.Min(1, float64(j.r)/edge)
		}
		fv.fpx = int(math.Round(float64(s.Side) * scale))
		fv.shot = shotOf(s.Box[3], th)
		fv.yaw = yawOf(s, p.Metrics.Width, p.Metrics.Height, th)
	}

	j.identityRule(fvs, mx, medoid)
	j.faceFlags(fvs, mx)
}

// shotOf maps the subject box height share to a shot type (4.5).
func shotOf(fh float64, th Thresholds) string {
	switch {
	case fh >= th.ShotClose:
		return "close"
	case fh >= th.ShotBust:
		return "bust"
	case fh >= th.ShotHalf:
		return "half"
	}
	return "full"
}

// yawOf is the head turn from the 5 landmarks with roll removed (4.5); ""
// when unknown.
func yawOf(f Face, w, h int, th Thresholds) string {
	if f.AlignSide < th.YawMinAlign || f.Score < th.YawMinScore {
		return ""
	}
	W, H := float64(max(w, 1)), float64(max(h, 1))
	rx, ry := f.Kps[0][0]*W, f.Kps[0][1]*H // right eye (image left)
	lx, ly := f.Kps[1][0]*W, f.Kps[1][1]*H // left eye
	nx, ny := f.Kps[2][0]*W, f.Kps[2][1]*H // nose
	t := math.Atan2(ly-ry, lx-rx)
	mxp, myp := (rx+lx)/2, (ry+ly)/2
	c, s := math.Cos(-t), math.Sin(-t)
	rotX := func(x, y float64) float64 { return (x-mxp)*c - (y-myp)*s }
	er, el, nose := rotX(rx, ry), rotX(lx, ly), rotX(nx, ny)
	if el-er == 0 {
		return ""
	}
	r := (nose - er) / (el - er)
	switch {
	case r < 0 || r > 1:
		return "profile"
	case math.Abs(r-0.5) < th.YawFront:
		return "front"
	case math.Abs(r-0.5) < th.YawQuarter:
		return "three_quarter"
	}
	return "profile"
}

// identityRule is section 6.3.
func (j *judge) identityRule(fvs []*photo, mx *cosMatrix, ref int) {
	th := j.th
	var subj []*photo
	for _, p := range fvs {
		if p.fv.row >= 0 {
			subj = append(subj, p)
		}
	}
	n := len(subj)
	j.id.n = n
	refSet := j.id.refSet
	for _, p := range subj {
		if refSet {
			v := r4(mx.cos(p.fv.row, ref))
			p.fv.refSim = &v
		}
	}
	setRule := n >= th.IdentityMin

	// Components over cos >= SameCos.
	uf := newUnionFind(n)
	for a := 0; a < n; a++ {
		for b := a + 1; b < n; b++ {
			if mx.cos(subj[a].fv.row, subj[b].fv.row) >= th.SameCos {
				uf.union(a, b)
			}
		}
	}
	size := map[int]int{}
	for i := range subj {
		size[uf.find(i)]++
	}
	holds := func(row int) int {
		for i, p := range subj {
			if p.fv.row == row {
				return uf.find(i)
			}
		}
		return -1
	}
	main := -1
	if refSet {
		main = holds(ref)
	}
	if main < 0 {
		med := holds(ref)
		for i := range subj {
			r := uf.find(i)
			if main < 0 || size[r] > size[main] || (size[r] == size[main] && r == med && main != med) {
				main = r
			}
		}
	}
	for r, s := range size {
		if r != main {
			j.id.second = max(j.id.second, s)
		}
	}
	if main >= 0 {
		j.id.main = size[main]
	}
	j.id.twoPeople = setRule && !refSet && j.id.second >= max(3, int(math.Ceil(th.TwoPeople*float64(n))))

	// Cohesion c_i and its robust spread.
	c := make([]float64, n)
	for i, p := range subj {
		var cs []float64
		for k, o := range subj {
			if k != i {
				cs = append(cs, mx.cos(p.fv.row, o.fv.row))
			}
		}
		c[i] = median(cs)
		if len(cs) > 0 {
			v := r4(c[i])
			p.fv.cohes = &v
		}
	}
	m := median(c)
	dev := make([]float64, n)
	for i := range c {
		dev[i] = math.Abs(c[i] - m)
	}
	d := math.Max(1.4826*median(dev), th.MADFloor)

	for i, p := range subj {
		fv := p.fv
		var top []float64
		for k, o := range subj {
			if k != i && uf.find(k) == main {
				top = append(top, mx.cos(fv.row, o.fv.row))
			}
		}
		slices.SortFunc(top, func(a, b float64) int { return cmp.Compare(b, a) })
		top = top[:min(3, len(top))]
		absLow := false
		if len(top) > 0 {
			sum := 0.0
			for _, v := range top {
				sum += v
			}
			absLow = sum/float64(len(top)) < th.SameCos
		}
		refLow := refSet && fv.row != ref && mx.cos(fv.row, ref) < th.SameCos
		apart := uf.find(i) != main
		outlier := c[i] < math.Max(th.OutlierFloor, m-th.OutlierK*d)
		if !setRule {
			// too few faces to compare within the set: the reference only
			absLow, apart, outlier = refLow, false, false
		}
		rel := outlier || refLow || apart
		profile := fv.yaw == "profile"
		if !j.id.twoPeople && fv.yaw != "" && !profile && absLow && rel {
			p.set(FlagNotThem, LevelWarn, noteNotThem(refLow))
			continue
		}
		why := ""
		switch {
		case j.id.twoPeople && apart:
			why = "two people in the set"
		case profile && (rel || absLow):
			why = "a side view"
		case rel != absLow, fv.yaw == "" && rel && absLow:
			why = "on the edge of the set"
		}
		if why != "" {
			p.set(FlagIdentityCheck, LevelInfo, noteIdentityCheck(why))
		}
	}
	// A subject too small to embed cannot be compared.
	if setRule || refSet {
		for _, p := range fvs {
			s := p.fv.subjectFace()
			if s != nil && p.fv.row < 0 {
				p.set(FlagIdentityCheck, LevelInfo, noteIdentityCheck("a small face"))
			}
		}
	}
}

// faceFlags are the per-photo face flags other than identity.
func (j *judge) faceFlags(fvs []*photo, mx *cosMatrix) {
	th := j.th
	var faceSharp, smoothRatio []float64
	for _, p := range fvs {
		if s := p.fv.subjectFace(); s != nil {
			faceSharp = append(faceSharp, s.Sharp)
			if float64(s.AlignSide) >= th.SmoothedSide && p.Face.FrameSharp > 0 {
				smoothRatio = append(smoothRatio, s.Sharp/p.Face.FrameSharp)
			}
		}
	}
	medFace := 0.0
	if len(faceSharp) >= th.SetMin {
		medFace = median(faceSharp)
	}
	medSmooth := 0.0
	if len(smoothRatio) >= th.SetMin {
		medSmooth = median(smoothRatio)
	}
	noFace := 0
	for _, p := range fvs {
		if p.fv.subj < 0 {
			noFace++
		}
	}
	noFaceLevel := LevelWarn
	if noFace <= 2 && float64(noFace) <= 0.10*float64(len(fvs)) {
		noFaceLevel = LevelInfo
	}

	for _, p := range fvs {
		fv := p.fv
		s := fv.subjectFace()
		if s == nil {
			p.set(FlagNoFace, noFaceLevel, noteNoFace())
			continue
		}
		subjRow := fv.row
		for i, f := range p.Face.Faces {
			if i == fv.subj || f.Score < th.OthersScore || float64(f.Side) < th.OthersSide*float64(s.Side) {
				continue
			}
			// a cosine at or above SameCos is a mirror or a reflection
			if r := mx.row(p.Metrics.SHA256, i); subjRow >= 0 && r >= 0 && f.Embedded && mx.cos(subjRow, r) >= th.SameCos {
				continue
			}
			p.set(FlagOthers, LevelWarn, "")
			break
		}
		if float64(fv.fpx) < th.SmallFace {
			p.set(FlagSmallFace, LevelWarn, noteSmallFace(fv.fpx))
		}
		if faceCut(s, th) {
			p.set(FlagFaceCut, LevelInfo, "")
		}
		// a face in deep shadow is flagged dark; its sharpness cannot be judged
		soft := s.Sharp < th.FaceFloor && s.MeanY >= th.DarkFace
		if medFace > 0 {
			soft = soft && s.Sharp < th.FaceBlurRel*medFace
		}
		if soft {
			rel := 0.0
			if medFace > 0 {
				rel = s.Sharp / medFace
			}
			p.set(FlagFaceBlurry, LevelWarn, noteFaceBlurry(rel))
		}
		if p.Metrics.Decoded && s.MeanY < th.BacklitRel*p.Metrics.LumaMean && s.MeanY < th.BacklitMax {
			p.set(FlagBacklit, LevelInfo, "")
		}
		if medSmooth > 0 && float64(s.AlignSide) >= th.SmoothedSide && p.Face.FrameSharp > 0 &&
			s.Sharp/p.Face.FrameSharp < th.SmoothedRel*medSmooth {
			p.set(FlagSmoothed, LevelInfo, "")
		}
	}
	j.textFlags(fvs)
}

// faceCut: the subject's box is within FaceCutBox of an edge and a
// landmark within FaceCutKps of the same edge.
func faceCut(f *Face, th Thresholds) bool {
	x, y, w, h := f.Box[0], f.Box[1], f.Box[2], f.Box[3]
	near := func(box bool, kp func(k [2]float64) bool) bool {
		if !box {
			return false
		}
		for _, k := range f.Kps {
			if kp(k) {
				return true
			}
		}
		return false
	}
	b, k := th.FaceCutBox, th.FaceCutKps
	return near(x <= b, func(p [2]float64) bool { return p[0] <= k }) ||
		near(x+w >= 1-b, func(p [2]float64) bool { return p[0] >= 1-k }) ||
		near(y <= b, func(p [2]float64) bool { return p[1] <= k }) ||
		near(y+h >= 1-b, func(p [2]float64) bool { return p[1] >= 1-k })
}

// textFlags are text (info) and watermark (warn) from the job's text
// boxes: a watermark repeats in the same spot, or is a small corner box.
func (j *judge) textFlags(fvs []*photo) {
	th := j.th
	boxes := map[*photo][]TextBox{}
	var with []*photo
	for _, p := range fvs {
		if !p.Face.TextChecked {
			continue
		}
		for _, b := range p.Face.Text {
			if b[4] >= th.TextScore {
				boxes[p] = append(boxes[p], b)
			}
		}
		if len(boxes[p]) > 0 {
			with = append(with, p)
		}
	}
	for _, p := range with {
		repeat, corner := 0, false
		for _, b := range boxes[p] {
			n := 0
			for _, q := range with {
				if q == p {
					continue
				}
				for _, o := range boxes[q] {
					if iou(b, o) >= th.WatermarkIoU {
						n++
						break
					}
				}
			}
			repeat = max(repeat, n)
			x, y, w, h := b[0], b[1], b[2], b[3]
			cs := th.CornerShare
			inX := x+w <= cs || x >= 1-cs
			inY := y+h <= cs || y >= 1-cs
			corner = corner || (inX && inY && h < th.CornerHeight)
		}
		switch {
		case repeat >= th.WatermarkRep:
			p.set(FlagWatermark, LevelWarn, noteWatermark(repeat))
		case corner:
			p.set(FlagWatermark, LevelWarn, noteWatermark(0))
		default:
			p.set(FlagText, LevelInfo, "")
		}
	}
}

// iou of two x, y, w, h boxes.
func iou(a, b TextBox) float64 {
	x0, y0 := math.Max(a[0], b[0]), math.Max(a[1], b[1])
	x1, y1 := math.Min(a[0]+a[2], b[0]+b[2]), math.Min(a[1]+a[3], b[1]+b[3])
	if x1 <= x0 || y1 <= y0 {
		return 0
	}
	in := (x1 - x0) * (y1 - y0)
	u := a[2]*a[3] + b[2]*b[3] - in
	if u <= 0 {
		return 0
	}
	return in / u
}
