package photocheck

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

type kps = [5][2]float64

// Landmarks (fractions of a 3000 x 4000 photo): right eye, left eye, nose,
// mouth corners.
var (
	frontal = kps{{0.45, 0.40}, {0.55, 0.40}, {0.50, 0.45}, {0.46, 0.50}, {0.54, 0.50}}
	quarter = kps{{0.45, 0.40}, {0.55, 0.40}, {0.53, 0.45}, {0.48, 0.50}, {0.55, 0.50}}
	profile = kps{{0.45, 0.40}, {0.55, 0.40}, {0.57, 0.45}, {0.52, 0.50}, {0.56, 0.50}}
)

// rotate turns landmarks by deg about their eye midpoint, in pixels.
func rotate(k kps, deg float64, w, h float64) kps {
	t := deg * math.Pi / 180
	cx, cy := (k[0][0]+k[1][0])/2*w, (k[0][1]+k[1][1])/2*h
	var out kps
	for i, p := range k {
		x, y := p[0]*w-cx, p[1]*h-cy
		out[i] = [2]float64{(x*math.Cos(t) - y*math.Sin(t) + cx) / w, (x*math.Sin(t) + y*math.Cos(t) + cy) / h}
	}
	return out
}

func faceSha(i int) string { return fmt.Sprintf("%064x", 3000+i+1) }

// faceInput is photo i with one embedded face; fh is the box height share.
func faceInput(i int, k kps, fh float64) Input {
	m := good(fmt.Sprintf("f%02d.jpg", i), 3000+i)
	side := int(fh * 4000)
	return Input{Metrics: m, Face: &FaceData{Algo: FaceAlgo, FrameSharp: 300, Faces: []Face{{
		Box: [4]float64{0.5 - fh*0.375, 0.3, fh * 0.75, fh}, Kps: k, Score: 0.9, Side: side, AlignSide: min(side, 400),
		Sharp: 100, MeanY: 120, ClipHi: 0.01, Embedded: true,
	}}}}
}

// faceSet is n face photos: all one yaw at bust size, or ("") a mix of
// yaws and shots.
func faceSet(n int, yaw string) []Input {
	in := make([]Input, n)
	for i := range in {
		switch yaw {
		case "front":
			in[i] = faceInput(i, frontal, 0.3)
		case "profile":
			in[i] = faceInput(i, profile, 0.3)
		default:
			ks := []kps{frontal, quarter, frontal, profile, quarter}
			fhs := []float64{0.5, 0.3, 0.15, 0.09}
			in[i] = faceInput(i, ks[i%len(ks)], fhs[i%len(fhs)])
		}
	}
	return in
}

// toHalf is float32 -> binary16, round to nearest (enough for cosines).
func toHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xff) - 127 + 15
	frac := b & 0x7fffff
	switch {
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint(14 - exp)
		h := uint16(frac >> shift)
		if frac>>(shift-1)&1 == 1 {
			h++
		}
		return sign | h
	case exp >= 31:
		return sign | 0x7c00
	}
	h := sign | uint16(exp)<<10 | uint16(frac>>13)
	if frac&0x1000 != 0 {
		h++
	}
	return h
}

// encodeCos writes an n-face matrix as the face job does.
func encodeCos(n int, cos func(i, j int) float64) string {
	b := make([]byte, 0, n*(n-1))
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			b = binary.LittleEndian.AppendUint16(b, toHalf(float32(cos(i, j))))
		}
	}
	return base64.StdEncoding.EncodeToString(b)
}

// identityOf is the Identity for the first n faceInput photos.
func identityOf(n int, cos func(i, j int) float64) *Identity {
	id := &Identity{Model: FaceModel}
	for i := 0; i < n; i++ {
		id.Faces = append(id.Faces, faceSha(i)+":0")
	}
	id.Cos = encodeCos(n, cos)
	return id
}

// outlierCos: the last k faces are someone else (0.1 to everyone).
func outlierCos(n, k int) func(i, j int) float64 {
	return func(i, j int) float64 {
		if i >= n-k || j >= n-k {
			return 0.1
		}
		return 0.6
	}
}

// groupCos: consecutive groups of the given sizes, within and across.
func groupCos(sizes []int, within, across float64) func(i, j int) float64 {
	groupOf := func(i int) int {
		for g, s := range sizes {
			if i < s {
				return g
			}
			i -= s
		}
		return len(sizes)
	}
	return func(i, j int) float64 {
		if groupOf(i) == groupOf(j) {
			return within
		}
		return across
	}
}

func flagged(r *Review, f Flag) []string {
	var out []string
	for _, it := range r.Items {
		if slices.Contains(it.Flags, f) {
			out = append(out, it.File)
		}
	}
	return out
}

func files(from, to int) []string {
	var out []string
	for i := from; i < to; i++ {
		out = append(out, fmt.Sprintf("f%02d.jpg", i))
	}
	return out
}

// TestIdentity is the section 6.3 rule.
func TestIdentity(t *testing.T) {
	t.Run("one stranger", func(t *testing.T) {
		r := Judge(faceSet(20, "front"), SetState{Identity: identityOf(20, outlierCos(20, 1))}, Options{})
		if got := flagged(r, FlagNotThem); !slices.Equal(got, files(19, 20)) {
			t.Errorf("not_them %v", got)
		}
		it := itemOf(t, r, "f19.jpg")
		if it.Notes[FlagNotThem] != "This face looks different from the rest of the set." || it.Usable {
			t.Errorf("f19 %+v", it)
		}
		if c := itemOf(t, r, "f00.jpg").Face; c == nil || c.Cohesion == nil || math.Abs(*c.Cohesion-0.6) > 0.01 || c.Yaw != "front" {
			t.Errorf("f00 face %+v", c)
		}
		if f := findingOf(r, "identity_outliers"); f == nil || f.Op == nil || f.Op.Op != "set_aside" || !slices.Equal(f.Op.Files, files(19, 20)) {
			t.Errorf("identity_outliers %+v", f)
		}
	})
	t.Run("three siblings", func(t *testing.T) {
		r := Judge(faceSet(20, "front"), SetState{Identity: identityOf(20, groupCos([]int{17, 3}, 0.6, 0.15))}, Options{})
		if got := flagged(r, FlagNotThem); !slices.Equal(got, files(17, 20)) {
			t.Errorf("not_them %v", got)
		}
		if findingOf(r, "two_people") != nil {
			t.Error("3 of 20 is not two people")
		}
	})
	t.Run("two people", func(t *testing.T) {
		r := Judge(faceSet(20, "front"), SetState{Identity: identityOf(20, groupCos([]int{12, 8}, 0.6, 0.1))}, Options{})
		if got := flagged(r, FlagNotThem); len(got) != 0 {
			t.Errorf("not_them %v", got)
		}
		if got := flagged(r, FlagIdentityCheck); !slices.Equal(got, files(12, 20)) {
			t.Errorf("identity_check %v", got)
		}
		if n := itemOf(t, r, "f15.jpg").Notes[FlagIdentityCheck]; n != "Hard to compare (two people in the set). Make sure it is them." {
			t.Errorf("note %q", n)
		}
		f := findingOf(r, "two_people")
		if f == nil || f.Message != "These photos seem to show two different people (12 and 8 photos)." {
			t.Errorf("two_people %+v", f)
		}
	})
	t.Run("profiles are never not_them", func(t *testing.T) {
		r := Judge(faceSet(20, "profile"), SetState{Identity: identityOf(20, func(int, int) float64 { return 0.3 })}, Options{})
		if got := flagged(r, FlagNotThem); len(got) != 0 {
			t.Errorf("not_them %v", got)
		}
		if n := itemOf(t, r, "f05.jpg").Notes[FlagIdentityCheck]; !strings.Contains(n, "a side view") {
			t.Errorf("side view note %q", n)
		}
	})
	t.Run("a reference picks the main component", func(t *testing.T) {
		set := SetState{Identity: identityOf(20, groupCos([]int{12, 8}, 0.6, 0.1)), Reference: faceSha(15)}
		r := Judge(faceSet(20, "front"), set, Options{})
		if got := flagged(r, FlagNotThem); !slices.Equal(got, files(0, 12)) {
			t.Errorf("not_them %v", got)
		}
		if findingOf(r, "two_people") != nil || r.Reference != "f15.jpg" || !itemOf(t, r, "f15.jpg").Reference {
			t.Errorf("reference %q, findings %v", r.Reference, ids(r))
		}
		it := itemOf(t, r, "f00.jpg")
		if it.Notes[FlagNotThem] != "This face looks different from the rest of the set and from your reference photo." ||
			it.Face.RefSim == nil || math.Abs(*it.Face.RefSim-0.1) > 0.01 {
			t.Errorf("f00 %+v", it)
		}
	})
	t.Run("a face too small to embed", func(t *testing.T) {
		in := faceSet(11, "front")
		in[10].Face.Faces[0].AlignSide, in[10].Face.Faces[0].Embedded = 30, false
		r := Judge(in, SetState{Identity: identityOf(10, outlierCos(10, 0))}, Options{})
		it := itemOf(t, r, "f10.jpg")
		if it.Notes[FlagIdentityCheck] != "Hard to compare (a small face). Make sure it is them." || slices.Contains(it.Flags, FlagNotThem) {
			t.Errorf("f10 %+v", it)
		}
		if it.Face.Yaw != "" {
			t.Errorf("yaw %q at align side 30", it.Face.Yaw)
		}
	})
	t.Run("few faces: only the reference", func(t *testing.T) {
		in := faceSet(4, "front")
		r := Judge(in, SetState{Identity: identityOf(4, outlierCos(4, 1))}, Options{})
		if got := flagged(r, FlagNotThem); len(got) != 0 {
			t.Errorf("no reference, 4 faces: %v", got)
		}
		r = Judge(in, SetState{Identity: identityOf(4, outlierCos(4, 1)), Reference: faceSha(0)}, Options{})
		if got := flagged(r, FlagNotThem); !slices.Equal(got, files(3, 4)) {
			t.Errorf("with a reference: %v", got)
		}
	})
	t.Run("MAD scaling", func(t *testing.T) {
		in := faceSet(25, "front")
		total := 0
		for seed := 0; seed < 200; seed++ {
			rng := rand.New(rand.NewPCG(uint64(seed), 1))
			v := make([][]float64, 25)
			for i := range v {
				v[i] = make([]float64, 25)
				for j := i + 1; j < 25; j++ {
					v[i][j] = math.Max(-1, math.Min(1, 0.6+0.05*rng.NormFloat64()))
				}
			}
			r := Judge(in, SetState{Identity: identityOf(25, func(i, j int) float64 { return v[i][j] })}, Options{})
			total += len(flagged(r, FlagNotThem))
		}
		if total != 0 {
			t.Errorf("%d not_them over 200 genuine sets", total)
		}
	})
	t.Run("one person", func(t *testing.T) {
		r := Judge(faceSet(8, "front"), SetState{Identity: identityOf(8, outlierCos(8, 0))}, Options{})
		if len(flagged(r, FlagNotThem))+len(flagged(r, FlagIdentityCheck)) != 0 {
			t.Errorf("a clean set: %v %v", flagged(r, FlagNotThem), flagged(r, FlagIdentityCheck))
		}
		if !slices.Contains(r.Basis, "faces") || r.Faces.Checked != 8 {
			t.Errorf("basis %v checked %d", r.Basis, r.Faces.Checked)
		}
	})
}

// TestIdentityMatrix covers DecodeCos and SetSig.
func TestIdentityMatrix(t *testing.T) {
	cos := func(i, j int) float64 { return float64(i*10+j) / 100 }
	id := identityOf(5, cos)
	v, err := DecodeCos(id.FaceMatrix)
	if err != nil || len(v) != 10 {
		t.Fatalf("%v %d", err, len(v))
	}
	m := newCosMatrix(id)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			want := 1.0
			if i < j {
				want = cos(i, j)
			} else if i > j {
				want = cos(j, i)
			}
			if got := m.cos(i, j); math.Abs(got-want) > 1e-3 {
				t.Errorf("cos(%d,%d) = %v, want %v", i, j, got, want)
			}
		}
	}
	for _, h := range []struct {
		bits uint16
		want float32
	}{{0x3c00, 1}, {0xbc00, -1}, {0x3800, 0.5}, {0x0000, 0}, {0x0001, 5.9604645e-8}, {0x7bff, 65504}} {
		if got := half(h.bits); got != h.want {
			t.Errorf("half(%#x) = %v, want %v", h.bits, got, h.want)
		}
	}
	if _, err := DecodeCos(FaceMatrix{Faces: []string{"a", "b", "c"}, Cos: base64.StdEncoding.EncodeToString([]byte{1, 2})}); err == nil {
		t.Error("a short matrix decoded")
	}
	if _, err := DecodeCos(FaceMatrix{Faces: []string{"a"}, Cos: "!!"}); err == nil {
		t.Error("bad base64 decoded")
	}
	a, b := SetSig([]string{"b", "a", "c"}), SetSig([]string{"c", "b", "a"})
	if a != b || len(a) != 64 || a == SetSig([]string{"a", "b"}) {
		t.Errorf("SetSig %s %s", a, b)
	}
}

// TestYawShot covers yaw from landmarks and shot from the box height.
func TestYawShot(t *testing.T) {
	th := DefaultThresholds()
	f := func(k kps) Face { return Face{Kps: k, Score: 0.9, AlignSide: 200} }
	for name, c := range map[string]struct {
		k    kps
		want string
	}{
		"frontal":       {frontal, "front"},
		"three-quarter": {quarter, "three_quarter"},
		"profile":       {profile, "profile"},
		"rolled 30":     {rotate(frontal, 30, 3000, 4000), "front"},
		"rolled -25":    {rotate(quarter, -25, 3000, 4000), "three_quarter"},
		"nose past eye": {kps{{0.45, 0.4}, {0.55, 0.4}, {0.40, 0.45}, {0.45, 0.5}, {0.5, 0.5}}, "profile"},
	} {
		if got := yawOf(f(c.k), 3000, 4000, th); got != c.want {
			t.Errorf("%s: %q, want %q", name, got, c.want)
		}
	}
	low := f(frontal)
	low.Score = 0.65
	small := f(frontal)
	small.AlignSide = 39
	if yawOf(low, 3000, 4000, th) != "" || yawOf(small, 3000, 4000, th) != "" {
		t.Error("yaw known on a weak or small face")
	}
	for fh, want := range map[float64]string{0.09: "full", 0.15: "half", 0.3: "bust", 0.5: "close", 0.42: "close", 0.11: "half"} {
		if got := shotOf(fh, th); got != want {
			t.Errorf("fh %v: %s, want %s", fh, got, want)
		}
	}
	// Through the judge, and "This is intended" with no face is a back view.
	in := []Input{faceInput(0, frontal, 0.09), faceInput(1, quarter, 0.5), faceInput(2, frontal, 0.3)}
	in[2].Face.Faces = nil
	in[2].Decision = DecisionIntended
	r := Judge(in, SetState{}, Options{})
	if s := itemOf(t, r, "f00.jpg").Face; s.Shot != "full" || s.Yaw != "front" || s.Px != 106 || s.Count != 1 {
		t.Errorf("f00 %+v", s)
	}
	b := itemOf(t, r, "f02.jpg")
	if b.Face.Shot != "back" || !slices.Equal(b.Waived, []Flag{FlagNoFace}) || len(b.Flags) != 0 {
		t.Errorf("back view %+v %+v", b.Face, b)
	}
	if r.Coverage.Shot["full"] != 1 || r.Coverage.Shot["close"] != 1 || r.Coverage.Yaw["three_quarter"] != 1 {
		t.Errorf("coverage %+v", r.Coverage)
	}
}

// TestOthers covers the others flag and its mirror exception.
func TestOthers(t *testing.T) {
	two := func(cos, score float64) []Input {
		in := faceInput(0, frontal, 0.3)
		s := in.Face.Faces[0]
		o := s
		o.Box = [4]float64{0.1, 0.3, 0.1, 0.15}
		o.Side, o.AlignSide, o.Score = s.Side/2, s.Side/2, score
		in.Face.Faces = append(in.Face.Faces, o)
		return []Input{in}
	}
	id := func(cos float64) *Identity {
		return &Identity{FaceMatrix: FaceMatrix{Faces: []string{faceSha(0) + ":0", faceSha(0) + ":1"},
			Cos: encodeCos(2, func(int, int) float64 { return cos })}}
	}
	r := Judge(two(0.1, 0.9), SetState{Identity: id(0.1)}, Options{})
	it := r.Items[0]
	if !slices.Contains(it.Flags, FlagOthers) || it.Face.Count != 2 || it.Face.Box == nil || (*it.Face.Box)[0] != 0.5-0.3*0.375 {
		t.Errorf("a second person: %v %+v", it.Flags, it.Face)
	}
	if f := findingOf(r, "to_look_at"); f == nil || f.Message != "1 photo to look at: 1 with other people." {
		t.Errorf("to_look_at %+v", f)
	}
	if r := Judge(two(0.7, 0.9), SetState{Identity: id(0.7)}, Options{}); slices.Contains(r.Items[0].Flags, FlagOthers) {
		t.Error("a mirror image counted as another person")
	}
	if r := Judge(two(0.1, 0.7), SetState{Identity: id(0.1)}, Options{}); slices.Contains(r.Items[0].Flags, FlagOthers) {
		t.Error("a weak second face counted")
	}
	// No face at all: a warn for many, info for one or two in a big set.
	in := faceSet(20, "front")
	in[0].Face.Faces = nil
	r = Judge(in, SetState{}, Options{})
	if it := itemOf(t, r, "f00.jpg"); !slices.Contains(it.Flags, FlagNoFace) || it.Level != LevelInfo {
		t.Errorf("one no_face in 20: %v %s", it.Flags, it.Level)
	}
	for i := 0; i < 5; i++ {
		in[i].Face.Faces = nil
	}
	if it := itemOf(t, Judge(in, SetState{}, Options{}), "f00.jpg"); it.Level != LevelWarn {
		t.Errorf("five no_face in 20: level %s", it.Level)
	}
}

// TestWatermark covers text boxes: a repeated or small corner box is a
// watermark, a single scene box is text.
func TestWatermark(t *testing.T) {
	in := faceSet(5, "front")
	for i := range in {
		in[i].Face.TextChecked = true
	}
	for i := 0; i < 3; i++ {
		in[i].Face.Text = []TextBox{{0.30 + 0.01*float64(i), 0.05, 0.40, 0.06, 0.9}}
	}
	in[3].Face.Text = []TextBox{{0.40, 0.45, 0.20, 0.10, 0.9}}
	in[4].Face.Text = []TextBox{{0.88, 0.90, 0.10, 0.05, 0.8}, {0.4, 0.4, 0.1, 0.1, 0.3}}
	r := Judge(in, SetState{}, Options{})
	for i := 0; i < 3; i++ {
		it := itemOf(t, r, fmt.Sprintf("f%02d.jpg", i))
		if !slices.Contains(it.Flags, FlagWatermark) ||
			it.Notes[FlagWatermark] != "A watermark or overlay text in the same spot as in 2 other photos. The LoRA can learn to draw it." {
			t.Errorf("repeated box %d: %v %q", i, it.Flags, it.Notes[FlagWatermark])
		}
	}
	if it := itemOf(t, r, "f03.jpg"); !slices.Equal(it.Flags, []Flag{FlagText}) || it.Level != LevelInfo {
		t.Errorf("scene text: %v", it.Flags)
	}
	if it := itemOf(t, r, "f04.jpg"); !slices.Contains(it.Flags, FlagWatermark) || slices.Contains(it.Flags, FlagText) ||
		it.Notes[FlagWatermark] != "A watermark or overlay text. The LoRA can learn to draw it." {
		t.Errorf("corner box: %v %v", it.Flags, it.Notes)
	}
	in[3].Face.TextChecked = false
	if it := itemOf(t, Judge(in, SetState{}, Options{}), "f03.jpg"); len(it.Flags) != 0 {
		t.Errorf("unchecked text read: %v", it.Flags)
	}
}

// TestIdentityFaceFlags covers small_face, face_cut, face_blurry, backlit,
// smoothed and the face-lit dark and bright.
func TestIdentityFaceFlags(t *testing.T) {
	in := faceSet(10, "front")
	in[0].Face.Faces[0].Side = 150 // fpx 44
	in[1].Face.Faces[0].Box = [4]float64{0.0, 0.3, 0.2, 0.3}
	in[1].Face.Faces[0].Kps[0] = [2]float64{0.02, 0.4}
	in[2].Face.Faces[0].Sharp = 5
	in[3].Face.Faces[0].MeanY = 40
	in[4].Face.Faces[0].Sharp = 30 // smoothed: a soft face in a sharp frame
	in[5].Metrics.LumaMean = 20    // the frame is dark, the face is lit
	in[6].Face.Faces[0].ClipHi = 0.3
	in[7].Metrics.Sharp = 2 // a soft frame with a sharp face is not blurry
	r := Judge(in, SetState{}, Options{})
	want := map[string][]Flag{
		"f00.jpg": {FlagSmallFace},
		"f01.jpg": {FlagFaceCut},
		"f02.jpg": {FlagFaceBlurry, FlagSmoothed},
		"f03.jpg": {FlagDark, FlagBacklit},
		"f04.jpg": {FlagSmoothed},
		"f05.jpg": {},
		"f06.jpg": {FlagBright},
		"f07.jpg": {},
	}
	for f, w := range want {
		it := itemOf(t, r, f)
		if !slices.Equal(it.Flags, w) {
			t.Errorf("%s: %v, want %v (%v)", f, it.Flags, w, it.Notes)
		}
	}
	if n := itemOf(t, r, "f03.jpg").Notes[FlagDark]; n != "The face is in deep shadow." {
		t.Errorf("dark note %q", n)
	}
	if n := itemOf(t, r, "f00.jpg").Notes[FlagSmallFace]; n != "The face trains at only about 44 px across, too few to learn it from." {
		t.Errorf("small face note %q", n)
	}
}
