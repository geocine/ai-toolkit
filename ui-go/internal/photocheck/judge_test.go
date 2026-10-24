package photocheck

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/color"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/testutil"
)

// good is a sharp, well exposed, colourful 3000 x 4000 camera photo with a
// hash of its own: the judge finds nothing in it.
func good(file string, seed int) Metrics {
	r := rand.New(rand.NewPCG(uint64(seed), 7))
	h := r.Uint64()
	layout := make([]byte, 48)
	for i := range layout {
		layout[i] = byte(r.IntN(256))
	}
	return Metrics{
		Version: MetricsVersion, File: file, SHA256: fmt.Sprintf("%064x", seed+1), Bytes: 4_000_000,
		Format: "jpeg", Width: 3000, Height: 4000, Orientation: 1, Decoded: true, Source: "server", Mode: "rgb",
		DHash: HashHex(h), DHashFlip: HashHex(r.Uint64()), Sharp: 300, LumaMean: 120, LumaStd: 50,
		ClipLo: 0.01, ClipHi: 0.01, ClipHiCenter: 0.01, Colorful: 40, ChromaSpread: 20, Flat: 0.3,
		Layout: base64.StdEncoding.EncodeToString(layout), JPEGQ: 92,
		Exif: &ExifInfo{Make: "Apple", Model: "iPhone 15"},
	}
}

// goodSet is n good photos named p00.jpg, p01.jpg, ...
func goodSet(n int) []Input {
	in := make([]Input, n)
	for i := range in {
		in[i] = Input{Metrics: good(fmt.Sprintf("p%02d.jpg", i), 1000+i)}
	}
	return in
}

func itemOf(t testing.TB, r *Review, file string) Item {
	t.Helper()
	for _, it := range r.Items {
		if it.File == file {
			return it
		}
	}
	t.Fatalf("no item %s", file)
	return Item{}
}

func findingOf(r *Review, id string) *Finding {
	for i := range r.Findings {
		if r.Findings[i].ID == id {
			return &r.Findings[i]
		}
	}
	return nil
}

func ids(r *Review) []string {
	var out []string
	for _, f := range r.Findings {
		out = append(out, f.ID)
	}
	return out
}

// TestJudgeFlags is the section 16 "TestMetrics (flags)" table: each
// fixture is measured, judged with 10 sharp scenes, and gets its flag.
func TestJudgeFlags(t *testing.T) {
	var scenes []Input
	for i := 0; i < 10; i++ {
		scenes = append(scenes, Input{Metrics: measureBytes(t, fmt.Sprintf("s%d.jpg", i),
			testutil.JPEG(testutil.Scene(101+i, 1200, 900), 90))})
	}
	sc := testutil.Scene(1, 1200, 900)
	big := testutil.Scene(2, 3000, 2000)
	cases := []struct {
		name  string
		file  string
		data  []byte
		r     int
		want  []Flag
		never []Flag
	}{
		{"scene", "plain.jpg", testutil.JPEG(sc, 90), 1024, nil,
			[]Flag{FlagBlurry, FlagDark, FlagBright, FlagFlat, FlagGrayscale, FlagBars, FlagSmall, FlagTiny, FlagScreenshot}},
		{"blur", "blur.jpg", testutil.JPEG(testutil.Blur(sc, 3), 90), 1024, []Flag{FlagBlurry}, nil},
		{"dark", "dark.jpg", testutil.JPEG(testutil.Darken(sc), 90), 1024, []Flag{FlagDark}, []Flag{FlagBars}},
		{"bright", "bright.jpg", testutil.JPEG(testutil.Brighten(sc), 90), 1024, []Flag{FlagBright}, nil},
		{"white backdrop", "white.jpg", testutil.JPEG(testutil.WhiteBackdrop(900, 1200), 90), 1024, nil,
			[]Flag{FlagBright, FlagBars}},
		{"flat", "flat.jpg", testutil.JPEG(testutil.Flatten(sc), 90), 1024, []Flag{FlagFlat}, nil},
		{"gray", "gray.jpg", testutil.JPEG(testutil.Gray(sc), 90), 1024, []Flag{FlagGrayscale}, nil},
		{"sepia", "sepia.jpg", testutil.JPEG(testutil.Sepia(sc), 90), 1024, []Flag{FlagGrayscale}, nil},
		{"saturated", "sat.jpg", testutil.JPEG(testutil.Saturate(sc), 90), 1024, []Flag{FlagStrongColor}, nil},
		{"screenshot", "ui.png", testutil.PNG(testutil.UIScreen(1170, 2532)), 1024, []Flag{FlagScreenshot},
			[]Flag{FlagFromScreen}},
		{"screen-size frame", "frame.jpg", testutil.JPEG(testutil.Scene(3, 1920, 1080), 90), 1024,
			[]Flag{FlagFromScreen}, []Flag{FlagScreenshot}},
		{"letterbox", "bars.jpg", testutil.JPEG(testutil.Letterbox(sc, 0.1, color.Black), 90), 1024, []Flag{FlagBars}, nil},
		{"tiny", "tiny.jpg", testutil.JPEG(testutil.Scene(2, 300, 400), 90), 1024, []Flag{FlagTiny}, []Flag{FlagSmall}},
		// PHO section 16 names 700x900, but its edge (794) is above 0.75 x
		// 1024 under the section 4.3 rule; 640x900 (edge 759) is the same
		// case just under it.
		{"small at 1024", "small.jpg", testutil.JPEG(testutil.Scene(2, 640, 900), 90), 1024, []Flag{FlagSmall}, []Flag{FlagTiny}},
		{"small at 768", "small.jpg", testutil.JPEG(testutil.Scene(2, 640, 900), 90), 768, nil, []Flag{FlagSmall, FlagTiny}},
		{"700x900 at 1024", "small.jpg", testutil.JPEG(testutil.Scene(2, 700, 900), 90), 1024, nil, []Flag{FlagSmall, FlagTiny}},
		{"q40", "q40.jpg", testutil.JPEG(sc, 40), 1024, []Flag{FlagCompressed}, nil},
		{"gray16", "g16.png", testutil.Gray16PNG(sc), 1024, []Flag{FlagGray16}, nil},
		{"grey+alpha 16", "ga16.png", testutil.GrayAlpha16PNG(sc), 1024, nil, []Flag{FlagGray16, FlagTransparent}},
		{"transparent", "tr.png", testutil.TransparentPNG(sc), 1024, []Flag{FlagTransparent}, nil},
		{"rotated", "rot.jpg", testutil.WithExif(testutil.JPEG(sc, 90), 6, "2024:05:06 07:08:09", "Apple", "iPhone 15"), 1024,
			[]Flag{FlagRotated}, nil},
		{"generated", "gen.png", testutil.PNGWithText("parameters", "a photo"), 1024, []Flag{FlagGenerated}, nil},
		{"junk", "junk.jpg", []byte("not an image"), 1024, []Flag{FlagUnreadable}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := measureBytes(t, c.file, c.data)
			r := Judge(append(slices.Clone(scenes), Input{Metrics: m}), SetState{}, Options{Resolutions: []int{c.r}})
			it := itemOf(t, r, c.file)
			for _, f := range c.want {
				if !slices.Contains(it.Flags, f) {
					t.Errorf("flags %v, want %s", it.Flags, f)
				}
				if it.Notes[f] == "" {
					t.Errorf("%s has no note", f)
				}
			}
			for _, f := range c.never {
				if slices.Contains(it.Flags, f) {
					t.Errorf("flags %v, must not hold %s (%s)", it.Flags, f, it.Notes[f])
				}
			}
			if c.name == "rotated" && (it.Width != 900 || it.Height != 1200) {
				t.Errorf("rotated: %dx%d, want the upright 900x1200", it.Width, it.Height)
			}
		})
	}
	t.Run("upscaled", func(t *testing.T) {
		up := measureBytes(t, "up.jpg", testutil.JPEG(testutil.Upscaled(big, 2), 90))
		native := measureBytes(t, "native.jpg", testutil.JPEG(big, 90))
		r := Judge(append(slices.Clone(scenes), Input{Metrics: up}, Input{Metrics: native}), SetState{}, Options{})
		u, n := itemOf(t, r, "up.jpg"), itemOf(t, r, "native.jpg")
		if !slices.Contains(u.Flags, FlagUpscaled) || slices.Contains(n.Flags, FlagUpscaled) {
			t.Errorf("upscaled %v, native %v", u.Flags, n.Flags)
		}
		if u.Px != n.Px || u.Px != 1024 {
			t.Errorf("px %d vs native %d", u.Px, n.Px)
		}
	})
	t.Run("scenes are clean", func(t *testing.T) {
		r := Judge(scenes, SetState{}, Options{})
		for _, it := range r.Items {
			if len(it.Flags) > 0 {
				t.Errorf("%s: %v %v", it.File, it.Flags, it.Notes)
			}
		}
	})
}

// TestJudgeReview checks the review's shape and totals.
func TestJudgeReview(t *testing.T) {
	in := goodSet(12)
	in[3].Pending = true
	in = append(in, Input{Metrics: good("stored.jpg", 5000), Existing: true})
	slices.Reverse(in)
	r := Judge(in, SetState{Convert: []string{"bmp"}}, Options{Resolutions: []int{512, 768}})
	if r.Total != 12 || r.Pending != 1 || r.Resolution != 768 || r.Kind != KindPerson || r.MetricsVersion != MetricsVersion {
		t.Errorf("total %d pending %d res %d kind %s", r.Total, r.Pending, r.Resolution, r.Kind)
	}
	if r.Usable != 11 || r.Flagged != 0 {
		t.Errorf("usable %d flagged %d", r.Usable, r.Flagged)
	}
	for i, it := range r.Items {
		if it.File != fmt.Sprintf("p%02d.jpg", i) {
			t.Fatalf("items not in file order: %d is %s", i, it.File)
		}
		if it.Flags == nil || it.Notes == nil {
			t.Fatalf("%s: nil flags or notes", it.File)
		}
	}
	if p := itemOf(t, r, "p03.jpg"); !p.Pending || p.Usable {
		t.Errorf("pending item: %+v", p)
	}
	if findingOf(r, "pending") == nil {
		t.Errorf("no pending finding: %v", ids(r))
	}
	if r.Thresholds["dup_hamming"] != 5 || r.Thresholds["blur_floor"] != 25 || len(r.Thresholds) != len(DefaultThresholds().Map()) {
		t.Errorf("thresholds %v", r.Thresholds)
	}
	if !slices.Equal(r.Convert, []string{"bmp"}) || r.Groups == nil || r.SetAside == nil || r.Hints == nil || r.Findings == nil {
		t.Errorf("convert %v groups %v set_aside %v", r.Convert, r.Groups, r.SetAside)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{`"groups":[]`, `"set_aside":[]`, `"basis":["pixels"]`, `"faces":{"available":false`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("review JSON lacks %s", k)
		}
	}

	empty := Judge(nil, SetState{Convert: []string{"bmp", "gif"}}, Options{})
	if empty.Total != 0 || empty.Grade != "F" || len(empty.Convert) != 2 || findingOf(empty, "count_none") == nil {
		t.Errorf("empty: %+v", empty)
	}
}

// TestJudgeFormat covers the file and format flags and their copy.
func TestJudgeFormat(t *testing.T) {
	heic := Metrics{Version: 1, File: "IMG_1.heic", Bytes: 9, Format: "heic", Width: 4032, Height: 3024, Source: "server",
		Error: "cannot decode heic here"}
	huge := good("huge.png", 1)
	huge.Width, huge.Height, huge.Decoded, huge.Error = 20000, 10000, false, "too large to analyse"
	big := good("big.jpg", 2)
	big.Width, big.Height, big.Decoded, big.Error = 12000, 10000, false, "too large to analyse"
	in := append(goodSet(10),
		Input{Metrics: heic},
		Input{Metrics: huge},
		Input{Metrics: big},
		Input{Metrics: good("a.jpg", 3)},
		Input{Metrics: good("a.png", 4)},
		Input{Metrics: good("a__face.jpg", 5), Parent: "p01.jpg"},
	)
	r := Judge(in, SetState{}, Options{})
	h := itemOf(t, r, "IMG_1.heic")
	if !slices.Equal(h.Flags, []Flag{FlagUnsupported}) || !strings.Contains(h.Notes[FlagUnsupported], "Export it as JPEG") ||
		!strings.HasPrefix(h.Notes[FlagUnsupported], "HEIC files are skipped") {
		t.Errorf("heic: %v %q", h.Flags, h.Notes[FlagUnsupported])
	}
	r2 := Judge(in, SetState{Convert: []string{"heic"}}, Options{})
	if n := itemOf(t, r2, "IMG_1.heic").Notes[FlagUnsupported]; strings.Contains(n, "Export") {
		t.Errorf("a convertible heic still says %q", n)
	}
	if it := itemOf(t, r, "huge.png"); !slices.Contains(it.Flags, FlagHuge) || slices.Contains(it.Flags, FlagUnreadable) ||
		it.Notes[FlagHuge] != "200 MP: the trainer skips images over about 179 MP." {
		t.Errorf("huge: %v %v", it.Flags, it.Notes)
	}
	if it := itemOf(t, r, "big.jpg"); len(it.Flags) != 0 {
		t.Errorf("a 120 MP photo the trainer reads: %v", it.Flags)
	}
	a, b := itemOf(t, r, "a.jpg"), itemOf(t, r, "a.png")
	if !slices.Contains(a.Flags, FlagSameStem) || a.Notes[FlagSameStem] !=
		"a.jpg and a.png have the same name before the extension, so they share one caption." || !slices.Contains(b.Flags, FlagSameStem) {
		t.Errorf("same stem: %v %v", a.Notes, b.Flags)
	}
	if d := itemOf(t, r, "a__face.jpg"); !slices.Contains(d.Flags, FlagDerived) || d.Parent != "p01.jpg" {
		t.Errorf("derived: %+v", d)
	}
	fp := findingOf(r, "format_problems")
	if fp == nil || fp.Op == nil || fp.Op.Op != "convert" ||
		fp.Message != "4 files the trainer reads wrongly or skips (not read by the trainer, too large, sharing a caption)." {
		t.Errorf("format_problems: %+v", fp)
	}
	stem := Judge([]Input{{Metrics: good("x.jpg", 1)}, {Metrics: good("x.webp", 2)}}, SetState{}, Options{})
	if f := findingOf(stem, "format_problems"); f == nil || f.Op.Op != "rename_same_stem" {
		t.Errorf("same stem only: %+v", f)
	}
	stored := Judge([]Input{{Metrics: good("y.png", 1)}, {Metrics: good("y.jpg", 2), Existing: true}}, SetState{}, Options{})
	if it := itemOf(t, stored, "y.png"); !slices.Contains(it.Flags, FlagSameStem) || len(stored.Items) != 1 {
		t.Errorf("an upload beside a stored file with its stem: %v", it.Flags)
	}
}

// TestJudgeDecisions: "Keep anyway" waives warn flags (not unreadable);
// "This is intended" waives its own list.
func TestJudgeDecisions(t *testing.T) {
	in := goodSet(10)
	in[0].Metrics.Width, in[0].Metrics.Height = 300, 400 // tiny
	in[0].Decision = DecisionKeep
	in[1].Metrics.ChromaSpread = 1 // grayscale
	in[1].Decision = DecisionIntended
	in[2].Metrics.Decoded, in[2].Metrics.Error = false, "cannot decode" // unreadable
	in[2].Decision = DecisionKeep
	r := Judge(in, SetState{}, Options{})
	tiny := itemOf(t, r, "p00.jpg")
	if len(tiny.Flags) != 0 || !slices.Equal(tiny.Waived, []Flag{FlagTiny}) || !tiny.Usable || tiny.Notes[FlagTiny] == "" {
		t.Errorf("keep: %+v", tiny)
	}
	gray := itemOf(t, r, "p01.jpg")
	if len(gray.Flags) != 0 || !slices.Equal(gray.Waived, []Flag{FlagGrayscale}) {
		t.Errorf("intended: %+v", gray)
	}
	if bad := itemOf(t, r, "p02.jpg"); !slices.Contains(bad.Flags, FlagUnreadable) || bad.Usable {
		t.Errorf("keep cannot make an unreadable file usable: %+v", bad)
	}
	if r.Usable != 9 || r.Counts[FlagTiny] != 0 {
		t.Errorf("usable %d counts %v", r.Usable, r.Counts)
	}
}

// TestKeeper: in focus first, then the bytes-per-pixel bucket, then
// sharpness; a stored copy always keeps.
func TestKeeper(t *testing.T) {
	sc := testutil.Scene(5, 1600, 1200)
	q95 := measureBytes(t, "q95.jpg", testutil.JPEG(sc, 95))
	q40 := measureBytes(t, "q40.jpg", testutil.JPEG(sc, 40))
	soft := measureBytes(t, "soft.jpg", testutil.JPEG(testutil.Blur(sc, 1.5), 100))
	// JPEG blocking raises the Laplacian; the bucket must win over it.
	q40.Sharp = max(q40.Sharp, q95.Sharp*1.2)
	r := Judge([]Input{{Metrics: q40}, {Metrics: q95}}, SetState{}, Options{})
	if len(r.Groups) != 1 || r.Groups[0].Kind != "duplicate" || r.Groups[0].Keep != "q95.jpg" || !r.Groups[0].Confirmed {
		t.Fatalf("groups %+v", r.Groups)
	}
	it := itemOf(t, r, "q40.jpg")
	if !slices.Contains(it.Flags, FlagDuplicate) || it.SameAs != "q95.jpg" || it.Usable ||
		it.Notes[FlagDuplicate] != "Same picture as q95.jpg (resized, re-saved or mirrored). q95.jpg is the better copy." {
		t.Errorf("q40: %+v", it)
	}
	if k := itemOf(t, r, "q95.jpg"); !k.Keeper || slices.Contains(k.Flags, FlagDuplicate) || k.Group != "g1" {
		t.Errorf("q95: %+v", k)
	}

	// A copy far out of focus loses even with more bytes per pixel.
	soft.Sharp = q95.Sharp * 0.2
	r = Judge([]Input{{Metrics: soft}, {Metrics: q95}}, SetState{}, Options{})
	if len(r.Groups) != 1 || r.Groups[0].Keep != "q95.jpg" {
		t.Errorf("out of focus: %+v (bytes %d vs %d)", r.Groups, soft.Bytes, q95.Bytes)
	}

	// The same bucket and both in focus: the sharper copy keeps.
	a, b := good("b.jpg", 1), good("a.jpg", 1)
	a.SHA256 = ""
	a.Thumb32 = q95.Thumb32
	b.Thumb32 = q95.Thumb32
	b.Sharp = 280
	b.Bytes = a.Bytes + 10
	r = Judge([]Input{{Metrics: b}, {Metrics: a}}, SetState{}, Options{})
	if len(r.Groups) != 1 || r.Groups[0].Keep != "b.jpg" {
		t.Errorf("sharper keeps: %+v", r.Groups)
	}
	b.Thumb32 = ""
	if r := Judge([]Input{{Metrics: b}, {Metrics: a}}, SetState{}, Options{}); len(r.Groups) != 1 || r.Groups[0].Kind != "similar" {
		t.Errorf("no thumbnail, no confirmation: %+v", r.Groups)
	}

	// In assess with a dataset, the stored copy keeps.
	r = Judge([]Input{{Metrics: q95}, {Metrics: q40, Existing: true}}, SetState{}, Options{})
	if len(r.Items) != 1 || len(r.Groups) != 1 || r.Groups[0].Keep != "q40.jpg" {
		t.Fatalf("stored keeper: items %d groups %+v", len(r.Items), r.Groups)
	}
	if n := itemOf(t, r, "q95.jpg").Notes[FlagDuplicate]; !strings.HasSuffix(n, "q40.jpg is the better copy, already in the dataset.") {
		t.Errorf("stored note %q", n)
	}

	// Identical bytes are a duplicate without pixels.
	x, y := good("x.jpg", 9), good("y.jpg", 10)
	y.SHA256 = x.SHA256
	r = Judge([]Input{{Metrics: x}, {Metrics: y}}, SetState{}, Options{})
	if len(r.Groups) != 1 || r.Groups[0].Distance != 0 || r.Groups[0].Keep != "x.jpg" {
		t.Errorf("same sha: %+v", r.Groups)
	}
}

// TestJudgeDuplicateConfirm: a close hash without an NCC confirmation is
// only similar (decision 5).
func TestJudgeDuplicateConfirm(t *testing.T) {
	const w, h = 72, 64
	a, b := make([]float32, w*h), make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ramp := float32(200 - 10*(x/8))
			qx, qy := x%8 < 4, y%8 < 4
			pa, pb := float32(-60), float32(-60)
			if qx == qy {
				pa = 60
			}
			if qx {
				pb = 60
			}
			a[y*w+x], b[y*w+x] = ramp+pa, ramp+pb
		}
	}
	ma, mb := good("a.jpg", 1), good("b.jpg", 2)
	for _, c := range []struct {
		m    *Metrics
		grid []float32
	}{{&ma, a}, {&mb, b}} {
		hh, ff := DHash(c.grid, w, h)
		c.m.DHash, c.m.DHashFlip = HashHex(hh), HashHex(ff)
		c.m.Thumb32 = base64.StdEncoding.EncodeToString(thumb32(c.grid, w, h))
	}
	r := Judge([]Input{{Metrics: ma}, {Metrics: mb}}, SetState{}, Options{})
	if len(r.Groups) != 1 || r.Groups[0].Kind != "similar" || r.Groups[0].Confirmed {
		t.Fatalf("groups %+v", r.Groups)
	}
	for _, f := range []string{"a.jpg", "b.jpg"} {
		it := itemOf(t, r, f)
		if !slices.Equal(it.Flags, []Flag{FlagSimilar}) || it.SameAs == "" || !it.Usable {
			t.Errorf("%s: %+v", f, it)
		}
	}
}

// TestBlurScale: blur is judged at the 1024 scale, so a 3 px blur on a
// 12 MP photo is caught; a set that is soft throughout is judged against
// the floor only when it is too small for the relative rule.
func TestBlurScale(t *testing.T) {
	var in []Input
	for i := 0; i < 9; i++ {
		in = append(in, Input{Metrics: measureBytes(t, fmt.Sprintf("s%d.jpg", i),
			testutil.JPEG(testutil.Scene(201+i, 1600, 1200), 90))})
	}
	blur := measureBytes(t, "blur12mp.jpg", testutil.JPEG(testutil.Blur(testutil.Scene(7, 4000, 3000), 3), 90))
	r := Judge(append(in, Input{Metrics: blur}), SetState{}, Options{})
	it := itemOf(t, r, "blur12mp.jpg")
	if !slices.Contains(it.Flags, FlagBlurry) || it.SharpRel >= 0.35 || !strings.Contains(it.Notes[FlagBlurry], "× the set's typical sharpness") {
		t.Errorf("12 MP blur: sharp %.1f rel %.3f flags %v", blur.Sharp, it.SharpRel, it.Flags)
	}
	for _, x := range in {
		if slices.Contains(itemOf(t, r, x.Metrics.File).Flags, FlagBlurry) {
			t.Errorf("%s flagged blurry", x.Metrics.File)
		}
	}

	soft := goodSet(10)
	for i := range soft {
		soft[i].Metrics.Sharp = 10 + float64(i)/10
	}
	if r := Judge(soft, SetState{}, Options{}); r.Counts[FlagBlurry] != 0 {
		t.Errorf("10 equally soft photos: %d blurry (the relative rule needs a sharper set)", r.Counts[FlagBlurry])
	}
	few := soft[:5]
	r = Judge(few, SetState{}, Options{})
	if r.Counts[FlagBlurry] != 5 {
		t.Errorf("5 soft photos: %d blurry by the floor", r.Counts[FlagBlurry])
	}
	if n := r.Items[0].Notes[FlagBlurry]; n != "Soft even in its sharpest part, so the LoRA learns the blur." {
		t.Errorf("floor-only note %q", n)
	}
	few[0].Metrics.Sharp = 30
	if r := Judge(few, SetState{}, Options{}); slices.Contains(r.Items[0].Flags, FlagBlurry) {
		t.Error("above the floor yet blurry in a small set")
	}
}

// withBits flips n low bits of a hash.
func withBits(h uint64, n int) uint64 { return h ^ (1<<n - 1) }

// TestBurst: EXIF times within 2 s and a near hash are one moment.
func TestBurst(t *testing.T) {
	pair := func(t1, t2 string, d int) []Input {
		a, b := good("a.jpg", 1), good("b.jpg", 2)
		h, _ := ParseHash(a.DHash)
		b.DHash, b.DHashFlip = HashHex(withBits(h, d)), HashHex(^h)
		a.DHashFlip = HashHex(^withBits(h, 40))
		a.Exif = &ExifInfo{Time: t1, Make: "Apple"}
		b.Exif = &ExifInfo{Time: t2, Make: "Apple"}
		if t1 == "" {
			a.Exif, b.Exif = nil, nil
		}
		return []Input{{Metrics: a}, {Metrics: b}}
	}
	r := Judge(pair("2026-05-01T14:03:22.100", "2026-05-01T14:03:23.300", 9), SetState{}, Options{})
	if len(r.Groups) != 1 || r.Groups[0].Kind != "burst" || r.Groups[0].Distance != 9 {
		t.Fatalf("1.2 s apart, 9 bits: %+v", r.Groups)
	}
	nk := itemOf(t, r, "b.jpg")
	k := itemOf(t, r, "a.jpg")
	if !slices.Equal(nk.Flags, []Flag{FlagBurst}) || nk.SameAs != "a.jpg" || slices.Contains(k.Flags, FlagBurst) ||
		nk.Notes[FlagBurst] != "Taken within 2 seconds of a.jpg. A burst counts as one photo toward variety." {
		t.Errorf("burst flags: keeper %v, other %v %q", k.Flags, nk.Flags, nk.SameAs)
	}
	if r.Usable != 1 {
		t.Errorf("a burst counts once: usable %d", r.Usable)
	}
	for _, c := range []struct {
		name   string
		t1, t2 string
	}{{"3 s apart", "2026-05-01T14:03:22", "2026-05-01T14:03:25"}, {"no exif", "", ""}} {
		r := Judge(pair(c.t1, c.t2, 9), SetState{}, Options{})
		for _, g := range r.Groups {
			if g.Kind == "burst" {
				t.Errorf("%s: a burst %+v", c.name, g)
			}
		}
		if len(r.Groups) != 1 || r.Groups[0].Kind != "similar" {
			t.Errorf("%s: 9 bits is still similar: %+v", c.name, r.Groups)
		}
	}
}

// caps builds n captioned good photos; fill(i) is caption i.
func caps(n int, fill func(i int) string) []Input {
	in := goodSet(n)
	for i := range in {
		c := fill(i)
		in[i].Caption = &c
	}
	return in
}

// TestCaptions covers the caption words and the variety keys.
func TestCaptions(t *testing.T) {
	outfit := caps(10, func(i int) string {
		if i < 7 {
			return "sks woman wearing a White T-shirt, smiling"
		}
		return "sks woman in a red dress"
	})
	r := Judge(outfit, SetState{}, Options{Trigger: "sks"})
	f := findingOf(r, "same_outfit")
	if f == nil || f.Message != "\"white t-shirt\" is in 7 of 10 captions." || r.Coverage.Outfits["red dress"] != 3 {
		t.Errorf("same_outfit %+v, outfits %v", f, r.Coverage.Outfits)
	}
	if !slices.Contains(r.Coverage.Basis, "captions") {
		t.Errorf("basis %v", r.Coverage.Basis)
	}
	shirt := caps(10, func(int) string { return "a woman in a shirt" })
	if r := Judge(shirt, SetState{}, Options{}); findingOf(r, "same_outfit") != nil || len(r.Coverage.Outfits) != 0 {
		t.Errorf("a bare garment counted: %v", r.Coverage.Outfits)
	}
	place := caps(10, func(i int) string {
		if i < 6 {
			return "a woman standing in a bright kitchen"
		}
		return "a woman at the beach, golden hour"
	})
	r = Judge(place, SetState{}, Options{})
	if f := findingOf(r, "same_place"); f == nil || f.Message != "6 of 10 captions place them in a kitchen." {
		t.Errorf("same_place %+v places %v", f, r.Coverage.Places)
	}
	if r.Coverage.Light["evening"] != 4 {
		t.Errorf("light %v", r.Coverage.Light)
	}
	backdrop := parseCaption("Portrait against a plain backdrop, in the living room", "")
	if !slices.Equal(backdrop.places, []string{"living room", "backdrop"}) {
		t.Errorf("places %v", backdrop.places)
	}
	for in, want := range map[string]string{"in a coffee shop": "coffee shop", "at the cosy coffee shop": "coffee shop",
		"in a shoe shop": "shop", "in her bright kitchen": "kitchen"} {
		if c := parseCaption("A woman sitting "+in, ""); !slices.Equal(c.places, []string{want}) {
			t.Errorf("%q places %v, want %q", in, c.places, want)
		}
	}
	if c := parseCaption("A woman in a blue denim jacket and grey tee", ""); !slices.Equal(c.outfits, []string{"blue denim jacket", "grey t-shirt"}) {
		t.Errorf("outfits %v", c.outfits)
	}

	sun := func(k int) []Input {
		return caps(10, func(i int) string {
			if i < k {
				return "ohwx man wearing sunglasses on a street"
			}
			return "ohwx man on a street"
		})
	}
	r = Judge(sun(2), SetState{}, Options{Trigger: "ohwx"})
	it := itemOf(t, r, "p00.jpg")
	if !slices.Contains(it.Flags, FlagMentionsCovered) || it.Notes[FlagMentionsCovered] != "The caption mentions sunglasses." {
		t.Errorf("sunglasses: %v %v", it.Flags, it.Notes)
	}
	if f := findingOf(r, "look_mismatch"); f == nil || f.Level != LevelInfo || len(f.Files) != 2 {
		t.Errorf("2/10 sunglasses: %+v", f)
	}
	if r := Judge(sun(10), SetState{}, Options{}); findingOf(r, "look_mismatch") != nil {
		t.Error("10/10 sunglasses is the look, not a mismatch")
	}

	// The trigger goes first: a trigger word that is a lexicon word is
	// not a mention.
	tr := caps(1, func(int) string { return "hat woman smiling, [trigger]" })
	if r := Judge(tr, SetState{}, Options{Trigger: "hat"}); len(r.Items[0].Flags) != 0 {
		t.Errorf("trigger word matched: %v", r.Items[0].Flags)
	}
	words := map[string]Flag{
		"two women laughing":         FlagMentionsPeople,
		"with her boyfriend":         FlagMentionsPeople,
		"a logo on the wall":         FlagMentionsText,
		"black and white photo":      FlagMentionsTone,
		"a screenshot of an app":     FlagMentionsScreen,
		"a sepia toned portrait":     FlagMentionsTone,
		"wearing a beanie, outdoors": FlagMentionsCovered,
	}
	for text, want := range words {
		r := Judge(caps(1, func(int) string { return text }), SetState{}, Options{})
		if !slices.Contains(r.Items[0].Flags, want) {
			t.Errorf("%q: %v, want %s", text, r.Items[0].Flags, want)
		}
	}
	if r := Judge(caps(1, func(int) string { return "a capable chatter in a hatchback" }), SetState{}, Options{}); len(r.Items[0].Flags) != 0 {
		t.Errorf("part words matched: %v", r.Items[0].Flags)
	}
	people := Judge(caps(10, func(i int) string {
		if i == 0 {
			return "a group of friends"
		}
		return "a woman"
	}), SetState{}, Options{})
	if f := findingOf(people, "people_in_captions"); f == nil || f.Message != "1 caption mentions other people." {
		t.Errorf("people_in_captions %+v", f)
	}
}

// dated sets EXIF times on a set, start + step*i.
func dated(in []Input, start string, step time.Duration) []Input {
	t0, _ := time.Parse("2006-01-02T15:04:05", start)
	for i := range in {
		in[i].Metrics.Exif = &ExifInfo{Time: t0.Add(time.Duration(i) * step).Format("2006-01-02T15:04:05"), Make: "Apple"}
	}
	return in
}

// TestSessions covers sessions, one_session and era_span.
func TestSessions(t *testing.T) {
	in := dated(goodSet(12), "2026-05-01T10:00:00", 5*time.Minute) // 8 within 40 minutes
	for i := 8; i < 12; i++ {
		in[i].Metrics.Exif.Time = fmt.Sprintf("2026-06-%02dT10:00:00", i)
	}
	r := Judge(in, SetState{}, Options{})
	f := findingOf(r, "one_session")
	if f == nil || f.Message != "8 of 12 photos were taken within one hour." {
		t.Errorf("one_session %+v", f)
	}
	if r.Coverage.Sessions != 5 || r.Coverage.LargestSession != 8 || r.Coverage.Dated != 12 || r.Coverage.SpanDays != 41 {
		t.Errorf("coverage %+v", r.Coverage)
	}
	if it := itemOf(t, r, "p00.jpg"); it.Session != 1 || it.Captured != "2026-05-01T10:00:00" {
		t.Errorf("p00 session %d captured %q", it.Session, it.Captured)
	}
	if itemOf(t, r, "p11.jpg").Session != 5 {
		t.Error("sessions are numbered in time order")
	}
	if !slices.Contains(r.Hints, hintSessions) {
		t.Errorf("hints %v", r.Hints)
	}

	spread := dated(goodSet(12), "2026-01-01T12:00:00", 49*time.Hour)
	if r := Judge(spread, SetState{}, Options{}); findingOf(r, "one_session") != nil || r.Coverage.Sessions != 12 {
		t.Errorf("spread: %v sessions %d", ids(r), r.Coverage.Sessions)
	}
	era := dated(goodSet(10), "2021-01-01T12:00:00", 160*24*time.Hour) // 4 years
	r = Judge(era, SetState{}, Options{})
	if f := findingOf(r, "era_span"); f == nil || f.Message != "Photos span 4 years. Haircut and age changes can blur the likeness." {
		t.Errorf("era_span %+v (span %d days)", f, r.Coverage.SpanDays)
	}
	just := dated(goodSet(10), "2021-01-01T12:00:00", 130*24*time.Hour) // 3.2 years
	if f := findingOf(Judge(just, SetState{}, Options{}), "era_span"); f == nil || !strings.HasPrefix(f.Message, "Photos span more than 3 years.") {
		t.Errorf("a 3.2-year span: %+v", f)
	}
	short := dated(goodSet(10), "2024-01-01T12:00:00", 60*24*time.Hour)
	if r := Judge(short, SetState{}, Options{}); findingOf(r, "era_span") != nil {
		t.Errorf("a 1.5-year span: %v", ids(r))
	}
	// Daytime only, no evening light: the light hint.
	day := dated(goodSet(8), "2026-03-01T09:00:00", 24*time.Hour)
	if r := Judge(day, SetState{}, Options{}); !slices.Contains(r.Hints, hintLight) {
		t.Errorf("daytime hints %v", r.Hints)
	}
}

// TestGrade walks the grade cascade.
func TestGrade(t *testing.T) {
	checked := SetState{Faces: FacesState{Available: true, State: "done", Checked: 1}}
	sessionGap := func(in []Input) []Input { return dated(in, "2026-05-01T10:00:00", time.Minute) }
	outfitGap := func(in []Input) []Input {
		for i := range in {
			c := "a woman in a white t-shirt"
			in[i].Caption = &c
		}
		return in
	}
	cases := []struct {
		name  string
		in    []Input
		set   SetState
		grade string
		prov  bool
	}{
		{"nothing readable", []Input{{Metrics: Metrics{File: "a.heic", Format: "heic", Error: "x"}}}, checked, "F", false},
		{"4 usable", goodSet(4), checked, "F", false},
		{"8 usable", goodSet(8), checked, "D", false},
		{"12 usable", goodSet(12), checked, "C", false},
		{"17 usable", goodSet(17), checked, "B", false},
		{"25 usable", goodSet(25), checked, "A", false},
		{"25 usable, faces unchecked", goodSet(25), SetState{}, "B", true},
		{"25 usable style", goodSet(25), SetState{Kind: KindStyle}, "A", false},
		{"25 usable object", goodSet(25), SetState{Kind: KindObject}, "A", false},
		{"32 usable, 1 gap", sessionGap(goodSet(32)), checked, "B", false},
		{"32 usable, 2 gaps", outfitGap(sessionGap(goodSet(32))), checked, "C", false},
	}
	for _, c := range cases {
		r := Judge(c.in, c.set, Options{})
		if r.Grade != c.grade || r.Provisional != c.prov || r.GradeLine == "" {
			t.Errorf("%s: grade %s provisional %v (usable %d, findings %v)", c.name, r.Grade, r.Provisional, r.Usable, ids(r))
		}
	}
	if r := Judge(goodSet(8), checked, Options{}); r.GradeLine != "Not enough yet. Add 7." {
		t.Errorf("D line %q", r.GradeLine)
	}
	if r := Judge(goodSet(8), checked, Options{}); findingOf(r, "count_very_low") == nil ||
		!slices.Equal(r.Hints[:1], []string{"Add 7 more photos of just them."}) {
		t.Errorf("8 usable: %v %v", ids(r), r.Hints)
	}
	if r := Judge(goodSet(12), SetState{Kind: KindStyle}, Options{}); findingOf(r, "count_low") == nil ||
		r.Findings[0].Message != "12 usable photos. Aim for 15-30." ||
		r.Hints[0] != "Add 8 more images in this style, of other subjects." {
		t.Errorf("style 12: %v %v", r.Findings, r.Hints)
	}
	big := Judge(goodSet(55), SetState{}, Options{})
	if f := findingOf(big, "count_high"); f == nil || f.Op == nil || f.Op.Op != "keep_best" || f.Op.N != 30 {
		t.Errorf("count_high %+v", f)
	}

	// Identity-driven grades (an otherwise A set of 30).
	r := Judge(faceSet(30, "front"), SetState{Identity: identityOf(30, outlierCos(30, 1))}, Options{})
	if r.Grade != "C" || r.Counts[FlagNotThem] != 1 {
		t.Errorf("one not_them: grade %s counts %v usable %d", r.Grade, r.Counts, r.Usable)
	}
	r = Judge(faceSet(30, "front"), SetState{Identity: identityOf(30, outlierCos(30, 10))}, Options{})
	if r.Grade != "F" || r.Counts[FlagNotThem] != 10 {
		t.Errorf("a third not them: grade %s counts %v", r.Grade, r.Counts)
	}
	// The share is of faced photos: 9 of 25 faces, with 10 no-face photos
	// that would bring it under 30% of every checked photo.
	noFace := faceSet(25, "front")
	for i := 0; i < 10; i++ {
		noFace = append(noFace, Input{Metrics: good(fmt.Sprintf("n%02d.jpg", i), 4000+i), Face: &FaceData{Algo: FaceAlgo, FrameSharp: 300}})
	}
	r = Judge(noFace, SetState{Identity: identityOf(25, outlierCos(25, 9))}, Options{})
	if r.Grade != "F" || r.Counts[FlagNotThem] != 9 || r.Faces.Checked != 35 {
		t.Errorf("not_them over faced photos: grade %s counts %v checked %d", r.Grade, r.Counts, r.Faces.Checked)
	}
	r = Judge(faceSet(30, "front"), SetState{Identity: identityOf(30, groupCos([]int{18, 12}, 0.6, 0.1))}, Options{})
	if r.Grade != "C" || findingOf(r, "two_people") == nil || r.Hints[0] != hintTwoPeople {
		t.Errorf("two people: grade %s %v %v", r.Grade, ids(r), r.Hints)
	}
	if r := Judge(faceSet(30, ""), SetState{Identity: identityOf(30, outlierCos(30, 0))}, Options{}); r.Grade != "A" || r.Provisional {
		t.Errorf("checked faces, one person: grade %s %v %v", r.Grade, ids(r), r.Hints)
	}
}

// TestJudgeFindings checks finding order, copy and ops.
func TestJudgeFindings(t *testing.T) {
	in := goodSet(16)
	in[0].Metrics.Sharp = 1 // blurry
	in[1].Metrics.LumaMean = 20
	in[2].Metrics.Orientation = 6
	in[3].Metrics.Width, in[3].Metrics.Height = 600, 700 // small
	in[4].Metrics.Width, in[4].Metrics.Height = 640, 800
	in[5].Metrics.Width, in[5].Metrics.Height = 700, 760
	in[6].Metrics.Width, in[6].Metrics.Height = 720, 720
	in[9].Metrics.Width, in[9].Metrics.Height = 600, 600
	in[7].Metrics.Colorful = 150
	dup := in[8].Metrics
	dup.File, dup.Bytes = "p08_copy.jpg", 100
	in = append(in, Input{Metrics: dup})
	r := Judge(in, SetState{Faces: FacesState{Available: true}}, Options{})
	got := ids(r)
	want := []string{"count_very_low", "duplicates", "low_res_share", "to_look_at", "orientation_tags", "faces_unchecked"}
	if !slices.Equal(got, want) {
		t.Fatalf("findings %v, want %v", got, want)
	}
	if f := findingOf(r, "to_look_at"); f.Message != "2 photos to look at: 1 blurry, 1 too dark." || len(f.Files) != 2 {
		t.Errorf("to_look_at %+v", f)
	}
	if f := findingOf(r, "duplicates"); f.Message != "1 photo is a copy of another one." ||
		!slices.Equal(f.Files, []string{"p08_copy.jpg"}) || f.Op.Op != "set_aside_flagged" {
		t.Errorf("duplicates %+v", f)
	}
	if f := findingOf(r, "low_res_share"); f.Message != "5 of 17 photos train below 1024 px." {
		t.Errorf("low_res %+v", f)
	}
	if f := findingOf(r, "orientation_tags"); f.Op.Op != "bake_orientation" {
		t.Errorf("orientation %+v", f)
	}
	if r.Usable != 9 || r.Findings[0].Message != "Only 9 usable photos. A likeness rarely holds with fewer than 10." {
		t.Errorf("usable %d: %q", r.Usable, r.Findings[0].Message)
	}
	if r.Flagged != 8 {
		t.Errorf("flagged %d", r.Flagged)
	}
	// Black and white in a third of the set.
	gm := goodSet(12)
	for i := 0; i < 4; i++ {
		gm[i].Metrics.ChromaSpread = 1
	}
	if f := findingOf(Judge(gm, SetState{}, Options{}), "grayscale_mix"); f == nil || f.Message != "4 photos are black and white or toned." {
		t.Errorf("grayscale_mix %+v", f)
	}
	filt := goodSet(10)
	for i := 0; i < 6; i++ {
		filt[i].Metrics.Colorful = 120
	}
	if findingOf(Judge(filt, SetState{}, Options{}), "filter_consistent") == nil {
		t.Error("6 of 10 strong colour: no filter_consistent")
	}
	stale := Judge(goodSet(20), SetState{Faces: FacesState{Available: true, Checked: 20, Stale: true, Added: 3}}, Options{})
	if f := findingOf(stale, "faces_stale"); f == nil || f.Message != "3 photos were added after the face check." {
		t.Errorf("faces_stale %+v", f)
	}
}
