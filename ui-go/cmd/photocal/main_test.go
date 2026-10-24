package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/testutil"
)

var (
	fixturesOnce sync.Once
	fixturesDir  string
	fixturesErr  error
)

// fixtures writes the labelled fixture tree once for every test, through
// -write-fixtures.
func fixtures(t *testing.T) string {
	t.Helper()
	fixturesOnce.Do(func() {
		fixturesDir, fixturesErr = os.MkdirTemp("", "photocal-*")
		if fixturesErr != nil {
			return
		}
		var out, errb bytes.Buffer
		if code := run([]string{"-write-fixtures", fixturesDir}, &out, &errb); code != 0 || !strings.Contains(out.String(), "wrote 35 fixtures") {
			fixturesErr = fmt.Errorf("-write-fixtures: exit %d, stdout %q, stderr %q", code, out.String(), errb.String())
		}
	})
	if fixturesErr != nil {
		t.Fatal(fixturesErr)
	}
	return fixturesDir
}

func TestMain(m *testing.M) {
	code := m.Run()
	if fixturesDir != "" {
		os.RemoveAll(fixturesDir)
	}
	os.Exit(code)
}

func defaults(dir string) options {
	return options{dir: dir, kind: photocheck.KindPerson, workers: 4, maxPairs: 200_000}
}

func folderOf(t *testing.T, r *Report, name string) Folder {
	t.Helper()
	for _, f := range r.Folders {
		if f.Name == name {
			return f
		}
	}
	t.Fatalf("no folder %q in %+v", name, r.Folders)
	return Folder{}
}

// TestCalibrateFixtures runs the tool on the generated fixtures: every
// labelled folder hits its flag at the shipped thresholds, the good photos
// carry no warn flag, and the dup/ pairs are confirmed duplicates while no
// pair across folders comes close.
func TestCalibrateFixtures(t *testing.T) {
	dir := fixtures(t)
	o := defaults(dir)
	o.pairs = true
	r, err := calibrate(o)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"good/scene_01.jpg", "dup/a/mirror.jpg", "screens/screenshot.png", "misc/transparent.png"} {
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(name))); err != nil {
			t.Error(err)
		}
	}
	if r.Photos != 35 || len(r.Skipped) != 0 || r.Resolution != 1024 || r.Faces {
		t.Fatalf("photos %d, skipped %v, R %d, faces %v", r.Photos, r.Skipped, r.Resolution, r.Faces)
	}
	if r.Thresholds["blur_floor"] != photocheck.DefaultThresholds().BlurFloor {
		t.Errorf("thresholds %v", r.Thresholds)
	}
	for _, name := range []string{"blur", "dark", "bright", "mono", "screens", "dup", "good"} {
		f := folderOf(t, r, name)
		if f.Target == "" || f.Hits != f.Photos || len(f.Misses) != 0 {
			t.Errorf("%s: target %q, %d/%d hits, misses %+v", name, f.Target, f.Hits, f.Photos, f.Misses)
		}
	}
	if g := folderOf(t, r, "good"); g.Warned != 0 || g.Photos != 11 {
		t.Errorf("good: %d photos, %d warned, flags %v", g.Photos, g.Warned, g.Flags)
	}
	if b := folderOf(t, r, "blur"); b.Flags[photocheck.FlagBlurry] != 3 {
		t.Errorf("blur flags %v", b.Flags)
	}
	if d := folderOf(t, r, "dup"); d.Flags[photocheck.FlagDuplicate] != 3 {
		// dup/a: 3 photos, one keeper; dup/b: 2 photos, one keeper
		t.Errorf("dup flags %v", d.Flags)
	}
	misc := folderOf(t, r, "misc")
	if misc.Target != "" || misc.Hits != 0 || misc.Flags[photocheck.FlagTiny] != 1 || misc.Flags[photocheck.FlagBars] != 1 {
		t.Errorf("misc %+v", misc)
	}

	var sharp *Metric
	for i := range r.Metrics {
		if r.Metrics[i].Name == "face_px" {
			t.Errorf("face metric without -faces")
		}
		if r.Metrics[i].Name == "sharp" {
			sharp = &r.Metrics[i]
		}
	}
	if sharp == nil || len(sharp.Rows) != 8 {
		t.Fatalf("sharp %+v", sharp)
	}
	var good, blur Dist
	for _, d := range sharp.Rows {
		switch d.Folder {
		case "good":
			good = d
		case "blur":
			blur = d
		}
	}
	if good.N != 11 || blur.N != 3 || blur.P95 >= photocheck.DefaultThresholds().BlurFloor || good.P50 < 100 {
		t.Errorf("sharp good %+v, blur %+v", good, blur)
	}

	p := r.Pairs
	if p == nil {
		t.Fatal("no pairs")
	}
	// dup/a gives 3 pairs and dup/b 1; dup/a and dup/b are not compared
	if p.Dup != 4 || p.DupWithin != 4 || p.DupConfirmed != 4 || len(p.Worst) != 0 {
		t.Errorf("dup pairs %+v", p)
	}
	// 35 photos in folders of 3, 2, 2, 5, 11, 8, 2, 2: (35² - Σn²) / 2
	if p.CrossTotal != 495 || p.Cross != 495 || p.CrossWithin != 0 || p.CrossSimilar != 0 {
		t.Errorf("cross pairs %d of %d, within %d, similar %d", p.Cross, p.CrossTotal, p.CrossWithin, p.CrossSimilar)
	}
	if p.CrossHamming.P5 <= float64(photocheck.DefaultThresholds().SimilarHamming) || len(p.Closest) != 10 {
		t.Errorf("cross Hamming %+v, closest %d", p.CrossHamming, len(p.Closest))
	}
	for i := 1; i < len(p.Closest); i++ {
		if p.Closest[i].Hamming < p.Closest[i-1].Hamming {
			t.Errorf("closest not in order: %+v", p.Closest)
		}
	}
}

func TestPairSample(t *testing.T) {
	o := defaults(fixtures(t))
	o.pairs, o.maxPairs = true, 100
	r, err := calibrate(o)
	if err != nil {
		t.Fatal(err)
	}
	// stride ceil(495/100) = 5 takes pairs 1, 6, 11, ...
	if p := r.Pairs; p.CrossTotal != 495 || p.Cross != 99 || p.Dup != 4 {
		t.Errorf("sample %d of %d, dup %d", p.Cross, p.CrossTotal, p.Dup)
	}
}

func TestRunText(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"-pairs", fixtures(t)}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{
		"35 photos in ", "judged as one dataset", "faces not checked",
		"Flag hits per folder", "blurry or face blurry", "3 (100%)",
		"sharp: blurry below blur_floor (and blur_rel x the set median) [blur_floor 25, blur_rel 0.35]",
		"luma_mean: dark below dark_mean, bright above bright_mean [dark_mean 40, bright_mean 220]",
		"Hamming in dup/", "NCC across folders",
		"In dup/: 4/4 within dup_hamming 5, 4/4 confirmed (NCC >= dup_ncc 0.95)",
		"Across folders: 0/495 within dup_hamming",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("output lacks %q:\n%s", want, s)
		}
	}
	if strings.Contains(s, "Missed in") {
		t.Errorf("misses on the fixtures:\n%s", s)
	}
}

func TestRunJSON(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"-json", "-each", "-r", "512,768", "-kind", "style", fixtures(t)}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	var r Report
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("%v: %s", err, out.String())
	}
	if r.Photos != 35 || !r.Each || r.Kind != photocheck.KindStyle || r.Resolution != 768 || r.Pairs != nil || r.Skipped == nil {
		t.Errorf("report %+v", r)
	}
	// judged on their own, folders under set_min have no set median
	for _, m := range r.Metrics {
		if m.Name != "sharp_rel" {
			continue
		}
		for _, d := range m.Rows {
			if d.Folder != "good" && d.Folder != "misc" {
				t.Errorf("sharp_rel for %s with %d photos", d.Folder, d.N)
			}
		}
	}
}

// TestMisses lists the photos that missed their folder's flag, with what
// they got instead, and skips the folders a dataset walk skips.
func TestMisses(t *testing.T) {
	dir := t.TempDir()
	files := map[string][]byte{
		"blur/sharp.jpg":           testutil.JPEG(testutil.Scene(1, 1200, 900), 90),
		"blur/soft.jpg":            testutil.JPEG(testutil.Blur(testutil.Scene(2, 1200, 900), 4), 90),
		"good/tiny.jpg":            testutil.JPEG(testutil.Scene(3, 300, 400), 90),
		"top.jpg":                  testutil.JPEG(testutil.Scene(4, 1200, 900), 90),
		"top.txt":                  []byte("a photo"),
		".set_aside/hidden.jpg":    testutil.JPEG(testutil.Scene(5, 1200, 900), 90),
		"good/_controls/depth.jpg": testutil.JPEG(testutil.Scene(6, 1200, 900), 90),
		"good/._tiny.jpg":          []byte("AppleDouble junk"), // macOS resource fork
		".top.jpg":                 testutil.JPEG(testutil.Scene(7, 1200, 900), 90),
		"_extra/kept.jpg":          testutil.JPEG(testutil.Scene(8, 1200, 900), 90),
		"notes.md":                 []byte("# notes"),
	}
	if err := testutil.WriteTree(dir, files); err != nil {
		t.Fatal(err)
	}
	r, err := calibrate(defaults(dir))
	if err != nil {
		t.Fatal(err)
	}
	// the dataset walk skips dot entries and the files in _controls, and
	// reads any other _ folder
	if r.Photos != 5 || len(r.Skipped) != 0 || folderOf(t, r, "_extra").Photos != 1 || folderOf(t, r, "good").Photos != 1 {
		t.Fatalf("photos %d, skipped %v, folders %+v", r.Photos, r.Skipped, r.Folders)
	}
	if r.Folders[0].Name != "." || r.Folders[0].Target != "" {
		t.Errorf("top folder %+v", r.Folders[0])
	}
	blur := folderOf(t, r, "blur")
	if blur.Hits != 1 || len(blur.Misses) != 1 || blur.Misses[0].File != "blur/sharp.jpg" {
		t.Errorf("blur %+v", blur)
	}
	good := folderOf(t, r, "good")
	if good.Hits != 0 || len(good.Misses) != 1 || good.Misses[0].Flags[0] != photocheck.FlagTiny {
		t.Errorf("good %+v", good)
	}

	var out, errb bytes.Buffer
	if code := run([]string{"-v", dir}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	for _, want := range []string{"Missed in blur (blurry or face blurry):", "blur/sharp.jpg  no flags", "Missed in good (no warn flag):", "good/tiny.jpg  tiny", "(no label)"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), "._tiny") {
		t.Errorf("AppleDouble file measured:\n%s", out.String())
	}
}

// TestDupTarget counts a dup/ photo only when it is in a duplicate group:
// a near-copy that dup_hamming misses lands in a similar group and is a
// miss, listed with that group.
func TestDupTarget(t *testing.T) {
	dir := t.TempDir()
	near := testutil.Scene(40, 1200, 900)
	files := map[string][]byte{
		"dup/a/orig.jpg":   testutil.JPEG(testutil.Scene(41, 1200, 900), 90),
		"dup/a/resave.jpg": testutil.JPEG(testutil.Scene(41, 1200, 900), 60),
		"dup/b/orig.jpg":   testutil.JPEG(near, 90),
		"dup/b/crop.jpg":   testutil.JPEG(testutil.Crop(near, 0.05), 90), // dHash 6 bits away
	}
	if err := testutil.WriteTree(dir, files); err != nil {
		t.Fatal(err)
	}
	o := defaults(dir)
	o.pairs = true
	r, err := calibrate(o)
	if err != nil {
		t.Fatal(err)
	}
	d := folderOf(t, r, "dup")
	if d.Hits != 2 || len(d.Misses) != 2 || d.Groups["duplicate"] != 2 || d.Groups["similar"] != 2 {
		t.Fatalf("dup %+v", d)
	}
	for _, m := range d.Misses {
		if !strings.HasPrefix(m.File, "dup/b/") || !slices.Equal(m.Groups, []string{"similar"}) || !slices.Contains(m.Flags, photocheck.FlagSimilar) {
			t.Errorf("miss %+v", m)
		}
	}
	if p := r.Pairs; p.Dup != 2 || p.DupConfirmed != 1 || p.DupSimilar != 2 || len(p.Worst) != 1 {
		t.Errorf("pairs %+v", p)
	}

	var out, errb bytes.Buffer
	if code := run([]string{dir}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	for _, want := range []string{"in a duplicate group", "2 (50%)", "duplicate 2, similar 2", "dup/b/crop.jpg  similar; in a similar group"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

// TestFaces reads the face job's faces.json: the identity matrix reaches
// the judge (an unrelated face is not_them, the folder of others hits it,
// cohesion and -ref similarity show), and the face rules' inputs print.
func TestFaces(t *testing.T) {
	dir := t.TempDir()
	const subjects = 8
	files := map[string][]byte{}
	var keys []string // matrix rows: "<sha>:0", the subjects first
	add := func(name string, seed int) {
		b := testutil.WithExif(testutil.JPEG(testutil.Scene(seed, 1200, 900), 90), 1, fmt.Sprintf("2024:05:%02d 10:00:00", seed), "Apple", "iPhone 15")
		files[name] = b
		keys = append(keys, sha(b)+":0")
	}
	for i := range subjects {
		add(fmt.Sprintf("good/g%d.jpg", i+1), i+1)
	}
	add("other_person/x.jpg", 20)
	if err := testutil.WriteTree(dir, files); err != nil {
		t.Fatal(err)
	}
	// the subjects sit at 0.7 to one another, the other face at 0 to all
	var cos []byte
	for i := range keys {
		for j := i + 1; j < len(keys); j++ {
			v := float32(0)
			if j < subjects {
				v = 0.7
			}
			cos = binary.LittleEndian.AppendUint16(cos, half(v))
		}
	}
	face := photocheck.Face{Box: [4]float64{0.3, 0.2, 0.4, 0.5}, Score: 0.95, Side: 300, AlignSide: 300, Sharp: 80, MeanY: 60, Embedded: true,
		Kps: [5][2]float64{{0.42, 0.38}, {0.58, 0.38}, {0.5, 0.48}, {0.44, 0.58}, {0.56, 0.58}}}
	res := photocheck.FaceJobResult{Algo: photocheck.FaceAlgo, Model: photocheck.FaceModel, Items: map[string]photocheck.FaceData{},
		Matrix: photocheck.FaceMatrix{Faces: keys, Cos: base64.StdEncoding.EncodeToString(cos)}}
	for i, k := range keys {
		f := face
		if i == 0 {
			f.Sharp = 40 // half the set's face median
		}
		res.Items[strings.TrimSuffix(k, ":0")] = photocheck.FaceData{Algo: photocheck.FaceAlgo, Faces: []photocheck.Face{f}, FrameSharp: 100}
	}
	fj, _ := json.Marshal(res)
	fp := filepath.Join(t.TempDir(), "faces.json")
	if err := os.WriteFile(fp, fj, 0o644); err != nil {
		t.Fatal(err)
	}
	metric := func(r *Report, name string) map[string]Dist {
		t.Helper()
		for _, m := range r.Metrics {
			if m.Name == name {
				rows := map[string]Dist{}
				for _, d := range m.Rows {
					rows[d.Folder] = d
				}
				return rows
			}
		}
		t.Fatalf("no metric %s", name)
		return nil
	}

	o := defaults(dir)
	o.faces = fp
	r, err := calibrate(o)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Faces {
		t.Error("faces not read")
	}
	other := folderOf(t, r, "other_person")
	if other.Target == "" || other.Hits != 1 || other.Flags[photocheck.FlagNotThem] != 1 {
		t.Errorf("other_person %+v", other)
	}
	good := folderOf(t, r, "good")
	if good.Hits != subjects || good.Flags[photocheck.FlagNotThem] != 0 || good.Shot["close"] != subjects || good.Yaw["front"] != subjects {
		t.Errorf("good %+v", good)
	}
	coh := metric(r, "face_cohesion")
	if g, x := coh["good"], coh["other_person"]; g.N != subjects || math.Abs(g.P50-0.7) > 0.005 || x.N != 1 || x.P50 > 0.01 {
		t.Errorf("face_cohesion %+v", coh)
	}
	// side 300 at R 1024 on a 1200x900 photo (edge 1039.2)
	if px := metric(r, "face_px")["good"]; px.N != subjects || px.P50 != 296 {
		t.Errorf("face_px %+v", px)
	}
	if fh := metric(r, "face_height")["good"]; fh.P50 != 0.5 {
		t.Errorf("face_height %+v", fh)
	}
	if fs := metric(r, "face_sharp")["good"]; fs.P5 != 40 || fs.P50 != 80 {
		t.Errorf("face_sharp %+v", fs)
	}
	// 9 faces, median 80: g1 at 40 reads 0.5
	if rel := metric(r, "face_sharp_rel")["good"]; rel.N != subjects || rel.P5 != 0.5 || rel.P50 != 1 {
		t.Errorf("face_sharp_rel %+v", rel)
	}
	if sm := metric(r, "face_smooth_rel")["good"]; sm.N != subjects || sm.P5 != 0.5 || sm.P50 != 1 {
		t.Errorf("face_smooth_rel %+v", sm)
	}
	if y := metric(r, "face_mean_y")["good"]; y.P50 != 60 {
		t.Errorf("face_mean_y %+v", y)
	}
	if lr := metric(r, "face_luma_rel")["good"]; lr.N != subjects || lr.P50 <= 0 {
		t.Errorf("face_luma_rel %+v", lr)
	}
	if rs := metric(r, "face_ref_sim"); len(rs) != 0 {
		t.Errorf("face_ref_sim without -ref %+v", rs)
	}

	o.ref = "good/g2.jpg"
	r, err = calibrate(o)
	if err != nil {
		t.Fatal(err)
	}
	rs := metric(r, "face_ref_sim")
	if g, x := rs["good"], rs["other_person"]; g.N != subjects || math.Abs(g.P50-0.7) > 0.005 || g.P95 != 1 || x.N != 1 || x.P50 > 0.01 {
		t.Errorf("face_ref_sim %+v", rs)
	}
	if f := folderOf(t, r, "other_person"); f.Hits != 1 {
		t.Errorf("other_person with -ref %+v", f)
	}

	var out, errb bytes.Buffer
	if code := run([]string{"-faces", fp, dir}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	for _, want := range []string{
		"with face data", "Shot and head turn per folder [shot_close 0.42, shot_bust 0.24, shot_half 0.11, yaw_front 0.15, yaw_quarter 0.35, yaw_min_align 40]",
		"close 8", "front 8",
		"face_sharp: subject face sharpness; face blurry below face_floor [face_floor 15]",
		"face_sharp_rel: face sharp / the set's face median; face blurry below face_blur_rel (set_min faces) [face_blur_rel 0.35, set_min 8]",
		"face_luma_rel: face mean_y / luma_mean; in shadow below backlit_rel [backlit_rel 0.6]",
		"face_mean_y: subject face mean luma; in shadow below backlit_max [backlit_max 70]",
		"[smoothed_rel 0.5, smoothed_side 96, set_min 8]",
		"[face_height 0.03, shot_close 0.42, shot_bust 0.24, shot_half 0.11]",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}

	o.ref = "other_person/missing.jpg"
	if _, err := calibrate(o); err == nil || !strings.Contains(err.Error(), "not a photo") {
		t.Errorf("missing ref: %v", err)
	}
	o.ref, o.faces = "", filepath.Join(dir, "good", "g1.jpg")
	if _, err := calibrate(o); err == nil {
		t.Error("a JPEG read as faces.json")
	}
}

func sha(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// half is v as IEEE 754 binary16, for the normal values the tests use.
func half(v float32) uint16 {
	if v == 0 {
		return 0
	}
	b := math.Float32bits(v)
	exp := int(b>>23&0xff) - 127 + 15
	return uint16(b>>16)&0x8000 | uint16(exp)<<10 | uint16(b>>13&0x3ff)
}

func TestRunErrors(t *testing.T) {
	empty := t.TempDir()
	for _, c := range []struct {
		args []string
		code int
		want string
	}{
		{nil, 2, "usage: photocal"},
		{[]string{"a", "b"}, 2, "usage: photocal"},
		{[]string{"-kind", "pet", empty}, 2, `kind "pet"`},
		{[]string{"-r", "32", empty}, 2, "outside 64-8192"},
		{[]string{"-ref", "a.jpg", empty}, 2, "-ref needs -faces"},
		{[]string{"-nope", empty}, 2, "flag provided but not defined"},
		{[]string{filepath.Join(empty, "missing")}, 1, "photocal:"},
		{[]string{empty}, 1, "no photos under"},
		{[]string{"-h"}, 0, "-write-fixtures"},
	} {
		var out, errb bytes.Buffer
		code := run(c.args, &out, &errb)
		if code != c.code || !strings.Contains(errb.String(), c.want) {
			t.Errorf("%v: exit %d, stderr %q; want %d and %q", c.args, code, errb.String(), c.code, c.want)
		}
	}
}

func TestDist(t *testing.T) {
	v := make([]float64, 20)
	for i := range v {
		v[20-1-i] = float64(i + 1) // unsorted on purpose
	}
	if d := dist(v); d != (Dist{N: 20, P5: 1, P25: 5, P50: 10, P75: 15, P95: 19}) {
		t.Errorf("%+v", d)
	}
	if d := dist([]float64{7}); d != (Dist{N: 1, P5: 7, P25: 7, P50: 7, P75: 7, P95: 7}) {
		t.Errorf("%+v", d)
	}
	if d := dist(nil); d != (Dist{}) {
		t.Errorf("%+v", d)
	}
}

func TestNum(t *testing.T) {
	for v, want := range map[float64]string{0: "0", 0.25: "0.25", 0.00031: "0.0003", 3.14159: "3.1416", 25: "25", 12.345: "12.35", 1039.2305: "1039.2", 200: "200", -0.05: "-0.05"} {
		if got := num(v); got != want {
			t.Errorf("num(%v) = %q, want %q", v, got, want)
		}
	}
}
