package photocheck

import (
	"bytes"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/imgmeta"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/testutil"
)

// cacheKey is a 22-character cache suffix as the trainer writes it.
const cacheKey = "AbCdEfGhIjKlMnOpQrSt_-"

func (s *svc) exists(rel string) bool {
	_, err := os.Stat(filepath.Join(s.dir, filepath.FromSlash(rel)))
	return err == nil
}

func (s *svc) read(t *testing.T, rel string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(s.dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func (s *svc) fix(t *testing.T, apply bool, ops ...FixOp) *Fix {
	t.Helper()
	fx, err := ApplyFix(s.ctx, s.dir, FixRequest{Ops: ops, Apply: apply})
	if err != nil {
		t.Fatal(err)
	}
	if fx.Applied != apply || fx.Review == nil || fx.Changes == nil {
		t.Fatalf("fix %+v", fx)
	}
	return fx
}

func changesOf(fx *Fix) string {
	var out []string
	for _, c := range fx.Changes {
		out = append(out, c.Op+" "+c.File+">"+c.To)
	}
	return strings.Join(out, ", ")
}

func TestPurgeDerived(t *testing.T) {
	dir := t.TempDir()
	files := map[string][]byte{".thumbs/IMG_1.jpg.jpg": {1}, ".thumbs/IMG_1.jpg.png": {1}, ".thumbs/IMG_1_2.jpg.jpg": {1}}
	for _, c := range cacheDirs {
		for _, stem := range []string{"IMG_1", "IMG_1_2", "IMG_1__face", "IMG_10"} {
			files[c+"/"+stem+"_"+cacheKey+".safetensors"] = []byte{1}
		}
		files[c+"/IMG_1_short.safetensors"] = []byte{1}
	}
	if err := testutil.WriteTree(dir, files); err != nil {
		t.Fatal(err)
	}
	PurgeDerived(dir, "IMG_1.jpg")
	for name := range files {
		_, err := os.Stat(filepath.Join(dir, filepath.FromSlash(name)))
		gone := strings.Contains(name, "/IMG_1_"+cacheKey) || strings.HasPrefix(name, ".thumbs/IMG_1.jpg.")
		if gone != os.IsNotExist(err) {
			t.Errorf("%s: removed %v, want %v", name, os.IsNotExist(err), gone)
		}
	}
}

func TestFix(t *testing.T) {
	s := setupService(t)
	files := map[string][]byte{
		"IMG_1.jpg":             scene(71),
		"IMG_1.txt":             []byte("a caption"),
		".thumbs/IMG_1.jpg.jpg": {1, 2, 3},
		"g16.png":               testutil.Gray16PNG(testutil.Scene(72, 600, 600)),
		"t.png":                 testutil.TransparentPNG(testutil.Scene(73, 600, 600)),
		"t.txt":                 []byte("transparent"),
		"s.jpg":                 scene(74),
		"s.png":                 testutil.PNG(testutil.Scene(75, 600, 600)),
		"s.txt":                 []byte("shared"),
	}
	for _, c := range cacheDirs {
		for _, stem := range []string{"IMG_1", "IMG_1_2", "IMG_1__face"} {
			files[c+"/"+stem+"_"+cacheKey+".safetensors"] = []byte{1}
		}
	}
	s.write(t, files)
	s.settled(t, Options{})

	// a preview changes nothing
	fx := s.fix(t, false, FixOp{Op: "set_aside", Files: []string{"IMG_1.jpg"}})
	if changesOf(fx) != "set_aside IMG_1.jpg>.set_aside/IMG_1.jpg" || !s.exists("IMG_1.jpg") || s.exists(".set_aside") {
		t.Fatalf("preview: %s", changesOf(fx))
	}
	if itemByFile(fx.Review, "IMG_1.jpg") != nil || len(fx.Review.SetAside) != 1 {
		t.Fatal("the preview review should show the photo left out")
	}

	// apply moves the photo, its caption and its thumb, and purges its caches only
	fx = s.fix(t, true, FixOp{Op: "set_aside", Files: []string{"IMG_1.jpg"}})
	for _, rel := range []string{".set_aside/IMG_1.jpg", ".set_aside/IMG_1.txt", ".set_aside/.thumbs/IMG_1.jpg.jpg"} {
		if !s.exists(rel) {
			t.Errorf("%s missing", rel)
		}
	}
	if s.exists("IMG_1.jpg") || s.exists("IMG_1.txt") || s.exists(".thumbs/IMG_1.jpg.jpg") {
		t.Error("left behind")
	}
	for _, c := range cacheDirs {
		if s.exists(c + "/IMG_1_" + cacheKey + ".safetensors") {
			t.Errorf("%s not purged", c)
		}
		if !s.exists(c+"/IMG_1_2_"+cacheKey+".safetensors") || !s.exists(c+"/IMG_1__face_"+cacheKey+".safetensors") {
			t.Errorf("%s: other stems purged", c)
		}
	}
	if len(fx.Review.SetAside) != 1 || fx.Review.SetAside[0].File != ".set_aside/IMG_1.jpg" {
		t.Fatalf("set_aside %+v", fx.Review.SetAside)
	}

	// restore renames on a clash
	s.write(t, map[string][]byte{"IMG_1.jpg": scene(76)})
	fx = s.fix(t, true, FixOp{Op: "restore", Files: []string{".set_aside/IMG_1.jpg"}})
	if changesOf(fx) != "restore .set_aside/IMG_1.jpg>IMG_1_2.jpg" || !s.exists("IMG_1_2.jpg") || string(s.read(t, "IMG_1_2.txt")) != "a caption" {
		t.Fatalf("restore: %s", changesOf(fx))
	}
	if len(fx.Review.SetAside) != 0 {
		t.Fatalf("set_aside after restore %+v", fx.Review.SetAside)
	}

	// convert: 16-bit grey to 8-bit PNG in place, transparent to JPEG on white
	fx = s.fix(t, true, FixOp{Op: "convert"})
	if got := changesOf(fx); got != "convert g16.png>, convert t.png>t.jpg" {
		t.Fatalf("convert: %s", got)
	}
	img, err := png.Decode(bytes.NewReader(s.read(t, "g16.png")))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := img.(*image.Gray); !ok {
		t.Errorf("g16.png is %T, want 8-bit grey", img)
	}
	if _, err := jpeg.Decode(bytes.NewReader(s.read(t, "t.jpg"))); err != nil {
		t.Fatalf("t.jpg: %v", err)
	}
	if !s.exists(".set_aside/t.png") || s.exists("t.png") || string(s.read(t, "t.txt")) != "transparent" {
		t.Error("t.png: the original goes aside, the caption stays with t.jpg")
	}
	// the same format keeps its name, and the original still goes aside
	if !s.exists(".set_aside/g16.png") || !bytes.Equal(s.read(t, ".set_aside/g16.png"), files["g16.png"]) {
		t.Error("g16.png: the 16-bit original should be in .set_aside/")
	}
	for _, f := range []string{"g16.png", "t.jpg"} {
		if it := itemByFile(fx.Review, f); it == nil || slices.Contains(it.Flags, FlagGray16) || slices.Contains(it.Flags, FlagTransparent) {
			t.Errorf("%s after convert: %+v", f, it)
		}
	}

	// rename_same_stem keeps the trainer's photo and copies the caption
	fx = s.fix(t, true, FixOp{Op: "rename_same_stem"})
	if got := changesOf(fx); got != "rename_same_stem s.png>s_2.png" {
		t.Fatalf("rename: %s", got)
	}
	if !s.exists("s.jpg") || !s.exists("s_2.png") || string(s.read(t, "s_2.txt")) != "shared" || string(s.read(t, "s.txt")) != "shared" {
		t.Error("rename_same_stem files")
	}

	var ue *UserError
	if _, err := ApplyFix(s.ctx, s.dir, FixRequest{Ops: []FixOp{{Op: "nope"}}}); !errors.As(err, &ue) || !strings.Contains(err.Error(), "set_aside, restore") {
		t.Errorf("unknown op: %v", err)
	}
	if _, err := ApplyFix(s.ctx, s.dir, FixRequest{Ops: []FixOp{{Op: "set_aside", Files: []string{"missing.jpg"}}}}); !errors.As(err, &ue) {
		t.Errorf("missing file: %v", err)
	}
	if _, err := ApplyFix(s.ctx, s.dir, FixRequest{Ops: []FixOp{{Op: "face_crops"}}}); !errors.As(err, &ue) || !ue.Conflict {
		t.Errorf("face_crops without faces: %v, want a 409", err)
	}
}

func TestFixFlaggedAndBest(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{
		"a.jpg": scene(81), "a_copy.jpg": scene(81), "b.jpg": scene(82), "c.jpg": scene(83), "d.jpg": scene(84),
		"tiny.jpg": testutil.JPEG(testutil.Scene(85, 300, 400), 90),
	})
	r := s.settled(t, Options{})
	copy_, keeper := "a_copy.jpg", "a.jpg"
	if it := itemByFile(r, "a.jpg"); !it.Keeper {
		copy_, keeper = keeper, copy_
	}
	fx := s.fix(t, true, FixOp{Op: "set_aside_flagged"})
	if got, want := changesOf(fx), "set_aside "+copy_+">.set_aside/"+copy_+", set_aside tiny.jpg>.set_aside/tiny.jpg"; got != want {
		t.Fatalf("flagged: %s, want %s", got, want)
	}
	if fx.Changes[0].Reason != "duplicate of "+keeper {
		t.Errorf("reason %q", fx.Changes[0].Reason)
	}
	// keep_best after a restore sees the restored photos again
	fx = s.fix(t, false, FixOp{Op: "restore", All: true}, FixOp{Op: "keep_best", N: 2})
	var aside int
	for _, c := range fx.Changes {
		if c.Op == "set_aside" {
			aside++
		}
	}
	if aside != 4 || len(fx.Review.Items) != 2 {
		t.Fatalf("keep_best 2 of 6: %s", changesOf(fx))
	}
	if !s.exists(".set_aside/" + copy_) {
		t.Fatal("the preview wrote")
	}
}

func TestFixBake(t *testing.T) {
	s := setupService(t)
	rot := testutil.WithExif(testutil.JPEG(testutil.Scene(91, 800, 600), 92), 6, "2026:05:01 14:03:22", "Apple", "iPhone 15")
	s.write(t, map[string][]byte{"r.jpg": rot, "r.txt": []byte("cap")})
	r := s.settled(t, Options{})
	if it := itemByFile(r, "r.jpg"); !slices.Contains(it.Flags, FlagRotated) || it.Width != 600 {
		t.Fatalf("before: %+v", it)
	}
	fx := s.fix(t, true, FixOp{Op: "bake_orientation"})
	if got := changesOf(fx); got != "bake_orientation r.jpg>" {
		t.Fatalf("bake: %s", got)
	}
	data := s.read(t, "r.jpg")
	cfg, err := jpeg.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width != 600 || cfg.Height != 800 {
		t.Fatalf("baked %dx%d, %v", cfg.Width, cfg.Height, err)
	}
	sn := imgmeta.SniffHead(data)
	if sn.Exif.Orientation != 1 || sn.Exif.Make != "Apple" || sn.Exif.Time == "" {
		t.Fatalf("exif %+v", sn.Exif)
	}
	if it := itemByFile(fx.Review, "r.jpg"); slices.Contains(it.Flags, FlagRotated) || it.Width != 600 {
		t.Fatalf("after: %+v", it)
	}
	// the original is kept in .set_aside/, the caption stays with the new file
	if !bytes.Equal(s.read(t, ".set_aside/r.jpg"), rot) || string(s.read(t, "r.txt")) != "cap" {
		t.Fatal("bake: the original should be in .set_aside/ and the caption in place")
	}
	if len(fx.Review.SetAside) != 1 || fx.Review.SetAside[0].File != ".set_aside/r.jpg" {
		t.Fatalf("set_aside after bake %+v", fx.Review.SetAside)
	}
	entries, _ := os.ReadDir(s.dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".r.jpg") {
			t.Errorf("left a temp file %s", e.Name())
		}
	}
	// restore brings it back next to the baked one
	fx = s.fix(t, true, FixOp{Op: "restore", All: true})
	if got := changesOf(fx); got != "restore .set_aside/r.jpg>r_2.jpg" || !bytes.Equal(s.read(t, "r_2.jpg"), rot) {
		t.Fatalf("restore: %s", got)
	}
}

// A preview of a same-format bake writes nothing and shows the original
// set aside.
func TestFixBakePreview(t *testing.T) {
	s := setupService(t)
	rot := testutil.WithExif(testutil.JPEG(testutil.Scene(92, 800, 600), 92), 6, "", "", "")
	s.write(t, map[string][]byte{"r.jpg": rot})
	s.settled(t, Options{})
	fx := s.fix(t, false, FixOp{Op: "bake_orientation"})
	if got := changesOf(fx); got != "bake_orientation r.jpg>" {
		t.Fatalf("preview: %s", got)
	}
	if s.exists(".set_aside") || !bytes.Equal(s.read(t, "r.jpg"), rot) {
		t.Fatal("the preview wrote")
	}
	if len(fx.Review.SetAside) != 1 || itemByFile(fx.Review, "r.jpg") == nil {
		t.Fatalf("preview review: set aside %+v", fx.Review.SetAside)
	}
}

func TestFixFaceCrops(t *testing.T) {
	s := setupService(t)
	tree := map[string][]byte{}
	for i := range 4 {
		tree[string(rune('a'+i))+".jpg"] = testutil.JPEG(testutil.Scene(100+i, 1500, 2000), 92)
	}
	s.write(t, tree)
	s.settled(t, Options{})
	rows := map[string][]byte{}
	for f := range tree {
		b, _ := json.Marshal(oneFace(0.10, 200))
		rows[s.sha(t, f)] = b
	}
	if err := state.PutPhotoFaces(s.ctx, rows, FaceAlgo); err != nil {
		t.Fatal(err)
	}
	r := s.settled(t, Options{})
	if it := itemByFile(r, "a.jpg"); it.Face == nil || it.Face.Shot != "full" || !it.Usable {
		t.Fatalf("a.jpg %+v", it)
	}
	fx := s.fix(t, true, FixOp{Op: "face_crops", Max: 2})
	if got := changesOf(fx); got != "face_crops a.jpg>a__face.jpg, face_crops b.jpg>b__face.jpg" {
		t.Fatalf("crops: %s", got)
	}
	cfg, err := jpeg.DecodeConfig(bytes.NewReader(s.read(t, "a__face.jpg")))
	if err != nil || cfg.Width != 520 || cfg.Height != 520 {
		t.Fatalf("crop %dx%d, %v", cfg.Width, cfg.Height, err)
	}
	if s.exists("a__face.txt") {
		t.Error("a crop gets no caption")
	}
	it := itemByFile(fx.Review, "a__face.jpg")
	if it == nil || it.Parent != "a.jpg" || !slices.Contains(it.Flags, FlagDerived) {
		t.Fatalf("crop item %+v", it)
	}
	// setting the parent aside takes its crop along
	fx = s.fix(t, true, FixOp{Op: "set_aside", Files: []string{"a.jpg"}})
	if got := changesOf(fx); got != "set_aside a.jpg>.set_aside/a.jpg, set_aside a__face.jpg>.set_aside/a__face.jpg" {
		t.Fatalf("parent aside: %s", got)
	}
	fx = s.fix(t, true, FixOp{Op: "remove_face_crops"})
	if got := changesOf(fx); got != "remove_face_crops b__face.jpg>, remove_face_crops .set_aside/a__face.jpg>" {
		t.Fatalf("remove: %s", got)
	}
	if s.exists("b__face.jpg") || s.exists(".set_aside/a__face.jpg") || !s.exists("b.jpg") {
		t.Fatal("remove_face_crops files")
	}
	if set, _ := state.GetPhotoSet(s.ctx, s.dir); len(set.Derived) != 0 {
		t.Fatalf("derived left %v", set.Derived)
	}
}

func TestFixCaptions(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{
		"s.jpg":            scene(201),
		"s.png":            testutil.PNG(testutil.Scene(202, 600, 600)),
		"s.txt":            []byte("shared"),
		"x.jpg":            scene(203),
		"x.txt":            []byte("x jpg"),
		".set_aside/x.png": testutil.PNG(testutil.Scene(204, 600, 600)),
		".set_aside/x.txt": []byte("x png"),
	})
	s.settled(t, Options{})
	text := func(rel string) string {
		if !s.exists(rel) {
			return "<none>"
		}
		return string(s.read(t, rel))
	}
	asideFrom := func(r *Review, file string) string {
		for _, a := range r.SetAside {
			if a.File == file {
				return a.From
			}
		}
		return "<none>"
	}

	// setting aside one of two same-stem photos copies the caption: the
	// other keeps it
	s.fix(t, true, FixOp{Op: "set_aside", Files: []string{"s.png"}})
	if text("s.txt") != "shared" || text(".set_aside/s.txt") != "shared" || !s.exists(".set_aside/s.png") {
		t.Fatalf("same stem: s.txt %q, .set_aside/s.txt %q", text("s.txt"), text(".set_aside/s.txt"))
	}

	// a caption with that stem in .set_aside/ renames the photo, never
	// replaces the caption, and the path it came from is kept
	fx := s.fix(t, true, FixOp{Op: "set_aside", Files: []string{"x.jpg"}})
	if got := changesOf(fx); got != "set_aside x.jpg>.set_aside/x_2.jpg" {
		t.Fatalf("clash aside: %s", got)
	}
	if text(".set_aside/x.txt") != "x png" || text(".set_aside/x_2.txt") != "x jpg" {
		t.Fatalf("aside captions: x %q, x_2 %q", text(".set_aside/x.txt"), text(".set_aside/x_2.txt"))
	}
	if from := asideFrom(fx.Review, ".set_aside/x_2.jpg"); from != "x.jpg" {
		t.Fatalf("from %q, want x.jpg", from)
	}
	if from := asideFrom(s.settled(t, Options{}), ".set_aside/x_2.jpg"); from != "x.jpg" {
		t.Fatalf("stored from %q, want x.jpg", from)
	}

	// restore puts it back under the name it came from
	fx = s.fix(t, true, FixOp{Op: "restore", Files: []string{".set_aside/x_2.jpg"}})
	if got := changesOf(fx); got != "restore .set_aside/x_2.jpg>x.jpg" || text("x.txt") != "x jpg" {
		t.Fatalf("restore home: %s, x.txt %q", got, text("x.txt"))
	}
	if s.exists(".set_aside/" + asideFromFile) {
		t.Error("the origin record outlived the restore")
	}

	// a restore onto a stem with a caption renames instead of overwriting
	fx = s.fix(t, true, FixOp{Op: "restore", Files: []string{".set_aside/x.png"}})
	if got := changesOf(fx); got != "restore .set_aside/x.png>x_2.png" {
		t.Fatalf("restore clash: %s", got)
	}
	if text("x.txt") != "x jpg" || text("x_2.txt") != "x png" || s.exists(".set_aside/x.txt") {
		t.Fatalf("restored captions: x %q, x_2 %q", text("x.txt"), text("x_2.txt"))
	}
}

func TestFixChecksEveryOpFirst(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{"a.jpg": scene(211), "b.jpg": scene(212)})
	s.settled(t, Options{})
	for _, bad := range []FixOp{
		{Op: "set_aside_flagged", Flags: []Flag{"bogus"}},
		{Op: "keep_best", N: -1},
		{Op: "set_aside", Files: []string{"missing.jpg"}},
		{Op: "restore"},
	} {
		_, err := ApplyFix(s.ctx, s.dir, FixRequest{Apply: true,
			Ops: []FixOp{{Op: "set_aside", Files: []string{"a.jpg"}}, bad}})
		var ue *UserError
		if !errors.As(err, &ue) {
			t.Errorf("%s: %v, want a 400", bad.Op, err)
		}
		if !s.exists("a.jpg") || s.exists(".set_aside/a.jpg") {
			t.Fatalf("%s: a.jpg moved before the bad op failed", bad.Op)
		}
	}
}

func TestReviewKind(t *testing.T) {
	s := setupService(t)
	var ue *UserError
	if _, err := ReviewDataset(s.ctx, s.dir, Options{Kind: "pet"}); !errors.As(err, &ue) || ue.Conflict {
		t.Fatalf("kind pet: %v, want a 400", err)
	}
}

func TestUprightNoTurn(t *testing.T) {
	rgba := image.NewRGBA(image.Rect(0, 0, 4, 3))
	for i := 3; i < len(rgba.Pix); i += 4 {
		rgba.Pix[i] = 0xff
	}
	if got := upright(rgba, 1); got != image.Image(rgba) {
		t.Error("an upright opaque RGBA should come back as it is")
	}
	yc := image.NewYCbCr(image.Rect(2, 2, 6, 5), image.YCbCrSubsampleRatio420)
	got, ok := upright(yc, 1).(*image.RGBA)
	if !ok || got.Bounds() != image.Rect(0, 0, 4, 3) {
		t.Fatalf("YCbCr upright: %T %v", got, got.Bounds())
	}
	if turned := upright(yc, 6); turned.Bounds() != image.Rect(0, 0, 3, 4) {
		t.Fatalf("orientation 6: %v", turned.Bounds())
	}
}

func TestCropRect(t *testing.T) {
	// a face near the bottom-right corner shifts the square, never shrinks it
	if r, ok := cropRect([4]float64{0.9, 0.9, 0.08, 0.08}, 2000, 2000); ok {
		t.Fatalf("2.6 x 160 px is under 512: %v", r)
	}
	r, ok := cropRect([4]float64{0.9, 0.9, 0.1, 0.1}, 2000, 2000)
	if !ok || r != image.Rect(1480, 1480, 2000, 2000) {
		t.Fatalf("corner crop %v %v", r, ok)
	}
}

// A .set_aside/.from.json written by hand cannot send a restore outside
// the dataset; a clean in-dataset origin is still followed.
func TestFixRestoreIgnoresOutsideOrigin(t *testing.T) {
	s := setupService(t)
	s.write(t, map[string][]byte{
		".set_aside/x.jpg": scene(61),
		".set_aside/w.jpg": scene(62),
		".set_aside/y.jpg": scene(63),
		".set_aside/" + asideFromFile: []byte(`{
			"x.jpg": {"from": "../../evil/x.jpg"},
			"w.jpg": {"from": "C:/evil/w.jpg"},
			"y.jpg": {"from": "sub/y.jpg"}
		}`),
	})
	s.settled(t, Options{})
	fx := s.fix(t, true, FixOp{Op: "restore", All: true})
	if got := changesOf(fx); got != "restore .set_aside/w.jpg>w.jpg, restore .set_aside/x.jpg>x.jpg, restore .set_aside/y.jpg>sub/y.jpg" {
		t.Fatalf("restore: %s", got)
	}
	if !s.exists("x.jpg") || !s.exists("w.jpg") || !s.exists("sub/y.jpg") {
		t.Fatal("restored photos missing")
	}
	root := filepath.Dir(filepath.Dir(s.dir))
	if _, err := os.Stat(filepath.Join(root, "evil")); err == nil {
		t.Fatal("restore wrote outside the dataset")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(s.dir), "evil")); err == nil {
		t.Fatal("restore wrote outside the dataset")
	}
}

// panicMagic starts a test-only image format whose decoder panics, as a
// decoder bug on a hostile file would.
const panicMagic = "AITKPANIC"

func init() {
	image.RegisterFormat("aitkpanic", panicMagic,
		func(io.Reader) (image.Image, error) { panic("decoder bug") },
		func(io.Reader) (image.Config, error) {
			return image.Config{ColorModel: color.RGBAModel, Width: 640, Height: 480}, nil
		})
}

// A decoder that panics during an applied convert fails the request; the
// original stays where it was and the dataset is not left locked.
func TestFixConvertDecoderPanic(t *testing.T) {
	s := setupService(t)
	bad := append([]byte(panicMagic), make([]byte, 64)...)
	s.write(t, map[string][]byte{"x.bmp": bad, "a.jpg": scene(64)})
	s.settled(t, Options{})
	if _, err := ApplyFix(s.ctx, s.dir, FixRequest{Ops: []FixOp{{Op: "convert", Files: []string{"x.bmp"}}}, Apply: true}); err == nil ||
		!strings.Contains(err.Error(), "cannot decode x.bmp") {
		t.Fatalf("convert: %v, want a decode error", err)
	}
	if !bytes.Equal(s.read(t, "x.bmp"), bad) || s.exists(".set_aside/x.bmp") {
		t.Fatal("the original should stay in place")
	}
	done := make(chan error, 1)
	go func() {
		_, err := ApplyFix(s.ctx, s.dir, FixRequest{Ops: []FixOp{{Op: "set_aside", Files: []string{"a.jpg"}}}, Apply: true})
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("the dataset stayed locked")
	}
}
