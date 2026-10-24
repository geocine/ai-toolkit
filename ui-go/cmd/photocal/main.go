// Command photocal calibrates the photo check thresholds. It measures every
// photo under a folder, groups the photos by their labelled subfolder (good/,
// blur/, dup/, screens/, dark/, bright/, mono/, other_person/, or any other
// name), judges them with the shipped thresholds and prints, per folder, the
// hit rate of the flag the label stands for, every flag that fired, and the
// p5/p25/median/p75/p95 of each measurement. -pairs adds the dHash and NCC
// distributions of the pairs inside dup/ (inside each dup/<set>/ when dup/
// has subfolders) against pairs from different folders.
//
// The output feeds a threshold change in internal/photocheck/flags.go; this
// tool changes nothing on disk except under -write-fixtures.
//
//	go run ./cmd/photocal [-pairs] [-r 1024] [-kind person] [-faces faces.json] <folder>
//	go run ./cmd/photocal -write-fixtures <dir>   # the generated fixtures, labelled
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"io"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"

	"github.com/geocine/aitoolkit/internal/photocheck"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

// options are the command line.
type options struct {
	dir         string
	pairs       bool
	each        bool
	jsonOut     bool
	verbose     bool
	kind        photocheck.Kind
	resolutions []int
	faces       string
	ref         string
	workers     int
	maxPairs    int
}

func run(args []string, stdout, stderr io.Writer) int {
	fl := flag.NewFlagSet("photocal", flag.ContinueOnError)
	fl.SetOutput(stderr)
	fl.Usage = func() {
		fmt.Fprintln(stderr, "usage: photocal [flags] <folder>\n       photocal -write-fixtures <dir>")
		fl.PrintDefaults()
	}
	var o options
	var res, kind, fixtures string
	fl.BoolVar(&o.pairs, "pairs", false, "print dHash and NCC distributions of dup/ pairs against cross-folder pairs")
	fl.BoolVar(&o.each, "each", false, "judge each folder as its own dataset (default: the whole tree as one)")
	fl.BoolVar(&o.jsonOut, "json", false, "print the report as JSON")
	fl.BoolVar(&o.verbose, "v", false, "list every file that missed its folder's flag")
	fl.StringVar(&res, "r", "1024", "training resolutions, e.g. 1024 or 512,768,1024")
	fl.StringVar(&kind, "kind", string(photocheck.KindPerson), "dataset kind: person, style or object")
	fl.StringVar(&o.faces, "faces", "", "faces.json from the face job, for the face flags (not_them, others)")
	fl.StringVar(&o.ref, "ref", "", "the \"this is them\" photo, relative to the folder (needs -faces)")
	fl.IntVar(&o.workers, "j", runtime.NumCPU(), "photos measured at once")
	fl.IntVar(&o.maxPairs, "max-pairs", 200_000, "cross-folder pairs compared at most (an even sample above it)")
	fl.StringVar(&fixtures, "write-fixtures", "", "write the generated fixtures as a labelled tree into this folder and exit")
	if err := fl.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fixtures != "" {
		n, err := writeFixtures(fixtures)
		if err != nil {
			fmt.Fprintln(stderr, "photocal:", err)
			return 1
		}
		fmt.Fprintf(stdout, "wrote %d fixtures to %s\n", n, fixtures)
		return 0
	}
	if fl.NArg() != 1 {
		fl.Usage()
		return 2
	}
	o.dir = fl.Arg(0)
	o.kind = photocheck.Kind(kind)
	switch o.kind {
	case photocheck.KindPerson, photocheck.KindStyle, photocheck.KindObject:
	default:
		fmt.Fprintf(stderr, "photocal: kind %q is not person, style or object\n", kind)
		return 2
	}
	var err error
	if o.resolutions, err = photocheck.ParseResolutionList(res); err != nil {
		fmt.Fprintln(stderr, "photocal:", err)
		return 2
	}
	if o.ref != "" && o.faces == "" {
		fmt.Fprintln(stderr, "photocal: -ref needs -faces")
		return 2
	}
	o.workers = max(1, o.workers)

	rep, err := calibrate(o)
	if err != nil {
		fmt.Fprintln(stderr, "photocal:", err)
		return 1
	}
	for _, s := range rep.Skipped {
		fmt.Fprintln(stderr, "photocal: skipped", s)
	}
	if o.jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(stderr, "photocal:", err)
			return 1
		}
		return 0
	}
	printReport(stdout, rep, o.verbose)
	return 0
}

// Report is everything photocal prints (-json writes it as is).
type Report struct {
	Dir        string             `json:"dir"`
	Kind       photocheck.Kind    `json:"kind"`
	Resolution int                `json:"resolution"`
	Each       bool               `json:"each"`
	Faces      bool               `json:"faces"`
	Photos     int                `json:"photos"`
	Skipped    []string           `json:"skipped"`
	Thresholds map[string]float64 `json:"thresholds"`
	Folders    []Folder           `json:"folders"`
	Metrics    []Metric           `json:"metrics"`
	Pairs      *Pairs             `json:"pairs,omitempty"`
}

// Folder is one labelled subfolder's flag hits.
type Folder struct {
	Name   string                  `json:"name"`
	Photos int                     `json:"photos"`
	Target string                  `json:"target,omitempty"` // what a hit means for this label; "" for an unknown label
	Hits   int                     `json:"hits"`
	Warned int                     `json:"warned"` // photos with a warn flag
	Flags  map[photocheck.Flag]int `json:"flags"`
	Groups map[string]int          `json:"groups"` // photos in a duplicate, burst or similar group, keepers included
	Shot   map[string]int          `json:"shot,omitempty"`
	Yaw    map[string]int          `json:"yaw,omitempty"`
	Misses []Miss                  `json:"misses"` // photos that missed the target
}

// Miss is one photo that missed its folder's target, with the flags it got
// and the kinds of group it is in.
type Miss struct {
	File   string            `json:"file"`
	Flags  []photocheck.Flag `json:"flags"`
	Groups []string          `json:"groups,omitempty"`
}

// Metric is one measurement's distribution per folder.
type Metric struct {
	Name string `json:"name"`
	Note string `json:"note,omitempty"` // the thresholds that read it
	Rows []Dist `json:"rows"`
}

// Dist is a nearest-rank five-number summary.
type Dist struct {
	Folder string  `json:"folder,omitempty"`
	N      int     `json:"n"`
	P5     float64 `json:"p5"`
	P25    float64 `json:"p25"`
	P50    float64 `json:"median"`
	P75    float64 `json:"p75"`
	P95    float64 `json:"p95"`
}

// Pairs compares near-copies in dup/ with unrelated photos. Hamming is
// min(direct, mirrored); NCC is max(direct, mirrored) over thumb32, the
// judge's own reading of a pair.
type Pairs struct {
	Dup          int        `json:"dup"`
	Cross        int        `json:"cross"`
	CrossTotal   int        `json:"cross_total"` // before sampling
	DupHamming   Dist       `json:"dup_hamming"`
	CrossHamming Dist       `json:"cross_hamming"`
	DupNCC       Dist       `json:"dup_ncc"`
	CrossNCC     Dist       `json:"cross_ncc"`
	DupWithin    int        `json:"dup_within"`    // Hamming <= dup_hamming
	DupConfirmed int        `json:"dup_confirmed"` // ... and NCC >= dup_ncc
	DupSimilar   int        `json:"dup_similar"`   // Hamming <= similar_hamming
	CrossWithin  int        `json:"cross_within"`
	CrossConfirm int        `json:"cross_confirmed"`
	CrossSimilar int        `json:"cross_similar"`
	Worst        []PairInfo `json:"worst"` // dup/ pairs a duplicate check misses, farthest first
	Closest      []PairInfo `json:"closest"`
}

// PairInfo is one pair of photos.
type PairInfo struct {
	A       string  `json:"a"`
	B       string  `json:"b"`
	Hamming int     `json:"hamming"`
	NCC     float64 `json:"ncc"`
}

// photo is one measured file.
type photo struct {
	rel    string // slash path under the folder
	folder string // first path element; "." for files at the top
	set    string // dup/<set> for pair grouping, else folder
	in     photocheck.Input
	item   photocheck.Item
	groups []string // kinds of the groups it is in: duplicate, burst, similar
	// the subject face against the judged set's medians (0: not available,
	// as the judge reads them only with set_min faces)
	faceSharpRel float64 // face sharp / the median face sharp
	smoothRel    float64 // (face sharp / frame sharp) / the median of that
}

// controlsDir holds a photo's control images; the dataset walk skips the
// files in it (photocheck walk.go).
const controlsDir = "_controls"

// imageExt is what photocal measures: what the trainer reads plus what the
// check lists as unsupported.
func imageExt(ext string) bool {
	ext = strings.ToLower(ext)
	return photocheck.TrainerExts[ext] || photocheck.ConvertibleExts[ext]
}

func calibrate(o options) (*Report, error) {
	st, err := os.Stat(o.dir)
	if err != nil {
		return nil, err
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("%s is not a folder", o.dir)
	}
	photos, skipped, err := measureTree(o.dir, o.workers)
	if err != nil {
		return nil, err
	}
	if len(photos) == 0 {
		return nil, fmt.Errorf("no photos under %s", o.dir)
	}
	set := photocheck.SetState{Kind: o.kind}
	if o.faces != "" {
		if err := loadFaces(o, photos, &set); err != nil {
			return nil, err
		}
	}
	opts := photocheck.Options{Kind: o.kind, Resolutions: o.resolutions}
	if o.each {
		for _, ps := range byFolder(photos) {
			judge(ps, set, opts)
		}
	} else {
		judge(photos, set, opts)
	}

	rep := &Report{
		Dir: o.dir, Kind: o.kind, Resolution: opts.R(), Each: o.each, Faces: o.faces != "",
		Photos: len(photos), Skipped: skipped, Thresholds: photocheck.DefaultThresholds().Map(),
	}
	if rep.Skipped == nil {
		rep.Skipped = []string{}
	}
	for _, ps := range byFolder(photos) {
		rep.Folders = append(rep.Folders, folderReport(ps))
	}
	rep.Metrics = metricReports(photos, rep.Faces)
	if o.pairs {
		rep.Pairs = pairReport(photos, o.maxPairs)
	}
	return rep, nil
}

// measureTree measures every photo under dir, skipping what the dataset
// walk (photocheck walkFolder) skips: every entry whose name starts with
// "." (.set_aside, macOS ._ AppleDouble files) and the files directly
// inside a _controls folder. A file that cannot be read at all is skipped
// with its error; one that does not decode is kept (it is flagged).
func measureTree(dir string, workers int) ([]*photo, []string, error) {
	var photos []*photo
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p == dir {
			return nil
		}
		name := d.Name()
		if strings.HasPrefix(name, ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() || !d.Type().IsRegular() || !imageExt(filepath.Ext(name)) {
			return nil
		}
		if filepath.Base(filepath.Dir(p)) == controlsDir {
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		parts := strings.Split(rel, "/")
		ph := &photo{rel: rel, folder: ".", set: "."}
		if len(parts) > 1 {
			ph.folder, ph.set = parts[0], parts[0]
			if parts[0] == "dup" && len(parts) > 2 {
				ph.set = "dup/" + parts[1]
			}
		}
		photos = append(photos, ph)
		return nil
	})
	if err != nil {
		return nil, nil, err
	}

	errs := make([]error, len(photos))
	next := make(chan int)
	var wg sync.WaitGroup
	for range min(workers, max(1, len(photos))) {
		wg.Go(func() {
			for i := range next {
				ph := photos[i]
				full := filepath.Join(dir, filepath.FromSlash(ph.rel))
				ph.in.Metrics, errs[i] = photocheck.Measure(full, ph.rel)
				if b, err := os.ReadFile(strings.TrimSuffix(full, filepath.Ext(full)) + ".txt"); err == nil {
					c := string(b)
					ph.in.Caption = &c
				}
			}
		})
	}
	for i := range photos {
		next <- i
	}
	close(next)
	wg.Wait()

	var kept []*photo
	var skipped []string
	for i, ph := range photos {
		if errs[i] != nil {
			skipped = append(skipped, fmt.Sprintf("%s: %v", ph.rel, errs[i]))
			continue
		}
		kept = append(kept, ph)
	}
	return kept, skipped, nil
}

// loadFaces reads the face job's faces.json into the photos' inputs and
// the set's identity matrix.
func loadFaces(o options, photos []*photo, set *photocheck.SetState) error {
	b, err := os.ReadFile(o.faces)
	if err != nil {
		return err
	}
	var res photocheck.FaceJobResult
	if err := json.Unmarshal(b, &res); err != nil {
		return fmt.Errorf("%s: %w", o.faces, err)
	}
	for _, ph := range photos {
		if fd, ok := res.Items[ph.in.Metrics.SHA256]; ok {
			ph.in.Face = &fd
		}
	}
	set.Identity = &photocheck.Identity{FaceMatrix: res.Matrix, Model: res.Model}
	set.Faces = photocheck.FacesState{Available: true, State: "done", Checked: len(res.Items)}
	if o.ref != "" {
		ref := path.Clean(filepath.ToSlash(o.ref))
		for _, ph := range photos {
			if ph.rel == ref {
				set.Reference = ph.in.Metrics.SHA256
			}
		}
		if set.Reference == "" {
			return fmt.Errorf("-ref %s is not a photo under %s", o.ref, o.dir)
		}
	}
	return nil
}

// judge runs the photo judge over ps and stores each photo's item, the
// groups it is in and its face readings against the set's medians.
func judge(ps []*photo, set photocheck.SetState, opts photocheck.Options) {
	in := make([]photocheck.Input, len(ps))
	byFile := map[string]*photo{}
	for i, ph := range ps {
		in[i] = ph.in
		byFile[ph.rel] = ph
	}
	r := photocheck.Judge(in, set, opts)
	for _, it := range r.Items {
		if ph := byFile[it.File]; ph != nil {
			ph.item = it
		}
	}
	for _, g := range r.Groups {
		for _, f := range g.Files {
			if ph := byFile[f]; ph != nil && !slices.Contains(ph.groups, g.Kind) {
				ph.groups = append(ph.groups, g.Kind)
			}
		}
	}

	// face_blurry and smoothed compare with set medians (photocheck
	// faceFlags); the judge reads them only with set_min faces.
	th := photocheck.DefaultThresholds()
	var sharp, ratio []float64
	for _, ph := range ps {
		if s := subjectFace(ph); s != nil {
			sharp = append(sharp, s.Sharp)
			if smoothable(ph, s, th) {
				ratio = append(ratio, s.Sharp/ph.in.Face.FrameSharp)
			}
		}
	}
	medSharp, medRatio := 0.0, 0.0
	if len(sharp) >= th.SetMin {
		medSharp = median(sharp)
	}
	if len(ratio) >= th.SetMin {
		medRatio = median(ratio)
	}
	for _, ph := range ps {
		s := subjectFace(ph)
		if s == nil {
			continue
		}
		if medSharp > 0 {
			ph.faceSharpRel = s.Sharp / medSharp
		}
		if medRatio > 0 && smoothable(ph, s, th) {
			ph.smoothRel = s.Sharp / ph.in.Face.FrameSharp / medRatio
		}
	}
}

// subjectFace is the face the judge took as the photo's subject (the one
// whose box the item carries), or nil.
func subjectFace(ph *photo) *photocheck.Face {
	f := ph.item.Face
	if f == nil || f.Box == nil || ph.in.Face == nil {
		return nil
	}
	for i := range ph.in.Face.Faces {
		if ph.in.Face.Faces[i].Box == *f.Box {
			return &ph.in.Face.Faces[i]
		}
	}
	return nil
}

// smoothable: the smoothed rule reads this face (large enough to align,
// with a frame sharpness to compare with).
func smoothable(ph *photo, s *photocheck.Face, th photocheck.Thresholds) bool {
	return float64(s.AlignSide) >= th.SmoothedSide && ph.in.Face.FrameSharp > 0
}

func median(v []float64) float64 {
	s := slices.Clone(v)
	slices.Sort(s)
	n := len(s)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// byFolder groups photos by folder, in name order with "." first.
func byFolder(photos []*photo) [][]*photo {
	m := map[string][]*photo{}
	var names []string
	for _, ph := range photos {
		if _, ok := m[ph.folder]; !ok {
			names = append(names, ph.folder)
		}
		m[ph.folder] = append(m[ph.folder], ph)
	}
	sort.Strings(names)
	out := make([][]*photo, 0, len(names))
	for _, n := range names {
		ps := m[n]
		slices.SortFunc(ps, func(a, b *photo) int { return strings.Compare(a.rel, b.rel) })
		out = append(out, ps)
	}
	return out
}

// target is what a labelled folder expects of its photos.
type target struct {
	what string
	hit  func(*photo) bool
}

func hasAny(fs ...photocheck.Flag) func(*photo) bool {
	return func(ph *photo) bool {
		for _, f := range fs {
			if slices.Contains(ph.item.Flags, f) {
				return true
			}
		}
		return false
	}
}

var targets = map[string]target{
	"good": {"no warn flag", func(ph *photo) bool { return ph.item.Level != photocheck.LevelWarn }},
	"blur": {"blurry or face blurry", hasAny(photocheck.FlagBlurry, photocheck.FlagFaceBlurry)},
	// keeper or copy: a similar or burst group alone means dup_hamming or
	// dup_ncc missed the copy, so it is a miss (listed with its group)
	"dup":          {"in a duplicate group", func(ph *photo) bool { return slices.Contains(ph.groups, "duplicate") }},
	"screens":      {"screenshot or saved from a screen", hasAny(photocheck.FlagScreenshot, photocheck.FlagFromScreen)},
	"dark":         {"too dark", hasAny(photocheck.FlagDark)},
	"bright":       {"blown out", hasAny(photocheck.FlagBright)},
	"mono":         {"black and white", hasAny(photocheck.FlagGrayscale)},
	"other_person": {"someone else?, other people or check the face", hasAny(photocheck.FlagNotThem, photocheck.FlagOthers, photocheck.FlagIdentityCheck)},
}

func folderReport(ps []*photo) Folder {
	f := Folder{Name: ps[0].folder, Photos: len(ps), Flags: map[photocheck.Flag]int{}, Groups: map[string]int{}, Misses: []Miss{}}
	t, known := targets[f.Name]
	if known {
		f.Target = t.what
	}
	count := func(m *map[string]int, k string) {
		if k == "" {
			return
		}
		if *m == nil {
			*m = map[string]int{}
		}
		(*m)[k]++
	}
	for _, ph := range ps {
		it := ph.item
		for _, fl := range it.Flags {
			f.Flags[fl]++
		}
		for _, g := range ph.groups {
			f.Groups[g]++
		}
		if it.Face != nil {
			count(&f.Shot, it.Face.Shot)
			count(&f.Yaw, it.Face.Yaw)
		}
		if it.Level == photocheck.LevelWarn {
			f.Warned++
		}
		if !known {
			continue
		}
		if t.hit(ph) {
			f.Hits++
		} else {
			f.Misses = append(f.Misses, Miss{File: ph.rel, Flags: slices.Clone(it.Flags), Groups: slices.Clone(ph.groups)})
		}
	}
	return f
}

// metricDef is one printed measurement.
type metricDef struct {
	name string
	note string
	face bool // only with -faces
	get  func(ph *photo) (float64, bool)
}

func measured(ph *photo) bool { return ph.in.Metrics.Decoded }

func plain(f func(m photocheck.Metrics) float64) func(*photo) (float64, bool) {
	return func(ph *photo) (float64, bool) { return f(ph.in.Metrics), measured(ph) }
}

func optional(f func(m photocheck.Metrics) *float64) func(*photo) (float64, bool) {
	return func(ph *photo) (float64, bool) {
		v := f(ph.in.Metrics)
		if v == nil || !measured(ph) {
			return 0, false
		}
		return *v, true
	}
}

var metricDefs = []metricDef{
	{"edge", "sqrt(w x h); small below small_rel x R", false, func(ph *photo) (float64, bool) {
		m := ph.in.Metrics
		return math.Sqrt(float64(m.Width) * float64(m.Height)), m.Width > 0 && m.Height > 0
	}},
	{"short_side", "tiny below tiny_short", false, func(ph *photo) (float64, bool) {
		m := ph.in.Metrics
		return float64(min(m.Width, m.Height)), m.Width > 0 && m.Height > 0
	}},
	{"aspect", "long / short; aspect_info, aspect_warn", false, func(ph *photo) (float64, bool) {
		m := ph.in.Metrics
		if m.Width <= 0 || m.Height <= 0 {
			return 0, false
		}
		return float64(max(m.Width, m.Height)) / float64(min(m.Width, m.Height)), true
	}},
	{"sharp", "blurry below blur_floor (and blur_rel x the set median)", false, plain(func(m photocheck.Metrics) float64 { return m.Sharp })},
	{"sharp_rel", "sharp / the set median; blurry below blur_rel (set_min photos)", false, func(ph *photo) (float64, bool) {
		return ph.item.SharpRel, ph.item.SharpRel > 0
	}},
	{"luma_mean", "dark below dark_mean, bright above bright_mean", false, plain(func(m photocheck.Metrics) float64 { return m.LumaMean })},
	{"luma_std", "flat below flat_std", false, plain(func(m photocheck.Metrics) float64 { return m.LumaStd })},
	{"clip_lo", "", false, plain(func(m photocheck.Metrics) float64 { return m.ClipLo })},
	{"clip_hi", "", false, plain(func(m photocheck.Metrics) float64 { return m.ClipHi })},
	{"clip_hi_center", "bright above bright_center", false, plain(func(m photocheck.Metrics) float64 { return m.ClipHiCenter })},
	{"flat", "", false, plain(func(m photocheck.Metrics) float64 { return m.Flat })},
	{"chroma_spread", "grayscale below gray_chroma", false, plain(func(m photocheck.Metrics) float64 { return m.ChromaSpread })},
	{"colorful", "flagged above strong_color", false, plain(func(m photocheck.Metrics) float64 { return m.Colorful })},
	{"bars", "flagged above bars", false, plain(func(m photocheck.Metrics) float64 { return m.Bars })},
	{"noise", "flagged above noisy", false, optional(func(m photocheck.Metrics) *float64 { return m.Noise })},
	{"detail", "maybe enlarged above upscaled_db", false, optional(func(m photocheck.Metrics) *float64 { return m.Detail })},
	{"jpeg_q", "flagged below compressed_q (known values only)", false, func(ph *photo) (float64, bool) {
		q := ph.in.Metrics.JPEGQ
		return float64(q), q > 0
	}},
	{"face_px", "subject face side at training size; small_face below small_face", true, func(ph *photo) (float64, bool) {
		f := ph.item.Face
		return float64(faceOr(f).Px), f != nil && f.Count > 0
	}},
	{"face_height", "subject box height / image height; faces count from face_height; shot close from shot_close, bust from shot_bust, half from shot_half", true, func(ph *photo) (float64, bool) {
		f := ph.item.Face
		if f == nil || f.Box == nil {
			return 0, false
		}
		return f.Box[3], true
	}},
	{"face_sharp", "subject face sharpness; face blurry below face_floor", true, func(ph *photo) (float64, bool) {
		s := subjectFace(ph)
		if s == nil {
			return 0, false
		}
		return s.Sharp, true
	}},
	{"face_sharp_rel", "face sharp / the set's face median; face blurry below face_blur_rel (set_min faces)", true, func(ph *photo) (float64, bool) {
		return ph.faceSharpRel, ph.faceSharpRel > 0
	}},
	{"face_mean_y", "subject face mean luma; in shadow below backlit_max", true, func(ph *photo) (float64, bool) {
		s := subjectFace(ph)
		if s == nil || !measured(ph) {
			return 0, false
		}
		return s.MeanY, true
	}},
	{"face_luma_rel", "face mean_y / luma_mean; in shadow below backlit_rel", true, func(ph *photo) (float64, bool) {
		s := subjectFace(ph)
		if s == nil || !measured(ph) || ph.in.Metrics.LumaMean <= 0 {
			return 0, false
		}
		return s.MeanY / ph.in.Metrics.LumaMean, true
	}},
	{"face_smooth_rel", "(face sharp / frame sharp) / the set median; smoothed below smoothed_rel (align side from smoothed_side, set_min faces)", true, func(ph *photo) (float64, bool) {
		return ph.smoothRel, ph.smoothRel > 0
	}},
	{"face_cohesion", "median cosine to the other subjects; an outlier below outlier_floor or median - outlier_k x MAD (mad_floor); not_them also needs a top-3 cosine below same_cos (identity_min faces)", true, func(ph *photo) (float64, bool) {
		f := ph.item.Face
		if f == nil || f.Cohesion == nil {
			return 0, false
		}
		return *f.Cohesion, true
	}},
	{"face_ref_sim", "cosine to the reference face; not_them below same_cos", true, func(ph *photo) (float64, bool) {
		f := ph.item.Face
		if f == nil || f.RefSim == nil {
			return 0, false
		}
		return *f.RefSim, true
	}},
}

func faceOr(f *photocheck.ItemFace) photocheck.ItemFace {
	if f == nil {
		return photocheck.ItemFace{}
	}
	return *f
}

func metricReports(photos []*photo, faces bool) []Metric {
	folders := byFolder(photos)
	var out []Metric
	for _, d := range metricDefs {
		if d.face && !faces {
			continue
		}
		m := Metric{Name: d.name, Note: d.note, Rows: []Dist{}}
		for _, ps := range folders {
			var v []float64
			for _, ph := range ps {
				if x, ok := d.get(ph); ok {
					v = append(v, x)
				}
			}
			if len(v) == 0 {
				continue
			}
			ds := dist(v)
			ds.Folder = ps[0].folder
			m.Rows = append(m.Rows, ds)
		}
		out = append(out, m)
	}
	return out
}

// dist is the nearest-rank p5, p25, median, p75 and p95 of v.
func dist(v []float64) Dist {
	if len(v) == 0 {
		return Dist{}
	}
	s := slices.Clone(v)
	slices.Sort(s)
	rank := func(p float64) float64 {
		i := int(math.Ceil(p/100*float64(len(s)))) - 1
		return round4(s[min(max(i, 0), len(s)-1)])
	}
	return Dist{N: len(s), P5: rank(5), P25: rank(25), P50: rank(50), P75: rank(75), P95: rank(95)}
}

func round4(v float64) float64 { return math.Round(v*1e4) / 1e4 }

// hashed is a photo's pair-comparison data.
type hashed struct {
	ph         *photo
	h, flip    uint64
	thumb, mir []byte
}

func pairOf(a, b hashed) PairInfo {
	ncc := max(photocheck.NCC32(a.thumb, b.thumb), photocheck.NCC32(a.thumb, b.mir))
	return PairInfo{A: a.ph.rel, B: b.ph.rel, Hamming: min(photocheck.Hamming(a.h, b.h), photocheck.Hamming(a.h, b.flip)), NCC: round4(ncc)}
}

// pairReport compares every pair inside each dup/ set with pairs from
// different folders; above maxPairs cross pairs it takes an even sample.
func pairReport(photos []*photo, maxPairs int) *Pairs {
	var hs []hashed
	for _, ph := range photos {
		m := ph.in.Metrics
		h, err1 := photocheck.ParseHash(m.DHash)
		flip, err2 := photocheck.ParseHash(m.DHashFlip)
		th := photocheck.DecodeThumb32(m.Thumb32)
		if !m.Decoded || err1 != nil || err2 != nil || th == nil {
			continue
		}
		hs = append(hs, hashed{ph: ph, h: h, flip: flip, thumb: th, mir: photocheck.Mirror32(th)})
	}
	th := photocheck.DefaultThresholds()
	p := &Pairs{Worst: []PairInfo{}, Closest: []PairInfo{}}

	perFolder := map[string]int{}
	for _, x := range hs {
		perFolder[x.ph.folder]++
	}
	sq := 0
	for _, n := range perFolder {
		sq += n * n
	}
	p.CrossTotal = (len(hs)*len(hs) - sq) / 2
	stride := 1
	if maxPairs > 0 && p.CrossTotal > maxPairs {
		stride = (p.CrossTotal + maxPairs - 1) / maxPairs
	}

	var dupH, dupN, crossH, crossN []float64
	var dups, cross []PairInfo
	k := 0
	for i := range hs {
		for j := i + 1; j < len(hs); j++ {
			a, b := hs[i], hs[j]
			switch {
			case a.ph.folder == "dup" && a.ph.set == b.ph.set:
				pi := pairOf(a, b)
				dups = append(dups, pi)
				dupH, dupN = append(dupH, float64(pi.Hamming)), append(dupN, pi.NCC)
			case a.ph.folder != b.ph.folder:
				k++
				if (k-1)%stride != 0 {
					continue
				}
				pi := pairOf(a, b)
				cross = append(cross, pi)
				crossH, crossN = append(crossH, float64(pi.Hamming)), append(crossN, pi.NCC)
			}
		}
	}
	p.Dup, p.Cross = len(dups), len(cross)
	p.DupHamming, p.DupNCC = dist(dupH), dist(dupN)
	p.CrossHamming, p.CrossNCC = dist(crossH), dist(crossN)
	for _, x := range dups {
		if x.Hamming <= th.DupHamming {
			p.DupWithin++
			if x.NCC >= th.DupNCC {
				p.DupConfirmed++
			}
		}
		if x.Hamming <= th.SimilarHamming {
			p.DupSimilar++
		}
	}
	for _, x := range cross {
		if x.Hamming <= th.DupHamming {
			p.CrossWithin++
			if x.NCC >= th.DupNCC {
				p.CrossConfirm++
			}
		}
		if x.Hamming <= th.SimilarHamming {
			p.CrossSimilar++
		}
	}
	// dup/ pairs that would not be confirmed duplicates, farthest first
	for _, x := range dups {
		if x.Hamming > th.DupHamming || x.NCC < th.DupNCC {
			p.Worst = append(p.Worst, x)
		}
	}
	slices.SortStableFunc(p.Worst, func(a, b PairInfo) int {
		if a.Hamming != b.Hamming {
			return b.Hamming - a.Hamming
		}
		return cmpFloat(a.NCC, b.NCC)
	})
	p.Worst = p.Worst[:min(len(p.Worst), 10)]
	// unrelated pairs nearest to merging
	slices.SortStableFunc(cross, func(a, b PairInfo) int {
		if a.Hamming != b.Hamming {
			return a.Hamming - b.Hamming
		}
		return cmpFloat(b.NCC, a.NCC)
	})
	p.Closest = append(p.Closest, cross[:min(len(cross), 10)]...)
	return p
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// printReport writes the report as text.
func printReport(w io.Writer, r *Report, verbose bool) {
	faces := "faces not checked (pass -faces for the face flags)"
	if r.Faces {
		faces = "with face data"
	}
	scope := "judged as one dataset"
	if r.Each {
		scope = "each folder judged on its own"
	}
	fmt.Fprintf(w, "%d photos in %s: kind %s, R %d, %s, %s.\n", r.Photos, r.Dir, r.Kind, r.Resolution, scope, faces)
	fmt.Fprintln(w, "Thresholds are the shipped values in internal/photocheck/flags.go.")

	fmt.Fprintln(w, "\nFlag hits per folder")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "folder\tphotos\ttarget\thit\twarned\tgroups\tflags")
	for _, f := range r.Folders {
		hit := "-"
		target := "(no label)"
		if f.Target != "" {
			target = f.Target
			hit = fmt.Sprintf("%d (%s)", f.Hits, pct(f.Hits, f.Photos))
		}
		fmt.Fprintf(tw, "%s\t%d\t%s\t%s\t%d\t%s\t%s\n", f.Name, f.Photos, target, hit, f.Warned, countList(f.Groups), flagList(f.Flags))
	}
	tw.Flush()
	fmt.Fprintln(w, "groups counts photos in each kind of group, keepers included.")

	if r.Faces {
		fmt.Fprintln(w, "\nShot and head turn per folder"+noteValues("shot_close shot_bust shot_half yaw_front yaw_quarter yaw_min_align", r.Thresholds))
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "folder\tshot\thead turn")
		for _, f := range r.Folders {
			if len(f.Shot) > 0 || len(f.Yaw) > 0 {
				fmt.Fprintf(tw, "%s\t%s\t%s\n", f.Name, countList(f.Shot), countList(f.Yaw))
			}
		}
		tw.Flush()
	}

	for _, f := range r.Folders {
		if len(f.Misses) == 0 {
			continue
		}
		fmt.Fprintf(w, "\nMissed in %s (%s):\n", f.Name, f.Target)
		show := f.Misses
		if !verbose && len(show) > 5 {
			show = show[:5]
		}
		for _, m := range show {
			fl := "no flags"
			if len(m.Flags) > 0 {
				fl = joinFlags(m.Flags)
			}
			if len(m.Groups) > 0 {
				fl += "; in a " + strings.Join(m.Groups, " and a ") + " group"
			}
			fmt.Fprintf(w, "  %s  %s\n", m.File, fl)
		}
		if len(show) < len(f.Misses) {
			fmt.Fprintf(w, "  ... and %d more (-v lists them all)\n", len(f.Misses)-len(show))
		}
	}

	fmt.Fprintln(w, "\nMeasurements per folder (nearest-rank percentiles)")
	for _, m := range r.Metrics {
		if len(m.Rows) == 0 {
			continue
		}
		head := m.Name
		if m.Note != "" {
			head += ": " + m.Note + noteValues(m.Note, r.Thresholds)
		}
		fmt.Fprintf(w, "\n%s\n", head)
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
		fmt.Fprintln(tw, "folder\tn\tp5\tp25\tmedian\tp75\tp95\t")
		for _, d := range m.Rows {
			fmt.Fprintf(tw, "%s\t%d\t%s\t%s\t%s\t%s\t%s\t\n", d.Folder, d.N, num(d.P5), num(d.P25), num(d.P50), num(d.P75), num(d.P95))
		}
		tw.Flush()
	}

	if p := r.Pairs; p != nil {
		th := r.Thresholds
		fmt.Fprintln(w, "\nPairs: Hamming is min(direct, mirrored) dHash bits; NCC is max(direct, mirrored) on thumb32")
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
		fmt.Fprintln(tw, "\tpairs\tp5\tp25\tmedian\tp75\tp95\t")
		row := func(name string, d Dist) {
			fmt.Fprintf(tw, "%s\t%d\t%s\t%s\t%s\t%s\t%s\t\n", name, d.N, num(d.P5), num(d.P25), num(d.P50), num(d.P75), num(d.P95))
		}
		row("Hamming in dup/", p.DupHamming)
		row("Hamming across folders", p.CrossHamming)
		row("NCC in dup/", p.DupNCC)
		row("NCC across folders", p.CrossNCC)
		tw.Flush()
		if p.Cross < p.CrossTotal {
			fmt.Fprintf(w, "Across folders: an even sample of %d of %d pairs (-max-pairs).\n", p.Cross, p.CrossTotal)
		}
		fmt.Fprintf(w, "In dup/: %s within dup_hamming %s, %s confirmed (NCC >= dup_ncc %s), %s within similar_hamming %s.\n",
			frac(p.DupWithin, p.Dup), num(th["dup_hamming"]), frac(p.DupConfirmed, p.Dup), num(th["dup_ncc"]),
			frac(p.DupSimilar, p.Dup), num(th["similar_hamming"]))
		fmt.Fprintf(w, "Across folders: %s within dup_hamming, %s confirmed, %s within similar_hamming.\n",
			frac(p.CrossWithin, p.Cross), frac(p.CrossConfirm, p.Cross), frac(p.CrossSimilar, p.Cross))
		if len(p.Worst) > 0 {
			fmt.Fprintln(w, "dup/ pairs not confirmed as duplicates:")
			for _, x := range p.Worst {
				fmt.Fprintf(w, "  %s  %s  Hamming %d  NCC %s\n", x.A, x.B, x.Hamming, num(x.NCC))
			}
		}
		if len(p.Closest) > 0 {
			fmt.Fprintln(w, "Closest pairs across folders:")
			for _, x := range p.Closest[:min(len(p.Closest), 5)] {
				fmt.Fprintf(w, "  %s  %s  Hamming %d  NCC %s\n", x.A, x.B, x.Hamming, num(x.NCC))
			}
		}
	}
}

// noteValues appends the current value of every threshold key a note names.
func noteValues(note string, th map[string]float64) string {
	var keys []string
	seen := map[string]bool{}
	for _, word := range strings.FieldsFunc(note, func(r rune) bool { return r == ' ' || r == ',' || r == '(' || r == ')' || r == ';' }) {
		if v, ok := th[word]; ok && !seen[word] {
			seen[word] = true
			keys = append(keys, word+" "+num(v))
		}
	}
	if len(keys) == 0 {
		return ""
	}
	return " [" + strings.Join(keys, ", ") + "]"
}

func flagList(m map[photocheck.Flag]int) string {
	s := make(map[string]int, len(m))
	for f, n := range m {
		s[string(f)] = n
	}
	return countList(s)
}

// countList is "a 3, b 1": most first, then by name; "none" when empty.
func countList(m map[string]int) string {
	if len(m) == 0 {
		return "none"
	}
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.SortFunc(ks, func(a, b string) int {
		if m[a] != m[b] {
			return m[b] - m[a]
		}
		return strings.Compare(a, b)
	})
	parts := make([]string, len(ks))
	for i, k := range ks {
		parts[i] = fmt.Sprintf("%s %d", k, m[k])
	}
	return strings.Join(parts, ", ")
}

func joinFlags(fs []photocheck.Flag) string {
	s := make([]string, len(fs))
	for i, f := range fs {
		s[i] = string(f)
	}
	return strings.Join(s, ", ")
}

func pct(n, of int) string {
	if of == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f%%", 100*float64(n)/float64(of))
}

func frac(n, of int) string { return fmt.Sprintf("%d/%d", n, of) }

// num prints a value with fewer decimals the larger it is (4 below 10, 2
// below 100, else 1) and no trailing zeros.
func num(v float64) string {
	prec := 4
	switch a := math.Abs(v); {
	case a >= 100:
		prec = 1
	case a >= 10:
		prec = 2
	}
	s := strconv.FormatFloat(v, 'f', prec, 64)
	if !strings.Contains(s, ".") {
		return s
	}
	s = strings.TrimRight(s, "0")
	return strings.TrimSuffix(s, ".")
}

// writeFixtures writes the generated photo fixtures (internal/testutil) as a
// labelled tree, so the tool runs on them the way it runs on a real set:
// ten good camera photos plus a high-key studio portrait, and the photos
// each label stands for. misc/ holds fixtures with no label of their own.
func writeFixtures(dir string) (int, error) {
	const w, h = 1200, 900
	cam := func(img image.Image, q, day int) []byte {
		t := fmt.Sprintf("2024:05:%02d 10:00:00", day)
		return testutil.WithExif(testutil.JPEG(img, q), 1, t, "Apple", "iPhone 15")
	}
	files := map[string][]byte{
		"good/white_backdrop.jpg": cam(testutil.WhiteBackdrop(900, 1200), 90, 11),

		"blur/blur_1.jpg": cam(testutil.Blur(testutil.Scene(11, w, h), 3), 90, 12),
		"blur/blur_2.jpg": cam(testutil.Blur(testutil.Scene(12, w, h), 3), 90, 13),
		"blur/blur_3.jpg": cam(testutil.Blur(testutil.Scene(13, w, h), 4), 90, 14),

		"dup/a/orig.jpg":   cam(testutil.Scene(20, w, h), 90, 15),
		"dup/a/half.jpg":   testutil.JPEG(testutil.Resize(testutil.Scene(20, w, h), w/2, h/2), 70),
		"dup/a/mirror.jpg": testutil.JPEG(testutil.FlipH(testutil.Scene(20, w, h)), 90),
		"dup/b/orig.jpg":   cam(testutil.Scene(21, w, h), 90, 16),
		"dup/b/resave.jpg": testutil.JPEG(testutil.Scene(21, w, h), 60),

		"screens/screenshot.png": testutil.PNG(testutil.UIScreen(1170, 2532)),
		"screens/frame.jpg":      testutil.JPEG(testutil.Scene(22, 1920, 1080), 90),

		"dark/dark_1.jpg": cam(testutil.Darken(testutil.Scene(23, w, h)), 90, 17),
		"dark/dark_2.jpg": cam(testutil.Darken(testutil.Scene(24, w, h)), 90, 18),

		"bright/bright_1.jpg": cam(testutil.Brighten(testutil.Scene(25, w, h)), 90, 19),
		"bright/bright_2.jpg": cam(testutil.Brighten(testutil.Scene(26, w, h)), 90, 20),

		"mono/gray.jpg":  cam(testutil.Gray(testutil.Scene(27, w, h)), 90, 21),
		"mono/sepia.jpg": cam(testutil.Sepia(testutil.Scene(28, w, h)), 90, 22),

		"misc/tiny.jpg":        testutil.JPEG(testutil.Scene(29, 300, 400), 90),
		"misc/compressed.jpg":  testutil.JPEG(testutil.Scene(30, w, h), 40),
		"misc/letterbox.jpg":   testutil.JPEG(testutil.Letterbox(testutil.Scene(31, w, h), 0.1, color.Black), 90),
		"misc/flat.jpg":        testutil.JPEG(testutil.Flatten(testutil.Scene(32, w, h)), 90),
		"misc/saturated.jpg":   testutil.JPEG(testutil.Saturate(testutil.Scene(33, w, h)), 90),
		"misc/noisy.jpg":       testutil.JPEG(testutil.Noise(testutil.Scene(34, w, h), 12, 1), 90),
		"misc/upscaled.jpg":    testutil.JPEG(testutil.Upscaled(testutil.Scene(35, 3000, 2000), 2), 90),
		"misc/transparent.png": testutil.TransparentPNG(testutil.Scene(36, w, h)),
	}
	for i := range 10 {
		files[fmt.Sprintf("good/scene_%02d.jpg", i+1)] = cam(testutil.Scene(i+1, w, h), 90, i+1)
	}
	if err := testutil.WriteTree(dir, files); err != nil {
		return 0, err
	}
	return len(files), nil
}
