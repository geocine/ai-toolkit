package photocheck

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"maps"
	"math"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
)

// The photo service: stored dataset photos analysed in the background pool
// and cached by content, judged by the same Judge the studio's metrics go
// through, plus the fixes, settings, forget and GC around them.

// settleWait is how long a review made after a change (settings, fix)
// waits for photos it has not analysed yet.
const settleWait = 3 * time.Second

// maxCaption caps a caption file a review reads.
const maxCaption = 1 << 20

// UserError is a request the caller can correct: a 400, or a 409 when
// Conflict is set. Its message is written for the user.
type UserError struct {
	Conflict bool
	Message  string
}

func (e *UserError) Error() string { return e.Message }

func badRequest(format string, args ...any) error {
	return &UserError{Message: fmt.Sprintf(format, args...)}
}

func conflict(format string, args ...any) error {
	return &UserError{Conflict: true, Message: fmt.Sprintf(format, args...)}
}

func isNotExist(err error) bool { return errors.Is(err, fs.ErrNotExist) }

// dirLocks serialises the writers of one dataset folder's photo state
// (ingest, settings, fixes, forget).
var dirLocks sync.Map

func lockDir(dir string) func() {
	v, _ := dirLocks.LoadOrStore(filepath.Clean(dir), &sync.Mutex{})
	mu := v.(*sync.Mutex)
	mu.Lock()
	return mu.Unlock
}

// snapshot is a dataset's photos and photo state at one moment.
type snapshot struct {
	dir   string
	items []*entry
	aside []*entry
	set   *state.PhotoSet
	job   *store.Job
	faces map[string]*FaceData // by sha256
	from  map[string]string    // set-aside rel to the dataset path it came from, when the two differ
}

// originOf is the dataset path a set-aside photo came from.
func (s *snapshot) originOf(e *entry) string {
	if from := s.from[e.rel]; from != "" {
		return from
	}
	return e.rel
}

// clone copies the snapshot so a dry run of fix ops can change it freely.
// The photo state is shared: only an applied op writes to it.
func (s *snapshot) clone() *snapshot {
	c := *s
	c.items, c.aside = cloneEntries(s.items), cloneEntries(s.aside)
	c.from = maps.Clone(s.from)
	return &c
}

func cloneEntries(es []*entry) []*entry {
	out := make([]*entry, len(es))
	for i, e := range es {
		x := *e
		out[i] = &x
	}
	return out
}

// loadSet is the folder's photo state, an empty one when it has none.
func loadSet(ctx context.Context, dir string) (*state.PhotoSet, error) {
	set, err := state.GetPhotoSet(ctx, dir)
	if err != nil {
		return nil, err
	}
	if set == nil {
		set = &state.PhotoSet{Dir: filepath.Clean(dir), Decisions: map[string]string{}, Derived: map[string]string{}}
	}
	return set, nil
}

// load reads a dataset: it ingests a finished face job, walks the photos,
// reads the cached analysis and queues the misses (urgently, waiting up to
// wait, when analyse is set; set-aside photos only as warm-up), and marks
// what it read as seen.
func load(ctx context.Context, dir string, wait time.Duration, analyse bool) (*snapshot, error) {
	return loadWith(ctx, dir, wait, analyse, false)
}

// loadWith is load for a caller that may already hold the folder's lock.
func loadWith(ctx context.Context, dir string, wait time.Duration, analyse, locked bool) (*snapshot, error) {
	dir = filepath.Clean(dir)
	st, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("%s is not a folder", dir)
	}
	s := &snapshot{dir: dir}
	unlock := func() {}
	if !locked {
		unlock = lockDir(dir)
	}
	s.set, err = loadSet(ctx, dir)
	if err == nil && analyse {
		var ierr error
		s.job, ierr = ingestFaces(ctx, dir, s.set)
		if ierr != nil {
			log.Printf("photocheck: face results for %s: %v", dir, ierr)
		}
	}
	unlock()
	if err != nil {
		return nil, err
	}
	if s.items, s.aside, err = walkPhotos(dir); err != nil {
		return nil, err
	}
	if err := resolve(ctx, s.items, wait, analyse, true); err != nil {
		return nil, err
	}
	if err := resolve(ctx, s.aside, 0, analyse, false); err != nil {
		return nil, err
	}
	s.from = readAsideFrom(dir, s.aside)
	shas := s.shas(true)
	if err := state.TouchPhotos(ctx, shas, time.Now()); err != nil {
		return nil, err
	}
	rows, err := state.GetPhotoFaces(ctx, shas, FaceAlgo)
	if err != nil {
		return nil, err
	}
	s.faces = map[string]*FaceData{}
	for sha, b := range rows {
		var fd FaceData
		if json.Unmarshal(b, &fd) == nil {
			s.faces[sha] = &fd
		}
	}
	return s, nil
}

// resolve fills in the entries' sha256 and metrics from the cache. With
// analyse, the misses go to the pool (urgent ones first in line), and the
// call waits up to wait for them; whatever is still running stays pending.
func resolve(ctx context.Context, es []*entry, wait time.Duration, analyse, urgent bool) error {
	var known []string
	for _, e := range es {
		sha, ok, err := state.LookupFileSHA256(ctx, e.abs, e.info)
		if err != nil && !isNotExist(err) {
			return err
		}
		if ok {
			e.sha = sha
			known = append(known, sha)
		}
	}
	rows, err := state.GetPhotoMetrics(ctx, known, MetricsVersion)
	if err != nil {
		return err
	}
	type miss struct {
		e *entry
		t *task
	}
	var misses []miss
	for _, e := range es {
		if e.sha != "" {
			if m, ok := decodeMetrics(rows[e.sha], e.rel); ok {
				e.m = &m
				continue
			}
		}
		if analyse {
			misses = append(misses, miss{e, analysis.analyse(e.abs, e.rel, e.info, urgent)})
		}
	}
	if len(misses) == 0 {
		return nil
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	expired := wait <= 0
	for _, x := range misses {
		if !expired {
			select {
			case <-x.t.done:
			case <-timer.C:
				expired = true
			case <-ctx.Done():
				expired = true
			}
		}
		select {
		case <-x.t.done:
		default:
			continue
		}
		if x.t.err != nil {
			m := Metrics{Version: MetricsVersion, File: x.e.rel, Bytes: x.e.info.Size(), Source: "server",
				Format: strings.TrimPrefix(strings.ToLower(path.Ext(x.e.rel)), "."), Orientation: 1, Error: "cannot read this file"}
			x.e.m = &m
			continue
		}
		m := x.t.m
		m.File = x.e.rel
		x.e.m, x.e.sha = &m, x.t.sha
	}
	return nil
}

// shas lists the known sha256 of the photos (and of the set-aside ones too
// when aside is set), each once.
func (s *snapshot) shas(aside bool) []string {
	seen := map[string]bool{}
	var out []string
	add := func(es []*entry) {
		for _, e := range es {
			if e.sha != "" && !seen[e.sha] {
				seen[e.sha] = true
				out = append(out, e.sha)
			}
		}
	}
	add(s.items)
	if aside {
		add(s.aside)
	}
	return out
}

// fileOf maps sha256 to the first photo with that content.
func (s *snapshot) fileOf() map[string]string {
	out := map[string]string{}
	for _, e := range s.items {
		if e.sha != "" && out[e.sha] == "" {
			out[e.sha] = e.rel
		}
	}
	return out
}

func captionExt(opts Options) string {
	if ext := strings.TrimPrefix(opts.CaptionExt, "."); ext != "" {
		return ext
	}
	return "txt"
}

// readCaption is the caption next to a photo, nil when there is none.
func readCaption(abs, ext string) *string {
	p := strings.TrimSuffix(abs, filepath.Ext(abs)) + "." + ext
	st, err := os.Stat(p)
	if err != nil || !st.Mode().IsRegular() || st.Size() > maxCaption {
		return nil
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return nil
	}
	c := string(b)
	return &c
}

// inputs are the judge's inputs for the dataset's photos.
func (s *snapshot) inputs(opts Options) []Input {
	ext := captionExt(opts)
	files := s.fileOf()
	in := make([]Input, 0, len(s.items))
	for _, e := range s.items {
		x := Input{Caption: readCaption(e.abs, ext)}
		if e.m == nil {
			x.Metrics = Metrics{Version: MetricsVersion, File: e.rel, Bytes: e.info.Size(), Source: "server"}
			x.Pending = true
			in = append(in, x)
			continue
		}
		x.Metrics = *e.m
		x.Metrics.File = e.rel
		if e.sha != "" {
			x.Face = s.faces[e.sha]
			x.Decision = Decision(s.set.Decisions[e.sha])
			if parent := s.set.Derived[e.sha]; parent != "" && files[parent] != e.rel {
				x.Parent = files[parent]
			}
		}
		in = append(in, x)
	}
	return in
}

// setState is the judge's view of the dataset's photo state.
func (s *snapshot) setState(opts Options) SetState {
	id := storedIdentity(s.set)
	shas := s.shas(false)
	faced := map[string]bool{}
	for sha := range s.faces {
		faced[sha] = true
	}
	return SetState{
		Kind:      Kind(s.set.Kind),
		Reference: s.set.Reference,
		Identity:  id,
		Faces:     facesState(s.job, id, shas, faced),
		SetAside:  s.asideItems(opts),
		Convert:   ConvertFormats(),
	}
}

// asideItems lists the set-aside photos with their flags, judged against
// the dataset's photos (which join only the duplicate pass), so a copy of a
// photo still in the set says so.
func (s *snapshot) asideItems(opts Options) []SetAsideItem {
	out := []SetAsideItem{}
	if len(s.aside) == 0 {
		return out
	}
	var in []Input
	for _, e := range s.items {
		if e.m != nil {
			m := *e.m
			m.File = e.rel
			in = append(in, Input{Metrics: m, Existing: true})
		}
	}
	for _, e := range s.aside {
		if e.m == nil {
			in = append(in, Input{Metrics: Metrics{Version: MetricsVersion, File: asideFile(e.rel)}, Pending: true})
			continue
		}
		m := *e.m
		m.File = asideFile(e.rel)
		in = append(in, Input{Metrics: m, Decision: Decision(s.set.Decisions[e.sha])})
	}
	r := Judge(in, SetState{Kind: Kind(s.set.Kind), Convert: ConvertFormats()},
		Options{Kind: opts.Kind, Resolutions: opts.Resolutions, CaptionExt: opts.CaptionExt})
	flags := map[string][]Flag{}
	for _, it := range r.Items {
		flags[it.File] = it.Flags
	}
	for _, e := range s.aside {
		f := flags[asideFile(e.rel)]
		if f == nil {
			f = []Flag{}
		}
		out = append(out, SetAsideItem{File: asideFile(e.rel), From: s.originOf(e), Flags: f})
	}
	return out
}

// judge judges the snapshot.
func (s *snapshot) judge(opts Options) *Review {
	return Judge(s.inputs(opts), s.setState(opts), opts)
}

// ReviewDataset judges a dataset's stored photos (GET photos/check,
// preflight; the plan's photocheck.Review, renamed because Review is the
// result type):
// the trainer's view of the folder plus .set_aside/ as its own list, from
// the cache, analysing misses in the background pool. It waits up to
// opts.Wait for them; photos still being analysed come back pending, and
// the analysis carries on, so callers poll until Pending is 0. A face job
// that finished since the last review is ingested first.
func ReviewDataset(ctx context.Context, dir string, opts Options) (*Review, error) {
	if err := checkKind(opts.Kind); err != nil {
		return nil, err
	}
	s, err := load(ctx, dir, opts.Wait, true)
	if err != nil {
		return nil, err
	}
	return s.judge(opts), nil
}

// Warm queues photos for analysis behind any a request waits on, so the
// first review after an upload finds them cached. rels are relative to dir;
// files that are not photos are ignored.
func Warm(dir string, rels []string) {
	for _, rel := range rels {
		if !isPhoto(rel) {
			continue
		}
		abs := filepath.Join(dir, filepath.FromSlash(rel))
		st, err := os.Stat(abs)
		if err != nil || !st.Mode().IsRegular() {
			continue
		}
		analysis.analyse(abs, filepath.ToSlash(rel), st, false)
	}
}

// Assess judges metrics measured elsewhere (POST /v1/photos/assess) with
// the same rules as ReviewDataset. With datasetDir, the dataset's cached photos
// join the duplicate pass as existing items; nothing is measured on the
// request (misses are only queued for warm-up). No items returns an empty
// review that still lists the formats the server converts.
func Assess(ctx context.Context, req AssessRequest, datasetDir string) (*Review, error) {
	if len(req.Items) > MaxAssessItems {
		return nil, badRequest("at most %d photos per assess, got %d", MaxAssessItems, len(req.Items))
	}
	if err := checkKind(req.Kind); err != nil {
		return nil, err
	}
	res, err := ParseResolutions(req.Resolution)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	if req.Pick < 0 || req.Pick > MaxAssessItems {
		return nil, badRequest("pick must be between 0 and %d", MaxAssessItems)
	}
	opts := Options{Kind: req.Kind, Resolutions: res, Pick: req.Pick}
	set := SetState{Convert: ConvertFormats(), Faces: FacesState{Available: FacesAvailable, State: "none"}}
	var stored *snapshot
	if datasetDir != "" {
		if stored, err = load(ctx, datasetDir, 0, false); err != nil {
			return nil, err
		}
		set.Kind, set.Reference = Kind(stored.set.Kind), stored.set.Reference
		var misses []string
		for _, e := range stored.items {
			if e.m == nil {
				misses = append(misses, e.rel)
			}
		}
		Warm(stored.dir, misses)
	}
	var in []Input
	seen := map[string]bool{}
	for i, raw := range req.Items {
		m, err := cleanMetrics(raw)
		if err != nil {
			return nil, badRequest("items[%d]: %v", i, err)
		}
		if seen[m.File] {
			return nil, badRequest("items[%d]: %s is listed twice", i, m.File)
		}
		seen[m.File] = true
		x := Input{Metrics: m, Decision: req.Decisions[m.File]}
		if x.Decision != "" && x.Decision != DecisionKeep && x.Decision != DecisionIntended {
			return nil, badRequest("decisions[%s] must be keep or intended", m.File)
		}
		if stored != nil && x.Decision == "" && m.SHA256 != "" {
			x.Decision = Decision(stored.set.Decisions[m.SHA256])
		}
		if c, ok := req.Captions[m.File]; ok {
			x.Caption = &c
		}
		in = append(in, x)
	}
	if stored != nil {
		for _, e := range stored.items {
			if e.m != nil && !seen[e.rel] {
				in = append(in, Input{Metrics: *e.m, Existing: true})
			}
		}
	}
	return Judge(in, set, opts), nil
}

func checkKind(k Kind) error {
	switch k {
	case "", KindPerson, KindStyle, KindObject:
		return nil
	}
	return badRequest("kind must be person, style or object")
}

// cleanMetrics validates one posted Metrics and clamps its values, so a
// client cannot push the judge outside the ranges it was written for.
func cleanMetrics(m Metrics) (Metrics, error) {
	f := strings.ReplaceAll(strings.TrimSpace(m.File), `\`, "/")
	if f == "" {
		return m, errors.New("file is required")
	}
	if len(f) > 512 {
		return m, errors.New("file name is too long")
	}
	f = strings.TrimPrefix(path.Clean("/"+f), "/")
	switch m.Version {
	case 0:
		m.Version = MetricsVersion
	case MetricsVersion:
	default:
		return m, fmt.Errorf("metrics version %d is not supported (this server measures version %d)", m.Version, MetricsVersion)
	}
	m.File = f
	if m.Source != "server" {
		m.Source = "browser"
	}
	m.Format = clip(strings.ToLower(m.Format), 16)
	m.Mode = clip(m.Mode, 16)
	m.Error = clip(m.Error, 200)
	if b, err := hex.DecodeString(m.SHA256); err != nil || len(b) != 32 {
		m.SHA256 = ""
	}
	m.SHA256 = strings.ToLower(m.SHA256)
	m.Bytes = max(0, min(m.Bytes, 1<<40))
	m.Width = max(0, min(m.Width, 1<<20))
	m.Height = max(0, min(m.Height, 1<<20))
	if m.Orientation < 1 || m.Orientation > 8 {
		m.Orientation = 1
	}
	if _, err := ParseHash(m.DHash); err != nil {
		m.DHash = ""
	}
	if _, err := ParseHash(m.DHashFlip); err != nil {
		m.DHashFlip = ""
	}
	if DecodeThumb32(m.Thumb32) == nil {
		m.Thumb32 = ""
	}
	if l, err := base64.StdEncoding.DecodeString(m.Layout); err != nil || len(l) != 3*layoutCells {
		m.Layout = ""
	}
	m.Sharp = clampMetric(m.Sharp, 1e7)
	m.LumaMean = clampMetric(m.LumaMean, 255)
	m.LumaStd = clampMetric(m.LumaStd, 255)
	m.ClipLo = clampMetric(m.ClipLo, 1)
	m.ClipHi = clampMetric(m.ClipHi, 1)
	m.ClipHiCenter = clampMetric(m.ClipHiCenter, 1)
	m.Alpha = clampMetric(m.Alpha, 1)
	m.Flat = clampMetric(m.Flat, 1)
	m.Bars = clampMetric(m.Bars, 1)
	m.Colorful = clampMetric(m.Colorful, 1000)
	m.ChromaSpread = clampMetric(m.ChromaSpread, 1000)
	for _, p := range []**float64{&m.Noise, &m.Detail} {
		if *p != nil {
			v := clampMetric(**p, 1000)
			*p = &v
		}
	}
	m.JPEGQ = max(0, min(m.JPEGQ, 100))
	m.PNGColorType = max(0, min(m.PNGColorType, 6))
	m.PNGBitDepth = max(0, min(m.PNGBitDepth, 16))
	if e := m.Exif; e != nil {
		x := ExifInfo{Time: clip(e.Time, 32), Make: clip(e.Make, 128), Model: clip(e.Model, 128), Software: clip(e.Software, 128)}
		m.Exif = &x
		if x == (ExifInfo{}) {
			m.Exif = nil
		}
	}
	return m, nil
}

func clampMetric(v, hi float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
		return 0
	}
	return math.Min(v, hi)
}

func clip(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// Forget drops a dataset's photo analysis (DELETE photos/analysis; dataset
// delete calls it before removing the folder): the metrics and face rows of
// its photos, set-aside ones included, and its photo_sets row with the
// reference, decisions and face similarities. The photos are untouched.
func Forget(ctx context.Context, dir string) error {
	dir = filepath.Clean(dir)
	defer lockDir(dir)()
	seen := map[string]bool{}
	var shas []string
	add := func(sha string) {
		if sha != "" && !seen[sha] {
			seen[sha] = true
			shas = append(shas, sha)
		}
	}
	items, aside, err := walkPhotos(dir)
	if err != nil && !isNotExist(err) {
		return err
	}
	for _, e := range append(items, aside...) {
		if sha, ok, _ := state.LookupFileSHA256(ctx, e.abs, e.info); ok {
			add(sha)
		}
	}
	set, err := state.GetPhotoSet(ctx, dir)
	if err != nil {
		return err
	}
	if set != nil {
		add(set.Reference)
		for sha := range set.Decisions {
			add(sha)
		}
		for crop, parent := range set.Derived {
			add(crop)
			add(parent)
		}
		if id := storedIdentity(set); id != nil {
			for _, f := range id.Faces {
				sha, _, _ := strings.Cut(f, ":")
				add(sha)
			}
		}
	}
	return state.ForgetPhotos(ctx, dir, shas)
}
