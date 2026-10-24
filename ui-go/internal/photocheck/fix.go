package photocheck

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"log"
	"math"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/collation"
	"github.com/geocine/aitoolkit/internal/state"
)

// Photo fixes (POST photos/fix). Ops run in order against the dataset as
// the earlier ops left it. Without apply nothing is written: the changes
// are worked out on a copy of the dataset in memory and the review is the
// judge's view of that copy. Photos are never deleted: set_aside moves them
// (with their caption and thumbs) into .set_aside/, and a format change
// moves the original there too. Only remove_face_crops deletes, and only
// the crops face_crops wrote. Every change purges the photo's thumbs and
// trainer caches. The handler checks for training and caption jobs first.

// faceCropSuffix names a face crop: <stem>__face.jpg.
const faceCropSuffix = "__face"

// Face crop geometry (section 9).
const (
	cropScale   = 2.6  // crop side over face side
	cropDrop    = 0.25 // centre shift down, in face heights
	cropMinSide = 512  // source px
	cropMaxFH   = 0.20 // face height over image height: the offer's limit
)

// ApplyFix runs a photos/fix request on a dataset folder.
func ApplyFix(ctx context.Context, dir string, req FixRequest) (*Fix, error) {
	if len(req.Ops) == 0 {
		return nil, badRequest("ops is required; valid ops: %s", strings.Join(FixOps, ", "))
	}
	for _, op := range req.Ops {
		if !slices.Contains(FixOps, op.Op) {
			return nil, badRequest("unknown op %q; valid ops: %s", op.Op, strings.Join(FixOps, ", "))
		}
	}
	if err := checkKind(req.Kind); err != nil {
		return nil, err
	}
	res, err := ParseResolutions(req.Resolution)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	opts := Options{Kind: req.Kind, Resolutions: res}
	dir = filepath.Clean(dir)
	unlock := func() {}
	if req.Apply {
		unlock = lockDir(dir)
	}
	s, err := loadWith(ctx, dir, settleWait, true, req.Apply)
	if err != nil {
		unlock()
		return nil, err
	}
	ops := make([]FixOp, len(req.Ops))
	for i, op := range req.Ops {
		if len(op.Files) == 0 {
			op.Files = req.Files
		}
		ops[i] = op
	}
	if req.Apply {
		// a dry run on a copy first, so a bad parameter in any op fails the
		// request before the first file moves
		dry := &fixer{ctx: ctx, s: s.clone(), opts: opts, changes: []Change{}, dirty: true}
		for _, op := range ops {
			if err := dry.run(op); err != nil {
				unlock()
				return nil, err
			}
		}
	}
	f := &fixer{ctx: ctx, s: s, apply: req.Apply, opts: opts, changes: []Change{}, dirty: true}
	for _, op := range ops {
		if err := f.run(op); err != nil {
			// what the earlier ops did on disk stays, and so does the state
			// they carried to new content
			if serr := f.save(); serr != nil {
				log.Printf("photocheck: fix %s: %v", dir, serr)
			}
			unlock()
			return nil, err
		}
	}
	if err := f.save(); err != nil {
		unlock()
		return nil, err
	}
	unlock()
	out := &Fix{Changes: f.changes, Applied: req.Apply}
	if !req.Apply {
		out.Review = f.review()
		return out, nil
	}
	if out.Review, err = ReviewDataset(ctx, dir, Options{Kind: req.Kind, Resolutions: res, Wait: settleWait}); err != nil {
		return nil, err
	}
	return out, nil
}

// fixer runs one request's ops over a snapshot it keeps up to date.
type fixer struct {
	ctx         context.Context
	s           *snapshot
	apply       bool
	opts        Options
	changes     []Change
	setChanged  bool
	fromChanged bool // s.from changed: the .set_aside/ origins need writing

	dirty bool // the snapshot changed since r was judged
	in    []Input
	r     *Review
}

// review judges the snapshot as the ops so far left it.
func (f *fixer) review() *Review {
	if f.dirty || f.r == nil {
		f.in = f.s.inputs(f.opts)
		f.r = Judge(f.in, f.s.setState(f.opts), f.opts)
		f.dirty = false
	}
	return f.r
}

// save writes what the applied ops changed besides the files: the photo
// state and the set-aside origins.
func (f *fixer) save() error {
	if !f.apply {
		return nil
	}
	var errs []error
	if f.fromChanged {
		errs = append(errs, writeAsideFrom(f.s.dir, f.s.from, f.s.aside))
		f.fromChanged = false
	}
	if f.setChanged {
		f.s.set.UpdatedAt = 0
		errs = append(errs, state.PutPhotoSet(f.ctx, f.s.set))
		f.setChanged = false
	}
	return errors.Join(errs...)
}

func (f *fixer) change(c Change) {
	f.changes = append(f.changes, c)
	f.dirty = true
}

func (f *fixer) run(op FixOp) error {
	switch op.Op {
	case "set_aside":
		return f.setAside(op.Files)
	case "restore":
		return f.restore(op.Files, op.All)
	case "set_aside_flagged":
		return f.setAsideFlagged(op.Flags)
	case "keep_best":
		return f.keepBest(op.N)
	case "bake_orientation":
		return f.bake(op.Files)
	case "convert":
		return f.convert(op.Files)
	case "rename_same_stem":
		return f.renameSameStem()
	case "face_crops":
		return f.faceCrops(op.Max)
	case "remove_face_crops":
		return f.removeFaceCrops()
	}
	return badRequest("unknown op %q; valid ops: %s", op.Op, strings.Join(FixOps, ", "))
}

// item is the dataset photo at rel, nil when none.
func (f *fixer) item(rel string) *entry {
	for _, e := range f.s.items {
		if e.rel == rel {
			return e
		}
	}
	return nil
}

func (f *fixer) asideEntry(rel string) *entry {
	for _, e := range f.s.aside {
		if e.rel == rel {
			return e
		}
	}
	return nil
}

// takenIn reports a name in use in the dataset (aside false) or in
// .set_aside/: by a photo in the snapshot, or by any file on disk.
func (f *fixer) takenIn(aside bool) func(rel string) bool {
	return func(rel string) bool {
		list, root := f.s.items, f.s.dir
		if aside {
			list, root = f.s.aside, filepath.Join(f.s.dir, SetAsideDir)
		}
		for _, e := range list {
			if strings.EqualFold(e.rel, rel) {
				return true
			}
		}
		_, err := os.Lstat(filepath.Join(root, filepath.FromSlash(rel)))
		return err == nil
	}
}

// uniqueName is rel, or <stem>_2<ext>, _3, … when rel is taken.
func uniqueName(rel string, taken func(string) bool) string {
	if !taken(rel) {
		return rel
	}
	ext := path.Ext(rel)
	stem := strings.TrimSuffix(rel, ext)
	for n := 2; ; n++ {
		if c := fmt.Sprintf("%s_%d%s", stem, n, ext); !taken(c) {
			return c
		}
	}
}

// destTaken reports a name a photo moving into the dataset (aside false)
// or into .set_aside/ cannot take: a photo has it, or a photo or a caption
// there has its stem. A move then never replaces a caption there, and
// never hands one to a photo that did not have it.
func (f *fixer) destTaken(aside bool) func(rel string) bool {
	list, root := f.s.items, f.s.dir
	if aside {
		list, root = f.s.aside, filepath.Join(f.s.dir, SetAsideDir)
	}
	taken := f.takenIn(aside)
	return func(rel string) bool {
		if taken(rel) {
			return true
		}
		stem := stemOf(rel)
		for _, e := range list {
			if strings.EqualFold(stemOf(e.rel), stem) {
				return true
			}
		}
		return exists(filepath.Join(root, filepath.FromSlash(stem)+".txt"))
	}
}

// stemOf is a dataset path without its extension.
func stemOf(rel string) string { return strings.TrimSuffix(rel, path.Ext(rel)) }

// captionFor is the caption file of the photo at abs.
func captionFor(abs string) string { return strings.TrimSuffix(abs, filepath.Ext(abs)) + ".txt" }

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

// What move does with a photo's caption.
const (
	captionStays = iota // it stays where it is
	captionMoves
	captionCopies // another photo there has the same stem and keeps it
)

// captionMode is how the photo e, moving out of list, takes its caption:
// a copy when another photo in list shares its stem.
func captionMode(e *entry, list []*entry) int {
	for _, x := range list {
		if x != e && strings.EqualFold(stemOf(x.rel), stemOf(e.rel)) {
			return captionCopies
		}
	}
	return captionMoves
}

// move renames a photo (and its .txt, by caption) from one path to
// another, moving its thumbs along and purging the trainer caches at both
// places. It never replaces a caption at the new place.
func move(from, to string, caption int) error {
	src, dst := captionFor(from), captionFor(to)
	withCaption := caption != captionStays && exists(src)
	if withCaption && exists(dst) {
		return fmt.Errorf("%s already has a caption", filepath.Base(dst))
	}
	if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
		return err
	}
	if err := os.Rename(from, to); err != nil {
		return err
	}
	if withCaption {
		var err error
		if caption == captionCopies {
			err = copyFile(src, dst)
		} else {
			err = os.Rename(src, dst)
		}
		if err != nil {
			return err
		}
	}
	moveThumbs(from, to)
	purgeCaches(from)
	purgeCaches(to)
	return nil
}

// shaOf is a photo's sha256, hashed now when the snapshot does not know it
// yet (a photo moved since its last review, say).
func (f *fixer) shaOf(e *entry) string {
	if e.sha == "" && e.info != nil {
		if sum, err := state.FileSHA256(f.ctx, e.abs, nil); err == nil {
			e.sha = sum
		}
	}
	return e.sha
}

// fill gives a photo the snapshot has no metrics for (a set-aside one is
// only warmed up) its cached metrics, so the ops after a restore judge it.
func (f *fixer) fill(e *entry) {
	if e.m != nil || f.shaOf(e) == "" {
		return
	}
	rows, err := state.GetPhotoMetrics(f.ctx, []string{e.sha}, MetricsVersion)
	if err != nil {
		return
	}
	if m, ok := decodeMetrics(rows[e.sha], e.rel); ok {
		e.m = &m
	}
}

// settle waits for the pool to let go of a photo before it is moved,
// rewritten or removed.
func (f *fixer) settle(e *entry) { analysis.settle(f.ctx, e.abs, e.info) }

func (f *fixer) dropItem(e *entry) {
	f.s.items = slices.DeleteFunc(f.s.items, func(x *entry) bool { return x == e })
}

func (f *fixer) dropAside(e *entry) {
	f.s.aside = slices.DeleteFunc(f.s.aside, func(x *entry) bool { return x == e })
}

func sortEntries(es []*entry) {
	slices.SortStableFunc(es, func(a, b *entry) int { return collation.Compare(a.rel, b.rel) })
}

// moveAside moves one photo into .set_aside/ (a clash there, by name or
// by stem, gets _2, _3, and the path it came from is recorded). With
// caption, its caption goes along, or a copy when another photo shares it.
// Its face crops go with it.
func (f *fixer) moveAside(e *entry, reason string, caption bool) error {
	to := uniqueName(e.rel, f.destTaken(true))
	abs := filepath.Join(f.s.dir, SetAsideDir, filepath.FromSlash(to))
	mode := captionStays
	if caption {
		mode = captionMode(e, f.s.items)
	}
	if f.apply {
		f.settle(e)
		if err := move(e.abs, abs, mode); err != nil {
			return err
		}
	}
	from := e.rel
	f.dropItem(e)
	e.rel, e.abs = to, abs
	f.s.aside = append(f.s.aside, e)
	sortEntries(f.s.aside)
	f.setFrom(to, from)
	f.change(Change{File: from, Op: "set_aside", To: asideFile(to), Reason: reason})
	if !caption || len(f.s.set.Derived) == 0 || f.shaOf(e) == "" {
		return nil
	}
	for _, c := range slices.Clone(f.s.items) {
		if f.s.set.Derived[f.shaOf(c)] == e.sha {
			if err := f.moveAside(c, "a crop of "+from, true); err != nil {
				return err
			}
		}
	}
	return nil
}

// setFrom records the dataset path the set-aside photo rel came from (""
// once it is out of .set_aside/).
func (f *fixer) setFrom(rel, from string) {
	if from == rel {
		from = ""
	}
	if f.s.from[rel] == from {
		return
	}
	if f.s.from == nil {
		f.s.from = map[string]string{}
	}
	if from == "" {
		delete(f.s.from, rel)
	} else {
		f.s.from[rel] = from
	}
	f.fromChanged = true
}

func (f *fixer) setAside(files []string) error {
	if len(files) == 0 {
		return badRequest("set_aside needs files")
	}
	for _, file := range files {
		_, rel, ok := photoPath(f.s.dir, file)
		if !ok {
			return badRequest("%q is not a photo in this dataset", file)
		}
		if strings.HasPrefix(rel, SetAsideDir+"/") {
			continue // already left out
		}
		e := f.item(rel)
		if e == nil {
			if f.asideEntry(rel) != nil {
				continue
			}
			return badRequest("no photo named %s in this dataset", rel)
		}
		if err := f.moveAside(e, "", true); err != nil {
			return err
		}
	}
	return nil
}

func (f *fixer) restore(files []string, all bool) error {
	var picked []*entry
	if all {
		picked = slices.Clone(f.s.aside)
	} else {
		if len(files) == 0 {
			return badRequest("restore needs files or all: true")
		}
		for _, file := range files {
			_, rel, ok := photoPath(f.s.dir, file)
			if !ok {
				return badRequest("%q is not a photo in this dataset", file)
			}
			rel = strings.TrimPrefix(rel, SetAsideDir+"/")
			e := f.asideEntry(rel)
			if e == nil {
				if f.item(rel) != nil {
					continue // already in the dataset
				}
				return badRequest("no photo named %s in %s/", rel, SetAsideDir)
			}
			picked = append(picked, e)
		}
	}
	for _, e := range picked {
		f.fill(e)
		// back to the path it came from, else its .set_aside/ name, else
		// that with _2, _3 (a clash by name or by stem)
		taken := f.destTaken(false)
		to := f.s.originOf(e)
		if taken(to) {
			to = uniqueName(e.rel, taken)
		}
		abs := filepath.Join(f.s.dir, filepath.FromSlash(to))
		if f.apply {
			f.settle(e)
			if err := move(e.abs, abs, captionMode(e, f.s.aside)); err != nil {
				return err
			}
		}
		from := e.rel
		f.setFrom(from, "")
		f.dropAside(e)
		e.rel, e.abs = to, abs
		f.s.items = append(f.s.items, e)
		sortEntries(f.s.items)
		f.change(Change{File: asideFile(from), Op: "restore", To: to})
	}
	return nil
}

// flagNote is the reason a flag gives in a change.
func flagNote(it Item, fl Flag) string {
	if fl == FlagDuplicate && it.SameAs != "" {
		return "duplicate of " + it.SameAs
	}
	return flagReason(fl)
}

func (f *fixer) setAsideFlagged(flags []Flag) error {
	if len(flags) == 0 {
		flags = DefaultSetAsideFlags
	}
	for _, fl := range flags {
		if _, ok := Catalog[fl]; !ok {
			return badRequest("unknown flag %q", fl)
		}
	}
	r := f.review()
	for _, it := range r.Items {
		if it.Pending {
			continue
		}
		for _, fl := range flags {
			if !slices.Contains(it.Flags, fl) {
				continue
			}
			if e := f.item(it.File); e != nil {
				if err := f.moveAside(e, flagNote(it, fl), true); err != nil {
					return err
				}
			}
			break
		}
	}
	return nil
}

func (f *fixer) keepBest(n int) error {
	if n < 0 {
		return badRequest("keep_best n must be positive")
	}
	r := f.review()
	pk := PickBest(f.in, r, n, f.opts)
	pending := map[string]bool{}
	for _, it := range r.Items {
		pending[it.File] = it.Pending
	}
	for _, lo := range pk.LeftOut {
		if pending[lo.File] {
			continue
		}
		if e := f.item(lo.File); e != nil {
			if err := f.moveAside(e, lo.Reason, true); err != nil {
				return err
			}
		}
	}
	return nil
}

// targets are the photos an op works on: the named ones, or every photo
// whose review item passes pick.
func (f *fixer) targets(op string, files []string, pick func(Item) bool) ([]Item, error) {
	r := f.review()
	byFile := map[string]Item{}
	for _, it := range r.Items {
		byFile[it.File] = it
	}
	var out []Item
	if len(files) == 0 {
		for _, it := range r.Items {
			if !it.Pending && pick(it) {
				out = append(out, it)
			}
		}
		return out, nil
	}
	for _, file := range files {
		_, rel, ok := photoPath(f.s.dir, file)
		it, found := byFile[rel]
		if !ok || !found {
			return nil, badRequest("%s: no photo named %s in this dataset", op, file)
		}
		if !it.Pending {
			out = append(out, it)
		}
	}
	return out, nil
}

// rewrite replaces a photo with a new file written by write (from the
// original's path to the new path). A new extension takes the same stem
// when it is free (else _2, …, with the caption copied along) and moves the
// original into .set_aside/. The new file is measured right away when
// applied; in a preview it borrows the old metrics with patch applied.
func (f *fixer) rewrite(e *entry, ext, op, reason string, write func(src, dst string) error, patch func(*Metrics)) error {
	from := e.rel
	if isJPEG(path.Ext(from)) && isJPEG(ext) {
		ext = path.Ext(from) // .jpeg stays .jpeg
	}
	to := stemOf(from) + ext
	sameFile := strings.EqualFold(path.Ext(from), ext)
	if sameFile {
		to = from
	} else {
		taken := f.takenIn(false)
		to = uniqueName(to, func(rel string) bool {
			if rel == from {
				return false
			}
			// a new stem must not land on another photo's caption
			return taken(rel) || (!strings.EqualFold(stemOf(rel), stemOf(from)) &&
				exists(filepath.Join(f.s.dir, filepath.FromSlash(stemOf(rel))+".txt")))
		})
	}
	abs := filepath.Join(f.s.dir, filepath.FromSlash(to))
	oldSha := e.sha
	var m *Metrics
	sha := ""
	if f.apply {
		f.settle(e)
		if sameFile {
			tmp := filepath.Join(filepath.Dir(e.abs), "."+filepath.Base(e.abs)+".orig")
			if err := os.Rename(e.abs, tmp); err != nil {
				return err
			}
			if err := write(tmp, abs); err != nil {
				os.Rename(tmp, e.abs)
				return err
			}
			os.Remove(tmp)
			PurgeDerived(f.s.dir, from)
		} else {
			if err := write(e.abs, abs); err != nil {
				return err
			}
			if stemOf(to) != stemOf(from) {
				if c := captionFor(e.abs); exists(c) {
					copyFile(c, captionFor(abs))
				}
			}
			if err := f.moveAside(e, "the original of "+to, false); err != nil {
				os.Remove(abs)
				return err
			}
			f.changes = f.changes[:len(f.changes)-1] // the convert change says it
		}
		got, gotSha, err := measureNow(abs, to)
		if err != nil {
			return err
		}
		m, sha = &got, gotSha
		f.carry(oldSha, sha)
	} else {
		if e.m != nil && e.m.Decoded {
			c := *e.m
			c.File, c.Format, c.SHA256 = to, strings.TrimPrefix(strings.ToLower(ext), "."), ""
			if c.Format == "jpg" {
				c.Format = "jpeg"
			}
			c.Orientation = 1
			if patch != nil {
				patch(&c)
			}
			m = &c
		}
		if !sameFile {
			if err := f.moveAside(e, "", false); err != nil {
				return err
			}
			f.changes = f.changes[:len(f.changes)-1]
		}
	}
	if sameFile {
		e.m, e.sha = m, sha
	} else {
		st, _ := os.Stat(abs)
		n := &entry{rel: to, abs: abs, info: st, sha: sha, m: m}
		if st == nil {
			n.info = e.info
		}
		f.s.items = append(f.s.items, n)
		sortEntries(f.s.items)
	}
	c := Change{File: from, Op: op, Reason: reason}
	if to != from {
		c.To = to
	}
	f.change(c)
	return nil
}

func isJPEG(ext string) bool {
	ext = strings.ToLower(ext)
	return ext == ".jpg" || ext == ".jpeg"
}

// carry moves the photo state of old content onto new content: its
// decision, the reference, and its place as a crop's parent.
func (f *fixer) carry(oldSha, newSha string) {
	set := f.s.set
	if oldSha == "" || newSha == "" || oldSha == newSha {
		return
	}
	if d, ok := set.Decisions[oldSha]; ok {
		set.Decisions[newSha] = d
		delete(set.Decisions, oldSha)
		f.setChanged = true
	}
	if set.Reference == oldSha {
		set.Reference = newSha
		f.setChanged = true
	}
	for crop, parent := range set.Derived {
		if parent == oldSha {
			set.Derived[crop] = newSha
			f.setChanged = true
		}
	}
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	data, err := io.ReadAll(in)
	if err != nil {
		return err
	}
	return writeFileAtomic(dst, data)
}

func (f *fixer) bake(files []string) error {
	items, err := f.targets("bake_orientation", files, func(it Item) bool { return slices.Contains(it.Flags, FlagRotated) })
	if err != nil {
		return err
	}
	for _, it := range items {
		e := f.item(it.File)
		if e == nil || e.m == nil || e.m.Orientation <= 1 {
			continue
		}
		ext := strings.ToLower(path.Ext(it.File))
		switch ext {
		case ".jpg", ".jpeg", ".png":
		case ".webp":
			ext = ".jpg"
			if e.m.Alpha > 0 {
				ext = ".png"
			}
		default:
			continue
		}
		write := func(src, dst string) error {
			defer analysis.holdBig(src)()
			img, ex, err := decodeUpright(src)
			if err != nil {
				return err
			}
			return writeImage(dst, img, ex)
		}
		if err := f.rewrite(e, ext, "bake_orientation", "turned upright", write, nil); err != nil {
			return err
		}
	}
	return nil
}

func (f *fixer) convert(files []string) error {
	items, err := f.targets("convert", files, func(it Item) bool {
		_, _, ok := convertTarget(it.File, it.Flags)
		return ok
	})
	if err != nil {
		return err
	}
	for _, it := range items {
		flags := slices.Concat(it.Flags, it.Waived)
		ext, reason, ok := convertTarget(it.File, flags)
		e := f.item(it.File)
		if !ok || e == nil { // nothing this server can do for it

			continue
		}
		patch := func(m *Metrics) {
			m.Mode, m.Alpha, m.PNGColorType, m.PNGBitDepth, m.Animated = "rgb", 0, 0, 0, false
			if m.Format == "jpeg" {
				m.JPEGQ = jpegQuality
			}
			if area := float64(m.Width) * float64(m.Height); area > hugeTarget {
				k := math.Sqrt(hugeTarget / area)
				m.Width, m.Height = int(float64(m.Width)*k), int(float64(m.Height)*k)
			}
		}
		if err := f.rewrite(e, ext, "convert", reason, convertFile, patch); err != nil {
			return err
		}
	}
	return nil
}

func (f *fixer) renameSameStem() error {
	r := f.review()
	groups := map[string][]string{}
	var stems []string
	for _, it := range r.Items {
		if it.Pending || !slices.Contains(it.Flags, FlagSameStem) {
			continue
		}
		st := stemOf(it.File)
		if groups[st] == nil {
			stems = append(stems, st)
		}
		groups[st] = append(groups[st], it.File)
	}
	slices.SortFunc(stems, collation.Compare)
	for _, st := range stems {
		files := groups[st]
		// the photo the trainer reads keeps the name (and the caption)
		keep := files[0]
		for _, file := range files {
			if TrainerExts[strings.ToLower(path.Ext(file))] {
				keep = file
				break
			}
		}
		for _, file := range files {
			if file == keep {
				continue
			}
			e := f.item(file)
			if e == nil {
				continue
			}
			to := f.freeStem(file)
			abs := filepath.Join(f.s.dir, filepath.FromSlash(to))
			if f.apply {
				f.settle(e)
				if err := move(e.abs, abs, captionStays); err != nil {
					return err
				}
				if c := captionFor(e.abs); exists(c) {
					if err := copyFile(c, captionFor(abs)); err != nil {
						return err
					}
				}
			}
			e.rel, e.abs = to, abs
			sortEntries(f.s.items)
			f.change(Change{File: file, Op: "rename_same_stem", To: to, Reason: "shared a caption with " + keep})
		}
	}
	return nil
}

// freeStem is rel renamed to <stem>_2<ext> (_3, …): a stem no other photo
// or caption in the dataset uses.
func (f *fixer) freeStem(rel string) string {
	ext := path.Ext(rel)
	stem := stemOf(rel)
	for n := 2; ; n++ {
		c := fmt.Sprintf("%s_%d", stem, n)
		used := exists(filepath.Join(f.s.dir, filepath.FromSlash(c)+".txt"))
		for _, e := range f.s.items {
			if strings.EqualFold(stemOf(e.rel), c) {
				used = true
				break
			}
		}
		if !used {
			return c + ext
		}
	}
}

// cropPlan is one face crop to write.
type cropPlan struct {
	parent *entry
	fh     float64
	rect   image.Rectangle
}

func (f *fixer) faceCrops(maxCrops int) error {
	r := f.review()
	if r.Faces.Checked == 0 {
		return conflict("Check faces first: face crops need the face check's results.")
	}
	cropped := map[string]bool{}
	for _, it := range r.Items {
		if it.Parent != "" {
			cropped[it.Parent] = true
		}
	}
	var plans []cropPlan
	for _, it := range r.Items {
		if !it.Usable || it.Parent != "" || cropped[it.File] || it.Face == nil || it.Face.Box == nil ||
			(it.Face.Shot != "half" && it.Face.Shot != "full") {
			continue
		}
		e := f.item(it.File)
		if e == nil || f.shaOf(e) == "" {
			continue
		}
		box := *it.Face.Box
		if box[3] >= cropMaxFH {
			continue
		}
		rect, ok := cropRect(box, it.Width, it.Height)
		if !ok {
			continue
		}
		plans = append(plans, cropPlan{parent: e, fh: box[3], rect: rect})
	}
	slices.SortStableFunc(plans, func(a, b cropPlan) int {
		if a.fh != b.fh {
			if a.fh < b.fh {
				return -1
			}
			return 1
		}
		return collation.Compare(a.parent.rel, b.parent.rel)
	})
	limit := maxCrops
	if limit <= 0 {
		limit = max(3, int(math.Floor(0.30*float64(r.Usable))))
	}
	if len(plans) > limit {
		plans = plans[:limit]
	}
	for _, p := range plans {
		to := uniqueName(stemOf(p.parent.rel)+faceCropSuffix+".jpg", f.takenIn(false))
		abs := filepath.Join(f.s.dir, filepath.FromSlash(to))
		if f.apply {
			release := analysis.holdBig(p.parent.abs)
			img, ex, err := decodeUpright(p.parent.abs)
			if err != nil {
				release()
				return err
			}
			rect := p.rect
			if b := img.Bounds(); !rect.In(b) {
				// the stored size disagrees with the decode: crop what is there
				rect = rect.Intersect(b)
			}
			sub, ok := img.(interface {
				SubImage(image.Rectangle) image.Image
			})
			if !ok {
				release()
				return fmt.Errorf("cannot crop %s", p.parent.rel)
			}
			err = writeImage(abs, sub.SubImage(rect), ex)
			release()
			if err != nil {
				return err
			}
			m, sha, err := measureNow(abs, to)
			if err != nil {
				return err
			}
			f.s.set.Derived[sha] = p.parent.sha
			f.setChanged = true
			st, _ := os.Stat(abs)
			f.s.items = append(f.s.items, &entry{rel: to, abs: abs, info: st, sha: sha, m: &m})
			sortEntries(f.s.items)
		}
		f.change(Change{File: p.parent.rel, Op: "face_crops", To: to, Reason: "a close-up of the face"})
	}
	return nil
}

// cropRect is the section 9 crop for a subject face box (fractions of the
// upright w x h photo): a square 2.6 face sides across, centred a quarter
// face height below the face, shifted (never shrunk) to stay in the frame.
// ok is false when the square would be under 512 px.
func cropRect(box [4]float64, w, h int) (image.Rectangle, bool) {
	if w <= 0 || h <= 0 {
		return image.Rectangle{}, false
	}
	bw, bh := box[2]*float64(w), box[3]*float64(h)
	side := math.Min(cropScale*math.Max(bw, bh), float64(min(w, h)))
	if side < cropMinSide {
		return image.Rectangle{}, false
	}
	cx := box[0]*float64(w) + bw/2
	cy := box[1]*float64(h) + bh/2 + cropDrop*bh
	s := int(math.Round(side))
	x0 := min(max(0, int(math.Round(cx-side/2))), w-s)
	y0 := min(max(0, int(math.Round(cy-side/2))), h-s)
	return image.Rect(x0, y0, x0+s, y0+s), true
}

func (f *fixer) removeFaceCrops() error {
	derived := f.s.set.Derived
	for _, list := range [][]*entry{slices.Clone(f.s.items), slices.Clone(f.s.aside)} {
		for _, e := range list {
			if derived[f.shaOf(e)] == "" {
				continue
			}
			aside := f.asideEntry(e.rel) == e
			rel := e.rel
			if aside {
				rel = asideFile(e.rel)
			}
			if f.apply {
				f.settle(e)
				if err := os.Remove(e.abs); err != nil && !isNotExist(err) {
					return err
				}
				os.Remove(captionFor(e.abs))
				PurgeDerived(f.s.dir, rel)
				delete(derived, e.sha)
				f.setChanged = true
			}
			if aside {
				f.setFrom(e.rel, "")
				f.dropAside(e)
			} else {
				f.dropItem(e)
			}
			f.change(Change{File: rel, Op: "remove_face_crops", Reason: "a face crop"})
		}
	}
	return nil
}
