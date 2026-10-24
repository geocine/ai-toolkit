package photocheck

import (
	"encoding/base64"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/collation"
)

// DefaultPick is the default N per kind.
func DefaultPick(k Kind) int {
	switch k {
	case KindStyle:
		return 30
	case KindObject:
		return 20
	}
	return 25
}

// pickItem is one photo as the pick sees it.
type pickItem struct {
	file    string
	in      Input
	it      *Item
	warn    bool // an active warn flag: half quality
	forced  bool // "Keep anyway" on a photo the trainer can read
	dupOf   string
	burst   string // burst group id
	hashed  bool
	h, hf   uint64
	layout  []byte // decoded Metrics.Layout, nil when missing
	keys    []string
	caption bool
	shot    string
	yaw     string
	face    float64 // subject face sharpness, when known
	hasFace bool
	q       float64
	gain    float64 // the gain it was picked with
}

func (p *pickItem) active(f Flag) bool { return slices.Contains(p.it.Flags, f) }

// Similarity weights (section 8 step 3); [cal].
const (
	simHash, simLayout, simSession, simShot, simYaw, simTags = 0.30, 0.25, 0.15, 0.10, 0.10, 0.10
)

// PickBest picks the best n photos of a judged set for quality and
// variety (section 8). r is Judge's review of the same inputs. It is
// deterministic: ties go to the first file. n <= 0 picks the kind's
// default. Stored photos (Existing) are never picked.
func PickBest(in []Input, r *Review, n int, opts Options) Pick {
	if n <= 0 {
		n = DefaultPick(r.Kind)
	}
	byFile := make(map[string]*Item, len(r.Items))
	for i := range r.Items {
		byFile[r.Items[i].File] = &r.Items[i]
	}
	dupOf, burstOf := map[string]string{}, map[string]string{}
	for _, g := range r.Groups {
		for _, f := range g.Files {
			switch {
			case g.Kind == groupDuplicate && f != g.Keep:
				dupOf[f] = g.Keep
			case g.Kind == groupBurst:
				burstOf[f] = g.ID
			}
		}
	}
	var items []*pickItem
	for _, x := range in {
		it := byFile[x.Metrics.File]
		if x.Existing || it == nil {
			continue
		}
		p := &pickItem{file: it.File, in: x, it: it, warn: it.Level == LevelWarn, dupOf: dupOf[it.File], burst: burstOf[it.File]}
		p.forced = x.Decision == DecisionKeep && !slices.ContainsFunc(it.Flags, func(f Flag) bool { return trainerUnreadable[f] })
		if h, err := ParseHash(x.Metrics.DHash); err == nil {
			p.h, p.hf, p.hashed = h, h, true
			if hf, err := ParseHash(x.Metrics.DHashFlip); err == nil {
				p.hf = hf
			}
		}
		if l, err := base64.StdEncoding.DecodeString(x.Metrics.Layout); err == nil && len(l) == 3*layoutCells {
			p.layout = l
		}
		if x.Caption != nil {
			p.caption = true
			p.keys = parseCaption(*x.Caption, opts.Trigger).keys()
		}
		if it.Face != nil {
			p.shot, p.yaw = it.Face.Shot, it.Face.Yaw
			if it.Face.Box != nil && x.Face != nil {
				for _, f := range x.Face.Faces {
					if f.Box == *it.Face.Box {
						p.face, p.hasFace = f.Sharp, true
						break
					}
				}
			}
		}
		items = append(items, p)
	}
	slices.SortStableFunc(items, func(a, b *pickItem) int { return collation.Compare(a.file, b.file) })

	pk := &picker{r: r, n: n, opts: opts, items: items, out: Pick{N: n, Kept: []string{}, LeftOut: []LeftOut{}}}
	pk.run()
	return pk.out
}

// picker holds one PickBest call.
type picker struct {
	r     *Review
	n     int
	opts  Options
	items []*pickItem
	pool  []*pickItem
	sim   [][]float32
	out   Pick

	picked []int // indices into pool, in pick order
	in     []bool
	cover  []float64
	reason map[string]string
}

func (k *picker) run() {
	k.reason = map[string]string{}
	var derived []*pickItem
	for _, p := range k.items {
		switch {
		case p.it.Pending:
			k.reason[p.file] = "not checked yet"
		case p.in.Parent != "":
			derived = append(derived, p)
		case p.forced:
			k.pool = append(k.pool, p)
		case p.dupOf != "":
			k.reason[p.file] = "duplicate of " + p.dupOf
		default:
			if f, ok := k.hardOut(p); ok {
				k.reason[p.file] = flagReason(f)
				continue
			}
			k.pool = append(k.pool, p)
		}
	}
	k.quality()
	k.similarity()
	k.in = make([]bool, len(k.pool))
	k.cover = make([]float64, len(k.pool))

	// A pool no larger than N: keep all, one per burst.
	distinct := map[string]bool{}
	count := 0
	for _, p := range k.pool {
		if p.burst == "" {
			count++
		} else if !distinct[p.burst] {
			distinct[p.burst] = true
			count++
		}
	}
	if count <= k.n {
		order := k.byQuality()
		slices.SortStableFunc(order, func(a, b int) int { return int(b2f(k.pool[b].forced) - b2f(k.pool[a].forced)) })
		for _, i := range order {
			if k.allowed(i, false) {
				k.add(i)
			}
		}
		slices.Sort(k.picked) // file order reads better than quality order here
		if count < k.n {
			note := fmt.Sprintf("Only %d photos pass, so all are kept. Add %d more", count, k.n-count)
			if len(k.r.Hints) > 0 {
				note += ": " + lowerFirst(k.r.Hints[0])
			} else {
				note += "."
			}
			k.out.Note = note
		}
	} else {
		k.greedy()
	}

	keptFiles := map[string]bool{}
	for _, i := range k.picked {
		k.out.Kept = append(k.out.Kept, k.pool[i].file)
		keptFiles[k.pool[i].file] = true
	}
	for _, p := range derived {
		if _, out := k.hardOut(p); keptFiles[p.in.Parent] && (!out || p.forced) {
			k.out.Kept = append(k.out.Kept, p.file)
			keptFiles[p.file] = true
		} else {
			k.reason[p.file] = "kept only with " + p.in.Parent
		}
	}
	for i, p := range k.pool {
		if !k.in[i] && k.reason[p.file] == "" {
			k.reason[p.file] = k.leftReason(i)
		}
	}
	for _, p := range k.items {
		if !keptFiles[p.file] {
			k.out.LeftOut = append(k.out.LeftOut, LeftOut{File: p.file, Reason: k.reason[p.file]})
		}
	}
}

// hardOut is the first flag that keeps a photo out of the pool.
func (k *picker) hardOut(p *pickItem) (Flag, bool) {
	for _, f := range pickHardOut {
		if p.active(f) {
			return f, true
		}
	}
	return "", false
}

func flagReason(f Flag) string {
	switch f {
	case FlagUnreadable, FlagUnsupported, FlagHuge:
		return "not read by the trainer"
	case FlagScreenshot:
		return "a screenshot"
	case FlagOthers:
		return "other people in it"
	}
	return Catalog[f].Label
}

// quality is section 8 step 2.
func (k *picker) quality() {
	var faceS, frameS, bpp []float64
	bppOf := func(p *pickItem) float64 {
		px := float64(p.in.Metrics.Width) * float64(p.in.Metrics.Height)
		if px <= 0 {
			return 0
		}
		return float64(p.in.Metrics.Bytes) / px
	}
	for _, p := range k.pool {
		if p.hasFace {
			faceS = append(faceS, p.face)
		}
		if p.in.Metrics.Decoded {
			frameS = append(frameS, p.in.Metrics.Sharp)
		}
		bpp = append(bpp, bppOf(p))
	}
	R := float64(k.r.Resolution)
	if R <= 0 {
		R = DefaultR
	}
	for _, p := range k.pool {
		var sum, wsum float64
		term := func(w, v float64) { sum += w * v; wsum += w }
		switch {
		case p.hasFace:
			term(0.35, rank(faceS, p.face))
		case p.in.Metrics.Decoded:
			term(0.35, rank(frameS, p.in.Metrics.Sharp))
		}
		m := p.in.Metrics
		if m.Width > 0 && m.Height > 0 {
			term(0.25, math.Min(1, math.Round(math.Sqrt(float64(m.Width)*float64(m.Height)))/R))
		}
		if m.Decoded {
			e := 1.0
			if p.active(FlagDark) {
				e -= 0.5
			}
			if p.active(FlagBright) {
				e -= 0.5
			}
			if p.active(FlagFlat) {
				e -= 0.25
			}
			term(0.15, math.Max(0, e))
		}
		if m.Bytes > 0 && m.Width > 0 {
			term(0.10, rank(bpp, bppOf(p)))
		}
		if k.r.Kind == KindPerson && p.it.Face != nil && p.it.Face.Box != nil {
			term(0.15, math.Min(1, float64(p.it.Face.Px)/192))
		}
		if wsum > 0 {
			p.q = sum / wsum
		}
		if p.warn {
			p.q *= 0.5
		}
	}
}

// rank is the percentile rank of v in vs (0..1); 1 for a single value.
func rank(vs []float64, v float64) float64 {
	if len(vs) <= 1 {
		return 1
	}
	less, equal := 0, 0
	for _, x := range vs {
		switch {
		case x < v:
			less++
		case x == v:
			equal++
		}
	}
	return (float64(less) + 0.5*float64(max(equal-1, 0))) / float64(len(vs)-1)
}

// similarity is section 8 step 3, over the pool.
func (k *picker) similarity() {
	n := len(k.pool)
	k.sim = make([][]float32, n)
	flat := make([]float32, n*n)
	for i := range k.sim {
		k.sim[i] = flat[i*n : (i+1)*n]
		k.sim[i][i] = 1
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			s := float32(pairSim(k.pool[i], k.pool[j]))
			k.sim[i][j], k.sim[j][i] = s, s
		}
	}
}

func pairSim(a, b *pickItem) float64 {
	var sum, wsum float64
	term := func(w, v float64) { sum += w * v; wsum += w }
	if a.hashed && b.hashed {
		d := min(Hamming(a.h, b.h), Hamming(a.h, b.hf))
		term(simHash, math.Max(0, 1-float64(d)/32))
	}
	if a.layout != nil && b.layout != nil {
		term(simLayout, layoutSim(a.layout, b.layout))
	}
	if a.it.Session > 0 && b.it.Session > 0 {
		term(simSession, b2f(a.it.Session == b.it.Session))
	}
	if a.shot != "" && b.shot != "" {
		term(simShot, b2f(a.shot == b.shot))
	}
	if a.yaw != "" && b.yaw != "" {
		term(simYaw, b2f(a.yaw == b.yaw))
	}
	if a.caption && b.caption && len(a.keys)+len(b.keys) > 0 {
		inter := 0
		for _, x := range a.keys {
			if slices.Contains(b.keys, x) {
				inter++
			}
		}
		term(simTags, float64(inter)/float64(len(a.keys)+len(b.keys)-inter))
	}
	if wsum == 0 {
		return 0
	}
	return sum / wsum
}

// layoutSim is LayoutSim on decoded grids (the pick compares every pair,
// so each grid is decoded once).
func layoutSim(pa, pb []byte) float64 {
	var sum float64
	for i := 0; i < layoutCells; i++ {
		dl := (float64(pa[3*i]) - float64(pb[3*i])) / 2.55
		da := float64(pa[3*i+1]) - float64(pb[3*i+1])
		db := float64(pa[3*i+2]) - float64(pb[3*i+2])
		sum += math.Sqrt(dl*dl + da*da + db*db)
	}
	return math.Min(1, math.Max(0, 1-sum/layoutCells/50))
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// sessionCap is the most photos one EXIF session may give.
func (k *picker) sessionCap() int { return max(2, (k.n+3)/4) }

// allowed checks the pick constraints: one per burst group and, when
// capped, at most sessionCap per session.
func (k *picker) allowed(i int, capSessions bool) bool {
	if k.in[i] {
		return false
	}
	p := k.pool[i]
	for _, j := range k.picked {
		q := k.pool[j]
		if p.burst != "" && q.burst == p.burst {
			return false
		}
	}
	if capSessions && p.it.Session > 0 && k.sessionCount(p.it.Session) >= k.sessionCap() {
		return false
	}
	return true
}

func (k *picker) sessionCount(s int) int {
	n := 0
	for _, j := range k.picked {
		if k.pool[j].it.Session == s {
			n++
		}
	}
	return n
}

// coverGain is the coverage part of i's gain.
func (k *picker) coverGain(i int) float64 {
	sum := 0.0
	row := k.sim[i]
	for j, c := range k.cover {
		if d := float64(row[j]) - c; d > 0 {
			sum += d
		}
	}
	return sum / float64(len(k.pool))
}

func (k *picker) add(i int) float64 {
	g := k.coverGain(i)
	k.pool[i].gain = g + 0.3*k.pool[i].q
	k.in[i] = true
	k.picked = append(k.picked, i)
	for j := range k.pool {
		k.cover[j] = math.Max(k.cover[j], float64(k.sim[i][j]))
	}
	return g
}

// byQuality is the pool in descending quality (ties: file order).
func (k *picker) byQuality() []int {
	order := make([]int, len(k.pool))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int {
		switch qa, qb := k.pool[a].q, k.pool[b].q; {
		case qa > qb:
			return -1
		case qa < qb:
			return 1
		}
		return 0
	})
	return order
}

// bucketOf is the shot bucket the seeds and quotas count.
func bucketOf(p *pickItem) string {
	switch p.shot {
	case "close", "bust":
		return "close"
	case "half", "full":
		return p.shot
	}
	return ""
}

// minimums are the per-bucket floors for N photos.
func (k *picker) minimums() map[string]int {
	n := float64(k.n)
	return map[string]int{"full": int(math.Ceil(0.10 * n)), "half": int(math.Ceil(0.20 * n)),
		"close": int(math.Ceil(0.40 * n)), "profile": 2}
}

func (k *picker) bucketCount(b string) int {
	c := 0
	for _, i := range k.picked {
		if b == "profile" && k.pool[i].yaw == "profile" || b != "profile" && bucketOf(k.pool[i]) == b {
			c++
		}
	}
	return c
}

func (k *picker) inBucket(i int, b string) bool {
	if b == "profile" {
		return k.pool[i].yaw == "profile"
	}
	return bucketOf(k.pool[i]) == b
}

// greedy is section 8 steps 4-7.
func (k *picker) greedy() {
	first := -1.0
	note := func(g float64) {
		if first < 0 {
			first = g
		}
	}
	for i, p := range k.pool {
		if p.forced && len(k.picked) < k.n && k.allowed(i, false) {
			note(k.add(i))
		}
	}
	faces := k.r.Kind == KindPerson && k.r.Faces.Checked > 0
	mins := k.minimums()
	if faces {
		buckets := []string{"full", "half"}
		if k.n >= 15 {
			buckets = append(buckets, "profile")
		}
		for _, b := range buckets {
			if len(k.picked) >= k.n || k.bucketCount(b) >= mins[b] {
				continue
			}
			for _, i := range k.byQuality() {
				if k.inBucket(i, b) && k.allowed(i, true) {
					note(k.add(i))
					break
				}
			}
		}
	}
	// Lazy greedy: coverage gains only shrink as picks are added, so a
	// stale gain is an upper bound and only the top of the list needs a
	// fresh one. Ties go to the first file, as in a full scan.
	ub := make([]float64, len(k.pool))
	for i := range ub {
		ub[i] = math.Inf(1)
	}
	stopped := false
	for len(k.picked) < k.n {
		best, bestGain, bestCover := -1, math.Inf(-1), 0.0
		for {
			top := -1
			for i := range k.pool {
				if !k.in[i] && (top < 0 || ub[i] > ub[top]) && k.allowed(i, true) {
					top = i
				}
			}
			if top < 0 {
				break
			}
			c := k.coverGain(top)
			ub[top] = c + 0.3*k.pool[top].q
			if ub[top] > bestGain || (ub[top] == bestGain && top < best) {
				best, bestGain, bestCover = top, ub[top], c
			}
			// best is final once no other candidate can beat it
			beaten := false
			for i := range k.pool {
				if i != best && !k.in[i] && (ub[i] > bestGain || (ub[i] == bestGain && i < best)) && k.allowed(i, true) {
					beaten = true
					break
				}
			}
			if !beaten {
				break
			}
		}
		if best < 0 {
			break
		}
		if len(k.picked) >= min(10, k.n) && first > 0 && bestCover < 0.01*first {
			stopped = true
			break
		}
		note(k.add(best))
	}
	if faces {
		k.quota(mins)
	}
	if stopped {
		s := fmt.Sprintf("%d more would not add variety.", k.n-len(k.picked))
		if h := k.missingHint(mins); h != "" {
			s += " " + h
		}
		k.out.Note = s
	}
}

// quota swaps in a shot bucket's best photo while that bucket is below
// half its minimum, for the lowest-gain pick of the most over-filled one.
// A short bucket with no candidate left is skipped, so a later one still
// gets its swap.
func (k *picker) quota(mins map[string]int) {
	for range k.n {
		if !k.swapShort(mins) {
			return
		}
	}
}

// swapShort makes one quota swap for the first short bucket that has an
// allowed candidate and an over-filled bucket to drop from. It reports
// whether it swapped.
func (k *picker) swapShort(mins map[string]int) bool {
	for _, short := range []string{"full", "half", "close"} {
		if float64(k.bucketCount(short)) >= float64(mins[short])/2 {
			continue
		}
		cand := -1
		for _, i := range k.byQuality() {
			if k.inBucket(i, short) && k.allowed(i, true) {
				cand = i
				break
			}
		}
		if cand < 0 {
			continue
		}
		over, excess := "", 0
		for _, b := range []string{"full", "half", "close"} {
			if e := k.bucketCount(b) - mins[b]; b != short && e > excess {
				over, excess = b, e
			}
		}
		if over == "" {
			continue
		}
		drop := -1
		for pos, i := range k.picked {
			if k.inBucket(i, over) && !k.pool[i].forced && (drop < 0 || k.pool[i].gain < k.pool[k.picked[drop]].gain) {
				drop = pos
			}
		}
		if drop < 0 {
			continue
		}
		k.in[k.picked[drop]] = false
		k.picked = slices.Delete(k.picked, drop, drop+1)
		k.recover()
		k.add(cand)
		return true
	}
	return false
}

// recover rebuilds the coverage after a pick is dropped.
func (k *picker) recover() {
	for j := range k.cover {
		k.cover[j] = 0
		for _, i := range k.picked {
			k.cover[j] = math.Max(k.cover[j], float64(k.sim[i][j]))
		}
	}
}

// missingHint is the hint for the first shot bucket below its minimum.
func (k *picker) missingHint(mins map[string]int) string {
	if k.r.Kind != KindPerson || k.r.Faces.Checked == 0 {
		return ""
	}
	switch {
	case k.bucketCount("full") < mins["full"] || k.bucketCount("half") < mins["half"]:
		return hintNoFull
	case k.bucketCount("close") < mins["close"]:
		return hintNoClose
	case k.n >= 15 && k.bucketCount("profile") < mins["profile"]:
		return hintProfiles
	}
	return ""
}

// leftReason says in plain words why a pool photo was not picked.
func (k *picker) leftReason(i int) string {
	p := k.pool[i]
	for _, j := range k.picked {
		if q := k.pool[j]; p.burst != "" && q.burst == p.burst {
			return "same burst as " + q.file
		}
	}
	if s := p.it.Session; s > 0 {
		if c := k.sessionCount(s); c >= k.sessionCap() {
			return fmt.Sprintf("same hour as %d photos already picked", c)
		}
	}
	if p.warn {
		for _, f := range p.it.Flags {
			if p.it.Notes[f] != "" && Catalog[f].Level == LevelWarn {
				return flagReason(f)
			}
		}
	}
	best, bestSim := -1, float32(0)
	for _, j := range k.picked {
		if s := k.sim[i][j]; s > bestSim {
			best, bestSim = j, s
		}
	}
	if best >= 0 && bestSim >= 0.5 {
		return "very like " + k.pool[best].file + ", which was picked"
	}
	return fmt.Sprintf("only %d were asked for", k.n)
}

func lowerFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToLower(s[:1]) + s[1:]
}
