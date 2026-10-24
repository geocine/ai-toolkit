package photocheck

import (
	"cmp"
	"math"
	"slices"
	"strconv"

	"github.com/geocine/aitoolkit/internal/collation"
)

// Group kinds.
const (
	groupDuplicate = "duplicate"
	groupBurst     = "burst"
	groupSimilar   = "similar"
)

// unionFind over photo indices.
type unionFind []int

func newUnionFind(n int) unionFind {
	u := make(unionFind, n)
	for i := range u {
		u[i] = i
	}
	return u
}

func (u unionFind) find(i int) int {
	for u[i] != i {
		u[i] = u[u[i]]
		i = u[i]
	}
	return i
}

func (u unionFind) union(a, b int) { u[u.find(a)] = u.find(b) }

// components lists the sets with two or more members, each in index order.
func (u unionFind) components() [][]int {
	by := map[int][]int{}
	for i := range u {
		r := u.find(i)
		by[r] = append(by[r], i)
	}
	var out [][]int
	for _, c := range by {
		if len(c) > 1 {
			out = append(out, c)
		}
	}
	return out
}

// pair is a scored pair of photos (indices into judge.all).
type pair struct {
	a, b int
	dist int
}

// groupPhotos finds duplicate, burst and similar groups (section 4.2) and
// flags their members. Stored photos (Existing) join, but a group needs at
// least one new photo, and a stored photo is always the keeper.
func (j *judge) groupPhotos() {
	th := j.th
	ps := j.all
	for _, p := range ps {
		if p.Pending || p.Metrics.DHash == "" {
			continue
		}
		h, err1 := ParseHash(p.Metrics.DHash)
		hf, err2 := ParseHash(p.Metrics.DHashFlip)
		if err1 != nil {
			continue
		}
		if err2 != nil {
			hf = h
		}
		p.h, p.hf, p.hashed = h, hf, true
		p.thumb = DecodeThumb32(p.Metrics.Thumb32)
	}
	related := func(a, b *photo) bool {
		if a.Pending || b.Pending || (a.Existing && b.Existing) {
			return false
		}
		// a face crop is never grouped with its parent
		return a.Parent != b.Metrics.File && b.Parent != a.Metrics.File
	}

	// Duplicates: the same bytes, or a close hash confirmed by NCC.
	dup := newUnionFind(len(ps))
	var dupEdges, near []pair
	for i := range ps {
		for k := i + 1; k < len(ps); k++ {
			a, b := ps[i], ps[k]
			if !related(a, b) {
				continue
			}
			if a.Metrics.SHA256 != "" && a.Metrics.SHA256 == b.Metrics.SHA256 {
				dup.union(i, k)
				dupEdges = append(dupEdges, pair{i, k, 0})
				continue
			}
			if !a.hashed || !b.hashed {
				continue
			}
			d := min(Hamming(a.h, b.h), Hamming(a.h, b.hf))
			if d <= th.DupHamming && confirmNCC(a.thumb, b.thumb, th.DupNCC) {
				dup.union(i, k)
				dupEdges = append(dupEdges, pair{i, k, d})
				continue
			}
			if d <= max(th.SimilarHamming, th.BurstHamming) {
				near = append(near, pair{i, k, d})
			}
		}
	}
	var groups []Group
	for _, c := range dup.components() {
		g, keep, ok := j.makeGroup(groupDuplicate, c, dupEdges, dup)
		if !ok {
			continue
		}
		groups = append(groups, g)
		for _, i := range c {
			p := ps[i]
			if p == keep {
				p.keeper = true
				continue
			}
			p.dupOf = keep.Metrics.File
			p.sameAs = keep.Metrics.File
			if !p.Existing {
				p.set(FlagDuplicate, LevelWarn, noteDuplicate(keep.Metrics.File, keep.Existing))
			}
		}
	}

	// Bursts and similar pairs, between photos that are not already copies
	// of one another and are not leaving as copies.
	burst := newUnionFind(len(ps))
	sim := newUnionFind(len(ps))
	var burstEdges, simEdges []pair
	for _, e := range near {
		a, b := ps[e.a], ps[e.b]
		if dup.find(e.a) == dup.find(e.b) || a.dupOf != "" || b.dupOf != "" {
			continue
		}
		if e.dist <= th.BurstHamming && a.dated && b.dated &&
			math.Abs(a.when.Sub(b.when).Seconds()) <= th.BurstSeconds {
			burst.union(e.a, e.b)
			burstEdges = append(burstEdges, e)
			continue
		}
		if e.dist <= th.SimilarHamming {
			simEdges = append(simEdges, e)
		}
	}
	for _, c := range burst.components() {
		g, keep, ok := j.makeGroup(groupBurst, c, burstEdges, burst)
		if !ok {
			continue
		}
		groups = append(groups, g)
		for _, i := range c {
			p := ps[i]
			p.burstID = g.Kind + ":" + g.Keep
			if p == keep {
				p.keeper = true
				continue
			}
			p.burstOf = keep.Metrics.File
			if p.sameAs == "" {
				p.sameAs = keep.Metrics.File
			}
			if !p.Existing {
				p.set(FlagBurst, LevelInfo, noteBurst(keep.Metrics.File))
			}
		}
	}
	for _, e := range simEdges {
		if burst.find(e.a) != burst.find(e.b) {
			sim.union(e.a, e.b)
		}
	}
	for _, c := range sim.components() {
		g, _, ok := j.makeGroup(groupSimilar, c, simEdges, sim)
		if !ok {
			continue
		}
		groups = append(groups, g)
		for _, i := range c {
			p := ps[i]
			near := j.closest(i, c, simEdges)
			if p.sameAs == "" {
				p.sameAs = near
			}
			if !p.Existing {
				p.set(FlagSimilar, LevelInfo, noteSimilar(near))
			}
		}
	}

	order := map[string]int{groupDuplicate: 0, groupBurst: 1, groupSimilar: 2}
	slices.SortFunc(groups, func(a, b Group) int {
		if c := cmp.Compare(order[a.Kind], order[b.Kind]); c != 0 {
			return c
		}
		return collation.Compare(a.Files[0], b.Files[0])
	})
	ids := map[string]string{} // kind:keep -> id
	for i := range groups {
		groups[i].ID = "g" + strconv.Itoa(i+1)
		ids[groups[i].Kind+":"+groups[i].Keep] = groups[i].ID
	}
	// Each photo names one group: its duplicate group, else its burst,
	// else its similar group.
	for _, g := range groups {
		for _, f := range g.Files {
			if p := j.byFile[f]; p != nil && p.group == "" {
				p.group = g.ID
			}
		}
	}
	for _, p := range ps {
		if p.burstID != "" {
			p.burstID = ids[p.burstID]
		}
	}
	j.groups = groups
}

// makeGroup builds the Group for one component and picks its keeper; ok is
// false when it holds no new photo.
func (j *judge) makeGroup(kind string, c []int, edges []pair, u unionFind) (Group, *photo, bool) {
	members := make([]*photo, len(c))
	own := false
	for i, k := range c {
		members[i] = j.all[k]
		own = own || !members[i].Existing
	}
	if !own {
		return Group{}, nil, false
	}
	root := u.find(c[0])
	dist := math.MaxInt
	for _, e := range edges {
		if u.find(e.a) == root && u.find(e.b) == root {
			dist = min(dist, e.dist)
		}
	}
	if dist == math.MaxInt {
		dist = 0
	}
	files := make([]string, len(members))
	for i, p := range members {
		files[i] = p.Metrics.File
	}
	slices.SortFunc(files, collation.Compare)
	keep := j.keeperOf(members)
	g := Group{Kind: kind, Files: files, Keep: keep.Metrics.File, Distance: dist}
	if kind == groupDuplicate {
		g.Confirmed = true // every edge is the same bytes or NCC-confirmed
	}
	return g, keep, true
}

// closest is the member of c nearest to photo i over the given edges
// (ties: file order).
func (j *judge) closest(i int, c []int, edges []pair) string {
	best, bestD := "", math.MaxInt
	for _, e := range edges {
		o := -1
		switch i {
		case e.a:
			o = e.b
		case e.b:
			o = e.a
		}
		if o < 0 || !slices.Contains(c, o) {
			continue
		}
		f := j.all[o].Metrics.File
		if e.dist < bestD || (e.dist == bestD && collation.Compare(f, best) < 0) {
			best, bestD = f, e.dist
		}
	}
	return best
}

// keeperOf picks the copy to keep (section 4.2): a stored photo when there
// is one; then in focus (sharp >= KeeperFocus x the best), the higher
// compression bucket floor(log2(bytes/pixel)), sharper, more pixels, more
// bytes, and the first file.
func (j *judge) keeperOf(ms []*photo) *photo {
	cands := ms
	if slices.ContainsFunc(ms, func(p *photo) bool { return p.Existing }) {
		cands = slices.DeleteFunc(slices.Clone(ms), func(p *photo) bool { return !p.Existing })
	}
	best := 0.0
	for _, p := range cands {
		best = max(best, p.Metrics.Sharp)
	}
	focus := func(p *photo) int {
		if p.Metrics.Sharp >= j.th.KeeperFocus*best {
			return 1
		}
		return 0
	}
	return slices.MinFunc(cands, func(a, b *photo) int {
		if c := cmp.Compare(focus(b), focus(a)); c != 0 {
			return c
		}
		if c := cmp.Compare(bucket(b.Metrics), bucket(a.Metrics)); c != 0 {
			return c
		}
		if c := cmp.Compare(b.Metrics.Sharp, a.Metrics.Sharp); c != 0 {
			return c
		}
		pa := int64(a.Metrics.Width) * int64(a.Metrics.Height)
		pb := int64(b.Metrics.Width) * int64(b.Metrics.Height)
		if c := cmp.Compare(pb, pa); c != 0 {
			return c
		}
		if c := cmp.Compare(b.Metrics.Bytes, a.Metrics.Bytes); c != 0 {
			return c
		}
		return collation.Compare(a.Metrics.File, b.Metrics.File)
	})
}

// bucket is floor(log2(bytes per pixel)); a photo with no size sorts last.
func bucket(m Metrics) int {
	px := float64(m.Width) * float64(m.Height)
	if px <= 0 || m.Bytes <= 0 {
		return math.MinInt32
	}
	return int(math.Floor(math.Log2(float64(m.Bytes) / px)))
}

// confirmNCC is the duplicate confirmation: the thumbnails correlate at
// least floor, directly or mirrored.
func confirmNCC(a, b []byte, floor float64) bool {
	if a == nil || b == nil {
		return false
	}
	return max(NCC32(a, b), NCC32(a, Mirror32(b))) >= floor
}
