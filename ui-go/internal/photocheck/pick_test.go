package photocheck

import (
	"encoding/base64"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"
)

// layoutOf is a 4x4 grid of one colour per seed: photos with the same
// seed look alike.
func layoutOf(seed int) string {
	r := rand.New(rand.NewPCG(uint64(seed), 99))
	b := make([]byte, 48)
	for i := range b {
		b[i] = byte(r.IntN(256))
	}
	return base64.StdEncoding.EncodeToString(b)
}

func keptOf(p Pick) map[string]bool {
	m := map[string]bool{}
	for _, f := range p.Kept {
		m[f] = true
	}
	return m
}

// TestPick covers "pick the best N" (section 8).
func TestPick(t *testing.T) {
	t.Run("clusters, bursts, hard-outs and keep", func(t *testing.T) {
		var in []Input
		for c := 0; c < 3; c++ {
			base := rand.New(rand.NewPCG(uint64(c), 3)).Uint64()
			for k := 0; k < 5; k++ {
				m := good(fmt.Sprintf("c%d_%d.jpg", c, k), 100+10*c+k)
				m.Layout = layoutOf(c)
				m.Sharp = 300 + float64(k)
				if c == 0 { // one burst: near hashes and EXIF times within a second
					m.DHash = HashHex(base ^ uint64(1)<<(8*k))
					m.Exif = &ExifInfo{Time: fmt.Sprintf("2026-05-01T14:03:22.%d00", k), Make: "Apple"}
				}
				in = append(in, Input{Metrics: m})
			}
		}
		for s := 0; s < 5; s++ {
			m := good(fmt.Sprintf("s%d.jpg", s), 200+s)
			m.Layout = layoutOf(10 + s)
			in = append(in, Input{Metrics: m})
		}
		tiny := good("tiny.jpg", 300)
		tiny.Width, tiny.Height, tiny.Layout = 400, 500, layoutOf(20)
		shot := good("screen.png", 301)
		shot.Width, shot.Height, shot.Exif, shot.UIBand, shot.Layout = 1170, 2532, nil, true, layoutOf(21)
		soft := good("soft_keep.jpg", 302)
		soft.Sharp, soft.Layout = 1, layoutOf(22)
		in = append(in, Input{Metrics: tiny}, Input{Metrics: shot}, Input{Metrics: soft, Decision: DecisionKeep})

		r := Judge(in, SetState{}, Options{Pick: 8})
		if r.Pick == nil {
			t.Fatal("no pick")
		}
		p := *r.Pick
		if p.N != 8 || len(p.Kept) != 8 {
			t.Fatalf("kept %d of %d: %v (%s)", len(p.Kept), p.N, p.Kept, p.Note)
		}
		kept := keptOf(p)
		for c := 0; c < 3; c++ {
			n := 0
			for f := range kept {
				if strings.HasPrefix(f, fmt.Sprintf("c%d_", c)) {
					n++
				}
			}
			if n == 0 || (c == 0 && n > 1) {
				t.Errorf("cluster %d: %d picked (%v)", c, n, p.Kept)
			}
		}
		if kept["tiny.jpg"] || kept["screen.png"] || !kept["soft_keep.jpg"] {
			t.Errorf("hard-outs or keep: %v", p.Kept)
		}
		reasons := map[string]string{}
		for _, l := range p.LeftOut {
			reasons[l.File] = l.Reason
		}
		if reasons["tiny.jpg"] != "too small" || reasons["screen.png"] != "a screenshot" || len(p.LeftOut) != len(in)-8 {
			t.Errorf("left out %v", p.LeftOut)
		}
		burst := 0
		for f, why := range reasons {
			if strings.HasPrefix(f, "c0_") && strings.HasPrefix(why, "same burst as c0_") {
				burst++
			}
		}
		if burst != 4 {
			t.Errorf("burst reasons: %v", reasons)
		}

		// Deterministic, whatever the input order, and the same through
		// PickBest.
		shuffled := slices.Clone(in)
		rand.New(rand.NewPCG(5, 5)).Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
		r2 := Judge(shuffled, SetState{}, Options{Pick: 8})
		if !slices.Equal(r2.Pick.Kept, p.Kept) {
			t.Errorf("order changed the pick: %v vs %v", r2.Pick.Kept, p.Kept)
		}
		if again := PickBest(in, r, 8, Options{}); !slices.Equal(again.Kept, p.Kept) {
			t.Errorf("PickBest %v vs Judge %v", again.Kept, p.Kept)
		}
	})

	t.Run("stop rule on a near-identical set", func(t *testing.T) {
		in := goodSet(30)
		for i := range in {
			in[i].Metrics.DHash, in[i].Metrics.DHashFlip = "", ""
			in[i].Metrics.Layout = layoutOf(1)
		}
		p := PickBest(in, Judge(in, SetState{}, Options{}), 20, Options{})
		if len(p.Kept) != 10 || !strings.HasPrefix(p.Note, "10 more would not add variety.") {
			t.Errorf("kept %d, note %q", len(p.Kept), p.Note)
		}
		reasons := map[string]bool{}
		for _, l := range p.LeftOut {
			reasons[l.Reason] = true
		}
		if len(reasons) != 1 || !strings.HasPrefix(p.LeftOut[0].Reason, "very like ") {
			t.Errorf("reasons %v", reasons)
		}
	})

	t.Run("a pool smaller than N keeps all", func(t *testing.T) {
		in := goodSet(6)
		in[5].Metrics.Width, in[5].Metrics.Height = 300, 400
		r := Judge(in, SetState{}, Options{})
		p := PickBest(in, r, 10, Options{})
		if len(p.Kept) != 5 || p.Note != "Only 5 photos pass, so all are kept. Add 5 more: add 10 more photos of just them." {
			t.Errorf("kept %v, note %q", p.Kept, p.Note)
		}
		if !slices.Equal(p.Kept, []string{"p00.jpg", "p01.jpg", "p02.jpg", "p03.jpg", "p04.jpg"}) ||
			len(p.LeftOut) != 1 || p.LeftOut[0] != (LeftOut{File: "p05.jpg", Reason: "too small"}) {
			t.Errorf("kept %v left out %v", p.Kept, p.LeftOut)
		}
		if d := PickBest(in, r, 0, Options{}); d.N != 25 {
			t.Errorf("default N %d", d.N)
		}
	})

	t.Run("a full-length seed", func(t *testing.T) {
		var in []Input
		for i := 0; i < 23; i++ {
			fh := 0.5
			if i >= 20 {
				fh = 0.09
			}
			x := faceInput(i, frontal, fh)
			x.Metrics.Layout = layoutOf(i % 4)
			if i >= 20 {
				x.Metrics.Sharp = 100 // weaker, and smaller faces: never first on quality
			}
			in = append(in, x)
		}
		r := Judge(in, SetState{}, Options{})
		p := PickBest(in, r, 10, Options{})
		full := 0
		for _, f := range p.Kept {
			if f >= "f20.jpg" {
				full++
			}
		}
		if len(p.Kept) != 10 || full < 1 {
			t.Errorf("kept %v (%d full)", p.Kept, full)
		}
	})

	t.Run("a half quota with no full-length photo", func(t *testing.T) {
		// N = 20: mins full 2, half 4, close 8. The picks are 19 close and
		// 1 half; the pool has 3 more half photos and no full-length one, so
		// full has no candidate and must not stop the half swaps.
		k := &picker{n: 20, r: &Review{Kind: KindPerson}}
		for i := 0; i < 30; i++ {
			k.pool = append(k.pool, &pickItem{file: fmt.Sprintf("c%02d.jpg", i), it: &Item{}, shot: "close", q: 0.5})
		}
		for i := 0; i < 4; i++ {
			k.pool = append(k.pool, &pickItem{file: fmt.Sprintf("h%d.jpg", i), it: &Item{}, shot: "half", q: 0.4})
		}
		n := len(k.pool)
		k.sim = make([][]float32, n)
		for i := range k.sim {
			k.sim[i] = make([]float32, n)
		}
		k.in, k.cover = make([]bool, n), make([]float64, n)
		for i := 0; i < 19; i++ {
			k.add(i)
		}
		k.add(30)
		k.quota(k.minimums())
		if h, c := k.bucketCount("half"), k.bucketCount("close"); h != 2 || c != 18 || len(k.picked) != 20 {
			t.Errorf("half %d close %d of %d", h, c, len(k.picked))
		}
	})

	t.Run("duplicates, crops and stored photos", func(t *testing.T) {
		in := goodSet(12)
		cp := in[0].Metrics
		cp.File, cp.Bytes = "p00_copy.jpg", 10
		crop := good("p01__face.jpg", 77)
		stored := good("stored.jpg", 78)
		in = append(in, Input{Metrics: cp}, Input{Metrics: crop, Parent: "p01.jpg"}, Input{Metrics: stored, Existing: true})
		r := Judge(in, SetState{}, Options{})
		p := PickBest(in, r, 12, Options{})
		kept := keptOf(p)
		if kept["p00_copy.jpg"] || kept["stored.jpg"] || !kept["p01__face.jpg"] || len(p.Kept) != 13 {
			t.Errorf("kept %v", p.Kept)
		}
		for _, l := range p.LeftOut {
			if l.File == "p00_copy.jpg" && l.Reason != "duplicate of p00.jpg" {
				t.Errorf("copy reason %q", l.Reason)
			}
			if l.File == "stored.jpg" {
				t.Error("a stored photo is listed")
			}
		}
		p = PickBest(in, r, 11, Options{})
		if keptOf(p)["p01.jpg"] != keptOf(p)["p01__face.jpg"] {
			t.Errorf("a crop without its parent: %v", p.Kept)
		}
	})

	t.Run("session cap", func(t *testing.T) {
		in := dated(goodSet(20), "2026-05-01T10:00:00", 3*time.Minute)
		for i := 10; i < 20; i++ {
			in[i].Metrics.Exif.Time = fmt.Sprintf("2026-06-%02dT10:00:00", i)
		}
		p := PickBest(in, Judge(in, SetState{}, Options{}), 8, Options{})
		first := 0
		for _, f := range p.Kept {
			if f < "p10.jpg" {
				first++
			}
		}
		if first > 2 || len(p.Kept) != 8 {
			t.Errorf("one session gave %d of %v", first, p.Kept)
		}
		found := false
		for _, l := range p.LeftOut {
			found = found || strings.HasPrefix(l.Reason, "same hour as 2 photos already picked")
		}
		if !found {
			t.Errorf("no session reason: %v", p.LeftOut)
		}
	})
}
