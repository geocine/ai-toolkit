package photocheck

import (
	"encoding/base64"
	"image"
	"math"
	"slices"

	"github.com/geocine/aitoolkit/internal/imgmeta"
)

// Sizes of the section 3 copies.
const (
	sharpEdge   = 1024 // S: area-equivalent edge
	workLong    = 512  // W: long side
	colourLong  = 128  // C: long side
	detailCrop  = 1024 // detail: centre crop side
	noiseMaxPix = 24_000_000
)

// MeasureImage measures a decoded photo (section 3). sn is the sniffed
// header of the same file (orientation, EXIF, PNG and JPEG facts), rel the
// dataset-relative name and bytes the file size. SHA256 is left empty: the
// caller hashes the file.
func MeasureImage(img image.Image, sn imgmeta.Sniff, rel string, bytes int64) Metrics {
	b := img.Bounds()
	if sn.Width == 0 || sn.Height == 0 {
		sn.Width, sn.Height = b.Dx(), b.Dy()
	}
	m := headerMetrics(sn, rel, bytes)
	if b.Dx() <= 0 || b.Dy() <= 0 {
		m.Error = "image has no pixels"
		return m
	}
	p := toPlanes(img, sn.Format)
	o := m.Orientation
	m.Width, m.Height = p.w, p.h
	if o >= 5 {
		m.Width, m.Height = p.h, p.w
	}
	m.Decoded = true
	m.Mode = p.mode

	// S, W and C in stored orientation, then upright.
	sw, sh := fitArea(p.w, p.h, sharpEdge)
	sRaw := boxDown(p.y, p.ys, p.w, p.h, sw, sh)
	ww, wh := fitLong(sw, sh, workLong)
	wRaw := BoxDown(sRaw, sw, sh, ww, wh)
	cw, ch := fitLong(p.w, p.h, colourLong)
	var cr, cg, cb []float32
	if p.cb != nil {
		cy := BoxDown(sRaw, sw, sh, cw, ch)
		ccb := boxDown(p.cb, p.cs, p.cw, p.ch, cw, ch)
		ccr := boxDown(p.cr, p.cs, p.cw, p.ch, cw, ch)
		cr, cg, cb = ycbcrToRGB(cy, ccb, ccr)
	} else {
		cr = boxDown(p.r, p.w, p.w, p.h, cw, ch)
		cg = boxDown(p.g, p.w, p.w, p.h, cw, ch)
		cb = boxDown(p.b, p.w, p.w, p.h, cw, ch)
	}
	if sRaw == nil || wRaw == nil || cr == nil || cg == nil || cb == nil {
		m.Decoded, m.Mode = false, ""
		m.Error = "cannot read the decoded pixels"
		return m
	}
	s, sw, sh := orient(sRaw, sw, sh, o)
	wk, ww, wh := orient(wRaw, ww, wh, o)
	cr, _, _ = orient(cr, cw, ch, o)
	cg, _, _ = orient(cg, cw, ch, o)
	cb, cw, ch = orient(cb, cw, ch, o)

	m.Sharp = r4(sharpness(s, sw, sh))
	h, flip := DHash(wk, ww, wh)
	m.DHash, m.DHashFlip = HashHex(h), HashHex(flip)
	m.Thumb32 = base64.StdEncoding.EncodeToString(thumb32(wk, ww, wh))
	ex := exposure(wk, ww, wh)
	m.LumaMean, m.LumaStd = r4(ex.mean), r4(ex.std)
	m.ClipLo, m.ClipHi, m.ClipHiCenter = r4(ex.clipLo), r4(ex.clipHi), r4(ex.clipHiCenter)
	m.Flat = r4(flatShare(wk))
	m.Bars = r4(bars(wk, ww, wh))
	m.UIBand = uiBand(wk, ww, wh)
	m.Colorful = r4(colorfulness(cr, cg, cb))
	m.ChromaSpread = r4(chromaSpread(cr, cg, cb))
	m.Layout = base64.StdEncoding.EncodeToString(layout(cr, cg, cb, cw, ch))
	if p.a != nil {
		awr, ahr := ww, wh
		if o >= 5 {
			awr, ahr = wh, ww
		}
		a := boxDown(p.a, p.w, p.w, p.h, awr, ahr)
		n := 0
		for _, v := range a {
			if v < 250 {
				n++
			}
		}
		m.Alpha = r4(float64(n) / float64(len(a)))
	}
	if p.w*p.h <= noiseMaxPix {
		if v, ok := immerkaer(p.y, p.ys, p.w, p.h); ok {
			v = r4(v)
			m.Noise = &v
		}
	}
	if v, ok := detail(p.y, p.ys, p.w, p.h); ok {
		v = r4(v)
		m.Detail = &v
	}
	return m
}

// r4 rounds to 4 decimals, which keeps cached JSON short.
func r4(v float64) float64 { return math.Round(v*1e4) / 1e4 }

// sharpness is the nearest-rank p90 of per-tile Laplacian variance on S,
// after a 3x3 Gaussian; both filters cover the valid region only.
func sharpness(s []float32, w, h int) float64 {
	if w < 5 || h < 5 {
		return 0
	}
	// Gaussian [1 2 1]/4 along rows, then columns: g is (w-2) x (h-2).
	gw, gh := w-2, h-2
	t := make([]float32, gw*h)
	for y := 0; y < h; y++ {
		r := s[y*w : (y+1)*w]
		for x := 0; x < gw; x++ {
			t[y*gw+x] = (r[x] + 2*r[x+1] + r[x+2]) / 4
		}
	}
	g := make([]float32, gw*gh)
	for y := 0; y < gh; y++ {
		for x := 0; x < gw; x++ {
			g[y*gw+x] = (t[y*gw+x] + 2*t[(y+1)*gw+x] + t[(y+2)*gw+x]) / 4
		}
	}
	lw, lh := gw-2, gh-2
	nx, ny := min(max(w/32, 1), 8), min(max(h/32, 1), 8)
	var vals []float64
	for ty := 0; ty < ny; ty++ {
		y0, y1 := ty*lh/ny, (ty+1)*lh/ny
		for tx := 0; tx < nx; tx++ {
			x0, x1 := tx*lw/nx, (tx+1)*lw/nx
			var sum, sq float64
			n := 0
			for y := y0; y < y1; y++ {
				c := (y + 1) * gw
				for x := x0; x < x1; x++ {
					i := c + x + 1
					l := float64(g[i-1] + g[i+1] + g[i-gw] + g[i+gw] - 4*g[i])
					sum += l
					sq += l * l
					n++
				}
			}
			if n > 0 {
				mean := sum / float64(n)
				vals = append(vals, math.Max(0, sq/float64(n)-mean*mean))
			}
		}
	}
	if len(vals) == 0 {
		return 0
	}
	slices.Sort(vals)
	return vals[int(math.Ceil(0.9*float64(len(vals))))-1]
}

type exposureStats struct {
	mean, std, clipLo, clipHi, clipHiCenter float64
}

// exposure reads W: mean, standard deviation, the shares at or below 5 and
// at or above 250, and the share at or above 250 inside the central
// 60% x 60%.
func exposure(wk []float32, w, h int) exposureStats {
	var e exposureStats
	n := float64(len(wk))
	if n == 0 {
		return e
	}
	var sum, sq float64
	var lo, hi int
	for _, v := range wk {
		f := float64(v)
		sum += f
		sq += f * f
		if v <= 5 {
			lo++
		}
		if v >= 250 {
			hi++
		}
	}
	e.mean = sum / n
	e.std = math.Sqrt(math.Max(0, sq/n-e.mean*e.mean))
	e.clipLo, e.clipHi = float64(lo)/n, float64(hi)/n
	x0, x1 := centre(w)
	y0, y1 := centre(h)
	hc, nc := 0, 0
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			if wk[y*w+x] >= 250 {
				hc++
			}
			nc++
		}
	}
	if nc > 0 {
		e.clipHiCenter = float64(hc) / float64(nc)
	}
	return e
}

// centre is the middle 60% of a side, at least one pixel.
func centre(n int) (int, int) {
	a, b := int(math.Round(0.2*float64(n))), int(math.Round(0.8*float64(n)))
	if b <= a {
		a, b = n/2, n/2+1
	}
	return a, b
}

// flatShare is the share of W in its 6 largest of 32 luma bins (Y>>3).
func flatShare(wk []float32) float64 {
	if len(wk) == 0 {
		return 0
	}
	var bins [32]int
	for _, v := range wk {
		bins[int(clamp8(v))>>3]++
	}
	s := bins[:]
	slices.Sort(s)
	top := 0
	for _, c := range s[len(s)-6:] {
		top += c
	}
	return float64(top) / float64(len(wk))
}

// Bar detection (section 3 "bars").
const (
	barStd     = 3.0
	barDark    = 24.0
	barLight   = 232.0
	barMaxFrac = 0.25
	barPairMin = 0.02
	barPairTol = 0.25
	barOneMin  = 0.04
	barStep    = 40.0
)

// barScan is how far uniform dark or light lines reach in from one edge.
type barScan struct {
	frac  float64
	n     int
	light bool
	mean  float64 // mean of the bar lines
	inner float64 // mean of the next 2 lines inward; NaN when there are none
}

// scanBar walks lines (rows or columns) in from an edge. line(i) returns
// the mean and std of the i-th line from that edge; total is the side.
func scanBar(total int, line func(i int) (mean, std float64)) barScan {
	var s barScan
	limit := int(math.Floor(barMaxFrac * float64(total)))
	var sum float64
	for i := 0; i < limit; i++ {
		m, sd := line(i)
		if sd >= barStd || (m >= barDark && m <= barLight) {
			break
		}
		light := m > barLight
		if i == 0 {
			s.light = light
		} else if light != s.light {
			break
		}
		sum += m
		s.n++
	}
	s.frac = float64(s.n) / float64(total)
	s.inner = math.NaN()
	if s.n > 0 {
		s.mean = sum / float64(s.n)
		if s.n+2 <= total {
			a, _ := line(s.n)
			b, _ := line(s.n + 1)
			s.inner = (a + b) / 2
		}
	}
	return s
}

// bars is the area share of W covered by counted edge bars: a matching pair
// on opposite edges, or a single edge with a sharp step inward.
func bars(wk []float32, w, h int) float64 {
	if w < 4 || h < 4 {
		return 0
	}
	row := func(y int) (float64, float64) { return lineStats(wk[y*w:(y+1)*w], 1, w) }
	col := func(x int) (float64, float64) { return lineStats(wk[x:], w, h) }
	top := scanBar(h, row)
	bottom := scanBar(h, func(i int) (float64, float64) { return row(h - 1 - i) })
	left := scanBar(w, col)
	right := scanBar(w, func(i int) (float64, float64) { return col(w - 1 - i) })
	t, b := countBars(top, bottom)
	l, r := countBars(left, right)
	return 1 - (1-t-b)*(1-l-r)
}

// countBars returns the counted fractions of two opposite edges.
func countBars(a, b barScan) (float64, float64) {
	if a.frac >= barPairMin && b.frac >= barPairMin && a.light == b.light &&
		math.Abs(a.frac-b.frac) <= barPairTol*math.Max(a.frac, b.frac) {
		return a.frac, b.frac
	}
	single := func(s barScan) float64 {
		if s.frac >= barOneMin && !math.IsNaN(s.inner) && math.Abs(s.inner-s.mean) >= barStep {
			return s.frac
		}
		return 0
	}
	return single(a), single(b)
}

// lineStats is the mean and std of n samples starting at v[0], step apart.
func lineStats(v []float32, step, n int) (float64, float64) {
	var sum, sq float64
	for i := 0; i < n; i++ {
		f := float64(v[i*step])
		sum += f
		sq += f * f
	}
	m := sum / float64(n)
	return m, math.Sqrt(math.Max(0, sq/float64(n)-m*m))
}

// uiBand reports a phone status bar along the top of W: a band of
// max(8, round(0.04 H)) rows that is mostly one level, with at least 1% of
// pixels far from it (|Y - median| > 48) in at least 3 separate runs of
// columns (runs split by 2 or more clean columns).
func uiBand(wk []float32, w, h int) bool {
	rows := min(h, max(8, int(math.Round(0.04*float64(h)))))
	band := wk[:rows*w]
	if len(band) == 0 {
		return false
	}
	sorted := slices.Clone(band)
	slices.Sort(sorted)
	med := float64(sorted[len(sorted)/2])
	if len(sorted)%2 == 0 {
		med = (float64(sorted[len(sorted)/2-1]) + med) / 2
	}
	dirty := make([]bool, w)
	d := 0
	for y := 0; y < rows; y++ {
		for x := 0; x < w; x++ {
			if math.Abs(float64(band[y*w+x])-med) > 48 {
				d++
				dirty[x] = true
			}
		}
	}
	share := float64(d) / float64(len(band))
	if share < 0.01 || 1-share < 0.85 {
		return false
	}
	runs, gap := 0, 2
	for _, dx := range dirty {
		if dx {
			if gap >= 2 {
				runs++
			}
			gap = 0
		} else {
			gap++
		}
	}
	return runs >= 3
}

// ycbcrToRGB converts JPEG (JFIF, full-range BT.601) planes to RGB.
func ycbcrToRGB(y, cb, cr []float32) (r, g, b []float32) {
	n := len(y)
	r, g, b = make([]float32, n), make([]float32, n), make([]float32, n)
	for i := 0; i < n && i < len(cb) && i < len(cr); i++ {
		yy, u, v := float64(y[i]), float64(cb[i])-128, float64(cr[i])-128
		r[i] = clampF(yy + 1.402*v)
		g[i] = clampF(yy - 0.344136*u - 0.714136*v)
		b[i] = clampF(yy + 1.772*u)
	}
	return r, g, b
}

func clampF(v float64) float32 { return float32(math.Min(255, math.Max(0, v))) }

// colorfulness is Hasler-Süsstrunk M3 on C.
func colorfulness(r, g, b []float32) float64 {
	n := float64(len(r))
	if n == 0 {
		return 0
	}
	var srg, syb, qrg, qyb float64
	for i := range r {
		rg := float64(r[i]) - float64(g[i])
		yb := (float64(r[i])+float64(g[i]))/2 - float64(b[i])
		srg += rg
		syb += yb
		qrg += rg * rg
		qyb += yb * yb
	}
	mrg, myb := srg/n, syb/n
	vrg, vyb := math.Max(0, qrg/n-mrg*mrg), math.Max(0, qyb/n-myb*myb)
	return math.Sqrt(vrg+vyb) + 0.3*math.Sqrt(mrg*mrg+myb*myb)
}

// chromaSpread is sqrt(var(Cb) + var(Cr)) on C, BT.601 full range.
func chromaSpread(r, g, b []float32) float64 {
	n := float64(len(r))
	if n == 0 {
		return 0
	}
	var scb, scr, qcb, qcr float64
	for i := range r {
		R, G, B := float64(r[i]), float64(g[i]), float64(b[i])
		cb := 128 - 0.168736*R - 0.331264*G + 0.5*B
		cr := 128 + 0.5*R - 0.418688*G - 0.081312*B
		scb += cb
		scr += cr
		qcb += cb * cb
		qcr += cr * cr
	}
	mcb, mcr := scb/n, scr/n
	return math.Sqrt(math.Max(0, qcb/n-mcb*mcb) + math.Max(0, qcr/n-mcr*mcr))
}

// srgbLinear maps 8-bit sRGB to linear light; fractional inputs are
// interpolated.
var srgbLinear = func() (t [256]float64) {
	for i := range t {
		c := float64(i) / 255
		if c <= 0.04045 {
			t[i] = c / 12.92
		} else {
			t[i] = math.Pow((c+0.055)/1.055, 2.4)
		}
	}
	return t
}()

func linear(v float32) float64 {
	f := math.Min(255, math.Max(0, float64(v)))
	i := int(f)
	if i >= 255 {
		return srgbLinear[255]
	}
	t := f - float64(i)
	return srgbLinear[i]*(1-t) + srgbLinear[i+1]*t
}

// lab converts sRGB (0-255) to CIELAB under D65.
func lab(r, g, b float32) (float64, float64, float64) {
	R, G, B := linear(r), linear(g), linear(b)
	x := (0.4124564*R + 0.3575761*G + 0.1804375*B) / 0.95047
	y := 0.2126729*R + 0.7151522*G + 0.0721750*B
	z := (0.0193339*R + 0.1191920*G + 0.9503041*B) / 1.08883
	fx, fy, fz := labF(x), labF(y), labF(z)
	return 116*fy - 16, 500 * (fx - fy), 200 * (fy - fz)
}

func labF(t float64) float64 {
	const d = 6.0 / 29
	if t > d*d*d {
		return math.Cbrt(t)
	}
	return t/(3*d*d) + 4.0/29
}

// layout is the 4x4 grid of mean CIELAB on C, 3 bytes per cell:
// round(L*2.55), a+128, b+128, clamped to 0..255.
func layout(r, g, b []float32, w, h int) []byte {
	n := len(r)
	L, A, B := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := 0; i < n; i++ {
		l, a, bb := lab(r[i], g[i], b[i])
		L[i], A[i], B[i] = float32(l), float32(a), float32(bb)
	}
	gl, ga, gb := BoxDown(L, w, h, 4, 4), BoxDown(A, w, h, 4, 4), BoxDown(B, w, h, 4, 4)
	out := make([]byte, 0, 3*layoutCells)
	for i := 0; i < layoutCells && i < len(gl); i++ {
		out = append(out, clamp8(gl[i]*2.55), clamp8(ga[i]+128), clamp8(gb[i]+128))
	}
	return out
}

// immerkaer is the Immerkaer noise sigma on full-resolution luma:
// sqrt(pi/2) * sum|I*N| / (6 (W-2)(H-2)), N = [1 -2 1; -2 4 -2; 1 -2 1].
func immerkaer(y []uint8, stride, w, h int) (float64, bool) {
	if w < 3 || h < 3 {
		return 0, false
	}
	var sum int64
	for r := 1; r < h-1; r++ {
		a, b, c := y[(r-1)*stride:], y[r*stride:], y[(r+1)*stride:]
		for x := 1; x < w-1; x++ {
			v := int(a[x-1]) - 2*int(a[x]) + int(a[x+1]) -
				2*int(b[x-1]) + 4*int(b[x]) - 2*int(b[x+1]) +
				int(c[x-1]) - 2*int(c[x]) + int(c[x+1])
			if v < 0 {
				v = -v
			}
			sum += int64(v)
		}
	}
	return math.Sqrt(math.Pi/2) * float64(sum) / (6 * float64(w-2) * float64(h-2)), true
}

// detail is the PSNR (dB) of a centre crop of full-resolution luma (up to
// 1024 x 1024) against itself halved with the box filter and doubled back
// bilinearly. A high value means the photo holds nothing finer than half
// its size: it was probably enlarged. Info only.
func detail(y []uint8, stride, w, h int) (float64, bool) {
	cw, ch := min(w, detailCrop)&^1, min(h, detailCrop)&^1
	if cw < 4 || ch < 4 {
		return 0, false
	}
	x0, y0 := (w-cw)/2, (h-ch)/2
	crop := make([]float32, cw*ch)
	for r := 0; r < ch; r++ {
		row := y[(y0+r)*stride+x0:]
		for x := 0; x < cw; x++ {
			crop[r*cw+x] = float32(row[x])
		}
	}
	half := BoxDown(crop, cw, ch, cw/2, ch/2)
	back := upBilinear2(half, cw/2, ch/2)
	var se float64
	for i, v := range crop {
		d := float64(v) - float64(back[i])
		se += d * d
	}
	mse := se / float64(len(crop))
	if mse < 1e-10 {
		return 100, true
	}
	return math.Min(100, 10*math.Log10(255*255/mse)), true
}
