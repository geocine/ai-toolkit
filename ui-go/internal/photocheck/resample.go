package photocheck

import "math"

// The working copies are area averages: every destination pixel is the mean
// of the source area it covers, with fractional weights for the source
// pixels on its edges. The filter is separable (rows first into a float32
// buffer, then columns) and works in float32 planes. studio/src/lib/photos.ts
// implements the same arithmetic; testdata/dhash_golden.json pins both:
//
//	lo = (i*n)/m, hi = ((i+1)*n)/m          // source span of output i (n -> m)
//	w_j = (min(hi, j+1) - max(lo, j)) * m/n // for j = floor(lo) .. ceil(hi)-1
//	out_i = sum_j w_j * src_j               // summed in float64, stored as float32
//
// The same formula samples up when m > n (each output then reads one or two
// source pixels), but the callers never ask for that on a photo's copies.

// boxAxis is the weight table of one axis: output i reads src[start[i]+k]
// with weight wts[off[i]+k] for k < off[i+1]-off[i].
type boxAxis struct {
	start []int
	off   []int
	wts   []float64
}

func newBoxAxis(n, m int) boxAxis {
	a := boxAxis{start: make([]int, m), off: make([]int, m+1)}
	inv := float64(m) / float64(n)
	for i := 0; i < m; i++ {
		lo := float64(i*n) / float64(m)
		hi := float64((i+1)*n) / float64(m)
		j0 := int(math.Floor(lo))
		j1 := int(math.Ceil(hi))
		if j1 > n {
			j1 = n
		}
		if j1 <= j0 {
			j1 = j0 + 1
		}
		a.start[i] = j0
		for j := j0; j < j1; j++ {
			ov := math.Min(hi, float64(j+1)) - math.Max(lo, float64(j))
			a.wts = append(a.wts, ov*inv)
		}
		a.off[i+1] = len(a.wts)
	}
	return a
}

// BoxDown area-averages a w x h float32 plane (row-major) to dw x dh.
func BoxDown(src []float32, w, h, dw, dh int) []float32 {
	return boxDown(src, w, w, h, dw, dh)
}

// boxDown is BoxDown over a plane of any pixel type with a row stride.
func boxDown[T uint8 | float32](src []T, stride, w, h, dw, dh int) []float32 {
	if w <= 0 || h <= 0 || dw <= 0 || dh <= 0 || len(src) < (h-1)*stride+w {
		return nil
	}
	ax, ay := newBoxAxis(w, dw), newBoxAxis(h, dh)
	tmp := make([]float32, h*dw)
	for y := 0; y < h; y++ {
		row := src[y*stride : y*stride+w]
		out := tmp[y*dw : (y+1)*dw]
		for i := 0; i < dw; i++ {
			j := ax.start[i]
			var s float64
			for _, wt := range ax.wts[ax.off[i]:ax.off[i+1]] {
				s += float64(wt * float64(row[j]))
				j++
			}
			out[i] = float32(s)
		}
	}
	dst := make([]float32, dw*dh)
	acc := make([]float64, dw)
	for i := 0; i < dh; i++ {
		clear(acc)
		y := ay.start[i]
		for _, wt := range ay.wts[ay.off[i]:ay.off[i+1]] {
			row := tmp[y*dw : (y+1)*dw]
			for x, v := range row {
				acc[x] += float64(wt * float64(v))
			}
			y++
		}
		out := dst[i*dw : (i+1)*dw]
		for x, v := range acc {
			out[x] = float32(v)
		}
	}
	return dst
}

// fitArea is the size of the copy whose area-equivalent edge
// sqrt(w*h) is at most edge; it never grows the image.
func fitArea(w, h int, edge float64) (int, int) {
	f := math.Min(1, edge/math.Sqrt(float64(w)*float64(h)))
	return max(1, int(math.Round(float64(w)*f))), max(1, int(math.Round(float64(h)*f)))
}

// fitLong is the size of the copy whose long side is at most long; it never
// grows the image.
func fitLong(w, h, long int) (int, int) {
	l := max(w, h)
	if l <= long {
		return w, h
	}
	f := float64(long) / float64(l)
	return max(1, int(math.Round(float64(w)*f))), max(1, int(math.Round(float64(h)*f)))
}

// orient applies EXIF orientation o (1..8) to a w x h plane and returns the
// upright plane and its size. Orientations 5-8 swap the sides.
func orient(src []float32, w, h, o int) ([]float32, int, int) {
	if o < 2 || o > 8 || len(src) < w*h {
		return src, w, h
	}
	ow, oh := w, h
	if o >= 5 {
		ow, oh = h, w
	}
	dst := make([]float32, ow*oh)
	for y := 0; y < oh; y++ {
		for x := 0; x < ow; x++ {
			var sx, sy int
			switch o {
			case 2: // mirror
				sx, sy = w-1-x, y
			case 3: // rotate 180
				sx, sy = w-1-x, h-1-y
			case 4: // flip
				sx, sy = x, h-1-y
			case 5: // transpose
				sx, sy = y, x
			case 6: // rotate 90 clockwise
				sx, sy = y, h-1-x
			case 7: // transverse
				sx, sy = w-1-y, h-1-x
			case 8: // rotate 90 counter-clockwise
				sx, sy = w-1-y, x
			}
			dst[y*ow+x] = src[sy*w+sx]
		}
	}
	return dst, ow, oh
}

// upBilinear2 doubles a w x h plane with bilinear sampling on pixel centres
// (source x = out x/2 - 0.25), clamped at the edges.
func upBilinear2(src []float32, w, h int) []float32 {
	ow, oh := 2*w, 2*h
	xs0, xs1, fx := axis2(w)
	ys0, ys1, fy := axis2(h)
	dst := make([]float32, ow*oh)
	for y := 0; y < oh; y++ {
		r0, r1, ty := src[ys0[y]*w:], src[ys1[y]*w:], fy[y]
		for x := 0; x < ow; x++ {
			a := float64(r0[xs0[x]])*(1-fx[x]) + float64(r0[xs1[x]])*fx[x]
			b := float64(r1[xs0[x]])*(1-fx[x]) + float64(r1[xs1[x]])*fx[x]
			dst[y*ow+x] = float32(a*(1-ty) + b*ty)
		}
	}
	return dst
}

func axis2(n int) (i0, i1 []int, f []float64) {
	i0, i1, f = make([]int, 2*n), make([]int, 2*n), make([]float64, 2*n)
	for o := range 2 * n {
		s := float64(o)/2 - 0.25
		a := int(math.Floor(s))
		t := s - float64(a)
		b := a + 1
		a, b = min(max(a, 0), n-1), min(max(b, 0), n-1)
		i0[o], i1[o], f[o] = a, b, t
	}
	return
}
