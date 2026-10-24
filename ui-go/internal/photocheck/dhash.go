package photocheck

import (
	"encoding/base64"
	"fmt"
	"math"
	"math/bits"
	"strconv"
)

// DHash is the 64-bit difference hash of a working copy (w is ww x wh luma,
// row-major): the copy is area-averaged to a 9x8 grid P, and bit i = y*8+x
// (x < 8) is set when P(x,y) > P(x+1,y), stored as h |= 1 << (63-i). flip
// is the same over the column-mirrored grid P'(x,y) = P(8-x,y), so a
// mirrored copy of a photo has flip equal to the photo's h.
func DHash(w []float32, ww, wh int) (h, flip uint64) {
	p := BoxDown(w, ww, wh, 9, 8)
	if len(p) != 72 {
		return 0, 0
	}
	for y := 0; y < 8; y++ {
		r := p[y*9 : y*9+9]
		for x := 0; x < 8; x++ {
			bit := uint64(1) << (63 - (y*8 + x))
			if r[x] > r[x+1] {
				h |= bit
			}
			if r[8-x] > r[7-x] {
				flip |= bit
			}
		}
	}
	return h, flip
}

// Hamming is the number of differing bits.
func Hamming(a, b uint64) int { return bits.OnesCount64(a ^ b) }

// HashHex writes a hash as 16 lowercase hex characters.
func HashHex(h uint64) string { return fmt.Sprintf("%016x", h) }

// ParseHash reads a hash written by HashHex.
func ParseHash(s string) (uint64, error) {
	if len(s) != 16 {
		return 0, fmt.Errorf("dhash %q is not 16 hex characters", s)
	}
	h, err := strconv.ParseUint(s, 16, 64)
	if err != nil {
		return 0, fmt.Errorf("dhash %q is not hex", s)
	}
	return h, nil
}

// thumbSide is the edge of thumb32.
const thumbSide = 32

// thumb32 is the 32x32 luma thumbnail of a working copy (aspect squashed),
// rounded to uint8, row-major.
func thumb32(w []float32, ww, wh int) []byte {
	p := BoxDown(w, ww, wh, thumbSide, thumbSide)
	if p == nil {
		return nil
	}
	out := make([]byte, len(p))
	for i, v := range p {
		out[i] = clamp8(v)
	}
	return out
}

// DecodeThumb32 reads Metrics.Thumb32; nil when it is missing or not 1024
// bytes.
func DecodeThumb32(s string) []byte {
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil || len(b) != thumbSide*thumbSide {
		return nil
	}
	return b
}

// Mirror32 mirrors a 32x32 thumbnail left to right; nil when b is not 1024
// bytes.
func Mirror32(b []byte) []byte {
	if len(b) != thumbSide*thumbSide {
		return nil
	}
	out := make([]byte, len(b))
	for y := 0; y < thumbSide; y++ {
		for x := 0; x < thumbSide; x++ {
			out[y*thumbSide+x] = b[y*thumbSide+thumbSide-1-x]
		}
	}
	return out
}

// NCC32 is the normalised cross-correlation of two 32x32 thumbnails,
// sum((a-ā)(b-b̄)) / sqrt(sum((a-ā)²) sum((b-b̄)²)). It is 0 (undefined, so
// never a confirmation) when the sizes differ or either thumbnail's variance
// is below 1.
func NCC32(a, b []byte) float64 {
	n := len(a)
	if n == 0 || n != len(b) {
		return 0
	}
	var sa, sb float64
	for i := range a {
		sa += float64(a[i])
		sb += float64(b[i])
	}
	ma, mb := sa/float64(n), sb/float64(n)
	var ab, aa, bb float64
	for i := range a {
		da, db := float64(a[i])-ma, float64(b[i])-mb
		ab += da * db
		aa += da * da
		bb += db * db
	}
	if aa/float64(n) < 1 || bb/float64(n) < 1 {
		return 0
	}
	return ab / math.Sqrt(aa*bb)
}

// layoutCells is the 4x4 CIELAB grid behind Metrics.Layout.
const layoutCells = 16

// LayoutSim compares two Metrics.Layout grids: clamp(1 - meanΔE76/50, 0, 1)
// over the 16 cells. It is 0 when either grid is missing or malformed.
func LayoutSim(a, b string) float64 {
	pa, err1 := base64.StdEncoding.DecodeString(a)
	pb, err2 := base64.StdEncoding.DecodeString(b)
	if err1 != nil || err2 != nil || len(pa) != 3*layoutCells || len(pb) != 3*layoutCells {
		return 0
	}
	var sum float64
	for i := 0; i < layoutCells; i++ {
		dl := (float64(pa[3*i]) - float64(pb[3*i])) / 2.55
		da := float64(pa[3*i+1]) - float64(pb[3*i+1])
		db := float64(pa[3*i+2]) - float64(pb[3*i+2])
		sum += math.Sqrt(dl*dl + da*da + db*db)
	}
	return math.Min(1, math.Max(0, 1-sum/layoutCells/50))
}

func clamp8(v float32) byte {
	switch {
	case v <= 0:
		return 0
	case v >= 255:
		return 255
	}
	return byte(math.Round(float64(v)))
}
