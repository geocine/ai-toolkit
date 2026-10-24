package imgmeta

import "math"

// stdLuma is the IJG (Annex K) luminance table in zigzag order, the order
// DQT stores tables in. It matches image/jpeg's unscaled table.
var stdLuma = [64]float64{
	16, 11, 12, 14, 12, 10, 16, 14,
	13, 14, 18, 17, 16, 19, 24, 40,
	26, 24, 22, 22, 24, 49, 35, 37,
	29, 40, 58, 51, 61, 60, 57, 51,
	56, 55, 64, 72, 92, 78, 64, 68,
	87, 69, 55, 56, 80, 109, 81, 87,
	95, 98, 103, 104, 103, 62, 77, 113,
	121, 112, 100, 120, 92, 101, 103, 99,
}

// JPEGQuality estimates the IJG quality factor (1..100) from a DQT segment
// payload (the bytes after the length field, which may hold several
// tables). It uses the luma table (id 0): s = mean(100·Q[i]/T[i]) against
// the standard table T, then q = (200-s)/2 when s <= 100, else 5000/s. It
// returns 0 when the payload holds no complete luma table.
func JPEGQuality(dqt []byte) int {
	p := 0
	for p < len(dqt) {
		pq, tq := dqt[p]>>4, dqt[p]&0x0F
		p++
		size := 64
		if pq == 1 {
			size = 128
		} else if pq != 0 {
			return 0
		}
		if p+size > len(dqt) {
			return 0
		}
		if tq != 0 {
			p += size
			continue
		}
		var sum float64
		for i := 0; i < 64; i++ {
			v := float64(dqt[p+i])
			if pq == 1 {
				v = float64(uint16(dqt[p+2*i])<<8 | uint16(dqt[p+2*i+1]))
			}
			if v == 0 {
				return 0 // not a valid quantiser
			}
			sum += 100 * v / stdLuma[i]
		}
		s := sum / 64
		var q float64
		if s <= 100 {
			q = (200 - s) / 2
		} else {
			q = 5000 / s
		}
		return int(math.Max(1, math.Min(100, math.Round(q))))
	}
	return 0
}
