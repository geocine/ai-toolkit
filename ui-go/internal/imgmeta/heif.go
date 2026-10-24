package imgmeta

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
)

// Limits for reading a HEIF container. The meta box of a phone HEIC is a
// few KB (a grid photo lists a few dozen tiles) and its Exif item is a few
// tens of KB.
const (
	maxMetaSize  = 16 << 20
	maxExifSize  = 4 << 20
	maxTopBoxes  = 64
	maxHEIFItems = 4096
)

var (
	errNoHEIF = errors.New("imgmeta: not a HEIF file")
	errNoMeta = errors.New("imgmeta: no meta box")
	errNoExif = errors.New("imgmeta: no Exif item")
	errBadBox = errors.New("imgmeta: malformed box")
)

// heifFormat names the image format a ftyp box at the start of head
// declares: "heic" (HEVC), "avif" (AV1), "heif" (another HEIF codec), or "".
func heifFormat(head []byte) string {
	if len(head) < 16 || string(head[4:8]) != "ftyp" {
		return ""
	}
	n := int(binary.BigEndian.Uint32(head))
	if n < 16 || n > len(head) {
		n = min(len(head), 256)
	}
	heic, avif, heif := false, false, false
	check := func(b string) {
		switch b {
		case "heic", "heix", "heim", "heis", "hevc", "hevx", "hevm", "hevs":
			heic = true
		case "avif", "avis":
			avif = true
		case "mif1", "msf1", "mif2":
			heif = true
		}
	}
	major := string(head[8:12])
	check(major)
	for p := 16; p+4 <= n; p += 4 {
		check(string(head[p : p+4]))
	}
	// The major brand decides when it names a codec; an AVIF often lists
	// mif1 and a HEIC lists mif1 too.
	switch {
	case major == "avif" || major == "avis":
		return "avif"
	case heic:
		return "heic"
	case avif:
		return "avif"
	case heif:
		return "heif"
	}
	return ""
}

// IsHEIF reports whether head starts with an ISOBMFF ftyp box of an image
// brand of the HEIF family (HEIC, HEIF or AVIF).
func IsHEIF(head []byte) bool { return heifFormat(head) != "" }

// HEIFExif returns the TIFF bytes of the Exif item of a HEIF file (HEIC,
// HEIF or AVIF): the item named in meta/iinf, located through meta/iloc
// (file offsets or the meta box's idat), with the 4-byte TIFF header offset
// skipped. The bytes start with "II" or "MM" and can go to ParseTIFF.
func HEIFExif(r io.ReaderAt, size int64) ([]byte, error) {
	var ftyp [64]byte
	n, _ := r.ReadAt(ftyp[:], 0)
	if !IsHEIF(ftyp[:n]) {
		return nil, errNoHEIF
	}
	meta, err := readMetaBox(r, size)
	if err != nil {
		return nil, err
	}
	m, err := parseHEIFMeta(meta)
	if err != nil {
		return nil, err
	}
	return m.exif(func(off, n int64) ([]byte, error) {
		if off < 0 || n < 0 || off > size || n > size-off {
			return nil, errBadBox
		}
		b := make([]byte, n)
		if got, _ := r.ReadAt(b, off); int64(got) < n {
			return nil, errShort
		}
		return b, nil
	}, size)
}

// readMetaBox finds the top-level meta box and returns its body (version
// and flags included).
func readMetaBox(r io.ReaderAt, size int64) ([]byte, error) {
	var off int64
	for i := 0; i < maxTopBoxes && off+8 <= size; i++ {
		var h [16]byte
		n, _ := r.ReadAt(h[:], off)
		if n < 8 {
			break
		}
		bsize := int64(binary.BigEndian.Uint32(h[:4]))
		typ := string(h[4:8])
		hdr := int64(8)
		switch bsize {
		case 0:
			bsize = size - off
		case 1:
			if n < 16 {
				return nil, errBadBox
			}
			bsize = int64(binary.BigEndian.Uint64(h[8:16]))
			hdr = 16
		}
		if bsize < hdr || bsize > size-off {
			return nil, errBadBox
		}
		if typ == "meta" {
			body := bsize - hdr
			if body > maxMetaSize {
				return nil, errBadBox
			}
			b := make([]byte, body)
			if got, _ := r.ReadAt(b, off+hdr); int64(got) < body {
				return nil, errShort
			}
			return b, nil
		}
		off += bsize
	}
	return nil, errNoMeta
}

// heifItem is one item of meta/iinf with its location from meta/iloc.
type heifItem struct {
	typ     string
	method  int // iloc construction_method: 0 file offset, 1 idat offset
	base    int64
	extents [][2]int64 // offset, length
	located bool
	w, h    int // ispe
}

type heifMeta struct {
	primary uint32
	items   map[uint32]*heifItem
	idat    []byte
}

func (m *heifMeta) item(id uint32) *heifItem {
	it := m.items[id]
	if it == nil && len(m.items) < maxHEIFItems {
		it = &heifItem{}
		m.items[id] = it
	}
	if it == nil {
		return &heifItem{} // over the cap: a throwaway
	}
	return it
}

// parseHEIFMeta reads pitm, iinf, iloc, idat and the ispe properties out of
// a meta box body.
func parseHEIFMeta(meta []byte) (*heifMeta, error) {
	if len(meta) < 4 {
		return nil, errBadBox
	}
	m := &heifMeta{items: map[uint32]*heifItem{}}
	var props [][]byte // ipco children, 1-based in ipma
	var ipma []byte
	eachBox(meta[4:], func(typ string, body []byte) bool {
		switch typ {
		case "pitm":
			c := cursor{b: body}
			v := c.u8()
			c.skip(3)
			if v == 0 {
				m.primary = uint32(c.u16())
			} else {
				m.primary = c.u32()
			}
		case "iinf":
			parseIINF(m, body)
		case "iloc":
			parseILOC(m, body)
		case "idat":
			m.idat = body
		case "iprp":
			eachBox(body, func(t string, b []byte) bool {
				switch t {
				case "ipco":
					eachBox(b, func(pt string, pb []byte) bool {
						props = append(props, append([]byte(pt), pb...))
						return len(props) < maxHEIFItems
					})
				case "ipma":
					ipma = b
				}
				return true
			})
		}
		return true
	})
	if ipma != nil {
		parseIPMA(m, ipma, props)
	}
	return m, nil
}

func parseIINF(m *heifMeta, body []byte) {
	c := cursor{b: body}
	v := c.u8()
	c.skip(3)
	if v == 0 {
		c.u16()
	} else {
		c.u32()
	}
	if c.bad {
		return
	}
	eachBox(c.rest(), func(typ string, b []byte) bool {
		if typ != "infe" {
			return true
		}
		e := cursor{b: b}
		ver := e.u8()
		e.skip(3)
		if ver < 2 {
			return true // v0/v1 entries carry no item type
		}
		var id uint32
		if ver == 2 {
			id = uint32(e.u16())
		} else {
			id = e.u32()
		}
		e.u16() // item_protection_index
		typ4 := e.n(4)
		if !e.bad {
			m.item(id).typ = string(typ4)
		}
		return len(m.items) < maxHEIFItems
	})
}

func parseILOC(m *heifMeta, body []byte) {
	c := cursor{b: body}
	ver := c.u8()
	c.skip(3)
	if ver > 2 {
		return
	}
	a, b := c.u8(), c.u8()
	offSize, lenSize, baseSize := int(a>>4), int(a&0x0F), int(b>>4)
	idxSize := 0
	if ver == 1 || ver == 2 {
		idxSize = int(b & 0x0F)
	}
	var count uint32
	if ver < 2 {
		count = uint32(c.u16())
	} else {
		count = c.u32()
	}
	for i := uint32(0); i < count && !c.bad && i < maxHEIFItems; i++ {
		var id uint32
		if ver < 2 {
			id = uint32(c.u16())
		} else {
			id = c.u32()
		}
		method := 0
		if ver == 1 || ver == 2 {
			method = int(c.u16() & 0x0F)
		}
		c.u16() // data_reference_index
		base := c.uintN(baseSize)
		n := int(c.u16())
		// Bound the extent loop by the bytes left: with every size 0 an
		// extent reads nothing, so keep at most one (it carries no data).
		if per := idxSize + offSize + lenSize; per == 0 {
			n = min(n, 1)
		} else if n*per > len(c.rest()) {
			return
		}
		var ext [][2]int64
		for j := 0; j < n && !c.bad; j++ {
			c.uintN(idxSize)
			off := c.uintN(offSize)
			l := c.uintN(lenSize)
			if len(ext) < 64 {
				ext = append(ext, [2]int64{off, l})
			}
		}
		if c.bad {
			return
		}
		it := m.item(id)
		it.method, it.base, it.extents, it.located = method, base, ext, true
	}
}

func parseIPMA(m *heifMeta, body []byte, props [][]byte) {
	c := cursor{b: body}
	ver := c.u8()
	flags := uint32(c.u8())<<16 | uint32(c.u16())
	count := c.u32()
	for i := uint32(0); i < count && !c.bad && i < maxHEIFItems; i++ {
		var id uint32
		if ver < 1 {
			id = uint32(c.u16())
		} else {
			id = c.u32()
		}
		n := int(c.u8())
		for j := 0; j < n && !c.bad; j++ {
			var idx int
			if flags&1 != 0 {
				idx = int(c.u16() & 0x7FFF)
			} else {
				idx = int(c.u8() & 0x7F)
			}
			if c.bad || idx < 1 || idx > len(props) {
				continue
			}
			p := props[idx-1]
			if string(p[:4]) != "ispe" || len(p) < 16 {
				continue
			}
			// ispe: type, version+flags, then width and height.
			it := m.item(id)
			it.w = int(binary.BigEndian.Uint32(p[8:12]) & 0x3FFFFFFF)
			it.h = int(binary.BigEndian.Uint32(p[12:16]) & 0x3FFFFFFF)
		}
	}
}

// primarySize is the ispe of the primary item (for a grid photo, the whole
// picture), or zeros.
func (m *heifMeta) primarySize() (int, int) {
	if it := m.items[m.primary]; it != nil {
		return it.w, it.h
	}
	return 0, 0
}

// exif reads the first located Exif item, through read for file offsets.
func (m *heifMeta) exif(read func(off, n int64) ([]byte, error), size int64) ([]byte, error) {
	var it *heifItem
	var bestID uint32
	for id, x := range m.items {
		if x.typ == "Exif" && x.located && len(x.extents) > 0 && (it == nil || id < bestID) {
			it, bestID = x, id
		}
	}
	if it == nil {
		return nil, errNoExif
	}
	var data []byte
	for _, e := range it.extents {
		off, n := it.base+e[0], e[1]
		switch it.method {
		case 0:
			if n == 0 && len(it.extents) == 1 { // 0 = to the end of the file
				n = size - off
			}
			if n < 0 || int64(len(data))+n > maxExifSize {
				return nil, errBadBox
			}
			b, err := read(off, n)
			if err != nil {
				return nil, err
			}
			data = append(data, b...)
		case 1:
			if n == 0 && len(it.extents) == 1 {
				n = int64(len(m.idat)) - off
			}
			if off < 0 || n < 0 || off > int64(len(m.idat)) || n > int64(len(m.idat))-off ||
				int64(len(data))+n > maxExifSize {
				return nil, errBadBox
			}
			data = append(data, m.idat[off:off+n]...)
		default:
			return nil, errNoExif
		}
	}
	// The item starts with exif_tiff_header_offset (big-endian u32), the
	// number of bytes before the TIFF header.
	if len(data) < 4 {
		return nil, errShort
	}
	skip := int64(binary.BigEndian.Uint32(data))
	if skip > int64(len(data))-4 {
		return nil, errShort
	}
	tiff := bytes.TrimPrefix(data[4+skip:], exifPrefix)
	if len(tiff) < 8 || !(bytes.HasPrefix(tiff, []byte("II*\x00")) || bytes.HasPrefix(tiff, []byte("MM\x00*"))) {
		return nil, errNotTIFF
	}
	return tiff, nil
}

// sniffHEIF fills the format, the primary item's stored size and the Exif
// found inside head.
func sniffHEIF(head []byte, s *Sniff) {
	s.Format = heifFormat(head)
	r := bytes.NewReader(head)
	meta, err := readMetaBox(r, int64(len(head)))
	if err != nil {
		return
	}
	m, err := parseHEIFMeta(meta)
	if err != nil {
		return
	}
	s.Width, s.Height = m.primarySize()
	tiff, err := m.exif(func(off, n int64) ([]byte, error) {
		if off < 0 || n < 0 || off > int64(len(head)) || n > int64(len(head))-off {
			return nil, errShort
		}
		return head[off : off+n], nil
	}, int64(len(head)))
	if err != nil {
		return
	}
	if e, err := ParseTIFF(tiff); err == nil {
		s.Exif, s.HasExif = e, true
	}
}

// eachBox calls fn with the type and body of each box in b. It stops at the
// first malformed header or when fn returns false.
func eachBox(b []byte, fn func(typ string, body []byte) bool) {
	for len(b) >= 8 {
		n := uint64(binary.BigEndian.Uint32(b))
		typ := string(b[4:8])
		hdr := uint64(8)
		switch n {
		case 0:
			n = uint64(len(b))
		case 1:
			if len(b) < 16 {
				return
			}
			n = binary.BigEndian.Uint64(b[8:16])
			hdr = 16
		}
		if n < hdr || n > uint64(len(b)) {
			return
		}
		if !fn(typ, b[hdr:n]) {
			return
		}
		b = b[n:]
	}
}

// cursor reads big-endian fields; once a read runs past the end it sets bad
// and returns zeros from then on.
type cursor struct {
	b   []byte
	p   int
	bad bool
}

func (c *cursor) n(k int) []byte {
	if c.bad || k < 0 || c.p+k > len(c.b) {
		c.bad = true
		return make([]byte, max(k, 0))
	}
	s := c.b[c.p : c.p+k]
	c.p += k
	return s
}

func (c *cursor) skip(k int)   { c.n(k) }
func (c *cursor) u8() byte     { return c.n(1)[0] }
func (c *cursor) u16() uint16  { return binary.BigEndian.Uint16(c.n(2)) }
func (c *cursor) u32() uint32  { return binary.BigEndian.Uint32(c.n(4)) }
func (c *cursor) rest() []byte { return c.b[min(c.p, len(c.b)):] }

// uintN reads a 0, 4 or 8 byte unsigned field (iloc sizes); other sizes
// mark the cursor bad.
func (c *cursor) uintN(k int) int64 {
	switch k {
	case 0:
		return 0
	case 4:
		return int64(c.u32())
	case 8:
		v := binary.BigEndian.Uint64(c.n(8))
		if v > 1<<62 {
			c.bad = true
			return 0
		}
		return int64(v)
	}
	c.bad = true
	return 0
}
