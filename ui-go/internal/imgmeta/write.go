package imgmeta

import (
	"bytes"
	"encoding/binary"
	"errors"
	"strings"
	"time"
)

var (
	errNotJPEG = errors.New("imgmeta: not a JPEG")
	errBadAPP1 = errors.New("imgmeta: not an Exif APP1 segment")
	errNoSOS   = errors.New("imgmeta: JPEG ends before its image data")
)

// BuildAPP1 returns a JPEG APP1 marker segment (FF E1, length, "Exif\0\0",
// little-endian TIFF) holding IFD0 Make and Model, and an Exif IFD with
// DateTimeOriginal and SubSecTimeOriginal. It never writes Orientation (the
// pixels it goes with are already upright), Software or GPS. It returns nil
// when e has none of those fields.
func BuildAPP1(e Exif) []byte {
	keep := Exif{Make: e.Make, Model: e.Model, Time: e.Time, SubSec: e.SubSec}
	if keep.Make == "" && keep.Model == "" && exifTime(keep.Time) == "" {
		return nil
	}
	return APP1(EncodeTIFF(keep, binary.LittleEndian))
}

// APP1 wraps TIFF bytes into an Exif APP1 marker segment, or returns nil
// when they do not fit in one segment.
func APP1(tiff []byte) []byte {
	n := 2 + len(exifPrefix) + len(tiff)
	if n > 0xFFFF {
		return nil
	}
	b := make([]byte, 0, 2+n)
	b = append(b, 0xFF, 0xE1, byte(n>>8), byte(n))
	b = append(b, exifPrefix...)
	return append(b, tiff...)
}

// EncodeTIFF writes e as a TIFF EXIF block in the given byte order: IFD0
// Make, Model, Orientation (when 1..8), Software and the Exif IFD pointer;
// the Exif IFD DateTimeOriginal and SubSecTimeOriginal. Empty fields are
// left out. Time may be "2006-01-02T15:04:05" (optionally with a ".123"
// fraction, used when SubSec is empty) or EXIF's "2006:01:02 15:04:05".
func EncodeTIFF(e Exif, bo binary.ByteOrder) []byte {
	var ifd0, sub []tiffEntry
	if e.Make != "" {
		ifd0 = append(ifd0, asciiEntry(tagMake, e.Make))
	}
	if e.Model != "" {
		ifd0 = append(ifd0, asciiEntry(tagModel, e.Model))
	}
	if e.Orientation >= 1 && e.Orientation <= 8 {
		ifd0 = append(ifd0, tiffEntry{tag: tagOrientation, typ: typeShort, count: 1, short: uint16(e.Orientation)})
	}
	if e.Software != "" {
		ifd0 = append(ifd0, asciiEntry(tagSoftware, e.Software))
	}
	subsec := digits(e.SubSec)
	if t := exifTime(e.Time); t != "" {
		sub = append(sub, asciiEntry(tagDateTimeOri, t))
		if i := strings.IndexByte(e.Time, '.'); subsec == "" && i >= 0 {
			subsec = digits(e.Time[i+1:])
		}
	}
	if subsec != "" {
		sub = append(sub, asciiEntry(tagSubSecOri, subsec))
	}
	if len(sub) > 0 {
		ifd0 = append(ifd0, tiffEntry{tag: tagExifIFD, typ: typeLong, count: 1})
	}

	ifdLen := func(n int) int { return 2 + 12*n + 4 }
	ifd0Off := 8
	subOff := ifd0Off + ifdLen(len(ifd0))
	dataOff := subOff
	if len(sub) > 0 {
		dataOff += ifdLen(len(sub))
	}
	var data []byte
	place := func(es []tiffEntry) {
		for i := range es {
			if len(es[i].data) > 4 {
				es[i].off = uint32(dataOff + len(data))
				data = append(data, es[i].data...)
				if len(data)&1 == 1 { // values start on a word boundary
					data = append(data, 0)
				}
			}
		}
	}
	place(ifd0)
	place(sub)

	b := make([]byte, 8, dataOff+len(data))
	if bo == binary.BigEndian {
		copy(b, "MM\x00*")
	} else {
		copy(b, "II*\x00")
	}
	bo.PutUint32(b[4:], uint32(ifd0Off))
	var w [4]byte
	u16 := func(v uint16) { bo.PutUint16(w[:], v); b = append(b, w[:2]...) }
	u32 := func(v uint32) { bo.PutUint32(w[:], v); b = append(b, w[:]...) }
	writeIFD := func(es []tiffEntry) {
		u16(uint16(len(es)))
		for _, x := range es {
			u16(x.tag)
			u16(x.typ)
			u32(x.count)
			var v [4]byte
			switch {
			case x.tag == tagExifIFD:
				bo.PutUint32(v[:], uint32(subOff))
			case x.typ == typeShort:
				bo.PutUint16(v[:], x.short)
			case len(x.data) <= 4:
				copy(v[:], x.data)
			default:
				bo.PutUint32(v[:], x.off)
			}
			b = append(b, v[:]...)
		}
		u32(0) // no next IFD
	}
	writeIFD(ifd0)
	if len(sub) > 0 {
		writeIFD(sub)
	}
	return append(b, data...)
}

type tiffEntry struct {
	tag, typ uint16
	count    uint32
	short    uint16
	data     []byte // ASCII value with its NUL
	off      uint32 // where data lands when it does not fit in 4 bytes
}

func asciiEntry(tag uint16, s string) tiffEntry {
	if len(s) > maxStringLen {
		s = s[:maxStringLen]
	}
	d := append([]byte(s), 0)
	return tiffEntry{tag: tag, typ: typeASCII, count: uint32(len(d)), data: d}
}

// exifTime turns Exif.Time (ISO or EXIF layout, any fraction ignored) into
// EXIF's "2006:01:02 15:04:05", or "" when it is not a date.
func exifTime(s string) string {
	if len(s) < len(isoTimeLayout) {
		return ""
	}
	head := s[:len(isoTimeLayout)]
	for _, layout := range []string{isoTimeLayout, exifTimeLayout} {
		if t, err := time.Parse(layout, head); err == nil {
			return t.Format(exifTimeLayout)
		}
	}
	return ""
}

// InjectJPEG returns jpeg with app1 (from BuildAPP1 or APP1) as its Exif
// segment: every existing Exif APP1 is dropped and app1 goes right after
// SOI, or after a leading APP0 (JFIF). An empty app1 only strips the old
// Exif. The image data is copied unchanged.
func InjectJPEG(jpeg, app1 []byte) ([]byte, error) {
	if len(jpeg) < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
		return nil, errNotJPEG
	}
	if len(app1) > 0 {
		if len(app1) < 4+len(exifPrefix) || app1[0] != 0xFF || app1[1] != 0xE1 ||
			int(binary.BigEndian.Uint16(app1[2:])) != len(app1)-2 || !bytes.HasPrefix(app1[4:], exifPrefix) {
			return nil, errBadAPP1
		}
	}
	type seg struct {
		start, end int
		marker     byte
	}
	var segs []seg
	sos := -1
	p := 2
	for p+1 < len(jpeg) && sos < 0 {
		if jpeg[p] != 0xFF {
			return nil, errNotJPEG
		}
		m := jpeg[p+1]
		switch {
		case m == 0xFF:
			p++
		case m == 0xDA:
			sos = p
		case m == 0xD9:
			return nil, errNoSOS
		case m == 0x01 || (m >= 0xD0 && m <= 0xD7):
			segs = append(segs, seg{p, p + 2, m})
			p += 2
		default:
			if p+4 > len(jpeg) {
				return nil, errNoSOS
			}
			end := p + 2 + int(binary.BigEndian.Uint16(jpeg[p+2:]))
			if end < p+4 || end > len(jpeg) {
				return nil, errNoSOS
			}
			segs = append(segs, seg{p, end, m})
			p = end
		}
	}
	if sos < 0 {
		return nil, errNoSOS
	}
	out := make([]byte, 0, len(jpeg)+len(app1))
	out = append(out, 0xFF, 0xD8)
	placed := len(app1) == 0
	for i, s := range segs {
		if !placed && !(i == 0 && s.marker == 0xE0) {
			out = append(out, app1...)
			placed = true
		}
		if s.marker == 0xE1 && bytes.HasPrefix(jpeg[min(s.start+4, s.end):s.end], exifPrefix) {
			continue
		}
		out = append(out, jpeg[s.start:s.end]...)
	}
	if !placed {
		out = append(out, app1...)
	}
	return append(out, jpeg[sos:]...), nil
}
