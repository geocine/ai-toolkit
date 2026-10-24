// Package imgmeta reads the facts a photo check needs from the first bytes of
// an image file: EXIF (orientation, capture time, camera, software), the
// format and stored size, PNG pixel format and text keys, WebP animation,
// JPEG components and quality, and the C2PA "made by a model" marker. It also
// pulls EXIF out of a HEIF container and writes a small EXIF block back into a
// JPEG, so a HEIC converted to JPEG keeps its date and camera.
//
// Everything here works on untrusted bytes: every read is bounds-checked and
// no input, truncated or garbage, makes a function panic.
package imgmeta

import (
	"bytes"
	"encoding/binary"
	"errors"
	"strings"
	"time"
)

// Exif is the part of EXIF the photo checks use. Orientation is 1..8 and 1
// when the tag is absent or out of range. Time is DateTimeOriginal as
// "2006-01-02T15:04:05" (camera-local, no zone) and SubSec the digits of
// SubSecTimeOriginal; both are empty when absent.
type Exif struct {
	Orientation int
	Time        string
	SubSec      string
	Make        string
	Model       string
	Software    string
}

// Tags this package reads and writes.
const (
	tagImageWidth  = 0x0100
	tagImageLength = 0x0101
	tagMake        = 0x010F
	tagModel       = 0x0110
	tagOrientation = 0x0112
	tagSoftware    = 0x0131
	tagExifIFD     = 0x8769
	tagDateTimeOri = 0x9003
	tagSubSecOri   = 0x9291
)

// TIFF field types.
const (
	typeASCII = 2
	typeShort = 3
	typeLong  = 4
)

const (
	maxIFDEntries  = 512 // real IFDs hold a few dozen
	maxStringLen   = 255
	exifTimeLayout = "2006:01:02 15:04:05"
	isoTimeLayout  = "2006-01-02T15:04:05"
)

var (
	errNotTIFF = errors.New("imgmeta: not a TIFF header")
	errShort   = errors.New("imgmeta: truncated")

	exifPrefix = []byte("Exif\x00\x00")
)

// ParseTIFF reads Exif from a TIFF-structured EXIF block (the bytes after
// "Exif\0\0" in a JPEG APP1, a PNG eXIf chunk, a WebP EXIF chunk or a HEIF
// Exif item). Both byte orders are read. Tags whose values fall outside b are
// skipped, so a truncated block still yields what it holds.
func ParseTIFF(b []byte) (Exif, error) {
	e, _, err := parseTIFF(b)
	return e, err
}

// tiffDims is IFD0's ImageWidth and ImageLength, set only for TIFF files.
type tiffDims struct{ w, h int }

func parseTIFF(b []byte) (Exif, tiffDims, error) {
	e := Exif{Orientation: 1}
	var d tiffDims
	if len(b) < 8 {
		return e, d, errShort
	}
	var bo binary.ByteOrder
	switch {
	case b[0] == 'I' && b[1] == 'I' && b[2] == 42 && b[3] == 0:
		bo = binary.LittleEndian
	case b[0] == 'M' && b[1] == 'M' && b[2] == 0 && b[3] == 42:
		bo = binary.BigEndian
	default:
		return e, d, errNotTIFF
	}
	t := tiff{b: b, bo: bo}
	ifd0 := int64(bo.Uint32(b[4:8]))
	exifIFD := int64(-1)
	t.walk(ifd0, func(tag, typ uint16, count uint32, val []byte) {
		switch tag {
		case tagOrientation:
			if o := t.uint(typ, count, val); o >= 1 && o <= 8 {
				e.Orientation = o
			}
		case tagMake:
			e.Make = t.ascii(typ, count, val)
		case tagModel:
			e.Model = t.ascii(typ, count, val)
		case tagSoftware:
			e.Software = t.ascii(typ, count, val)
		case tagExifIFD:
			if typ == typeLong && count == 1 {
				exifIFD = int64(bo.Uint32(val))
			}
		case tagImageWidth:
			d.w = t.uint(typ, count, val)
		case tagImageLength:
			d.h = t.uint(typ, count, val)
		}
	})
	if exifIFD > 0 && exifIFD != ifd0 {
		t.walk(exifIFD, func(tag, typ uint16, count uint32, val []byte) {
			switch tag {
			case tagDateTimeOri:
				e.Time = exifToISO(t.ascii(typ, count, val))
			case tagSubSecOri:
				e.SubSec = digits(t.ascii(typ, count, val))
			}
		})
	}
	return e, d, nil
}

// tiff reads IFD entries out of one EXIF block.
type tiff struct {
	b  []byte
	bo binary.ByteOrder
}

// walk calls fn for each entry of the IFD at off. val is the 4-byte value
// field: the value itself when it fits, else an offset into t.b.
func (t tiff) walk(off int64, fn func(tag, typ uint16, count uint32, val []byte)) {
	if off < 8 || off+2 > int64(len(t.b)) {
		return
	}
	n := int(t.bo.Uint16(t.b[off:]))
	if n > maxIFDEntries {
		n = maxIFDEntries
	}
	p := off + 2
	for i := 0; i < n; i++ {
		if p+12 > int64(len(t.b)) {
			return
		}
		ent := t.b[p : p+12]
		fn(t.bo.Uint16(ent[0:]), t.bo.Uint16(ent[2:]), t.bo.Uint32(ent[4:]), ent[8:12])
		p += 12
	}
}

// uint reads a single SHORT or LONG value, or returns 0.
func (t tiff) uint(typ uint16, count uint32, val []byte) int {
	if count != 1 {
		return 0
	}
	switch typ {
	case typeShort:
		return int(t.bo.Uint16(val))
	case typeLong:
		v := t.bo.Uint32(val)
		if v > 1<<30 {
			return 0
		}
		return int(v)
	}
	return 0
}

// ascii reads an ASCII value up to its first NUL, trimmed and capped.
func (t tiff) ascii(typ uint16, count uint32, val []byte) string {
	if typ != typeASCII || count == 0 {
		return ""
	}
	var s []byte
	if count <= 4 {
		s = val[:count]
	} else {
		off := int64(t.bo.Uint32(val))
		end := off + int64(count)
		if off < 0 || end > int64(len(t.b)) {
			return ""
		}
		s = t.b[off:end]
	}
	if i := bytes.IndexByte(s, 0); i >= 0 {
		s = s[:i]
	}
	if len(s) > maxStringLen {
		s = s[:maxStringLen]
	}
	return strings.TrimSpace(strings.ToValidUTF8(string(s), ""))
}

// exifToISO turns "2006:01:02 15:04:05" into "2006-01-02T15:04:05"; anything
// that is not a valid date gives "" (cameras write "0000:00:00 00:00:00" or
// blanks when the clock was never set).
func exifToISO(s string) string {
	if len(s) < len(exifTimeLayout) {
		return ""
	}
	t, err := time.Parse(exifTimeLayout, s[:len(exifTimeLayout)])
	if err != nil {
		return ""
	}
	return t.Format(isoTimeLayout)
}

// digits keeps the leading decimal digits of s (SubSecTimeOriginal is
// "123" or "123  " in practice).
func digits(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return s[:i]
		}
	}
	return s
}

// FromJPEG reads the Exif APP1 of a JPEG from the first bytes of the file.
// The bool is false when there is no readable Exif block before the image
// data.
func FromJPEG(head []byte) (Exif, bool) {
	var e Exif
	ok := false
	walkJPEG(head, func(marker byte, seg []byte) bool {
		if marker == 0xE1 && bytes.HasPrefix(seg, exifPrefix) {
			if x, err := ParseTIFF(seg[len(exifPrefix):]); err == nil {
				e, ok = x, true
				return false
			}
		}
		return true
	})
	if !ok {
		return Exif{}, false
	}
	return e, true
}

// FromPNG reads the eXIf chunk of a PNG from the first bytes of the file.
func FromPNG(head []byte) (Exif, bool) {
	var e Exif
	ok := false
	walkPNG(head, func(typ string, data []byte) bool {
		if typ == "eXIf" {
			if x, err := ParseTIFF(data); err == nil {
				e, ok = x, true
			}
			return false
		}
		return true
	})
	if !ok {
		return Exif{}, false
	}
	return e, true
}

// FromWebP reads the EXIF chunk of a WebP from the first bytes of the file.
// Some writers keep the JPEG "Exif\0\0" prefix inside the chunk; it is
// skipped.
func FromWebP(head []byte) (Exif, bool) {
	var e Exif
	ok := false
	walkWebP(head, func(fourcc string, data []byte) bool {
		if fourcc == "EXIF" {
			data = bytes.TrimPrefix(data, exifPrefix)
			if x, err := ParseTIFF(data); err == nil {
				e, ok = x, true
			}
			return false
		}
		return true
	})
	if !ok {
		return Exif{}, false
	}
	return e, true
}

// walkJPEG calls fn with each marker segment's payload (after the length
// field) up to the first SOS or EOI, and stops early when fn returns false.
// A segment cut off by the end of head is passed with what is there.
func walkJPEG(head []byte, fn func(marker byte, seg []byte) bool) {
	if len(head) < 4 || head[0] != 0xFF || head[1] != 0xD8 {
		return
	}
	p := 2
	for p+1 < len(head) {
		if head[p] != 0xFF {
			return // lost sync: not a marker where one must be
		}
		m := head[p+1]
		if m == 0xFF { // fill byte
			p++
			continue
		}
		p += 2
		switch {
		case m == 0xD9 || m == 0xDA: // EOI, SOS: header ends
			return
		case m == 0x01 || (m >= 0xD0 && m <= 0xD7): // standalone markers
			continue
		}
		if p+2 > len(head) {
			return
		}
		n := int(binary.BigEndian.Uint16(head[p:]))
		if n < 2 {
			return
		}
		end := p + n
		if end > len(head) {
			end = len(head)
		}
		if !fn(m, head[p+2:end]) {
			return
		}
		p += n
	}
}

var pngSig = []byte("\x89PNG\r\n\x1a\n")

// walkPNG calls fn with each chunk's type and data until IEND or the end of
// head. A chunk cut off by the end of head is passed with what is there.
func walkPNG(head []byte, fn func(typ string, data []byte) bool) {
	if !bytes.HasPrefix(head, pngSig) {
		return
	}
	p := len(pngSig)
	for p+8 <= len(head) {
		n := int64(binary.BigEndian.Uint32(head[p:]))
		typ := string(head[p+4 : p+8])
		start := int64(p + 8)
		end := start + n
		if end > int64(len(head)) {
			end = int64(len(head))
		}
		if !fn(typ, head[start:end]) || typ == "IEND" {
			return
		}
		next := start + n + 4 // data + CRC
		if next > int64(len(head)) {
			return
		}
		p = int(next)
	}
}

// walkWebP calls fn with each RIFF chunk's FourCC and payload. A chunk cut
// off by the end of head is passed with what is there.
func walkWebP(head []byte, fn func(fourcc string, data []byte) bool) {
	if len(head) < 12 || string(head[0:4]) != "RIFF" || string(head[8:12]) != "WEBP" {
		return
	}
	p := 12
	for p+8 <= len(head) {
		fourcc := string(head[p : p+4])
		n := int64(binary.LittleEndian.Uint32(head[p+4:]))
		start := int64(p + 8)
		end := start + n
		if end > int64(len(head)) {
			end = int64(len(head))
		}
		if !fn(fourcc, head[start:end]) {
			return
		}
		next := start + n + n&1 // chunks are padded to an even size
		if next > int64(len(head)) {
			return
		}
		p = int(next)
	}
}
