package imgmeta

import (
	"bytes"
	"encoding/binary"
	"regexp"
	"strings"
)

// HeadSize is how much of a file SniffHead expects: the metadata of camera
// and phone photos sits well inside it.
const HeadSize = 256 << 10

// Sniff is what the first bytes of an image file say about it. Width and
// Height are as stored, before EXIF orientation (for HEIF, the primary
// item's ispe). Format is one of jpeg png webp gif bmp tiff heic heif avif
// jxl other.
type Sniff struct {
	Format         string
	Width          int
	Height         int
	PNGColorType   int // IHDR colour type; 0 when not a PNG
	PNGBitDepth    int
	JPEGComponents int // SOF component count; 4 is CMYK
	JPEGQuality    int // IJG quality from the luma DQT; 0 unknown
	Animated       bool
	C2PAGenerated  bool // a C2PA manifest says trainedAlgorithmicMedia
	Generated      bool // PNG generator text keys, EXIF Software or C2PA
	HasExif        bool
	Exif           Exif // Orientation is 1 when there is no EXIF
}

// Gray16 reports a 16-bit greyscale PNG (IHDR colour type 0, depth 16), the
// one PNG kind the trainer's loader clips to white. Grey+alpha 16 (type 4)
// and RGB 16 open correctly scaled.
func (s Sniff) Gray16() bool {
	return s.Format == "png" && s.PNGColorType == 0 && s.PNGBitDepth == 16
}

// generatorSoftware matches EXIF Software written by image models and
// their front ends.
var generatorSoftware = regexp.MustCompile(`(?i)stable diffusion|midjourney|dall.?e|comfyui|firefly|imagen|flux|novelai`)

// generatorKeys are PNG text keys image-model front ends write: A1111 and
// Forge "parameters", ComfyUI "prompt" and "workflow".
var generatorKeys = map[string]bool{"parameters": true, "prompt": true, "workflow": true}

// c2paAI is the IPTC digital source type C2PA uses for media a model made.
// Camera phones sign ordinary photos with C2PA too, so only this string,
// not the manifest itself, marks a photo as generated.
var c2paAI = []byte("trainedAlgorithmicMedia")

// SniffHead reads format, size and metadata from the first bytes of a file
// (HeadSize is enough). It never fails: what it cannot read stays zero.
func SniffHead(head []byte) Sniff {
	s := Sniff{Format: "other", Exif: Exif{Orientation: 1}}
	switch {
	case len(head) >= 3 && head[0] == 0xFF && head[1] == 0xD8 && head[2] == 0xFF:
		sniffJPEG(head, &s)
	case bytes.HasPrefix(head, pngSig):
		sniffPNG(head, &s)
	case len(head) >= 12 && string(head[0:4]) == "RIFF" && string(head[8:12]) == "WEBP":
		sniffWebP(head, &s)
	case bytes.HasPrefix(head, []byte("GIF87a")) || bytes.HasPrefix(head, []byte("GIF89a")):
		s.Format = "gif"
		if len(head) >= 10 {
			s.Width = int(binary.LittleEndian.Uint16(head[6:]))
			s.Height = int(binary.LittleEndian.Uint16(head[8:]))
		}
	case len(head) >= 2 && head[0] == 'B' && head[1] == 'M':
		s.Format = "bmp"
		if len(head) >= 26 {
			s.Width = abs(int(int32(binary.LittleEndian.Uint32(head[18:]))))
			s.Height = abs(int(int32(binary.LittleEndian.Uint32(head[22:])))) // negative = top-down
		}
	case bytes.HasPrefix(head, []byte("II*\x00")) || bytes.HasPrefix(head, []byte("MM\x00*")):
		s.Format = "tiff"
		if e, d, err := parseTIFF(head); err == nil {
			s.Exif, s.HasExif = e, true
			s.Width, s.Height = d.w, d.h
		}
	case IsHEIF(head):
		sniffHEIF(head, &s)
	case bytes.HasPrefix(head, []byte{0xFF, 0x0A}) ||
		bytes.HasPrefix(head, []byte("\x00\x00\x00\x0CJXL \r\n\x87\n")):
		s.Format = "jxl"
	}
	if generatorSoftware.MatchString(s.Exif.Software) || s.C2PAGenerated {
		s.Generated = true
	}
	return s
}

func sniffJPEG(head []byte, s *Sniff) {
	s.Format = "jpeg"
	var jumbf []byte
	walkJPEG(head, func(m byte, seg []byte) bool {
		switch {
		case m == 0xE1 && !s.HasExif && bytes.HasPrefix(seg, exifPrefix):
			if e, err := ParseTIFF(seg[len(exifPrefix):]); err == nil {
				s.Exif, s.HasExif = e, true
			}
		case m == 0xEB && len(seg) > 8 && seg[0] == 'J' && seg[1] == 'P':
			// APP11 JUMBF: "JP", box instance, sequence number, then the
			// box bytes; a manifest spans several segments.
			if len(jumbf) < HeadSize {
				jumbf = append(jumbf, seg[8:]...)
			}
		case m == 0xDB && s.JPEGQuality == 0:
			s.JPEGQuality = JPEGQuality(seg)
		case isSOF(m) && s.JPEGComponents == 0 && len(seg) >= 6:
			s.Height = int(binary.BigEndian.Uint16(seg[1:]))
			s.Width = int(binary.BigEndian.Uint16(seg[3:]))
			s.JPEGComponents = int(seg[5])
		}
		return true
	})
	s.C2PAGenerated = c2paGenerated(jumbf)
}

// isSOF is every start-of-frame marker (C0-CF less DHT C4, JPG C8, DAC CC).
func isSOF(m byte) bool {
	return m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC
}

func sniffPNG(head []byte, s *Sniff) {
	s.Format = "png"
	walkPNG(head, func(typ string, data []byte) bool {
		switch typ {
		case "IHDR":
			if len(data) >= 10 {
				s.Width = int(binary.BigEndian.Uint32(data[0:]) & 0x7FFFFFFF)
				s.Height = int(binary.BigEndian.Uint32(data[4:]) & 0x7FFFFFFF)
				s.PNGBitDepth = int(data[8])
				s.PNGColorType = int(data[9])
			}
		case "acTL": // APNG: only the first frame trains
			s.Animated = true
		case "eXIf":
			if e, err := ParseTIFF(data); err == nil {
				s.Exif, s.HasExif = e, true
			}
		case "tEXt", "iTXt", "zTXt":
			key, text := data, []byte(nil)
			if i := bytes.IndexByte(key, 0); i >= 0 {
				key, text = key[:i], key[i+1:]
			}
			if generatorKeys[strings.ToLower(string(key))] {
				s.Generated = true
			}
			// NovelAI writes no EXIF: its name sits in a "Software" text
			// chunk. zTXt and compressed iTXt text are not inflated.
			if text != nil && strings.EqualFold(string(key), "Software") && typ != "zTXt" {
				if typ == "iTXt" {
					text = itxtText(text)
				}
				if text != nil && generatorSoftware.Match(text) {
					s.Generated = true
				}
			}
		case "caBX": // C2PA in PNG
			if c2paGenerated(data) {
				s.C2PAGenerated = true
			}
		}
		return true
	})
}

// itxtText is the text of an iTXt chunk after its keyword's NUL:
// compression flag and method, language tag, translated keyword, then the
// text. It is nil when the text is compressed or the fields run short.
func itxtText(b []byte) []byte {
	if len(b) < 2 || b[0] != 0 {
		return nil
	}
	b = b[2:]
	for range 2 { // language tag, translated keyword
		i := bytes.IndexByte(b, 0)
		if i < 0 {
			return nil
		}
		b = b[i+1:]
	}
	return b
}

func sniffWebP(head []byte, s *Sniff) {
	s.Format = "webp"
	walkWebP(head, func(fourcc string, data []byte) bool {
		switch fourcc {
		case "VP8X":
			if len(data) >= 10 {
				if data[0]&0x02 != 0 {
					s.Animated = true
				}
				s.Width = int(uint32(data[4])|uint32(data[5])<<8|uint32(data[6])<<16) + 1
				s.Height = int(uint32(data[7])|uint32(data[8])<<8|uint32(data[9])<<16) + 1
			}
		case "ANIM", "ANMF":
			s.Animated = true
		case "VP8 ":
			// Frame tag (3 bytes), start code 9d 01 2a, then 14-bit sizes.
			if s.Width == 0 && len(data) >= 10 && data[3] == 0x9D && data[4] == 0x01 && data[5] == 0x2A {
				s.Width = int(binary.LittleEndian.Uint16(data[6:]) & 0x3FFF)
				s.Height = int(binary.LittleEndian.Uint16(data[8:]) & 0x3FFF)
			}
		case "VP8L":
			// Signature 0x2f, then width-1 and height-1 in 14 bits each.
			if s.Width == 0 && len(data) >= 5 && data[0] == 0x2F {
				v := binary.LittleEndian.Uint32(data[1:])
				s.Width = int(v&0x3FFF) + 1
				s.Height = int(v>>14&0x3FFF) + 1
			}
		case "EXIF":
			if e, err := ParseTIFF(bytes.TrimPrefix(data, exifPrefix)); err == nil {
				s.Exif, s.HasExif = e, true
			}
		case "C2PA":
			if c2paGenerated(data) {
				s.C2PAGenerated = true
			}
		}
		return true
	})
}

// c2paGenerated reports whether JUMBF bytes hold a C2PA manifest that names
// the trainedAlgorithmicMedia source type.
func c2paGenerated(b []byte) bool {
	return len(b) > 0 && bytes.Contains(b, []byte("c2pa")) && bytes.Contains(b, c2paAI)
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
