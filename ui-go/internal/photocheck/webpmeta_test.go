package photocheck

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/geocine/aitoolkit/internal/imgmeta"
	"github.com/geocine/aitoolkit/internal/testutil"
)

// webpWithExif is SceneWebP (128x96) in a VP8X container with an EXIF
// chunk of orientation o after the image data, and pad bytes of another
// chunk between them.
func webpWithExif(t *testing.T, o, pad int) []byte {
	t.Helper()
	src := testutil.SceneWebP()
	if string(src[12:16]) != "VP8 " {
		t.Fatalf("SceneWebP starts with %q", src[12:16])
	}
	chunk := func(b *bytes.Buffer, fourcc string, data []byte) {
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], uint32(len(data)))
		b.WriteString(fourcc)
		b.Write(n[:])
		b.Write(data)
		if len(data)%2 == 1 {
			b.WriteByte(0)
		}
	}
	var body bytes.Buffer
	body.WriteString("WEBP")
	vp8x := make([]byte, 10)
	vp8x[0] = 0x08 // EXIF
	vp8x[4], vp8x[7] = 127, 95
	chunk(&body, "VP8X", vp8x)
	body.Write(src[12:]) // the VP8 chunk, header included
	if pad > 0 {
		chunk(&body, "ZZZZ", make([]byte, pad))
	}
	chunk(&body, "EXIF", imgmeta.EncodeTIFF(imgmeta.Exif{Orientation: o}, binary.LittleEndian))
	var out bytes.Buffer
	out.WriteString("RIFF")
	var n [4]byte
	binary.LittleEndian.PutUint32(n[:], uint32(body.Len()))
	out.Write(n[:])
	out.Write(body.Bytes())
	return out.Bytes()
}

// A WebP's EXIF chunk comes after the image data, so in a big one it is
// past the 256 KB head; Measure still honours its orientation.
func TestMeasureWebPExifPastHead(t *testing.T) {
	for _, tc := range []struct {
		name string
		pad  int
	}{{"small", 0}, {"big", imgmeta.HeadSize + 64<<10}} {
		data := webpWithExif(t, 6, tc.pad)
		if tc.pad > 0 && len(data) <= imgmeta.HeadSize {
			t.Fatalf("%s: %d bytes is not past the head", tc.name, len(data))
		}
		m := measureBytes(t, tc.name+".webp", data)
		if !m.Decoded || m.Orientation != 6 || m.Width != 96 || m.Height != 128 {
			t.Errorf("%s: decoded %v orientation %d, %dx%d; want 6, 96x128", tc.name, m.Decoded, m.Orientation, m.Width, m.Height)
		}
	}
}
