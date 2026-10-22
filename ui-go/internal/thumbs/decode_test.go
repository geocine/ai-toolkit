package thumbs

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/png"
	"testing"
)

// a PNG whose header claims 60000x60000 (a decompression bomb) is refused
// before any pixel is allocated; a normal one decodes
func TestDecodeRefusesHugeCanvas(t *testing.T) {
	var buf bytes.Buffer
	png.Encode(&buf, image.NewGray(image.Rect(0, 0, 2, 2)))
	small := buf.Bytes()
	if img, err := Decode(bytes.NewReader(small)); err != nil || img.Bounds().Dx() != 2 {
		t.Fatalf("small: %v", err)
	}
	bomb := append([]byte(nil), small...)
	// IHDR data starts at 16: width, height (big endian); then fix the chunk CRC
	binary.BigEndian.PutUint32(bomb[16:], 60000)
	binary.BigEndian.PutUint32(bomb[20:], 60000)
	binary.BigEndian.PutUint32(bomb[29:], crc32.ChecksumIEEE(bomb[12:29]))
	if _, err := Decode(bytes.NewReader(bomb)); err == nil {
		t.Fatal("60000x60000 decoded")
	}
}
