package media

import (
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
)

// serveAlbumArt is the Next route ui/src/app/api/audio/art/[...audioPath]/route.ts,
// which the front server falls back to when no thumb exists or can be made:
// embedded album art from the file's ID3v2 tag, reading only the tag.
func serveAlbumArt(w http.ResponseWriter, r *http.Request, resolved string) {
	textHeader := func() { w.Header().Set("Content-Type", "text/plain;charset=UTF-8") }
	st, err := os.Stat(resolved)
	if err != nil || !st.Mode().IsRegular() {
		textHeader()
		plain(w, http.StatusNotFound, "File not found")
		return
	}

	// a thumb written by the sampler or a previous call: tags read at most once per file
	thumbPath := filepath.Join(filepath.Dir(resolved), ".thumbs", filepath.Base(resolved)+".jpg")
	if ts, err := os.Stat(thumbPath); err == nil && ts.Mode().IsRegular() {
		data, err := os.ReadFile(thumbPath)
		if err != nil {
			artError(w, err)
			return
		}
		writeArt(w, r, "image/jpeg", data)
		return
	}

	f, err := os.Open(resolved)
	if err != nil {
		artError(w, err)
		return
	}
	defer f.Close()
	header := make([]byte, 10)
	f.ReadAt(header, 0)
	if header[0] != 0x49 || header[1] != 0x44 || header[2] != 0x33 {
		textHeader()
		plain(w, http.StatusNotFound, "No ID3 tag")
		return
	}
	tagSize := synchsafeToInt(header[6], header[7], header[8], header[9])
	tag := make([]byte, min(10+tagSize, 4_000_000))
	if _, err := f.ReadAt(tag, 0); err != nil && err != io.EOF {
		artError(w, err)
		return
	}
	mime, data, ok := extractArtFromTag(tag)
	if !ok {
		textHeader()
		plain(w, http.StatusNotFound, "No album art found")
		return
	}
	if mime == "image/jpeg" {
		// cache as the file's thumb; best effort
		if err := os.MkdirAll(filepath.Dir(thumbPath), 0o755); err == nil {
			err = os.WriteFile(thumbPath, data, 0o644)
			if err != nil {
				log.Printf("album art thumb not written: %v", err)
			}
		} else {
			log.Printf("album art thumb not written: %v", err)
		}
	}
	writeArt(w, r, mime, data)
}

func writeArt(w http.ResponseWriter, r *http.Request, mime string, data []byte) {
	h := w.Header()
	h.Set("Content-Type", mime)
	h.Set("Content-Length", strconv.Itoa(len(data)))
	h.Set("Cache-Control", "public, max-age=604800, immutable")
	w.WriteHeader(http.StatusOK)
	if r.Method != http.MethodHead {
		w.Write(data)
	}
}

func artError(w http.ResponseWriter, err error) {
	log.Printf("Error extracting album art: %v", err)
	w.Header().Set("Content-Type", "text/plain;charset=UTF-8")
	plain(w, http.StatusInternalServerError, "Internal Server Error")
}

func synchsafeToInt(b0, b1, b2, b3 byte) int {
	return int(b0&0x7f)<<21 | int(b1&0x7f)<<14 | int(b2&0x7f)<<7 | int(b3&0x7f)
}

func deUnsync(b []byte) []byte {
	out := make([]byte, 0, len(b))
	for i := 0; i < len(b); i++ {
		out = append(out, b[i])
		if b[i] == 0xff && i+1 < len(b) && b[i+1] == 0x00 {
			i++
		}
	}
	return out
}

// readNullTerminated returns the latin1 text (unused for wide strings) and
// the offset after the terminator.
func readNullTerminated(buf []byte, start int, wide bool) (string, int) {
	if wide {
		i := start
		for i+1 < len(buf) && !(buf[i] == 0 && buf[i+1] == 0) {
			i += 2
		}
		return "", i + 2
	}
	i := start
	for i < len(buf) && buf[i] != 0 {
		i++
	}
	end := min(i, len(buf))
	if start > end {
		return "", i + 1
	}
	runes := make([]rune, 0, end-start)
	for _, c := range buf[start:end] {
		runes = append(runes, rune(c))
	}
	return string(runes), i + 1
}

func slice(b []byte, from, to int) []byte {
	from = max(0, min(from, len(b)))
	to = max(from, min(to, len(b)))
	return b[from:to]
}

// extractArtFromTag is the upstream ID3v2.2/2.3/2.4 PIC/APIC reader.
func extractArtFromTag(buf []byte) (string, []byte, bool) {
	if len(buf) < 10 || buf[0] != 0x49 || buf[1] != 0x44 || buf[2] != 0x33 {
		return "", nil, false
	}
	verMajor := buf[3]
	flags := buf[5]
	tagSize := synchsafeToInt(buf[6], buf[7], buf[8], buf[9])
	tagEnd := min(10+tagSize, len(buf))
	tagData := slice(buf, 10, tagEnd)
	if flags&0x80 != 0 {
		tagData = deUnsync(tagData)
	}
	offset := 0
	// skip extended header
	if (verMajor == 3 || verMajor == 4) && flags&0x40 != 0 && len(tagData) >= 4 {
		var extSize int
		if verMajor == 4 {
			extSize = synchsafeToInt(tagData[0], tagData[1], tagData[2], tagData[3])
		} else {
			extSize = int(int32(uint32(tagData[0])<<24 | uint32(tagData[1])<<16 | uint32(tagData[2])<<8 | uint32(tagData[3])))
		}
		offset += 4 + max(0, extSize)
	}
	for offset < len(tagData) {
		if tagData[offset] == 0x00 {
			break
		}
		if verMajor == 2 {
			// ID3v2.2: 3-byte frame ID, 3-byte size
			if offset+6 > len(tagData) {
				break
			}
			id := string(tagData[offset : offset+3])
			size := int(tagData[offset+3])<<16 | int(tagData[offset+4])<<8 | int(tagData[offset+5])
			offset += 6
			if blank(id) || size <= 0 || offset+size > len(tagData) {
				break
			}
			if id == "PIC" && size > 6 {
				frame := tagData[offset : offset+size]
				fmtName := lower(string(slice(frame, 1, 4)))
				mime := "image/jpeg"
				if fmtName == "png" {
					mime = "image/png"
				}
				enc := frame[0]
				wide := enc == 1 || enc == 2
				_, p := readNullTerminated(frame, 5, wide)
				if p < len(frame) {
					if img := frame[p:]; len(img) > 64 {
						return mime, append([]byte(nil), img...), true
					}
				}
			}
			offset += size
			continue
		}
		// ID3v2.3/v2.4: 4-byte frame ID, 4-byte size, 2-byte flags
		if offset+10 > len(tagData) {
			break
		}
		id := string(tagData[offset : offset+4])
		var size int
		if verMajor == 4 {
			size = synchsafeToInt(tagData[offset+4], tagData[offset+5], tagData[offset+6], tagData[offset+7])
		} else {
			size = int(int32(uint32(tagData[offset+4])<<24 | uint32(tagData[offset+5])<<16 | uint32(tagData[offset+6])<<8 | uint32(tagData[offset+7])))
		}
		flag2 := tagData[offset+9]
		offset += 10
		if blank(id) || size <= 0 || offset+size > len(tagData) {
			break
		}
		if id == "APIC" {
			frame := tagData[offset : offset+size]
			if verMajor == 4 && flag2&0x02 != 0 {
				frame = deUnsync(frame)
			}
			enc := frame[0]
			mime, p := readNullTerminated(frame, 1, false)
			if mime == "" {
				mime = "image/jpeg"
			}
			if p < len(frame) {
				p++ // picture type byte
			}
			wide := enc == 1 || enc == 2
			_, p = readNullTerminated(frame, p, wide)
			if p < len(frame) {
				if img := frame[p:]; len(img) > 64 {
					return mime, append([]byte(nil), img...), true
				}
			}
		}
		offset += size
	}
	return "", nil, false
}

// blank is `!id.trim()` on a latin1 frame id.
func blank(id string) bool {
	for _, c := range id {
		switch c {
		case ' ', '\t', '\n', '\r', '\v', '\f', 0xa0:
		default:
			return false
		}
	}
	return true
}

func lower(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return string(b)
}
