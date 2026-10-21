package jobfiles

import (
	"errors"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/geocine/aitoolkit/internal/httpx"
)

// maxTailBytes caps the initial payload. The client renders the log through
// a terminal emulator that collapses \r rewrites (progress bars), so send a
// generous byte tail and let the emulator trim.
const maxTailBytes = 5 * 1024 * 1024

type logResponse struct {
	Log    string `json:"log"`
	Offset int64  `json:"offset"`
	Reset  bool   `json:"reset"`
}

// logTail returns the job log incrementally. The client sends the byte
// offset it already consumed; without one (or past the end, i.e. the log was
// truncated by a fresh run) it gets a fresh tail. Mirrors
// jobs/[jobID]/log/route.ts.
func logTail(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}
	logPath := filepath.Join(folder, "log.txt")
	if _, err := os.Stat(logPath); err != nil {
		httpx.OK(w, logResponse{Log: "", Offset: 0, Reset: true})
		return
	}

	offset := math.NaN()
	if r.URL.Query().Has("offset") {
		offset = jsParseInt(r.URL.Query().Get("offset"))
	}

	resp, err := readLog(logPath, offset)
	if err != nil {
		log.Printf("Error reading log file: %v", err)
		httpx.OK(w, logResponse{Log: "Error reading log file", Offset: 0, Reset: true})
		return
	}
	httpx.OK(w, resp)
}

func readLog(logPath string, offset float64) (logResponse, error) {
	st, err := os.Stat(logPath)
	if err != nil {
		return logResponse{}, err
	}
	size := st.Size()
	f, err := os.Open(logPath)
	if err != nil {
		return logResponse{}, err
	}
	defer f.Close()

	if math.IsNaN(offset) || offset > float64(size) {
		start := max(0, size-maxTailBytes)
		text, err := readRange(f, start, size)
		if err != nil {
			return logResponse{}, err
		}
		// drop a partial first line if we started mid-file
		if start > 0 {
			if i := strings.IndexByte(text, '\n'); i != -1 {
				text = text[i+1:]
			}
		}
		return logResponse{Log: text, Offset: size, Reset: true}, nil
	}

	// incremental: only the bytes appended since the last offset
	if offset < -1 {
		// FileHandle.read rejects positions below -1 (ERR_OUT_OF_RANGE)
		return logResponse{}, errors.New("offset out of range")
	}
	if offset == -1 {
		// position -1 means "current position": a fresh handle reads from 0
		// into a buffer one byte longer than the file, leaving a trailing NUL
		text, err := readRange(f, 0, size)
		if err != nil {
			return logResponse{}, err
		}
		return logResponse{Log: text + "\x00", Offset: size, Reset: false}, nil
	}
	text, err := readRange(f, int64(offset), size)
	if err != nil {
		return logResponse{}, err
	}
	return logResponse{Log: text, Offset: size, Reset: false}, nil
}

func readRange(f *os.File, start, end int64) (string, error) {
	n := end - start
	if n <= 0 {
		return "", nil
	}
	buf := make([]byte, n)
	read, err := f.ReadAt(buf, start)
	if err != nil && read == 0 {
		return "", err
	}
	// a short read leaves the rest of the buffer zeroed, like Buffer.alloc
	return decodeUTF8(buf), nil
}
