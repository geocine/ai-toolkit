package media

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"

	"github.com/geocine/aitoolkit/internal/collation"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/safepath"
)

// POST /api/files/info mirrors ui/src/app/api/files/info/route.ts: the
// header of a .safetensors file under the datasets or training folder
// (metadata, every tensor's dtype/shape/size, totals per dtype), for the
// file browser's info button. Only the header is read.

// safetensors header: u64 LE byte length, then that many bytes of JSON.
const maxSafetensorsHeader = 256 * 1024 * 1024

var dtypeBytes = map[string]float64{
	"F64": 8, "I64": 8, "U64": 8,
	"F32": 4, "I32": 4, "U32": 4,
	"F16": 2, "BF16": 2, "I16": 2, "U16": 2,
	"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1,
	"F4": 0.5, "U4": 0.5, "I4": 0.5,
}

type tensorInfo struct {
	Name  string    `json:"name"`
	DType any       `json:"dtype"`
	Shape []float64 `json:"shape"`
	Numel float64   `json:"numel"`
	Bytes float64   `json:"bytes"`
}

type dtypeCount struct {
	Count float64 `json:"count"`
	Numel float64 `json:"numel"`
	Bytes float64 `json:"bytes"`
}

func filesInfo(w http.ResponseWriter, r *http.Request) {
	internal := func(err error) {
		log.Printf("Error reading safetensors info: %v", err)
		textPlain(w, http.StatusInternalServerError, "Internal Server Error")
	}
	body, isNull, err := readBody(r)
	if err != nil || isNull {
		internal(errors.Join(err, errors.New("invalid body")))
		return
	}
	filePath, ok := str(body, "filePath")
	if !ok || filePath == "" {
		textPlain(w, http.StatusBadRequest, "filePath is required")
		return
	}
	decoded, err := decodeURIComponent(filePath)
	if err != nil {
		internal(err)
		return
	}
	rt, err := roots(r.Context())
	if err != nil {
		internal(err)
		return
	}
	allowed := []string{rt.Datasets, rt.Training}
	resolved := safepath.Resolve(decoded, allowed)
	if resolved == "" {
		log.Printf("Access denied: %s not in %s", decoded, strings.Join(allowed, ", "))
		textPlain(w, http.StatusForbidden, "Access denied")
		return
	}
	st, err := os.Stat(resolved)
	if err != nil {
		textPlain(w, http.StatusNotFound, "File not found")
		return
	}
	if !st.Mode().IsRegular() {
		textPlain(w, http.StatusBadRequest, "Not a file")
		return
	}
	if st.Size() < 8 {
		textPlain(w, http.StatusBadRequest, "Not a safetensors file")
		return
	}

	header, headerSize, status, msg := readSafetensorsHeader(resolved, st.Size())
	if status != 0 {
		textPlain(w, status, msg)
		return
	}

	metadata, ok := header.Get("__metadata__")
	if !ok || metadata == nil {
		metadata = jsonx.NewObject()
	}
	tensors := []tensorInfo{}
	counts := jsonx.NewObject() // insertion order, like the JS object
	var totalParams, totalBytes float64
	for _, name := range header.Keys() {
		if name == "__metadata__" {
			continue
		}
		v, _ := header.Get(name)
		t, _ := v.(*jsonx.Object)
		var dtype any
		shape := []float64{}
		var offsets []any
		if t != nil {
			dtype, _ = t.Get("dtype")
			if s, ok := t.Get("shape"); ok {
				if arr, ok := s.([]any); ok {
					for _, d := range arr {
						shape = append(shape, num(d))
					}
				}
			}
			if o, ok := t.Get("data_offsets"); ok {
				offsets, _ = o.([]any)
			}
		}
		numel := 1.0
		for _, d := range shape {
			numel *= d
		}
		ds, _ := dtype.(string)
		var bytes float64
		if len(offsets) >= 2 {
			bytes = num(offsets[1]) - num(offsets[0])
		} else {
			bytes = numel * dtypeBytes[ds]
		}
		tensors = append(tensors, tensorInfo{Name: name, DType: dtype, Shape: shape, Numel: numel, Bytes: bytes})
		totalParams += numel
		totalBytes += bytes
		c, _ := counts.Get(ds)
		dc, _ := c.(*dtypeCount)
		if dc == nil {
			dc = &dtypeCount{}
			counts.Set(ds, dc)
		}
		dc.Count++
		dc.Numel += numel
		dc.Bytes += bytes
	}
	sort.SliceStable(tensors, func(i, j int) bool { return collation.Compare(tensors[i].Name, tensors[j].Name) < 0 })

	out := jsonx.NewObject()
	out.Set("filePath", resolved)
	out.Set("fileSize", st.Size())
	out.Set("headerSize", headerSize)
	out.Set("metadata", metadata)
	out.Set("tensors", tensors)
	out.Set("tensorCount", len(tensors))
	out.Set("totalParams", totalParams)
	out.Set("totalBytes", totalBytes)
	out.Set("dtypeCounts", counts)
	httpx.OK(w, out)
}

// readSafetensorsHeader reads and parses the JSON header. A non-zero status
// is upstream's answer for a file that is not safetensors.
func readSafetensorsHeader(path string, size int64) (header *jsonx.Object, headerSize int64, status int, msg string) {
	const unparseable = "Not a safetensors file (unparseable header)"
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, http.StatusBadRequest, unparseable
	}
	defer f.Close()
	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, 0, http.StatusBadRequest, unparseable
	}
	n := binary.LittleEndian.Uint64(lenBuf[:])
	if n > maxSafetensorsHeader || n > uint64(size-8) {
		return nil, 0, http.StatusBadRequest, "Not a safetensors file (bad header length)"
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, 0, http.StatusBadRequest, unparseable
	}
	h, err := jsonx.ParseObject(buf)
	if err != nil {
		return nil, 0, http.StatusBadRequest, unparseable
	}
	return h, int64(n), 0, ""
}

// num is a JSON number as JavaScript sees it (0 for anything else).
func num(v any) float64 {
	switch n := v.(type) {
	case json.Number:
		f, _ := n.Float64()
		return f
	case float64:
		return n
	}
	return 0
}
