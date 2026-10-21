package jobfiles

import (
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/httpx"
)

// isMedia matches upstream's suffix list exactly, including "mp3" without
// the dot.
func isMedia(name string) bool {
	for _, s := range []string{".png", ".jpg", ".jpeg", ".webp", ".mp4", "mp3"} {
		if strings.HasSuffix(name, s) {
			return true
		}
	}
	return false
}

// samples lists the job's sample files: media, plus .txt files that are not
// the prompt sidecar of a media sample (text samples from LLM models). Only
// regular files count, so the .thumbs/.tmp folders stay out. Mirrors
// jobs/[jobID]/samples/route.ts.
func samples(w http.ResponseWriter, r *http.Request) {
	folder, ok := jobFolder(w, r)
	if !ok {
		return
	}
	samplesFolder := filepath.Join(folder, "samples")
	if _, err := os.Stat(samplesFolder); err != nil {
		httpx.OK(w, map[string]any{"samples": []string{}})
		return
	}
	entries, err := os.ReadDir(samplesFolder)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	var names []string
	for _, e := range entries {
		if e.Type().IsRegular() {
			names = append(names, e.Name())
		}
	}
	stems := map[string]bool{}
	for _, n := range names {
		if isMedia(n) {
			stems[jsSlice(n, lastIndexOf(n, "."))] = true
		}
	}
	out := []string{}
	for _, n := range names {
		if isMedia(n) || (strings.HasSuffix(n, ".txt") && !stems[jsSlice(n, -4)]) {
			out = append(out, filepath.Join(samplesFolder, n))
		}
	}
	slices.SortFunc(out, cmpJS)
	for i := range out {
		out[i] = decodeUTF8([]byte(out[i]))
	}
	httpx.OK(w, map[string]any{"samples": out})
}
