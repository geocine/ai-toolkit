// Package artifacts finds what a training job writes into its folder
// (<training root>/<job name>): checkpoints and samples. Shared by the /v1
// jobs API, the event observer and auto-upload so they agree on what a
// checkpoint is.
package artifacts

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Checkpoint is a saved model file of a job.
type Checkpoint struct {
	Name     string    `json:"name"`
	Path     string    `json:"path"`
	Step     *int      `json:"step"` // nil for the final, step-less file
	Size     int64     `json:"size"`
	Modified time.Time `json:"modified"`
	SHA256   string    `json:"sha256,omitempty"`
}

// Sample is a generated sample image or video.
type Sample struct {
	Name     string    `json:"name"`
	Path     string    `json:"path"`
	Step     *int      `json:"step"`
	Index    *int      `json:"index"`
	Modified time.Time `json:"modified"`
}

var (
	// <name>_000000020.safetensors (ai-toolkit pads steps to 9 digits)
	stepSuffix = regexp.MustCompile(`_(\d+)\.(safetensors|pt|ckpt|bin)$`)
	// samples/<ms>__<step>_<index>.<ext>
	sampleName = regexp.MustCompile(`^\d+__(\d+)_(\d+)\.[A-Za-z0-9]+$`)
	modelExts  = map[string]bool{".safetensors": true, ".ckpt": true}
	sampleExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".mp4": true,
		".webm": true, ".gif": true, ".wav": true, ".mp3": true, ".flac": true}
)

// Checkpoints lists model files at the top of jobFolder (diffusers-format
// saves are folders; those count too when they hold safetensors), sorted by
// step with the final file last. Missing folder -> empty list.
func Checkpoints(jobFolder string) ([]Checkpoint, error) {
	entries, err := os.ReadDir(jobFolder)
	if os.IsNotExist(err) {
		return []Checkpoint{}, nil
	}
	if err != nil {
		return nil, err
	}
	out := []Checkpoint{}
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if e.IsDir() || !modelExts[strings.ToLower(filepath.Ext(name))] {
			continue
		}
		c := Checkpoint{Name: name, Path: filepath.Join(jobFolder, name), Size: info.Size(), Modified: info.ModTime().UTC()}
		if m := stepSuffix.FindStringSubmatch(name); m != nil {
			if n, err := strconv.Atoi(m[1]); err == nil {
				c.Step = &n
			}
		}
		out = append(out, c)
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i].Step, out[j].Step
		switch {
		case a == nil && b == nil:
			return out[i].Name < out[j].Name
		case a == nil:
			return false
		case b == nil:
			return true
		}
		return *a < *b
	})
	return out, nil
}

// Samples lists jobFolder/samples, sorted by step then index.
func Samples(jobFolder string) ([]Sample, error) {
	dir := filepath.Join(jobFolder, "samples")
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return []Sample{}, nil
	}
	if err != nil {
		return nil, err
	}
	out := []Sample{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || strings.HasPrefix(name, ".") || !sampleExts[strings.ToLower(filepath.Ext(name))] {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		s := Sample{Name: name, Path: filepath.Join(dir, name), Modified: info.ModTime().UTC()}
		if m := sampleName.FindStringSubmatch(name); m != nil {
			step, _ := strconv.Atoi(m[1])
			idx, _ := strconv.Atoi(m[2])
			s.Step, s.Index = &step, &idx
		}
		out = append(out, s)
	}
	sort.SliceStable(out, func(i, j int) bool {
		si, sj := intOr(out[i].Step, -1), intOr(out[j].Step, -1)
		if si != sj {
			return si < sj
		}
		if ii, ij := intOr(out[i].Index, -1), intOr(out[j].Index, -1); ii != ij {
			return ii < ij
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

func intOr(p *int, d int) int {
	if p == nil {
		return d
	}
	return *p
}

// SHA256File hashes a file.
func SHA256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
