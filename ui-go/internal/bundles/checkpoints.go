package bundles

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

// How ai-toolkit resumes (jobs/process/BaseSDTrainProcess.py):
// get_latest_save_path globs <training_folder>/<config.name>/<name>*.safetensors,
// <name>*.pt and <name>* (diffusers saves are folders) and takes the one with
// the newest ctime; load_training_state_from_metadata then reads the step
// from its safetensors metadata (training_info.step; aitk_meta.yaml in a
// diffusers folder), and optimizer.pt next to it is loaded when present.
// Saves are named <name>_<9-digit step> (<name> alone for the final one).

// cacheDirs are the folders ai-toolkit caches into next to dataset images
// (toolkit/dataloader_mixins.py): latents, text embeddings, CLIP vision.
var cacheDirs = map[string]bool{"_latent_cache": true, "_t_e_cache": true, "_clip_vision_cache": true}

// IsCacheDir reports whether a dataset folder name is one of ai-toolkit's caches.
func IsCacheDir(name string) bool { return cacheDirs[name] }

// inCache reports whether a dataset-relative path lies in a cache folder.
func inCache(rel string) bool {
	for _, seg := range strings.Split(rel, "/") {
		if cacheDirs[seg] {
			return true
		}
	}
	return false
}

var stepRe = regexp.MustCompile(`^_(\d+)$`)

// FinalStep sorts the final, step-less save after every stepped one.
const FinalStep = math.MaxInt

// checkpointOf says whether a job folder entry (top-level name) is one of
// the job's saves and its filename step (FinalStep for <name> itself).
// Model files are .safetensors/.ckpt/.pt (not optimizer.pt); folders are
// diffusers saves.
func checkpointOf(name, job string, dir bool) (int, bool) {
	base := name
	if !dir {
		ext := strings.ToLower(path.Ext(name))
		if (ext != ".safetensors" && ext != ".ckpt" && ext != ".pt") || name == "optimizer.pt" {
			return 0, false
		}
		base = strings.TrimSuffix(name, path.Ext(name))
	}
	if base == job {
		return FinalStep, true
	}
	rest, ok := strings.CutPrefix(base, job)
	if !ok {
		return 0, false
	}
	if m := stepRe.FindStringSubmatch(rest); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil {
			return n, true
		}
	}
	return 0, false
}

// metaStep reads training_info.step from a safetensors header.
func metaStep(r io.Reader) (int, bool) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil || n == 0 || n > 64<<20 {
		return 0, false
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return 0, false
	}
	var head struct {
		Meta map[string]string `json:"__metadata__"`
	}
	if json.Unmarshal(buf, &head) != nil {
		return 0, false
	}
	var info struct {
		Step *int `json:"step"`
	}
	if json.Unmarshal([]byte(head.Meta["training_info"]), &info) != nil || info.Step == nil {
		return 0, false
	}
	return *info.Step, true
}

// CheckpointStep is the step ai-toolkit resumes at from a save: its
// metadata's training_info.step (a safetensors file, or aitk_meta.yaml in a
// diffusers folder).
func CheckpointStep(p string) (int, bool) {
	st, err := os.Stat(p)
	if err != nil {
		return 0, false
	}
	if st.IsDir() {
		b, err := os.ReadFile(filepath.Join(p, "aitk_meta.yaml"))
		if err != nil {
			return 0, false
		}
		var meta struct {
			TrainingInfo struct {
				Step *int `yaml:"step"`
			} `yaml:"training_info"`
		}
		if yaml.Unmarshal(b, &meta) != nil || meta.TrainingInfo.Step == nil {
			return 0, false
		}
		return *meta.TrainingInfo.Step, true
	}
	if !strings.EqualFold(path.Ext(p), ".safetensors") {
		return 0, false
	}
	f, err := os.Open(p)
	if err != nil {
		return 0, false
	}
	defer f.Close()
	return metaStep(f)
}
