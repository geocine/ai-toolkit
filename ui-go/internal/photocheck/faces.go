package photocheck

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/photocheck/pyjob"
	"github.com/geocine/aitoolkit/internal/state"
)

// The face check job (section 12): the photos it reads, the config the
// worker hands photo_faces.py and the models it fetches. The script and the
// model pins live in pyjob; ingest.go reads the job's faces.json back.

// This server can run the face check: reviews say faces.available.
func init() { FacesAvailable = true }

// FaceThreads is the face job's cv2 thread count: the server shares the
// box with training (section 17).
const FaceThreads = 2

// FileRef is one photo a face job reads: dataset-relative, with "/".
type FileRef struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
}

// faceModelKeys are the models a face check fetches before its first photo:
// the detector and the recognizer, plus the text detector when text is on.
// yunet5 is fetched only on OpenCV 5.x when yunet4 does not load.
func faceModelKeys(text bool) []string {
	if text {
		return []string{"yunet4", "sface", "ppocr"}
	}
	return []string{"yunet4", "sface"}
}

// FaceModelsDir is where face checks keep their models: <models>/opencv_zoo,
// with models the server's models path (settings.ModelsPath).
func FaceModelsDir(modelsPath string) string {
	return filepath.Join(modelsPath, pyjob.ModelsSubdir)
}

// FaceDownload is how many bytes a face check still has to fetch into
// modelsDir before its first photo. A file of the pinned size counts as
// there; the job itself checks the sha256 and fetches it again when it
// differs.
func FaceDownload(modelsDir string, text bool) int64 {
	var n int64
	for _, key := range faceModelKeys(text) {
		m := pyjob.Models[key]
		if st, err := os.Stat(filepath.Join(modelsDir, m.File())); err != nil || st.Size() != m.Bytes {
			n += m.Bytes
		}
	}
	return n
}

// FaceJobName names a face check of the dataset started at t, like the
// caption jobs: <dataset>_faces_<UTC yyyymmdd_hhmmss>.
func FaceJobName(dataset string, t time.Time) string {
	return dataset + "_faces_" + t.UTC().Format("20060102_150405")
}

// FaceFiles lists the photos a face check of dir reads: the photos a review
// lists (what the trainer reads, plus the formats it skips; set-aside ones
// are left out), each content once, in walk order, so the stored set
// signature matches the review's and a fresh check is not stale.
// convertible are the listed photos in formats the trainer skips: the job
// tries them and records an error for one it cannot read. No photos, or
// over MaxFaceJob, is a *UserError (a 400).
func FaceFiles(ctx context.Context, dir string) (files []FileRef, convertible []string, err error) {
	dir = filepath.Clean(dir)
	items, _, err := walkPhotos(dir)
	if err != nil {
		return nil, nil, err
	}
	if len(items) == 0 {
		return nil, nil, badRequest("This dataset has no photos to check.")
	}
	if len(items) > MaxFaceJob {
		return nil, nil, badRequest("This dataset has %d photos; the face check takes at most %d. Split the dataset.", len(items), MaxFaceJob)
	}
	seen := map[string]bool{}
	files = make([]FileRef, 0, len(items))
	for _, e := range items {
		sha, err := state.FileSHA256(ctx, e.abs, e.info)
		if isNotExist(err) {
			continue // removed since the walk
		}
		if err != nil {
			return nil, nil, err
		}
		if seen[sha] {
			continue // the job measures each content once
		}
		seen[sha] = true
		files = append(files, FileRef{File: e.rel, SHA256: sha})
		if !TrainerExts[strings.ToLower(path.Ext(e.rel))] {
			convertible = append(convertible, e.rel)
		}
	}
	if len(files) == 0 {
		return nil, nil, badRequest("This dataset has no photos to check.")
	}
	return files, convertible, nil
}

// faceProcess is config.process[0] of a face check (section 12.2). The
// worker adds sqlite_db_path when it writes .job_config.json.
type faceProcess struct {
	Type       string         `json:"type"`
	Algo       int            `json:"algo"`
	DatasetDir string         `json:"dataset_dir"`
	Files      []FileRef      `json:"files"`
	Text       bool           `json:"text"`
	Threads    int            `json:"threads"`
	Out        string         `json:"out"`
	Models     map[string]any `json:"models"`
}

// BuildFaceJobConfig is the job_config of a face check named name over
// files of the dataset folder dir: {"job": "extension", "config": {"name",
// "process": [...]}} with the model pins and modelsDir (section 12.2).
// With text, the job also looks for text and watermarks.
func BuildFaceJobConfig(dir, name string, files []FileRef, text bool, modelsDir string) (json.RawMessage, error) {
	if name == "" {
		return nil, fmt.Errorf("face check: no job name")
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("face check: no photos")
	}
	if len(files) > MaxFaceJob {
		return nil, fmt.Errorf("face check: %d photos, at most %d", len(files), MaxFaceJob)
	}
	models := map[string]any{"dir": modelsDir}
	for key, m := range pyjob.Models {
		models[key] = m
	}
	p := faceProcess{
		Type:       pyjob.ProcessType,
		Algo:       FaceAlgo,
		DatasetDir: filepath.Clean(dir),
		Files:      files,
		Text:       text,
		Threads:    FaceThreads,
		Out:        FaceResultFile,
		Models:     models,
	}
	cfg := map[string]any{
		"job":    "extension",
		"config": map[string]any{"name": name, "process": []faceProcess{p}},
	}
	b, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	return b, nil
}
