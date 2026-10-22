// Package bundles packs a training run into one zip and reads it back
// (docs/headless.md, "Run bundles"): the job's dataset(s), its folder with
// checkpoints and optimizer state, and a manifest. Unzipped into a toolkit
// it is the layout ai-toolkit resumes from (<training>/<name>/<name>_<step>
// .safetensors + optimizer.pt), and it reads the hand-made zips that predate
// the manifest (data/ + output/<name>/ with config.yaml).
//
// The package is the format only; handlers/v1 serves it (download, import
// tasks, chunked uploads, the remote library).
package bundles

import (
	"encoding/json"
	"time"
)

// Format is the manifest version this server writes and reads.
const Format = 1

// ManifestName is the manifest's entry name at the zip root.
const ManifestName = "aitk_bundle.json"

// Limits for reading a bundle.
const (
	MaxBytes   = 1 << 40 // uncompressed total
	MaxEntries = 200000
)

// Manifest is aitk_bundle.json.
type Manifest struct {
	Format          int             `json:"format"`
	Name            string          `json:"name"`
	CreatedAt       time.Time       `json:"created_at"`
	ServerVersion   string          `json:"server_version"`
	UpstreamVersion *string         `json:"upstream_version"`
	Type            string          `json:"type"`
	Step            int             `json:"step"`
	TotalSteps      *int            `json:"total_steps"`
	Config          json.RawMessage `json:"config"` // dataset/training paths as aitk:// (internal/portable)
	Policy          Policy          `json:"policy"`
	Datasets        []DatasetRef    `json:"datasets"`
	JobDir          string          `json:"job_dir"`     // "output/<name>"
	Checkpoints     []string        `json:"checkpoints"` // saves in the job folder, oldest first (the last one is resumed from)
	Optimizer       bool            `json:"optimizer"`
	Samples         int             `json:"samples"`
	Caches          bool            `json:"caches"` // the datasets' ai-toolkit caches (_latent_cache, _t_e_cache, ...) are included
	BaseModel       string          `json:"base_model,omitempty"`
	Arch            string          `json:"arch,omitempty"`
	Warnings        []string        `json:"warnings"`
}

// Policy is the portable part of a job policy (remotes are per server).
type Policy struct {
	AutoResume     *bool `json:"auto_resume"`
	MaxAutoResumes *int  `json:"max_auto_resumes"`
}

// DatasetRef is one dataset in the bundle: its name on the exporting server
// and the zip folder holding it (data, data_1, ...).
type DatasetRef struct {
	Name     string `json:"name"`
	Dir      string `json:"dir"`
	Images   int    `json:"images"`
	Captions int    `json:"captions"`
}

// DatasetDir is the zip folder of the job's i-th dataset.
func DatasetDir(i int) string {
	if i == 0 {
		return "data"
	}
	return "data_" + itoa(i)
}

// Summary is the sidecar <id>.json next to <id>.zip in a remote's bundles/
// folder, so a library lists without opening zips.
type Summary struct {
	Format        int       `json:"format"`
	Name          string    `json:"name"`
	Step          int       `json:"step"`
	TotalSteps    *int      `json:"total_steps"`
	Size          int64     `json:"size"`
	SHA256        string    `json:"sha256,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
	Images        int       `json:"images"`
	Datasets      []string  `json:"datasets"`
	BaseModel     string    `json:"base_model,omitempty"`
	Arch          string    `json:"arch,omitempty"`
	Checkpoints   []string  `json:"checkpoints"`
	Optimizer     bool      `json:"optimizer"`
	ServerVersion string    `json:"server_version"`
}

// SummaryOf derives the sidecar from a manifest.
func SummaryOf(m *Manifest, size int64, sha string) Summary {
	s := Summary{Format: m.Format, Name: m.Name, Step: m.Step, TotalSteps: m.TotalSteps, Size: size, SHA256: sha,
		CreatedAt: m.CreatedAt, BaseModel: m.BaseModel, Arch: m.Arch, Checkpoints: m.Checkpoints,
		Optimizer: m.Optimizer, ServerVersion: m.ServerVersion, Datasets: []string{}}
	if s.Checkpoints == nil {
		s.Checkpoints = []string{}
	}
	for _, d := range m.Datasets {
		s.Images += d.Images
		s.Datasets = append(s.Datasets, d.Name)
	}
	return s
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}
