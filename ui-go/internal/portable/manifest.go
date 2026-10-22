// Package portable keeps jobs continuable on another server: a job whose
// policy has sync_remote pushes its resumable state (datasets, newest
// checkpoint, optimizer state, loss log, log) to that remote with a manifest
// and a lease, and any server can import it and resume. See docs/headless.md,
// "Portable jobs".
package portable

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/state"
)

// Schema is the manifest version this server writes and reads.
const Schema = 1

// LeaseTTL is how long a lease holds without renewal; renewals run every
// LeaseRenew while the job is active here.
const (
	LeaseTTL   = 5 * time.Minute
	LeaseRenew = time.Minute
)

// File is one file of a job's portable state. Key is its remote key; files
// live under a content-addressed key (files/<sha16>/<name>) so the manifest
// is the only commit point: a manifest never points at a half-overwritten
// optimizer.pt.
type File struct {
	Name   string `json:"name"`
	Key    string `json:"key"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
	Step   *int   `json:"step"`
}

// Lease says which server is running (or about to run) the job.
type Lease struct {
	NodeID      string    `json:"node_id"`
	NodeLabel   string    `json:"node_label"`
	HeartbeatAt time.Time `json:"heartbeat_at"`
	ExpiresAt   time.Time `json:"expires_at"`
}

// Held reports whether the lease belongs to another node and is still valid.
func (l *Lease) Held(nodeID string, now time.Time) bool {
	return l != nil && l.NodeID != nodeID && now.Before(l.ExpiresAt)
}

// Manifest is jobs/<name>/manifest.json.
type Manifest struct {
	Schema int    `json:"schema"`
	Name   string `json:"name"`
	// RunID names the run across servers: the id of the job that first
	// synced it, kept by every import. A job with the same name but another
	// run id must never write over it.
	RunID      string           `json:"run_id,omitempty"`
	Type       string           `json:"type"`
	Ref        *string          `json:"ref"`
	GpuIDs     string           `json:"gpu_ids"`
	Config     json.RawMessage  `json:"config"`
	Policy     state.Policy     `json:"policy"`
	Datasets   []string         `json:"datasets"`
	Step       int              `json:"step"`
	TotalSteps *int             `json:"total_steps"`
	State      string           `json:"state"`
	Files      []File           `json:"files"`
	Samples    []File           `json:"samples"`
	Attempts   []*state.Attempt `json:"attempts"`
	Lease      *Lease           `json:"lease"`
	UpdatedAt  time.Time        `json:"updated_at"`
}

// DatasetManifest is datasets/<name>/manifest.json.
type DatasetManifest struct {
	Name  string        `json:"name"`
	Files []DatasetFile `json:"files"`
}

// DatasetFile is one dataset file; its content is blobs/<sha256>.
type DatasetFile struct {
	Path   string `json:"path"` // slash-separated, relative to the dataset folder
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

func jobKey(name, rest string) string { return "jobs/" + name + "/" + rest }
func manifestKey(name string) string  { return jobKey(name, "manifest.json") }
func datasetKey(name string) string   { return "datasets/" + name + "/manifest.json" }
func blobKey(sha string) string       { return "blobs/" + sha }
func fileKey(name, sha, file string) string {
	return jobKey(name, "files/"+sha[:16]+"/"+file)
}

// ValidName rejects names that would escape their remote folder.
func ValidName(name string) error {
	if name == "" || name == "." || name == ".." || strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") {
		return fmt.Errorf("%q cannot be used as a portable name", name)
	}
	return nil
}

// readManifest returns the job's manifest, or nil when the remote has none.
func readManifest(ctx context.Context, c remote.Client, name string) (*Manifest, error) {
	b, err := c.GetBytes(ctx, manifestKey(name))
	if errors.Is(err, remote.ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("manifest of %s: %w", name, err)
	}
	if m.Schema > Schema {
		return nil, fmt.Errorf("manifest of %s is schema %d; this server reads up to %d", name, m.Schema, Schema)
	}
	return &m, nil
}

func writeManifest(ctx context.Context, c remote.Client, m *Manifest) error {
	m.Schema = Schema
	m.UpdatedAt = time.Now().UTC().Truncate(time.Millisecond)
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return c.PutBytes(ctx, manifestKey(m.Name), b)
}

// nodeLabel is a human hint for which machine holds a lease.
func nodeLabel() string {
	host, _ := os.Hostname()
	for _, env := range []string{"RUNPOD_POD_ID", "VAST_CONTAINERLABEL", "HOSTNAME_LABEL"} {
		if v := os.Getenv(env); v != "" {
			return host + " (" + v + ")"
		}
	}
	return host
}

func runKey(jobID string) string { return "portable.run." + jobID }

// RunID is the portable run the local job continues: the id it was imported
// with, else its own id.
func RunID(ctx context.Context, jobID string) string {
	if v, ok, _ := state.Setting(ctx, runKey(jobID)); ok && v != "" {
		return v
	}
	return jobID
}

func setRunID(ctx context.Context, jobID, runID string) error {
	if runID == "" || runID == jobID {
		return nil
	}
	return state.SetSetting(ctx, runKey(jobID), runID)
}

// ErrOtherRun: the remote holds a different run under the job's name.
var ErrOtherRun = errors.New("the remote holds a different run with this name")

// newestStep is the highest checkpoint step among files (-1 for none).
func newestStep(files []File) int {
	n := -1
	for _, f := range files {
		if f.Step != nil && *f.Step > n {
			n = *f.Step
		}
	}
	return n
}

func newLease(ctx context.Context) (*Lease, error) {
	id, err := state.NodeID(ctx)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC().Truncate(time.Millisecond)
	return &Lease{NodeID: id, NodeLabel: nodeLabel(), HeartbeatAt: now, ExpiresAt: now.Add(LeaseTTL)}, nil
}
