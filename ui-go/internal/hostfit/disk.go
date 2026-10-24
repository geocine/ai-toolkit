package hostfit

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/geocine/aitoolkit/internal/imagecheck"
)

const (
	// WorkGB is the room for outputs, checkpoints, optimizer state and the
	// latent/text caches. UNCONFIRMED tuning.
	WorkGB = 15.0
	// diskMargin and diskFloorGB shape DiskGB = ceil10((peak + model + work) x 1.10), at least 40.
	diskMargin  = 1.10
	diskFloorGB = 40
	// DefaultImagePeakGB is today's image while it pulls (6.57 GB compressed
	// plus 13.75 GB unpacked, IMAGE-PLAN §0.1): the budget when the image
	// cannot be read. Whether the daemon keeps the compressed blobs after
	// extraction is UNCONFIRMED, so both are counted.
	DefaultImagePeakGB = 20.3
	// imageHeadroomGB is what a disk needs beyond the image peak before the
	// image even starts (disk_image).
	imageHeadroomGB = 2
	// sizeTimeout bounds the model size lookup inside NeedFor.
	sizeTimeout = 10 * time.Second
)

// Measured sizes of today's Go-server image (the fork base plus the Go
// layers), used when its registry cannot be read.
const (
	defaultDownloadBytes = 6_567_282_014
	defaultUnpackedBytes = 13_750_901_627
)

// ImagePeakGB is the disk the image takes while it pulls: compressed plus
// unpacked bytes, to 0.1 GB. An image that was not read (nil) gets
// DefaultImagePeakGB.
func ImagePeakGB(img *imagecheck.Image) float64 {
	if img == nil || img.DownloadBytes <= 0 {
		return DefaultImagePeakGB
	}
	unpacked := img.UnpackedBytes
	if unpacked <= 0 {
		unpacked = int64(float64(img.DownloadBytes) * 2.1)
	}
	return round1(float64(img.DownloadBytes+unpacked) / 1e9)
}

// NeedFor is the disk a launch should ask for: the image while it unpacks,
// the model download and the run's work files, plus 10 %, rounded up to 10
// GB. Without a model, with an arch the download table does not know, or
// when the model size is unknown, DiskGB is at least
// MinDiskGBUnknownModel and model_source is "default", so a launch can
// still hold a base model (MASTER §3.10.4). sizer nil uses the table sizes.
func NeedFor(img *imagecheck.Image, model *ModelRef, sizer *ModelSizer) Need {
	n := Need{ImagePeakGB: ImagePeakGB(img), WorkGB: WorkGB, ModelSource: "none"}
	if model != nil {
		ctx, cancel := context.WithTimeout(context.Background(), sizeTimeout)
		n.ModelGB, n.ModelSource = sizer.Size(ctx, *model)
		cancel()
		n.ModelGB = round1(n.ModelGB)
	}
	n.DiskGB = max(ceil10((n.ImagePeakGB+n.ModelGB+n.WorkGB)*diskMargin), diskFloorGB)
	if _, known := modelFiles[archOf(model)]; !known || n.ModelSource == "none" {
		n.DiskGB = max(n.DiskGB, MinDiskGBUnknownModel)
		n.ModelSource = "default"
	}
	return n
}

func archOf(m *ModelRef) string {
	if m == nil {
		return ""
	}
	return m.Arch
}

// diskChecks judge the host's disk (hostGB, the most it can allocate; 0 =
// chosen at launch) and the disk the launch asks for (askedGB; 0 = not
// chosen yet) against need. Below the image peak blocks (disk_image);
// below the need is a warn (disk_model) with a set_disk action when the
// host allows the needed size.
func diskChecks(need Need, hostGB float64, askedGB int, model *ModelRef) []imagecheck.Check {
	floor := need.ImagePeakGB + imageHeadroomGB
	peak := int(math.Round(need.ImagePeakGB))
	hostAllows := hostGB == 0 || hostGB >= float64(need.DiskGB)
	switch {
	case hostGB > 0 && hostGB < floor:
		return []imagecheck.Check{{Level: imagecheck.LevelBlock, Code: "disk_image",
			Message: fmt.Sprintf("%d GB is needed just to download and unpack the image; this host has %d GB.", peak, int(hostGB)),
			Fix:     fmt.Sprintf("Pick a host with at least %d GB.", need.DiskGB),
			Action:  &imagecheck.CheckAction{Kind: "filter_hosts"}}}
	case askedGB > 0 && float64(askedGB) < floor:
		c := imagecheck.Check{Level: imagecheck.LevelBlock, Code: "disk_image",
			Message: fmt.Sprintf("%d GB is needed just to download and unpack the image; the launch asks for %d GB.", peak, askedGB),
			Fix:     fmt.Sprintf("Use at least %d GB.", need.DiskGB)}
		if hostAllows {
			c.Action = &imagecheck.CheckAction{Kind: "set_disk", Value: need.DiskGB}
		}
		return []imagecheck.Check{c}
	}
	size := float64(askedGB)
	if askedGB == 0 {
		size = hostGB
	}
	if size <= 0 || size >= float64(need.DiskGB) {
		return nil
	}
	msg := fmt.Sprintf("%d GB fits the image but leaves little room for a base model and checkpoints.", int(size))
	if model != nil && need.ModelGB > 0 {
		msg = fmt.Sprintf("%d GB fits the image but not %s (about %.0f GB) plus checkpoints.", int(size), ArchName(model.Arch), need.ModelGB)
	}
	c := imagecheck.Check{Level: imagecheck.LevelWarn, Code: "disk_model", Message: msg}
	if hostAllows {
		c.Fix = fmt.Sprintf("Use %d GB.", need.DiskGB)
		c.Action = &imagecheck.CheckAction{Kind: "set_disk", Value: need.DiskGB}
	} else {
		c.Fix = fmt.Sprintf("Pick a host with at least %d GB.", need.DiskGB)
		c.Action = &imagecheck.CheckAction{Kind: "filter_hosts"}
	}
	return []imagecheck.Check{c}
}

func ceil10(x float64) int { return int(math.Ceil(x/10-1e-9)) * 10 }

func round1(x float64) float64 { return math.Round(x*10) / 10 }
