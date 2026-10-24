package imagecheck

import (
	"errors"
	"fmt"
	"strings"
)

// Check levels, worst first. "ok" is the absence of checks.
const (
	LevelBlock = "block"
	LevelWarn  = "warn"
	LevelInfo  = "info"
)

// Check is one finding about an image or a host, with copy the studio shows
// as is: message in plain words, fix with the technical fact (IMAGE-PLAN
// §6.4). The JSON shape is the /v1 ImageCheckItem.
type Check struct {
	Level   string       `json:"level"` // block | warn | info
	Code    string       `json:"code"`  // stable: image_denied, host_cuda, ...
	Message string       `json:"message"`
	Fix     string       `json:"fix,omitempty"`
	Action  *CheckAction `json:"action,omitempty"`
}

// CheckAction is a button the studio renders next to a check.
type CheckAction struct {
	Kind  string `json:"kind"` // filter_hosts | show_all | set_disk | use_image | pick_gpu | edit_template
	Value any    `json:"value,omitempty"`
}

// SettingsURL is where the owner makes the default image's package public.
const SettingsURL = "https://github.com/users/geocine/packages/container/aitoolkit-go/settings"

// DeniedCopy is the message and fix for an image a registry (or a provider
// pulling it) refused anonymously. The default image gets the "not
// published yet" copy, since that is the likely cause until its owner
// publishes it (MASTER §3.4).
func DeniedCopy(repo string) (message, fix string) {
	if r, err := ParseRef(repo); err == nil {
		repo = r.Name()
	}
	if repo == DefaultRepo {
		return "The server image " + DefaultRepo + " isn't published yet, or it's still private.",
			"Its owner publishes it once with ui-go/scripts/build_and_push.sh --push and makes the package public. Until then GPU servers can't start it."
	}
	return "The image " + repo + " wasn't found, or it's private.",
		"GPU servers pull it without a login, so the package must be public."
}

const (
	stillLaunch  = "You can still launch; problems will show in the provider's log."
	refFormatFix = "Use a form like " + DefaultRepo + ":latest."
)

// ErrorCheck is the check for an image that could not be read. Registry
// trouble (unreachable, rate limited, a registry outside the allowlist) is
// a warn, never a block; an image that is missing, private or built for
// another platform blocks. Its image_not_found fix is generic: a caller that
// can list the repo's tags uses NotFoundCheck instead, which names the
// newest build and offers it (IMAGE-PLAN §6.4).
func ErrorCheck(err error) Check {
	var e *Error
	if !errors.As(err, &e) {
		return Check{Level: LevelWarn, Code: "image_unreachable", Message: "Couldn't check the image right now.", Fix: stillLaunch}
	}
	switch e.Kind {
	case KindBadRef:
		return Check{Level: LevelBlock, Code: "image_ref", Message: "That isn't an image name.", Fix: refFormatFix}
	case KindNotAllowed:
		host := e.Repo
		if i := strings.IndexByte(host, '/'); i > 0 {
			host = host[:i]
		}
		return Check{Level: LevelWarn, Code: "image_unchecked", Message: "Images on " + host + " can't be checked.", Fix: stillLaunch}
	case KindDenied:
		msg, fix := DeniedCopy(e.Repo)
		return Check{Level: LevelBlock, Code: "image_denied", Message: msg, Fix: fix}
	case KindNotFound:
		msg := "There's no tag " + e.Tag + " for this image."
		if e.Tag == "" {
			msg = "There's no image with that digest in " + e.Repo + "."
		}
		return Check{Level: LevelBlock, Code: "image_not_found", Message: msg, Fix: "Pick a published tag."}
	case KindNoPlatform:
		return Check{Level: LevelBlock, Code: "image_platform",
			Message: "This image is built for " + archList(e.Platforms) + " only. Rented GPU servers here run amd64.",
			Fix:     "Rebuild with --platform linux/amd64."}
	case KindRateLimited:
		return Check{Level: LevelWarn, Code: "image_rate_limited", Message: "Docker Hub's anonymous limit is nearly used on this server.",
			Fix: "The GPU host pulls on its own limit; GHCR images aren't affected."}
	}
	return Check{Level: LevelWarn, Code: "image_unreachable", Message: "Couldn't check the image right now.", Fix: stillLaunch}
}

// NotFoundCheck is ErrorCheck with the newest build filled in for
// image_not_found: fix "Newest build: 0.13.34 · f53a407." and a use_image
// action pinned to that build. tags is Tags() for the image's repo (newest
// first); any other error, or no tags, is plain ErrorCheck.
func NotFoundCheck(err error, tags []TagInfo) Check {
	c := ErrorCheck(err)
	var e *Error
	if c.Code != "image_not_found" || len(tags) == 0 || !errors.As(err, &e) || e.Repo == "" {
		return c
	}
	newest := tags[0]
	build, version := newest.Revision, ""
	for _, t := range tags {
		if t.Digest != newest.Digest {
			continue
		}
		if version == "" && versionTag.MatchString(t.Tag) {
			version = t.Tag
		}
		if build == "" && !t.Moving {
			build = t.Tag
		}
	}
	if build == "" {
		build = newest.Tag
	}
	c.Fix = "Newest build: " + build + "."
	if version != "" {
		c.Fix = "Newest build: " + version + " · " + build + "."
	}
	c.Action = &CheckAction{Kind: "use_image", Value: e.Repo + ":" + build}
	return c
}

// archList is "arm64", "arm64 and arm/v7", or "another platform".
func archList(platforms []string) string {
	var archs []string
	for _, p := range platforms {
		if _, a, ok := strings.Cut(p, "/"); ok {
			archs = append(archs, a)
		}
	}
	switch len(archs) {
	case 0:
		return "another platform"
	case 1:
		return archs[0]
	}
	return strings.Join(archs[:len(archs)-1], ", ") + " and " + archs[len(archs)-1]
}

// ImageChecks are the image-side checks for an image that was read: is it
// the Go server the studio drives, on the port and API it expects, and does
// it say which GPUs it runs on. Host-side checks (driver, GPU, disk, pull
// time) are hostfit's.
func ImageChecks(img *Image) []Check {
	checks := []Check{}
	req := img.Requires
	if !img.GoServer {
		checks = append(checks, Check{Level: LevelBlock, Code: "image_not_go",
			Message: "This image doesn't include the server the studio talks to.",
			Fix:     "Use " + DefaultRepo + ", or an image built with ui-go/docker/Dockerfile.",
			Action:  &CheckAction{Kind: "use_image", Value: DefaultRepo + ":latest"}})
	} else {
		if req.Port != ServerPort {
			checks = append(checks, Check{Level: LevelBlock, Code: "image_port",
				Message: fmt.Sprintf("The image doesn't expose port %d, so the studio can't reach it.", ServerPort),
				Fix:     fmt.Sprintf("EXPOSE %d in the Dockerfile.", ServerPort)})
		}
		if req.API != APIVersion {
			checks = append(checks, Check{Level: LevelWarn, Code: "image_api",
				Message: "This build speaks a different API version than this server.",
				Fix:     "Some studio features may not work on it."})
		}
	}
	if len(req.SM) == 0 && len(req.PTX) == 0 {
		checks = append(checks, Check{Level: LevelInfo, Code: "image_sm_unknown",
			Message: "Couldn't tell which GPUs this image supports.",
			Fix:     "It has no PyTorch CUDA build information."})
	}
	return checks
}

// Worst is the worst level among checks: block, warn, info, or "ok".
func Worst(checks []Check) string {
	rank := map[string]int{LevelInfo: 1, LevelWarn: 2, LevelBlock: 3}
	worst := "ok"
	for _, c := range checks {
		if rank[c.Level] > rank[worst] {
			worst = c.Level
		}
	}
	return worst
}
