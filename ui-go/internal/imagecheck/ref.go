package imagecheck

import (
	"regexp"
	"strings"
)

// Ref is a parsed image reference (distribution/reference grammar, Docker's familiar defaults).
type Ref struct {
	Domain string // as the user means it: "ghcr.io", "docker.io", "localhost:5000"
	API    string // registry API host: "ghcr.io", "registry-1.docker.io"
	Repo   string // "geocine/aitoolkit-go", "library/alpine"
	Tag    string // as typed; "latest" when neither tag nor digest was given, "" for a digest alone
	Digest string // "sha256:<64 hex>"; when both are given the digest is what gets resolved
}

const (
	dockerDomain = "docker.io"
	dockerAPI    = "registry-1.docker.io"
	maxRefLen    = 255
)

var (
	// One path component: lowercase alphanumerics joined by ".", "_", "__" or dashes.
	pathComponent = regexp.MustCompile(`^[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*$`)
	tagPattern    = regexp.MustCompile(`^[\w][\w.-]{0,127}$`)
	digestPattern = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
	domainPattern = regexp.MustCompile(`^(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*|\[[0-9a-fA-F:.]+\])(?::[0-9]{1,5})?$`)
)

// ParseRef parses s the way docker pull does. It rejects uppercase repos, bad
// tags (^[\w][\w.-]{0,127}$), bad digests, more than 255 characters and any
// whitespace.
func ParseRef(s string) (Ref, error) {
	if s == "" {
		return Ref{}, Errorf(KindBadRef, "the image reference is empty")
	}
	if len(s) > maxRefLen {
		return Ref{}, Errorf(KindBadRef, "the image reference is longer than %d characters", maxRefLen)
	}
	if strings.IndexFunc(s, isSpace) >= 0 {
		return Ref{}, Errorf(KindBadRef, "the image reference contains whitespace")
	}

	var r Ref
	name := s
	if at := strings.IndexByte(s, '@'); at >= 0 {
		name, r.Digest = s[:at], s[at+1:]
		if !digestPattern.MatchString(r.Digest) {
			return Ref{}, Errorf(KindBadRef, "%q is not a sha256 digest", r.Digest)
		}
	}
	// A tag follows the last ':' after the last '/' (a ':' before it is a port).
	if c := strings.LastIndexByte(name, ':'); c > strings.LastIndexByte(name, '/') {
		tag := name[c+1:]
		name = name[:c]
		if !tagPattern.MatchString(tag) {
			return Ref{}, Errorf(KindBadRef, "%q is not a valid tag", tag)
		}
		r.Tag = tag
	}
	if name == "" {
		return Ref{}, Errorf(KindBadRef, "the image reference has no repository")
	}

	// The first component is a registry when it looks like a host.
	path := name
	if slash := strings.IndexByte(name, '/'); slash >= 0 {
		first := name[:slash]
		if strings.ContainsAny(first, ".:") || first == "localhost" || strings.ToLower(first) != first {
			if !domainPattern.MatchString(first) {
				return Ref{}, Errorf(KindBadRef, "%q is not a valid registry host", first)
			}
			r.Domain = strings.ToLower(first)
			path = name[slash+1:]
		}
	}
	for _, part := range strings.Split(path, "/") {
		if !pathComponent.MatchString(part) {
			if strings.ToLower(part) != part {
				return Ref{}, Errorf(KindBadRef, "repository names must be lowercase (%q)", path)
			}
			return Ref{}, Errorf(KindBadRef, "%q is not a valid repository name", path)
		}
	}

	switch r.Domain {
	case "", dockerDomain, "index.docker.io", dockerAPI:
		r.Domain, r.API = dockerDomain, dockerAPI
		if !strings.Contains(path, "/") {
			path = "library/" + path
		}
	default:
		r.API = r.Domain
	}
	r.Repo = path
	if r.Tag == "" && r.Digest == "" {
		r.Tag = "latest"
	}
	return r, nil
}

func isSpace(c rune) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' || c == 0x85 || c == 0xA0
}

// Name is the familiar repository name: "ghcr.io/geocine/aitoolkit-go",
// "alpine", "x/y" (Docker Hub drops its domain, and "library/" when nothing
// else follows it: "library/a/b" stays as it is).
func (r Ref) Name() string {
	if r.Domain == dockerDomain {
		if rest := strings.TrimPrefix(r.Repo, "library/"); rest != r.Repo && !strings.Contains(rest, "/") {
			return rest
		}
		return r.Repo
	}
	return r.Domain + "/" + r.Repo
}

// String is the familiar form: "ghcr.io/geocine/aitoolkit-go:latest",
// "alpine:3.20", "alpine@sha256:...", "a/b:t@sha256:..." (tag and digest both
// kept, as docker prints them).
func (r Ref) String() string {
	s := r.Name()
	if r.Tag != "" {
		s += ":" + r.Tag
	}
	if r.Digest != "" {
		s += "@" + r.Digest
	}
	return s
}

// WithTag drops the digest.
func (r Ref) WithTag(tag string) Ref {
	r.Tag, r.Digest = tag, ""
	return r
}

// Pinned is the reference by digest: "ghcr.io/geocine/aitoolkit-go@sha256:...".
func (r Ref) Pinned(digest string) string { return r.Name() + "@" + digest }

// Split gives the QuickPod template fields image_path and version_tag. A
// reference pinned only by digest has no tag (""); one with both keeps the tag
// as typed.
func (r Ref) Split() (imagePath, tag string) { return r.Name(), r.Tag }
