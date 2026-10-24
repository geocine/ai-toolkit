package imagecheck

import (
	"errors"
	"fmt"
)

// Kind names why an image could not be read. Kinds map to check codes and to
// the friendly copy the studio shows.
type Kind string

const (
	KindBadRef      Kind = "bad_ref"
	KindNotAllowed  Kind = "not_allowed"  // registry outside the allowlist: no request was made
	KindDenied      Kind = "denied"       // token 401/403: private or missing (GHCR cannot tell which)
	KindNotFound    Kind = "not_found"    // manifest 404 MANIFEST_UNKNOWN: tag or digest missing
	KindNoPlatform  Kind = "no_platform"  // index has no linux/amd64
	KindRateLimited Kind = "rate_limited" // 429, or Docker Hub remaining <= 5
	KindUnreachable Kind = "unreachable"  // dial, TLS, timeout
	KindBadResponse Kind = "bad_response" // non-JSON, oversize, digest mismatch, unknown media type
)

// Error is every failure the package returns.
type Error struct {
	Kind    Kind
	Status  int    // HTTP status, 0 for network errors
	Message string // never contains a token or an Authorization header

	// What was asked, for the friendly copy (ErrorCheck). Empty when the
	// reference did not parse.
	Repo      string   // familiar repository name: "ghcr.io/geocine/aitoolkit-go"
	Tag       string   // tag as asked; "" for a digest reference
	Digest    string   // digest as asked
	Platforms []string // KindNoPlatform: what the image is built for

	cause error // the network error behind KindUnreachable (errors.Is(err, netguard.ErrBlocked))
}

func (e *Error) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("imagecheck: %s (HTTP %d): %s", e.Kind, e.Status, e.Message)
	}
	return fmt.Sprintf("imagecheck: %s: %s", e.Kind, e.Message)
}

// Unwrap is the network error behind the failure, if any.
func (e *Error) Unwrap() error { return e.cause }

// Errorf builds an *Error with no HTTP status.
func Errorf(kind Kind, format string, args ...any) *Error {
	return &Error{Kind: kind, Message: fmt.Sprintf(format, args...)}
}

// KindOf is err's Kind, or "" when err is not (or does not wrap) an *Error.
func KindOf(err error) Kind {
	var e *Error
	if errors.As(err, &e) {
		return e.Kind
	}
	return ""
}

// withRef fills in what was asked, so the copy can name it.
func (e *Error) withRef(r Ref) *Error {
	if e.Repo == "" {
		e.Repo, e.Tag, e.Digest = r.Name(), r.Tag, r.Digest
	}
	return e
}
