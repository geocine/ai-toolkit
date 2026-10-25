package imagecheck

import (
	"errors"
	"strings"
	"testing"
)

const testDigest = "sha256:d89f0123456789abcdef0123456789abcdef0123456789abcdef0123459e30b0"

func TestParseRef(t *testing.T) {
	cases := []struct {
		in   string
		want Ref
		str  string
	}{
		{"alpine", Ref{"docker.io", "registry-1.docker.io", "library/alpine", "latest", ""}, "alpine:latest"},
		{"alpine:3.20", Ref{"docker.io", "registry-1.docker.io", "library/alpine", "3.20", ""}, "alpine:3.20"},
		{"docker.io/x/y", Ref{"docker.io", "registry-1.docker.io", "x/y", "latest", ""}, "x/y:latest"},
		{"index.docker.io/x/y:1", Ref{"docker.io", "registry-1.docker.io", "x/y", "1", ""}, "x/y:1"},
		{"docker.io/alpine", Ref{"docker.io", "registry-1.docker.io", "library/alpine", "latest", ""}, "alpine:latest"},
		{"ostris/aitoolkit:0.13.39", Ref{"docker.io", "registry-1.docker.io", "ostris/aitoolkit", "0.13.39", ""}, "ostris/aitoolkit:0.13.39"},
		{"ghcr.io/a/b:c", Ref{"ghcr.io", "ghcr.io", "a/b", "c", ""}, "ghcr.io/a/b:c"},
		{"ghcr.io/geocine/aitoolkit-go", Ref{"ghcr.io", "ghcr.io", "geocine/aitoolkit-go", "latest", ""}, "ghcr.io/geocine/aitoolkit-go:latest"},
		{"GHCR.io/geocine/aitoolkit-go:f53a407", Ref{"ghcr.io", "ghcr.io", "geocine/aitoolkit-go", "f53a407", ""}, "ghcr.io/geocine/aitoolkit-go:f53a407"},
		{"localhost:5000/a", Ref{"localhost:5000", "localhost:5000", "a", "latest", ""}, "localhost:5000/a:latest"},
		{"localhost/a/b", Ref{"localhost", "localhost", "a/b", "latest", ""}, "localhost/a/b:latest"},
		{"[::1]:5000/a", Ref{"[::1]:5000", "[::1]:5000", "a", "latest", ""}, "[::1]:5000/a:latest"},
		{"quay.io/org/sub/repo_x__y.z-w:v1.2_3", Ref{"quay.io", "quay.io", "org/sub/repo_x__y.z-w", "v1.2_3", ""}, "quay.io/org/sub/repo_x__y.z-w:v1.2_3"},
		// Docker Hub drops "library/" only when nothing else follows it.
		{"docker.io/library/a/b", Ref{"docker.io", "registry-1.docker.io", "library/a/b", "latest", ""}, "library/a/b:latest"},
		{"library/a/b:1", Ref{"docker.io", "registry-1.docker.io", "library/a/b", "1", ""}, "library/a/b:1"},
		{"library/alpine", Ref{"docker.io", "registry-1.docker.io", "library/alpine", "latest", ""}, "alpine:latest"},
		// A tag next to a digest is kept as typed; the digest is what gets resolved.
		{"host:5000/a/b:t@" + testDigest, Ref{"host:5000", "host:5000", "a/b", "t", testDigest}, "host:5000/a/b:t@" + testDigest},
		{"alpine@" + testDigest, Ref{"docker.io", "registry-1.docker.io", "library/alpine", "", testDigest}, "alpine@" + testDigest},
	}
	for _, c := range cases {
		got, err := ParseRef(c.in)
		if err != nil {
			t.Errorf("ParseRef(%q): %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("ParseRef(%q) = %+v, want %+v", c.in, got, c.want)
		}
		if got.String() != c.str {
			t.Errorf("ParseRef(%q).String() = %q, want %q", c.in, got.String(), c.str)
		}
		// The familiar form parses back to the same reference.
		if again, err := ParseRef(got.String()); err != nil || again != got {
			t.Errorf("round trip of %q: %+v %v", got.String(), again, err)
		}
	}
}

func TestParseRefRejects(t *testing.T) {
	bad := []string{
		"",
		"Alpine",
		"ghcr.io/Geocine/aitoolkit-go",
		"ghcr.io/geocine/AITOOLKIT",
		"alpine:-bad",
		"alpine:.bad",
		"alpine:" + strings.Repeat("a", 129),
		"alpine:bad/tag",
		"alpine@sha256:abc",
		"alpine@md5:" + strings.Repeat("a", 32),
		"alpine@" + strings.ToUpper(testDigest),
		"alpine latest",
		" alpine",
		"alpine\n",
		"alpine\t:1",
		"ghcr.io/" + strings.Repeat("a", 250),
		"ghcr.io/",
		"ghcr.io//x",
		"a//b",
		"-a/b",
		"a/b-",
		"a/b..c",
		":latest",
		"bad_host.io/a",
		"host:port/a",
	}
	for _, s := range bad {
		r, err := ParseRef(s)
		if err == nil {
			t.Errorf("ParseRef(%q) = %+v, want an error", s, r)
			continue
		}
		var e *Error
		if !errors.As(err, &e) || e.Kind != KindBadRef || KindOf(err) != KindBadRef {
			t.Errorf("ParseRef(%q): %v is not a bad_ref *Error", s, err)
		}
	}
	// 128 characters is the longest tag.
	if _, err := ParseRef("alpine:" + strings.Repeat("a", 128)); err != nil {
		t.Errorf("128-char tag: %v", err)
	}
}

func TestRefHelpers(t *testing.T) {
	r, err := ParseRef("ghcr.io/geocine/aitoolkit-go@" + testDigest)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.WithTag("f53a407"); got.Digest != "" || got.Tag != "f53a407" || got.String() != "ghcr.io/geocine/aitoolkit-go:f53a407" {
		t.Errorf("WithTag: %+v", got)
	}
	if got := r.Pinned(testDigest); got != "ghcr.io/geocine/aitoolkit-go@"+testDigest {
		t.Errorf("Pinned: %q", got)
	}
	if r.Name() != DefaultRepo {
		t.Errorf("Name: %q", r.Name())
	}

	// Name drops the registry for Docker Hub and keeps it elsewhere.
	r, _ = ParseRef("ghcr.io/geocine/aitoolkit-go:f53a407")
	if p, tag := r.Name(), r.Tag; p != "ghcr.io/geocine/aitoolkit-go" || tag != "f53a407" {
		t.Errorf("Name: %q %q", p, tag)
	}
	r, _ = ParseRef("alpine")
	if p, tag := r.Name(), r.Tag; p != "alpine" || tag != "latest" {
		t.Errorf("Name docker hub: %q %q", p, tag)
	}
	if p := (Ref{Domain: "docker.io", API: "registry-1.docker.io", Repo: "library/alpine", Digest: testDigest}).Name(); p != "alpine" {
		t.Errorf("Name digest-only: %q", p)
	}
	r, _ = ParseRef("ghcr.io/geocine/aitoolkit-go:f53a407@" + testDigest)
	if p, tag := r.Name(), r.Tag; p != "ghcr.io/geocine/aitoolkit-go" || tag != "f53a407" {
		t.Errorf("Name tag and digest: %q %q", p, tag)
	}
	if got := r.Pinned(r.Digest); got != "ghcr.io/geocine/aitoolkit-go@"+testDigest {
		t.Errorf("Pinned with a tag: %q", got)
	}
	if p := (Ref{Domain: "docker.io", API: "registry-1.docker.io", Repo: "library/a/b", Tag: "1"}).Name(); p != "library/a/b" {
		t.Errorf("Name library sub-path: %q", p)
	}
}

func TestErrorText(t *testing.T) {
	e := &Error{Kind: KindDenied, Status: 403, Message: "private or missing"}
	if e.Error() != "imagecheck: denied (HTTP 403): private or missing" {
		t.Errorf("%q", e.Error())
	}
	if Errorf(KindUnreachable, "dial %s", "ghcr.io").Error() != "imagecheck: unreachable: dial ghcr.io" {
		t.Error("Errorf text")
	}
	if KindOf(errors.New("plain")) != "" || KindOf(nil) != "" {
		t.Error("KindOf of a non-imagecheck error")
	}
}
