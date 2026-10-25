package hostfit

import (
	"slices"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
)

const (
	rev    = registrymock.Revision // f53a407: :latest, :cu130 and :0.13.34 point at it
	newer  = "9a1b2c3"
	goRepo = DefaultRepo
)

// withNewerLatest publishes 9a1b2c3 (0.13.39) and moves :latest to it.
func withNewerLatest(t *testing.T, images []registrymock.Image) []registrymock.Image {
	images = retag(t, images, registrymock.DefaultRepo, rev, newer, map[string]string{
		imagecheck.LabelRevision: newer, imagecheck.LabelVersion: "0.13.39",
	}, "2026-10-30T09:00:00Z")
	return retag(t, images, registrymock.DefaultRepo, newer, "latest", nil, "")
}

func oldDriverHost() Host {
	return Host{GPUName: "NVIDIA GeForce RTX 4090", SM: 89, GPUVendor: "nvidia", GPUClass: "geforce", CPUArch: "amd64",
		CUDAMax: "12.8", DiskGB: 500, InetDownMbps: 900, DiskBWMBps: 2000, VRAMGB: 23.99}
}

func recommend(t *testing.T, c *imagecheck.Client, in ImageRecommendInput) ImageRecommendation {
	t.Helper()
	out, err := RecommendImage(t.Context(), c, nil, in)
	if err != nil {
		t.Fatal(err)
	}
	if out.Alternatives == nil || out.Checks == nil {
		t.Fatalf("nil slices: %+v", out)
	}
	return out
}

// 1 and 9: the published build this server runs fits, and the why lines
// carry the numbers.
func TestRecommendSame(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	host := FromVast(vastRow(t, 38341566)) // RTX 5090, CUDA 13.0, 1376 Mbps
	out := recommend(t, c, ImageRecommendInput{Host: host, ServerRevision: rev + "1c2b3d4e5f60718293a4b5c6d7e8f9012", Model: &ModelRef{Arch: "flux"}})
	ch := out.Choice
	if ch == nil || ch.Channel != "same" || ch.Image != goRepo+":"+rev || *ch.LaunchRef != goRepo+":"+rev || ch.Level != "info" {
		t.Fatalf("choice: %+v", ch)
	}
	if ch.Label != "AI Toolkit 0.13.34 · build f53a407" || ch.Revision != rev || ch.Flavor != "cu130" || !strings.HasPrefix(ch.Digest, "sha256:") {
		t.Errorf("facts: %+v", ch)
	}
	want := []string{"Same build as this server", "Needs CUDA 13.0; this host has 13.0", "Runs on sm_120 (RTX 5090)",
		"6.6 GB download, about 3 min at 1376 Mbps", "Pinned to build f53a407 (" + shortDigest(ch.Digest) + ")"}
	if !slices.Equal(ch.Why, want) {
		t.Errorf("why:\n%q\nwant\n%q", ch.Why, want)
	}
	// :latest and :cu130 are the same image: not listed twice.
	if len(out.Alternatives) != 0 {
		t.Errorf("alternatives: %+v", out.Alternatives)
	}
	if !hasCode(out.Checks, "pull_eta") || hasCode(out.Checks, "image_unpinned") {
		t.Errorf("checks: %v", codesOf(out.Checks))
	}
	// A 4090 runs the sm_86 build.
	h4090 := FromVast(vastRow(t, 51256153))
	if out := recommend(t, c, ImageRecommendInput{Host: h4090, ServerRevision: rev}); !slices.Contains(out.Choice.Why, "Runs on sm_89 through the sm_86 build") {
		t.Errorf("4090 why: %q", out.Choice.Why)
	}
}

// 2: this server's build isn't published: the newest build, with an info.
func TestRecommendSameUnpublished(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	out := recommend(t, c, ImageRecommendInput{Host: FromVast(vastRow(t, 38341566)), ServerRevision: "abc1234"})
	if out.Choice == nil || out.Choice.Channel != "latest" || out.Choice.Why[0] != "Newest build" || *out.Choice.LaunchRef != goRepo+":"+rev {
		t.Fatalf("choice: %+v", out.Choice)
	}
	if c := checkOf(t, out.Checks, "image_same_unpublished"); c.Message != "This server's build (abc1234) isn't published; using the newest build." {
		t.Errorf("info: %+v", c)
	}
	if c := checkOf(t, out.Checks, "image_unpinned"); c.Message != "latest is build f53a407 today." {
		t.Errorf("pin: %+v", c)
	}
}

// 3: a host whose driver is too old for the only flavor: nothing fits, and
// the fix is to filter hosts.
func TestRecommendNothingFits(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	out := recommend(t, c, ImageRecommendInput{Host: oldDriverHost(), ServerRevision: rev})
	if out.Choice != nil || out.Action == nil || out.Action.Kind != "filter_hosts" {
		t.Fatalf("choice %+v action %+v", out.Choice, out.Action)
	}
	if !slices.Equal(codesOf(out.Checks), []string{"host_cuda"}) {
		t.Errorf("checks: %v", codesOf(out.Checks))
	}
	if len(out.Alternatives) == 0 || out.Alternatives[0].Level != "block" || out.Alternatives[0].LaunchRef != nil {
		t.Errorf("alternatives: %+v", out.Alternatives)
	}
}

// 4: a cu126 flavor channel wins the CUDA 12.8 host.
func TestRecommendFlavor(t *testing.T) {
	_, c := newMock(t, retag(t, registrymock.Fixtures(), registrymock.DefaultRepo, "cu126test", "cu126", nil, ""), registrymock.Faults{})
	out := recommend(t, c, ImageRecommendInput{Host: oldDriverHost(), ServerRevision: rev})
	ch := out.Choice
	if ch == nil || ch.Channel != "flavor" || ch.Flavor != "cu126" || ch.Image != goRepo+":cu126" {
		t.Fatalf("choice: %+v", ch)
	}
	if ch.Why[0] != "Different flavor (cu126) for this host's driver" || ch.Why[1] != "Needs CUDA 12.6; this host has 12.8" ||
		!strings.HasSuffix(ch.Label, " · cu126") || *ch.LaunchRef != goRepo+":cu126" || !hasCode(out.Checks, "image_unpinned") {
		t.Errorf("flavor choice: %+v", ch)
	}
	if len(out.Alternatives) != 1 || out.Alternatives[0].Channel != "same" || out.Alternatives[0].Level != "block" {
		t.Errorf("alternatives: %+v", out.Alternatives)
	}
}

// 5 and 6: custom images.
func TestRecommendCustom(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	host := FromVast(vastRow(t, 38341566))
	out := recommend(t, c, ImageRecommendInput{Host: host, Custom: "ostris/aitoolkit:latest"})
	if out.Choice != nil || out.Action != nil || !slices.Equal(codesOf(out.Checks), []string{"image_not_go"}) {
		t.Errorf("ostris: choice %+v action %+v checks %v", out.Choice, out.Action, codesOf(out.Checks))
	}
	out = recommend(t, c, ImageRecommendInput{Host: host, Custom: "quay.io/someone/trainer:1"})
	if out.Choice == nil || out.Choice.Channel != "custom" || out.Choice.Level != "warn" || *out.Choice.LaunchRef != "quay.io/someone/trainer:1" {
		t.Fatalf("quay.io: %+v", out.Choice)
	}
	if c := checkOf(t, out.Checks, "image_unchecked"); c.Message != "Images on quay.io can't be checked." {
		t.Errorf("unchecked: %+v", c)
	}
	// A missing tag names the newest build and offers it.
	out = recommend(t, c, ImageRecommendInput{Custom: goRepo + ":0.14.0"})
	nf := checkOf(t, out.Checks, "image_not_found")
	if out.Choice != nil || nf.Fix != "Newest build: 0.13.34 · f53a407." || nf.Action == nil || nf.Action.Value != goRepo+":"+rev {
		t.Errorf("not found: %+v", nf)
	}
}

// An image that cannot be read and has no fallback (an unchecked registry)
// has no known size: no disk block and no pull time, and the launch is
// allowed with the warn (IMAGE-PLAN §4.3).
func TestRecommendCustomUnknownSize(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	host := FromVast(vastRow(t, 38341566)) // 1376 Mbps
	host.DiskGB = 16
	out := recommend(t, c, ImageRecommendInput{Host: host, Custom: "quay.io/someone/trainer:1", Model: &ModelRef{Arch: "flux"}})
	if out.Choice == nil || out.Choice.Level != "warn" || out.Choice.LaunchRef == nil {
		t.Fatalf("choice: %+v checks %v", out.Choice, codesOf(out.Checks))
	}
	for _, code := range []string{"disk_image", "disk_model", "pull_eta", "pull_slow"} {
		if hasCode(out.Checks, code) {
			t.Errorf("%s for an image of unknown size: %v", code, codesOf(out.Checks))
		}
	}
}

// 7: a QuickPod template runs only its own image; an unpinned tag gets the
// edit_template info.
func TestRecommendTemplate(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	host := FromQuickPod(quickPodOffers(t)[0])
	out := recommend(t, c, ImageRecommendInput{Host: host, TemplateImage: goRepo + ":latest", ServerRevision: rev})
	if out.Choice == nil || out.Choice.Channel != "template" || *out.Choice.LaunchRef != goRepo+":latest" || out.Choice.Why[0] != "The image of the QuickPod template" {
		t.Fatalf("template: %+v", out.Choice)
	}
	tag := checkOf(t, out.Checks, "image_template_tag")
	if tag.Message != "Change the tag in the template's image path to f53a407." || tag.Action.Kind != "edit_template" || tag.Action.Value != rev {
		t.Errorf("edit_template: %+v", tag)
	}
	if len(out.Alternatives) != 1 || out.Alternatives[0].Image != goRepo+":"+rev || out.Alternatives[0].Label != "Change the tag in the template's image path to f53a407" {
		t.Errorf("alternatives: %+v", out.Alternatives)
	}
	// The template's tag blocks here (an old build that drops sm_89) while
	// the newest build runs: no choice (QuickPod runs only the template's
	// image), and the fix is to edit the template's tag.
	images := retag(t, withNewerLatest(t, registrymock.Fixtures()), registrymock.DefaultRepo, rev, rev,
		map[string]string{imagecheck.LabelSM: "90,100,120"}, "")
	_, c2 := newMock(t, images, registrymock.Faults{})
	h4090 := FromVast(vastRow(t, 51256153))
	stuck := recommend(t, c2, ImageRecommendInput{Host: h4090, TemplateImage: goRepo + ":" + rev})
	if stuck.Choice != nil {
		t.Fatalf("choice outside the template: %+v", stuck.Choice)
	}
	if stuck.Action == nil || stuck.Action.Kind != "edit_template" || stuck.Action.Value != newer {
		t.Errorf("action: %+v", stuck.Action)
	}
	if !hasCode(stuck.Checks, "gpu_sm") || !hasCode(stuck.Checks, "image_template_tag") {
		t.Errorf("checks: %v", codesOf(stuck.Checks))
	}
	if n := len(stuck.Alternatives); n != 2 || stuck.Alternatives[1].Image != goRepo+":"+newer || stuck.Alternatives[1].Level == "block" {
		t.Errorf("alternatives: %+v", stuck.Alternatives)
	}

	pinned := recommend(t, c, ImageRecommendInput{Host: host, TemplateImage: goRepo + ":" + rev})
	if hasCode(pinned.Checks, "image_template_tag") || len(pinned.Alternatives) != 0 {
		t.Errorf("pinned template: %v %+v", codesOf(pinned.Checks), pinned.Alternatives)
	}
}

// 8: prefer latest puts the newest build first and says this server runs
// an older one; the default prefers this server's build. When this
// server's build cannot run, the newer one is chosen and says so first.
func TestRecommendPrefer(t *testing.T) {
	images := withNewerLatest(t, registrymock.Fixtures())
	_, c := newMock(t, images, registrymock.Faults{})
	host := FromVast(vastRow(t, 38341566))

	same := recommend(t, c, ImageRecommendInput{Host: host, ServerRevision: rev})
	if same.Choice.Channel != "same" || len(same.Alternatives) != 1 || same.Alternatives[0].Channel != "latest" || same.Alternatives[0].Revision != newer {
		t.Fatalf("prefer same: %+v alts %+v", same.Choice, same.Alternatives)
	}
	latest := recommend(t, c, ImageRecommendInput{Host: host, ServerRevision: rev, Prefer: "latest"})
	if latest.Choice.Channel != "latest" || latest.Choice.Revision != newer || *latest.Choice.LaunchRef != goRepo+":"+newer {
		t.Fatalf("prefer latest: %+v", latest.Choice)
	}
	skew := checkOf(t, latest.Checks, "image_skew")
	if skew.Message != "This is a newer build than this server (f53a407 → 9a1b2c3)." || skew.Action.Value != goRepo+":"+rev {
		t.Errorf("skew: %+v", skew)
	}

	// This server's build drops sm_89: a 4090 gets the newer build, named as such.
	images = retag(t, images, registrymock.DefaultRepo, rev, rev, map[string]string{imagecheck.LabelSM: "90,100,120"}, "")
	_, c = newMock(t, images, registrymock.Faults{})
	out := recommend(t, c, ImageRecommendInput{Host: FromVast(vastRow(t, 51256153)), ServerRevision: rev, ServerVersion: "0.13.34"})
	if out.Choice == nil || out.Choice.Channel != "latest" || out.Choice.Why[0] != "Newer build than this server (0.13.34 · f53a407 → 0.13.39 · 9a1b2c3)" {
		t.Fatalf("newer build: %+v", out.Choice)
	}
	if hasCode(out.Checks, "image_skew") {
		t.Error("skew info while this server's build can't run here")
	}
}

// The default image not published yet: nothing to choose, and the copy
// says so (MASTER §3.4).
func TestRecommendNotPublished(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{Private: map[string]bool{registrymock.DefaultRepo: true}})
	out := recommend(t, c, ImageRecommendInput{Host: FromVast(vastRow(t, 38341566)), ServerRevision: rev})
	if out.Choice != nil || out.Action != nil {
		t.Fatalf("choice %+v action %+v", out.Choice, out.Action)
	}
	if d := checkOf(t, out.Checks, "image_denied"); !strings.Contains(d.Message, "isn't published yet") {
		t.Errorf("denied: %+v", d)
	}
}

func TestLaunchRef(t *testing.T) {
	_, c := newMock(t, nil, registrymock.Faults{})
	ctx := t.Context()
	latest, err := c.Inspect(ctx, goRepo+":latest", "", false)
	if err != nil {
		t.Fatal(err)
	}
	if ref, info := LaunchRef(ctx, c, latest, goRepo+":latest", false); ref != goRepo+":"+rev || info == nil || info.Code != "image_unpinned" {
		t.Errorf("latest: %s %+v", ref, info)
	}
	if ref, _ := LaunchRef(ctx, c, latest, goRepo+":latest", true); ref != goRepo+"@"+latest.Digest {
		t.Errorf("digest ref: %s", ref)
	}
	if ref, info := LaunchRef(ctx, c, latest, goRepo+":"+rev, false); ref != goRepo+":"+rev || info != nil {
		t.Errorf("sha tag: %s %+v", ref, info)
	}
	old, err := c.Inspect(ctx, goRepo+":oldsha1", "", false)
	if err != nil {
		t.Fatal(err)
	}
	if ref, info := LaunchRef(ctx, c, old, goRepo+":oldsha1", false); ref != goRepo+":oldsha1" || info == nil || info.Message != "This tag can move." {
		t.Errorf("unlabelled Go image: %s %+v", ref, info)
	}
	if got := shortDigest("sha256:d89f0000000000000000000000000000000000000000000000000000059e30b"); got != "sha256:d89f…59e30b" {
		t.Errorf("shortDigest %q", got)
	}
}
