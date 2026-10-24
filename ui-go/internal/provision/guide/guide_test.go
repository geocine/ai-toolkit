package guide

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/mockkit"
	_ "github.com/geocine/aitoolkit/internal/provision/quickpod"
	_ "github.com/geocine/aitoolkit/internal/provision/runpod"
	_ "github.com/geocine/aitoolkit/internal/provision/vast"
)

var update = flag.Bool("update", false, "rewrite studio/src/lib/guides.json and the generated blocks of docs/providers.md")

var (
	guidesJSON = filepath.Join("..", "..", "..", "studio", "src", "lib", "guides.json")
	providerMD = filepath.Join("..", "..", "..", "docs", "providers.md")
	kinds      = []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod}
)

func mustFor(t *testing.T, k provision.Kind, o Options) provision.SetupGuide {
	t.Helper()
	g, err := For(k, o)
	if err != nil {
		t.Fatalf("%s: %v", k, err)
	}
	return g
}

func lookup(t *testing.T, g provision.SetupGuide, label string) provision.GuideField {
	t.Helper()
	for _, f := range g.Fields {
		if f.Label == label {
			return f
		}
	}
	t.Fatalf("%s: no field %q", g.Kind, label)
	return provision.GuideField{}
}

func field(t *testing.T, g provision.SetupGuide, label string) string {
	t.Helper()
	return lookup(t, g, label).Value
}

func note(t *testing.T, g provision.SetupGuide, label string) string {
	t.Helper()
	return lookup(t, g, label).Note
}

// The values come from the spec bodies (runpod.md §13, vast.md §8,
// quickpod.md §9) and PLAN §10.
func TestGuideFields(t *testing.T) {
	rp := mustFor(t, provision.RunPod, Options{})
	for label, want := range map[string]string{
		"Container Image": provision.DefaultImage, "Container Disk": "100", "Expose HTTP Ports": "8675",
		"Visibility": "Private", "Name": TemplateName,
	} {
		if got := field(t, rp, label); got != want {
			t.Errorf("runpod %s = %q, want %q", label, got, want)
		}
	}
	if rp.EnvBlock != "AI_TOOLKIT_AUTH="+TokenPlaceholder+"\nAITK_CORS_ORIGINS="+OriginPlaceholder || rp.DockerOptions != "" {
		t.Errorf("runpod env block %q, docker options %q", rp.EnvBlock, rp.DockerOptions)
	}
	if i := slices.IndexFunc(rp.Env, func(e provision.GuideEnv) bool { return e.Name == "HF_TOKEN" }); i < 0 || rp.Env[i].Value != "{{ RUNPOD_SECRET_hf_token }}" || !rp.Env[i].Secret {
		t.Errorf("runpod HF_TOKEN: %+v", rp.Env)
	}
	if rp.URLPattern != "https://<pod id>-8675.proxy.runpod.net" || !strings.Contains(note(t, rp, "Container Disk"), "wiped when the pod stops") {
		t.Errorf("runpod url %q / wipe note missing", rp.URLPattern)
	}

	va := mustFor(t, provision.Vast, Options{})
	if va.DockerOptions != "-p 8675:8675 -e OPEN_BUTTON_PORT=8675 -e AITK_CORS_ORIGINS="+OriginPlaceholder || va.EnvBlock != "" {
		t.Errorf("vast docker options %q", va.DockerOptions)
	}
	for label, want := range map[string]string{
		"Image Path:Tag": provision.DefaultImage, "Select Launch Mode": "docker ENTRYPOINT", "Disk Space": "100",
		"Extra Filters": "direct_port_count >= 1 cuda_max_good >= 13.0 compute_cap >= 800", "Template Visibility": "Private",
	} {
		if got := field(t, va, label); got != want {
			t.Errorf("vast %s = %q, want %q", label, got, want)
		}
	}
	if va.URLPattern != "http://<ip>:<port>" || !strings.Contains(va.Steps[0], "Account → Environment Variables") || !strings.Contains(va.Steps[0], "AI_TOOLKIT_AUTH = "+TokenPlaceholder) {
		t.Errorf("vast url %q, first step %q", va.URLPattern, va.Steps[0])
	}
	// PLAN §10: the private-template route too, and a warning that account
	// env also reaches the servers the studio creates with their own token
	if !strings.Contains(va.Steps[0], "-e AI_TOOLKIT_AUTH="+TokenPlaceholder) || !strings.Contains(va.Steps[0], "servers the studio creates") {
		t.Errorf("vast first step lacks the template route or the account-env warning: %q", va.Steps[0])
	}

	qp := mustFor(t, provision.QuickPod, Options{})
	// both are required: the port, and a fallback token so the server never boots open
	if qp.DockerOptions != "-p 8675:8675 -e AI_TOOLKIT_AUTH="+TokenPlaceholder+" -e AITK_CORS_ORIGINS="+OriginPlaceholder {
		t.Errorf("quickpod docker options %q", qp.DockerOptions)
	}
	for label, want := range map[string]string{
		"Template Type": "GPU", "Runtime": "Pod", "Docker Image Path": provision.DefaultImage, "Disk Space": "100",
		"Launch Mode": "Docker Entrypoint", "Template Name": TemplateName,
	} {
		if got := field(t, qp, label); got != want {
			t.Errorf("quickpod %s = %q, want %q", label, got, want)
		}
	}
	if qp.URLPattern != "https://<pod id>-8675.proxy.quickpod.io/" || !strings.Contains(strings.Join(qp.Steps, " "), "http://<ip>:<port>") {
		t.Errorf("quickpod url %q", qp.URLPattern)
	}

	for _, k := range kinds {
		g := mustFor(t, k, Options{})
		if g.Kind != k || g.Port != provision.Port || g.Image != provision.DefaultImage || g.DiskGB != provision.DefaultDiskGB || g.Token != "" {
			t.Errorf("%s header: %+v", k, g)
		}
		if g.Steps[len(g.Steps)-1] != LastStep {
			t.Errorf("%s: last step %q", k, g.Steps[len(g.Steps)-1])
		}
		// the steps are actions in order: the one before LastStep is where
		// the server is running and its URL is found, not a closing note
		if n := len(g.Steps); n < 2 || !strings.Contains(g.Steps[n-2], "running") {
			t.Errorf("%s: step before the last is %q, want the running step", k, g.Steps[n-2])
		}
		if g.ConsoleURL == "" || g.KeysURL == "" || len(g.KeyScopes) == 0 {
			t.Errorf("%s: console, keys or scopes missing", k)
		}
	}
}

// A hand-made server is only as good as the host picked for it: every guide
// says, in the step where a host is picked, the image's CUDA floor, driver
// branch and oldest GPU, and the values are the default image's.
func TestGuideHostRequirements(t *testing.T) {
	req, ok := imagecheck.FallbackRequirements(imagecheck.DefaultRepo)
	if !ok {
		t.Fatal("no fallback requirements for the default image")
	}
	if req.CUDAMin != HostCUDAMin || req.DriverMin != HostDriverMin || len(req.FlashAttnSM) == 0 || slices.Min(req.FlashAttnSM) != HostSMMin {
		t.Errorf("guide host needs CUDA %s, driver %d, sm_%d; the image needs CUDA %s, driver %d, flash attention from sm_%d",
			HostCUDAMin, HostDriverMin, HostSMMin, req.CUDAMin, req.DriverMin, slices.Min(req.FlashAttnSM))
	}
	pick := map[provision.Kind]string{provision.RunPod: "Deploy On-Demand", provision.Vast: "RENT", provision.QuickPod: "GPU Search"}
	for _, k := range kinds {
		g := mustFor(t, k, Options{})
		i := slices.IndexFunc(g.Steps, func(s string) bool { return strings.Contains(s, pick[k]) })
		if i < 0 {
			t.Fatalf("%s: no host-picking step (%q)", k, pick[k])
		}
		for _, want := range []string{"CUDA 13.0 or newer (580+)", "RTX 30-series, A100 or newer GPU", "still bills"} {
			if !strings.Contains(g.Steps[i], want) {
				t.Errorf("%s: host-picking step lacks %q: %q", k, want, g.Steps[i])
			}
		}
	}
	rp := mustFor(t, provision.RunPod, Options{})
	if !strings.Contains(strings.Join(rp.Steps, " "), "CUDA version filter to 13.0") {
		t.Errorf("runpod: no CUDA version filter in %q", rp.Steps)
	}
	va := mustFor(t, provision.Vast, Options{})
	if n := note(t, va, "Extra Filters"); !strings.Contains(n, "CUDA 13.0 or newer (580+)") {
		t.Errorf("vast Extra Filters note %q", n)
	}
	qp := mustFor(t, provision.QuickPod, Options{})
	if !strings.Contains(strings.Join(qp.Steps, " "), "max CUDA of 13.0 or newer") {
		t.Errorf("quickpod: no max CUDA in %q", qp.Steps)
	}
}

// The guide's keys page and scopes are the adapter's, so the studio's key
// form and the manual guide never disagree.
func TestGuideMatchesAdapters(t *testing.T) {
	t.Setenv("AITK_PROVISION_PROVIDERS", "")
	for _, k := range kinds {
		c, ok := provision.Lookup(k)
		if !ok {
			t.Fatalf("%s adapter not registered", k)
		}
		info, g := c.Info(), mustFor(t, k, Options{})
		if info.KeysURL != g.KeysURL || !slices.Equal(info.KeyScopes, g.KeyScopes) {
			t.Errorf("%s: guide keys %s %v, adapter %s %v", k, g.KeysURL, g.KeyScopes, info.KeysURL, info.KeyScopes)
		}
	}
}

func TestGuideOptions(t *testing.T) {
	tok := provision.NewToken()
	o := Options{Image: "ghcr.io/geocine/aitoolkit-go:abc1234", DiskGB: 150, CORSOrigin: "https://studio.example.com,http://localhost:5174", Token: tok}
	for _, k := range kinds {
		g := mustFor(t, k, o)
		b, _ := json.Marshal(g)
		s := string(b)
		if strings.Contains(s, TokenPlaceholder) || strings.Contains(s, OriginPlaceholder) || strings.Contains(s, provision.DefaultImage) {
			t.Errorf("%s: placeholder or default left: %s", k, s)
		}
		if g.Token != tok || !strings.Contains(s, "AI_TOOLKIT_AUTH="+tok) && !strings.Contains(s, "AI_TOOLKIT_AUTH = "+tok) {
			t.Errorf("%s: token not filled in", k)
		}
		if g.DiskGB != 150 || !strings.Contains(s, `"value":"150"`) || !strings.Contains(s, "studio.example.com,http://localhost:5174") {
			t.Errorf("%s: disk or origin not filled in: %s", k, s)
		}
	}
	if g := mustFor(t, provision.Vast, Options{CORSOrigin: "*"}); !strings.HasSuffix(g.DockerOptions, "AITK_CORS_ORIGINS=*") {
		t.Errorf("origin *: %q", g.DockerOptions)
	}

	for name, bad := range map[string]Options{
		"small disk":      {DiskGB: 10},
		"huge disk":       {DiskGB: 5000},
		"image space":     {Image: "ghcr.io/x y"},
		"image shell":     {Image: "ghcr.io/x;rm"},
		"origin path":     {CORSOrigin: "https://studio.example.com/app"},
		"origin scheme":   {CORSOrigin: "ftp://studio.example.com"},
		"origin space":    {CORSOrigin: "https://a.example, https://b.example"},
		"origin quote":    {CORSOrigin: `https://a.example"`},
		"token space":     {Token: "aitk_abc def ghi"},
		"token shell":     {Token: "aitk_abcdef$(id)"},
		"token too short": {Token: "abc"},
	} {
		if _, err := For(provision.QuickPod, bad); provision.ClassOf(err) != provision.ClassInvalid {
			t.Errorf("%s: err %v, want invalid", name, err)
		}
	}
	if _, err := For("lambda", Options{}); provision.ClassOf(err) != provision.ClassNotFound {
		t.Errorf("unknown kind: %v", err)
	}
	if code, _, _, ok := provision.V1(func() error { _, err := For(provision.RunPod, Options{DiskGB: 1}); return err }()); !ok || code != "invalid_request" {
		t.Errorf("v1 code %q", code)
	}
}

// Our own sentences are sentence case: once the console labels, the field
// values and a few names are taken out, a capitalised word may only start a
// sentence.
func TestSentenceCase(t *testing.T) {
	names := []string{"Hugging Face", "RunPod", "QuickPod", "Vast", "NVIDIA", "Servers", "Pod"}
	for _, k := range kinds {
		g := mustFor(t, k, Options{})
		allowed := append(slices.Clone(ConsoleLabels(k)), names...)
		var sentences []string
		sentences = append(sentences, g.Steps...)
		for _, f := range g.Fields {
			if !slices.Contains(ConsoleLabels(k), f.Label) {
				t.Errorf("%s: field label %q is not a listed console label", k, f.Label)
			}
			allowed = append(allowed, f.Value) // console values, verbatim ("Private", "Docker Entrypoint")
			sentences = append(sentences, f.Note)
		}
		for _, e := range g.Env {
			sentences = append(sentences, e.Note)
		}
		slices.SortFunc(allowed, func(a, b string) int { return len(b) - len(a) }) // longest first
		for _, s := range sentences {
			stripped := s
			for _, a := range allowed {
				stripped = strings.ReplaceAll(stripped, a, "#")
			}
			for _, w := range midSentenceCaps(stripped) {
				t.Errorf("%s: %q is capitalised mid-sentence in %q", k, w, s)
			}
		}
	}
}

// The lint itself: a lone capitalised word mid-sentence is caught, a label
// or a sentence start is not.
func TestSentenceCaseLint(t *testing.T) {
	for s, want := range map[string]string{
		"Open the Billing page and click Next.": "Billing Next",
		"Open the page. Click next: Then wait.": "",
		"Click RENT and wait for the GPU.":      "",
	} {
		if got := strings.Join(midSentenceCaps(s), " "); got != want {
			t.Errorf("%q: flagged %q, want %q", s, got, want)
		}
	}
}

var (
	capWord       = regexp.MustCompile(`\b[A-Z][a-z]+\b`)
	sentenceStart = regexp.MustCompile(`(^|[.:!?] )$`)
)

// midSentenceCaps lists the capitalised words in s that do not start a
// sentence (at the start, or after ". ", ": ", "! " or "? ").
func midSentenceCaps(s string) []string {
	var out []string
	for _, m := range capWord.FindAllStringIndex(s, -1) {
		if !sentenceStart.MatchString(s[:m[0]]) {
			out = append(out, s[m[0]:m[1]])
		}
	}
	return out
}

// studio/src/lib/guides.json is All(placeholders); -update regenerates it.
func TestGuidesJSON(t *testing.T) {
	all, err := All(Options{})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(all); err != nil {
		t.Fatal(err)
	}
	if *update {
		if err := os.WriteFile(guidesJSON, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(guidesJSON)
	if err != nil {
		t.Fatalf("%v (run go test ./internal/provision/guide -update)", err)
	}
	if !bytes.Equal(bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n")), buf.Bytes()) {
		t.Errorf("%s is stale: run go test ./internal/provision/guide -update", guidesJSON)
	}
}

// docs/providers.md carries the manual steps and the "not verified" list
// generated from the same code; -update rewrites the marked blocks.
func TestProvidersDoc(t *testing.T) {
	t.Setenv("AITK_PROVISION_PROVIDERS", "")
	raw, err := os.ReadFile(providerMD)
	if err != nil {
		t.Fatal(err)
	}
	doc := strings.ReplaceAll(string(raw), "\r\n", "\n")
	blocks := map[string]string{}
	var unconfirmed strings.Builder
	for _, k := range kinds {
		blocks["guide:"+string(k)] = Markdown(mustFor(t, k, Options{}))
		c, _ := provision.Lookup(k)
		for _, s := range c.Info().Unconfirmed {
			unconfirmed.WriteString("- " + provision.Label(k) + ": " + s + ".\n")
		}
	}
	blocks["unconfirmed"] = unconfirmed.String()

	got := doc
	for name, body := range blocks {
		open, end := "<!-- "+name+" -->\n", "<!-- /"+name+" -->"
		i, j := strings.Index(got, open), strings.Index(got, end)
		if i < 0 || j < i {
			t.Fatalf("%s lacks the %q ... %q block", providerMD, open, end)
		}
		got = got[:i+len(open)] + body + got[j:]
	}
	if *update && got != doc {
		if err := os.WriteFile(providerMD, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		doc = got
	}
	if got != doc {
		t.Errorf("%s is stale: run go test ./internal/provision/guide -update", providerMD)
	}
	for _, f := range mockkit.FaultNames() {
		if !strings.Contains(doc, "`"+f+"`") {
			t.Errorf("%s does not list the fault key %s", providerMD, f)
		}
	}
}
