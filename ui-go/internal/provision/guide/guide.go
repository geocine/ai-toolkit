// Package guide is the single source of the manual template instructions
// (GET /v1/providers/{kind}/setup, studio/src/lib/guides.json, docs/providers.md).
//
// Field labels are copied verbatim from each provider's console (spec:
// runpod.md §13, vast.md §8, quickpod.md §9); the sentences around them are
// ours, in sentence case. guide_test.go keeps guides.json and the generated
// blocks of docs/providers.md equal to All(Options{}).
package guide

import (
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/geocine/aitoolkit/internal/provision"
)

// Placeholders left in a guide when Options has no token or origin. The
// studio replaces them in the browser (studio/src/lib/guides.ts), so a token
// never travels to the server.
const (
	TokenPlaceholder  = "<a long random token>"
	OriginPlaceholder = "<your studio origin>"
)

// TemplateName is the template name every guide suggests.
const TemplateName = "aitk headless"

// LastStep closes every guide.
const LastStep = "When it shows running, copy its URL and add it on the Servers page with the same token."

type Options struct {
	Image      string // default provision.DefaultImage
	DiskGB     int    // default provision.DefaultDiskGB
	CORSOrigin string // "" -> placeholder "<your studio origin>"
	Token      string // "" -> placeholder "<a long random token>"
}

var (
	tokenRe  = regexp.MustCompile(`^[A-Za-z0-9._~-]{8,512}$`)
	originRe = regexp.MustCompile(`^https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?$|^https?://\[[0-9A-Fa-f:.]+\](:[0-9]{1,5})?$`)
)

// normalize fills the defaults and refuses values that would break the
// copied env block or docker options (whitespace, quotes, shell characters).
func normalize(k provision.Kind, o Options) (Options, error) {
	o.Image = strings.TrimSpace(o.Image)
	if o.Image == "" {
		o.Image = provision.DefaultImage
	}
	if len(o.Image) > 1024 || strings.IndexFunc(o.Image, unicode.IsSpace) >= 0 || strings.ContainsAny(o.Image, "\"'`$\\;|&<>") {
		return o, provision.Errorf(k, provision.ClassInvalid, 0, "image must be an image reference without spaces")
	}
	switch {
	case o.DiskGB == 0:
		o.DiskGB = provision.DefaultDiskGB
	case o.DiskGB < 20 || o.DiskGB > 2000:
		return o, provision.Errorf(k, provision.ClassInvalid, 0, "disk_gb must be 20 to 2000")
	}
	o.CORSOrigin = strings.TrimSpace(o.CORSOrigin)
	if o.CORSOrigin == "" {
		o.CORSOrigin = OriginPlaceholder
	} else {
		for _, p := range strings.Split(o.CORSOrigin, ",") {
			if p != "*" && !validOrigin(p) {
				return o, provision.Errorf(k, provision.ClassInvalid, 0, "cors_origin must be * or scheme://host[:port], comma separated")
			}
		}
	}
	if o.Token != "" && !tokenRe.MatchString(o.Token) {
		return o, provision.Errorf(k, provision.ClassInvalid, 0, "the token may hold only letters, digits and ._~-")
	}
	return o, nil
}

func validOrigin(s string) bool {
	if !originRe.MatchString(s) {
		return false
	}
	u, err := url.Parse(s)
	return err == nil && u.Host != "" && u.User == nil
}

// For builds the manual setup guide for k.
func For(k provision.Kind, o Options) (provision.SetupGuide, error) {
	o, err := normalize(k, o)
	if err != nil {
		return provision.SetupGuide{}, err
	}
	token := o.Token
	if token == "" {
		token = TokenPlaceholder
	}
	var g provision.SetupGuide
	switch k {
	case provision.RunPod:
		g = runpodGuide(o, token)
	case provision.Vast:
		g = vastGuide(o, token)
	case provision.QuickPod:
		g = quickpodGuide(o, token)
	default:
		return provision.SetupGuide{}, provision.Errorf(k, provision.ClassNotFound, 0, "unknown provider")
	}
	g.Kind, g.Image, g.Port, g.DiskGB = k, o.Image, provision.Port, o.DiskGB
	g.Steps = append(g.Steps, LastStep)
	g.Token = o.Token
	return g, nil
}

// All is the guide of every kind (studio/src/lib/guides.json is All(Options{})).
func All(o Options) (map[provision.Kind]provision.SetupGuide, error) {
	out := map[provision.Kind]provision.SetupGuide{}
	for _, k := range []provision.Kind{provision.RunPod, provision.Vast, provision.QuickPod} {
		g, err := For(k, o)
		if err != nil {
			return nil, err
		}
		out[k] = g
	}
	return out, nil
}

// ConsoleLabels are the console labels (verbatim) and names the guide for k
// may write in Title Case. Everything else it says is sentence case.
func ConsoleLabels(k provision.Kind) []string { return labels[k] }

var labels = map[provision.Kind][]string{
	provision.RunPod: {
		"Templates", "New Template", "Name", "Template Type", "Compute type", "Visibility", "Container Image",
		"Container Registry Credentials", "Container Start Command", "Container Disk", "Expose HTTP Ports",
		"Environment Variables", "Save Template", "Settings", "Secrets", "Pods", "Deploy", "Change Template",
		"Deploy On-Demand", "Connect", "HTTP Service", "Hugging Face",
	},
	provision.Vast: {
		"Account", "Environment Variables", "Templates", "+ New", "Config", "Template Name", "Image Path:Tag",
		"Select Launch Mode", "Extra Filters", "Disk Space", "Template Visibility", "Docker Options",
		"Create & Use", "RENT", "IP Port Info", "Open", "Hugging Face",
	},
	provision.QuickPod: {
		"Settings", "Templates", "Create New Template", "Template Type", "Runtime", "Template Name",
		"Docker Image Path", "Disk Space", "Launch Mode", "Docker Entrypoint", "Private Docker Repository",
		"Docker Entrypoint Run Options", "On Start Script", "Docker Options", "Create Template", "GPU Search",
		"My Templates", "Create", "Pods", "Connect", "Exposed Ports", "HTTPS", "HTTP", "Secure API keys",
		"Pods read", "Pods write", "Templates read", "Account read", "Profile read",
	},
}

// ---- RunPod (spec: runpod.md §7, §13) ----

func runpodGuide(o Options, token string) provision.SetupGuide {
	env := []provision.GuideEnv{
		{Name: "AI_TOOLKIT_AUTH", Value: token, Secret: true, Note: "The server asks for it on every request. A RunPod secret works too: {{ RUNPOD_SECRET_aitk_token }}."},
		{Name: "AITK_CORS_ORIGINS", Value: o.CORSOrigin, Note: "Lets your studio call the server from the browser."},
		{Name: "HF_TOKEN", Value: "{{ RUNPOD_SECRET_hf_token }}", Secret: true, Note: "Optional, for gated base models: save your Hugging Face token as the RunPod secret hf_token first."},
	}
	return provision.SetupGuide{
		Env:      env,
		EnvBlock: envBlock(env[:2]),
		Fields: []provision.GuideField{
			{Label: "Name", Value: TemplateName},
			{Label: "Visibility", Value: "Private", Note: "The template holds your token."},
			{Label: "Container Image", Value: o.Image, Note: "A public image, so Container Registry Credentials stay empty."},
			{Label: "Container Disk", Value: strconv.Itoa(o.DiskGB), Note: "GB for models, datasets and checkpoints. It is wiped when the pod stops, so keep results on a remote and terminate the pod when you are done."},
			{Label: "Expose HTTP Ports", Value: strconv.Itoa(provision.Port)},
		},
		Steps: []string{
			"Sign in at console.runpod.io, open Templates and click New Template.",
			"Set Template Type to Pod and Compute type to NVIDIA GPU, then fill in the fields above. Leave Container Start Command empty: the image starts the server itself.",
			"Under Environment Variables, add the variables above.",
			"For gated base models, also add HF_TOKEN = {{ RUNPOD_SECRET_hf_token }} after saving your Hugging Face token as the secret hf_token under Settings → Secrets.",
			"Click Save Template.",
			"Open Pods, click Deploy and pick a GPU. Click Change Template, choose " + TemplateName + " and click Deploy On-Demand.",
			"When the pod shows running, open Connect → HTTP Service: the " + strconv.Itoa(provision.Port) + " row links to https://<pod id>-" + strconv.Itoa(provision.Port) + ".proxy.runpod.net.",
		},
		URLPattern: "https://<pod id>-" + strconv.Itoa(provision.Port) + ".proxy.runpod.net",
		ConsoleURL: "https://console.runpod.io",
		KeysURL:    "https://console.runpod.io/user/settings",
		KeyScopes:  []string{"pods:read", "pods:write", "catalog:read", "templates:read"},
	}
}

// ---- Vast.ai (spec: vast.md §5.7, §8) ----

func vastGuide(o Options, token string) provision.SetupGuide {
	port := strconv.Itoa(provision.Port)
	env := []provision.GuideEnv{
		{Name: "AI_TOOLKIT_AUTH", Value: token, Secret: true, Note: "Under Account → Environment Variables, so it stays out of the template. Or, with Template Visibility Private, as -e AI_TOOLKIT_AUTH=" + token + " in Docker Options. An account-wide value also reaches the servers the studio creates for you."},
		{Name: "OPEN_BUTTON_PORT", Value: port, Note: "Points the instance's Open button at the server."},
		{Name: "AITK_CORS_ORIGINS", Value: o.CORSOrigin, Note: "Lets your studio call the server from the browser."},
		{Name: "HF_TOKEN", Value: "<your Hugging Face token>", Secret: true, Note: "Optional, for gated base models. Also under Account → Environment Variables."},
	}
	return provision.SetupGuide{
		Env:           env,
		DockerOptions: "-p " + port + ":" + port + " " + envFlags(env[1:3]),
		Fields: []provision.GuideField{
			{Label: "Template Name", Value: TemplateName},
			{Label: "Image Path:Tag", Value: o.Image, Note: "The full registry path. Another tag works too."},
			{Label: "Select Launch Mode", Value: "docker ENTRYPOINT", Note: "Runs the image as built, which starts the server."},
			{Label: "Extra Filters", Value: "direct_port_count >= 1", Note: "Only hosts that can give port " + port + " a public port."},
			{Label: "Disk Space", Value: strconv.Itoa(o.DiskGB), Note: "GB, fixed for the life of the instance."},
			{Label: "Template Visibility", Value: "Private"},
		},
		Steps: []string{
			// Vast puts account env into every instance, including the ones the
			// studio launches with their own AI_TOOLKIT_AUTH; which of the two
			// wins is not confirmed, hence the warning.
			"Open Account → Environment Variables (cloud.vast.ai/account/) and add AI_TOOLKIT_AUTH = " + token + ". Vast puts it into every instance, so it stays out of the template. Or, with Template Visibility Private, add -e AI_TOOLKIT_AUTH=" + token + " to Docker Options instead. An account-wide AI_TOOLKIT_AUTH also reaches the servers the studio creates for you, which get their own token, so use the template if you also create servers from the studio. Add HF_TOKEN under Account → Environment Variables for gated base models.",
			"Open Templates (cloud.vast.ai/templates/), click + New and stay on the Config tab.",
			"Fill in the fields above and paste the docker options into Docker Options.",
			"Click Create & Use, pick an offer and click RENT.",
			"When the instance shows running, open IP Port Info: the line ending in -> " + port + "/tcp gives the URL http://<ip>:<port>. It is plain http only, so the token crosses the internet unencrypted, and a studio served over https cannot call it directly: use the studio the server serves at /studio/.",
		},
		URLPattern: "http://<ip>:<port>",
		ConsoleURL: "https://cloud.vast.ai/templates/",
		KeysURL:    "https://cloud.vast.ai/manage-keys/",
		KeyScopes:  []string{"user_read", "instance_read", "instance_write"},
	}
}

// ---- QuickPod (spec: quickpod.md §5.3, §6.3, §9) ----

func quickpodGuide(o Options, token string) provision.SetupGuide {
	port := strconv.Itoa(provision.Port)
	env := []provision.GuideEnv{
		{Name: "AI_TOOLKIT_AUTH", Value: token, Secret: true, Note: "In the template's docker options, so the server never starts without a token."},
		{Name: "AITK_CORS_ORIGINS", Value: o.CORSOrigin, Note: "Lets your studio call the server from the browser."},
	}
	return provision.SetupGuide{
		Env: env,
		// Both are required: the port, plus a fallback token so the server
		// never boots open if QuickPod ignores per-pod options (PLAN §6.3).
		DockerOptions: "-p " + port + ":" + port + " " + envFlags(env),
		Fields: []provision.GuideField{
			{Label: "Template Type", Value: "GPU"},
			{Label: "Runtime", Value: "Pod", Note: "Not VM."},
			{Label: "Template Name", Value: TemplateName},
			{Label: "Docker Image Path", Value: o.Image, Note: "The tag goes in the path: there is no separate tag field."},
			{Label: "Disk Space", Value: strconv.Itoa(o.DiskGB), Note: "GB. Only the default: you pick the size again at launch."},
			{Label: "Launch Mode", Value: "Docker Entrypoint", Note: "Uses the image's entrypoint, which starts the server."},
		},
		Steps: []string{
			"Sign in at console.quickpod.io and add credit under Settings. QuickPod support is in beta. For the studio to create pods for you later, a key under Settings → Secure API keys needs Pods read, Pods write, Templates read, Account read and Profile read.",
			"Open Templates and click Create New Template.",
			"Fill in the fields above. Leave Private Docker Repository unticked, and leave Docker Entrypoint Run Options and On Start Script empty.",
			"Paste the docker options into Docker Options. Keep both parts: -p " + port + ":" + port + " publishes the port, and AI_TOOLKIT_AUTH keeps the server from starting without a token.",
			"Click Create Template. On GPU Search, pick an offer, choose " + TemplateName + " under My Templates and click Create.",
			"On Pods, wait for running and click Connect. Under Exposed Ports, the " + port + " row has an HTTPS link (https://<pod id>-" + port + ".proxy.quickpod.io/) and an HTTP link (http://<ip>:<port>).",
		},
		URLPattern: "https://<pod id>-" + port + ".proxy.quickpod.io/",
		ConsoleURL: "https://console.quickpod.io/templates",
		KeysURL:    "https://console.quickpod.io/settings",
		KeyScopes:  []string{"pods:read", "pods:write", "templates:read", "account:read", "profile:read"},
	}
}

// envBlock is RunPod's "K=V" lines.
func envBlock(env []provision.GuideEnv) string {
	lines := make([]string, len(env))
	for i, e := range env {
		lines[i] = e.Name + "=" + e.Value
	}
	return strings.Join(lines, "\n")
}

// envFlags is "-e K=V -e K=V" for docker options.
func envFlags(env []provision.GuideEnv) string {
	parts := make([]string, len(env))
	for i, e := range env {
		parts[i] = "-e " + e.Name + "=" + e.Value
	}
	return strings.Join(parts, " ")
}
