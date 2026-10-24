package registrymock

import (
	"embed"
	"encoding/json"
	"maps"
	"slices"
	"strings"
)

// testdata holds public registry answers saved during development (no IPs,
// no tokens): the ostris/aitoolkit amd64 manifest and a config of the same
// build, and geocine/aitoolkit:latest and :0.11.1 manifests with their
// configs.
//
//go:embed testdata/*.json
var testdata embed.FS

// Fixture facts other packages assert on.
const (
	// DefaultRepo is the server image's repository on GHCR.
	DefaultRepo = "geocine/aitoolkit-go"
	// Revision is the build the default fixtures publish as :<sha>, :latest,
	// :cu130 and :0.13.34.
	Revision = "f53a407"
	// RevisionFull is its 40-character commit.
	RevisionFull = "f53a4071c2b3d4e5f60718293a4b5c6d7e8f9012"
	// Version is the upstream version.py the fixtures carry.
	Version = "0.13.34"
	// BaseUnpacked is the measured unpacked size of the fork base.
	BaseUnpacked = "13689223168"
)

// GoLayerSizes are the layers ui-go/docker/Dockerfile adds above the base:
// aitk-server, web/dist, studio/dist, start-go.sh and the start.sh edit.
var GoLayerSizes = []int64{24_812_345, 1_203_456, 2_345_678, 1_024, 8_192}

func read(name string) []byte {
	b, err := testdata.ReadFile("testdata/" + name)
	if err != nil {
		panic(err)
	}
	return b
}

// layersOf is the layer list of a saved manifest.
func layersOf(name string) []Layer {
	var m struct {
		Layers []struct {
			Digest string `json:"digest"`
			Size   int64  `json:"size"`
		} `json:"layers"`
	}
	if err := json.Unmarshal(read(name), &m); err != nil {
		panic(err)
	}
	out := make([]Layer, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = Layer{l.Digest, l.Size}
	}
	return out
}

// GoLabels are the labels build_and_push.sh writes on a cu130 build
// (IMAGE-PLAN §1.3).
func GoLabels() map[string]string {
	return map[string]string{
		"org.opencontainers.image.title":            "aitoolkit-go",
		"org.opencontainers.image.version":          Version,
		"org.opencontainers.image.revision":         RevisionFull,
		"org.opencontainers.image.created":          "2026-10-24T09:00:00Z",
		"org.opencontainers.image.source":           "https://github.com/geocine/ai-toolkit",
		"org.opencontainers.image.base.name":        "nvidia/cuda:13.0.3-runtime-ubuntu24.04",
		"io.github.geocine.aitk.schema":             "1",
		"io.github.geocine.aitk.server":             "go",
		"io.github.geocine.aitk.api":                "1",
		"io.github.geocine.aitk.revision":           Revision,
		"io.github.geocine.aitk.upstream":           "c22f21fa958a",
		"io.github.geocine.aitk.flavor":             "cu130",
		"io.github.geocine.aitk.cuda":               "13.0",
		"io.github.geocine.aitk.driver_min":         "580",
		"io.github.geocine.aitk.torch":              "2.13.0+cu130",
		"io.github.geocine.aitk.sm":                 "75,80,86,90,100,120",
		"io.github.geocine.aitk.ptx":                "",
		"io.github.geocine.aitk.sm.flash_attn":      "80,90,100,120",
		"io.github.geocine.aitk.platform":           "linux/amd64",
		"io.github.geocine.aitk.port":               "8675",
		"io.github.geocine.aitk.health":             "/v1/health",
		"io.github.geocine.aitk.size.base_unpacked": BaseUnpacked,
	}
}

// goConfig is the geocine/aitoolkit:latest config turned into a Go-server
// image: labels (nil keeps the base's), extra env entries, the history line
// of the aitk-server COPY, and the given creation time.
func goConfig(labels map[string]string, env map[string]string, created string) []byte {
	var cfg map[string]any
	if err := json.Unmarshal(read("geocine_latest_config.json"), &cfg); err != nil {
		panic(err)
	}
	c := cfg["config"].(map[string]any)
	if labels != nil {
		merged := map[string]any{}
		for k, v := range c["Labels"].(map[string]any) {
			merged[k] = v
		}
		for k, v := range labels {
			merged[k] = v
		}
		c["Labels"] = merged
	}
	if len(env) > 0 {
		var out []any
		for _, kv := range c["Env"].([]any) {
			name, _, _ := strings.Cut(kv.(string), "=")
			if _, replaced := env[name]; !replaced {
				out = append(out, kv)
			}
		}
		for _, k := range slices.Sorted(maps.Keys(env)) {
			out = append(out, k+"="+env[k])
		}
		c["Env"] = out
	}
	history := cfg["history"].([]any)
	for _, h := range []string{
		"COPY /out/aitk-server /app/ai-toolkit/ui-go/aitk-server # buildkit",
		"COPY /web/dist /app/ai-toolkit/ui-go/web/dist # buildkit",
		"COPY /studio/dist /app/ai-toolkit/ui-go/studio/dist # buildkit",
		"COPY ui-go/docker/start-go.sh /tmp/start-go.sh # buildkit",
		"RUN /bin/sh -c sed -i 's/\\r$//' /start.sh /tmp/start-go.sh # buildkit",
	} {
		history = append(history, map[string]any{"created": created, "created_by": h, "comment": "buildkit.dockerfile.v0"})
	}
	cfg["history"] = history
	cfg["created"] = created
	b, err := json.Marshal(cfg)
	if err != nil {
		panic(err)
	}
	return b
}

// Fixtures are the images simulate mode and the tests serve:
//   - ghcr.io/geocine/aitoolkit-go :f53a407, :latest, :cu130, :0.13.34 (one
//     labelled cu130 build), :oldsha1 (a Go image without labels, bare
//     manifest), :cu126test (a synthetic cu126 flavor), :arm64only;
//   - ghcr.io/geocine/aitoolkit :latest (CUDA 13.0.3) and :0.11.1 (CUDA
//     12.8.1): the Next.js images, not the Go server;
//   - docker.io/ostris/aitoolkit:latest (upstream, not the Go server);
//   - docker.io/library/alpine:3.20 (no CUDA at all, a manifest list).
//
// The Go-server config plants HF_TOKEN and AWS_SECRET_ACCESS_KEY env entries,
// which the check must never echo.
func Fixtures() []Image {
	base := layersOf("geocine_latest_manifest.json")
	goLayers := append([]Layer(nil), base...)
	for i, size := range GoLayerSizes {
		goLayers = append(goLayers, Layer{digestOf([]byte("go-layer|" + Revision + "|" + string(rune('a'+i)))), size})
	}
	secrets := map[string]string{"HF_TOKEN": "hf_planted_secret", "AWS_SECRET_ACCESS_KEY": "planted-secret"}
	goCfg := goConfig(GoLabels(), secrets, "2026-10-24T09:00:00Z")

	cu126 := GoLabels()
	for k, v := range map[string]string{
		"io.github.geocine.aitk.revision": "c126c12", "io.github.geocine.aitk.flavor": "cu126",
		"io.github.geocine.aitk.cuda": "12.6", "io.github.geocine.aitk.driver_min": "560",
		"io.github.geocine.aitk.torch": "2.13.0+cu126", "io.github.geocine.aitk.sm": "50,60,70,75,80,86,90",
		"io.github.geocine.aitk.sm.flash_attn": "80,90",
	} {
		cu126[k] = v
	}
	cu126Cfg := goConfig(cu126, map[string]string{
		"NVIDIA_REQUIRE_CUDA": "cuda>=12.6 brand=unknown,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=tesla,driver>=535,driver<536 brand=tesla,driver>=550,driver<551",
		"CUDA_VERSION":        "12.6.3",
	}, "2026-10-20T09:00:00Z")

	var out []Image
	for _, tag := range []string{Revision, "latest", "cu130", Version} {
		out = append(out, Image{Repo: DefaultRepo, Tag: tag, Config: goCfg, Layers: goLayers, Index: true})
	}
	return append(out,
		Image{Repo: DefaultRepo, Tag: "oldsha1", Config: goConfig(nil, nil, "2026-07-01T09:00:00Z"), Layers: goLayers},
		Image{Repo: DefaultRepo, Tag: "cu126test", Config: cu126Cfg, Layers: goLayers, Index: true},
		Image{Repo: DefaultRepo, Tag: "arm64only", Platforms: []string{"linux/arm64"}, Config: goCfg, Layers: goLayers, Index: true},
		Image{Repo: "geocine/aitoolkit", Tag: "latest", Config: read("geocine_latest_config.json"), Layers: base, Index: true},
		Image{Repo: "geocine/aitoolkit", Tag: "0.11.1", Config: read("geocine_0.11.1_config.json"),
			Layers: layersOf("geocine_0.11.1_manifest.json"), Index: true},
		Image{Host: Hub, Repo: "ostris/aitoolkit", Tag: "latest", Config: read("ostris_config.json"),
			Layers: layersOf("ostris_latest_manifest.json"), Index: true},
		Image{Host: Hub, Repo: "library/alpine", Tag: "3.20", Platforms: []string{"linux/amd64", "linux/arm64/v8", "linux/arm/v7"},
			Config:     []byte(`{"architecture":"amd64","os":"linux","created":"2026-09-06T22:20:07Z","config":{"Env":["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],"Cmd":["/bin/sh"]},"rootfs":{"type":"layers","diff_ids":["sha256:63ca1fbb43ae5034640e5e6cb3e083e05c290072c5366fcaa9d62435a4cced85"]},"history":[{"created":"2026-09-06T22:20:07Z","created_by":"ADD alpine-minirootfs-3.20.8-x86_64.tar.gz / # buildkit"}]}`),
			LayerSizes: []int64{3_626_545}, Index: true, SchemaV2List: true, NoAttestation: true},
	)
}
