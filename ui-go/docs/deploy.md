# Deploying

Production is **one image**, configured only through environment variables:
`ghcr.io/geocine/aitoolkit-go`. It contains upstream's whole toolkit (Python,
CUDA, torch, extensions), the Go server on port **8675** (API, `/mcp`, events,
management UI) and its job worker. Nothing else needs to run beside it: no
compose, no database server, no sidecar.

A GPU server is treated as disposable. With a remote configured, every
running job keeps its resumable state there, and the same image started on
another provider picks the job up and continues from the last checkpoint.

## Quick start

```bash
docker run -d --gpus all -p 8675:8675 \
  -e AI_TOOLKIT_AUTH=<long random token> \
  -e AITK_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
  -e AITK_S3_BUCKET=<bucket> \
  -e AITK_S3_ACCESS_KEY_ID=<key id> -e AITK_S3_SECRET_ACCESS_KEY=<secret> \
  -e AITK_SYNC_REMOTE=s3 -e AITK_AUTO_RESUME=1 -e AITK_IMPORT_ON_BOOT=all \
  -e HF_TOKEN=<hf token, for gated base models> \
  ghcr.io/geocine/aitoolkit-go:latest
```

Then point a client at `http://<host>:8675` with `Authorization: Bearer <token>`:
the `/v1` API (`/v1/openapi.json`), MCP at `/mcp`, the console at `/`, or the
studio at `/studio/` (installable on a phone home screen: open it there and
"Add to Home Screen").

## Providers

The container needs a GPU, port 8675 exposed over HTTP, and the env vars.

| Provider | How |
|---|---|
| RunPod | Template: container image above, *Expose HTTP Ports* `8675`, env vars in the template. Reach it at `https://<pod id>-8675.proxy.runpod.net`. `PUBLIC_KEY` enables SSH (upstream's start script). Stop and terminate go through RunPod's API v2. |
| Vast.ai | Image above; *Docker options* `-p 8675:8675 -e ...`. Use the mapped port shown on the instance. |
| QuickPod | Template: *Docker Image Path* above, *Docker Options* `-p 8675:8675 -e AI_TOOLKIT_AUTH=... -e ...`. Use the URL under *Connect*, *Exposed Ports*. |
| Lambda, any VM | `docker run` as in the quick start (NVIDIA Container Toolkit installed). |

Or create one from the studio: docs/providers.md (step-by-step template
guides for all three are there too).

No server yet? Run a controller on your laptop: docs/controller.md.

Disks can be ephemeral: training state lives on the remote. Mount
`/root/.cache/huggingface` only to keep base models across restarts on the
same machine.

## The image

`ghcr.io/geocine/aitoolkit-go` is a **public** GHCR package, so hosts pull it
without a login. There is one flavor, `cu130`: linux/amd64 only, on
`nvidia/cuda:13.0.3-runtime-ubuntu24.04` with torch `2.13.0+cu130`, about
6.5 GB to download and 13.7 GB unpacked.

While the package is missing or private, the studio says "The server image
ghcr.io/geocine/aitoolkit-go isn't published yet, or it's still private." and
GPU servers cannot start it. Offers and the suggested GPU still work: hosts
are judged against the image's built-in requirements (the ones below).

**Tags**
| Tag | Moves | Use |
|---|---|---|
| `:<sha>` | never | One build: the short commit (`git rev-parse --short HEAD`), the same value as `version` in `/v1/health`. GPU servers created from the studio run this tag. |
| `:latest` | every build | The newest build. Fine for `docker run`. A launch from the studio resolves it to its `:<sha>` first, because RunPod caches `:latest`. |
| `:cu130` | every build | The newest build of the cu130 flavor: the same as `latest` today, and the slot a second flavor would sit beside. |
| `:<version>` (`0.13.34`) | every build | Upstream's `version.py`, as a friendly name. Every fork build on that version overwrites it. |
| `:<yymmddHHMM>` | never | The build time, kept for older scripts. |

A hand-made template can pin `:<sha>` to stay on one build. A `:<sha>` is
never moved: `build_and_push.sh --push` refuses to push one that already
exists with a different image unless `ALLOW_RETAG=1`.

**Labels.** The Go layer says what hosts the image runs on. Every value is
read from the freshly built base, never typed by hand. The image check reads
the labels first. An image without them (an older tag, `ostris/aitoolkit`,
your own build) is judged from its `NVIDIA_REQUIRE_CUDA` env and a built-in
table instead.
| Label (`io.github.geocine.aitk.`) | Example | Meaning |
|---|---|---|
| `server` | `go` | Runs the Go server the studio talks to: only such an image can be launched from the studio. |
| `api` | `1` | The `/v1` major version. Another value is a warning. |
| `revision`, `upstream` | `f53a407`, `c22f21fa958a` | The build (its `:<sha>` tag) and the upstream commit of the base. |
| `flavor`, `torch` | `cu130`, `2.13.0+cu130` | torch's CUDA build. |
| `cuda` | `13.0` | The CUDA floor, from the base's `NVIDIA_REQUIRE_CUDA`. Hosts are gated on this. |
| `driver_min` | `580` | The driver branch for that CUDA, for display. |
| `sm`, `ptx` | `75,80,86,90,100,120`, empty | GPU architectures torch has kernels (and PTX) for. |
| `sm.flash_attn` | `80,90,100,120` | The architectures flash-attn has kernels for. |
| `platform`, `port`, `health` | `linux/amd64`, `8675`, `/v1/health` | |
| `size.base_unpacked` | `13689223168` | The base's unpacked bytes, for the disk need. |
| `schema` | `1` | Bumped when a label changes meaning. |

The standard `org.opencontainers.image.*` labels are set too: `title`,
`version` (upstream's version, replacing Ubuntu's `24.04` inherited from the
base), `revision` (the full commit), `created` (the commit time in UTC),
`source` (links the package to `github.com/geocine/ai-toolkit`) and
`base.name`. Read them with
`docker image inspect -f '{{json .Config.Labels}}' ghcr.io/geocine/aitoolkit-go:<sha>`,
or without pulling with `aitk-server image-check` (below).

**Host requirements** (the cu130 build). The server checks these before it
rents a host and again at launch (docs/headless.md, "Images and hosts").
| | Needs | Otherwise |
|---|---|---|
| CPU | amd64 | arm64 hosts (GH200, GB10) are blocked. |
| GPU vendor | NVIDIA | AMD (MI300X) is blocked: there is no ROCm build. |
| GPU | sm_80 or newer: A100, A30, RTX 30-series, RTX A2000-A6000, A10, A40, RTX 40-series, L4, L40, L40S, RTX Ada cards, H100, H200, B200, B300, RTX 50-series, RTX PRO Blackwell | sm_75 (T4, RTX 20-series, GTX 16-series) runs torch but not flash attention and has no native bf16: hidden from offers, a warning when shown. sm_70 and older (V100, Titan V, P100, GTX 10-series) have no kernels and are blocked. The B300 runs the sm_100 kernels and is not yet tested. |
| Driver | CUDA 13.0 or newer: driver branch 580 or newer | Blocked on every GPU, data-center cards too: the image ships CUDA's forward-compatibility libraries, but they are not verified with it. A driver that cannot be read is a warning. On RunPod, RunPod itself places the pod on a CUDA 13.0 host. |
| Disk | 22 GB just for the image (6.5 GB download + 13.7 GB unpacked + 2 GB) | Blocked below that. The disk a launch asks for is (image + model + 15 GB for checkpoints and caches) × 1.1, rounded up to 10 GB, at least 40 GB, and 100 GB when the model is not known (FLUX.1-dev: 80 GB). Less is a warning. |
| Download | nothing | The pull time is estimated from the host's reported speed: about 3 minutes at 1376 Mbps, 25 at 73 Mbps. Over 10 minutes is a warning. |

After a launch, what a pre-check cannot see is checked on the machine: a
pull far slower than predicted (3× the estimate, at least 15 minutes) is
`pull_slow`; a failed pull is classified from the provider's text (tag
missing, private, wrong platform, disk full, driver too old) as
`image_pull_failed`; a ready server whose `version` is not the build asked
for gets a warning (the host had an older copy cached); and the server's
CUDA self-check runs one tiny kernel on every GPU (`cuda_failed` when it
cannot).

### Building and publishing

```bash
ui-go/scripts/build_and_push.sh            # upstream's base, then the Go layer with its labels
ui-go/scripts/build_and_push.sh --push     # push <sha>, <timestamp>, <version>, cu130, latest; then verify <sha>
ui-go/scripts/build_and_push.sh --verify   # verify the pushed <sha> of HEAD again, without building
bash ui-go/scripts/testdata/build_and_push_test.sh   # the script's own tests: fake docker, offline
```

Both builds are linux/amd64. The script probes the base once (torch, its
CUDA and kernel targets, flash-attn) and fails when the base does not look
as the labels expect: no CUDA build of torch, no kernel list, or a CUDA or
flash-attn version it has no table entry for. `IMAGE`, `UPSTREAM_REF`,
`REBUILD_BASE=1`, `BASE_INSTALLER=uv` and `ALLOW_RETAG=1` adjust it (header
of the script).

Publishing the first time:
1. On an amd64 machine with Docker (and buildx, for the `:<sha>` guard), run
   `build_and_push.sh --push`. GHCR creates the package **private**, so the
   verify step fails with "denied" and the settings URL.
2. Open `https://github.com/users/geocine/packages/container/aitoolkit-go/settings`
   and set the visibility to Public.
3. Run `build_and_push.sh --verify`. It runs `image-check --require-go`
   inside the pushed image and must exit 0.

Push again whenever the server changes. A GPU server created from the
studio runs the controller's own build when that `:<sha>` is published, and
the newest build otherwise (it then says so).

### Checking an image

```bash
aitk-server image-check [--require-go] [--platform linux/amd64] [--host-json host.json] [--json] <image>
docker run --rm --entrypoint /app/ai-toolkit/ui-go/aitk-server ghcr.io/geocine/aitoolkit-go:<sha> \
  image-check --require-go ghcr.io/geocine/aitoolkit-go:<sha>
```

It reads the image's manifest, config and labels from the registry (an
anonymous pull, no layers downloaded) and prints what the image needs and
every check. `--host-json` also checks it against a host (`HostFacts`,
docs/headless.md), `--require-go` blocks when the `sm` label is missing, and
`go run . image-check ...` works from `ui-go/`. Exit 0 when nothing blocks,
1 on a block or an error, 2 when anonymous pulls are denied (private or
missing). Only GHCR and Docker Hub are read; other registries are never
contacted.

## Environment

**Access**
| Variable | Meaning |
|---|---|
| `AI_TOOLKIT_AUTH` | Bearer token for `/api`, `/v1` and `/mcp` (admin). Named, revocable keys can be added with `POST /v1/keys`. Without a token and without keys the server is open. |
| `HF_TOKEN` | Hugging Face token passed to trainers (gated base models) and the default for `hf` remotes. |
| `AITK_CORS_ORIGINS` | Origins allowed to call the API from a browser app hosted elsewhere (the studio, a desktop webview): comma list, or `*`. Unset = same origin only. A server with no token and no API key also refuses writes (POST, PUT, PATCH, DELETE) from pages on other origins not listed here, so a website you visit cannot drive it. |

**Remotes** (created or updated on every start)
| Variable | Meaning |
|---|---|
| `AITK_S3_ENDPOINT`, `AITK_S3_BUCKET`, `AITK_S3_ACCESS_KEY_ID`, `AITK_S3_SECRET_ACCESS_KEY` | An S3-compatible remote (Cloudflare R2, AWS S3, MinIO, Backblaze B2, ...). |
| `AITK_S3_REGION` (empty: `auto` for R2, else the bucket's own), `AITK_S3_PREFIX`, `AITK_S3_PATH_STYLE` (`1` for MinIO), `AITK_S3_NAME` (`s3`) | Its options and the name it gets. |
| `AITK_HF_REPO`, `AITK_HF_REPO_TYPE` (`model`), `AITK_HF_PATH`, `AITK_HF_TOKEN` (else `HF_TOKEN`), `AITK_HF_NAME` (`hf`) | A private Hugging Face repo remote (created if missing). |
| `AITK_REMOTES` | Any number of remotes as JSON: `[{"name","kind":"s3"\|"hf"\|"fs","config":{...},"credentials":{...}}]` (shapes in docs/headless.md). |

**Job defaults** (apply to jobs created without a policy, from any API)
| Variable | Meaning |
|---|---|
| `AITK_SYNC_REMOTE` | Remote that keeps jobs portable (checkpoint + optimizer state + dataset + manifest). |
| `AITK_SYNC_KEEP_CHECKPOINTS` | Older checkpoints the sync remote keeps besides the newest: a number, or `all` (default: only the newest). A job's `policy.keep_checkpoints` overrides it. Keep them if you pick the best step after training, since a destroyed pod takes its local copies with it. |
| `AITK_AUTO_UPLOAD_REMOTE` | Remote every new checkpoint is also uploaded to (e.g. a Hugging Face repo for the LoRAs). |
| `AITK_AUTO_RESUME` (`0`) / `AITK_MAX_AUTO_RESUMES` (`3`) | Re-queue a job whose trainer died without finishing, up to N times in a row. |

**Failover**
| Variable | Meaning |
|---|---|
| `AITK_IMPORT_ON_BOOT` | `all` (every job on the remote that was cut off mid-run: queued, running or interrupted; not ones that completed, failed or that you stopped) or a comma list of job names to import and start at boot. |
| `AITK_IMPORT_REMOTE` | Remote to import from (default `AITK_SYNC_REMOTE`). |
| `AITK_IMPORT_WAIT` (`15m`) | How long to keep waiting for another server's lease to expire. |
| `AITK_IMPORT_FORCE` (`0`) | Take jobs over even while another server's lease is valid (only when you know it is gone). |

**Bootstrap from run bundles** (docs/bundle-format.md; see "Bootstrapping a fresh server" below)
| Variable | Meaning |
|---|---|
| `AITK_BOOTSTRAP` | Comma list of run bundles to import once at boot: an `http(s)://` URL (e.g. a signed `/v1/jobs/{id}/bundle` link from another server, a presigned S3 URL), `<remote>:<key>` of a configured remote (`s3:bundles/my_lora_step2000.zip`), or a path inside the container (`/workspace/run.zip`). |
| `AITK_BOOTSTRAP_START` (`1`) | Queue each imported job so training continues from its newest save; `0` only imports. |
| `AITK_BOOTSTRAP_PREFETCH` (`1`) | While a bundle downloads, fetch its base model (`model.name_or_path`, when it is a Hugging Face diffusers repo) into the HF cache with `HF_TOKEN`; the job starts when both are ready. |

**Unattended runs** (docs/headless.md, "Unattended runs"; all changeable at runtime via `/v1/notify` and `/v1/lifecycle`)
| Variable | Meaning |
|---|---|
| `AITK_NOTIFY` | Comma list of notification sinks: `https://ntfy.sh/<topic>` (self-hosted: `ntfy+https://<host>/<topic>`), a Discord webhook URL, `telegram://<bot token>@<chat id>`, `pushover://<api token>@<user key>`, any other `https://` URL (JSON webhook). |
| `AITK_NOTIFY_EVENTS` (all) | Kinds to send: `started,resumed,sample,checkpoint,completed,failed,interrupted,lease_lost,sync_failed,lifecycle`. |
| `AITK_NOTIFY_SAMPLE_EVERY` (`15m`) | At most one sample grid per job this often. |
| `AITK_PUBLIC_URL` | This server's public URL, for tap-through links (default: RunPod's proxy URL or Vast.ai's mapped port when detected) and for signed download links (`POST /v1/links`; default: the URL the client called). |
| `AITK_STUDIO_URL` | Where links open the studio (default `<public url>/studio/` when the image serves it). |
| `AITK_WHEN_DONE` (`keep`) | `stop` or `destroy` this server once every queue is idle and the last job reached the trigger, after verifying the finished jobs' results are on a remote (never when they are only here). |
| `AITK_WHEN_DONE_DELAY` (`10m`) | Countdown before acting; cancel, extend or skip it from the studio, the API or MCP. A job queued meanwhile cancels it; only training runs arm it (not caption jobs or inference engines), and it waits for bundle transfers and uploads in flight. |
| `AITK_WHEN_DONE_ON` (`completed`) | `any` also triggers when the last job failed. |
| `AITK_WHEN_DONE_BUNDLE` | A remote to keep each finished run in as a resumable bundle (`bundles/<name>_step<step>.zip`, docs/bundle-format.md) before acting. Pushed when missing, verified by size, and preferred over the sync and upload checks. It holds the newest checkpoint, or every one when the job keeps older checkpoints (`keep_checkpoints`); `AITK_WHEN_DONE_BUNDLE_CHECKPOINTS=all|latest` forces either. |
| `AITK_WHEN_DONE_DRY_RUN` (`0`) | Log and emit events instead of calling the provider. |
| `AITK_WHEN_DONE_PROVIDER` (auto) | `runpod`, `vast`, `quickpod`, `command` or `exit`. Auto: `AITK_WHEN_DONE_CMD`, else RunPod (`RUNPOD_POD_ID`), else Vast.ai (`CONTAINER_ID` + `CONTAINER_API_KEY`), else QuickPod (`CONTAINER_LABEL` + `QUICKPOD_PORT_8675` + `AITK_QUICKPOD_API_KEY`), else the process exits. |
| `AITK_RUNPOD_API_KEY` | A full RunPod key (the pod's own `RUNPOD_API_KEY` may only be allowed to stop, not terminate; then the pod is stopped and its disk still billed). |
| `AITK_VAST_API_KEY` | Overrides Vast.ai's per-instance `CONTAINER_API_KEY`. |
| `AITK_QUICKPOD_API_KEY` | A QuickPod key with `pods:write`: QuickPod gives a pod no key of its own, so without it the pod cannot stop or destroy itself. |
| `AITK_WHEN_DONE_CMD` | Shell command that ends the server instead (gets `AITK_WHEN_DONE_ACTION=stop\|destroy`). |

**Provisioning** (creating GPU servers from this server; docs/providers.md)
| Variable | Meaning |
|---|---|
| `AITK_PROVISION_RUNPOD_KEY`, `AITK_PROVISION_VAST_KEY`, `AITK_PROVISION_QUICKPOD_KEY` | Provider keys used when none was saved through the API. Never the self-terminate keys above, so a GPU server does not become a launcher. |
| `AITK_PROVISION_PROVIDERS` (all) | Comma list of providers to offer (`runpod,vast,quickpod`). |
| `AITK_PROVISION_MAX_RUNNING` (`2`) | Live machines per provider, 1 to 20, unless set per provider through the API. |
| `AITK_PROVISION_READY_TIMEOUT` (`20m`) | When a machine that never answered is marked failed. It is not terminated and still bills. |
| `AITK_PROVISION_SIMULATE` (`0`) | `1` answers every provider from in-process mocks: no credits, no provider calls. Local testing only. |
| `AITK_PROVISION_SIMULATE_DELAY` (`15s`) | How long a simulated machine takes to come up. |
| `AITK_PROVISION_SIMULATE_REGISTRY` (unset) | `real` keeps simulate's mock providers but checks images against the real public registries and sizes models from Hugging Face (anonymous reads), instead of the in-process mocks. |
| `AITK_IMAGECHECK_ALLOW_LOOPBACK_PROXY` (`0`) | `1` lets the image check go through an `HTTPS_PROXY` or `HTTP_PROXY` on this machine. It otherwise dials public addresses only. |

**Controller** (no GPU or trainer; creates and tracks GPU servers; docs/controller.md)
| Variable | Meaning |
|---|---|
| `AITK_CONTROLLER` (`0`) | `1`, `true` or `yes` runs this server as a controller (same as `--controller`): its own databases, no trainer, no console, and `127.0.0.1` unless `--host` is given. A reachable address needs `AI_TOOLKIT_AUTH`. |
| `AITK_CONTROLLER_DIR` | Where a controller keeps its databases (same as `--data-dir`; default `<user config dir>/aitk-controller`). |
| `AITK_ALLOWED_HOSTS` | Extra `Host` names an open controller (no token, no API key) answers besides `127.0.0.1`, `[::1]`, `localhost` and the host of `AITK_PUBLIC_URL`: comma list of `host` or `host:port`. |

**Server** (rarely needed)
`PORT` (8675), `AI_TOOLKIT_UI_DIR` (frontend dir), `MODELS_PATH`,
`AI_TOOLKIT_DB_JOURNAL_MODE`, `AI_TOOLKIT_EVENT_RETENTION` (`336h`),
`AI_TOOLKIT_GO_DB`, `AI_TOOLKIT_ROOT`, `AITK_STUDIO_DIR` (studio build served at
`/studio/`, default `ui-go/studio/dist`).

## How failover works

1. A job with a sync remote pushes its dataset once (content addressed),
   then after every checkpoint the newest checkpoint and `optimizer.pt`, the
   loss log, log and config, and finally a manifest that names them. It holds
   a lease in that manifest, renewed every minute, valid for five.
2. The server goes away. Nothing is lost beyond the steps since the last
   checkpoint (`save.save_every` sets that window).
3. The image starts somewhere else with the same env. With
   `AITK_IMPORT_ON_BOOT`, it waits for the old lease to expire (at most five
   minutes), downloads and verifies (sha256) the state, recreates the job and
   starts it; ai-toolkit continues from the checkpoint. Without it, a client
   calls `POST /v1/portable/import` (or `aitk.portable.import` over MCP).
4. If the old server comes back, it sees it lost the lease and stops its copy.

## Bootstrapping a fresh server

The slow part of a new GPU server is getting the run onto it. A run bundle
(docs/bundle-format.md) carries everything: dataset with its latent and
text-embedding caches, the newest save, `optimizer.pt`, config. One
`docker run` turns an empty machine into the run, continuing:

```bash
docker run -d --gpus all -p 8675:8675 \
  -e AI_TOOLKIT_AUTH=<token> -e HF_TOKEN=<hf token> \
  -e AITK_S3_ENDPOINT=... -e AITK_S3_BUCKET=... -e AITK_S3_ACCESS_KEY_ID=... -e AITK_S3_SECRET_ACCESS_KEY=... \
  -e AITK_BOOTSTRAP=s3:bundles/my_lora_step2000.zip \
  ghcr.io/geocine/aitoolkit-go:latest
```

At boot, for each ref in `AITK_BOOTSTRAP`, in order:
1. If this ref was imported before and its job still exists, it is skipped
   (restarts never import twice). The bundle's job name and step are read
   up front when possible (a library item's `.json` summary, one ranged GET
   of a URL's first MiB, a local file); a job of that name at the same or a
   later step is skipped, an older one is left alone with a log line.
2. Its base model starts downloading into the HF cache in parallel
   (`AITK_BOOTSTRAP_PREFETCH`, below), the bundle is downloaded and imported
   like `POST /v1/bundles/imports` (task visible at `GET /v1/bundles/imports`,
   `bundle.import` events, `ref` set).
3. With `AITK_BOOTSTRAP_START=1` (default) the job is queued once the
   prefetch finished: ai-toolkit picks up the newest save and `optimizer.pt`.

A failed import is logged (`bootstrap <ref>: attempt n failed: ...`) and
retried after 30 s, 2 min and 8 min when the cause can pass (network, remote,
a URL not there yet); a bundle that is invalid or conflicts with a job or
dataset here is not retried.

Make bundles with `POST /v1/jobs/{id}/bundle/push {remote}` (library at
`bundles/` on the remote) or download one (`GET /v1/jobs/{id}/bundle`, or a
signed link from `POST /v1/links` that the new server can fetch without a
token). A hand-made zip of `data/` + `output/<name>/` works too.

Prefetch: when `model.name_or_path` is a Hugging Face repo id with a
`model_index.json` (diffusers format), the server runs
`DiffusionPipeline.download(repo)` with the toolkit's Python and `HF_TOKEN`:
model_index.json and the component folders only, not single-file
checkpoints next to them. Other repos, local paths and the extra repos some
archs load (e.g. accuracy-recovery adapters for quantized models) are left
to the trainer; a failed prefetch only adds a warning to the task and the
trainer downloads instead.

## Local testing

`ui-go/docker/compose.local.yml` stands up MinIO as R2 and two servers as two
providers on one machine, and `ui-go/scripts/failover_test.py` runs a real
SD 1.5 job through it: train on node-a, remove node-a mid-training, watch
node-b pick the job up from the env alone and finish it.

```bash
ui-go/scripts/build_and_push.sh            # build the image locally
python ui-go/scripts/failover_test.py      # ~10 minutes, GPU required
```
