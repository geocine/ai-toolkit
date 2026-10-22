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
| RunPod | Template: container image above, *Expose HTTP Ports* `8675`, env vars in the template. Reach it at `https://<pod id>-8675.proxy.runpod.net`. `PUBLIC_KEY` enables SSH (upstream's start script). |
| Vast.ai | Image above; *Docker options* `-p 8675:8675 -e ...`. Use the mapped port shown on the instance. |
| Lambda, any VM | `docker run` as in the quick start (NVIDIA Container Toolkit installed). |

Disks can be ephemeral: training state lives on the remote. Mount
`/root/.cache/huggingface` only to keep base models across restarts on the
same machine.

## Environment

**Access**
| Variable | Meaning |
|---|---|
| `AI_TOOLKIT_AUTH` | Bearer token for `/api`, `/v1` and `/mcp` (admin). Named, revocable keys can be added with `POST /v1/keys`. Without a token and without keys the server is open. |
| `HF_TOKEN` | Hugging Face token passed to trainers (gated base models) and the default for `hf` remotes. |
| `AITK_CORS_ORIGINS` | Origins allowed to call the API from a browser app hosted elsewhere (the studio, a desktop webview): comma list, or `*`. Unset = same origin only. |

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
| `AITK_WHEN_DONE_BUNDLE` | A remote to keep each finished run in as a resumable bundle (`bundles/<name>_step<step>.zip`, docs/bundle-format.md) before acting. Pushed when missing, verified by size, and preferred over the sync and upload checks. |
| `AITK_WHEN_DONE_DRY_RUN` (`0`) | Log and emit events instead of calling the provider. |
| `AITK_WHEN_DONE_PROVIDER` (auto) | `runpod`, `vast`, `command` or `exit`. Auto: `AITK_WHEN_DONE_CMD`, else RunPod (`RUNPOD_POD_ID`), else Vast.ai (`CONTAINER_ID` + `CONTAINER_API_KEY`), else the process exits. |
| `AITK_RUNPOD_API_KEY` | A full RunPod key (the pod's own `RUNPOD_API_KEY` may only be allowed to stop, not terminate; then the pod is stopped and its disk still billed). |
| `AITK_VAST_API_KEY` | Overrides Vast.ai's per-instance `CONTAINER_API_KEY`. |
| `AITK_WHEN_DONE_CMD` | Shell command that ends the server instead (gets `AITK_WHEN_DONE_ACTION=stop\|destroy`). |

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
