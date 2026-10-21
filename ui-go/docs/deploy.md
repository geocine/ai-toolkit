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
  -e AITK_R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
  -e AITK_R2_BUCKET=<bucket> \
  -e AITK_R2_ACCESS_KEY_ID=<key id> -e AITK_R2_SECRET_ACCESS_KEY=<secret> \
  -e AITK_SYNC_REMOTE=r2 -e AITK_AUTO_RESUME=1 -e AITK_IMPORT_ON_BOOT=all \
  -e HF_TOKEN=<hf token, for gated base models> \
  ghcr.io/geocine/aitoolkit-go:latest
```

Then point a client at `http://<host>:8675` with `Authorization: Bearer <token>`:
the `/v1` API (`/v1/openapi.json`), MCP at `/mcp`, or the console at `/`.

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

**Remotes** (created or updated on every start)
| Variable | Meaning |
|---|---|
| `AITK_R2_ENDPOINT`, `AITK_R2_BUCKET`, `AITK_R2_ACCESS_KEY_ID`, `AITK_R2_SECRET_ACCESS_KEY` | An S3-compatible remote (Cloudflare R2, AWS S3, MinIO, Backblaze B2, ...). |
| `AITK_R2_REGION` (`auto`), `AITK_R2_PREFIX`, `AITK_R2_PATH_STYLE` (`1` for MinIO), `AITK_R2_NAME` (`r2`) | Its options and the name it gets. |
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
| `AITK_IMPORT_ON_BOOT` | `all` (every unfinished job on the remote) or a comma list of job names to import and start at boot. |
| `AITK_IMPORT_REMOTE` | Remote to import from (default `AITK_SYNC_REMOTE`). |
| `AITK_IMPORT_WAIT` (`15m`) | How long to keep waiting for another server's lease to expire. |
| `AITK_IMPORT_FORCE` (`0`) | Take jobs over even while another server's lease is valid (only when you know it is gone). |

**Server** (rarely needed)
`PORT` (8675), `AI_TOOLKIT_UI_DIR` (frontend dir), `MODELS_PATH`,
`AI_TOOLKIT_DB_JOURNAL_MODE`, `AI_TOOLKIT_EVENT_RETENTION` (`336h`),
`AI_TOOLKIT_GO_DB`, `AI_TOOLKIT_ROOT`.

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

## Local testing

`ui-go/docker/compose.local.yml` stands up MinIO as R2 and two servers as two
providers on one machine, and `ui-go/scripts/failover_test.py` runs a real
SD 1.5 job through it: train on node-a, remove node-a mid-training, watch
node-b pick the job up from the env alone and finish it.

```bash
ui-go/scripts/build_and_push.sh            # build the image locally
python ui-go/scripts/failover_test.py      # ~10 minutes, GPU required
```
