# AI Toolkit (geocine fork)

A fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) that adds a headless Go server, studio UI, versioned HTTP `/v1` API, and MCP server under [`ui-go/`](ui-go/), published as the public container image `ghcr.io/geocine/aitoolkit-go`.

## How this fork differs from upstream

- **Branch layout:** `main` mirrors [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit); fork changes live on `geocine`.
- **Upstream trainer untouched:** Everything the fork adds lives under `ui-go/` (plus this README). Upstream's Python trainers (`run.py`), models, extensions, Prisma schema (`ui/prisma/schema.prisma`), and Next.js UI (`ui/`) are left unmodified.
- **Go server (`ui-go/`):** A single binary (`aitk-server`) on port `8675` that replaces upstream's Next.js backend and job queue worker against the same `aitk_db.db`, job folders, and `run.py`, storing fork state separately in `aitk_go.db`.
- **HTTP surfaces on `:8675`:**
  - `/api/*` — upstream-compatible UI API (checked against upstream by `ui-go/parity/`).
  - `/v1/*` — headless JSON API (`/v1/openapi.json`) with named API keys, signed download links, SSE/long-poll events, dataset caption and photo checks, S3/R2 and Hugging Face remotes, portable job failover, resumable run bundles (`AITK_BOOTSTRAP`), notifications, unattended stop/destroy lifecycle, and GPU server provisioning/recommendations across RunPod, Vast.ai, and QuickPod.
  - `/mcp` — Streamable HTTP MCP server in code mode (`aitk_api`, `aitk_run`, `aitk_view`) over `/v1`.
- **Built-in UIs:** Management console at `/` (`ui-go/web`) and Studio at `/studio/` (`ui-go/studio`), plus a `--studio-api` mode that runs with no GPU, Python, or upstream checkout to manage rented GPU servers.
- **Container image:** `ghcr.io/geocine/aitoolkit-go` layers the Go server and UIs on top of upstream's `docker/Dockerfile` (`linux/amd64`, CUDA 13.0, driver 580+, RTX 30-series / A100 or newer).

## Quick start

### Docker

Requires an `amd64` host with an NVIDIA GPU (RTX 30-series / A100 or newer, `sm_80+`), a CUDA 13.0+ driver (`580+`), and the NVIDIA Container Toolkit.

```bash
docker pull ghcr.io/geocine/aitoolkit-go:latest

docker run -d --gpus all -p 8675:8675 \
  -e AI_TOOLKIT_AUTH=<long-random-token> \
  -e HF_TOKEN=<hf-token-for-gated-models> \
  ghcr.io/geocine/aitoolkit-go:latest
```

Connect to `http://<host>:8675` with `Authorization: Bearer <token>`:

- `/` — management console
- `/studio/` — Studio UI (installable on a phone home screen)
- `/v1` — headless API (OpenAPI schema at `/v1/openapi.json`)
- `/mcp` — MCP server (Streamable HTTP)

For S3/R2 sync, automatic failover (`AITK_IMPORT_ON_BOOT`), run-bundle bootstrapping (`AITK_BOOTSTRAP`), notifications, and unattended shutdown (`AITK_WHEN_DONE`), see [`ui-go/docs/deploy.md`](ui-go/docs/deploy.md).

### Running locally

**Full server with local GPU and trainer** (requires Go 1.26, Node 22, and upstream's Python environment):

```bash
cd ui && npx prisma db push && cd ..   # once: create/migrate aitk_db.db with upstream's schema
cd ui-go
(cd web && npm ci && npm run build)    # console -> web/dist, served at /
(cd studio && npm ci && npm run build) # studio  -> studio/dist, served at /studio/
go run . --port 8675                   # API, UIs, /mcp, and queue worker on :8675
```

**Studio API only (no GPU or Python needed)** — holds provider keys and creates/tracks GPU servers on RunPod, Vast.ai, or QuickPod:

```bash
cd ui-go/studio && npm ci && npm run build
cd .. && go run . --studio-api         # listens on http://127.0.0.1:8675/studio/
```

## Documentation

- [`ui-go/README.md`](ui-go/README.md) — Go server layout, flags, parity tests, rebasing onto upstream, and image builds
- [`ui-go/docs/deploy.md`](ui-go/docs/deploy.md) — Docker image tags, labels, host requirements, env vars, failover, and bootstrapping
- [`ui-go/docs/providers.md`](ui-go/docs/providers.md) — RunPod, Vast.ai, and QuickPod setup, manual templates, and simulate mode
- [`ui-go/docs/headless.md`](ui-go/docs/headless.md) — `/v1` API contract, events, portable jobs, unattended runs, and MCP code mode
- [`ui-go/docs/studio-api.md`](ui-go/docs/studio-api.md) — running `--studio-api` locally or in a lightweight Docker image
- [`ui-go/docs/bundle-format.md`](ui-go/docs/bundle-format.md) — resumable run bundle zip format

For supported models, Python installation, dataset preparation, config examples, and CLI training (`run.py`), see upstream [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit).
