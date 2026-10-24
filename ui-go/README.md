# ai-toolkit Go server

A Go port of the ai-toolkit UI backend: the HTTP API of upstream's Next.js UI
(`ui/src/app/api`) and its job queue worker (`ui/cron`), working against the
same database, job folders and Python entry point (`run.py`).

The frontend lives in `web/` and is served by the Go server at `/`. What is
there now is a placeholder (API status, GPUs, jobs); replace the folder with
the real UI and keep `npm run build` emitting `web/dist/`.

## How this fork stays rebase-friendly

Everything the fork adds lives under `ui-go/`. No upstream file is modified,
so rebasing onto upstream never conflicts and every Python-side change
(models, trainers, extensions) arrives untouched.

The Go side depends on upstream only through these contracts, and each one is
checked automatically:

| Upstream contract | Where it lives | How drift is caught |
|---|---|---|
| HTTP API (paths, methods) | `ui/src/app/api/**/route.ts` | `go test ./parity/` (`TestRouteCoverage`) |
| Database schema | `ui/prisma/schema.prisma` | `go test ./parity/` (`TestSchemaDrift`), plus a schema check at startup |
| Response bodies and behavior | the Next routes | `ui-go/scripts/parity.sh` (both servers, same requests, JSON diff) |
| Job launch | `ui/cron/actions/startJob.ts`, `run.py`, `.job_config.json` | parity run + a real job |
| Docker image | `docker/Dockerfile`, `docker/start.sh` | the fork image is built *on top of* upstream's; the build fails if `start.sh` changes shape |

The schema is owned by upstream: the Go server never creates or alters tables.
Create or migrate the database with upstream's Prisma CLI (`npx prisma db push`
in `ui/`); the Docker image does this on start. The studio API is the one
exception: it creates its own from an embedded copy of the schema, kept equal
to upstream's by `go test ./parity/`.

## Rebasing onto upstream

```bash
git fetch upstream            # or update the local main that mirrors it
git rebase main               # from the go-react branch; no conflicts expected
cd ui-go
go build ./... && go vet ./...
go test ./...                 # route coverage + schema drift fail on anything new upstream
../ui-go/scripts/parity.sh    # optional: diff live responses against upstream's server
```

When `go test ./parity/` fails, it names the route or column to port. Mirror
the upstream file next to the matching Go package (`handlers/<group>`), then
rerun the tests.

## Layout

```
main.go                 flags, startup, static frontend (optional)
internal/server         route table (every handler package)
internal/store          SQLite access in Prisma's storage format (DateTime = ms, Boolean = 0/1)
internal/worker         queue loop and job launch (detached; Windows uses a relay like upstream)
internal/settings       cached settings (DATASETS_FOLDER, TRAINING_FOLDER, ...)
internal/httpx          JSON helpers, route registry, auth middleware
internal/{paths,safepath,proc,apicache,engines,jsonx}
handlers/<group>        one package per upstream route group
web/                    frontend (Vite + React placeholder), built to web/dist
parity/                 drift tests against upstream's files
docker/                 image layer over upstream's image; compose.local.yml for local tests
scripts/                build_and_push.sh (offline tests in scripts/testdata/), parity.sh, failover_test.py
```

## Running locally

```bash
cd ui && npx prisma db push && cd ..   # once: create/migrate aitk_db.db with upstream's schema
cd ui-go
(cd web && npm ci && npm run build)    # frontend -> web/dist, served at /
go run . --port 8675                   # API + frontend on :8675, queue worker included
```

No GPU or checkout? `go run . --studio-api` runs the studio API, which only
creates and tracks GPU servers, with its own databases and the studio at
`/studio/`: see [docs/studio-api.md](docs/studio-api.md).

For frontend work, run `npm run dev` in `web/` (Vite on :5173, `/api` proxied
to the Go server on :8675, or `AITK_API`).

Flags: `--port` (env `PORT`, default 8675), `--host`, `--ui-dir` (env
`AI_TOOLKIT_UI_DIR`, a built frontend to serve at `/`, default `ui-go/web/dist`
when present), `--no-worker` (when the Next worker is running instead; run
only one worker per database).

Env, same as upstream: `AI_TOOLKIT_AUTH` (bearer token for `/api/*` except
media routes), `MODELS_PATH`, `AI_TOOLKIT_DB_JOURNAL_MODE`, plus
`AI_TOOLKIT_ROOT` to point at a checkout other than `..`.

## Docker

Production is one image configured by env vars (no compose): see
[docs/deploy.md](docs/deploy.md) for the variables, provider setup and how a
job continues on another provider. To rent a GPU server on RunPod, Vast.ai
or QuickPod from the studio, or set one up by hand, see
[docs/providers.md](docs/providers.md).

```bash
ui-go/scripts/build_and_push.sh            # base: upstream's docker/Dockerfile at the merge-base; then the Go layer
ui-go/scripts/build_and_push.sh --push     # push <sha>, <timestamp>, <version>, cu130, latest; verify <sha>
ui-go/scripts/build_and_push.sh --verify   # re-check the pushed <sha> (after making the package public)
python ui-go/scripts/failover_test.py      # local failover test (docker/compose.local.yml, GPU)
```

The image is the public package `ghcr.io/geocine/aitoolkit-go` (linux/amd64,
CUDA 13.0, driver 580 or newer, RTX 30-series or A100 and newer). Its tags,
labels and host requirements, and how to publish it the first time, are in
[docs/deploy.md](docs/deploy.md#the-image). `IMAGE` (default
`ghcr.io/geocine/aitoolkit-go`), `UPSTREAM_REF` (default `main`),
`REBUILD_BASE=1`, `BASE_INSTALLER=uv` and `ALLOW_RETAG=1` adjust the build.

## Headless API

[docs/headless.md](docs/headless.md) is the contract for apps and agents:
the `/v1` API, events, recovery and resume, portable jobs, remotes and MCP
code mode.
