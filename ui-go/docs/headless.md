# Headless server

The Go server is a headless training node. Apps (a custom UI, a desktop app,
scripts) and agents (over MCP) connect to it directly; the UI it serves at `/`
is only a management console for stepping in.

It speaks two HTTP surfaces on the same port:

| Surface | Purpose | Contract |
|---|---|---|
| `/api/*` | Upstream-compatible API (the Next.js UI's) | Exactly upstream's behavior, checked by `parity/` and `scripts/parity.sh` |
| `/v1/*` | App API for clients and agents | This document; stable, versioned, structured errors |
| `/mcp` | MCP server in code mode | Two tools over `/v1` (below) |

`/api/*` keeps upstream's quirks on purpose. New clients use `/v1`.

## Fork state

Everything the fork adds lives in its own SQLite file, `aitk_go.db` next to
upstream's `aitk_db.db` (override: `AI_TOOLKIT_GO_DB`). Upstream's schema is
never extended. Tables (owned by `internal/state`, embedded migrations):

| Table | Holds |
|---|---|
| `events` | append-only event log (cursor = `id`) |
| `api_keys` | named API keys (sha256 of the secret, display prefix, last used, revoked) |
| `job_attempts` | one row per launch of a job: start/end, end status/reason, step at start/end |
| `job_policies` | per-job `auto_resume`, `max_auto_resumes`, `auto_upload_remote` |
| `remotes` | upload targets (S3/R2 or Hugging Face); secrets write-only |
| `uploads` | upload jobs and their progress |
| `file_hashes` | sha256 cache keyed by path + size + mtime |

## Conventions (`/v1`)

- JSON in and out; times are RFC 3339 UTC strings; sizes in bytes.
- Errors: `{"error": {"code": "<snake_case>", "message": "<human text>"}}` with
  an HTTP status that matches (400 `invalid_request`, 401 `unauthorized`, 403
  `forbidden`, 404 `not_found`, 409 `conflict`, 413 `too_large`, 422
  `invalid_config`, 500 `internal`, 502 `remote_failed`).
- Lists return `{"items": [...]}` (plus `next_cursor` when paged).
- `POST` creating a resource answers 201 with the resource; actions answer 200
  with the resource after the action (never the stale row upstream returns).

## Auth

`Authorization: Bearer <token>` on `/v1/*`, `/mcp` and the upstream `/api/*`
routes. A token is either `AI_TOOLKIT_AUTH` (admin, like upstream) or a named
API key (`aitk_<32 base62>`, created via `POST /v1/keys`). If neither
`AI_TOOLKIT_AUTH` is set nor any key was ever created, the server is open
(upstream's behavior). Once a key exists auth stays on, even after every key
is revoked. Media GETs under `/api/img|files|audio/art` stay public like
upstream. `GET /v1/health` is always public.

## Resources

### System
- `GET /v1/health` → `{ok, version, upstream_version, started_at}`
- `GET /v1/system` → `{gpus: [...], cpu: {...}, disks: [{path, free, total}], toolkit_root}`

### Jobs
`Job` (v1 view of upstream's row plus fork state):
```
{ id, name, type, ref, gpu_ids, state, status, info, step, total_steps, speed,
  queue_position, pid, created_at, updated_at,
  config,                          // parsed job_config object
  policy: {auto_resume, max_auto_resumes, auto_upload_remote, sync_remote},
  attempts: n, last_attempt: Attempt | null,
  latest_checkpoint: Checkpoint | null }
```
`state` is the client-facing lifecycle: `draft | queued | starting | running |
stopping | stopped | completed | failed | interrupted`. It derives from the
upstream `status` plus fork state (`interrupted` = the process died without
writing an ending, e.g. the server or machine went down; `starting` = running
with step 0 and no trainer progress yet).

- `GET /v1/jobs?state=&type=` → `{items: [Job]}` (newest first)
- `POST /v1/jobs {name, gpu_ids, config, type?, ref?, policy?}` → 201 Job (`draft`)
- `GET /v1/jobs/{id}` → Job
- `PATCH /v1/jobs/{id} {name?, gpu_ids?, config?, policy?}` → Job (409 while queued/running)
- `DELETE /v1/jobs/{id}?files=true|false` → 204 (409 while running)
- `POST /v1/jobs/{id}/start` → Job. Queues it and makes sure its GPU queue runs.
  Starting a job that has checkpoints resumes from the latest one (ai-toolkit
  does this itself for the same job folder).
- `POST /v1/jobs/{id}/stop` → Job (graceful: trainer saves state and exits)
- `POST /v1/jobs/{id}/kill` → Job (force)
- `POST /v1/jobs/{id}/save_now`, `/sample_now` → Job
- `GET /v1/jobs/{id}/attempts` → `{items: [Attempt]}`;
  `Attempt = {n, started_at, ended_at, end_state, end_reason, start_step, end_step, auto}`
- `GET /v1/jobs/{id}/log?offset=&limit=` → `{offset, next_offset, size, text}`
  (byte offsets into log.txt; `limit` default 64 KiB, max 1 MiB; negative
  `offset` = from the end)
- `GET /v1/jobs/{id}/loss?key=&after_step=&limit=&stride=` → `{keys: [..], key, points: [{step, value, wall_time}]}`
- `GET /v1/jobs/{id}/checkpoints?hash=true` → `{items: [Checkpoint]}`;
  `Checkpoint = {name, path, step, size, modified, sha256?}` (`step` null for the final file)
- `GET /v1/jobs/{id}/samples` → `{items: [{name, path, step, index, modified}]}`

### Queues
- `GET /v1/queues` → `{items: [{gpu_ids, running, current_job_id, queued: n}]}`
- `POST /v1/queues/{gpu_ids}/start`, `/stop` → queue

### Datasets
`Dataset = {name, path, images, captions, size, modified}`
- `GET /v1/datasets` → `{items: [Dataset]}`
- `POST /v1/datasets {name}` → 201 Dataset (name used as given if it is a safe folder name, else 400)
- `GET /v1/datasets/{name}?cursor=&limit=` → Dataset + `{items: [{file, caption, size, width?, height?}], next_cursor}`
- `POST /v1/datasets/{name}/files` multipart `files[]` → `{items: [file names]}`
- `POST /v1/datasets/{name}/archive` body = zip (or multipart `file`) → `{added: n, skipped: [..]}` (safe extraction: no traversal, no symlinks, size and count limits)
- `PUT /v1/datasets/{name}/captions/{file...} {caption}` → item
- `DELETE /v1/datasets/{name}/files/{file...}` → 204 (also removes its caption)
- `DELETE /v1/datasets/{name}` → 204

### Files
Generic access inside the allowed roots: `datasets`, `training`, `data`, `models`.
- `GET /v1/files?root=&path=` → `{root, path, items: [{name, type: file|dir, size, modified}]}`
- `GET /v1/files/content?root=&path=` → bytes (Range, ETag; `?thumb=1` for media)
- `PUT /v1/files/content?root=&path=&overwrite=` body = bytes → `{root, path, size, sha256}` (atomic write)
- `DELETE /v1/files/content?root=&path=` → 204

### Remote uploads
`Remote = {id, name, kind: s3|hf|fs, config, created_at}` where `config` is
`s3: {endpoint, bucket, region, prefix, path_style}` (R2: endpoint
`https://<account>.r2.cloudflarestorage.com`, region `auto`),
`hf: {repo_id, repo_type: model|dataset, private, path_in_repo, revision}` or
`fs: {path}` (a mounted shared volume). Every kind implements
`internal/remote.Client`.
Credentials (`s3: {access_key_id, secret_access_key}`, `hf: {token}`; HF falls
back to the HF_TOKEN setting) are accepted on create and never returned.
- `GET /v1/remotes`, `POST /v1/remotes`, `DELETE /v1/remotes/{id}`, `POST /v1/remotes/{id}/test` → `{ok, detail}`
- `POST /v1/uploads {remote, root, path}` or `{remote, job_id, checkpoint}` → 202 Upload
- `GET /v1/uploads?job_id=&state=` → `{items: [Upload]}`; `GET /v1/uploads/{id}`
- `Upload = {id, remote_id, job_id, root, path, url, state: queued|running|done|failed, bytes, bytes_done, sha256, error, created_at, updated_at}`
- A job policy `auto_upload_remote` uploads every new checkpoint (and the final file) to that remote.

### Settings and keys
- `GET /v1/settings` → `{training_folder, datasets_folder, data_root, models_path, hf_token_set, auto_resume_default, max_auto_resumes_default}`
- `PATCH /v1/settings {...}` (`hf_token` write-only) → settings
- `GET /v1/keys` → `{items: [{id, name, prefix, created_at, last_used_at}]}`;
  `POST /v1/keys {name}` → 201 `{id, name, prefix, key}` (`key` shown once);
  `DELETE /v1/keys/{id}` → 204

### Templates
Job configs are built from upstream's own UI definitions (read from
`ui/src/app/jobs/new/jobConfig.ts` and the extensions' `ui.tsx` at runtime, so
they follow upstream after a rebase).
- `GET /v1/templates` → `{archs: [{name, label, group, defaults}], default_config}`
- `POST /v1/templates/render {arch, name, datasets: [names], overrides: {"config.process[0].train.steps": 500, ...}}` → `{config}`
  (applies the arch's defaults like the UI does, then the dotted-path overrides)

## Events

Every change worth reacting to is an event in `events`, with a monotonically
increasing `id`. Clients keep the last id they saw and resume from it.

`Event = {id, type, job_id, at, data}`. Types:

| type | data |
|---|---|
| `job.created`, `job.deleted` | `{name}` |
| `job.state` | `{from, to, status, info}` |
| `job.progress` | `{step, total_steps, speed, info}` (coalesced, at most ~1/s per job) |
| `job.loss` | `{points: [{step, key, value}]}` (batched) |
| `job.checkpoint` | Checkpoint |
| `job.sample` | `{name, path, step, index}` |
| `job.log` | `{offset, text}` (only streamed, see below; not stored) |
| `job.attempt` | Attempt (on start and on end) |
| `upload.state` | Upload |
| `dataset.changed` | `{name}` |

- `GET /v1/events?after=&job_id=&types=a,b&limit=` → `{items: [Event], next_cursor}` (polling;
  `after=now` starts from the current end, skipping history)
- `GET /v1/events/stream?after=&job_id=&types=` → `text/event-stream`; each
  event is `id: <id>\nevent: <type>\ndata: <Event JSON>\n\n`; honors
  `Last-Event-ID`; `: ping` every 15 s. With `job_id` and `types` including
  `job.log`, log text is streamed live from `log.txt` (not persisted).

Events are kept for 14 days (`AI_TOOLKIT_EVENT_RETENTION`).

## Recovery and resume

Jobs run detached, so they survive a server restart. The supervisor makes sure
nothing is ever left claiming to run:

1. On boot and every 5 s, every job whose upstream `status` is `running` or
   `stopping` is checked: its pid must be alive **and** be that job's trainer
   (command line holds its `.job_config.json`). A live one is watched; a dead
   one is marked failed in upstream's table (`status: error`, info
   `Interrupted: ...`) and its attempt is closed as `interrupted`.
2. If the job's policy (or `auto_resume_default`) allows, an interrupted job is
   queued again (`job.attempt` with `auto: true`), up to `max_auto_resumes`
   consecutive interruptions. ai-toolkit continues from the latest checkpoint.
3. A job that failed on its own (traceback, non-zero exit with the trainer's
   own error) is never auto-resumed.

## Portable jobs (continue on another provider)

A GPU server is disposable. A job whose policy has `sync_remote` keeps its
resumable state on that remote, so if the server ends (pod deleted, provider
outage, credits ran out) any other server, on any provider, can import the
job and continue from the last checkpoint.

Remote layout (under the remote's prefix / repo path / folder):
```
jobs/<job name>/manifest.json            portable manifest (below)
jobs/<job name>/files/<file>             latest checkpoint(s), optimizer.pt, loss_log.db, log.txt, .job_config.json
jobs/<job name>/samples/<file>           samples (best effort)
datasets/<dataset name>/manifest.json    {files: [{path, sha256, size}]}
blobs/<sha256>                           dataset file contents, uploaded once, shared by all datasets
```

`manifest.json`:
```
{ schema: 1, name, type, ref, gpu_ids, config,     // config with dataset paths as "aitk://datasets/<name>/<rel>"
  policy, datasets: [names], step, total_steps, state,
  files: [{name, size, sha256, step}],             // what files/ holds; the newest checkpoint + optimizer.pt
  attempts: [Attempt],
  lease: {node_id, node_label, heartbeat_at, expires_at} | null,
  updated_at }
```

Sync (on the server running the job):
- When the job starts: datasets are pushed (only blobs the remote lacks), the
  manifest is written and the lease taken (`expires_at` = now + 5 min).
- On every `job.checkpoint`: the checkpoint and `optimizer.pt` (and the loss
  log and log) are uploaded, then the manifest is updated. Only the newest
  checkpoint is required for resuming; older ones are pruned remotely unless
  `keep_checkpoints` is set on the remote's config.
- While running, the lease is renewed every 60 s. On a clean end (completed,
  stopped, failed) the lease is released (`lease: null`).

Import (on the new server):
- `GET /v1/portable/jobs?remote=` → `{items: [{name, step, state, lease, updated_at, datasets}]}`
- `POST /v1/portable/import {remote, name, gpu_ids?, start?, force?}` → 201 Job.
  Refused with 409 when the lease is held by another node and not expired
  (`force: true` takes it over), or a local job of that name exists.
  Restores the datasets (verifying sha256), rewrites `aitk://datasets/...` to
  this server's datasets root, downloads `files/` into the job folder,
  creates the job with the same name (ai-toolkit resumes from the newest
  checkpoint it finds there), sets `sync_remote` to the same remote, takes the
  lease and, with `start: true`, queues it.
- `POST /v1/jobs/{id}/sync` → forces a sync now; `GET /v1/jobs/{id}/sync` →
  `{remote, last_sync_at, last_error, lease}`.

Leases are advisory: a server never starts a synced job whose remote lease is
held by another live node (`job.state` stays queued with an info saying so).

## MCP (code mode)

`/mcp` speaks MCP over Streamable HTTP (same bearer auth). Instead of one tool
per endpoint it exposes two:

- `aitk_api` — returns the TypeScript declaration of the `aitk` client (every
  `/v1` operation with typed params and results), optionally filtered by a
  search string.
- `aitk_run {code}` — runs TypeScript/JavaScript against that client in a
  sandbox and returns `{result, logs, error}`. The code is transpiled with
  esbuild and executed in an embedded JS runtime (goja) with no filesystem,
  network or process access: only `aitk.*` (calls `/v1` in-process with the
  caller's credentials), `console.log`, and `sleep(ms)`. Limits: 120 s wall
  time per call (`timeout_ms` up to 600 s), 1 MiB of logs, 256 API calls.

Example the agent might send to `aitk_run`:
```ts
const ds = await aitk.datasets.create({ name: "my_set" });
const { config } = await aitk.templates.render({ arch: "sd1", name: "my_lora", datasets: [ds.name],
  overrides: { "config.process[0].train.steps": 1500 } });
const job = await aitk.jobs.create({ name: "my_lora", gpu_ids: "0", config,
  policy: { auto_resume: true, auto_upload_remote: "r2" } });
await aitk.jobs.start(job.id);
return (await aitk.jobs.get(job.id)).state;
```
