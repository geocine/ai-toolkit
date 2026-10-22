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
  `forbidden`, 404 `not_found`, 409 `conflict` (409 `dataset_exists` when a
  dataset of that name holds other files), 413 `too_large`, 422
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
upstream. `GET /v1/health` is always public. A signed link (below) stands in
for the token on the one GET it was signed for.

### Signed links
`POST /v1/links {path, ttl_s?}` → `{url, path, expires_at}`. `path` is a server
path with its query; the answer appends `exp` and `sig` (HMAC-SHA256 over the
exact path and query, with a key generated once and kept in the fork
database). `url` is absolute (`AITK_PUBLIC_URL`, else the request's
`X-Forwarded-Proto/Host`, else its Host); `path` is the same, relative.
A `GET`/`HEAD` of exactly that URL is served without a bearer token until
`expires_at`; any change to the path or query, another method, or an expired
link answers 401. Links may only point at downloads:
`/v1/files/content?...` (add `download=1` for an attachment), `/v1/jobs/{id}/bundle?...`, `/v1/bundles/content?...`
and the media routes `/api/img|files|audio/art/...`. `ttl_s` defaults to 3600
(60 to 604800). Browsers use them for multi-GB downloads (`<a href>`, no blob
in memory); agents use them to hand files to people. Links cannot be revoked
one by one; they expire.

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
- `GET /v1/jobs/{id}/bundle?...` and `POST /v1/jobs/{id}/bundle/push`: see Run bundles

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
- `PUT /v1/datasets/{name}/files/{file...}` body = the file's bytes → item (creates or replaces it; may contain `/`; hidden names refused)
- `POST /v1/datasets/{name}/import {url, file?, caption?}` → `{added, files: [..], skipped: [..]}`: the server downloads an http(s) URL; a zip is unpacked like `/archive`, an image/video is saved as `file` (default: from Content-Disposition or the URL, the extension sniffed from the bytes when missing) with `caption` written beside it
- `PUT /v1/datasets/{name}/captions/{file...} {caption}` → item
- `DELETE /v1/datasets/{name}/files/{file...}` → 204 (also removes its caption)
- `DELETE /v1/datasets/{name}` → 204

### Files
Generic access inside the allowed roots: `datasets`, `training`, `data`, `models`.
- `GET /v1/files?root=&path=` → `{root, path, items: [{name, type: file|dir, size, modified}]}`
- `GET /v1/files/content?root=&path=` → bytes (Range, ETag; `?thumb=1` for media; `?download=1` adds `Content-Disposition: attachment`)
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
- `POST /v1/uploads/{id}/retry` → Upload (queued again; 409 while queued/running)
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
- `POST /v1/templates/render {arch, name, datasets: [names], overrides: {"config.process[0].train.steps": 500, ...}}` → `{config, warnings}`
  (applies the arch's defaults like the UI does, then the dotted-path overrides;
  `warnings` include the check below on the result)

The keys the trainer reads are read from the toolkit's Python at runtime too
(`toolkit/config_modules.py`, whose classes read their section with
`kwargs.get(key, default)`, and the processes' `get_conf` reads), so they
match the trainer that runs. Nothing in this list is hand-written except which
class reads which section (`internal/templates/fields.go`).
- `GET /v1/templates/fields?section=` → `{sections: [{path, class, fields: [{key, default, type, doc}]}], errors}`;
  paths are `process`, `train`, `train.ema_config`, `model`, `network`,
  `save`, `sample`, `sample.samples[]`, `datasets[]`, ...; `section=train` also
  returns the sections nested in it. `default` is the Python expression.
- `POST /v1/templates/check {config}` → `{problems: [{path, message, fix}], errors}`:
  keys the trainer ignores where they are, e.g. `config.process[0].loss_type`
  (`fix`: "it is read in train") or `train.ma_decay` (`fix`: "did you mean
  train.ema_config.ema_decay?"). Processes without a `train` or `model`
  section are not checked.
- `GET /v1/templates/examples` → `{items: [{name, file, arch, model, network, error}]}` (upstream's `config/examples`)
- `GET /v1/templates/examples/{name}` → `{name, file, arch, model, network, error, yaml, config}`

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
| `job.sync` | `{remote, action, step, files, error}` (portable sync; `action: lease_lost` when another node took the job) |
| `lifecycle.state` | Lifecycle (see Unattended runs) |
| `bundle.import`, `bundle.push` | BundleTask (see Run bundles; progress at most ~1/s) |

- `GET /v1/events?after=&job_id=&types=a,b&limit=&wait_ms=` → `{items: [Event], next_cursor}` (polling;
  `after=now` starts from the current end, skipping history; `wait_ms` up to
  60000 long-polls: with nothing newer than `after` it answers as soon as the
  next stored event arrives, or empty after `wait_ms`)
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

## Run bundles (data presets)

A run bundle is one zip holding a training run: the job's dataset(s) with
their ai-toolkit caches, its job folder (saves, `optimizer.pt`,
`config.yaml`, samples, log, loss log) and `aitk_bundle.json`. Upload it to
any server and training continues from the newest save. The format, how
paths are rewritten and how the resume step is found are specified in
[bundle-format.md](bundle-format.md); hand-made zips of `data/` +
`output/<name>/` (no manifest) are read too. Unlike portable jobs (a live
sync with a lease) a bundle is a snapshot you move by hand, keep in a
library, or bootstrap a fresh server from (`AITK_BOOTSTRAP`, docs/deploy.md).

`BundleTask` (imports and pushes run in the background):
```
{ id, kind: import|push, state: queued|running|done|failed,
  phase: "" | downloading | reading | restoring | building | uploading,
  bytes, bytes_done,              // of the current phase (bytes 0 = unknown)
  job_id, job_name, dataset,      // import: the job created, its first dataset
  source: url|remote|upload|local, ref,   // ref: the AITK_BOOTSTRAP entry
  remote, key, url,               // import source, or the push's target key and stored URL
  upload_id,                      // import: the staged zip; kept when the import fails
  skipped: [names], warnings: [..],
  error: {code, message} | null, created_at, updated_at }
```

Export
- `GET /v1/jobs/{id}/bundle?checkpoints=latest|all|none&optimizer=1&samples=1&caches=1`
  → `application/zip`, `Content-Disposition: attachment; filename=<name>_step<step>.zip`,
  exact `Content-Length` (entries are stored, not deflated); answers HEAD.
  Defaults: the newest save (plus the final one), `optimizer.pt`, samples and
  the dataset caches (`latents=` is an alias of `caches=`). Safe while the job
  runs: saves and `optimizer.pt` are read once their size is stable, the loss
  log is snapshotted. Browsers download it through a signed link.

Import
- `POST /v1/bundles/imports` JSON with one source and options → 202 BundleTask:
  `{url}` (the server downloads http(s)), `{remote, key}` (an object in a
  remote, e.g. a library item's `key`), or `{upload_id}` (a complete chunked
  upload); options `name?` (job name, default the bundle's; saves are renamed
  `<name>_<step>.safetensors` and `config.name` set so resume still works),
  `dataset?` (name of the first dataset, default the bundle's, else the job
  name), `gpu_ids?` (default "0"), `policy?` (as on POST /v1/jobs, default the
  bundle's auto_resume / max_auto_resumes), `start?` (queue it right away:
  ai-toolkit resumes from the newest save).
- For curl: the same endpoint takes the zip as the body, `Content-Type`
  anything but JSON, options in the query
  (`curl -T run.zip -H 'Content-Type: application/zip' '.../v1/bundles/imports?name=x&start=1'`),
  or multipart with fields `name, dataset, gpu_ids, start, policy` and a `file` part.
- With an `upload_id` (and the curl forms) the bundle is checked before the
  202: a job of that name answers 409 `conflict`; a dataset of that name with
  other files 409 `dataset_exists` (one with the same files is reused); too
  little disk 413. With `url`/`remote` those come back as the failed task's
  `error`, and its `upload_id` keeps the downloaded zip for a retry with
  another `name`/`dataset` (no second download).
- What it does: datasets go to the datasets folder, `output/<name>/` to the
  training folder (each through a temp folder renamed into place; a failure
  removes what it created), the config is rewritten to this server's paths
  (dataset `folder_path`s, every process's `training_folder`, `aitk://`
  paths), `.job_config.json` is regenerated, the job is created at the
  bundle's step with info `Imported from a bundle at step N`. Junk is skipped
  (hidden files and folders like `.ipynb_checkpoints`, `__MACOSX`,
  `output/generations`, stray root files) and listed in `skipped`.
- `GET /v1/bundles/imports?job_id=` → `{items: [BundleTask]}`; `GET /v1/bundles/imports/{id}` → BundleTask

Chunked uploads (for browsers behind proxies with body limits)
- `POST /v1/bundles/uploads {size, filename?}` → 201
  `BundleUpload = {id, filename, size, received, complete, chunk_size, created_at, updated_at, expires_at}`
  (413 when the disk lacks `size`)
- `PUT /v1/bundles/uploads/{id}` body = bytes with `Content-Range: bytes <start>-<end>/<size>`
  (or `?offset=<start>`) → BundleUpload. `chunk_size` is the suggested size
  (32 MiB); at most 128 MiB per PUT. A chunk may start anywhere up to
  `received` (re-sending one is harmless); a gap answers 409 whose message
  says where to resume; `GET` tells `received` after a reload.
- `GET /v1/bundles/uploads` → `{items: [BundleUpload]}` (to resume after a reload);
  `GET /v1/bundles/uploads/{id}`; `DELETE /v1/bundles/uploads/{id}` → 204.
- Sessions expire 24 h after their last chunk; a successful import deletes its upload.

Library (a remote's `bundles/` folder)
- `POST /v1/jobs/{id}/bundle/push {remote, bundle?, overwrite?, checkpoints?, optimizer?, samples?, caches?}`
  → 202 BundleTask (`kind: push`): builds the zip in the staging folder,
  uploads it to `bundles/<bundle>.zip` (default id `<name>_step<step>`) and
  writes the summary `bundles/<bundle>.json`. An existing id without
  `overwrite` answers 409 when `bundle` is given; with the default id it is
  only known once the bundle is built, so the task fails with code `conflict`.
  `GET /v1/bundles/pushes?job_id=`, `GET /v1/bundles/pushes/{id}`.
- `GET /v1/bundles?remote=` → `{items: [LibraryBundle]}` newest first:
  `LibraryBundle = {id, key, size, modified, summary: BundleSummary | null}`
  (null for a zip put there without a summary),
  `BundleSummary = {format, name, step, total_steps, size, sha256, created_at, images, datasets: [names], base_model, arch, checkpoints: [..], optimizer, server_version}`.
  Import one with `{remote, key}`.
- `GET /v1/bundles/content?remote=&id=` → the zip (fetched from the remote into
  the staging folder, then served with Range support); sign it with
  `POST /v1/links` for a browser download.
- `DELETE /v1/bundles/{id}?remote=` → 204 (the zip and its summary)

Staging files (uploads, downloads, zips being built) live in
`<training folder>/.aitk-bundles/`.

## Unattended runs

A rented GPU server should not need babysitting: it tells the user's phone
what happens, and when the work is done it can stop or destroy itself, but
only once the results are provably off the box.

### Notifications
`internal/notify` follows the event log and sends messages to every sink.
Kinds (the filter): `started`, `resumed` (by hand or auto-resumed),
`sample` (a grid of the newest step's samples as JPEG, at most once per
`sample_every_s` per job, default 15 min), `checkpoint` (synced to the sync
remote or uploaded), `completed` (with the result file, where it is kept and
the samples grid), `failed` (with the last log lines), `interrupted`,
`lease_lost`, `sync_failed` (at most every 30 min per job), `lifecycle`
(countdown, held, cancelled, acting, done, failed).

Sinks (URLs hold their secret: write-only, shown redacted):

| URL | Sends |
|---|---|
| `https://ntfy.sh/<topic>`; self-hosted `ntfy+https://<host>/<topic>`; auth `https://:<token>@host/topic` or `user:pass@` | ntfy publish: title, click link, priority, tags; the image as an attachment |
| `https://discord.com/api/webhooks/<id>/<token>` | an embed (title links to the run) with the image attached |
| `telegram://<bot token>@<chat id>` | `sendMessage`, or `sendPhoto` with a caption |
| any other `https://...` (`webhook+` forces it) | JSON `{kind, level, title, body, url, job_id, job_name, at, server, image_jpeg_base64?}` |

Links open the run in the studio (`<studio>/#/runs/<id>`) or, without one, in
the console (`<public>/#/jobs/<id>`); lifecycle messages open
`<studio>/#/lifecycle`. The public URL is `AITK_PUBLIC_URL`, else RunPod's
proxy URL (`https://<pod id>-8675.proxy.runpod.net`) or Vast.ai's mapped port;
the studio is `AITK_STUDIO_URL`, else `<public>/studio/` when this server
serves it. Delivery is queued per sink and never blocks the server: 5
attempts with backoff (honoring `Retry-After`) on network errors, 429 and 5xx.

`NotifySink = {id, kind: ntfy|discord|telegram|webhook, target, source: env|api, created_at}`
- `GET /v1/notify` → `{sinks: [NotifySink], events: [kind] | null, kinds: [kind], public_url, studio_url, sample_every_s}`
- `PATCH /v1/notify {events?: [kind] | null, public_url?, studio_url?, sample_every_s?}` → same
- `POST /v1/notify/sinks {url}` → 201 NotifySink (400 for an unknown URL form, 409 when already there)
- `DELETE /v1/notify/sinks/{id}` → 204
- `POST /v1/notify/test {sink?, message?}` → `{items: [{sink, kind, target, ok, detail}]}` (one attempt each, synchronous; 409 without sinks)

Sinks from `AITK_NOTIFY` are `source: env` and are replaced at every boot;
API ones persist (fork setting `notify.config`).

### When done
`internal/lifecycle` watches the queues. When nothing is queued, running or
stopping and the job that ended last ended `completed` (`on: any`: also
`failed`), it starts a countdown (`delay_s`). A job queued meanwhile cancels
it. At the deadline it checks every finished job (completed, failed, stopped,
interrupted) that has a checkpoint: its newest model file must be on a remote,
proven by the remote object's size. The first check that applies to a job
decides:

- with `AITK_WHEN_DONE_BUNDLE=<remote>`: the run is in that remote's library
  as a bundle holding the newest checkpoint (`bundles/<name>_step<step>.zip`,
  its size matching its summary; else it is pushed and checked again);
- `sync_remote` jobs: the portable manifest lists the file with the local size
  and the object exists with that size (else one final sync runs, then it is
  checked again);
- `auto_upload_remote` jobs: `jobs/<job>/<file>` exists with the local size
  (else it is uploaded and waited for).

If any job fails the check the server stays up (`state: held`) and says so
("results only on this server, not destroying"). Otherwise it stops or
destroys the server through the provider, detected from the environment:
RunPod (`RUNPOD_POD_ID`; `DELETE https://rest.runpod.io/v1/pods/{id}`,
falling back to `POST .../pods/{id}/stop` when the key may not terminate),
Vast.ai (`CONTAINER_ID` + `CONTAINER_API_KEY`; `DELETE
https://console.vast.ai/api/v0/instances/{id}/`, stop is `PUT` with
`{"state": "stopped"}`), `AITK_WHEN_DONE_CMD`, or a plain process exit.
With `dry_run` it only logs and emits events.

`Lifecycle = {mode: keep|stop|destroy, on: completed|any, delay_s, dry_run,
state, deadline, reason, provider: runpod|vast|command|exit, job_id,
job_name, results: [{job_id, job_name, safe, via, detail}], updated_at}`;
`state` is `off` (keep) | `waiting` | `countdown` | `verifying` | `acting` |
`held` | `cancelled` | `done` | `failed`.
- `GET /v1/lifecycle` → Lifecycle
- `PATCH /v1/lifecycle {mode?, on?, delay_s?, dry_run?}` → Lifecycle (persisted, though `AITK_WHEN_DONE*` set in the env win again at boot; a change decides afresh: a countdown restarts with the new delay)
- `POST /v1/lifecycle/cancel` → Lifecycle (`cancelled` until another job runs and ends, also across restarts; 409 when nothing is pending)
- `POST /v1/lifecycle/extend {minutes}` → Lifecycle (409 without a countdown)
- `POST /v1/lifecycle/now` → Lifecycle (skip the countdown; 409 while a job is queued or running, or with mode keep)

Every change is a `lifecycle.state` event carrying Lifecycle.

## MCP (code mode)

`/mcp` speaks MCP over Streamable HTTP (same bearer auth). Instead of one tool
per endpoint it exposes three:

- `aitk_api` — returns the TypeScript declaration of the `aitk` client (every
  `/v1` operation with typed params and results), optionally filtered by a
  search string.
- `aitk_run {code}` — runs TypeScript/JavaScript against that client in a
  sandbox and returns `{result, logs, error}`. The code is transpiled with
  esbuild and executed in an embedded JS runtime (goja) with no filesystem,
  network or process access: only `aitk.*` (calls `/v1` in-process with the
  caller's credentials), `console.log`, and `sleep(ms)`. Limits: 120 s wall
  time per call (`timeout_ms` up to 600 s), 1 MiB of logs, 256 API calls.
- `aitk_view {paths, max_px?}` — looks at images: up to 8 paths as the API
  returns them (a sample's `path`, a dataset's `path` joined with an item's
  `file`, or `<dataset>/<file>`), inside the datasets, training or data
  folders. Returns one JPEG preview per path (fit to `max_px`, default 512,
  max 1024; videos show their thumbnail) with a caption line, 8 MiB per call.

Everything a UI can do has an `aitk` op (a test walks the registered `/v1`
routes; only `GET /v1/events/stream` and `/v1/openapi.json` are left out:
`events.list({ after, wait_ms })` long-polls instead, `aitk_api` replaces the
OpenAPI document). Bodies that are not JSON:
- binary in: base64 params, 16 MiB per call (`datasets.putFile`,
  `datasets.uploadFiles`, `datasets.uploadArchive`, `bundles.uploadChunk`),
  or let the server fetch a URL (`datasets.importUrl`, `bundles.import({ url })`);
- binary out: ops returning `Link` (`files.link`, `bundles.download`,
  `links.create`) sign the download instead of fetching it, so the agent
  gets a URL it can hand to a person or to another server;
- `bundles.wait(id)` polls an import or push until it settles, like `jobs.wait`.

Example the agent might send to `aitk_run`:
```ts
const ds = await aitk.datasets.create({ name: "my_set" });
const { config, warnings } = await aitk.templates.render({ arch: "sd1", name: "my_lora", datasets: [ds.name],
  overrides: { "config.process[0].train.steps": 1500 } });
console.log(warnings); // e.g. an override in a section the trainer does not read
const job = await aitk.jobs.create({ name: "my_lora", gpu_ids: "0", config,
  policy: { auto_resume: true, auto_upload_remote: "s3" } });
await aitk.jobs.start(job.id);
return (await aitk.jobs.get(job.id)).state;
```

Moving a run to another server from an agent:
```ts
const job = (await aitk.jobs.list()).items.find((j) => j.name === "my_lora");
const { url } = await aitk.bundles.download(job.id, { ttl_s: 3600 });
// on the other server's MCP:
//   const t = await aitk.bundles.import({ url, start: true });
//   return await aitk.bundles.wait(t.id);
return url;
```
