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
| `settings` | fork settings; GPU providers keep `provision.key.<kind>` (the provider API key, write-only) and `provision.max_running.<kind>` |
| `machines` | GPU servers rented through `/v1/machines`: provider, ids, status, problem, token, spec (never env values), image digest and launch ref |
| `photo_metrics` | photo-check measurements per file content (sha256 + metrics version): sharpness, exposure, hashes, EXIF; dropped 90 days after a review last read them |
| `photo_faces` | face-check results per file content: boxes, landmarks, scores, face sharpness and light (no face templates) |
| `photo_sets` | per dataset folder: subject kind, reference photo, keep/intended decisions, face crops' parents and the last face check's face-to-face similarities; dropped with the folder |

## Conventions (`/v1`)

- JSON in and out; times are RFC 3339 UTC strings; sizes in bytes.
- Errors: `{"error": {"code": "<snake_case>", "message": "<human text>"}}` with
  an HTTP status that matches (400 `invalid_request`, 401 `unauthorized`, 403
  `forbidden`, 404 `not_found`, 409 `conflict` (409 `dataset_exists` when a
  dataset of that name holds other files), 413 `too_large`, 422
  `invalid_config`, 500 `internal`, 502 `remote_failed`; GPU providers add
  400 `provider_auth`, 402 `insufficient_credit`, 409 `no_capacity`, 422
  `image_blocked` and 429 `rate_limited`, see GPU providers and machines).
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
- `GET /v1/health` → `{ok, version, upstream_version, started_at, mode}`;
  `mode` is `"server"`, or `"studio-api"` when the server runs with
  `--studio-api` / `AITK_STUDIO_API=1`
- `GET /v1/system` → `{gpus: [...], cpu: {...}, disks: [{path, free, total}], toolkit_root}`

### Jobs
`Job` (v1 view of upstream's row plus fork state):
```
{ id, name, type, ref, gpu_ids, state, status, info, step, total_steps, speed,
  queue_position, pid, created_at, updated_at,
  config,                          // parsed job_config object
  policy: {auto_resume, max_auto_resumes, auto_upload_remote, sync_remote, keep_checkpoints},
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
- `PUT /v1/jobs/{id}/chosen {step | file, strength?, note?}` → Job (`chosen: {step, file, strength, note, at}`);
  `DELETE` clears it: the checkpoint picked as the run's result
- `POST /v1/jobs/preflight {config | job_id}` → `{ok, checks: [{level: block|warn|info, message, fix?}]}`: what
  would go wrong or be wasted before a GPU is paid for: keys the trainer ignores, a missing dataset
  folder, images it skips (it reads jpg, jpeg, png, webp), uncaptioned, trigger-less or cut-off
  captions, little disk, a missing local model, an unreachable sync remote, checkpoints that will not
  be kept, when done set to stop/destroy with no remote for the results. `ok` is false when a check blocks.
- `DELETE /v1/jobs/{id}?files=true|false` → 204 (409 while running); the output folder is deleted only with `files=true`
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
- `GET /v1/jobs/{id}/samples` → `{items: [{name, path, step, index, modified, prompt?, seed?}]}` (`prompt` and
  `seed` from the config's sample section, the trigger filled in)
- `GET /v1/jobs/{id}/bundle?...` and `POST /v1/jobs/{id}/bundle/push`: see Run bundles

### Queues
- `GET /v1/queues` → `{items: [{gpu_ids, running, current_job_id, queued: n}]}`. The queue `cpu` uses no
  GPU: face checks run there, so one starts at once beside a training run.
- `POST /v1/queues/{gpu_ids}/start`, `/stop` → queue

### Datasets
`Dataset = {name, path, images, captions, size, modified}`
- `GET /v1/datasets` → `{items: [Dataset]}`
- `POST /v1/datasets {name}` → 201 Dataset (name used as given if it is a safe folder name, else 400)
- `GET /v1/datasets/{name}?cursor=&limit=` → Dataset + `{items: [{file, caption, size, width?, height?}], next_cursor, caption_job, photo_job}`
  (`width` and `height` are upright, after the EXIF orientation; `photo_job` is the newest face check, or null)
- `POST /v1/datasets/{name}/files?on_conflict=replace|rename|skip` multipart `files[]` → `UploadResult = {items: [saved names], renamed: [{from, to}], skipped: [..]}`:
  `replace` (default) overwrites a file of that name; `rename` saves a different file under a taken name as `<stem>_2<ext>`
  (then `_3`, ...) and skips the same content, and a `.txt` in the same request follows its image's new name; `skip`
  never overwrites
- `POST /v1/datasets/{name}/archive` body = zip (or multipart `file`) → `{added: n, skipped: [..]}` (safe extraction: no traversal, no symlinks, size and count limits)
- `PUT /v1/datasets/{name}/files/{file...}` body = the file's bytes → item (creates or replaces it; may contain `/`; hidden names refused)
- `POST /v1/datasets/{name}/import {url, file?, caption?}` → `{added, files: [..], skipped: [..]}`: the server downloads an http(s) URL; a zip is unpacked like `/archive`, an image/video is saved as `file` (default: from Content-Disposition or the URL, the extension sniffed from the bytes when missing) with `caption` written beside it
- `POST /v1/datasets/{name}/caption {captioner?, model?, prompt?, recaption?, overrides?, gpu_ids?}` → 201 Job (type `caption`, ref the dataset folder) plus `warnings`, already queued:
- `GET /v1/datasets/{name}/captions/check?trigger=` → `{items: [{file, caption, flags}], total, flagged, counts}`:
  flags are `missing`, `no_trigger` (no `[trigger]` and no trigger word: the trainer
  prepends the word itself), `trigger_case` (`[Trigger]`, or `Jihan` for `jihan`: matched
  exactly, so not replaced), `truncated` (ends on a dangling word or comma: raise
  `caption.max_new_tokens`), `hedging`, `preamble`, `duplicate`
- `POST /v1/datasets/{name}/captions/fix {ops, trigger?, files?, apply?}` → `{changes: [{file, before, after}], applied}`:
  ops run in order: `ensure_trigger {text?}` (puts `[trigger]` first where no trigger is),
  `fix_trigger_case`, `replace {find, with, regex?}`, `fill_missing {caption}`, `strip`.
  Without `apply: true` it only previews.
  one of upstream's captioner models writes a `.txt` next to each file as it goes, skipping files
  that have one unless `recaption`. The config is upstream's default caption job with the
  captioner selected the way its dataset page does (`handleCaptionerTypeChange`); `overrides`
  keys are relative to `config.process[0]` (`caption.max_res`). 409 while one runs for the
  dataset. `GET /v1/datasets/{name}` returns the newest as `caption_job`.
- `PUT /v1/datasets/{name}/captions/{file...} {caption}` → item
- `DELETE /v1/datasets/{name}/files/{file...}` → 204 (also removes its caption)
- `DELETE /v1/datasets/{name}` → 204 (409 while a queued or running job, or a caption job, reads it; its photo analysis is forgotten)

#### Photo checks
What is wrong with each photo for training, found before a GPU is paid for.
The server measures every stored photo once (sharpness, exposure, hashes,
EXIF, format), keeps the numbers by file content, and judges the set with the
same rules the studio applies in the browser before upload. The shapes
(`PhotoReview`, `PhotoFix`, `PhotoMetrics`, `PhotoFixOp`, ...) are declared in
full by `aitk_api` and in `studio/src/lib/photoTypes.ts`.

- `POST /v1/photos/assess {items, kind?, resolution?, dataset?, pick?, decisions?, captions?}` → PhotoReview:
  judges `PhotoMetrics` measured elsewhere (the studio measures before upload). At most 2,000 items
  and a 4 MiB body (413 over it). With `dataset`, that dataset's stored photos join the duplicate pass
  (they show only in `groups` and `same_as`), and the readiness is the dataset's once these photos are
  added: `usable`, `grade`, `grade_line`, `provisional`, `basis`, `coverage`, `findings` and `hints`
  judge them together with the stored photos measured so far (their captions, decisions and face results
  included), while `items`, `total`, `flagged`, `counts` and `groups` stay the posted photos'. `pick`
  picks among the posted photos; when too few pass, its `note` gives that combined readiness's hint.
  `items: []` is valid and returns `convert`, the formats this server converts.
- `GET /v1/datasets/{name}/photos/check?kind=&resolution=&pick=&wait_ms=` → PhotoReview: walks the
  dataset like the trainer (dot folders and `_controls/` skipped), lists the photos left out into
  `.set_aside/` separately as `set_aside` and the originals a convert or a bake kept there as
  `originals` (`{file, from, of}`, `of` the photo made from it), and lists formats the trainer skips (heic, heif, avif, jxl, bmp, gif, tiff) as
  `unsupported`. Photos not measured yet are analysed in the background; the call waits up to `wait_ms`
  (default 3000, max 20000), then answers with `pending > 0`, so poll until `pending == 0`. `resolution`
  is the run's, `1024` or `512,768,1024`. `kind` (`person`, `style`, `object`) defaults to the stored
  kind, then `person`. `pick=n` adds `pick`, the best n.
- `POST /v1/datasets/{name}/photos/fix {ops, files?, apply?, kind?, resolution?}` →
  `{changes: [{file, op, to?, reason?}], applied, review}`. A preview unless `apply: true`. Ops run in
  order: `set_aside {files}`, `restore {files | all}`, `set_aside_flagged {flags?}` (default
  `duplicate`, `tiny`, `unreadable`), `keep_best {n?}` (the best n for quality and variety; the rest are
  set aside), `bake_orientation`, `convert` (bmp, gif, tiff, heic where `convert` lists it, gray16,
  transparent, cmyk and huge to JPEG or PNG with the same stem; a JPEG is re-encoded at quality 95;
  for both, the original is kept in `.set_aside/` and listed in `originals`, not `set_aside`: it is a
  backup, so `restore {all}` leaves it, and `restore {files: [it]}` puts it back in place of the photo made
  from it, which is removed while it is as the convert or bake left it; one changed since stays, and the
  original comes back beside it),
  `rename_same_stem`, `face_crops {max?}`, `remove_face_crops`. An unknown op is 400 listing the valid
  ops. With `apply`, 409 while a training run (queued, running or stopping) reads the dataset or a
  caption job writes it; a face check never blocks. Every change drops the photo's thumbnails and its
  trainer caches (`_latent_cache`, `_t_e_cache`, `_clip_vision_cache`) and emits `dataset.changed`.
- `PUT /v1/datasets/{name}/photos/settings {kind?, reference?, decisions?}` → PhotoReview: the subject
  kind, the "this is them" reference photo (by file; null clears) and per-photo decisions by file:
  `keep` (keep anyway: waives its warnings) or `intended` (no face, sunglasses or black and white on
  purpose); null clears. They are stored by content, so they survive a rename.
- `DELETE /v1/datasets/{name}/photos/analysis` → 204: forgets the dataset's scores, face results,
  face-to-face similarities, reference and decisions. The photos stay. Deleting the dataset forgets them too.
  Scores and face results for photos that another dataset also holds are kept for that dataset.
- `POST /v1/datasets/{name}/photos/faces {text?}` → 201 Job (type `photo_faces`, ref the dataset folder,
  `gpu_ids` `cpu`) plus `warnings`, already queued: the face check (below). 409 while one is queued,
  running or stopping for the dataset; 400 when it has no photos or more than 2,000 (split the dataset).

Leaving a photo out moves it, with its caption and thumbnail, to `<dataset>/.set_aside/<file>`; nothing
is deleted. The trainer, the captioner, the dataset listing, bundles and portable sync skip dot folders,
and `restore` moves it back (`_2`, `_3`, ... on a name clash). Set-aside photos keep their analysis.

A review's `items` carry per photo `flags` (with a plain `notes` line for each), `level`, `usable`, and
for copies `group`, `keeper` and `same_as`. The flags:

| Flags | Level | Meaning |
|---|---|---|
| `unreadable`, `unsupported`, `huge` | warn | The trainer skips it: it reads JPEG, PNG and WebP only, under about 179 MP. |
| `gray16`, `transparent` | warn | Read wrongly: 16-bit grey clips to white; transparency is dropped and the hidden colour trains. |
| `same_stem` | warn | Two files with the same name before the extension share one caption. |
| `animated`, `cmyk`, `rotated`, `generated`, `derived` | info | First frame only; colours can shift; stored sideways (the auto-captioner reads it sideways); made by an image model; a face crop. |
| `duplicate` | warn | The same picture as another photo (resized, re-saved or mirrored); the group's `keep` is the better copy. |
| `similar`, `burst` | info | Close to another photo; taken within 2 s of another (a burst counts once toward variety). |
| `tiny`, `small` | warn | Under 512 px on the short side; trains below the run's resolution (the trainer never scales photos up). |
| `blurry`, `dark`, `bright`, `screenshot`, `bars` | warn | Soft even in its sharpest part; mostly shadow; large white areas; a screen capture; black or flat bars on the edges. |
| `upscaled`, `flat`, `grayscale`, `strong_color`, `noisy`, `compressed`, `from_screen`, `aspect` | info | Enlarged, low contrast, no colour, a strong filter, grain, low JPEG quality, a screen size with no camera data, very wide or tall. |
| `mentions_covered`, `mentions_people`, `mentions_text`, `mentions_tone`, `mentions_screen` | info | The caption mentions sunglasses or a hat, other people, text, black and white, a screen. |
| `no_face`, `others`, `small_face`, `face_blurry`, `not_them`, `watermark` | warn | From the face check. |
| `face_cut`, `backlit`, `identity_check`, `smoothed`, `text` | info | From the face check. |

The rest of the review:
- `findings`: dataset-level `{id, level, message, fix?, files?, op?}`, ranked block, warn, info. `op` is a
  ready-to-send fixPhotos op (`duplicates` → `set_aside_flagged` with `duplicate`). Ids: `count_none`
  (block), `count_very_low`, `count_low`, `count_high`, `duplicates`, `format_problems`, `low_res_share`,
  `to_look_at`, `two_people`, `identity_outliers`, `one_session`, `same_outfit`, `same_place`,
  `grayscale_mix`, `filter_consistent`, `people_in_captions`, `look_mismatch`, `era_span`,
  `orientation_tags`, `shot_gap`, `angle_gap`, `face_crops_offer`, `faces_unchecked`, `faces_stale`, `pending`.
- `usable`: photos not set aside, not a spare copy of a duplicate, and with no warning left (`keep`
  waives them). A burst counts once.
- `grade` (`grade_line` says it in words), the first rule that matches: F under 5 usable (or over 30 %
  of the faces look like someone else), D under 10, C under 15, two coverage gaps, two different people
  or a photo that may show someone else, B under 20 or one gap, A otherwise. A person set whose faces are not checked is `provisional` and gets B at best. The gaps are
  the coverage rules `shot_gap`, `angle_gap`, `one_session`, `same_outfit` and `same_place`, counted
  only where there is data (`basis` lists what the grade rests on).
- `coverage` (dated photos, sessions, span, and shot, angle, outfit and place counts where known) and
  `hints`: at most 3 things to add.
- `pick` (with `pick=n`, or `keep_best`): `kept` in pick order and `left_out` with one reason each. It
  favours quality and variety and takes at most one photo per burst. It never picks files the trainer
  skips (unreadable, unsupported, huge), even when marked `keep`. Unless marked `keep`, it also leaves
  out files the trainer reads wrongly (gray16, transparent), tiny photos, screenshots and photos with
  someone else in them (not_them, others). Default n: person 25, style 30, object 20.
- `faces`: the face check's state (`available` is false on a server without it), `thresholds` (the
  numbers the rules use) and `convert`.

Uploads are measured right away in the background, so a check after upload answers fast. `jobs.preflight`
adds photo lines (copies, low resolution, files read wrongly, the grade); they never block.

Privacy: photos and everything derived from them stay on this server and in the browser that shows
them. There is no third-party API. Logs carry file names and scores only, and photo-check data never
goes through notifications (face check jobs send no job messages at all). Check faces only on photos
of yourself or of someone who agreed. Scores and face-to-face similarity numbers are kept with the
dataset in the server database. Face templates are not stored: they exist only in the face check's
memory, and the similarities between this dataset's faces cannot recognise the person in any other
photo. They are cleared by Forget analysis, by deleting the dataset and by the next face check. The
SFace weights (Apache-2.0) were trained by their authors on public face datasets (CASIA-WebFace,
VGGFace2 and MS-Celeb-1M per the SFace paper; MS-Celeb-1M was withdrawn by Microsoft in 2019). The
check runs only when you click Check faces, and only on your server.

##### Face check
Faces, other people in a photo and look-alikes need a model, so they are a job rather than part of
every review. `photos/faces` queues a `photo_faces` job on the `cpu` queue. It runs the toolkit's
Python (numpy, Pillow and OpenCV 4.10 or newer; no torch) with a script the server writes into the job
folder, and takes about a minute for 30 photos on two CPU threads. It reads every photo the review
lists (set-aside ones are left out), finds faces with YuNet and compares them with SFace; with
`text: true` it also looks for text and watermarks with PP-OCR. Like Auto caption it never resumes on
its own, syncs or uploads, and it never blocks photo fixes.

- Follow it with `GET /v1/datasets/{name}` (`photo_job`, the newest face check) or the review's
  `faces` (`state`: `none`, `queued`, `running`, `done`, `failed` or `stopped`; `checked`, `stale`
  once the photos changed since it (added, removed, set aside or replaced), `added` (photos it has not
  seen)). The next `photos/check` reads the result in and deletes
  the job's `faces.json`; from then on the review carries the face flags, `two_people`,
  `identity_outliers` and the face crops offer.
- `warnings`: photos in a format the check cannot read (heic, heif, avif, jxl: convert them first),
  and the one-time model download.
- The models come from the OpenCV model zoo (github.com/opencv/opencv_zoo) the first time, pinned by
  sha256, into `<models>/opencv_zoo` (`<models>` is the models path setting): YuNet 2023mar (MIT,
  0.2 MB; the 2026may export on OpenCV 5 when that one does not load) and SFace 2021dec (Apache-2.0,
  39 MB), plus PP-OCRv3 English text detection (Apache-2.0, 2.4 MB) with `text`. The request sends
  no photo data.
- Check an install with `python -I photo_faces.py --selftest [--models DIR]` using the toolkit's
  Python (the script is `internal/photocheck/pyjob/photo_faces.py`, and every face check's job folder
  has a copy). It prints the OpenCV version and the detector it loaded, and exits 0 when it works.

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
- `GET /v1/remotes`, `POST /v1/remotes`, `DELETE /v1/remotes/{id}` (409 while a running job syncs or uploads to it), `POST /v1/remotes/{id}/test` → `{ok, detail}`
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
- `GET /v1/templates/captioners` → `{captioners: [{name, label, group, defaults, model_options, caption_prompts, additional_sections, python_accepted, ...}], default_config, options}`:
  upstream's captioners (`ui/src/helpers/captionOptions.ts`, `captionJobConfig.ts`), read
  the same way as the archs; `defaults` are the dotted paths selecting one sets,
  `additional_sections` the settings its dialog shows (`caption.caption_prompt`, `caption.max_res`, ...)

### Training presets
The studio's presets (a full job config tuned for a kind of run, the knobs the
wizard asks for, sample prompts and a caption prompt per subject) are kept on
each server the studio connects to, so MCP agents can use them. The studio
keeps its own copy and syncs both ways: the newer `updated_at` (ms) wins; a
deletion is a tombstone (`deleted: true`).
- `GET /v1/presets?deleted=` → `{items: [Preset]}`
- `GET /v1/presets/{id}` → Preset
- `PUT /v1/presets/{id} {name, base, description?, builtIn?, edited?, defaults?, promptSets?, captionTemplates?, notes?, updated_at?}` → `{preset, problems}` (`problems`: keys of `base` the trainer ignores)
- `DELETE /v1/presets/{id}?updated_at=` → 204

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
| `machine.state` | Machine (see GPU providers and machines; on a change of status, problem, base URL, price or name; never the token) |
| `machine.deleted` | `{id}` (a machine forgotten with `DELETE /v1/machines/{id}`) |

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
  kept: the job's `policy.keep_checkpoints` (N older ones besides the newest,
  `-1` all), else `AITK_SYNC_KEEP_CHECKPOINTS`, else `keep_checkpoints` on
  the remote's config. Keep them when you will pick the best step afterwards.
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
| `https://ntfy.sh/<topic>`; self-hosted `ntfy+https://<host>/<topic>`; auth `ntfy+https://:<token>@host/topic` or `user:pass@` | ntfy publish: title, click link, priority, tags; the image as an attachment |
| `https://discord.com/api/webhooks/<id>/<token>` | an embed (title links to the run) with the image attached |
| `telegram://<bot token>@<chat id>` | `sendMessage`, or `sendPhoto` with a caption |
| `pushover://<api token>@<user key>`, optionally `/<device>` | a message with the link as its URL and the image attached; failures at high priority |
| any other `https://...` (`webhook+` forces it) | JSON `{kind, level, title, body, url, job_id, job_name, at, server, image_jpeg_base64?}` |

Links open the run in the studio (`<studio>/#/runs/<id>`) or, without one, in
the console (`<public>/#/jobs/<id>`); lifecycle messages open
`<studio>/#/lifecycle`. The public URL is `AITK_PUBLIC_URL`, else RunPod's
proxy URL (`https://<pod id>-8675.proxy.runpod.net`) or Vast.ai's mapped port;
the studio is `AITK_STUDIO_URL`, else `<public>/studio/` when this server
serves it. Delivery is queued per sink and never blocks the server: 5
attempts with backoff (honoring `Retry-After`) on network errors, 429 and 5xx.

`NotifySink = {id, kind: ntfy|discord|telegram|pushover|webhook, target, source: env|api, created_at}`
- `GET /v1/notify` → `{sinks: [NotifySink], events: [kind] | null, kinds: [kind], public_url, studio_url, sample_every_s}`
- `PATCH /v1/notify {events?: [kind] | null, public_url?, studio_url?, sample_every_s?}` → same
- `POST /v1/notify/sinks {url}` → 201 NotifySink (400 for an unknown URL form, 409 when already there)
- `DELETE /v1/notify/sinks/{id}` → 204
- `POST /v1/notify/test {sink?, message?}` → `{items: [{sink, kind, target, ok, detail}]}` (one attempt each, synchronous; 409 without sinks)

Sinks from `AITK_NOTIFY` are `source: env` and are replaced at every boot;
API ones persist (fork setting `notify.config`).

### When done
`internal/lifecycle` watches the queues. When nothing is queued, running or
stopping and the training run that ended last ended `completed` (`on: any`:
also `failed`, or `interrupted` and not resumed within 2 minutes), it starts a
countdown (`delay_s`). A job queued meanwhile cancels it. Only training runs
arm it: a caption job holds it while it runs but its ending arms nothing (you
are still preparing a run), and an inference engine neither holds nor arms it.
At the deadline it waits while bundle imports/pushes or remote uploads are in
flight. An ending it already acted on (or held, failed or was cancelled for)
is remembered across a restart, so a restarted server does not act on it
again. At the deadline it checks every finished job (completed, failed, stopped,
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
destroys the server through the provider, detected from the environment
(or named by `AITK_WHEN_DONE_PROVIDER`: `runpod`, `vast`, `quickpod`,
`command`, `exit`):
- RunPod (`RUNPOD_POD_ID`): REST v2, `DELETE https://api.runpod.io/v2/pods/{id}`
  terminates and `POST .../v2/pods/{id}/action {"action": "stop"}` stops. v2
  may refuse the pod-scoped `RUNPOD_API_KEY` RunPod injects; until v1 retires
  on 2026-11-15 the same call is then retried on
  `https://rest.runpod.io/v1/pods/{id}`. A terminate the key may not do falls
  back to stop. `AITK_RUNPOD_API_KEY` supplies a full key instead.
- Vast.ai (`CONTAINER_ID` + `CONTAINER_API_KEY`): `DELETE
  https://console.vast.ai/api/v0/instances/{id}/`, stop is `PUT` with
  `{"state": "stopped"}`.
- QuickPod (`CONTAINER_LABEL`, `QUICKPOD_PORT_8675` and
  `AITK_QUICKPOD_API_KEY`, which QuickPod does not inject: a launch with
  `self_destroy_key` puts it there): `GET
  https://api.quickpod.org/update/api/destroypod?pod_uuid=` removes the pod
  and its disk (falling back to stop), `.../stoppod` stops it.
- `AITK_WHEN_DONE_CMD`, or a plain process exit.

With `dry_run` it only logs and emits events.

`Lifecycle = {mode: keep|stop|destroy, on: completed|any, delay_s, dry_run,
state, deadline, reason, provider: runpod|vast|quickpod|command|exit, job_id,
job_name, results: [{job_id, job_name, safe, via, detail}], updated_at}`;
`state` is `off` (keep) | `waiting` | `countdown` | `verifying` | `acting` |
`held` | `cancelled` | `done` | `failed`.
- `GET /v1/lifecycle` → Lifecycle
- `PATCH /v1/lifecycle {mode?, on?, delay_s?, dry_run?}` → Lifecycle (persisted, though `AITK_WHEN_DONE*` set in the env win again at boot; a change decides afresh: a countdown restarts with the new delay)
- `POST /v1/lifecycle/cancel` → Lifecycle (`cancelled` until another job runs and ends, also across restarts; 409 when nothing is pending)
- `POST /v1/lifecycle/extend {minutes}` → Lifecycle (409 without a countdown)
- `POST /v1/lifecycle/now` → Lifecycle (skip the countdown; 409 while a job is queued or running, or with mode keep)

Every change is a `lifecycle.state` event carrying Lifecycle.

## GPU providers and machines

This server can rent GPU servers ("machines") that run this image on RunPod,
Vast.ai and QuickPod, watch them until they answer, and hand out
their URL and token so the studio or an agent can use them
(`internal/provision`, one adapter per provider written from its published
API). Usually the studio API (docs/studio-api.md: no GPU, on a laptop, a VPS
or in Docker) holds the keys and the machines do the training. Provider calls never happen on a read of
`/v1/providers`; offers and quotes are cached for 30 s.

- **Keys.** One key per provider, stored with `PUT /v1/providers/{kind}`
  after the provider accepts it (fork setting `provision.key.<kind>`,
  write-only: responses show `key_hint`, the last 4 characters). Without a
  stored key `AITK_PROVISION_<KIND>_KEY` applies (`AITK_PROVISION_RUNPOD_KEY`,
  `..._VAST_KEY`, `..._QUICKPOD_KEY`), never the `AITK_<KIND>_API_KEY` a pod
  uses to end itself.
- **Kill switch.** `AITK_PROVISION_PROVIDERS=runpod,vast` enables only those;
  any other `{kind}` answers 404 `not_found` "unknown provider".
- **Simulate.** `AITK_PROVISION_SIMULATE=1` runs the three provider mocks in
  this process and refuses the real provider hosts: machines come up as
  loopback copies of this server, `simulated: true`, and cost nothing
  (`AITK_PROVISION_SIMULATE_DELAY`, default `15s`; see docs/providers.md for
  the `sim_*` fault keys). For local testing only.

Shapes (`studio/src/lib/provisionTypes.ts` is the exact contract):

- `Provider = {kind: runpod|vast|quickpod, label, stage: stable|beta,
  capabilities, unconfirmed: [sentence], console_url, keys_url, key_scopes,
  configured, key_source: api|env|sim|"", key_hint, max_running, running,
  simulated}`. `capabilities` says what the provider can do:
  `template_required` (QuickPod), `cloud_choice` and `count_choice` (RunPod),
  `stop_start`, `stop_keeps_disk` (false on RunPod: a stop wipes the
  container disk), `https` (`yes` RunPod, `no` Vast, `unconfirmed`
  QuickPod), `self_destroy` (`builtin` Vast, `key` the others), `credit`,
  the env limits, `image_digest_ref`, `reports_inet`, `reports_disk_bw`.
- `Offer = {id, provider, gpu, gpu_count, vram_gb, price_per_hour,
  stopped_price_per_hour, location, reliability, availability, max_disk_gb,
  cloud?, https, host, fit}`. `price_per_hour` is the whole machine while it
  runs, with the disk asked for; `host` (driver, CUDA, bandwidth, disk) and
  `fit` (does the image run there) are null until the image check fills them.
- `Machine = {id, name, label, provider, provider_id, status, problem,
  detail, provider_status, gpu, gpu_count, vram_gb, location, image,
  image_digest, image_launch_ref, image_warnings, disk_gb, base_url, secure,
  relay_path, price_per_hour, stopped_price_per_hour, when_done,
  self_destroy_key, copied_remote, actions, self, simulated, warnings,
  created_at, updated_at, ready_at, ended_at}`. `actions` is what works now
  (`stop`, `start`, `terminate`, `forget`); `self` marks the machine this
  server runs on. It never holds the token.
- `MachineLink = {name, base_url, token, secure, relay_path}`;
  `MachineLogs = {text, source, truncated}`; `ProviderTemplate = {id, name,
  image, ours, publishes_port, has_token}`; `SetupGuide = {kind, image, port,
  disk_gb, env: [{name, value, secret, note}], env_block, docker_options,
  fields: [{label, value, note}], steps, url_pattern, console_url, keys_url,
  key_scopes, token?}`.
- `ProviderInstance = {provider_id, label, status, provider_status, gpu,
  gpu_count, price_per_hour, machine_id, token_known, created_at}`: one
  `aitk-` labelled server on the provider account. `status` is the
  provider's view (never `ready`); `machine_id` is the live machine here that
  tracks it (null: untracked, and it may still be billing); `token_known`
  says the provider echoed its `AI_TOOLKIT_AUTH`, so adopting it needs no
  token; `created_at` is null when the provider does not say.

Providers:
- `GET /v1/providers` → `{items: [Provider]}` (no provider call)
- `PUT /v1/providers/{kind} {api_key?, max_running?}` → Provider (the key is checked with the provider first: 400 `provider_auth` and nothing stored when it is rejected; `max_running` 1-20, default `AITK_PROVISION_MAX_RUNNING` or 2). Guarded.
- `DELETE /v1/providers/{kind}` → 204 (the stored key; an env key still applies)
- `POST /v1/providers/{kind}/test` → `{ok, detail, credit}` (200 even when not ok)
- `GET /v1/providers/{kind}/offers?gpu=&min_vram_gb=&max_vram_gb=&max_price=&gpu_count=&cloud=&disk_gb=&limit=&image=&model=&show_all=&fast=` → `{items: [Offer]}`, cheapest first.
  `gpu` is a substring of the GPU name; `max_vram_gb` is exclusive (offers
  below it); `max_price` is $/h for the whole machine; `cloud` is RunPod's
  `secure` (default) or `community`; `disk_gb` (20-2000, default 100) prices
  the disk; `limit` 1-200 (default 50); `image` is the image the hosts must
  run (default `ghcr.io/geocine/aitoolkit-go:latest`); `model` is a model
  arch (`flux`, `qwen_image`, ...) for VRAM badges and the disk need;
  `show_all=1` keeps offers the image cannot run, marked in `fit`; `fast=1`
  keeps hosts with at least 500 Mbps download and 300 MB/s disk, where the
  provider reports them (an offer that reports neither passes).
- `GET /v1/providers/{kind}/templates` → `{items: [ProviderTemplate]}` (QuickPod launches need one with `ours` and `publishes_port`)
- `GET /v1/providers/{kind}/setup?image=&disk_gb=&cors_origin=&generate_token=1` → SetupGuide: the manual instructions for the provider's console (docs/providers.md). It needs no key and never accepts a token; `generate_token=1` fills in a fresh one (returned in `token`), otherwise a placeholder stays.
- `GET /v1/providers/{kind}/instances` → `{items: [ProviderInstance]}`, newest first: every server on the account whose label starts with `aitk-`, terminated ones left out (one provider call; 400 without a key). Env values are never returned.
- `POST /v1/providers/{kind}/adopt {provider_id, name?, token?}` → **201** Machine: tracks an untracked instance from that list here. Guarded.

Machines (`{id}` is a machine id or the name of a live machine):
- `GET /v1/machines?active=1` → `{items: [Machine]}` newest first (`active`: not ended)
- `POST /v1/machines` LaunchRequest → **201** Machine (`creating`); **200** with the same Machine for a repeated `request_id`. Guarded.
- `GET /v1/machines/{id}?refresh=1` → Machine (`refresh` re-reads it from the provider, at most every 5 s across callers)
- `PATCH /v1/machines/{id} {name?, token?}` → Machine (`token` is the AI_TOOLKIT_AUTH a template set; it clears `token_mismatch` and re-probes now)
- `POST /v1/machines/{id}/stop` → Machine (409 unless `stop` is in `actions`)
- `POST /v1/machines/{id}/start` → Machine (409 unless `start` is in `actions`). Guarded.
- `POST /v1/machines/{id}/terminate {confirm_self?}` → Machine (`terminating`). Deletes the machine and its disk at the provider; irreversible. Sending it again while `terminating` asks the provider again (for a delete it accepted but has not carried out). On the machine this server runs on (`self`) it needs `confirm_self: true` (else 400).
- `DELETE /v1/machines/{id}?force=1` → 204: forgets an ended machine (or a failed one that never got a provider id and is not `never_listed`; else 409, because it may still bill). `force=1` forgets it here only: it keeps running and billing at the provider.
- `POST /v1/machines/{id}/link` → MachineLink: the only response with the machine's token; 409 unless the machine is `ready` and enforces its token. Guarded.
- `GET /v1/machines/{id}/logs?tail=200` → MachineLogs (tail up to 2000 lines, text clipped to its last 256 KiB)

`LaunchRequest = {provider, offer_id, max_price_per_hour, request_id?, name?,
image?, disk_gb?, gpu_count?, cloud?, template_id?, cors_origins?,
studio_url?, copy_remote?, copy_hf_token?, copy_notify?, when_done?,
self_destroy_key?, env?, skip_image_check?}`:
- `provider`, `offer_id` (from offers) and `max_price_per_hour` are required;
  `request_id` is 8-64 of `[A-Za-z0-9_-]`;
- `name` defaults to `<gpu> on <provider>` (1-64 characters, unique among
  live machines); `image` defaults to `ghcr.io/geocine/aitoolkit-go:latest`;
  `disk_gb` is 20-2000 and at most the offer's `max_disk_gb`;
- `gpu_count` and `cloud` are RunPod only; QuickPod needs `template_id` from
  its templates;
- `cors_origins` (each `scheme://host[:port]`) default to the request's
  `Origin`; `studio_url` becomes `AITK_STUDIO_URL` on the machine;
- `copy_remote` names an S3 or Hugging Face remote whose settings and keys go
  into the machine's env (`AITK_S3_*` / `AITK_HF_*`, plus `AITK_SYNC_REMOTE`
  and `AITK_AUTO_RESUME=1`); a shared-folder remote gives 400.
  `copy_hf_token` and `copy_notify` copy this server's Hugging Face token and
  notification sinks;
- `when_done` (`keep`, `stop`, `destroy`) defaults to `destroy` when
  `copy_remote` is set and destroy is possible (on QuickPod only with
  `self_destroy_key`), else `keep` (with a warning when a remote is set on
  QuickPod without the key). `stop` or `destroy` without `copy_remote` is 422
  `invalid_config`, because results must be kept somewhere;
- `self_destroy_key` puts the account key into the machine's env so it can
  delete itself (RunPod, QuickPod; Vast has its own per-instance key);
- `env` adds `AITK_*`, `HF_*`, `HUGGING_FACE_HUB_TOKEN` or `TZ` variables;
  reserved names (`AI_TOOLKIT_AUTH`, `AITK_CORS_ORIGINS`, `AITK_WHEN_DONE*`,
  `AITK_PROVISION_*`, `AITK_*_API_KEY`, provider prefixes, ...) and values
  over a provider's limits are 422 naming the variable, never echoing it;
- `skip_image_check` is honoured only for custom images or registries that
  cannot be read, never to skip a host-side block.

Every machine gets its own random `AI_TOOLKIT_AUTH`. Env values are never
stored or returned; the machine row keeps only their names.

Money rules:
- A launch rents a GPU and costs money from the moment it returns. Show the
  offer's `price_per_hour` and get the user's OK first.
- The offer is quoted again just before the launch; a price above
  `max_price_per_hour` is 409 `conflict` ("The price is now $0.52/h, above
  your limit of $0.47/h ...").
- `request_id` makes a retry return the same machine instead of renting twice.
- A create the provider did not answer clearly (a timeout, a 5xx, an
  unreadable answer) may still have been made: it returns the machine as
  `creating` with a `detail` saying so, and the watcher picks it up by its
  label when it appears (`never_listed` after 5 minutes if it does not).
- `max_running` per provider (default 2) caps live machines: 409 `conflict`
  at the cap.
- This server never terminates a machine on its own: a machine that is not
  ready in 20 minutes (`AITK_PROVISION_READY_TIMEOUT`) is `failed` with
  `not_ready_timeout` and is still billed until it is terminated.
- A stopped machine may still bill for its disk (`stopped_price_per_hour`).

Statuses (`status`); the watcher refreshes transitional ones every 10 s,
ready ones every minute and stopped ones every 5 min:

| status | meaning |
|---|---|
| `creating` | accepted; the provider is allocating (QuickPod: until the pod is listed) |
| `starting` | host assigned; the image is pulling or the container starting |
| `running` | the provider says running; the server does not answer with its token yet |
| `ready` | `GET /v1/system` on the machine answered 200 with its token |
| `stopping`, `stopped` | compute released; the disk may still bill |
| `terminating`, `terminated` | terminate sent / gone at the provider |
| `failed` | the launch was refused, the provider reported an error, or it was not ready in time |
| `unknown` | a provider state this server cannot map (raw value in `provider_status`) |

Problems (`problem`, with the human text in `detail`):

| problem | meaning |
|---|---|
| `token_mismatch` | the server answers but refuses our token (the template set its own AI_TOOLKIT_AUTH): `PATCH` it in |
| `open_server` | the server answers without any token; it is ready but cannot be linked |
| `no_port` | running over 3 min without a public port for 8675 |
| `not_ready_timeout` | not ready in time; still billed |
| `unreachable` | was ready, failing probes for over 5 min |
| `never_listed` | the create was accepted (QuickPod) or did not answer clearly, and no server appeared in 5 min; it may be billing, so it still counts toward `max_running` and forgetting it needs `force=1` |
| `provider_error` | the provider reports an error state |
| `not_refreshed` | a real machine while simulate is on (or the reverse); actions on it are 409 |
| `cuda_failed`, `pull_slow`, `image_pull_failed` | the image could not run its CUDA check, is pulling far slower than the host predicts, or could not be pulled |

Errors from providers: 400 `provider_auth` (the provider rejected the key;
never 401, which means this server's own token), 402 `insufficient_credit`,
409 `no_capacity` (the offer was taken: pick another), 409 `conflict`
(action not valid now, price rose, cap reached), 422 `invalid_config` (when
done or env rules), 422 `image_blocked` (the image cannot run on that host;
the checks are in `error.details`), 429 `rate_limited` with `Retry-After`
(the watcher pauses that provider too), 502 `remote_failed` (the provider did
not answer). Provider messages are passed through with any key redacted.

**Open-server guard.** On a server without `AI_TOOLKIT_AUTH` and without API
keys, `PUT /v1/providers/{kind}`, `POST /v1/providers/{kind}/adopt`,
`POST /v1/machines`, `.../start`, `.../link` and the relay answer 403
`forbidden` ("Set AI_TOOLKIT_AUTH on this server before it can rent GPUs for
you.") unless the request comes straight
from this machine (a loopback peer and no `X-Forwarded-For`, `X-Real-IP` or
`Forwarded` header). Code mode over `/mcp` counts as the MCP client's own
connection: an agent on this machine needs no token, one connecting from
elsewhere or through a proxy is refused.

**Relay** (for plain-http hosts, Vast and QuickPod, that a studio served over
https cannot call):
- `GET /v1/machines/{id}/relay/{path...}`
- `POST /v1/machines/{id}/relay/{path...}`
- `PUT /v1/machines/{id}/relay/{path...}`
- `PATCH /v1/machines/{id}/relay/{path...}`
- `DELETE /v1/machines/{id}/relay/{path...}`

each forward the request to `<base_url>/<path>` with its query and body
(streamed, SSE flushed). `Origin`, `Cookie`, `X-Forwarded-*` and
`Authorization` from the client are dropped and `Set-Cookie` from the
machine; `X-Forwarded-Prefix: /v1/machines/<id>/relay` lets the machine build
links that point back through the relay. A request that passed this server's
auth gets `Authorization: Bearer <machine token>`; the public media GETs and
signed links that reach the relay without a token are forwarded with no
credentials, so the machine decides. 409 `conflict` unless the machine is
ready with a `base_url`; 502 `remote_failed` when it cannot be reached; a
private address is refused unless the machine is simulated. Each request
costs two hops, and this server's `AITK_CORS_ORIGINS` applies. The relay has
no `aitk` op: agents use `machines.link` and talk to the machine (or its
`/mcp`) directly.

**Finding and adopting servers.** Machines live in this server's database.
If it is lost (or a machine was forgotten with `force=1`), the machines keep
running and billing at the provider under their `aitk-` labels. On any
server with the same provider key, `GET /v1/providers/{kind}/instances` lists
them and `POST /v1/providers/{kind}/adopt` tracks one again:
- the token comes from the provider's echo of the instance env
  (`AI_TOOLKIT_AUTH` in the RunPod pod `env` or Vast's `extra_env`) unless
  `token` is given. QuickPod returns no env, so `token` is required there,
  as it is when an echo is a RunPod secret reference
  (`{{ RUNPOD_SECRET_... }}`): 400 `invalid_request` without it. A wrong
  token gives the machine `token_mismatch`; `PATCH` the right one in;
- the machine gets back the id it was launched with (its `AITK_MACHINE_ID`,
  else the one its label was made from) when no row here has it, so a
  studio entry that pointed at it works again; `when_done` and
  `copied_remote` come from `AITK_WHEN_DONE` and `AITK_SYNC_REMOTE`, and
  `self_destroy_key` from whether an `AITK_*_API_KEY` is set; no other env
  value (tokens, keys) is stored, only the env names in `env_keys`. `image`
  and `disk_gb` were not recorded and stay empty;
- `name` defaults to `<gpu> on <provider>`; `created_at` is the adoption, so
  the 20-minute ready timeout counts from then. It is probed at once and
  watched like any other machine;
- 404 `not_found` for an id that is not an `aitk-` server on the account,
  409 `conflict` when a live machine here already tracks it. `max_running`
  does not apply: the server already bills.

### GPU recommendation

The recommender (`internal/gpupick`) suggests the default offer: **best
value** (most training speed per $/h) and **best speed** (fastest), from
the smallest VRAM tier the model fits, among offers that can run the image
fast. It only reads offers and never rents anything: the studio preselects
the value pick, and an agent shows both picks and asks before
`machines.launch`. Every tier query goes through the 30 s offers cache, so a
recommendation right after an offers request of the same shape costs no
provider call. Results are not cached themselves.

- `GET /v1/providers/{kind}/recommend?model=&image=&disk_gb=&cloud=&max_price=` → GpuRecommendation for one provider. 404 for an unknown or disabled `{kind}`, 400 `invalid_request` when the provider has no key ("Add a RunPod API key first"); a provider error ends the run with that error (429 `rate_limited` with `Retry-After`, 402, 400 `provider_auth`, 502).
- `GET /v1/hosts/recommend?providers=runpod,vast&model=&image=&disk_gb=&cloud=&max_price=` → GpuRecommendation across every enabled provider with a key, or the comma list in `providers`. A provider that fails is a note, not the status: the others still give picks. 400 when no listed provider has a key ("Add an API key for RunPod, Vast.ai or QuickPod first"), for a name that is not a provider, or when every listed provider is turned off by `AITK_PROVISION_PROVIDERS` (`providers: "quickpod" isn't enabled on this server`); a turned-off provider listed beside others is a `not_configured` note.

`model` is a model arch (`flux`, `qwen_image`, `wan22_14b`, ...); an arch
the VRAM table does not know is accepted, starts at 24 GB and adds the note
`model_unknown`. `image` is the image the picks are judged against (default
`ghcr.io/geocine/aitoolkit-go:latest`; the studio sends its Image row, or on
QuickPod the template's image). `disk_gb` is 20-2000 (default: what the
image and model need) and `max_price` is above 0. The code-mode ops are
`providers.recommend(kind, q)` and `hosts.recommend(q)`.

The rules:

1. **Disk.** The disk asked for is the larger of `disk_gb` and what the
   image and model need (`need.disk_gb`, at least 100 GB without a known
   model). When the image cannot be read, the default image's built-in
   requirements stand in and a note with the image check code
   (`image_denied`, `image_not_found`, `image_unreachable`) says so; a
   missing or private image also puts its sentence first in `message`.
2. **Ladder.** VRAM tiers 24, 32, 48, 80, 96, 141 and 180 GB. The walk
   starts at the smallest tier where the model fits or is tight (rounded up
   to the ladder, never below 24; 80 when it fits nowhere up to 80; 24 with
   no model). An offer's tier is the largest `L` with `vram ≥ L × 0.9`
   (host VRAM in GiB when known), so 22.49 GiB L4 and A10 cards are 24 GB
   cards, and 20 GB cards are below the ladder and ignored. There is one
   offers query per tier and provider (`gpu_count=1`, `fast=1`, `limit=200`,
   the tier's VRAM band), and the walk stops at the first tier with a pick:
   a higher tier is searched only when the lower one has nothing.
3. **Qualify.** An offer is left out for the first rule it fails, in this
   order:

   | code | left out when |
   |---|---|
   | `gpu_count` | it has more than one GPU (each job uses one) |
   | `price` | `max_price` is set and the offer costs more |
   | `cuda` | the driver is below the image's CUDA (`host_cuda`) |
   | `cuda_unknown` | the driver is unknown (`host_cuda_unknown`) |
   | `cuda_capacity` | RunPod has no free host with the CUDA it needs (`runpod_no_cuda_capacity`) |
   | `gpu_arch` | the image has no kernels for the GPU (`gpu_sm`), the GPU is sm_75 (`gpu_sm_flash_attn`, hidden by default), not NVIDIA (`gpu_vendor`) or unknown (`gpu_unknown`), or the host CPU is not amd64 (`host_arch`) |
   | `disk` | the host cannot hold the image (`disk_image`), or its `max_disk_gb` is below the disk asked for |
   | `inet` | it **reports** a download speed below 500 Mbps |
   | `disk_bw` | it **reports** a disk speed below 300 MB/s |
   | `reliability` | it reports a reliability below 0.95 |
   | `fit_block` | anything else blocks, or the offer was not checked against the image (note `not_decorated`) |
   | `speed_unknown` | the GPU has no speed rating |

   RunPod passes the CUDA rule through `host_cuda_placement`: RunPod itself
   places the pod on a host with the image's CUDA. A download or disk speed
   that is 0 or missing means "not reported" and passes (RunPod never
   reports them, and some QuickPod rows have none); only a reported value
   below the threshold leaves an offer out.
4. **Thresholds** (`thresholds` in the answer). 500 Mbps pulls the 6.54 GB
   image in about 3.5 min, where the 73 Mbps hosts seen on Vast and
   QuickPod take about 25 min; at 300 MB/s the disk stops being the
   bottleneck of the unpack. Their units are UNCONFIRMED: Vast's
   `inet_down` is read as Mbps and `disk_bw` as MB/s, and QuickPod's units
   are unknown. The 0.95 reliability floor is ours, not a provider's
   (UNCONFIRMED tuning; Vast reports 0-1, QuickPod 0-100, normalised to
   0-1).
5. **Rank** within the tier. Value is the highest `per_dollar = speed /
   price_per_hour`; when two offers are within 2 % of each other, one that
   reports both its download and disk speed ranks first. Speed is the
   highest `speed` (RTX 4090 = 1.00, as in `hosts.gpus`). Equal scores
   break by lower price, then higher reliability (none lowest), then
   provider order (RunPod, Vast.ai, QuickPod), then offer id. `default` is
   `value` when there is a pick; when one offer is both, `value` and
   `speed` hold the same offer.
6. **Reason** (`reason`, the one line to show with the pick):
   `{gpu} · {vram} GB · ${price}/h · {net} · {cuda}`. `net` is the reported
   download speed (`912 Mbps`, followed by `disk speed not reported` when
   only the disk speed is missing) or `net not reported`; `cuda` is `CUDA
   13.2 OK` for a host that reports its CUDA, or `CUDA 13.0+ by RunPod`. In
   a cross-provider answer ` on {provider}` follows the GPU: `RTX 4090 on
   Vast.ai · 24 GB · $0.34/h · 912 Mbps · CUDA 13.2 OK`.
7. **Nothing qualifies.** `value` and `speed` are null, `default` is `""`
   and `message` says why, from the run: "No offer from 24 to 180 GB has
   fast internet, enough disk (100 GB) and a driver for CUDA 13.0 right
   now. Most were left out for slow internet (12). Pick one yourself from
   the list.", or, when no offer came back at all, "No offers match these
   filters (disk 2000 GB). Change the filters or pick a provider with more
   stock." The offer list stays the way to pick by hand.

`GpuRecommendation = {providers, model, start_tier_gb, tier_gb, default:
value|"", value, speed, need, tiers, notes, thresholds, message}`
(`studio/src/lib/provisionTypes.ts` is the exact contract):
- `value` and `speed` are `GpuPick = {kind: value|speed, offer, tier_gb,
  speed, speed_source, per_dollar, reason}` or null. `offer` is the whole
  Offer as `providers.offers` returns it, so a client can show and launch
  it even when its own list does not hold it;
- `providers` is what was searched; `tier_gb` is the tier the picks come
  from (0 for none); `need` is the DiskNeed;
- `tiers: [{tier_gb, offers, qualified, rejected: {code: count}}]` is each
  tier walked, in order;
- `notes: [{provider, code, message, offers}]`: `not_configured` (no key,
  or not enabled), a `/v1` error code for a provider that failed
  (`rate_limited`, `provider_auth`, `remote_failed`, ...), `not_decorated`,
  and, with `provider: ""`, the image and `model_unknown` notes;
- `thresholds = {min_inet_down_mbps, min_disk_bw_mbps, min_reliability,
  ladder}`.

Arrays and records are never null. An agent picks, then asks:
```ts
const rec = await aitk.providers.recommend("vast", { model: "flux" });
if (!rec.value) return rec.message; // nothing qualifies: say so, and let the user pick from providers.offers
return { value: rec.value.reason, speed: rec.speed.reason, price: rec.value.offer.price_per_hour };
// after the user's OK: machines.launch({ provider: "vast", offer_id: rec.value.offer.id, max_price_per_hour, request_id })
```

## Images and hosts

Before a GPU server is rented this server reads what its image needs from
the registry, without pulling it, and matches that to the host: CPU
architecture, GPU vendor and compute capability, driver and CUDA, disk,
pull time and VRAM (`internal/imagecheck`, `internal/hostfit`). Offers carry
the result as `host` and `fit`, `POST /v1/machines` runs the same checks
again (422 `image_blocked` with the checks in `error.details`, and no
provider call is made), and a ready machine runs a CUDA self-check.

- **Registries.** GHCR and Docker Hub only, with anonymous pulls (images
  must be public). Any other registry is never contacted: the check is the
  warn `image_unchecked` and a launch is allowed. Registry trouble
  (unreachable, Docker Hub's limit nearly used) is a warn, never a block,
  and a check never answers 5xx because a registry is down.
- **Caches.** A tag's digest is kept 10 min, a missing or private tag 60 s,
  image facts by digest for the process lifetime; `refresh` re-resolves.
- **Not published yet.** While `ghcr.io/geocine/aitoolkit-go` cannot be
  read (not published, private, unreachable), its built-in requirements
  (`requires.source: "fallback"`: CUDA 13.0, linux/amd64, sm_75 to sm_120)
  still judge hosts, so offers and recommendations work. The check says
  once that the image isn't published yet; launches of it stay blocked.
- **Simulate.** With `AITK_PROVISION_SIMULATE=1` the registry and Hugging
  Face are in-process mocks (`AITK_PROVISION_SIMULATE_REGISTRY=real` reads
  the real registry).

Shapes (`studio/src/lib/serverImageTypes.ts` is the exact contract):

- `ImageCheckItem = {level: block|warn|info, code, message, fix?, action?}`;
  `action = {kind: filter_hosts|show_all|set_disk|use_image|pick_gpu|edit_template, value?}`
  is a button the studio shows next to the check.
- `ImageCheck = {ok, level: ok|info|warn|block, image, launch_ref, pinned,
  host, need, pull, vram, checks}`. `ok` is false when any check blocks;
  `image` (ImageFacts) is null when the image could not be read;
  `launch_ref` is the reference a launch sends (a `:<sha>` build tag when
  the image has one), null when anything blocks; `pinned` is
  `repo@sha256:...`.
- `ImageFacts = {ref, repo, tag?, digest, platform_digest, platforms,
  platform, download_bytes, largest_layer_bytes, layers, unpacked_bytes,
  unpacked_source, created, labels, env, exposed_ports, go_server, revision?,
  requires, checked_at, rate_limit?}`. `labels` keeps only
  `org.opencontainers.image.*` and `io.github.geocine.aitk.*`, and `env` only
  `CUDA_VERSION`, `NVIDIA_REQUIRE_CUDA` and `NVIDIA_DRIVER_CAPABILITIES`, so
  nothing secret in an image is echoed.
- `requires = {platform, cuda_min, driver_min, compat_branches, sm, ptx,
  flash_attn_sm, torch, flavor, port, api, sources, source?}`.
- `HostFacts = {provider?, gpu_name?, gpu_count?, vram_gb?, sm?, sm_source?,
  gpu_class?, gpu_vendor?, cpu_arch?, cuda_max?, cuda_source?, driver?,
  disk_gb?, inet_down_mbps?, disk_bw_mbps?, ram_gb?, placement_floor?}`; a
  missing field is unknown. From a caller, `gpu_name` alone fills in the
  GPU's architecture, class, vendor and VRAM from the GPU table.
- `DiskNeed = {disk_gb, image_peak_gb, model_gb, work_gb, model_source:
  hf|table|none|default}` (`disk_gb` is what a launch should ask for, at
  least 100 without a known model); `PullEstimate = {image_min, model_min,
  basis, estimate}`; `VramFit = {verdict: fits|tight|no|unknown, note,
  source: repo|vendor|estimate, recipe?}` (tight names the recipe that makes
  it fit; VRAM never blocks a launch).
- `ImageTag = {tag, digest, created, flavor?, cuda_min?, revision?, moving}`;
  `ImageChoice = {channel: same|latest|flavor|custom|template, label, image,
  launch_ref, digest, revision?, flavor?, level, why}`; `ImageRecommendation
  = {choice, alternatives, checks, action?}`; `GpuInfo = {name, sm, vram_gb,
  class, vendor, platform, aliases, speed, speed_source, verdict, note}`;
  `HostFit = {model, tiers: [{vram_gb, verdict, note, source, recipe?}],
  need}`; `CudaCheck = {ok, checked_at, torch, cuda, arch_list, devices:
  [{index, name, capability, driver}], error, failure}`.

Routes:
- `POST /v1/images/check {image, platform?, host?, offer?: {provider, offer_id}, model?, disk_gb?, refresh?}` → ImageCheck, 200 even when it blocks (400 only for an empty `image`). `offer` re-reads that offer's host through the 30 s quote cache and wins over `host`; with neither only the image-side checks run. `model` is the job's model section (`ModelRef = {arch, name_or_path?, quantize?, qtype?, low_vram?, resolution?, frames?}`) for the disk need and VRAM; `disk_gb` is the disk the launch will ask for.
- `GET /v1/images/tags?image=ghcr.io/geocine/aitoolkit-go&limit=8` → `{items: [ImageTag]}`, newest first: `latest`, the flavor channels, versions and recent build tags (`limit` up to 20). 400 for a bad or unchecked reference, 404 for a missing or private repo, 502 when the registry cannot be read.
- `POST /v1/images/recommend {offer?, host?, model?, prefer?, image?, template_id?, disk_gb?}` → ImageRecommendation. The choice is the build this server runs when it is published and fits (`prefer: "latest"` tries the newest first), else the newest build, else another flavor; a different trainer version is never chosen silently (its first `why` line says so). `choice` is null when every candidate blocks, with `action: filter_hosts` when a host filter would fix it. `image` checks a custom image as the only candidate; `template_id` (QuickPod) checks the image that template runs.
- `GET /v1/hosts/gpus` → `{items: [GpuInfo]}`: the GPU table (compute capability, nominal VRAM, class, the provider names it matches, relative training `speed` with RTX 4090 = 1) with the default image's `verdict` on each GPU and a `note`. The driver is a fact of each host, so `images.check` judges it, not this list; sm_75 GPUs are a warn because offers hide them by default.
- `POST /v1/hosts/fit {model, image?, gpus?: [{name?, vram_gb?, count?}], host?}` → HostFit: the VRAM verdict for each GPU given (a known `name` gives its VRAM) or, with none, for 16, 24, 32, 48 and 80 GB, and the disk the model needs with `image`. 400 without `model.arch`.
- `GET /v1/system/cuda?refresh=1` → CudaCheck, on the machine itself: runs a tiny CUDA kernel through the trainer's torch on every GPU (90 s at most, one run at a time, a success is cached; `refresh=1` runs it again). `failure` names the cause: `no_kernel_image`, `driver_too_old`, `no_gpu`, `torch_missing` or `timeout`. The provisioning service calls it once a machine is ready (`cuda_failed` when it fails; an older image without the route is skipped).

Check codes (`code`), by level:

| level | codes |
|---|---|
| block | `image_ref`, `image_denied` (private or missing; for this server's image, not published yet), `image_not_found` (`use_image` offers the newest build), `image_platform`, `image_not_go` (no Go server for the studio to talk to), `image_port`, `image_labels` (the `image-check --require-go` CLI only), `host_cuda`, `host_arch`, `gpu_vendor`, `gpu_sm`, `disk_image` |
| warn | `image_api`, `image_unchecked`, `image_unreachable`, `image_rate_limited`, `host_cuda_unknown`, `runpod_no_cuda_capacity`, `gpu_sm_flash_attn`, `gpu_sm_hidden` (hosts.gpus only), `disk_model`, `pull_slow`, `vram_no` (an info at launch) |
| info | `image_sm_unknown`, `image_unpinned`, `image_skew`, `image_same_unpublished`, `image_template_tag`, `host_cuda_placement`, `gpu_sm_unverified`, `gpu_unknown`, `pull_eta`, `vram_tight`, `ram_tight` |

An offer's `fit` is `{level, codes, summary, pull_min, vram}`: the same
host-side codes and a one-line summary ("Driver too old (CUDA 12.8)",
"≈ 3 min to pull"). The image's own failure is never in an offer's `fit`;
`images.check` shows it once.

The code-mode groups are `images` (`check`, `tags`, `recommend`), `hosts`
(`gpus`, `fit`, and `recommend` from GPU recommendation) and `system.cuda`. An agent checks before it rents:
```ts
const offers = (await aitk.providers.offers("vast", { model: "flux", limit: 5 })).items;
const offer = offers.find((o) => o.fit?.level !== "block");
const ref = { provider: "vast", offer_id: offer.id };
const rec = await aitk.images.recommend({ offer: ref, model: { arch: "flux" } });
if (!rec.choice) return rec.checks; // nothing fits this host; rec.action says what to change
const check = await aitk.images.check({ image: rec.choice.image, offer: ref, model: { arch: "flux" } });
return { ok: check.ok, launch_ref: check.launch_ref, disk_gb: check.need?.disk_gb, checks: check.checks };
```

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
routes; only `GET /v1/events/stream`, `/v1/openapi.json` and the machine
relay are left out: `events.list({ after, wait_ms })` long-polls instead,
`aitk_api` replaces the OpenAPI document, and an agent reaches a machine
through `machines.link`). Bodies that are not JSON:
- binary in: base64 params, 16 MiB per call (`datasets.putFile`,
  `datasets.uploadFiles`, `datasets.uploadArchive`, `bundles.uploadChunk`),
  or let the server fetch a URL (`datasets.importUrl`, `bundles.import({ url })`);
- binary out: ops returning `Link` (`files.link`, `bundles.download`,
  `links.create`) sign the download instead of fetching it, so the agent
  gets a URL it can hand to a person or to another server;
- `bundles.wait(id)` polls an import or push until it settles, like `jobs.wait`.

Photo checks are `datasets.checkPhotos` (poll while `pending > 0`), `datasets.fixPhotos`,
`datasets.photoSettings`, `datasets.forgetPhotos`, `datasets.checkFaces` and `photos.assess`
(Datasets, Photo checks). An agent can send a finding's `op` to `fixPhotos` as it is, and look at a
flagged photo with `aitk_view`. After `checkFaces`, poll `datasets.get` until `photo_job` settles, then
call `checkPhotos` for the face flags; only check faces on photos of the user or of someone who agreed.

GPU servers are the `providers` group (`list`, `setKey`, `removeKey`, `test`,
`offers`, `recommend`, `templates`, `setup`; `hosts.recommend` searches every
provider) and the `machines` group (`list`, `launch`,
`get`, `update`, `stop`, `start`, `terminate`, `forget`, `link`, `logs`, and
`wait(id, {until?, timeout_ms?, poll_ms?})`, which re-reads the machine until
its status is in `until`, default `ready`, `failed`, `stopped`, `terminated`,
or 100 s pass, and returns the last Machine). The agent asks before it
spends money:
```ts
const offer = (await aitk.providers.recommend("runpod", { model: "flux" })).value?.offer;
if (!offer) return "nothing qualifies: pick one with providers.offers";
// tell the user offer.gpu and offer.price_per_hour, and launch only after they agree
const m = await aitk.machines.launch({ provider: "runpod", offer_id: offer.id,
  max_price_per_hour: offer.price_per_hour, request_id: "lora_run_0001", copy_remote: "r2" });
const ready = await aitk.machines.wait(m.id); // up to 100 s; call it again in a later run while it is still starting
if (ready.status !== "ready" || ready.problem) return ready;
return await aitk.machines.link(m.id); // { base_url, token }: connect to that server's /mcp
```

Example the agent might send to `aitk_run`:
```ts
const ds = await aitk.datasets.create({ name: "my_set" });
const { config, warnings } = await aitk.templates.render({ arch: "sdxl", name: "my_lora", datasets: [ds.name],
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
