# Run bundle format (format 1)

A run bundle is one zip that brings a training run to any server: unzipped
into a toolkit it is exactly the layout ai-toolkit resumes from, plus the
dataset it trains on and a manifest that says how to put it back. This
server writes it (`GET /v1/jobs/{id}/bundle`, `POST /v1/jobs/{id}/bundle/push`)
and reads it (`POST /v1/bundles/imports`, `AITK_BOOTSTRAP`); docs/headless.md
has the API. The reader and writer are `internal/bundles`.

## Layout

```
aitk_bundle.json            the manifest (first entry, stored)
data/                       the job's first dataset
data_1/, data_2/, ...       its further datasets (data_<i> for config index i)
output/<name>/              the job folder, as ai-toolkit wrote it
```

Every path is relative and slash-separated; no entry may be absolute, contain
`..`, or be a symlink (the whole bundle is refused otherwise). Hidden names
(any path segment starting with `.`) are never part of a bundle and are
skipped on import: `.job_config.json` (regenerated), `.thumbs`,
`.aitk_size.json`, `.ipynb_checkpoints`, and so are `__MACOSX/`, `Thumbs.db`,
`desktop.ini`, `output/generations/` and files at the root other than the
manifest. If every entry sits under one shared top folder that is not
`data`, `data_<n>` or `output` (`run.zip -> run/data/...`), that folder is
stripped first.

Writers store every entry (method 0, no compression): checkpoints and images
do not compress, and stored sizes make the zip's size known before it is
written (downloads send an exact `Content-Length`). Readers accept stored
and deflated entries (hand-made zips are usually deflated). Zip64 is used
past 4 GiB.

### Job folder: `output/<name>/`

`<name>` is the job's `config.name`, which is also its folder name under the
training folder. What ai-toolkit writes there and reads back on resume
(`jobs/process/BaseSDTrainProcess.py`, `BaseTrainProcess.py`):

| Entry | Meaning |
|---|---|
| `<name>_<step>.safetensors` | a save at `<step>`, zero-padded to 9 digits (`lora_000000400.safetensors`). `.pt`/`.ckpt` saves are named the same way. |
| `<name>.safetensors` | the final save, written when training completes |
| `<name>_<step>/`, `<name>/` | a save as a diffusers folder (full fine-tunes with a diffusers `save_format`); its step is in `aitk_meta.yaml` (`training_info.step`) |
| `optimizer.pt` | optimizer state of the newest save (one file, overwritten at every save) |
| `config.yaml` | the config ai-toolkit wrote at its last start: the authority for a bundle without a manifest |
| `samples/<ms>__<step>_<i>.<ext>` | samples |
| `log.txt`, `loss_log.db`, `learnable_snr.json`, ... | everything else the trainer wrote; carried as is (`loss_log.db` is a consistent snapshot) |

How ai-toolkit resumes, which the format follows exactly:
1. `get_latest_save_path` globs `<training_folder>/<name>/<name>*.safetensors`,
   `<name>*.pt` and `<name>*` (folders), and takes the match with the
   **newest ctime**. Not the highest step: file time.
2. `load_training_state_from_metadata` reads the step to continue at from
   that save's safetensors metadata, `training_info` (a JSON string) `.step`
   (`aitk_meta.yaml` for a diffusers folder). `train.start_step` in the
   config overrides it.
3. `optimizer.pt` beside it is loaded when present (and the network did not
   change shape); without it training resumes with a fresh optimizer.

So a writer puts the saves **last, oldest first** (stepped saves by step,
the final save after them), and a reader extracts entries in zip order. The
newest save then has the newest ctime, after an import or a plain `unzip`.
Renaming a job renames its saves (`<old>_<step>` to `<new>_<step>`, `<old>`
to `<new>`, files and folders) so the `<name>*` glob still finds them; the
step inside the metadata does not depend on the name.

### Datasets: `data/`, `data_<i>/`

A dataset folder as ai-toolkit reads it: media files (`.jpg .jpeg .png
.webp .bmp .gif .tif .tiff`, videos, audio) with a caption per file
(`<stem>.txt`, the config's `caption_ext`), in any subfolders. The caches
ai-toolkit writes next to the images are included by default because
re-caching big models is slow: `_latent_cache/` (VAE latents),
`_t_e_cache/` (text embeddings), `_clip_vision_cache/`, in any subfolder.
Exporting with `caches=0` leaves them out. A cache is only reused by the
trainer when its settings match (they are keyed by file and settings), so a
stale cache costs nothing but space.

## `aitk_bundle.json`

```json
{
  "format": 1,
  "name": "lora",
  "created_at": "2026-10-22T03:00:00Z",
  "server_version": "df22a0601626",
  "upstream_version": "0.7.2",
  "type": "train",
  "step": 400,
  "total_steps": 5000,
  "config": { "job": "extension", "config": { "name": "lora", "process": [ ... ] }, "meta": { ... } },
  "policy": { "auto_resume": true, "max_auto_resumes": null },
  "datasets": [ { "name": "bali", "dir": "data", "images": 20, "captions": 20 } ],
  "job_dir": "output/lora",
  "checkpoints": [ "lora_000000200.safetensors", "lora_000000400.safetensors" ],
  "optimizer": true,
  "samples": 6,
  "caches": true,
  "base_model": "black-forest-labs/FLUX.1-dev",
  "arch": "flux",
  "warnings": []
}
```

| Field | Req. | Meaning |
|---|---|---|
| `format` | yes | `1`. Readers refuse a higher format. |
| `name` | yes | the job name = `config.config.name` = the job folder's name |
| `config` | yes | the job config (JSON, the server's job_config) with machine paths made portable (below); must be an object with `config.process[0]` |
| `job_dir` | no | `output/<name>` (the default); a single folder under `output/` |
| `datasets` | no | `[{name, dir, images?, captions?}]`: each dataset's name on the exporting server and its zip folder (`data`, `data_<i>`). `[]` or missing: no dataset. |
| `step` | no | the step a resume continues at: `training_info.step` of the newest save in the bundle (else its filename step), `0` without saves. Informational (the trainer reads the save itself), and what bootstrapping compares to skip a job that is already further. |
| `total_steps` | no | `config.process[0].train.steps` |
| `type` | no | the job type, `train` by default |
| `policy` | no | `{auto_resume, max_auto_resumes}` (nulls: server defaults). Remotes are per server and are not carried. |
| `checkpoints`, `optimizer`, `samples`, `caches` | no | what the bundle holds: save names oldest first, whether `optimizer.pt` and the dataset caches are in, the sample count |
| `base_model`, `arch` | no | `process[0].model.name_or_path` and `.arch`, for listings and model prefetch |
| `created_at`, `server_version`, `upstream_version` | no | provenance |
| `warnings` | no | what the writer could not include (a dataset outside the datasets folder, a loss log that could not be snapshotted, ...) |

Unknown fields are ignored, so later writers can add optional ones without a
new format number.

### Paths in `config`

On export every string in the config that is a path on the exporting server
is rewritten (the same mapping as portable jobs, `internal/portable`):

| Local path | In the bundle |
|---|---|
| `<datasets folder>/<dataset>/<rest>` | `aitk://datasets/<dataset>/<rest>` |
| `<training folder>/<rest>` | `aitk://training/<rest>` |
| `<toolkit root>/<rest>` | `aitk://root/<rest>` |

Datasets referenced from `process[0].datasets[*].folder_path` are bundled in
that order (`data`, `data_1`, ...), then any other dataset the config points
into (e.g. a `control_path`). A `folder_path` outside the datasets folder is
not bundled and is listed in `warnings`.

On import:
1. Each bundled dataset gets a local name (the caller's `dataset` for the
   first one, else its manifest name; without one the job name, `<job>_<i>`
   for further ones), and every path under `aitk://datasets/<manifest name>` maps
   to that folder.
2. Remaining `aitk://datasets|training|root` paths map to this server's
   folders.
3. Every process's `training_folder` becomes this server's training folder.
4. `config.name` (and a `meta.name` that matched it, beside or inside
   `config`) becomes the job name; saves are renamed to match.
5. `.job_config.json` is written into the job folder the way the worker
   writes it at launch.

## Legacy zips (no manifest)

Zips made by hand from a training machine (`data/` + `output/<name>/`) are
read the same way, with the manifest derived:
- the job folder is the one `output/<x>/` that holds `config.yaml`
  (`config.yml` or `config.json`); several are refused;
- the config is that file (YAML converted to JSON, key order kept); `name` is
  its `config.name`, else `<x>`; saves are matched on that name;
- `data/` is `process[0].datasets[0]`, `data_<i>/` is `datasets[i]`; the
  config's own `folder_path` for that index (e.g. `/workspace/data`) is what
  gets mapped to the restored folder; dataset names default to the job name;
- `step` is read from the newest save's metadata; `checkpoints`, `optimizer`,
  `samples`, `caches` from what is there;
- root files such as a trainer's top-level `config.yaml`, and
  `output/generations/`, are skipped.

## Resumable vs dataset only

- **Resumable**: `output/<name>/` holds at least one save of `<name>`. A
  started job continues at that save's step; with `optimizer.pt` it
  continues exactly (same optimizer moments and schedule), without it from
  the weights with a fresh optimizer.
- **Dataset only** (a "data preset"): no saves (`checkpoints=none`, or a zip
  of `data/` plus `output/<name>/config.yaml` only). Importing creates the dataset and a job
  with the config that starts from step 0.

The manifest says which one: `checkpoints` is empty for a dataset-only
bundle.
