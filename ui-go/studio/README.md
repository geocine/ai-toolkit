# Studio

The day-to-day training app for the headless ai-toolkit server: pick a
preset, drop the photos, fix the captions, launch, then watch the samples and
loss and pick a checkpoint. It is a static web app that talks to one or more
servers over the `/v1` API (`ui-go/docs/headless.md`). The operator console
(`ui-go/web`, served by the server itself) is for stepping in; this is for
training.

## What it does

- **Servers**: add as many as you like (base URL + token, kept in this
  browser's localStorage), test them, switch between them. Create a GPU
  server on RunPod, Vast.ai or QuickPod through the active server, or follow
  the step-by-step template guide (`ui-go/docs/providers.md`).
- **Presets**: a built-in *Krea 2 · character LoKr* preset made from
  `krea_lokr.yml` / `krea_lokr_male.yml` and your Krea notes, with female /
  male / man prompt sets and VLM caption prompt templates. Import any
  ai-toolkit YAML or JSON as a preset; export any preset back to YAML.
  Imports are checked for the two traps found in the Krea files (see below).
- **New run**: preset → name and trigger → drag in images (and `.txt`
  captions) → captions (inline editing, fill missing, copy the caption
  prompt) → knobs → review → create and start, optionally portable on a
  remote with auto-resume.
- **Runs**: live progress over the event stream, ETA, samples in grids by step
  per step, an SVG loss chart with checkpoint marks, checkpoints to download
  or push to a remote, sync / portability status, the trainer log.

## Fixes in the built-in preset

1. In `krea_lokr.yml`, `loss_type`, `do_differential_guidance` and
   `differential_guidance_scale` sat at the process level. The trainer reads
   them only from `train` (`toolkit/config_modules.py`, `TrainConfig`), so
   differential guidance was silently off. They are in `train` here.
2. In both files EMA was effectively off: the trainer reads EMA only from
   `train.ema_config: {use_ema, ema_decay}` (`EMAConfig`), but the files set
   `ema_config: null` plus loose `use_ema` / `ema_decay` (`ma_decay`, a typo,
   in the female file). The preset uses `ema_config: {use_ema: true,
   ema_decay: 0.99}`.

## Run, build, host

```bash
cd ui-go/studio
npm ci
npm run dev        # http://localhost:5174
npm run build      # static files in dist/ — host anywhere (any static host, a bucket, file server)
```

The server itself serves a build at `/studio/` when `ui-go/studio/dist` exists
(the Docker image builds it), so one URL works from a phone:
`https://<pod>-8675.proxy.runpod.net/studio/`, with that same URL as the
server's base URL (prefilled) and no CORS setup. The build carries a web app
manifest and an app-shell service worker (`public/sw.js`, which never caches
the API), so "Add to Home Screen" installs it.

In development, add a server with the base URL `/aitk`: the dev server
proxies it to `AITK_API` (default `http://127.0.0.1:8675`), so no CORS setup is
needed.

From a hosted build the studio calls the server cross-origin with
`Authorization: Bearer <token>`. Allow the studio's origin on the server:

```bash
docker run ... -e AITK_CORS_ORIGINS=https://studio.example.com ghcr.io/geocine/aitoolkit-go
# several: -e AITK_CORS_ORIGINS=https://studio.example.com,http://localhost:5175
```

Images are loaded from the server's public media routes (`/api/img/...`, no
token needed, like upstream's UI).

## Try GPU servers without credits

Run the server with simulated providers, then the dev studio:

```bash
cd ui-go
AITK_PROVISION_SIMULATE=1 AI_TOOLKIT_AUTH=dev go run .
cd studio && npm run dev     # in a second terminal
```

Add the server `/aitk` with token `dev`, then *Create a GPU server* with any
key. Nothing calls RunPod, Vast.ai or QuickPod; the keys that fail on purpose
(`sim_full`, `sim_broke`, ...) are listed in `ui-go/docs/providers.md`.
