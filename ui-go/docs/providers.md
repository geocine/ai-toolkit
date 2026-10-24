# GPU servers on RunPod, Vast.ai and QuickPod

A GPU server is the aitk image (`ghcr.io/geocine/aitoolkit-go`) running on a
rented GPU. This page covers renting one from the studio, setting one up by
hand, and what each provider does with your money and your data.

## Two ways to get a GPU server

1. **Create it from the studio.** Servers → *Create a GPU server*. The server
   the studio is connected to (usually the [studio API](studio-api.md)) rents
   the GPU with your provider API key, waits until the aitk server on it
   answers, and adds it to your servers. It also stops and terminates it for you.
2. **Set up a template yourself.** Servers → *Set up a template yourself*
   shows the steps below with your token filled in. Start the image at the
   provider by hand, then add its URL and token under Servers → *Add server*.
   This needs no server at all. Nothing filters the hosts for you here, so
   pick one the image runs on: a driver for CUDA 13.0 or newer (580+) and an
   RTX 30-series, A100 or newer GPU (docs/deploy.md, "The image"). Each guide
   below says how in the step where you pick the host.

## Where calls run and why

The studio never calls a provider. Every provider call runs on the server that
holds your provider keys:

| Provider | From a browser | Server URL |
|---|---|---|
| RunPod | API v2 answers the CORS preflight with 401 and no CORS headers. API v1 allows any origin but retires on 2026-11-15. | `https://<pod id>-8675.proxy.runpod.net` |
| Vast.ai | The preflight gets 204 with no `Access-Control-Allow-Origin`. | `http://<ip>:<port>` |
| QuickPod | Echoes the origin, but templates can only be made with a console login. | `https://<pod id>-8675.proxy.quickpod.io/` (not verified), or `http://<ip>:<port>` |

One code path for all three also keeps provider keys out of the browser: a key
is sent once to that server, stored there and never returned (only its
last four characters, `key_hint`).

**Mixed content.** A studio served over https cannot call an `http://` server.
When the server's URL is https, or the studio page itself is http, the studio
talks to the server directly. Otherwise (Vast.ai, QuickPod without the proxy)
it goes through that server's relay, `/v1/machines/{id}/relay/...`:

- every request makes two hops, browser → that server → GPU server;
- the relaying server's `AITK_CORS_ORIGINS` applies, and the GPU server
  needs none;
- uploads stream, events stream, and image thumbnails and signed download
  links pass through without a token (the server checks them itself);
- the relay → GPU server leg is still plain http.

## The server that holds your keys

The server you use creates and tracks the GPU servers. It keeps the provider
keys, the list of machines and their tokens in its database, and polls the
providers while anything runs. Keep it long-lived: if it is lost, its
machines **keep billing**. Every machine is labelled `aitk-…` at the provider,
so you can find and terminate it in the provider console.

Any aitk server can hold the keys. The usual one is the studio API: the Go
server with no GPU, Python or upstream checkout, run on your laptop, a VPS or
in Docker ([docs/studio-api.md](studio-api.md) has the build steps, the flags
and a small image):

```bash
./aitk-server --studio-api     # then open http://127.0.0.1:8675/studio/
```

It listens on `127.0.0.1` and keeps its databases in your user config
folder.

Or, with Docker and no build (a large CUDA image; no GPU needed):

```bash
docker run -d --name aitk-server -p 127.0.0.1:8675:8675 \
  -e AI_TOOLKIT_AUTH=<token> -e AITK_CORS_ORIGINS=<studio origin> \
  -e AI_TOOLKIT_GO_DB=/app/ai-toolkit/data/aitk_go.db \
  -v aitk-server:/app/ai-toolkit/data \
  ghcr.io/geocine/aitoolkit-go:latest
```

Then open `http://127.0.0.1:8675/studio/`. The volume keeps the keys and the
machine list across restarts. `-p 127.0.0.1:…` publishes it on loopback only.

## RunPod

- **Key.** Create one at <https://console.runpod.io/user/settings> with
  `pods:read`, `pods:write`, `catalog:read` and `templates:read`. Paste it
  in the studio, or set `AITK_PROVISION_RUNPOD_KEY` on the server that
  holds your keys.
- **Price.** The GPU price on the chosen cloud (Secure or Community) × GPU
  count, plus the container disk at $0.10 per GB per month (÷ 730 hours).
  RunPod picks the data center.
- **Stop vs terminate.** Stop releases the GPU and **wipes the container
  disk**; a stopped pod made here bills nothing, and starting it again can
  fail when the host's GPUs are taken. Terminate deletes the pod. Keep results
  on a remote.
- **URL.** `https://<pod id>-8675.proxy.runpod.net`. The proxy cuts a request
  that gets no response for 100 seconds.

Manual template:

<!-- guide:runpod -->
| Field | Value | Note |
|---|---|---|
| Name | `aitk headless` |  |
| Visibility | `Private` | The template holds your token. |
| Container Image | `ghcr.io/geocine/aitoolkit-go:latest` | A public image, so Container Registry Credentials stay empty. |
| Container Disk | `100` | GB for models, datasets and checkpoints. It is wiped when the pod stops, so keep results on a remote and terminate the pod when you are done. |
| Expose HTTP Ports | `8675` |  |

Environment variables:

```
AI_TOOLKIT_AUTH=<a long random token>
AITK_CORS_ORIGINS=<your studio origin>
```

1. Sign in at console.runpod.io, open Templates and click New Template.
2. Set Template Type to Pod and Compute type to NVIDIA GPU, then fill in the fields above. Leave Container Start Command empty: the image starts the server itself.
3. Under Environment Variables, add the variables above.
4. For gated base models, also add HF_TOKEN = {{ RUNPOD_SECRET_hf_token }} after saving your Hugging Face token as the secret hf_token under Settings → Secrets.
5. Click Save Template.
6. Open Pods, click Deploy, set the CUDA version filter to 13.0 or newer and pick an RTX 30-series, A100 or newer GPU: the image needs a driver for CUDA 13.0 or newer (580+). On an older host the image fails, and the rental still bills. Click Change Template, choose aitk headless and click Deploy On-Demand.
7. When the pod shows running, open Connect → HTTP Service: the 8675 row links to https://\<pod id>-8675.proxy.runpod.net.
8. When it shows running, copy its URL and add it on the Servers page with the same token.

URL: `https://<pod id>-8675.proxy.runpod.net`
<!-- /guide:runpod -->

## Vast.ai

- **Key.** Create one at <https://cloud.vast.ai/manage-keys/> with
  `user_read`, `instance_read` and `instance_write`, or set
  `AITK_PROVISION_VAST_KEY`.
- **Price.** `dph_total`: the GPU plus storage for your disk size. Offers are
  verified, rentable hosts with at least one direct port. A stopped instance
  keeps its disk and bills the storage (the offer's stopped price).
- **Stop vs destroy.** Stop keeps the disk, which is still billed. Destroy
  deletes everything.
- **URL.** `http://<ip>:<port>`, from the instance's mapped port. The port
  changes when the instance starts again.
- **Limits.** Plain http only: the token crosses the internet unencrypted.
  Vast also limits how often its API can be called, so the server spaces
  its calls, and offer searches have a daily quota.
- **Account env.** Vast puts Account → Environment Variables into every
  instance, including the ones the studio creates with their own
  `AI_TOOLKIT_AUTH`. Which of the two wins has not been checked, so if you
  also create servers from the studio, keep a hand-made server's token in a
  private template (`-e AI_TOOLKIT_AUTH=…` in Docker Options) instead.

Manual template:

<!-- guide:vast -->
| Field | Value | Note |
|---|---|---|
| Template Name | `aitk headless` |  |
| Image Path:Tag | `ghcr.io/geocine/aitoolkit-go:latest` | The full registry path. Another tag works too. |
| Select Launch Mode | `docker ENTRYPOINT` | Runs the image as built, which starts the server. |
| Extra Filters | `direct_port_count >= 1 cuda_max_good >= 13.0 compute_cap >= 800` | Only hosts that can give port 8675 a public port, with a driver for CUDA 13.0 or newer (580+) and an RTX 30-series, A100 or newer GPU. |
| Disk Space | `100` | GB, fixed for the life of the instance. |
| Template Visibility | `Private` |  |

Docker options:

```
-p 8675:8675 -e OPEN_BUTTON_PORT=8675 -e AITK_CORS_ORIGINS=<your studio origin>
```

1. Open Account → Environment Variables (cloud.vast.ai/account/) and add AI_TOOLKIT_AUTH = \<a long random token>. Vast puts it into every instance, so it stays out of the template. Or, with Template Visibility Private, add -e AI_TOOLKIT_AUTH=\<a long random token> to Docker Options instead. An account-wide AI_TOOLKIT_AUTH also reaches the servers the studio creates for you, which get their own token, so use the template if you also create servers from the studio. Add HF_TOKEN under Account → Environment Variables for gated base models.
2. Open Templates (cloud.vast.ai/templates/), click + New and stay on the Config tab.
3. Fill in the fields above and paste the docker options into Docker Options.
4. Click Create & Use, pick an offer and click RENT. The Extra Filters above keep the offers to hosts with a driver for CUDA 13.0 or newer (580+) and an RTX 30-series, A100 or newer GPU. On an older host the image fails, and the rental still bills.
5. When the instance shows running, open IP Port Info: the line ending in -> 8675/tcp gives the URL http://\<ip>:\<port>. It is plain http only, so the token crosses the internet unencrypted, and a studio served over https cannot call it directly: use the studio the server serves at /studio/.
6. When it shows running, copy its URL and add it on the Servers page with the same token.

URL: `http://<ip>:<port>`
<!-- /guide:vast -->

## QuickPod (beta)

- **Key.** Settings → Secure API keys at <https://console.quickpod.io/settings>
  with `pods:read`, `pods:write`, `templates:read`, `account:read` and
  `profile:read`. The `qpk_…` value is shown once. Or set
  `AITK_PROVISION_QUICKPOD_KEY`.
- **Template first.** QuickPod templates cannot be made with an API key, so
  create one by hand (below) and pick it in the studio when you create a
  server. Each listed template shows whether it runs the aitk image, publishes 8675
  and sets a token.
- **Price.** The hourly price plus the host's storage price for your disk
  size. A stopped pod bills the storage.
- **Stop vs destroy.** Stop pauses compute and keeps the storage, which is
  still billed. Destroy removes the pod and its disk.
- **URL.** `https://<pod id>-8675.proxy.quickpod.io/` (not verified for every
  image), else `http://<ip>:<port>`.
- **Limits.** Beta: switch it off with `AITK_PROVISION_PROVIDERS=runpod,vast`.
  Whether per-pod docker options add to the template's or replace them is not
  verified, which is why the template carries `-p 8675:8675` and a token of
  its own.

Manual template:

<!-- guide:quickpod -->
| Field | Value | Note |
|---|---|---|
| Template Type | `GPU` |  |
| Runtime | `Pod` | Not VM. |
| Template Name | `aitk headless` |  |
| Docker Image Path | `ghcr.io/geocine/aitoolkit-go:latest` | The tag goes in the path: there is no separate tag field. |
| Disk Space | `100` | GB. Only the default: you pick the size again at launch. |
| Launch Mode | `Docker Entrypoint` | Uses the image's entrypoint, which starts the server. |

Docker options:

```
-p 8675:8675 -e AI_TOOLKIT_AUTH=<a long random token> -e AITK_CORS_ORIGINS=<your studio origin>
```

1. Sign in at console.quickpod.io and add credit under Settings. QuickPod support is in beta. For the studio to create pods for you later, a key under Settings → Secure API keys needs Pods read, Pods write, Templates read, Account read and Profile read.
2. Open Templates and click Create New Template.
3. Fill in the fields above. Leave Private Docker Repository unticked, and leave Docker Entrypoint Run Options and On Start Script empty.
4. Paste the docker options into Docker Options. Keep both parts: -p 8675:8675 publishes the port, and AI_TOOLKIT_AUTH keeps the server from starting without a token.
5. Click Create Template. On GPU Search, pick an offer with a max CUDA of 13.0 or newer on an RTX 30-series, A100 or newer GPU: the image needs a driver for CUDA 13.0 or newer (580+). On an older host the image fails, and the rental still bills. Choose aitk headless under My Templates and click Create.
6. On Pods, wait for running and click Connect. Under Exposed Ports, the 8675 row has an HTTPS link (https://\<pod id>-8675.proxy.quickpod.io/) and an HTTP link (http://\<ip>:\<port>).
7. When it shows running, copy its URL and add it on the Servers page with the same token.

URL: `https://<pod id>-8675.proxy.quickpod.io/`
<!-- /guide:quickpod -->

## Which hosts are listed

*Create a GPU server* lists only hosts that can run the image you launch:
an NVIDIA amd64 GPU from the RTX 30-series and A100 on, a driver for CUDA
13.0 or newer, and enough disk (docs/deploy.md, "The image"). Each provider
filters as far as its API allows, and the aitk server does the rest, so every
offer carries the same `fit` whichever provider it comes from:

| Provider | The provider filters | The aitk server filters |
|---|---|---|
| RunPod | The pod is created with `gpu.minCudaVersion` `13.0`, so RunPod places it only on a host whose driver runs the image. | GPU types the image cannot run are left out of the catalog. A GPU with no free CUDA 13.0 host right now is a warning. RunPod reports no download or disk speed. |
| Vast.ai | The search sends `gpu_arch`, `cpu_arch`, `cuda_max_good >= 13.0`, `compute_cap` (the GPU architectures the image runs), `disk_space` (the disk need), plus `gpu_ram` when a model is chosen and `inet_down` and `disk_bw` for fast hosts. Never `driver_vers`: it is empty or stale on many hosts. | The GPU name, and the same checks again on each row. |
| QuickPod | Nothing: `/rentable` takes no filters. | Everything: amd64, NVIDIA, a GPU the image runs, `max_cuda` (or the driver version) at least 13.0, and `max_disk_size` at least the disk need. A host whose CUDA version is unknown stays, with a warning. |

The filters above the list:
- **What will you train?** (`model`) sets the VRAM floor (the smallest size
  the model fits, such as 24 GB for FLUX.1 with 8-bit), a VRAM badge on each
  offer, and the disk need.
- **Fast hosts only** (`fast`) keeps hosts with at least 500 Mbps download
  and 300 MB/s disk. A host that reports no speed passes. At 500 Mbps the
  image takes about 5 minutes to download and unpack, at 73 Mbps about 25.
- **Show all hosts** (`show_all`) also lists the hosts the image cannot run,
  marked *Won't run here* and not selectable, and the sm_75 GPUs (T4, RTX
  20-series), which run without flash attention, with a warning.

Each offer shows its fit as a dot and one line ("Driver too old (CUDA
12.8)", "≈ 3 min to pull"), and the disk defaults to the disk need. The Image
row under Configure says which build the server will run: the creating
server's own build when it is published, else the newest, pinned to its `:<sha>` tag.
On QuickPod it checks the template's image instead. A launch repeats every
check and is refused (`image_blocked`) before any provider call when the
host cannot run the image.

The suggested GPU (*Best value*, *Best speed*) comes from the smallest VRAM
tier the model fits (24, 32, 48, 80 GB and up), among one-GPU offers with a
CUDA 13.0 driver, a supported GPU, enough disk, at least 500 Mbps and
300 MB/s where reported, and reliability of 95 % or more:
[docs/headless.md, "GPU recommendation"](headless.md#gpu-recommendation).

## Money guards

- Every launch names a ceiling, `max_price_per_hour` (the studio sends the
  quoted price plus 2 %). The server quotes the offer again and refuses
  with 409 when the price is now above it.
- A launch carries a `request_id`: sending it again returns the same machine
  instead of renting a second GPU.
- `max_running` caps live machines per provider: 2 by default, 1 to 20, set
  per provider in the studio or with `AITK_PROVISION_MAX_RUNNING`.
- A server without `AI_TOOLKIT_AUTH` refuses to rent, start or link for
  anyone but a caller on the same machine.
- The MCP launch op tells agents to quote the price and get your OK first.
- The server that holds your keys never terminates anything on its own. A
  machine that is not ready after 20 minutes (`AITK_PROVISION_READY_TIMEOUT`)
  is marked failed and is still billed until you terminate it.

## When done

*When done* (`when_done`: `keep`, `stop` or `destroy`) is the server's own
lifecycle (docs/headless.md, "Unattended runs"): once the last job finishes
and its results are verified on a remote, the machine stops or deletes
itself. It needs *Keep results on* (a remote), because the lifecycle never
acts on results that exist only on the machine.

- With a remote chosen, it defaults to `destroy` (on QuickPod only with
  `self_destroy_key`), otherwise `keep`.
- **RunPod.** A pod can only stop itself with its own key, and a stopped pod
  loses its container disk. `self_destroy_key` gives it your key
  (`AITK_RUNPOD_API_KEY`) so it can delete itself.
- **Vast.ai.** Every instance has its own key; nothing extra is needed.
- **QuickPod.** `stop` and `destroy` need `self_destroy_key`
  (`AITK_QUICKPOD_API_KEY`); without it the launch is refused.

`self_destroy_key` puts your **account key** in the machine's environment. It
is then visible in the provider console and API, and to anything running in
the container. It is never preselected.

## Security

- Each machine gets a fresh token (`AI_TOOLKIT_AUTH`, `aitk_` plus 32 random
  bytes). That server keeps it and hands it to the studio only when the
  machine is linked; events and logs never carry it. It checks
  that the GPU server refuses requests without it, and flags a machine that
  answers without a token instead of adding it.
- Provider keys are write-only on the server that holds them.
- *Keep results on* copies that remote's credentials into the machine's
  environment as plain values. Anyone with the provider key or console access
  can read them. It is never preselected.
- Vast.ai, and QuickPod without the proxy, are plain http: the token travels
  unencrypted between you (or the relay) and the machine.

## Simulate

Try the whole flow with no credits. In `ui-go/` of a dev checkout:

```bash
AITK_PROVISION_SIMULATE=1 AI_TOOLKIT_AUTH=dev go run .
cd studio && npm run dev     # in a second terminal
```

Add the server `/aitk` with token `dev`. Each provider then answers from an
in-process mock, and every "GPU server" is this same server under another
address and token. Any key works; these keys make the mock fail on purpose:

| Key | What happens |
|---|---|
| `sim_bad` | The key is rejected. |
| `sim_broke` | The account has no credit: the key check shows $0.00 and launches fail. |
| `sim_full` | No capacity for the offer. |
| `sim_429` | Every third call is rate limited. |
| `sim_crash` | The machine crashes after it starts. |
| `sim_noport` | No public port for 8675. |
| `sim_open` | The server answers without a token. |
| `sim_slow` | Everything takes ten times longer. |
| `sim_replace` | QuickPod replaces the template's docker options (no port). |
| `sim_ignore` | QuickPod ignores the per-pod options (the template's token). |
| `sim_gone` | A deleted pod reads as terminated instead of missing. |
| `sim_nokernel` | The GPU has no kernel image for this build. |
| `sim_olddriver` | The host driver is too old for the image. |
| `sim_stale` | The server reports an older build than the image asked for. |

`AITK_PROVISION_SIMULATE_DELAY` (`15s`) sets how long a machine takes to come
up. Never enable simulate on a shared or public server.

## Not verified against the live API

<!-- unconfirmed -->
- RunPod: Whether a pod's own RUNPOD_API_KEY works on API v2.
- RunPod: The exact text RunPod returns when a GPU is out of stock.
- RunPod: Whether GET returns 404 or TERMINATED after a pod is deleted.
- Vast.ai: Whether actual_status reports scheduling or stopped.
- Vast.ai: Exact permission names for scoped keys.
- Vast.ai: Whether an account environment variable or the instance's own env wins for AI_TOOLKIT_AUTH.
- QuickPod: Whether per-pod docker options add to the template's or replace them.
- QuickPod: Whether the https proxy passes the Authorization header.
- QuickPod: The State values besides running and exited.
- QuickPod: The HTTP status for insufficient credit.
<!-- /unconfirmed -->
