# Design: the studio API as the fleet gateway

Status: proposal, not built. Nothing here describes current behaviour unless
it says "today".

## Roles

| Role | What it is | Trusted with |
|---|---|---|
| Orchestrator | The studio API on your own server (`ghcr.io/geocine/aitoolkit-studio`), persisted on a volume. | Everything: provider keys, configs, datasets, every machine credential. |
| Thin clients | The studio in a browser, and agents over MCP. Both talk only to the orchestrator. | Your orchestrator token. Never a machine token or a provider key. |
| Workers | Rented GPU servers (`ghcr.io/geocine/aitoolkit-worker`). Cattle: created, used and terminated by the orchestrator. | One job's config and dataset while it trains. |

Today the orchestrator already creates, probes, relays to, stops and
terminates workers, but it reaches them through their public URL: the
RunPod or QuickPod HTTPS proxy, or plain `http://ip:port` on Vast and
QuickPod direct. Each worker also runs its own `/mcp`, jobs and datasets, and
the orchestrator answers those routes with 409.

## Threat model

| Party | Can do today | After this design |
|---|---|---|
| Anyone on the path to a Vast or QuickPod-direct worker | Read and replay the machine token, read every request and response. | Nothing: the worker has no listening port; its one connection is TLS to your server. |
| The provider's HTTPS proxy (RunPod, QuickPod) | Terminates TLS, so it sees plaintext traffic. | Not on the path. |
| A web page you visit | Blocked by Host and Origin checks on an open loopback studio API. | Same checks, and the shell WebSocket checks `Origin` explicitly (CORS does not cover WebSockets). |
| A leaked machine token | Full control of that worker from anywhere. | Useless from anywhere but the tunnel: the worker answers only the orchestrator. |
| The GPU host's owner | Reads the dataset, config and outputs, on disk and in memory. | **Unchanged.** No transport fixes this. Mitigate by choosing hosts: RunPod secure cloud (data centres) by default, community clouds and Vast only by choice, and terminating (not stopping) when done. |
| Whoever holds your server or its bucket credentials | Everything. | Everything. Protect them like the provider keys they guard. |

## 1. Workers dial home (the tunnel)

The worker opens one long-lived outbound connection to the orchestrator and
everything travels through it. The orchestrator never dials the worker.

- **Transport.** `wss://<your domain>/v1/fleet/tunnel`, TLS terminated by
  your reverse proxy (Caddy with a Let's Encrypt certificate). The worker
  verifies the certificate with the system roots, and can pin it
  (`AITK_ORCHESTRATOR_PIN`, the SPKI sha256) for a self-signed setup.
- **Identity.** The orchestrator already makes a 256-bit token per machine
  (`provision.NewToken`). At create time it passes `AITK_STUDIO_API_URL` and
  that token in the worker's env. The provider can read the env, but the
  provider is the host and can read everything anyway.
- **Handshake (built).** The token never crosses the wire, and each side
  proves it to the other.
  - The orchestrator sends a fresh random challenge. The worker answers with
    `HMAC-SHA256(token, "aitk-tunnel-v1|worker|" + id + "|" + challenge +
    "|" + nonce)` and its own fresh nonce.
  - The orchestrator checks that in constant time, then answers the nonce
    with the `studio` role. The worker refuses an orchestrator that can't.
  - Because both challenges are fresh, an answer can't be replayed, so no
    clock limit or replay cache is needed.
  - This takes celld's idea of an HMAC handshake on tunnel setup, with a key
    per machine instead of one per fleet, inside TLS (which celld leaves to
    the network).
- **Streams.** The WebSocket carries [yamux](https://github.com/hashicorp/yamux)
  (Hashicorp's multiplexer, used by Consul and Nomad): many independent
  streams with their own flow control on one connection. Each stream is a
  plain HTTP/1.1 request served by the worker's existing mux, so every `/v1`
  route, SSE stream and download works unchanged. A shell is one stream of raw
  bytes.
- **One mode variable.** `AITK_MODE` decides what a server is. Nothing else
  turns UIs, MCP or the public port on or off.

  | `AITK_MODE` | Runs | Agents talk to | Listens on |
  |---|---|---|---|
  | `worker` (default in the GPU image) | Trainer only. No console, no studio, no `/mcp` of its own. It dials `AITK_STUDIO_API_URL` and is managed from there. | The remote studio's `/mcp` | `127.0.0.1` only |
  | `standalone` | An ordinary AI Toolkit server: console at `/`, `/v1`, `/mcp`. No studio. | This server's `/mcp` | `0.0.0.0` (set `AI_TOOLKIT_AUTH`: without it the server is open, as upstream) |
  | `studio` | Studio and trainer on one node. The console does not load. It can also manage remote workers. | The studio's `/mcp` | `0.0.0.0`, with GitHub sign-in or a token |
  | `studio-api` (default in the slim image) | Studio without a trainer: the orchestrator at `lora.aivan.io` | The studio's `/mcp` | As `studio` |

  - Every worker the orchestrator creates is `worker`. Its probe, relay,
    shell and MCP forwarding all go through the tunnel.
  - `studio-api` is only for the slim image, which sets it itself. The GPU
    image's binary refuses it. `AITK_STUDIO_API=1` (the older flag) stays as an
    alias in the slim image and source builds.
  - A `worker` without `AITK_STUDIO_API_URL` refuses to start, saying: set
    it, or use `AITK_MODE=standalone`.
  - This is a breaking change for hand-started GPU servers (a RunPod
    template, a plain `docker run`): they now need `AITK_MODE=standalone`
    to keep today's behaviour. `docs/deploy.md` and `docs/providers.md` say
    so.
- **Reconnects.** The worker reconnects with jittered backoff (1 s up to
  30 s). Pings every 20 s keep provider NAT and proxy idle timers open. The
  orchestrator marks a machine `unreachable` after 5 minutes without a tunnel,
  as it does today without a probe answer.
- **Where workers dial.** The orchestrator passes its own `AITK_PUBLIC_URL`
  as `AITK_STUDIO_API_URL`. A launch on a real provider is refused before
  anything is rented when that URL is missing or loopback, because the worker
  could never connect. Simulated launches are exempt.
- **Testing without a public server.**
  - **Local worker.** A worker container on the same machine dials
    `http://host.docker.internal:8675`. This covers the tunnel, the shell and
    MCP forwarding with no rented GPU; only training needs one.
  - **Rented GPU, local orchestrator.** Expose the local studio API with a
    throwaway tunnel (`cloudflared tunnel --url http://127.0.0.1:8675`, which
    gives an `https://….trycloudflare.com` URL with no account or DNS). Set
    that URL as `AITK_PUBLIC_URL`.
  - **Production.** `https://lora.aivan.io`.
- **Old workers.** Machines from images without tunnel support keep today's
  path: the probe and relay dial their public URL. The machine record shows
  which path it uses.

### Why not protobuf or gRPC

Protobuf is an encoding, not a security layer: the security comes from TLS
and the handshake above. It also would not make the link leaner where it
matters: control messages are tiny, and the heavy traffic (datasets,
checkpoints, shell bytes) is raw bytes in any encoding. Moving the `/v1` API
to gRPC would rewrite every route and client for no gain. The tunnel keeps
HTTP+JSON for control and raw bytes for bulk and shell traffic.

## 2. HTTPS and GitHub sign-in on the orchestrator

The orchestrator is served at `https://lora.aivan.io`, reachable from the
internet on 443. Browsers, agents and workers all reach it over TLS. A
compose file in `ui-go/docker/` runs the studio API image with Caddy:

- Caddy gets and renews the certificate for the domain and proxies to the
  studio API on the compose network.
- `AI_TOOLKIT_AUTH` is required (the studio API already refuses to start on a
  reachable address without it), and `AITK_PUBLIC_URL` is the domain.
- The data volume and Caddy's certificate volume persist.

### GitHub sign-in for the studio

- **Setup.** A GitHub OAuth App provides `AITK_GITHUB_CLIENT_ID` and
  `AITK_GITHUB_CLIENT_SECRET`. `AITK_GITHUB_USERS=geocine` lists who may sign
  in.
  - On the first sign-in the orchestrator stores the account's numeric GitHub
    id, and matches on that id from then on. Renaming the account, or someone
    later taking the name `geocine`, does not change who gets in.
- **Flow.** The authorization code flow with `state` and PKCE.
  - The orchestrator asks GitHub for no scopes: it only needs `GET /user`
    to learn who signed in.
  - It keeps no GitHub token after sign-in.
- **Session.** A server-side session in `aitk_go.db`, sent as a
  `__Host-aitk_session` cookie (`HttpOnly`, `Secure`, `SameSite=Lax`). It
  lasts 30 days and slides with use.
  - Every write and every WebSocket handshake also checks `Origin`.
  - Sign-out deletes the session.
  - A sessions list lets you revoke other devices.
- **Agents and workers** keep using bearer credentials: `AI_TOOLKIT_AUTH`, API
  keys and the tunnel handshake. GitHub sign-in is for browsers only.
- **Local testing.** A second OAuth App whose callback is
  `http://127.0.0.1:8675/v1/auth/github/callback`.
  - On `http://127.0.0.1` the cookie drops `__Host-` and `Secure`, which
    browsers require over plain http.
  - That is the only difference: the allowlist and flow are the same.
- With the GitHub variables unset, the studio keeps today's token prompt.

## 3. Shell sessions

- **Worker.** `GET /v1/shell/sessions/{id}/attach` upgrades to a WebSocket
  and attaches to a PTY (`creack/pty`, pure Go, Linux only behind a build
  tag; the Windows build gets a stub answering 501). Sessions live on the
  worker, independent of the connection: a dropped browser reattaches and
  replays a 256 KB scrollback ring, so `nvtop` or a long command keeps
  running. `POST /v1/shell/sessions` creates one, `GET` lists, `DELETE`
  kills.
- **Wire format.** Binary frames are raw PTY bytes in both directions. Text
  frames are control messages (`{"resize":[cols,rows]}`). Output is coalesced
  over 4 ms, with backpressure: the PTY read pauses while the socket is full.
- **Browser auth.** A browser cannot set `Authorization` on a WebSocket. The
  studio first calls `POST /v1/machines/{id}/shell/tickets`, which returns a
  single-use ticket valid for 30 s. The ticket goes in the
  `Sec-WebSocket-Protocol` header (not the URL, so it stays out of logs), and
  the handshake checks `Origin` against the orchestrator's own origin and
  `AITK_CORS_ORIGINS`.
- **Path.** Browser → orchestrator → tunnel stream → worker PTY. The
  machine credential never reaches the browser. The latency floor is the two
  network legs; there is no provider proxy in between.
- **Terminal.** xterm.js with the WebGL renderer, fit and image addons, in a
  shell panel on the machine page. TUIs (nvtop, btop, htop, vim, tmux)
  render in full colour.
- **Inline images.** `aitk-server imgcat <file>` on the worker prints an image
  with the iTerm inline-image protocol, which the image addon displays: sample
  images and loss plots in the terminal, with nothing added to the GPU image.
- **Refused** in studio API mode on its own box (409), and over a plain-http
  path to an old worker unless you turn that on per machine.
- **MCP.** `aitk.machines.exec({ id, command, timeout_s })` runs a command
  non-interactively and returns its exit code and output (truncated at
  1 MB). `aitk.machines.shell.list` and `.kill` manage the sessions.
  Interactive PTYs stay in the studio.

## 4. One gateway for agents and the studio

- Agents connect only to the orchestrator's `/mcp`. Code mode gains
  `aitk.on(machineId)`, which returns the worker's ops (`jobs.*`,
  `datasets.*`, `files.*` …) forwarded through the tunnel. The forwarding
  keeps the worker's error codes. Workers' own `/mcp` is off in tunnel-only
  mode.
- The studio does the same: a server picker chooses a worker, and its calls
  go through the orchestrator's relay instead of to the worker's URL.

## 5. Configs and datasets live on the orchestrator

- **Library.** Job configs and datasets are stored on the orchestrator. The
  job routes that answer 409 today become a library: create, edit,
  duplicate, version.
- **Run on a host.** `POST /v1/runs {config, dataset, machine}` pushes the
  dataset to the worker over the tunnel, starts the job there, streams
  progress and samples back, and pulls the outputs to the orchestrator.
  - The dataset push is chunked and resumable, checked with xxhash, and
    sends only the files the worker does not already have.
  - The outputs pulled back are checkpoints and the final LoRA.
- **Replicate.** The same config and dataset can run on several hosts. Each
  run records its machine, image digest and config version.
- **Storage.** Files live in a storage target, and workers pull from it
  directly, so dataset bytes don't pass through your server's uplink. Targets
  are below.

### Storage targets

Both are S3 APIs, so one `minio-go` client (already a dependency, used by
remotes) serves both. A target is a bucket plus a role: datasets, outputs or
backups.

| | Cloudflare R2 | Hugging Face storage bucket |
|---|---|---|
| Endpoint | `https://<account>.r2.cloudflarestorage.com`, region `auto` | `https://<namespace>.s3.hf.co`, region `us-east-1`, path-style only |
| Credentials | An R2 API token scoped to one bucket | S3 keys (`HFAK…`) generated from a Hugging Face token, with that token's permissions |
| Cost | Free tier: 10 GB-month, 1M writes and 10M reads a month, free egress (check Cloudflare's current page) | $18/TB-month private; free accounts list 100 GB private (whether buckets count is not stated) |
| Versioning | Off | None at all: an overwrite or delete is final |
| Best for | Configs, small datasets, backups | Datasets, LoRAs and checkpoints: workers pull at CDN speed (`GetObject` redirects to the nearest edge), next to the base models they already download from the Hub |

- **Budget guard.** Each target has a quota the orchestrator enforces before
  it writes: `quota_gb`, default 9 for R2, under the 10 GB free tier. It also
  enforces a monthly write cap, default 900k for R2.
  - Usage is measured by a nightly listing and counted locally between
    listings.
  - An upload that would cross a limit is refused with the numbers, and a
    notify event goes out at 80%.
  - Your $10 Cloudflare alert stays the backstop.
- **What workers get.**
  - **R2:** presigned URLs only, short-lived and per object: GET for inputs,
    PUT for outputs. A worker never holds a key.
  - **Hugging Face:** presigned URLs are not documented, so a worker gets two
    fine-grained tokens:
    - a read token for the datasets bucket;
    - a write token for a separate outputs bucket only.

    The host can then only overwrite outputs, never your datasets. Whether a
    token can be limited to one bucket must be checked when this is built.
    The docs imply it but don't show how.
- **Outputs to the Hub.** A finished LoRA can also be copied to a private
  model repo with the existing HF upload path.

## 6. A durable orchestrator

The orchestrator is the one thing that must not lose state: a lost machine
record is a server that may still be billing.

- **Continuous backup.** Ship every SQLite write to your bucket in the LTX
  format, as celld does. Use [Litestream](https://litestream.io) (same format,
  embeddable in Go), and restore from the bucket on boot when the volume is
  empty. One orchestrator node, so no leases or consensus are needed.
- **Restart-safe steps.** Create → ready → push dataset → train → pull outputs
  → terminate is stored as a step per run in the database. A restarted
  orchestrator resumes at the last finished step, the way celld Workflows
  resume their durable steps.
- **Safety net.** Every machine is labelled `aitk-…` at the provider and
  `providers.adopt` re-tracks it (today), so even a total loss can be
  recovered from the provider accounts.

### Why not run the orchestrator on celld

celld would give durable objects, replication to a bucket, workflows and
hibernating WebSockets, but:

- the orchestrator would have to be rewritten from Go into a Workers app;
- it is a beta (v0.6.3) with no Windows support;
- it needs a bucket even for one node, and with one node every write waits
  for a bucket round trip;
- its peer traffic is plaintext and must sit on a private network;
- GPU work is out of scope, and workers must never be celld nodes, because
  the bucket credentials give full control of the fleet.

The parts worth taking are its patterns: an HMAC handshake on tunnel setup, the LTX format for continuous SQLite backup, and steps
that survive a restart.

## Build order

| Step | Change | Tested by |
|---|---|---|
| 1 | Tunnel: handshake, yamux, the worker's tunnel-only listener, and probe and relay over the tunnel with a fallback. | Go tests with an in-process orchestrator and worker. A Linux container worker registered against a local studio API. |
| 2 | GitHub sign-in, the compose file with Caddy, and doc updates (including `AITK_PUBLIC_LISTEN`). | Go tests against a fake GitHub. A local OAuth App on `127.0.0.1`. `docker compose up` on `lora.aivan.io`. |
| 3 | Shell: PTY sessions, tickets, the xterm.js panel, `imgcat`, and the `exec` op. | Go tests (Linux). A browser session through the relay to a container worker. |
| 4 | `aitk.on(id)` MCP forwarding and the studio server picker. | codemode tests, plus the MCP parity rule (op, TS type, OpenAPI, headless.md). |
| 5 | Config and dataset library, R2 and Hugging Face storage targets with quotas, and runs pushed to hosts. | End-to-end on a simulated and then a real RunPod worker. |
| 6 | Litestream backup and restart-safe steps. | Kill the orchestrator mid-run; restore from the bucket. |

Each step ships alone. A worker image with step 1 must be pushed before the
orchestrator image that relies on it, because the orchestrator launches
workers at its own build's `:<sha>`.

## Decisions

- **Domain.** The orchestrator is `https://lora.aivan.io`, reachable from the
  internet on 443. DNS is not set up yet.
- **Sign-in.** GitHub, for `geocine` only, and testable locally.
- **Storage.** R2 (free tier, guarded by quota) and Hugging Face storage
  buckets. Backups (step 6) go to R2.
- **Workers.** Tunnel-only by default. `AITK_PUBLIC_LISTEN=1` opts a
  hand-started server into a public port.
