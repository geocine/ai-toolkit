# Studio API

The studio API is the Go server without the trainer: it holds your provider
keys, lists and recommends GPU offers, and creates, watches, stops, relays
and terminates GPU servers. It does not train; training runs on the GPU
servers it creates. Run it anywhere: your laptop, a VPS or Docker. Keep it
running while servers it launched exist; they keep billing if it is gone.
Every machine it creates is labelled `aitk-…` at the provider, so you can
always find and terminate it in the provider console.

It needs no GPU, no Python and no upstream checkout. It creates its own
databases, serves the studio at `/studio/`, and answers the provider,
machine, image and recommend routes, `/v1/health`, settings, remotes, keys,
notify, events and `/mcp`. Routes that need the trainer answer 409 (see
[Limits](#limits)).

## Build and run from source

You need Go 1.26 and Node 22.

```bash
cd ui-go/studio && npm ci && npm run build
cd .. && go build -o aitk-server .        # Windows: go build -o aitk-server.exe .
./aitk-server --studio-api               # then open http://127.0.0.1:8675/studio/
```

The binary serves the studio build in `studio/dist` next to it. `go run .
--studio-api` from `ui-go/` works too.

| Flag or env | Meaning |
|---|---|
| `AITK_MODE=studio-api`, `--studio-api`, `AITK_STUDIO_API=1` | Studio API mode (the slim image sets `AITK_MODE`). |
| `AITK_PUBLIC_URL` | This studio API's public https URL, such as `https://loraboard.com`. Every GPU server it creates is a worker that dials this URL for its tunnel, so a real launch is refused while it is unset (the browser's origin is used then) or points at this computer or a private address. See [GPU servers dial in](#gpu-servers-dial-in). |
| `--data-dir`, `AITK_STUDIO_API_DIR` | Where the databases live. Default: `%AppData%\aitk-studio-api` on Windows, `~/Library/Application Support/aitk-studio-api` on macOS, `~/.config/aitk-studio-api` on Linux. The folder is made private to you (0700) and the database files 0600; Windows keeps its own permissions. |
| `--port`, `PORT` | Port, default 8675. |
| `--host` | Address to listen on. Default `127.0.0.1` in studio API mode (the plain server listens on `0.0.0.0`). |
| `AI_TOOLKIT_AUTH` | Bearer token. Required when `--host` is not loopback: the studio API refuses to start on a reachable address without it, because it holds your provider keys. |
| `AITK_ALLOWED_HOSTS` | Extra `Host` names an open studio API (no token, no API key) answers, as a comma list of `host` or `host:port`. See [Host names](#host-names). |
| `AITK_CORS_ORIGINS` | Origins of a studio hosted elsewhere (see below). |
| `AITK_PROVISION_RUNPOD_KEY`, `AITK_PROVISION_VAST_KEY`, `AITK_PROVISION_QUICKPOD_KEY` | Provider keys used when none was saved in the studio. |
| `AITK_PROVISION_SIMULATE=1` | Try it with no credits: every provider answers from an in-process mock. The keys that fail on purpose are in [providers.md](providers.md#simulate). |
| `AITK_STUDIO_DIR` | A studio build to serve at `/studio/` instead of `studio/dist`. |

The databases are `aitk_db.db` (upstream's schema, created from a copy
embedded in the binary) and `aitk_go.db` (provider keys, machines, settings).
When a newer binary finds an `aitk_db.db` made by an older one, it keeps its
settings, moves the old file to `aitk_db.db.bak-<unix time>` and creates a new
one. You never need `npx prisma db push` here.

### Host names

An open studio API (no `AI_TOOLKIT_AUTH` and no API key) answers only
requests whose `Host` is `127.0.0.1`, `[::1]` or `localhost`, on any port.
Anything else gets 403. That stops a web page whose name was re-pointed at
127.0.0.1 (DNS rebinding) from driving your studio API. It also answers on:

- the host of `AITK_PUBLIC_URL` (`host`, or `host:port` when the URL has a
  port), so MCP code mode keeps working when you set it;
- every name in `AITK_ALLOWED_HOSTS`. An entry without a port matches any
  port. A studio or client reached through a proxy that passes on its own
  `Host` name needs that name here; a proxy that rewrites `Host` to
  `127.0.0.1` needs nothing. An MCP client behind a proxy that sets
  `X-Forwarded-Host` needs that name too (or in `AITK_PUBLIC_URL`), because
  code mode's calls carry it as their `Host`.

With a token set the check is off, since such a page cannot send the bearer
header.

## GPU servers dial in

Every GPU server the studio API creates runs `AITK_MODE=worker`.

- **The connection.** At boot, the worker opens one outbound connection to
  `AITK_STUDIO_API_URL/v1/fleet/tunnel`, and everything travels inside it:
  the readiness probe, the relay the studio uses and the CUDA check.
  - The worker listens on `127.0.0.1` only, so nothing on its public port
    answers.
  - The studio API never dials the worker's address, so the RunPod and
    QuickPod proxies are not in the path, and Vast's plain http port is not
    used.
- **Proving who is who.** Both sides prove they hold the machine's token
  without sending it: each answers the other's random challenge with an
  HMAC.
  - The studio API accepts a tunnel only from a machine it tracks and that has
    not ended.
  - Inside a stream the studio API still sends the token as a bearer, so the
    worker's own auth applies as usual.
- **Status.** A machine reached this way shows `tunnel: true` and
  `secure: true`. Its `base_url` is `http://<hex id>.tunnel.aitk`, which only
  this server can dial, and the studio uses its relay path. If the tunnel
  stays down for 5 minutes, the machine shows `unreachable`.
- **The URL must be public.** The worker must be able to reach the studio
  API, so `AITK_PUBLIC_URL` must be a public https URL.
  - **Testing from this computer:** `cloudflared tunnel --url
    http://127.0.0.1:8675` prints a throwaway `https://….trycloudflare.com`
    URL; set it as `AITK_PUBLIC_URL`.
  - **A worker container on the same machine** can dial
    `http://host.docker.internal:8675`. The worker allows plain http only to
    localhost, `host.docker.internal` and private addresses. The tunnel route
    answers under any `Host` name, even on an open studio API. To try it
    without renting anything:

    ```bash
    AITK_PROVISION_SIMULATE=1 ./aitk-server --studio-api     # 1. a simulated studio API
    # 2. in the studio, create a GPU server; then take its id and token:
    curl -s -X POST http://127.0.0.1:8675/v1/machines/<id>/link   # {"token": ...}
    # 3. a worker that is that machine (the GPU image, or any Linux box with the binary):
    docker run --rm -e AITK_MODE=worker -e AITK_STUDIO_API_URL=http://host.docker.internal:8675 \
      -e AITK_MACHINE_ID=<id> -e AI_TOOLKIT_AUTH=<token> ghcr.io/geocine/aitoolkit-worker:<sha>
    ```

    The machine then switches to `tunnel: true`, and the relay reaches the
    worker. On Docker Desktop, `host.docker.internal` reaches a studio API
    listening on `127.0.0.1`. On Linux, add
    `--add-host=host.docker.internal:host-gateway`, and start the studio API
    with `--host 0.0.0.0` and `AI_TOOLKIT_AUTH`.

A server started from an older image has no worker mode: it keeps listening
on its public port, and the studio API reaches it there as before.

## Connecting the studio

- **Easiest.** Open `http://127.0.0.1:8675/studio/`. It is the same origin,
  so no CORS and no token are needed on loopback. Then **Servers → Add
  server** with base URL `http://127.0.0.1:8675`, plus the token if you set
  one. The server shows a *Studio API* badge, and **Create a GPU server**
  rents your first GPU server through it.
- **Studio dev server.** `cd ui-go/studio && npm run dev` (port 5174). The
  Vite proxy sends `/aitk` to `AITK_API` (default `http://127.0.0.1:8675`), so
  add a server with base URL `/aitk`. No CORS is needed.
- **A studio hosted elsewhere** (for example `https://studio.example`). Start
  the studio API with `AITK_CORS_ORIGINS=https://studio.example` and add the
  server `http://127.0.0.1:8675`. Browsers treat an https page calling
  `http://127.0.0.1` differently: Chrome may ask for local network access, and
  Safari may block it as mixed content. This is not verified, so use the
  studio API's own `/studio/` when you can.

## Small image (optional)

A studio API image with the Go server and the studio only: no CUDA and no
Python, so it is far smaller than the GPU image and needs no GPU.

```bash
docker build -f ui-go/docker/studio-api.Dockerfile -t aitk-studio-api .        # from the repo root
docker run -d --name aitk-studio-api -p 127.0.0.1:8675:8675 \
  -e AI_TOOLKIT_AUTH=<a long random token> -v aitk-studio-api:/data aitk-studio-api
```

Then open `http://127.0.0.1:8675/studio/` and add the server with that token.
The token is required because the container listens on `0.0.0.0` inside.
The volume keeps the provider keys and the machine list across restarts, and
`-p 127.0.0.1:…` publishes the port on loopback only. Pass
`--build-arg GIT_SHA=$(git rev-parse --short HEAD)` to the build so
`/v1/health` reports the build.

## Deploy with Dokploy

`ui-go/docker/compose.studio.yml` runs `ghcr.io/geocine/aitoolkit-studio`
behind Dokploy's Traefik, with HTTPS.

1. **Create** a Compose service from this repo: branch `studio`, compose path
   `ui-go/docker/compose.studio.yml`.
2. **Environment**: set `AI_TOOLKIT_AUTH` (a long random token) and
   `AITK_PUBLIC_URL` (`https://loraboard.com`).
   - The deploy fails at once, naming the variable, if either is missing.
   - Optional: provider keys (`AITK_PROVISION_*_KEY`), `AITK_CORS_ORIGINS`,
     `AITK_PROVISION_SIMULATE=1`, and `AITK_STUDIO_TAG` to pin a build instead
     of `latest`.
3. **Domains**: service `studio`, container port `8675`, your host, HTTPS on.
   Dokploy adds the Traefik labels and network itself. The compose file
   publishes no host port.
4. **Registry**: the image is private unless you make the package public. Add
   `ghcr.io` under Dokploy's registries with a GitHub token that has
   `read:packages`.
5. **Deploy**, open `https://<host>/studio/` and add the server
   `https://<host>` with the token.

The `studio-data` volume holds the provider keys, machines and settings. It
survives redeploys, and Dokploy's Volume Backups can back it up. Deleting the
service with "volumes" ticked deletes it, along with the record of every GPU
server still billing.

## Connect an agent

Point an MCP client at `http://127.0.0.1:8675/mcp` (streamable HTTP), with
`Authorization: Bearer <token>` when `AI_TOOLKIT_AUTH` or an API key is set.
Code mode then has `aitk.providers.*`, `aitk.hosts.recommend`,
`aitk.images.*` and `aitk.machines.*`. Trainer ops answer with the
studio API's 409 sentence. On an open studio API an agent on this machine
needs no token; one that connects under a name other than localhost needs
that name in `AITK_ALLOWED_HOSTS`.

With `AITK_PROVISION_SIMULATE=1`, every simulated GPU server is this same
studio API under another address. Code mode through such a machine's `/mcp`
works when the studio API has no credentials at all, or when
`AI_TOOLKIT_AUTH` is set. With API keys only (no `AI_TOOLKIT_AUTH`) it answers
401, so set `AI_TOOLKIT_AUTH` for that test. Its `/v1/health` also reports
`mode: studio-api` and its studio pages show the studio API note: that is the
honest result, and a simulated click-through ends at *Ready and linked*.

## Limits

- No runs, captions, dataset uploads or face checks: those routes answer 409
  with "This server is the studio API (no GPU, no trainer): it creates and
  tracks GPU servers. Do this on a GPU server: …". Reads such as
  `GET /v1/jobs` and `GET /v1/datasets` answer from the studio API's own empty
  folders.
- `/v1/system/cuda` is refused (it runs Python). `GET /v1/system` still
  answers, but nothing here uses the GPUs it lists.
- The console is not served: `/` redirects to `/studio/`.
- The bundle, portable, lifecycle and upload loops do not run, so *When done*
  is for the GPU servers it launches, not for the studio API
  (`/v1/lifecycle` answers 409).
