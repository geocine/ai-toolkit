#!/usr/bin/env python3
"""Provider failover, end to end, on this machine (GPU required).

Runs ui-go/docker/compose.local.yml: MinIO plays Cloudflare R2 and two
containers of the same image play two GPU providers.

  1. node-a: an agent (MCP code mode) creates a dataset and an SD 1.5 LoRA job;
     images are uploaded over HTTP like an app would; the job is portable
     because the image is started with AITK_SYNC_REMOTE.
  2. after the step-40 checkpoint is synced, node-a is removed (the provider
     went away mid-training).
  3. node-b, a fresh server with an empty database, is started with the same
     env. It imports the job by itself (AITK_IMPORT_ON_BOOT=all) once node-a's
     lease expires and finishes it from the checkpoint. No API call is needed
     on node-b; the script only watches.

usage: python ui-go/scripts/failover_test.py [--keep]   (--keep leaves the stack up)
Stdlib only. Expect ~10 minutes, most of it waiting for the old lease to expire.
"""
import io
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE = ["docker", "compose", "-f", os.path.join(HERE, "..", "docker", "compose.local.yml")]
TOKEN = os.environ.get("AI_TOOLKIT_AUTH", "secret")
A, B = "http://127.0.0.1:18731", "http://127.0.0.1:18732"
JOB = "failover_lora_" + uuid.uuid4().hex[:4]
STEPS = 60


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def compose(*args):
    subprocess.run(COMPOSE + list(args), check=True)


def http(base, method, path, body=None, raw=None, ctype=None, timeout=120):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    if raw is not None:
        data, headers["Content-Type"] = raw, ctype
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return 0, None


def wait_up(base, what, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s, h = http(base, "GET", "/v1/health", timeout=5)
        if s == 200 and h and h.get("ok"):
            return
        time.sleep(2)
    raise SystemExit(f"{what} did not come up")


def mcp(base, code, timeout_ms=120000):
    """One aitk_run over MCP Streamable HTTP, like an agent's MCP client."""
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}

    def rpc(method, params, id_):
        msg = json.dumps({"jsonrpc": "2.0", "id": id_, "method": method, "params": params}).encode()
        req = urllib.request.Request(base + "/mcp", data=msg, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_ms / 1000 + 30) as r:
            text = r.read().decode()
            if r.headers.get("Content-Type", "").startswith("text/event-stream"):
                text = [l[5:].strip() for l in text.splitlines() if l.startswith("data:")][-1]
        return json.loads(text)

    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "failover-test", "version": "1"}}, 1)
    res = rpc("tools/call", {"name": "aitk_run", "arguments": {"code": code, "timeout_ms": timeout_ms}}, 2)["result"]
    out = res.get("structuredContent") or json.loads(res["content"][0]["text"])
    if out.get("error"):
        raise SystemExit(f"aitk_run failed: {out['error']}\nlogs: {out.get('logs')}")
    return out["result"]


def png(w, h, seed):
    rows = b"".join(b"\x00" + bytes(c for x in range(w) for c in (
        (x * 7 + seed * 50) % 256, (y * 5 + seed * 30) % 256, ((x ^ y) + seed * 70) % 256)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def multipart(files):
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for name, data in files:
        buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="files[]"; filename="{name}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n".encode())
        buf.write(data + b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def node_a():
    log("node-a: creating dataset (MCP)")
    ds = mcp(A, 'return (await aitk.datasets.create({ name: "failover_set" })).name;')
    files = []
    for i in range(4):
        files += [(f"p{i}.png", png(512, 512, i)), (f"p{i}.txt", f"ptrn style abstract pattern {i}".encode())]
    raw, ct = multipart(files)
    s, up = http(A, "POST", f"/v1/datasets/{ds}/files", raw=raw, ctype=ct)
    log(f"node-a: uploaded {len(up['items'])} files over HTTP ({s})")

    log("node-a: rendering an SD 1.5 template, creating and starting the job (MCP)")
    r = mcp(A, f"""
const r = await aitk.templates.render({{ arch: "sd15", name: "{JOB}", datasets: ["{ds}"], overrides: {{
  "config.process[0].model.arch": "sd1",
  "config.process[0].train.steps": {STEPS},
  "config.process[0].train.skip_first_sample": true,
  "config.process[0].save.save_every": 20,
  "config.process[0].sample.sample_every": 20,
  "config.process[0].sample.sample_steps": 10,
  "config.process[0].sample.samples": [{{ prompt: "ptrn style abstract pattern" }}],
  "config.process[0].network.linear": 8, "config.process[0].network.linear_alpha": 8,
  "config.process[0].datasets[0].resolution": [512],
}} }});
const job = await aitk.jobs.create({{ name: "{JOB}", gpu_ids: "0", config: r.config }});
await aitk.jobs.start(job.id);
return job.id;""")
    jid = r
    last = None
    while True:
        _, j = http(A, "GET", f"/v1/jobs/{jid}")
        _, sync = http(A, "GET", f"/v1/jobs/{jid}/sync")
        cur = (j["state"], j["step"], j["info"][:50], (sync or {}).get("step"))
        if cur != last:
            log(f"  node-a state={cur[0]:<10} step={cur[1]:<3} synced_step={cur[3]} info={cur[2]!r}")
            last = cur
        if j["state"] in ("failed", "completed"):
            raise SystemExit(f"job ended on node-a before the failover: {j['state']} {j['info']}")
        if j["step"] >= 45 and ((sync or {}).get("step") or 0) >= 40:
            return jid, sync
        time.sleep(2)


def node_b(t_gone):
    log("node-b: fresh server started with the same env; waiting for it to pick the job up")
    seen, last, t_import = None, None, None
    while True:
        _, jobs = http(B, "GET", "/v1/jobs")
        job = next((j for j in (jobs or {}).get("items", []) if j["name"] == JOB), None)
        if job is None:
            time.sleep(5)
            continue
        if t_import is None:
            t_import = time.time()
            log(f"node-b: imported {JOB} {t_import - t_gone:.0f}s after node-a went away (lease expiry)")
        cur = (job["state"], job["step"], job["info"][:50])
        if cur != last:
            log(f"  node-b state={cur[0]:<10} step={cur[1]:<3} info={cur[2]!r}")
            last = cur
            if job["state"] == "running" and seen is None and job["step"] > 0:
                seen = job["step"]
        if job["state"] in ("completed", "failed", "stopped"):
            return job, seen
        time.sleep(2)


def main():
    keep = "--keep" in sys.argv
    compose("rm", "-sf", "node-a", "node-b")
    compose("up", "-d", "minio", "bucket", "node-a")
    wait_up(A, "node-a")
    jid, sync = node_a()
    log(f"node-a: checkpoint at step {sync['step']} is on the remote; lease {json.dumps(sync.get('lease'))}")

    log("node-a: removing the container mid-training (provider gone)")
    compose("rm", "-sf", "node-a")
    t_gone = time.time()
    compose("--profile", "failover", "up", "-d", "node-b")
    wait_up(B, "node-b")
    job, first_step = node_b(t_gone)

    jid_b = job["id"]
    time.sleep(3)  # the supervisor records the attempt's ending on its next tick
    _, atts = http(B, "GET", f"/v1/jobs/{jid_b}/attempts")
    _, sync = http(B, "GET", f"/v1/jobs/{jid_b}/sync")
    _, cps = http(B, "GET", f"/v1/jobs/{jid_b}/checkpoints")
    _, loss = http(B, "GET", f"/v1/jobs/{jid_b}/loss?limit=1000")
    pts = (loss or {}).get("points", [])
    log("result:")
    log(f"  final state {job['state']} at step {job['step']} (first step seen on node-b: {first_step})")
    log("  attempts: " + ", ".join(f"#{a['n']} {a['start_step']}->{a['end_step']} {a['end_state']}" for a in atts["items"]))
    log("  checkpoints: " + ", ".join(c["name"] for c in cps["items"]))
    log(f"  loss points {len(pts)} covering steps {pts[0]['step'] if pts else '-'}..{pts[-1]['step'] if pts else '-'}")
    log(f"  remote: synced step {sync.get('step')}, lease {'released' if not sync.get('lease') else sync['lease']}")
    ok = (job["state"] == "completed" and job["step"] >= STEPS and first_step is not None and first_step >= 20
          and atts["items"][-1]["end_state"] == "completed" and not sync.get("lease"))
    log("PASS: continued on the new node from the synced checkpoint" if ok else "FAIL")
    if not keep:
        compose("down", "-v")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
