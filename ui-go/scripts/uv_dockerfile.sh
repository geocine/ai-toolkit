#!/usr/bin/env bash
# Derive a uv variant of upstream's docker/Dockerfile without editing it.
#
#   ui-go/scripts/uv_dockerfile.sh < docker/Dockerfile > Dockerfile.uv
#
# Every `pip install ...` in a RUN becomes `uv pip install --system ...` with the
# same requirements and the same meaning:
#   --no-cache-dir            -> --no-cache
#   --break-system-packages   -> kept (Ubuntu's python is EXTERNALLY-MANAGED)
#   --index-url X, --find-links X, -r FILE, pins, direct wheel URLs -> kept
#   -U                        -> --upgrade-package <name> per requirement on the
#                                line. pip's -U only upgrades the named packages;
#                                uv's --upgrade would upgrade every dependency
#                                too (e.g. replace the cu130 torch with PyPI's).
# uv is copied in from its pinned image right before the first install, with
# bytecode compilation on (pip's default, so first imports at boot stay fast).
#
# Anything outside that known shape (another flag, quoting, pip3, python -m pip,
# PIP_* env) fails the script rather than producing a silently different image;
# then extend the rules here after checking what upstream changed.
set -euo pipefail

UV_VERSION=${UV_VERSION:-0.12.24}

awk -v uv_version="$UV_VERSION" '
function die(msg) { printf "uv_dockerfile.sh: %s\n", msg > "/dev/stderr"; failed = 1; exit 1 }
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }

# Package name of a requirement token: a pin/spec ("natten==0.21.7+x") or a
# direct wheel URL (".../flash_attn-2.8.3+cu130...whl").
function req_name(tok,   n) {
    if (tok ~ /:\/\//) {
        n = tok; sub(/.*\//, "", n)
        if (n !~ /\.whl$/) die("direct URL is not a wheel: " tok)
        sub(/-.*/, "", n)
        return n
    }
    n = tok; sub(/[=<>!~\[;@].*/, "", n)
    if (n !~ /^[A-Za-z0-9][A-Za-z0-9._-]*$/) die("cannot parse requirement: " tok)
    return n
}

# One "pip install ..." command -> the uv command.
function translate(seg,   ntok, t, i, out, upgrade, reqs, nreq, has_r) {
    if (seg ~ /["\047`$(){}|;<>]/) die("unexpected shell syntax in: " seg)
    ntok = split(seg, t, /[ \t]+/)
    out = "uv pip install --system"; upgrade = 0; nreq = 0; has_r = 0
    for (i = 3; i <= ntok; i++) {
        if (t[i] == "--no-cache-dir") out = out " --no-cache"
        else if (t[i] == "--break-system-packages") out = out " " t[i]
        else if (t[i] == "-U" || t[i] == "--upgrade") upgrade = 1
        else if (t[i] == "--index-url" || t[i] == "--find-links" || t[i] == "-r") {
            if (i == ntok) die("missing value for " t[i] " in: " seg)
            if (t[i] == "-r") has_r = 1
            out = out " " t[i] " " t[i + 1]; i++
        }
        else if (t[i] ~ /^-/) die("unknown pip flag " t[i] " in: " seg)
        else { out = out " " t[i]; reqs[++nreq] = t[i] }
    }
    if (nreq == 0 && !has_r) die("no requirements in: " seg)
    if (upgrade) {
        if (has_r) die("-U with -r cannot be mapped to --upgrade-package: " seg)
        for (i = 1; i <= nreq; i++) out = out " --upgrade-package " req_name(reqs[i])
    }
    ntranslated++
    return out
}

function emit_uv_setup() {
    print "# --- uv (added by ui-go/scripts/uv_dockerfile.sh) ---"
    print "COPY --from=ghcr.io/astral-sh/uv:" uv_version " /uv /uvx /usr/local/bin/"
    # ARG, not ENV: visible to the RUN steps below, not baked into the image.
    print "ARG UV_LINK_MODE=copy"
    print "ARG UV_COMPILE_BYTECODE=1"
    print "ARG UV_HTTP_TIMEOUT=300"
    print "# --- end uv ---"
    setup_done = 1
}

function process(instr, phys,   joined, nseg, s, i, seg, out, cmd, rest, changed) {
    joined = instr
    gsub(/\n[ \t]*#[^\n]*/, "", joined)
    gsub(/\\\n/, " ", joined)
    cmd = joined; sub(/^[ \t]+/, "", cmd)
    if (cmd ~ /^#/ || cmd !~ /pip/) { printf "%s", phys; return }
    if (toupper(substr(cmd, 1, 4)) != "RUN ") die("pip outside a RUN: " cmd)
    if (cmd ~ /PIP_/) die("PIP_* setting would not apply to uv: " cmd)
    rest = substr(cmd, 5)
    nseg = split(rest, s, /&&/)
    out = ""
    for (i = 1; i <= nseg; i++) {
        seg = trim(s[i])
        if (seg ~ /^pip install /) { seg = translate(seg); changed = 1 }
        else if (seg ~ /(^|[^A-Za-z0-9_.-])pip[0-9.]*([ \t]|$)/) die("unhandled pip use: " seg)
        out = out (i == 1 ? "" : " && \\\n    ") seg
    }
    if (!changed) { printf "%s", phys; return }
    if (!setup_done) emit_uv_setup()
    print "RUN " out
}

{
    # A comment line inside a continued instruction does not end it.
    if (buf != "" && $0 ~ /^[ \t]*#/) { buf = buf $0 "\n"; next }
    buf = buf $0 "\n"
    if ($0 ~ /\\$/ && $0 !~ /^[ \t]*#/) next
    instr = buf; sub(/\n$/, "", instr)
    process(instr, buf)
    buf = ""
}

END {
    if (failed) exit 1
    if (buf != "") die("Dockerfile ends inside a continued line")
    if (ntranslated == 0) die("no pip install found; upstream changed shape")
}
'
