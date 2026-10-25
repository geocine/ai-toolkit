#!/usr/bin/env bash
# Offline tests for ui-go/scripts/build_and_push.sh: a fake docker and curl
# (this folder) on PATH, a throwaway git repo as the checkout, and checks on the
# arguments the script hands to docker and on the builds it refuses.
#
#   bash ui-go/scripts/testdata/build_and_push_test.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
script="$here/../build_and_push.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/bin" "$work/repo/docker"
# The shims have no .sh suffix, so a CRLF checkout may have converted them.
for s in docker curl; do
  tr -d '\r' <"$here/$s" >"$work/bin/$s"
  chmod +x "$work/bin/$s"
done
export PATH="$work/bin:$PATH"
export SHIM_LOG="$work/docker.log"

# The checkout: upstream's version.py and docker/Dockerfile, one commit on
# main, so the merge-base is HEAD. The user's git config stays out of it.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.com
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.com
export GIT_COMMITTER_DATE='2026-10-20T08:30:00+08:00'
repo="$work/repo"
printf 'VERSION = "0.13.34"\n' >"$repo/version.py"
printf '# upstream\nFROM --platform=linux/amd64 nvidia/cuda:13.0.3-runtime-ubuntu24.04\nRUN true\n' >"$repo/docker/Dockerfile"
git -C "$repo" init -q -b main
git -C "$repo" add -A
git -C "$repo" commit -q -m upstream
SHA=$(git -C "$repo" rev-parse --short HEAD)
IMAGE=ghcr.io/geocine/aitoolkit-go

failed=0
name=
fail() { echo "FAIL $name: $*" >&2; failed=1; }

# run_case NAME WANT_EXIT [VAR=value ...] [--push]
run_case() {
  name=$1
  local want=$2 args=() envs=() a
  shift 2
  for a in "$@"; do
    if [[ "$a" == --* ]]; then args+=("$a"); else envs+=("$a"); fi
  done
  : >"$SHIM_LOG"
  set +e
  (cd "$repo" && env ${envs[@]+"${envs[@]}"} bash "$script" ${args[@]+"${args[@]}"}) >"$work/out" 2>&1
  local got=$?
  set -e
  if [[ $got != "$want" ]]; then
    fail "exit $got, want $want"
    sed 's/^/    /' "$work/out" >&2
  fi
}
log_has() { grep -qE -- "$1" "$SHIM_LOG" || fail "no docker call matching: $1"; }
log_lacks() { ! grep -qE -- "$1" "$SHIM_LOG" || fail "unexpected docker call matching: $1"; }
out_has() { grep -qF -- "$1" "$work/out" || fail "output lacks: $1"; }
final_build() { grep -E '^build \| --platform .*ui-go/docker/Dockerfile' "$SHIM_LOG"; }
build_arg() { final_build | grep -qF -- "--build-arg | $1 |" || fail "final build lacks --build-arg $1"; }

# A fresh base: both builds are linux/amd64 and every label comes from the probe.
run_case "build" 0 SHIM_BASE_EXISTS=0
log_has '^build \| --platform \| linux/amd64 \| -f \| docker/Dockerfile .* -$'
log_has '^build \| --platform \| linux/amd64 \| -f \| ui-go/docker/Dockerfile'
log_has '^run \| --rm \| --platform \| linux/amd64 \| --entrypoint \| python3 '
log_has '^run \| --rm \| --platform \| linux/amd64 \| --entrypoint \| du \| aitoolkit-upstream:[0-9a-f]* \| -sxb \| /$'
build_arg "AITK_SM=75,80,86,90,100,120"
build_arg "AITK_PTX="
build_arg "AITK_CUDA=13.0"
build_arg "AITK_DRIVER_MIN=580"
build_arg "AITK_FLAVOR=cu130"
build_arg "AITK_TORCH=2.13.0+cu130"
build_arg "AITK_SM_FLASH_ATTN=80,90,100,120"
build_arg "AITK_VERSION=0.13.34"
build_arg "AITK_REVISION_SHORT=$SHA"
build_arg "AITK_REVISION=$(git -C "$repo" rev-parse HEAD)"
build_arg "AITK_UPSTREAM=$(git -C "$repo" rev-parse HEAD | cut -c1-12)"
build_arg "AITK_BASE_NAME=nvidia/cuda:13.0.3-runtime-ubuntu24.04"
build_arg "AITK_BASE_UNPACKED=13689223168"
build_arg "GIT_SHA=$SHA"
# The commit time in UTC, so a rebuild of the same commit gets the same label.
build_arg "AITK_CREATED=2026-10-20T00:30:00Z"
for t in "$SHA" 0.13.34 cu130 latest; do
  final_build | grep -qF -- "-t | $IMAGE:$t" || fail "final build lacks tag $t"
done
final_build | grep -qE -- "-t \| $IMAGE:[0-9]{10}( |$)" || fail "final build lacks the timestamp tag"
log_lacks '^push'
out_has "sm             75,80,86,90,100,120"
out_has "base_unpacked  13689223168 bytes"

# A reused base is not rebuilt.
run_case "reuse base" 0 SHIM_BASE_EXISTS=1
log_lacks '^build .* docker/Dockerfile .* -$'
log_has '^build \| --platform \| linux/amd64 \| -f \| ui-go/docker/Dockerfile'

# No flash-attn in the base: an empty label, not a failure.
run_case "no flash-attn" 0 SHIM_BASE_EXISTS=1 "SHIM_PROBE=torch=2.13.0+cu130
cuda=13.0
sm=75,80,86,90,100,120
ptx=
flash_attn="
build_arg "AITK_SM_FLASH_ATTN="

# Refusals: nothing is built on top of a base the labels cannot describe.
run_case "empty sm" 1 SHIM_BASE_EXISTS=1 "SHIM_PROBE=torch=2.13.0+cu130
cuda=13.0
sm=
ptx=
flash_attn=2.8.3"
out_has "torch reports no CUDA arch flags"
[[ -z "$(final_build)" ]] || fail "built the Go layer anyway"

run_case "cuda major mismatch" 1 SHIM_BASE_EXISTS=1 "SHIM_PROBE=torch=2.13.0+cu128
cuda=12.8
sm=75,80
ptx=
flash_attn=2.8.3"
out_has "NVIDIA_REQUIRE_CUDA 13.0 and torch CUDA 12.8 differ in major"
[[ -z "$(final_build)" ]] || fail "built the Go layer anyway"

run_case "unknown flash-attn" 1 SHIM_BASE_EXISTS=1 "SHIM_PROBE=torch=2.13.0+cu130
cuda=13.0
sm=75,80
ptx=
flash_attn=2.9.0"
out_has "flash-attn 2.9.0: add its SASS list"
[[ -z "$(final_build)" ]] || fail "built the Go layer anyway"

run_case "du not a number" 1 SHIM_BASE_EXISTS=1 "SHIM_DU=du: cannot read"
out_has "not a byte count"
[[ -z "$(final_build)" ]] || fail "built the Go layer anyway"

run_case "no cuda floor" 1 SHIM_BASE_EXISTS=1 "SHIM_ENV=PATH=/usr/bin"
out_has "has no NVIDIA_REQUIRE_CUDA cuda>= floor"

run_case "unknown cuda" 1 SHIM_BASE_EXISTS=1 "SHIM_ENV=NVIDIA_REQUIRE_CUDA=cuda>=14.0" "SHIM_PROBE=torch=3.0.0+cu140
cuda=14.0
sm=120
ptx=
flash_attn="
out_has "unknown CUDA 14.0"

# Push: every tag, then the image check from inside the image.
run_case "push" 0 SHIM_BASE_EXISTS=1 --push
for t in "$SHA" 0.13.34 cu130 latest; do log_has "^push \| $IMAGE:$t$"; done
log_has "^run \| --rm \| --platform \| linux/amd64 \| --entrypoint \| timeout \| $IMAGE:$SHA \| -k \| 15 \| 120 \| /app/ai-toolkit/ui-go/aitk-server \| image-check \| --require-go \| $IMAGE:$SHA$"
log_lacks '^curl'
out_has "Verified $IMAGE:$SHA"

# The retag guard: an existing :<sha> with another digest stops the push.
run_case "retag refused" 1 SHIM_BASE_EXISTS=1 SHIM_REMOTE_DIGEST=sha256:aaaa --push
out_has "already exists (sha256:aaaa)"
out_has "build_and_push.sh --verify"
log_lacks '^push'

run_case "retag allowed" 0 SHIM_BASE_EXISTS=1 SHIM_REMOTE_DIGEST=sha256:aaaa ALLOW_RETAG=1 --push
log_has "^push \| $IMAGE:$SHA$"

run_case "same image again" 0 SHIM_BASE_EXISTS=1 SHIM_REMOTE_DIGEST=sha256:aaaa "SHIM_LOCAL_DIGESTS=$IMAGE@sha256:aaaa" --push
log_has "^push \| $IMAGE:$SHA$"

run_case "no buildx" 0 SHIM_BASE_EXISTS=1 SHIM_BUILDX=0 SHIM_REMOTE_DIGEST=sha256:aaaa --push
out_has "docker buildx is missing"
log_has "^push \| $IMAGE:$SHA$"

# Verify failures name the package settings page and how to re-check.
run_case "verify denied" 1 SHIM_BASE_EXISTS=1 SHIM_IMAGECHECK_EXIT=2 --push
out_has "https://github.com/users/geocine/packages/container/aitoolkit-go/settings"
out_has "re-run: ui-go/scripts/build_and_push.sh --verify"
log_lacks '^curl'

run_case "verify cannot run" 1 SHIM_BASE_EXISTS=1 SHIM_IMAGECHECK_EXIT=127 --push
out_has "could not run the image check"

# A server without image-check starts serving instead; timeout stops it (124,
# or 137 when it had to be killed) and the anonymous GHCR token stands in.
run_case "old server, private" 1 SHIM_BASE_EXISTS=1 SHIM_HAS_IMAGECHECK=0 SHIM_CURL_CODE=403 --push
log_has 'image-check \| --require-go'
log_has '^curl \| .*--connect-timeout 10 --max-time 30 .*https://ghcr.io/token\?scope=repository:geocine/aitoolkit-go:pull'
out_has "token HTTP 403"
out_has "https://github.com/users/geocine/packages/container/aitoolkit-go/settings"
out_has "re-run: ui-go/scripts/build_and_push.sh --verify"

run_case "old server, public" 0 SHIM_BASE_EXISTS=1 SHIM_HAS_IMAGECHECK=0 SHIM_CURL_CODE=200 --push
log_has '^curl'
out_has "only the anonymous pull token was checked"

run_case "old server, killed" 0 SHIM_BASE_EXISTS=1 SHIM_HAS_IMAGECHECK=0 SHIM_TIMEOUT_EXIT=137 --push
out_has "only the anonymous pull token was checked"

# No answer from ghcr.io says nothing about the package: no "make it public".
run_case "old server, ghcr unreachable" 1 SHIM_BASE_EXISTS=1 SHIM_HAS_IMAGECHECK=0 SHIM_CURL_CODE=000 --push
out_has "could not reach ghcr.io"
out_has "re-run: ui-go/scripts/build_and_push.sh --verify"
! grep -qF "make the package public" "$work/out" || fail "told to make the package public on a network error"
! grep -qF "settings" "$work/out" || fail "named the settings page on a network error"

# --verify only re-checks the pushed :<sha>: no base, probe, build or push.
run_case "verify only" 0 --verify
log_lacks '^build'
log_lacks '^push'
log_lacks 'python3'
log_lacks '^buildx'
log_lacks '^pull'
log_has "^run \| --rm \| --platform \| linux/amd64 \| --entrypoint \| timeout \| $IMAGE:$SHA \| -k \| 15 \| 120 \| /app/ai-toolkit/ui-go/aitk-server \| image-check \| --require-go \| $IMAGE:$SHA$"
out_has "Verified $IMAGE:$SHA"

run_case "verify only, pulls" 0 SHIM_IMAGE_EXISTS=0 --verify
log_has "^pull \| --platform \| linux/amd64 \| $IMAGE:$SHA$"
log_lacks '^build'
log_has 'image-check \| --require-go'

run_case "verify only, not pullable" 1 SHIM_IMAGE_EXISTS=0 SHIM_PULL_EXIT=1 --verify
out_has "could not be pulled"
log_lacks 'image-check'

run_case "verify only, still private" 1 SHIM_IMAGECHECK_EXIT=2 --verify
out_has "https://github.com/users/geocine/packages/container/aitoolkit-go/settings"

run_case "push and verify" 1 --push --verify
out_has "not both"
log_lacks '^build'

run_case "unknown argument" 1 --pusj
out_has "unknown argument --pusj"
log_lacks '^build'

if [[ $failed != 0 ]]; then
  echo "build_and_push_test.sh: FAIL" >&2
  exit 1
fi
echo "build_and_push_test.sh: ok"
