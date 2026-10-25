#!/usr/bin/env bash
# Build the fork image: upstream's docker/Dockerfile (unchanged, from the
# upstream commit this branch sits on) as the base, then ui-go's Go layer.
#
#   ui-go/scripts/build_and_push.sh            # build only
#   ui-go/scripts/build_and_push.sh --push     # build, push all tags, verify
#   ui-go/scripts/build_and_push.sh --verify   # only verify the pushed :<sha> of HEAD
#
# --push needs `docker login ghcr.io` first, with a classic personal access
# token that has the write:packages scope (docs/deploy.md, "Publishing the
# first time").
#
# Env: IMAGE (default ghcr.io/geocine/aitoolkit-go), UPSTREAM_REF (default main,
# the branch that mirrors upstream), REBUILD_BASE=1 to rebuild the base image,
# BASE_INSTALLER=uv to build the base with uv instead of pip (opt-in; the
# Dockerfile is derived from upstream's by ui-go/scripts/uv_dockerfile.sh and
# the base is tagged aitoolkit-upstream:<sha12>-uv), ALLOW_RETAG=1 to replace
# an already pushed :<sha> that has a different digest.
#
# Both builds are linux/amd64. The Go layer carries labels that say what hosts
# the image runs on (CUDA floor, driver branch, torch and flash-attn SASS
# targets); they are read from the freshly built base, never typed by hand, and
# the build fails if the base does not look the way the labels expect.
# Tests: ui-go/scripts/testdata/build_and_push_test.sh (fake docker, offline).
set -euo pipefail

# Git Bash on Windows would rewrite /app/... container paths into C:/... ones.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

cd "$(git rev-parse --show-toplevel)"

IMAGE=${IMAGE:-ghcr.io/geocine/aitoolkit-go}
UPSTREAM_REF=${UPSTREAM_REF:-main}
PLATFORM=linux/amd64
PUSH=false
VERIFY_ONLY=false

die() { echo "build_and_push.sh: $*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    --push) PUSH=true ;;
    --verify) VERIFY_ONLY=true ;;
    *) die "unknown argument $arg (use --push or --verify)" ;;
  esac
done
if $PUSH && $VERIFY_ONLY; then die "--push already verifies; pass --push or --verify, not both"; fi

SHA=$(git rev-parse --short HEAD)
SETTINGS_URL="https://github.com/users/geocine/packages/container/$(basename "$IMAGE")/settings"
SERVER=/app/ai-toolkit/ui-go/aitk-server
RERUN="then re-run: ui-go/scripts/build_and_push.sh --verify"

# Verify $IMAGE:$SHA with the same Go code the studio uses, from inside the
# image (no Go needed here). A server built before the image-check subcommand
# existed ignores the argument and starts serving, so the check runs under
# timeout: 124 (137 once it had to be killed) means "older server", not a
# verdict. A string search of the binary cannot tell the two apart: the
# compiler folds short constant comparisons into integers.
verify_image() {
  local rc=0
  docker run --rm --platform "$PLATFORM" --entrypoint timeout "$IMAGE:$SHA" \
    -k 15 120 "$SERVER" image-check --require-go "$IMAGE:$SHA" || rc=$?
  case "$rc" in
    0)
      echo "Verified $IMAGE:$SHA"
      return
      ;;
    124 | 137) ;;
    125 | 126 | 127) die "could not run the image check inside $IMAGE:$SHA (docker exit $rc)" ;;
    *) die "$IMAGE:$SHA: anonymous pull fails or labels are wrong (exit $rc). If it says denied: make the package public at $SETTINGS_URL, $RERUN" ;;
  esac

  if [[ "$IMAGE" == ghcr.io/* ]] && command -v curl >/dev/null 2>&1; then
    # Older server: at least check that GHCR hands out an anonymous pull token,
    # which it refuses for a private (or missing) package. curl prints 000 when
    # no answer came back, which says nothing about the package.
    local repo=${IMAGE#ghcr.io/} code
    code=$(curl -s --connect-timeout 10 --max-time 30 -o /dev/null -w '%{http_code}' \
      "https://ghcr.io/token?scope=repository:$repo:pull" || true)
    case "${code:-000}" in
      200) echo "warning: this server has no image-check subcommand; only the anonymous pull token was checked" >&2 ;;
      000) die "could not reach ghcr.io to check anonymous pulls of $IMAGE; $RERUN" ;;
      *) die "anonymous pulls of $IMAGE are denied (token HTTP $code). Make the package public at $SETTINGS_URL, $RERUN" ;;
    esac
  else
    echo "warning: this server has no image-check subcommand; $IMAGE:$SHA was not verified" >&2
  fi
}

# Re-check an image pushed earlier (say, once the package is public) without
# building anything: a rebuild could move the immutable :<sha>.
if $VERIFY_ONLY; then
  if ! docker image inspect "$IMAGE:$SHA" >/dev/null 2>&1; then
    echo "== pulling $IMAGE:$SHA"
    docker pull --platform "$PLATFORM" "$IMAGE:$SHA" ||
      die "$IMAGE:$SHA is not here and could not be pulled; build and push it with --push"
  fi
  verify_image
  exit 0
fi

BASE_COMMIT=$(git merge-base HEAD "$UPSTREAM_REF")
BASE_INSTALLER=${BASE_INSTALLER:-pip}
case "$BASE_INSTALLER" in
  pip) BASE_IMAGE="aitoolkit-upstream:${BASE_COMMIT:0:12}" ;;
  uv) BASE_IMAGE="aitoolkit-upstream:${BASE_COMMIT:0:12}-uv" ;;
  *) echo "BASE_INSTALLER must be pip or uv, got: $BASE_INSTALLER" >&2; exit 1 ;;
esac
VERSION=$(sed -n 's/^VERSION = "\(.*\)"/\1/p' version.py)
SHA_FULL=$(git rev-parse HEAD)
TS=$(date +%y%m%d%H%M)

if ! git diff --quiet HEAD -- ui-go; then
  echo "warning: ui-go has uncommitted changes; the image is built from the working tree" >&2
fi

# The base only depends on the upstream commit, so an existing image for it is
# reused (the build cache alone is unreliable: its torch layers get evicted).
# REBUILD_BASE=1 forces a fresh one.
if [[ "${REBUILD_BASE:-}" != "1" ]] && docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "== base: reusing $BASE_IMAGE (upstream ${BASE_COMMIT:0:12})"
else
  echo "== base: upstream ${BASE_COMMIT:0:12} -> $BASE_IMAGE ($BASE_INSTALLER)"
  # The context is the upstream commit itself (git archive), so the base gets
  # exactly upstream's files and none of the local datasets/output/node_modules.
  DOCKERFILE=docker/Dockerfile
  ARCHIVE_ARGS=()
  if [[ "$BASE_INSTALLER" == uv ]]; then
    # Same upstream Dockerfile with pip installs rewritten to uv, added to the
    # archive root as Dockerfile.uv; the script fails if upstream's install
    # lines no longer match the shape it knows.
    UV_DIR=$(mktemp -d)
    trap 'rm -rf "$UV_DIR"' EXIT
    git show "$BASE_COMMIT:docker/Dockerfile" | bash ui-go/scripts/uv_dockerfile.sh >"$UV_DIR/Dockerfile.uv"
    DOCKERFILE=Dockerfile.uv
    ARCHIVE_ARGS=(--add-file="$UV_DIR/Dockerfile.uv")
  fi
  git archive --format=tar ${ARCHIVE_ARGS[@]+"${ARCHIVE_ARGS[@]}"} "$BASE_COMMIT" |
    docker build --platform "$PLATFORM" -f "$DOCKERFILE" \
      --build-arg GIT_COMMIT="$BASE_COMMIT" \
      --build-arg CACHEBUST="$BASE_COMMIT" \
      -t "$BASE_IMAGE" -
fi

# Probe the base once (no GPU, no network): torch's version, CUDA and SASS
# targets, and the installed flash-attn. One key=value per line, so the host
# needs no python or jq to read it.
PROBE_PY='
import importlib.metadata as md, torch
flags = (torch._C._cuda_getArchFlags() or "").split()
sm = sorted({int(f[3:]) for f in flags if f.startswith("sm_")})
ptx = sorted({int(f[8:]) for f in flags if f.startswith("compute_")})
def ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return ""
print("torch=" + torch.__version__)
print("cuda=" + (torch.version.cuda or ""))
print("sm=" + ",".join(map(str, sm)))
print("ptx=" + ",".join(map(str, ptx)))
print("flash_attn=" + ver("flash_attn"))
'
probe=$(docker run --rm --platform "$PLATFORM" --entrypoint python3 "$BASE_IMAGE" -c "$PROBE_PY") ||
  die "probing torch in $BASE_IMAGE failed"
probe_field() { sed -n "s/^$1=//p" <<<"$probe" | head -n 1; }
AITK_TORCH=$(probe_field torch)
TORCH_CUDA=$(probe_field cuda)
AITK_SM=$(probe_field sm)
AITK_PTX=$(probe_field ptx)
FLASH_ATTN=$(probe_field flash_attn)
[[ -n "$AITK_TORCH" ]] || die "torch is not importable in $BASE_IMAGE"
[[ -n "$AITK_SM" ]] || die "torch reports no CUDA arch flags in $BASE_IMAGE"
[[ -n "$TORCH_CUDA" ]] || die "torch in $BASE_IMAGE is not a CUDA build ($AITK_TORCH)"

# The host gate is the base's NVIDIA_REQUIRE_CUDA cuda>=X.Y floor, not the
# toolkit version: the container runtime refuses older drivers on that alone.
REQ=$(docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$BASE_IMAGE" |
  sed -n 's/^NVIDIA_REQUIRE_CUDA=cuda>=\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
[[ -n "$REQ" ]] || die "$BASE_IMAGE has no NVIDIA_REQUIRE_CUDA cuda>= floor"
[[ "${REQ%%.*}" == "${TORCH_CUDA%%.*}" ]] ||
  die "NVIDIA_REQUIRE_CUDA $REQ and torch CUDA $TORCH_CUDA differ in major"
AITK_CUDA=$REQ
AITK_FLAVOR=cu$(cut -d. -f1-2 <<<"$TORCH_CUDA" | tr -d .)

# Driver branch that first supports each CUDA minor (IMAGE-PLAN 3.2). Shown to
# people only; the checker gates on the cuda label.
case "$AITK_CUDA" in
  12.6) AITK_DRIVER_MIN=560 ;; 12.8) AITK_DRIVER_MIN=570 ;; 12.9) AITK_DRIVER_MIN=575 ;;
  13.0) AITK_DRIVER_MIN=580 ;; 13.1) AITK_DRIVER_MIN=590 ;; 13.2) AITK_DRIVER_MIN=595 ;;
  13.3) AITK_DRIVER_MIN=610 ;; 13.4) AITK_DRIVER_MIN=615 ;;
  *) die "unknown CUDA $AITK_CUDA: add its driver branch to build_and_push.sh" ;;
esac

# SASS targets of the prebuilt flash-attn wheel per version (its setup.py
# default; sm_86/89 run the sm_80 cubin). No flash-attn means an empty label.
case "$FLASH_ATTN" in
  "") AITK_SM_FLASH_ATTN= ;;
  2.8.3 | 2.8.3+*) AITK_SM_FLASH_ATTN=80,90,100,120 ;;
  *) die "flash-attn $FLASH_ATTN: add its SASS list to build_and_push.sh" ;;
esac

# Measured inside the base, not read from .Size: under Docker's containerd image
# store .Size is the compressed content (6.6 GB for a 13.7 GB base).
AITK_BASE_UNPACKED=$(docker run --rm --platform "$PLATFORM" --entrypoint du "$BASE_IMAGE" -sxb / | cut -f1) ||
  die "measuring $BASE_IMAGE failed"
[[ "$AITK_BASE_UNPACKED" =~ ^[0-9]+$ ]] || die "du in $BASE_IMAGE printed '$AITK_BASE_UNPACKED', not a byte count"
# First FROM of upstream's Dockerfile, skipping flags such as --platform=.
AITK_BASE_NAME=$(git show "$BASE_COMMIT:docker/Dockerfile" |
  awk '!done && toupper($1) == "FROM" { for (i = 2; i <= NF; i++) if ($i !~ /^--/) { print $i; done = 1; break } }')
# The commit time, not the build time: a cached rebuild of the same commit and
# base keeps its image id, so pushing it again counts as the same image below.
AITK_CREATED=$(TZ=UTC git show -s --date=format-local:%Y-%m-%dT%H:%M:%SZ --format=%cd HEAD)

cat <<EOF
== labels from $BASE_IMAGE
   torch          $AITK_TORCH
   cuda           $AITK_CUDA (torch CUDA $TORCH_CUDA)
   driver_min     $AITK_DRIVER_MIN
   flavor         $AITK_FLAVOR
   sm             $AITK_SM
   ptx            ${AITK_PTX:-(none)}
   flash_attn     ${FLASH_ATTN:-(not installed)} -> sm ${AITK_SM_FLASH_ATTN:-(none)}
   base           ${AITK_BASE_NAME:-(unknown)}
   base_unpacked  $AITK_BASE_UNPACKED bytes
EOF

TAGS=("$SHA" "$TS" "$VERSION" "$AITK_FLAVOR" latest)
echo "== go layer -> $IMAGE:{$(IFS=,; echo "${TAGS[*]}")}"
TAG_ARGS=()
for t in "${TAGS[@]}"; do TAG_ARGS+=(-t "$IMAGE:$t"); done
docker build --platform "$PLATFORM" -f ui-go/docker/Dockerfile \
  --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg GIT_SHA="$SHA" \
  --build-arg AITK_VERSION="$VERSION" --build-arg AITK_REVISION="$SHA_FULL" \
  --build-arg AITK_REVISION_SHORT="$SHA" --build-arg AITK_CREATED="$AITK_CREATED" \
  --build-arg AITK_UPSTREAM="${BASE_COMMIT:0:12}" --build-arg AITK_BASE_NAME="$AITK_BASE_NAME" \
  --build-arg AITK_FLAVOR="$AITK_FLAVOR" --build-arg AITK_CUDA="$AITK_CUDA" \
  --build-arg AITK_DRIVER_MIN="$AITK_DRIVER_MIN" --build-arg AITK_TORCH="$AITK_TORCH" \
  --build-arg AITK_SM="$AITK_SM" --build-arg AITK_PTX="$AITK_PTX" \
  --build-arg AITK_SM_FLASH_ATTN="$AITK_SM_FLASH_ATTN" \
  --build-arg AITK_BASE_UNPACKED="$AITK_BASE_UNPACKED" \
  "${TAG_ARGS[@]}" .

$PUSH || exit 0

# :<sha> is the tag GPU servers launch, so it must not move. A REBUILD_BASE=1
# can change unpinned pip deps under the same sha; the same image pushed again
# (its digest is among the local repo digests) is fine.
if docker buildx version >/dev/null 2>&1; then
  if remote=$(docker buildx imagetools inspect "$IMAGE:$SHA" --format '{{json .Manifest.Digest}}' 2>/dev/null); then
    remote=${remote//\"/}
    local_digests=$(docker image inspect -f '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE:$SHA" 2>/dev/null || true)
    if ! grep -qxF "$IMAGE@$remote" <<<"$local_digests" && [[ "${ALLOW_RETAG:-}" != 1 ]]; then
      die "$IMAGE:$SHA already exists ($remote) with a different image; set ALLOW_RETAG=1 to replace it," \
        "or run ui-go/scripts/build_and_push.sh --verify to only re-check the pushed one"
    fi
  fi
else
  echo "warning: docker buildx is missing, so an existing $IMAGE:$SHA is not checked before it is replaced" >&2
fi

for t in "${TAGS[@]}"; do docker push "$IMAGE:$t"; done
echo "Pushed $IMAGE:{$(IFS=,; echo "${TAGS[*]}")}"

verify_image
