#!/usr/bin/env bash
# Build the fork image: upstream's docker/Dockerfile (unchanged, from the
# upstream commit this branch sits on) as the base, then ui-go's Go layer.
#
#   ui-go/scripts/build_and_push.sh            # build only
#   ui-go/scripts/build_and_push.sh --push     # build and push all tags
#
# Env: IMAGE (default ghcr.io/geocine/aitoolkit-go), UPSTREAM_REF (default main,
# the branch that mirrors upstream), REBUILD_BASE=1 to rebuild the base image.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

IMAGE=${IMAGE:-ghcr.io/geocine/aitoolkit-go}
UPSTREAM_REF=${UPSTREAM_REF:-main}
PUSH=false
[[ "${1:-}" == "--push" ]] && PUSH=true

BASE_COMMIT=$(git merge-base HEAD "$UPSTREAM_REF")
BASE_IMAGE="aitoolkit-upstream:${BASE_COMMIT:0:12}"
VERSION=$(sed -n 's/^VERSION = "\(.*\)"/\1/p' version.py)
SHA=$(git rev-parse --short HEAD)
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
  echo "== base: upstream ${BASE_COMMIT:0:12} -> $BASE_IMAGE"
  # The context is the upstream commit itself (git archive), so the base gets
  # exactly upstream's files and none of the local datasets/output/node_modules.
  git archive --format=tar "$BASE_COMMIT" |
    docker build -f docker/Dockerfile \
      --build-arg GIT_COMMIT="$BASE_COMMIT" \
      --build-arg CACHEBUST="$BASE_COMMIT" \
      -t "$BASE_IMAGE" -
fi

TAGS=("$SHA" "$TS" "$VERSION" latest)
echo "== go layer -> $IMAGE:{$(IFS=,; echo "${TAGS[*]}")}"
TAG_ARGS=()
for t in "${TAGS[@]}"; do TAG_ARGS+=(-t "$IMAGE:$t"); done
docker build -f ui-go/docker/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg GIT_SHA="$SHA" "${TAG_ARGS[@]}" .

if $PUSH; then
  for t in "${TAGS[@]}"; do docker push "$IMAGE:$t"; done
  echo "Pushed $IMAGE:{$(IFS=,; echo "${TAGS[*]}")}"
fi
