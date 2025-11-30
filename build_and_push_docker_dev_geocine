#!/usr/bin/env bash

VERSION=dev
GIT_COMMIT=dev

echo "Docker builds from the repo, not this dir. Make sure changes are pushed to the repo."
echo "Building version: $VERSION"
# wait 2 seconds
sleep 2

# Build the image with cache busting
docker build --build-arg CACHEBUST=$(date +%s) --build-arg GIT_COMMIT=${GIT_COMMIT} -t aitoolkit-geocine:$VERSION -f Dockerfile .

# Tag with version
docker tag aitoolkit-geocine:$VERSION geocine/aitoolkit:$VERSION

# Push tag
echo "Pushing images to Docker Hub..."
docker push geocine/aitoolkit:$VERSION

echo "Successfully built and pushed geocine/aitoolkit:$VERSION"

