#!/usr/bin/env bash

# Extract version from version.py
if [ -f "version.py" ]; then
    VERSION=$(python3 -c "from version import VERSION; print(VERSION)")
    echo "Building version: $VERSION"
else
    echo "Error: version.py not found. Please create a version.py file with VERSION defined."
    exit 1
fi

echo "Docker builds from the repo, not this dir. Make sure changes are pushed to the repo."
echo "Building version: $VERSION and latest"
# wait 2 seconds
sleep 2

# Build the image with cache busting
docker build --build-arg CACHEBUST=$(date +%s) -t aitoolkit-geocine:$VERSION -f Dockerfile .

# Tag with version and latest
docker tag aitoolkit-geocine:$VERSION geocine/aitoolkit:$VERSION
docker tag aitoolkit-geocine:$VERSION geocine/aitoolkit:latest

# Push both tags
echo "Pushing images to Docker Hub..."
docker push geocine/aitoolkit:$VERSION
docker push geocine/aitoolkit:latest

echo "Successfully built and pushed geocine/aitoolkit:$VERSION and geocine/aitoolkit:latest"

