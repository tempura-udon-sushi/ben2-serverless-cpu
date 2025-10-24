#!/bin/bash

# Build script for BEN2 Serverless CPU Worker
# Usage: ./build.sh [tag]

set -e

IMAGE_NAME="ben2-serverless-cpu"
TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "======================================"
echo "Building BEN2 Serverless CPU Worker"
echo "======================================"
echo "Image: ${FULL_IMAGE}"
echo "Build context: Parent directory (..)"
echo ""

# Check if parent ComfyUI exists
if [ ! -d "../ComfyUI" ]; then
    echo "❌ Error: ComfyUI directory not found at ../ComfyUI"
    echo "Please ensure this script is run from ben2-serverless-cpu directory"
    exit 1
fi

echo "✓ ComfyUI directory found"
echo ""
echo "Starting build (this will take 10-15 minutes)..."
echo ""

# Build from parent directory to access ComfyUI
cd ..
docker build -f ben2-serverless-cpu/Dockerfile -t "${FULL_IMAGE}" .

echo ""
echo "======================================"
echo "✅ Build Complete!"
echo "======================================"
echo "Image: ${FULL_IMAGE}"
echo ""

# Show image size
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "Next steps:"
echo "1. Test locally: docker run -it ${FULL_IMAGE}"
echo "2. Push to registry: docker push your-registry/${FULL_IMAGE}"
echo "3. Deploy to RunPod with CPU worker type"
