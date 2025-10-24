#!/bin/bash
set -e

echo "=== Starting ComfyUI in CPU mode ==="
echo "Python: $(which python)"
echo "Working directory: $(pwd)"

# Start ComfyUI with --cpu flag
cd /comfyui
python main.py --cpu --listen 0.0.0.0 --port 8188 &
COMFYUI_PID=$!

echo "ComfyUI started with PID: $COMFYUI_PID"

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI server to start..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "✓ ComfyUI server is ready!"
        break
    fi
    echo "  Attempt $i/60: waiting..."
    sleep 2
done

# Start the RunPod handler
echo "Starting RunPod worker handler..."
python /comfyui_runpod_worker/handler.py
