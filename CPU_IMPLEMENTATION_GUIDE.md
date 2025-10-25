# CPU Serverless Implementation Guide

## Overview

This guide explains how we successfully implemented ComfyUI workflows on RunPod CPU serverless endpoints. Use this as a template for deploying other CPU-based workflows.

---

## ✅ What We Achieved

- **Working CPU endpoint** with ComfyUI + custom nodes
- **CPU-only PyTorch** (no CUDA dependencies)
- **Pre-loaded models**: BEN2, Florence-2, Llama 3.1 8B, NudeNet
- **Fast workflow**: 30-40s for background removal only
- **Full workflow**: 5+ minutes with AI models (Llama bottleneck)

---

## 🏗️ Architecture

### Base Image
```
runpod/worker-comfyui:5.4.1-base
```

### Key Components
1. **CPU-only PyTorch** - Prevents CUDA initialization errors
2. **Custom start script** - Adds `--cpu` flag to ComfyUI
3. **Pre-downloaded models** - Cached during build
4. **Custom nodes** - Installed and configured for CPU

---

## 📋 Step-by-Step Implementation

### 1. Dockerfile Structure

```dockerfile
FROM runpod/worker-comfyui:5.4.1-base

# Stage 1: Install dependencies
RUN apt-get update && apt-get install -y wget curl git

# Stage 2: Copy custom nodes
COPY ComfyUI/custom_nodes/YourCustomNode /comfyui/custom_nodes/YourCustomNode

# Stage 3: Install Python dependencies
RUN pip install --no-cache-dir \
    onnxruntime>=1.18.0 \
    transformers>=4.46.0 \
    your-other-deps

# Stage 4: Download models (HuggingFace, ONNX, etc.)
RUN mkdir -p /comfyui/models/your_model_type && \
    wget -O /comfyui/models/your_model_type/model.onnx \
    https://your-model-url.com/model.onnx

# Stage 5: **CRITICAL** - Install CPU-only PyTorch
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir \
    torch==2.5.0+cpu \
    torchvision==0.20.0+cpu \
    torchaudio==2.5.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Stage 6: Set CPU environment variables
ENV CUDA_VISIBLE_DEVICES="" \
    FORCE_CUDA=0 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1

# Stage 7: Add custom start script with --cpu flag
COPY your-project/start_cpu.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
```

### 2. Custom Start Script

**Critical**: The base image's start script doesn't include `--cpu` flag.

Create `start_cpu.sh`:

```bash
#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode
comfy-manager-set-mode offline || echo "Could not set offline mode" >&2

echo "Starting ComfyUI in CPU mode"

: "${COMFY_LOG_LEVEL:=DEBUG}"

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --cpu --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --cpu --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    python -u /handler.py
fi
```

**Key addition**: `--cpu` flag on `main.py`

### 3. Model Download Strategies

#### HuggingFace Models
```dockerfile
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('microsoft/Florence-2-base', cache_dir='/comfyui/models/LLM'); \
    AutoTokenizer.from_pretrained('microsoft/Florence-2-base', cache_dir='/comfyui/models/LLM')"
```

#### Direct Download (ONNX, etc.)
```dockerfile
RUN wget -O /comfyui/models/ben2_onnx/BEN2_Base.onnx \
    https://huggingface.co/.../BEN2_Base.onnx
```

#### Large Models (Llama, etc.)
```dockerfile
RUN wget -O /comfyui/models/LLM/model.gguf \
    https://huggingface.co/.../model.gguf
```

### 4. Workflow Configuration for CPU

**CRITICAL**: Modify your workflow JSON to use CPU providers:

```python
import json

# Load workflow
with open("workflow.json", "r") as f:
    workflow = json.load(f)

# Fix BEN2/ONNX nodes
for node_id, node in workflow.items():
    if node["class_type"] == "BEN2_ONNX_RemoveBg":
        node["inputs"]["provider"] = "CPU"  # Change from CUDA to CPU

# Fix Llama nodes
for node_id, node in workflow.items():
    if node["class_type"] == "LocalLLMLoader_AdvV2":
        node["inputs"]["n_gpu_layers"] = 0  # Force CPU-only

# Save modified workflow
with open("workflow_cpu.json", "w") as f:
    json.dump(workflow, f, indent=2)
```

### 5. Testing Script Template

```python
import requests
import json
import base64
from pathlib import Path
from datetime import datetime

ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "your_api_key"

# Load workflow
with open("workflow_cpu.json", "r") as f:
    prompt = json.load(f)

# Load input image
with open("input.png", "rb") as img:
    image_data = base64.b64encode(img.read()).decode('utf-8')

# Prepare payload
payload = {
    "input": {
        "workflow": prompt,
        "images": [
            {
                "name": "input_image.png",
                "image": image_data
            }
        ]
    }
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Submit job
async_endpoint = ENDPOINT_URL.replace("/runsync", "/run")
response = requests.post(async_endpoint, json=payload, headers=headers)
result = response.json()
job_id = result.get("id")

# Poll for completion
import time
endpoint_id = ENDPOINT_URL.split("/v2/")[1].split("/")[0]
status_endpoint = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

while True:
    time.sleep(5)
    status_response = requests.get(status_endpoint, headers=headers)
    status_result = status_response.json()
    
    status = status_result.get("status")
    print(f"Status: {status}")
    
    if status == "COMPLETED":
        # Extract images
        images = status_result.get("output", {}).get("images", [])
        for idx, img_data in enumerate(images):
            if isinstance(img_data, dict):
                image_b64 = img_data.get("data", "")  # RunPod uses 'data' key!
            else:
                image_b64 = img_data
            
            if image_b64:
                output_path = f"output_{idx}.png"
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_b64))
                print(f"Saved: {output_path}")
        break
    elif status == "FAILED":
        print(f"Failed: {status_result.get('error')}")
        break
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: CUDA Error
```
AssertionError: Torch not compiled with CUDA enabled
```

**Solution**: Install CPU-only PyTorch + add `--cpu` flag to start script

### Issue 2: Server Not Reachable
```
ComfyUI server (127.0.0.1:8188) not reachable
```

**Solution**: ComfyUI crashed during startup. Check:
- Custom nodes are CPU-compatible
- All dependencies installed
- No CUDA-only operations

### Issue 3: Empty Output Images
```
Image file is 0 bytes
```

**Solution**: Use `img_data.get("data", "")` not `img_data.get("image", "")`

### Issue 4: ONNXRuntime pthread Warnings
```
pthread_setaffinity_np failed
```

**Solution**: Add environment variables:
```dockerfile
ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8
```

---

## 📊 Performance Benchmarks

### BEN2 Background Removal Only
- **Time**: 30-40 seconds
- **vCPUs**: 4-8
- **Cost**: ~$0.0015-0.003 per job

### Full Workflow (BEN2 + Florence-2 + Llama 8B)
- **Time**: 5-8 minutes
- **vCPUs**: 8-16 recommended
- **Cost**: ~$0.03-0.05 per job
- **Bottleneck**: Llama 8B on CPU (60-80% of total time)

### Recommendations
- **Light workflows**: CPU is cost-effective
- **AI-heavy workflows**: Consider GPU or smaller models
- **Production**: Use GPU for <30s response time

---

## 🚀 Build & Deploy Workflow

```bash
# 1. Build image
docker build -f Dockerfile -t your-image:v1 .

# 2. Tag for Docker Hub
docker tag your-image:v1 username/your-image:v1
docker tag your-image:v1 username/your-image:latest

# 3. Push to Docker Hub
docker push username/your-image:v1
docker push username/your-image:latest

# 4. Update RunPod endpoint
# - Go to RunPod console
# - Edit endpoint
# - Update container image to username/your-image:v1
# - Save and wait for workers to restart
```

---

## ✅ Verification Checklist

Before deploying to production:

- [ ] CPU-only PyTorch installed
- [ ] Custom start script includes `--cpu` flag
- [ ] All models pre-downloaded during build
- [ ] Workflow JSON configured for CPU providers
- [ ] Test script returns valid output images
- [ ] Image size matches expectations
- [ ] Processing time acceptable for use case
- [ ] Cost per job within budget

---

## 📚 Additional Resources

- **RunPod worker-comfyui**: https://github.com/runpod-workers/worker-comfyui
- **PyTorch CPU wheels**: https://download.pytorch.org/whl/cpu
- **This implementation**: `zerocalory/ben2-serverless-cpu:v1.5-cpu-final`

---

## 🔄 Version History

- **v1.5-cpu-final** (Oct 25, 2025): ✅ Working CPU implementation
- **v1.4-cpu-pytorch** (Oct 25, 2025): Added CPU PyTorch, missing `--cpu` flag
- **v1.3-broken** (Oct 25, 2025): Wrong handler path
- **v1.2-cpu-fix** (Oct 25, 2025): Initial CPU attempt
- **v1.1-optimized** (Earlier): GPU version

