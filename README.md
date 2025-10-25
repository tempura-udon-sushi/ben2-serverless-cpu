# BEN2 Serverless Worker - CPU Version

✅ **Working CPU implementation** for ComfyUI serverless on RunPod.

**Current Version**: `v1.5-cpu-final` (Oct 25, 2025)

## 📚 Documentation

- **[CPU Implementation Guide](CPU_IMPLEMENTATION_GUIDE.md)** - Complete guide for implementing CPU workflows (reusable template)
- **[CPU Optimization Guide](CPU_OPTIMIZATION_GUIDE.md)** - Advanced optimization strategies to improve speed and reduce costs
- **[RunPod Deployment Guide](RUNPOD_DEPLOYMENT.md)** - Step-by-step deployment instructions
- **This README** - Quick reference and overview

## 🎯 Purpose

Testing and development version of BEN2 serverless worker that runs on CPU. Use this for:
- Testing workflows without GPU costs
- Development and debugging
- Low-volume processing
- Cost-effective batch jobs

## 📦 What's Inside

### Docker Image
- **Base**: `runpod/worker-comfyui:5.4.1-base`
- **PyTorch**: CPU-only (2.5.0+cpu)
- **Size**: ~13.6 GB
- **Hub**: `zerocalory/ben2-serverless-cpu:v1.5-cpu-final`

### Models (All Pre-loaded)
- **BEN2_Base.onnx** (~223 MB) - Background removal
- **Florence-2-base** (~850 MB) - Image captioning  
- **Llama-3.1-8B-Instruct** (~5.4 GB) - Safety classification
- **NudeNet** (~60 MB) - Content safety

### Custom Nodes
- ComfyUI_BEN2_ONNX
- comfyui-florence2
- ComfyUI_LocalJSONExtractor
- comfyui-custom-scripts
- comfyui-kjnodes
- save_image_no_metadata.py

## ✅ Quick Start

### Option 1: Use Pre-built Image

```bash
# Already on Docker Hub - ready to use!
zerocalory/ben2-serverless-cpu:v1.5-cpu-final
```

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for deployment instructions.

### Option 2: Build Your Own

See [CPU_IMPLEMENTATION_GUIDE.md](CPU_IMPLEMENTATION_GUIDE.md) for complete build instructions.

## 🔧 Build Instructions (Custom)

### Prerequisites
- Docker installed
- ~15 GB free disk space
- ComfyUI installation at `../ComfyUI`

### Build Locally

```bash
# From the parent directory (Comfy_vanila)
cd ben2-serverless-cpu

# Build the image
docker build -t ben2-serverless-cpu:latest .

# Or use the build script
./build.sh
```

### Build with Tag

```bash
docker build -t your-dockerhub/ben2-serverless-cpu:1.0 .
```

## 🚀 Deploy to RunPod

### 1. Push to Docker Hub

```bash
docker tag ben2-serverless-cpu:latest your-dockerhub/ben2-serverless-cpu:latest
docker push your-dockerhub/ben2-serverless-cpu:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to RunPod Serverless
2. Click "New Endpoint"
3. Select **CPU Worker Type**
4. Choose configuration:
   - **CPU3** (Compute-Optimized): 3+ GHz, DDR5
   - **CPU5** (Compute-Optimized): 5+ GHz, DDR5 (Recommended)
5. Instance: 4 vCPUs / 8 GB RAM (minimum)
6. Container Image: `your-dockerhub/ben2-serverless-cpu:latest`
7. Container Disk: 20 GB
8. Deploy

### 3. Test the Endpoint

```python
import requests
import base64
import json

# Your endpoint details
ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run"
API_KEY = "your-runpod-api-key"

# Load and encode image
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Prepare payload
payload = {
    "input": {
        "workflow": {...},  # Your workflow JSON
        "images": [
            {
                "name": "input_image.png",
                "image": image_b64
            }
        ]
    }
}

# Submit job
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
job_id = response.json()["id"]

print(f"Job submitted: {job_id}")
```

## ⚡ Performance Notes

### Tested Performance (v1.5-cpu-final)

#### Fast Workflow (BEN2 Only)
- **Time**: 30-40 seconds
- **Models**: Background removal only
- **vCPUs**: 4-8
- **Cost**: ~$0.0015-0.003 per job
- **Use Case**: Production-ready for background removal

#### Full Workflow (BEN2 + Florence + Llama)
- **Time**: 5-8 minutes
- **Models**: BG removal + image captioning + safety check
- **vCPUs**: 8-16 recommended
- **Cost**: ~$0.03-0.05 per job
- **Bottleneck**: Llama 8B (60-80% of processing time)
- **Use Case**: Testing only; use GPU for production

### CPU vs GPU Comparison
- **Cold Start**: ~30-60s (similar for both)
- **BG Removal Only**: 
  - GPU: 15-20s
  - CPU: 30-40s
- **Full AI Pipeline**:
  - GPU: 15-30s
  - CPU: 5-8 minutes
- **Cost per hour**:
  - CPU (4 vCPUs): ~$0.17/hr
  - GPU (RTX 4090): ~$0.70-1.40/hr

### Optimization Tips
1. **Use fast workflow** - Remove Llama for 10x speedup
2. **Scale vCPUs** - 8 vCPUs for better performance
3. **Keep workers warm** - Reduce cold start delays
4. **Consider GPU** - For AI-heavy workflows

## 🔍 Model Configuration

### BEN2 ONNX
- Provider: `CPU` (set in workflow)
- Threads: Auto (uses all available)

### Llama 3.1 8B
- Context: 4096 tokens
- Threads: Auto-detected
- GPU layers: 0 (CPU only)

### Florence-2
- Device: CPU
- Precision: fp32 (CPU doesn't support fp16)

## 📊 Expected Image Size

| Component | Size |
|-----------|------|
| Base Image | ~2.5 GB |
| Custom Nodes | ~50 MB |
| BEN2 ONNX | ~223 MB |
| Florence-2 | ~850 MB |
| Llama 3.1 8B | ~5.4 GB |
| NudeNet | ~60 MB |
| Dependencies | ~1 GB |
| **Total** | **~10 GB** |

## 🛠️ Troubleshooting

### Slow Performance
- Increase vCPU count
- Use CPU5 instances (5+ GHz)
- Reduce model context length

### Out of Memory
- Increase RAM (8 GB minimum, 16 GB recommended)
- Reduce batch size in workflow

### Model Loading Errors
- Check logs for specific model issues
- Verify all models downloaded during build

## 📝 Workflow Format

The worker expects ComfyUI API format workflows. See `/ComfyUI/user/default/workflows/API/` for examples.

## 🔐 Security

All models run locally within the container. No external API calls except:
- HuggingFace downloads during build (can be avoided with pre-cached models)

## 📄 License

Check individual model licenses:
- BEN2: Apache 2.0
- Florence-2: MIT
- Llama 3.1: Llama 3.1 Community License
- NudeNet: MIT

## 🆚 GPU Version

For production workloads, see the GPU version: `ben2-serverless-gpu`

---

**Note**: This is a CPU version for testing. For production, use the GPU version for 5-10x faster processing.
