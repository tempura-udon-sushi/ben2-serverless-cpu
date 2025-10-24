# BEN2 Serverless Worker - CPU Version

CPU-optimized Docker image for background removal using BEN2 ONNX, Florence-2, and safety checks.

## 🎯 Purpose

Testing and development version of BEN2 serverless worker that runs on CPU. Use this for:
- Testing workflows without GPU costs
- Development and debugging
- Low-volume processing
- Cost-effective batch jobs

## 📦 What's Inside

### Models (All Baked In)
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

## 🔧 Build Instructions

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

### CPU vs GPU
- **Cold Start**: ~30-60s (similar to GPU)
- **Processing Time**: 
  - GPU: 2-5s per image
  - CPU: 10-30s per image (varies by vCPU count)
- **Cost**: ~$0.0002/s (vs ~$0.0004/s GPU)

### Optimization Tips
1. **Use CPU5 instances** - 5+ GHz significantly faster
2. **Scale vCPUs** - 8 vCPUs recommended for production
3. **Keep workers warm** - Set idle timeout to reduce cold starts
4. **Batch processing** - Process multiple images per job

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
