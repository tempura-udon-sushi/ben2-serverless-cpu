# RunPod Serverless Deployment Guide - CPU Version

## ✅ Current Working Image

- **Docker Hub**: `zerocalory/ben2-serverless-cpu:v1.5-cpu-final`
- **Size**: ~13.6 GB
- **Type**: CPU-optimized (PyTorch CPU-only)
- **Models**: BEN2, Florence-2, Llama 3.1 8B, NudeNet
- **Status**: ✅ Tested and working (Oct 25, 2025)

## Step-by-Step Deployment

### 1. Go to RunPod Serverless

https://www.runpod.io/console/serverless

### 2. Create New Endpoint

Click **"+ New Endpoint"**

### 3. Basic Configuration

**Endpoint Name**: `ben2-cpu-test`

**Worker Type**: ⚠️ **SELECT CPU** (not GPU!)

### 4. CPU Configuration

Select **CPU5 (Compute-Optimized)**
- 5+ GHz DDR5 RAM NVMe
- Recommended for faster processing

**Instance Configuration**:
- **4 vCPUs / 8 GB RAM** (minimum)
- **8 vCPUs / 16 GB RAM** (recommended for production)

**Cost**: ~$0.0000467/s (4 vCPUs) or ~$0.0000933/s (8 vCPUs)

### 5. Container Configuration

**Container Image**: 
```
zerocalory/ben2-serverless-cpu:v1.5-cpu-final
```

**Container Disk**: `20 GB` (minimum, 25 GB recommended)

**Docker Command**: Leave empty (uses default `/start.sh`)

### 6. Environment Variables (Optional)

```bash
OMP_NUM_THREADS=0
MKL_NUM_THREADS=0
```

(Already set in image, but can override if needed)

### 7. Advanced Settings

**Max Workers**: `1` (for testing)

**Idle Timeout**: `30` seconds (adjust based on usage)

**Execution Timeout**: `300` seconds (5 minutes)

**GPU Type**: Leave empty (CPU worker)

### 8. Deploy!

Click **"Deploy"**

Wait for:
1. ✅ Image pull (~5-10 min first time)
2. ✅ Container start (~30s)
3. ✅ Status: Active

## Testing Your Endpoint

### Get Your Endpoint Details

After deployment, you'll see:
- **Endpoint ID**: `your-endpoint-id`
- **API URL**: `https://api.runpod.ai/v2/your-endpoint-id`

### Test with Python

Use your existing test script:

```python
import requests
import base64
import json
import time

# Your RunPod endpoint details
ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run"
API_KEY = "your-runpod-api-key"

# Load workflow
with open("ComfyUI/user/default/workflows/API/BG_remove_BEN2_simple_1st_API.json", "r") as f:
    workflow = json.load(f)

# Fix model paths for CPU
workflow["15"]["inputs"]["model"] = "models--microsoft--Florence-2-base"
workflow["4"]["inputs"]["provider"] = "CPU"  # Important!

# Load image
with open("test_images/your_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Prepare payload
payload = {
    "input": {
        "workflow": workflow,
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

# Poll for result
status_endpoint = f"https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/{job_id}"

while True:
    time.sleep(5)
    status_response = requests.get(status_endpoint, headers=headers)
    status_result = status_response.json()
    
    status = status_result.get("status")
    print(f"Status: {status}")
    
    if status == "COMPLETED":
        # Save output images
        images = status_result["output"]["images"]
        for idx, img_data in enumerate(images):
            img_bytes = base64.b64decode(img_data["data"])
            with open(f"output_{idx}.png", "wb") as f:
                f.write(img_bytes)
        print(f"✅ Saved {len(images)} image(s)")
        break
    elif status == "FAILED":
        print(f"❌ Failed: {status_result.get('error')}")
        break
```

## Expected Performance (CPU)

### Cold Start
- **First request**: ~60-90s (image pull + model load)
- **Subsequent**: ~5-10s (if worker stays warm)

### Processing Time
| Step | 4 vCPUs | 8 vCPUs |
|------|---------|---------|
| Image resize | 1-2s | 0.5-1s |
| BEN2 processing | 8-12s | 4-6s |
| Florence-2 caption | 10-15s | 5-8s |
| Llama safety check | 5-10s | 3-5s |
| **Total** | **25-40s** | **15-25s** |

### Cost per Image
- **4 vCPUs**: ~$0.0012 per image
- **8 vCPUs**: ~$0.0019 per image

Compare to GPU (RTX 4090):
- **Processing**: 5-10s
- **Cost**: ~$0.0025 per image

## Workflow Configuration for CPU

Make sure your workflow uses CPU settings:

### BEN2 Node (Node 4)
```json
{
  "inputs": {
    "provider": "CPU",  // ⚠️ Must be CPU, not CUDA
    "background_color": "none",
    "sensitivity": 0.7,
    "mask_blur": 0,
    "mask_offset": -1
  }
}
```

### Florence-2 Node (Node 8)
```json
{
  "inputs": {
    "precision": "fp32",  // CPU uses fp32, not fp16
    "task": "more_detailed_caption"
  }
}
```

### Llama Safety Classifier (Node 25)
```json
{
  "inputs": {
    "n_gpu_layers": 0,  // ⚠️ Must be 0 for CPU
    "temperature": 0.1,
    "max_new_tokens": 300
  }
}
```

## Monitoring & Logs

### View Logs
1. Go to your endpoint in RunPod console
2. Click on **Logs** tab
3. Monitor for errors or performance issues

### Common Issues

#### 1. Timeout Errors
**Solution**: Increase execution timeout to 600s (10 min)

#### 2. Out of Memory
**Solution**: 
- Upgrade to 8 vCPUs / 16 GB RAM
- Reduce max_new_tokens in Llama

#### 3. Slow Performance
**Solution**:
- Use CPU5 instances (5+ GHz)
- Increase vCPU count
- Keep workers warm during peak hours

## Scaling Strategy

### Low Volume (<10 images/hour)
- 1 worker, 4 vCPUs
- Idle timeout: 60s
- Cost-effective for testing

### Medium Volume (10-100 images/hour)
- 2-3 workers, 8 vCPUs each
- Idle timeout: 300s
- Balance cost and latency

### High Volume (>100 images/hour)
- Consider switching to GPU version
- CPU becomes expensive at scale

## Cost Comparison

| Volume | CPU (4 vCPUs) | CPU (8 vCPUs) | GPU (RTX 4090) |
|--------|---------------|---------------|----------------|
| 10/hour | $0.012/hr | $0.019/hr | $0.025/hr |
| 100/hour | $0.12/hr | $0.19/hr | $0.25/hr |
| 1000/hour | $1.20/hr | $1.90/hr | $2.50/hr |

**Sweet Spot**: CPU is best for <50 images/hour

## Next Steps

1. ✅ Deploy to RunPod (follow steps above)
2. 🧪 Test with your images
3. 📊 Monitor performance and costs
4. 🚀 Scale or switch to GPU as needed

---

**Support**: If you encounter issues, check the logs or contact RunPod support.
