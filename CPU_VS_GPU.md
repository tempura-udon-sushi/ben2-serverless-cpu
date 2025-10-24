# CPU vs GPU Version Comparison

## Key Differences

### Docker Image

| Aspect | CPU Version | GPU Version |
|--------|-------------|-------------|
| **Base Image** | `runpod/worker-comfyui:5.4.1-base` | `runpod/worker-comfyui:5.4.1-base` |
| **ONNX Runtime** | `onnxruntime` | `onnxruntime-gpu` |
| **llama-cpp-python** | CPU build with OpenBLAS | CUDA 12.4 build |
| **CUDA Support** | None | CUDA 12.4+ |
| **Size** | ~10 GB | ~10 GB |

### Performance

| Metric | CPU (4 vCPUs) | CPU (8 vCPUs) | GPU (RTX 4090) |
|--------|---------------|---------------|----------------|
| **Cold Start** | ~30-60s | ~30-60s | ~30-60s |
| **BEN2 Processing** | 8-15s | 5-10s | 1-2s |
| **Florence-2 Caption** | 10-20s | 6-12s | 2-4s |
| **Llama Safety Check** | 5-15s | 3-8s | 1-3s |
| **Total per Image** | ~25-50s | ~15-30s | ~5-10s |

### Cost (RunPod Serverless)

| Configuration | Cost per Second | Cost per Image* |
|---------------|----------------|-----------------|
| CPU (2 vCPUs, 4 GB) | $0.0000233/s | $0.0006/image |
| CPU (4 vCPUs, 8 GB) | $0.0000467/s | $0.0009/image |
| CPU (8 vCPUs, 16 GB) | $0.0000933/s | $0.0014/image |
| GPU (RTX 3070) | ~$0.0003/s | $0.0015/image |
| GPU (RTX 4090) | ~$0.0005/s | $0.0025/image |

*Estimated based on average processing time

### When to Use Each

#### Use CPU Version When:
✅ Testing and development  
✅ Low-volume processing (<100 images/day)  
✅ Cost is primary concern  
✅ Processing time <1 minute is acceptable  
✅ Learning/experimenting with workflows

#### Use GPU Version When:
✅ Production workloads  
✅ High-volume processing (>100 images/day)  
✅ Real-time or near-real-time processing needed  
✅ Processing time <10 seconds required  
✅ Multiple concurrent requests

## Technical Differences

### Environment Variables

**CPU Version:**
```dockerfile
ENV OMP_NUM_THREADS=0           # Use all CPU threads
ENV MKL_NUM_THREADS=0           # Use all MKL threads
ENV CUDA_VISIBLE_DEVICES=""     # Disable GPU
```

**GPU Version:**
```dockerfile
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Dependencies

**CPU-Specific:**
- `onnxruntime` (CPU)
- `llama-cpp-python` with OpenBLAS

**GPU-Specific:**
- `onnxruntime-gpu`
- `llama-cpp-python` with CUDA 12.4

### Workflow Adjustments

**BEN2 Node:**
```json
// CPU
"provider": "CPU"

// GPU  
"provider": "CUDA"
```

**Florence-2:**
```json
// CPU
"precision": "fp32"

// GPU
"precision": "fp16"
```

**Llama 3.1:**
```json
// CPU
"n_gpu_layers": 0

// GPU
"n_gpu_layers": -1  // All layers on GPU
```

## Optimization Tips

### CPU Version
1. **Use Compute-Optimized instances** (CPU5)
2. **Scale vCPUs** based on workload
3. **Batch processing** to amortize cold start
4. **Keep workers warm** during peak hours
5. **Consider caching** for repeated operations

### GPU Version
1. **Use latest RTX GPUs** (4090, 4080)
2. **Keep models in VRAM** (set `keep_model_loaded: true`)
3. **Batch multiple images** per request
4. **Monitor VRAM usage** (24GB recommended)
5. **Auto-scaling** for burst traffic

## Migration Path

### CPU → GPU
Simply change the RunPod endpoint configuration:
1. Stop CPU endpoint
2. Create new GPU endpoint
3. Use GPU Docker image
4. Update workflow JSON (provider: CUDA, precision: fp16)
5. Test and deploy

No code changes needed in client!

## Cost Optimization

### Hybrid Approach
```
┌─────────────────────────────────────┐
│         API Gateway / Router         │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
   Low Priority    High Priority
       │               │
    ┌──┴──┐         ┌──┴──┐
    │ CPU │         │ GPU │
    └─────┘         └─────┘
   Cheap/Slow    Expensive/Fast
```

Route based on:
- User tier (free → CPU, paid → GPU)
- Queue length (overflow to CPU)
- Time of day (off-peak → CPU)
- Request urgency flag

## Summary

| Use Case | Recommended |
|----------|-------------|
| Development | CPU |
| Testing | CPU |
| Demos | CPU |
| Low-volume (<10/hour) | CPU |
| Medium-volume (10-100/hour) | GPU |
| High-volume (>100/hour) | GPU |
| Real-time processing | GPU |
| Batch processing (overnight) | CPU |
| Cost-sensitive | CPU |
| Time-sensitive | GPU |

---

**Bottom Line:**  
CPU version is 5-10x cheaper but 3-5x slower.  
Use CPU for development/testing, GPU for production.
