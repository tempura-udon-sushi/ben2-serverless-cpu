# CPU Serverless Optimization Guide

## Overview

This guide covers advanced optimization strategies to improve performance and reduce costs for CPU-based serverless ComfyUI workflows.

---

## 📊 Current Baseline Performance

### v1.5-cpu-final (Unoptimized)
- **Fast workflow** (BEN2 only): 30-40s
- **Full workflow** (BEN2 + Florence + Llama): 5-8 minutes
- **Bottleneck**: Llama 8B (60-80% of time)
- **Image size**: 13.6 GB

---

## 🎯 Optimization Strategies

### 1. Model Quantization

#### What is it?
Reduce model precision from FP32 → INT8/INT4 to speed up inference.

#### Llama Quantization (Biggest Impact)

**Current**: Llama-3.1-8B-Instruct-Q5_K_M (5.4 GB)

**Options for faster CPU inference**:

| Model | Size | Speed vs Current | Accuracy |
|-------|------|------------------|----------|
| Q2_K | ~3.5 GB | **2-3x faster** | 85-90% |
| Q3_K_M | ~4.0 GB | **1.5-2x faster** | 90-95% |
| Q4_K_M | ~4.5 GB | **1.2-1.5x faster** | 95-97% |
| Q5_K_M | ~5.4 GB | Baseline | 98-99% |

**Implementation**:
```dockerfile
# Replace in Dockerfile
RUN wget -O /comfyui/models/LLM/Llama-3.1-8B-Instruct-Q2_K.gguf \
    https://huggingface.co/.../Llama-3.1-8B-Instruct-Q2_K.gguf
```

**Expected improvement**: 
- Full workflow: 5-8 min → **2-3 minutes**
- Cost reduction: ~60-70%

#### Florence-2 Quantization

Use smaller Florence-2 variant:

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| Florence-2-large | ~1.5 GB | Baseline | Best quality |
| Florence-2-base | ~850 MB | **1.3x faster** | Good quality ✅ Current |
| Florence-2-base-ft | ~850 MB | 1.3x faster | Fine-tuned |

**Already using base model - optimized!**

#### BEN2 ONNX Optimization

Enable ONNX Runtime optimizations:

```python
# In custom node code
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 8  # Match CPU cores
session_options.inter_op_num_threads = 2
session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

# Enable optimizations
session_options.add_session_config_entry('session.intra_op.allow_spinning', '1')
session_options.add_session_config_entry('session.inter_op.allow_spinning', '1')
```

**Expected improvement**: 10-20% faster BEN2 inference

---

### 2. Smaller Model Alternatives

#### Replace Llama 8B with Smaller Models

**Llama 3.2 1B** (~700 MB):
```dockerfile
RUN wget -O /comfyui/models/LLM/Llama-3.2-1B-Instruct-Q5_K_M.gguf \
    https://huggingface.co/...
```
- **Speed**: 8-10x faster than 8B
- **Quality**: Good for simple classification
- **Time**: 5-8 min → **45-60 seconds**

**TinyLlama 1.1B** (~600 MB):
```dockerfile
RUN wget -O /comfyui/models/LLM/TinyLlama-1.1B-Q5_K_M.gguf \
    https://huggingface.co/...
```
- **Speed**: 10-12x faster than 8B
- **Quality**: Basic classification
- **Time**: 5-8 min → **40-50 seconds**

**Phi-3-mini-4k** (~2.5 GB):
```dockerfile
RUN wget -O /comfyui/models/LLM/Phi-3-mini-4k-Q5_K_M.gguf \
    https://huggingface.co/...
```
- **Speed**: 3-4x faster than 8B
- **Quality**: Very good
- **Time**: 5-8 min → **90-120 seconds**

#### Replace Florence-2 with Faster Alternatives

**BLIP-2** (~2 GB, but faster):
- Faster inference
- Similar caption quality
- Better CPU optimization

**CLIP** (~500 MB):
- Much faster
- Classification only (no captions)
- Best for simple image-text matching

---

### 3. Workflow Optimizations

#### A. Conditional Execution

Skip unnecessary steps based on input:

```python
# Example: Skip Llama if Florence detects safe content
def should_run_safety_check(florence_caption):
    safe_keywords = ["landscape", "object", "product", "art"]
    if any(kw in florence_caption.lower() for kw in safe_keywords):
        return False  # Skip Llama
    return True
```

**Savings**: Skip 60-80% of processing time for safe images

#### B. Parallel Processing

Run independent models in parallel:

```python
# Instead of: Florence → Llama (sequential)
# Do: Florence + BEN2 → Llama (parallel)

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_florence = executor.submit(run_florence, image)
    future_ben2 = executor.submit(run_ben2, image)
    
    florence_result = future_florence.result()
    ben2_result = future_ben2.result()
```

**Savings**: 20-30% time reduction

#### C. Batch Processing

Process multiple images in one job:

```python
payload = {
    "input": {
        "workflow": prompt,
        "images": [
            {"name": "image1.png", "image": img1_b64},
            {"name": "image2.png", "image": img2_b64},
            {"name": "image3.png", "image": img3_b64}
        ]
    }
}
```

**Savings**: 
- Amortize cold start cost
- Share model loading time
- 50-70% cost reduction per image

#### D. Caching Strategies

Cache frequently processed images:

```python
import hashlib

def get_image_hash(image_data):
    return hashlib.md5(image_data).hexdigest()

# Check cache before processing
image_hash = get_image_hash(image_bytes)
cached_result = redis_client.get(f"result:{image_hash}")
if cached_result:
    return cached_result
```

**Savings**: 100% for duplicate images

---

### 4. Infrastructure Optimizations

#### A. CPU Selection

**Current**: Generic CPU (AVX2)

**Optimized**: Intel Ice Lake or AMD EPYC Milan (AVX512)

Benefits:
- 2x wider SIMD operations
- Faster matrix multiplications
- Better for AI workloads

**RunPod Configuration**:
```
Select: CPU5 (Compute-Optimized) + Newer processors
```

#### B. Thread Configuration

Optimize thread allocation:

```dockerfile
# In Dockerfile
ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    VECLIB_MAXIMUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    OMP_WAIT_POLICY=ACTIVE \
    OMP_PROC_BIND=close \
    OMP_PLACES=cores
```

**Expected improvement**: 10-15% faster

#### C. Memory Optimization

Pre-allocate and reuse memory:

```python
# In ComfyUI custom node
import torch

# Reuse tensor memory
torch.set_num_threads(8)
torch.set_num_interop_threads(2)

# Disable gradient computation (inference only)
torch.set_grad_enabled(False)
```

#### D. Container Size Optimization

**Current image**: 13.6 GB

**Optimization strategies**:

1. **Remove unused dependencies**:
```dockerfile
RUN pip uninstall -y unused-package1 unused-package2
```

2. **Use smaller base Python image**:
```dockerfile
# Consider python:3.12-slim if compatible
```

3. **Multi-stage build**:
```dockerfile
FROM python:3.12 AS builder
# Download and prepare models

FROM python:3.12-slim
# Copy only what's needed
COPY --from=builder /comfyui /comfyui
```

**Target**: 10-11 GB (20% reduction)

---

### 5. Advanced Techniques

#### A. Model Distillation

Train smaller models to mimic larger ones:

```python
# Example: Distill Llama-8B → TinyLlama-1B
# Use 8B outputs as training data for 1B
```

**Benefits**:
- 80-90% smaller model
- 70-80% of original accuracy
- 8-10x faster inference

#### B. Dynamic Batching

Group concurrent requests:

```python
# Handler modification
def process_batch(jobs):
    images = [job["image"] for job in jobs]
    # Process all at once
    results = model.batch_inference(images)
    return results
```

**Savings**: 30-50% cost per job at high volume

#### C. Early Exit

Stop processing when confidence is high:

```python
# In safety classification
def classify_with_early_exit(image, caption):
    # Quick classification first
    quick_score = simple_classifier(caption)
    
    if quick_score > 0.95 or quick_score < 0.05:
        return quick_score  # High confidence
    
    # Only run Llama if uncertain
    return llama_classify(image, caption)
```

**Savings**: 50-70% time for clear cases

---

## 📋 Recommended Optimization Roadmap

### Phase 1: Quick Wins (1-2 hours)
- [ ] Use Llama Q2_K quantization (2-3x speedup)
- [ ] Optimize thread configuration
- [ ] Enable ONNX Runtime optimizations

**Expected**: 5-8 min → **2-3 minutes**

### Phase 2: Workflow Changes (4-6 hours)
- [ ] Implement conditional Llama execution
- [ ] Add batch processing support
- [ ] Parallel Florence + BEN2 execution

**Expected**: 2-3 min → **60-90 seconds**

### Phase 3: Model Replacement (1-2 days)
- [ ] Replace Llama 8B with Llama 3.2 1B
- [ ] Test alternative Florence models
- [ ] Benchmark quality vs speed

**Expected**: 60-90s → **30-45 seconds**

### Phase 4: Advanced (1-2 weeks)
- [ ] Implement dynamic batching
- [ ] Add caching layer
- [ ] Model distillation experiments

**Expected**: 30-45s → **20-30 seconds** + cost savings

---

## 💰 Cost Impact Analysis

### Current Cost (v1.5-cpu-final)

| Workflow | Time | Cost/Job | Volume/Month | Monthly Cost |
|----------|------|----------|--------------|--------------|
| Fast (BEN2 only) | 35s | $0.0020 | 10,000 | $20 |
| Full (w/ Llama 8B) | 6m | $0.035 | 10,000 | $350 |

### Phase 1 Optimizations

| Workflow | Time | Cost/Job | Volume/Month | Monthly Cost | Savings |
|----------|------|----------|--------------|--------------|---------|
| Fast | 30s | $0.0017 | 10,000 | $17 | -15% |
| Full (Llama Q2_K) | 2.5m | $0.015 | 10,000 | $150 | **-57%** |

### Phase 2 Optimizations

| Workflow | Time | Cost/Job | Volume/Month | Monthly Cost | Savings |
|----------|------|----------|--------------|--------------|---------|
| Fast | 25s | $0.0015 | 10,000 | $15 | -25% |
| Full (conditional) | 1.5m | $0.008 | 10,000 | $80 | **-77%** |

### Phase 3 Optimizations

| Workflow | Time | Cost/Job | Volume/Month | Monthly Cost | Savings |
|----------|------|----------|--------------|--------------|---------|
| Fast | 20s | $0.0012 | 10,000 | $12 | -40% |
| Full (Llama 1B) | 40s | $0.0023 | 10,000 | $23 | **-93%** |

---

## 🧪 Testing Methodology

### Benchmark Each Optimization

```python
import time

def benchmark_workflow(workflow_func, iterations=10):
    times = []
    for i in range(iterations):
        start = time.time()
        result = workflow_func()
        end = time.time()
        times.append(end - start)
    
    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "p95": sorted(times)[int(len(times) * 0.95)]
    }

# Test current vs optimized
baseline = benchmark_workflow(run_baseline_workflow)
optimized = benchmark_workflow(run_optimized_workflow)

speedup = baseline["mean"] / optimized["mean"]
print(f"Speedup: {speedup:.2f}x")
```

### Quality Validation

```python
def validate_quality(original_results, optimized_results):
    """Compare outputs to ensure quality maintained"""
    
    # For classification
    accuracy = sum(o == n for o, n in zip(original_results, optimized_results)) / len(original_results)
    
    # For captions (BLEU score)
    from nltk.translate.bleu_score import sentence_bleu
    bleu_scores = [
        sentence_bleu([orig.split()], opt.split()) 
        for orig, opt in zip(original_results, optimized_results)
    ]
    
    return {
        "accuracy": accuracy,
        "mean_bleu": sum(bleu_scores) / len(bleu_scores)
    }
```

---

## ⚠️ Trade-offs to Consider

| Optimization | Speed Gain | Quality Impact | Implementation Effort |
|--------------|------------|----------------|----------------------|
| Llama Q2_K quantization | +200% | -10-15% | Low |
| Llama 1B model | +800% | -20-30% | Low |
| Conditional execution | +150% | None | Medium |
| Parallel processing | +25% | None | Medium |
| Batch processing | +50% | None | High |
| Model distillation | +600% | -15-25% | Very High |

---

## 📚 Implementation Resources

### Quantization Tools
- **llama.cpp**: GGUF quantization
- **ONNX Runtime**: INT8 quantization
- **PyTorch**: Dynamic quantization

### Benchmarking Tools
```bash
# Install
pip install pytest-benchmark locust

# Run benchmarks
pytest test_optimization.py --benchmark-only
```

### Monitoring
```python
# Add to handler.py
import time
import logging

logger = logging.getLogger(__name__)

def log_timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__}: {duration:.2f}s")
        return result
    return wrapper
```

---

## 🎯 Next Steps

1. **Choose optimization phase** based on needs
2. **Benchmark baseline** performance
3. **Implement changes** incrementally
4. **Validate quality** after each change
5. **Measure cost savings** in production
6. **Document learnings** for future projects

---

## 📊 Success Metrics

Track these KPIs:

- **Mean processing time** (target: <45s for full workflow)
- **P95 latency** (target: <60s)
- **Cost per 1000 jobs** (target: <$25)
- **Quality score** (target: >85% vs baseline)
- **Error rate** (target: <1%)

---

## 🔗 Related Documentation

- [CPU Implementation Guide](CPU_IMPLEMENTATION_GUIDE.md)
- [RunPod Deployment Guide](RUNPOD_DEPLOYMENT.md)
- [Main README](README.md)

