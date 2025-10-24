# BEN2 Serverless Worker - CPU Version
# For RunPod CPU Serverless deployment (testing/development)
# Expected size: ~9-10 GB

FROM runpod/worker-comfyui:5.4.1-base

LABEL maintainer="your-email@example.com"
LABEL description="ComfyUI BEN2 Background Removal Worker (CPU) with Safety Checks"
LABEL models="BEN2, Florence-2-base, NudeNet, Llama-3.1-8B"
LABEL worker.type="CPU"

WORKDIR /comfyui

# ============================================================================
# STAGE 1: Install System Dependencies
# ============================================================================

USER root

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# STAGE 2: Copy Custom Nodes
# ============================================================================

# Copy custom nodes from local installation
COPY ComfyUI/custom_nodes/ComfyUI_BEN2_ONNX /comfyui/custom_nodes/ComfyUI_BEN2_ONNX
COPY ComfyUI/custom_nodes/comfyui-florence2 /comfyui/custom_nodes/comfyui-florence2
COPY ComfyUI/custom_nodes/ComfyUI_LocalJSONExtractor /comfyui/custom_nodes/ComfyUI_LocalJSONExtractor
COPY ComfyUI/custom_nodes/comfyui-custom-scripts /comfyui/custom_nodes/comfyui-custom-scripts
COPY ComfyUI/custom_nodes/comfyui-kjnodes /comfyui/custom_nodes/comfyui-kjnodes
COPY ComfyUI/custom_nodes/save_image_no_metadata.py /comfyui/custom_nodes/save_image_no_metadata.py

# Copy workflows
COPY ComfyUI/user/default/workflows/API /comfyui/user/default/workflows/API

# ============================================================================
# STAGE 3: Install Python Dependencies (CPU-Optimized)
# ============================================================================

# Install ONNX Runtime (CPU version) - Much lighter than GPU version
RUN pip install --no-cache-dir onnxruntime>=1.16.0

# Install BEN2 ONNX dependencies
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_BEN2_ONNX/requirements.txt

# Install Florence-2 dependencies
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/comfyui-florence2/requirements.txt

# Install llama-cpp-python (CPU version with optimizations)
# CPU builds include BLAS/OpenBLAS for better performance
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir llama-cpp-python

# Install NudeNet for safety checking
RUN pip install --no-cache-dir nudenet

# Install additional dependencies
RUN pip install --no-cache-dir \
    safetensors>=0.3.0 \
    transformers>=4.39.0 \
    accelerate>=0.26.0 \
    timm \
    peft \
    matplotlib

# ============================================================================
# STAGE 4: Create Model Directories
# ============================================================================

RUN mkdir -p /comfyui/models/ben2_onnx && \
    mkdir -p /comfyui/models/llm && \
    mkdir -p /comfyui/models/LLM && \
    chmod -R 755 /comfyui/models

# ============================================================================
# STAGE 5: Download and Install Models
# ============================================================================

# Install huggingface-cli for reliable model downloads
RUN pip install -U "huggingface_hub[cli]"

# Download BEN2 ONNX model (~223 MB)
RUN echo "Downloading BEN2_Base.onnx..." && \
    hf download PramaLLC/BEN2 \
    --include "BEN2_Base.onnx" \
    --local-dir /comfyui/models/ben2_onnx && \
    echo "BEN2 model downloaded successfully"

# Download Llama 3.1 8B Instruct GGUF (~5.4 GB)
# Using Q5_K_M quantization for best quality/size balance
RUN echo "Downloading Llama-3.1-8B-Instruct-Q5_K_M.gguf..." && \
    hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --include "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" \
    --local-dir /comfyui/models/llm && \
    echo "Llama 3.1 model downloaded successfully"

# Pre-download Florence-2-base model files (download only, don't instantiate)
# Model will load properly at runtime by the custom node
RUN echo "Pre-downloading Florence-2-base model..." && \
    hf download microsoft/Florence-2-base \
    --local-dir /comfyui/models/LLM/models--microsoft--Florence-2-base && \
    echo "Florence-2 model files downloaded successfully"

# Pre-trigger NudeNet model download
# NudeNet auto-downloads on first use, so we'll trigger it during build
RUN echo "Pre-downloading NudeNet model..." && \
    python3 -c "from nudenet import NudeDetector; \
    print('Downloading NudeNet...'); \
    detector = NudeDetector(); \
    print('NudeNet downloaded and ready')" && \
    echo "NudeNet model downloaded successfully"

# ============================================================================
# STAGE 6: Verify Models
# ============================================================================

RUN echo "Verifying all models are present..." && \
    ls -lh /comfyui/models/ben2_onnx/BEN2_Base.onnx && \
    echo "Checking llm directory structure:" && \
    find /comfyui/models/llm -name "*.gguf" -type f && \
    ls -lh /comfyui/models/llm/*.gguf && \
    echo "Model verification complete"

# Display total model sizes
RUN echo "=== Model Size Summary ===" && \
    du -sh /comfyui/models/ben2_onnx/ && \
    du -sh /comfyui/models/llm/ && \
    du -sh /comfyui/models/LLM/ && \
    du -sh /comfyui/models/ && \
    echo "=========================="

# ============================================================================
# STAGE 7: Set Permissions
# ============================================================================

RUN chmod -R 755 /comfyui/models && \
    chmod -R 755 /comfyui/custom_nodes && \
    chown -R root:root /comfyui

# ============================================================================
# STAGE 8: Environment Variables (CPU-Optimized)
# ============================================================================

# CPU-specific: Use all available threads
ENV OMP_NUM_THREADS=0
ENV MKL_NUM_THREADS=0

# HuggingFace cache location
ENV HF_HOME=/comfyui/models/LLM
ENV TRANSFORMERS_CACHE=/comfyui/models/LLM

# NudeNet cache
ENV NUDENET_HOME=/root/.NudeNet

# Force CPU mode for frameworks
ENV CUDA_VISIBLE_DEVICES=""

# ============================================================================
# STAGE 9: Health Check & Metadata
# ============================================================================

# Add labels for tracking
LABEL build.date="2025-10-25"
LABEL version="1.0-cpu"
LABEL models.ben2="BEN2_Base.onnx"
LABEL models.florence="Florence-2-base"
LABEL models.llama="Llama-3.1-8B-Instruct-Q5_K_M"
LABEL models.nudenet="detector_v2_default_checkpoint"

# ============================================================================
# STAGE 10: Default Command
# ============================================================================

# Use the default ComfyUI worker start command
# This will be overridden by RunPod if needed
CMD ["/start.sh"]
