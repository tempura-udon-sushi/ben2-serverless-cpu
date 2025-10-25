# BEN2 Serverless Worker - CPU Version (Size Optimized)
# Target: <15 GB final image
# Expected size: ~12-13 GB

FROM runpod/worker-comfyui:5.4.1-base

LABEL maintainer="your-email@example.com"
LABEL description="ComfyUI BEN2 Background Removal Worker (CPU, Size Optimized)"
LABEL models="BEN2, Florence-2-base, NudeNet, Llama-3.1-8B"
LABEL worker.type="CPU"

WORKDIR /comfyui

# ============================================================================
# STAGE 1: System Setup
# ============================================================================

USER root

# Install runtime + build dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# STAGE 2: Copy Custom Nodes & Workflows
# ============================================================================

COPY ComfyUI/custom_nodes/ComfyUI_BEN2_ONNX /comfyui/custom_nodes/ComfyUI_BEN2_ONNX
COPY ComfyUI/custom_nodes/comfyui-florence2 /comfyui/custom_nodes/comfyui-florence2
COPY ComfyUI/custom_nodes/ComfyUI_LocalJSONExtractor /comfyui/custom_nodes/ComfyUI_LocalJSONExtractor
COPY ComfyUI/custom_nodes/comfyui-custom-scripts /comfyui/custom_nodes/comfyui-custom-scripts
COPY ComfyUI/custom_nodes/comfyui-kjnodes /comfyui/custom_nodes/comfyui-kjnodes
COPY ComfyUI/custom_nodes/save_image_no_metadata.py /comfyui/custom_nodes/save_image_no_metadata.py
COPY ComfyUI/user/default/workflows/BG_remove_BEN2_simple_1st.json /comfyui/user/default/workflows/BG_remove_BEN2_simple_1st.json
COPY ComfyUI/user/default/workflows/BG_remove_BEN2_simple_2nd.json /comfyui/user/default/workflows/BG_remove_BEN2_simple_2nd.json

# ============================================================================
# STAGE 3: Install Python Dependencies
# ============================================================================

# Install all Python packages in one layer to reduce size
RUN pip install --no-cache-dir \
    onnxruntime>=1.16.0 \
    nudenet \
    safetensors>=0.3.0 \
    transformers>=4.39.0 \
    accelerate>=0.26.0 \
    timm \
    peft \
    matplotlib \
    "huggingface_hub[cli]" && \
    # Install custom node requirements
    pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_BEN2_ONNX/requirements.txt && \
    pip install --no-cache-dir -r /comfyui/custom_nodes/comfyui-florence2/requirements.txt && \
    # Install llama-cpp-python with OpenBLAS
    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir llama-cpp-python && \
    # Clean up pip cache
    pip cache purge

# ============================================================================
# STAGE 4: Download Models & Cleanup Build Tools
# ============================================================================

RUN mkdir -p /comfyui/models/ben2_onnx /comfyui/models/llm /comfyui/models/LLM && \
    # Download BEN2 ONNX (~223 MB)
    echo "Downloading BEN2_Base.onnx..." && \
    hf download PramaLLC/BEN2 \
    --include "BEN2_Base.onnx" \
    --local-dir /comfyui/models/ben2_onnx && \
    # Download Llama 3.1 8B (~5.4 GB)
    echo "Downloading Llama-3.1-8B-Instruct-Q5_K_M.gguf..." && \
    hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --include "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" \
    --local-dir /comfyui/models/llm && \
    # Download Florence-2 (~850 MB)
    echo "Pre-downloading Florence-2-base model..." && \
    hf download microsoft/Florence-2-base \
    --local-dir /comfyui/models/LLM/models--microsoft--Florence-2-base && \
    # Pre-download NudeNet (~60 MB)
    echo "Pre-downloading NudeNet model..." && \
    python3 -c "from nudenet import NudeDetector; NudeDetector()" && \
    # IMPORTANT: Remove build tools to save ~1GB
    apt-get remove -y build-essential cmake && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Clean up HuggingFace cache
    rm -rf ~/.cache/huggingface/download/* && \
    # Verify models
    echo "=== Model Verification ===" && \
    ls -lh /comfyui/models/ben2_onnx/BEN2_Base.onnx && \
    find /comfyui/models/llm -name "*.gguf" -type f && \
    echo "=== Model Size Summary ===" && \
    du -sh /comfyui/models/ben2_onnx/ && \
    du -sh /comfyui/models/llm/ && \
    du -sh /comfyui/models/LLM/ && \
    du -sh /comfyui/models/ && \
    echo "==========================" && \
    # Set permissions
    chmod -R 755 /comfyui/models /comfyui/custom_nodes && \
    chown -R root:root /comfyui

# ============================================================================
# STAGE 5: Environment & Runtime
# ============================================================================

# Reinstall PyTorch for CPU-only to avoid CUDA dependencies
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir \
    torch==2.5.0+cpu \
    torchvision==0.20.0+cpu \
    torchaudio==2.5.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# CPU-specific environment
# Force CPU-only mode for PyTorch and ComfyUI
ENV CUDA_VISIBLE_DEVICES="" \
    FORCE_CUDA=0 \
    HF_HOME=/comfyui/models/LLM \
    TRANSFORMERS_CACHE=/comfyui/models/LLM \
    NUDENET_HOME=/root/.NudeNet \
    PYTHONUNBUFFERED=1

# Replace start script with CPU version that adds --cpu flag
COPY ben2-serverless-cpu/start_cpu.sh /start.sh
RUN chmod +x /start.sh

# Metadata
LABEL build.date="2025-10-25" \
      version="1.5-cpu-final" \
      models.ben2="BEN2_Base.onnx" \
      models.florence="Florence-2-base" \
      models.llama="Llama-3.1-8B-Instruct-Q5_K_M" \
      models.nudenet="detector_v2_default_checkpoint" \
      pytorch="cpu-only"

# Use CPU-optimized start script
CMD ["/start.sh"]
