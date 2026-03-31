# syntax=docker/dockerfile:1
# Use NVIDIA CUDA runtime as base for better GPU support
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv (pinned for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /bin/uv

# Set working directory
WORKDIR /app

# Create virtual environment
RUN uv venv --python 3.11

# Activate the virtual environment for subsequent RUN commands
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG CHATTERBOX_MULTILINGUAL_SHA=c33cc0286ad166f14dd143c02c2a1c3309ab4727
COPY requirements.multilingual.lock.txt ./

# Install all Python dependencies in a single layer with uv cache mount.
# The cache mount persists the uv download cache across builds so unchanged
# packages are not re-downloaded even when this layer is invalidated.
RUN --mount=type=cache,target=/root/.cache/uv \
    echo "$CHATTERBOX_MULTILINGUAL_SHA" | grep -Eq '^[0-9a-f]{40}$' \
    && sed "s/__CHATTERBOX_MULTILINGUAL_SHA__/${CHATTERBOX_MULTILINGUAL_SHA}/g" \
        requirements.multilingual.lock.txt > /tmp/requirements.multilingual.lock.resolved.txt \
    && uv pip install \
        setuptools \
        fastapi \
        "uvicorn[standard]" \
        python-dotenv \
        python-multipart \
        requests \
        psutil \
        pydub \
        sse-starlette \
    && uv pip install -r /tmp/requirements.multilingual.lock.resolved.txt \
    && uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install numba==0.61.2 llvmlite==0.44.0

# Copy application code (changes here don't invalidate dependency layer)
COPY app/ ./app/
COPY main.py ./

# Voice sample is optional and can be mounted at runtime.

# Create directories for model cache and voice library (separate from source code)
RUN mkdir -p /cache /voices /data/long_text_jobs

# Set default environment variables (prefer CUDA)
ENV PORT=4123
ENV EXAGGERATION=0.5
ENV CFG_WEIGHT=0.5
ENV TEMPERATURE=0.8
ENV VOICE_SAMPLE_PATH=/app/voice-sample.mp3
ENV MAX_CHUNK_LENGTH=280
ENV MAX_TOTAL_LENGTH=3000
ENV DEVICE=cuda
ENV MODEL_CACHE_DIR=/cache
ENV VOICE_LIBRARY_DIR=/voices
ENV HOST=0.0.0.0

# Long text TTS settings
ENV LONG_TEXT_DATA_DIR=/data/long_text_jobs
ENV LONG_TEXT_MAX_LENGTH=100000
ENV LONG_TEXT_CHUNK_SIZE=2500
ENV LONG_TEXT_SILENCE_PADDING_MS=200
ENV LONG_TEXT_JOB_RETENTION_DAYS=7
ENV LONG_TEXT_MAX_CONCURRENT_JOBS=3
ENV NUMBA_CACHE_DIR=/tmp/numba
ENV NUMBA_DISABLE_JIT=1

# NVIDIA/CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application using the new entry point
CMD ["python", "main.py"]
