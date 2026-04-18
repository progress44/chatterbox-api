# syntax=docker/dockerfile:1
# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Activate the virtual environment for subsequent RUN commands
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install all Python dependencies in a single layer with pip cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install \
        fastapi \
        "uvicorn[standard]" \
        python-dotenv \
        python-multipart \
        requests \
        psutil \
        pydub \
        sse-starlette \
    && pip install git+https://github.com/travisvn/chatterbox-multilingual.git@exp \
    && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124 \
    && pip install numba==0.61.2 llvmlite==0.44.0

# Copy application code (changes here don't invalidate dependency layer)
COPY app/ ./app/
COPY main.py ./

# Voice sample is optional and can be mounted at runtime.
COPY voice-sample.mp3 ./voice-sample.mp3

# Create runtime directories. /tmp/numba must be writable by the non-root
# runtime user that Olares injects into the container.
RUN mkdir -p /cache /voices /data/long_text_jobs \
    && install -d -m 0777 /tmp/numba

# Set default environment variables
ENV PORT=4123
ENV EXAGGERATION=0.5
ENV CFG_WEIGHT=0.5
ENV TEMPERATURE=0.8
ENV VOICE_SAMPLE_PATH=/app/voice-sample.mp3
ENV MAX_CHUNK_LENGTH=280
ENV MAX_TOTAL_LENGTH=3000
ENV DEVICE=auto
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

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with the new entry point
CMD ["python", "main.py"]
