FROM python:3.11-slim

WORKDIR /app

ARG PYTORCH_FLAVOR=cpu
ARG PYTORCH_VERSION=2.9.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    HF_HUB_CACHE=/data/huggingface/hub \
    TRANSFORMERS_CACHE=/data/huggingface/transformers \
    TORCH_HOME=/data/torch \
    NUMBA_CACHE_DIR=/data/numba \
    NUMBA_DISABLE_JIT=1 \
    CHATTERBOX_MODEL=turbo \
    CHATTERBOX_DEVICE=cpu \
    CHATTERBOX_ENABLE_DOCS=true \
    CHATTERBOX_MAX_TEXT_LENGTH=4000 \
    CHATTERBOX_DEFAULT_LANGUAGE=en \
    CHATTERBOX_DEFAULT_AUDIO_FORMAT=wav \
    CHATTERBOX_REF_VOICE_DIR=/data/reference-voices

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$PYTORCH_FLAVOR" != "cpu" ]; then \
        pip install --no-cache-dir \
            torch==${PYTORCH_VERSION} \
            torchaudio==${PYTORCH_VERSION} \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_FLAVOR}"; \
    else \
        pip install --no-cache-dir torch==${PYTORCH_VERSION} torchaudio==${PYTORCH_VERSION}; \
    fi && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps conformer==0.3.2 && \
    pip install --no-cache-dir --no-deps s3tokenizer==0.3.0 && \
    pip install --no-cache-dir --no-deps chatterbox-tts==0.1.6

COPY app.py .

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /data/huggingface/hub /data/huggingface/transformers /data/torch /data/reference-voices /data/numba && \
    chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
