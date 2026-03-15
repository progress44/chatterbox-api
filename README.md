# Chatterbox TTS API

A FastAPI-based HTTP wrapper around [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) text-to-speech models. This service provides a simple API to generate high-quality speech from text, supporting multiple models and an OpenAI-compatible endpoint.

## Features

- **Multiple Models:** Supports `turbo`, `english`, and `multilingual` models.
- **OpenAI Compatible:** Implements the `/v1/audio/speech` endpoint for easy integration with existing OpenAI-compatible clients.
- **Reference Voices:** Support for voice cloning/cloning via reference audio files (uploaded or from a pre-configured directory).
- **Multi-Format Output:** Supports `wav`, `flac`, and `ogg` audio formats.
- **Hardware Acceleration:** Native support for both CPU and NVIDIA GPU (CUDA) inference.
- **Multi-Arch Docker Images:** Published images support `amd64` and `arm64` (CPU).
- **Olares Ready:** Includes a Helm chart and Olares manifest for deployment in Olares environments.

## Getting Started

### Prerequisites

- Docker and Docker Compose (optional for local running)
- Python 3.11+ (for local development)
- NVIDIA Container Toolkit (for GPU acceleration)

### Running with Docker

The easiest way to run the service is using the pre-built images from GitHub Container Registry:

```bash
# Run CPU version
docker run -p 8000:8000 ghcr.io/progress44/rpi-system-chatterbox-tts:latest

# Run GPU version (requires NVIDIA GPU and container toolkit)
docker run --gpus all -p 8000:8000 ghcr.io/progress44/rpi-system-chatterbox-tts:latest-gpu
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATTERBOX_MODEL` | `turbo` | Default model to use (`turbo`, `english`, `multilingual`). |
| `CHATTERBOX_DEVICE` | `cpu` | Device for inference (`cpu`, `cuda`, `auto`). |
| `CHATTERBOX_DEFAULT_LANGUAGE` | `en` | Default language for multilingual model. |
| `CHATTERBOX_DEFAULT_AUDIO_FORMAT` | `wav` | Default output format (`wav`, `flac`, `ogg`). |
| `CHATTERBOX_REF_VOICE_DIR` | `/data/reference-voices` | Directory containing reference audio files for voice cloning. |
| `CHATTERBOX_MAX_TEXT_LENGTH` | `4000` | Maximum allowed characters per request. |
| `CHATTERBOX_ENABLE_DOCS` | `true` | Whether to enable FastAPI Swagger/ReDoc documentation. |

## API Endpoints

### 1. Synthesize Speech (Standard)
`POST /tts`

```json
{
  "text": "Hello world",
  "language": "en",
  "audio_format": "wav",
  "reference_voice": "my_voice.wav"
}
```

### 2. OpenAI Compatible Speech
`POST /v1/audio/speech`

Compatible with OpenAI's speech API.

```json
{
  "model": "turbo",
  "input": "The quick brown fox jumps over the lazy dog.",
  "voice": "alloy",
  "response_format": "mp3"
}
```

### 3. Speech with File Upload
`POST /v1/audio/speech/upload`

Allows uploading a reference audio file directly in the request (multipart/form-data).

## Deployment

### Helm Chart

A Helm chart is provided in the `chart/` directory.

```bash
helm install chatterboxtts ./chart
```

### Olares

This project is designed to be deployed as an Olares application. See `chart/OlaresManifest.yaml` for details.

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## License

This project incorporates [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox). Please refer to their license for usage terms of the underlying models.
