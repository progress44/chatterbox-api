# Chatterbox API

FastAPI wrapper around [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox)
that exposes text-to-speech endpoints, including OpenAI-compatible speech routes.

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /tts`
- `POST /v1/audio/speech`
- `POST /v1/audio/speech/upload`

## OpenAI-style request example

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"turbo","input":"Hello from Chatterbox API","response_format":"wav"}' \
  --output speech.wav
```

## Environment Variables

- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `4123`)
- `DEVICE` (`auto|cpu|cuda|mps`, default: `auto`)
- `USE_MULTILINGUAL_MODEL` (`true|false`, default: `true`)
- `VOICE_LIBRARY_DIR` (default: `./voices`)
- `MODEL_CACHE_DIR` (default: `./models`)
- `MAX_TOTAL_LENGTH` (default: `3000`)
- `MAX_CHUNK_LENGTH` (default: `280`)

## Docker

Published image:

- `ghcr.io/progress44/rpi-system-chatterbox-api:latest`

Run locally:

```bash
docker run --gpus all -p 4123:4123 ghcr.io/progress44/rpi-system-chatterbox-api:latest
```

## Olares Packaging

Olares chart and manifest are under:

- `olares/chatterboxapi`

Package manually:

```bash
helm package olares/chatterboxapi
```
