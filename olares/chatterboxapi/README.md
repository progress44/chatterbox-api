# Chatterbox API for Olares

This package deploys the published image:

- `ghcr.io/progress44/rpi-system-chatterbox-api:latest`

The app exposes text-to-speech endpoints at:

- `http://chatterboxapi-svc:4123`

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /tts`
- `POST /v1/audio/speech`
- `POST /v1/audio/speech/upload`

## Request example

```bash
curl -X POST http://chatterboxapi-svc:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"turbo","input":"Hello from Olares","response_format":"wav"}' \
  --output speech.wav
```

## Notes

- The first synthesis request may be slower while model files are downloaded.
- Hugging Face and torch caches persist under `userspace.appData`.
- Use Olares env variables `OLARES_USER_HUGGINGFACE_TOKEN` and
  `OLARES_USER_HUGGINGFACE_SERVICE` if needed.
