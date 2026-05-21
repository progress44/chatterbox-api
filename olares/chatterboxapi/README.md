# Chatterbox API for Olares

This package deploys Chatterbox API as an Olares shared application.

The Olares admin installs the shared GPU-backed API service once for the
cluster. Each user installation receives a lightweight user-space API entrance
that proxies to that shared service.

The shared service uses the published image:

- `ghcr.io/progress44/rpi-system-chatterbox-api:latest`

## Olares endpoints

Backend-to-backend clients inside Olares should use the hidden shared entrance:

- `http://chatterboxapi.shared.olares.com`

Browser clients or per-user integrations should use the normal user-space
entrance:

- `https://chatterboxapi.{OlaresID}.olares.com`

Inside the shared server namespace, the admin-installed service is exposed at:

- `http://chatterboxapi-svc:4123`

From another namespace, use:

- `http://chatterboxapi-svc.chatterboxapiserver-shared:4123`

User-space installations proxy through:

- `http://chatterboxapi-proxy:8080`

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

Shared Olares endpoint:

```bash
curl -X POST http://chatterboxapi.shared.olares.com/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"turbo","input":"Hello from Olares","response_format":"wav"}' \
  --output speech.wav
```

User-space endpoint:

```bash
curl -X POST https://chatterboxapi.{OlaresID}.olares.com/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"turbo","input":"Hello from Olares","response_format":"wav"}' \
  --output speech.wav
```

## Notes

- The first synthesis request may be slower while model files are downloaded.
- Hugging Face and torch caches persist under `userspace.appData`.
- Use Olares env variables `OLARES_USER_HUGGINGFACE_TOKEN` and
  `OLARES_USER_HUGGINGFACE_SERVICE` if needed.
