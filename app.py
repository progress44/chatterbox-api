import io
import logging
import os
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from threading import Lock
from typing import Literal

# Force a deterministic numba runtime before any Chatterbox/librosa imports.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/data/numba")

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS


MODEL_NAME = os.getenv("CHATTERBOX_MODEL", "turbo").strip().lower()
DEVICE = os.getenv("CHATTERBOX_DEVICE", "cpu").strip().lower()
DEFAULT_LANGUAGE = os.getenv("CHATTERBOX_DEFAULT_LANGUAGE", "en").strip().lower()
DEFAULT_AUDIO_FORMAT = os.getenv("CHATTERBOX_DEFAULT_AUDIO_FORMAT", "wav").strip().lower()
REF_VOICE_DIR = Path(os.getenv("CHATTERBOX_REF_VOICE_DIR", "/data/reference-voices"))
MAX_TEXT_LENGTH = int(os.getenv("CHATTERBOX_MAX_TEXT_LENGTH", "4000"))
ENABLE_DOCS = os.getenv("CHATTERBOX_ENABLE_DOCS", "true").strip().lower() == "true"
ALLOW_NOOP_WATERMARKER = os.getenv("CHATTERBOX_ALLOW_NOOP_WATERMARKER", "true").strip().lower() == "true"

SUPPORTED_AUDIO_FORMATS = {"wav", "flac", "ogg"}
MULTILINGUAL_MODEL_NAMES = {"multilingual", "mtl", "multi"}
TURBO_MODEL_NAMES = {"turbo", "tts_turbo"}
VOICE_FILE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
REQUEST_ID_HEADER = "X-Request-Id"


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("chatterbox-tts")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    language: str | None = Field(default=None, description="Use ISO code for multilingual mode, for example `en`, `sv`, `fr`.")
    audio_format: Literal["wav", "flac", "ogg"] = Field(default=DEFAULT_AUDIO_FORMAT)  # type: ignore[arg-type]
    reference_voice: str | None = Field(default=None, description="Optional file name from CHATTERBOX_REF_VOICE_DIR.")


class OpenAISpeechRequest(BaseModel):
    model: str | None = Field(default=None, description="Maps to CHATTERBOX_MODEL if omitted.")
    input: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    voice: str | None = Field(default=None, description="Optional file name from CHATTERBOX_REF_VOICE_DIR.")
    response_format: Literal["wav", "flac", "ogg"] = Field(default=DEFAULT_AUDIO_FORMAT)  # type: ignore[arg-type]
    language: str | None = Field(default=None)


class ServiceState:
    models = {}
    model_families = {}
    model_lock = Lock()
    startup_logged = False


state = ServiceState()
docs_url = "/docs" if ENABLE_DOCS else None
redoc_url = "/redoc" if ENABLE_DOCS else None
openapi_url = "/openapi.json" if ENABLE_DOCS else None

app = FastAPI(
    title="Chatterbox TTS API",
    description="HTTP wrapper around Resemble AI Chatterbox text-to-speech models.",
    version="0.1.0",
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)


def _request_id(request: Request | None) -> str:
    return getattr(getattr(request, "state", None), "request_id", "unknown")


def _client_ip(request: Request | None) -> str:
    if request is None or request.client is None:
        return "unknown"
    return request.client.host


def _log(event: str, **fields):
    payload = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
    logger.info("%s %s", event, payload)


def _log_warning(event: str, **fields):
    payload = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
    logger.warning("%s %s", event, payload)


def _log_exception(event: str, **fields):
    payload = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
    logger.exception("%s %s", event, payload)


def _reference_source(reference_voice_path: str | None) -> str:
    if not reference_voice_path:
        return "none"
    resolved = Path(reference_voice_path)
    with suppress(Exception):
        if resolved.resolve().is_relative_to(REF_VOICE_DIR.resolve()):
            return resolved.name
    return "uploaded"


def _log_startup():
    if state.startup_logged:
        return
    state.startup_logged = True
    _log(
        "startup",
        configured_device=DEVICE,
        configured_model=MODEL_NAME,
        cuda_available=torch.cuda.is_available(),
        default_audio_format=DEFAULT_AUDIO_FORMAT,
        default_language=DEFAULT_LANGUAGE,
        docs_enabled=ENABLE_DOCS,
        hf_home=os.getenv("HF_HOME", "/data/huggingface"),
        numba_cache_dir=os.getenv("NUMBA_CACHE_DIR", "/data/numba"),
        reference_voice_dir=str(REF_VOICE_DIR),
        torch_home=os.getenv("TORCH_HOME", "/data/torch"),
        transformers_cache=os.getenv("TRANSFORMERS_CACHE", "/data/huggingface/transformers"),
    )


def _normalize_model_name(model_name: str | None) -> str:
    requested = (model_name or MODEL_NAME).strip().lower()
    if requested in TURBO_MODEL_NAMES:
        return "turbo"
    if requested in MULTILINGUAL_MODEL_NAMES:
        return "multilingual"
    if requested in {"english", "tts", "default"}:
        return "english"
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported model `{requested}`. Available models: turbo, english, multilingual.",
    )


def _resolve_device() -> str:
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CHATTERBOX_DEVICE is set to cuda but CUDA is not available in the container.")
    return DEVICE


def _ensure_watermarker_runtime():
    try:
        import perth
    except ImportError as exc:  # pragma: no cover - dependency is expected in the runtime image
        raise RuntimeError("Perth watermarker dependency is unavailable. Ensure resemble-perth is installed.") from exc

    watermarker_cls = getattr(perth, "PerthImplicitWatermarker", None)
    if callable(watermarker_cls):
        return

    if ALLOW_NOOP_WATERMARKER:
        class _NoOpPerthImplicitWatermarker:
            def apply_watermark(self, wav, sample_rate=None):
                return wav

        perth.PerthImplicitWatermarker = _NoOpPerthImplicitWatermarker
        _log_warning(
            "watermarker_fallback_enabled",
            reason="PerthImplicitWatermarker_unavailable",
            workaround="noop_watermarker",
        )
        return

    raise RuntimeError(
        "PerthImplicitWatermarker is unavailable. This is commonly caused by an incompatible or missing "
        "setuptools/pkg_resources runtime with resemble-perth==1.0.1. Rebuild the image with setuptools<81 "
        "or upgrade resemble-perth to a version that removes the pkg_resources dependency."
    )


def _ensure_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    if len(normalized) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail=f"Text exceeds CHATTERBOX_MAX_TEXT_LENGTH ({MAX_TEXT_LENGTH}).")
    return normalized


def _resolve_reference_voice(reference_voice: str | None) -> str | None:
    if not reference_voice:
        return None

    candidate = (REF_VOICE_DIR / reference_voice).resolve()
    base = REF_VOICE_DIR.resolve()

    if not str(candidate).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Reference voice must stay within CHATTERBOX_REF_VOICE_DIR.")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Reference voice `{reference_voice}` was not found.")
    if candidate.suffix.lower() not in VOICE_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported reference voice file extension.")

    return str(candidate)


def _load_model(model_name: str | None = None):
    _log_startup()
    normalized_model = _normalize_model_name(model_name)

    if normalized_model in state.models:
        return state.models[normalized_model], state.model_families[normalized_model]

    with state.model_lock:
        if normalized_model in state.models:
            return state.models[normalized_model], state.model_families[normalized_model]

        device = _resolve_device()
        _ensure_watermarker_runtime()

        if normalized_model == "turbo":
            model = ChatterboxTurboTTS.from_pretrained(device=device)
            family = "turbo"
        elif normalized_model == "multilingual":
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            family = "multilingual"
        else:
            model = ChatterboxTTS.from_pretrained(device=device)
            family = "english"

        state.models[normalized_model] = model
        state.model_families[normalized_model] = family
        _log(
            "model_loaded",
            model_name=normalized_model,
            model_family=family,
            resolved_device=device,
        )
        return model, family


def _render_audio(
    *,
    text: str,
    language: str | None,
    audio_format: str,
    reference_voice_path: str | None,
    request_id: str,
    endpoint: str,
    model_name: str | None = None,
):
    normalized_model = _normalize_model_name(model_name)
    model, model_family = _load_model(normalized_model)
    normalized_text = _ensure_text(text)

    kwargs = {}
    if reference_voice_path:
        kwargs["audio_prompt_path"] = reference_voice_path

    if model_family == "multilingual":
        kwargs["language_id"] = (language or DEFAULT_LANGUAGE).strip().lower()
    elif language and language.strip().lower() != DEFAULT_LANGUAGE:
        raise HTTPException(status_code=400, detail=f"Language is only supported when the selected model is multilingual. Current model: {model_family}.")

    started_at = time.perf_counter()
    try:
        wav = model.generate(normalized_text, **kwargs)
    except Exception as exc:  # pragma: no cover - model runtime errors depend on environment
        _log_exception(
            "tts_generation_failed",
            endpoint=endpoint,
            language=(language or DEFAULT_LANGUAGE).strip().lower(),
            model_family=model_family,
            model_name=normalized_model,
            reference_source=_reference_source(reference_voice_path),
            request_id=request_id,
            response_format=audio_format,
            text_length=len(normalized_text),
        )
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()

    if getattr(wav, "ndim", 1) == 2:
        wav = wav[0]

    buffer = io.BytesIO()
    sf.write(buffer, wav, model.sr, format=audio_format.upper())
    audio_bytes = buffer.getvalue()
    _log(
        "tts_generation_succeeded",
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        endpoint=endpoint,
        language=(language or DEFAULT_LANGUAGE).strip().lower(),
        model_family=model_family,
        model_name=normalized_model,
        output_bytes=len(audio_bytes),
        reference_source=_reference_source(reference_voice_path),
        request_id=request_id,
        response_format=audio_format,
        sample_rate=model.sr,
        text_length=len(normalized_text),
    )
    return audio_bytes, model.sr


def _render_response(audio_bytes: bytes, audio_format: str) -> Response:
    media_types = {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }
    return Response(content=audio_bytes, media_type=media_types[audio_format])


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    _log(
        "request_started",
        client_ip=_client_ip(request),
        method=request.method,
        path=request.url.path,
        request_id=request_id,
    )

    try:
        response = await call_next(request)
    except Exception:
        _log_exception(
            "request_failed_uncaught",
            client_ip=_client_ip(request),
            method=request.method,
            path=request.url.path,
            request_id=request_id,
        )
        raise

    response.headers[REQUEST_ID_HEADER] = request_id
    _log(
        "request_completed",
        client_ip=_client_ip(request),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        method=request.method,
        path=request.url.path,
        request_id=request_id,
        status_code=response.status_code,
    )
    return response


@app.get("/")
async def root():
    return {
        "service": "chatterbox-tts",
        "default_model": _normalize_model_name(MODEL_NAME),
        "loaded_models": sorted(state.model_families.keys()),
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "docs": docs_url,
        "endpoints": ["/health", "/v1/models", "/v1/audio/speech", "/tts"],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_model": _normalize_model_name(MODEL_NAME),
        "loaded_models": state.model_families,
        "configured_device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "resolved_device": None if not state.models else _resolve_device(),
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "turbo", "object": "model", "family": "english"},
            {"id": "english", "object": "model", "family": "english"},
            {"id": "multilingual", "object": "model", "family": "multilingual"},
        ],
        "default": MODEL_NAME,
    }


@app.post("/tts")
async def synthesize(request: Request, payload: TTSRequest):
    request_id = _request_id(request)
    reference_voice_path = _resolve_reference_voice(request.reference_voice)
    audio_bytes, _sample_rate = _render_audio(
        text=payload.text,
        language=payload.language,
        audio_format=payload.audio_format,
        reference_voice_path=reference_voice_path,
        request_id=request_id,
        endpoint="/tts",
        model_name=MODEL_NAME,
    )
    return _render_response(audio_bytes, payload.audio_format)


@app.post("/v1/audio/speech")
async def openai_compatible_speech(request: Request, payload: OpenAISpeechRequest):
    request_id = _request_id(request)
    requested_model = _normalize_model_name(payload.model or MODEL_NAME)

    reference_voice_path = _resolve_reference_voice(payload.voice)
    audio_bytes, _sample_rate = _render_audio(
        text=payload.input,
        language=payload.language,
        audio_format=payload.response_format,
        reference_voice_path=reference_voice_path,
        request_id=request_id,
        endpoint="/v1/audio/speech",
        model_name=requested_model,
    )
    return _render_response(audio_bytes, payload.response_format)


@app.post("/v1/audio/speech/upload")
async def speech_with_upload(
    request: Request,
    input: str = Form(...),
    response_format: Literal["wav", "flac", "ogg"] = Form(DEFAULT_AUDIO_FORMAT),  # type: ignore[arg-type]
    language: str | None = Form(default=None),
    reference_audio: UploadFile | None = File(default=None),
):
    temp_path = None
    request_id = _request_id(request)
    try:
        if reference_audio is not None:
            suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(await reference_audio.read())
                temp_path = handle.name

        audio_bytes, _sample_rate = _render_audio(
            text=input,
            language=language,
            audio_format=response_format,
            reference_voice_path=temp_path,
            request_id=request_id,
            endpoint="/v1/audio/speech/upload",
            model_name=MODEL_NAME,
        )
        return _render_response(audio_bytes, response_format)
    finally:
        if temp_path:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    request_id = _request_id(request)
    _log_exception(
        "runtime_error",
        client_ip=_client_ip(request),
        path=request.url.path,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        headers={REQUEST_ID_HEADER: request_id},
        content={"detail": str(exc), "request_id": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = _request_id(request)
    log_fn = _log_warning if exc.status_code < 500 else _log_exception
    log_fn(
        "http_exception",
        client_ip=_client_ip(request),
        detail=exc.detail,
        path=request.url.path,
        request_id=request_id,
        status_code=exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        headers={REQUEST_ID_HEADER: request_id},
        content={"detail": exc.detail, "request_id": request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = _request_id(request)
    _log_warning(
        "request_validation_failed",
        client_ip=_client_ip(request),
        errors=exc.errors(),
        path=request.url.path,
        request_id=request_id,
        status_code=422,
    )
    return JSONResponse(
        status_code=422,
        headers={REQUEST_ID_HEADER: request_id},
        content={"detail": exc.errors(), "request_id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = _request_id(request)
    _log_exception(
        "unhandled_exception",
        client_ip=_client_ip(request),
        path=request.url.path,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        headers={REQUEST_ID_HEADER: request_id},
        content={"detail": "Internal server error.", "request_id": request_id},
    )
