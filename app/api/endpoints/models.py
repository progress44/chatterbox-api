"""
Model listing endpoints (OpenAI compatibility)
"""

from time import time

from fastapi import APIRouter

from app.models import ModelsResponse, ModelInfo
from app.core import add_route_aliases
from app.core.tts_model import get_model_info

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List models",
    description="List available models (OpenAI API compatibility)"
)
async def list_models():
    """List currently loaded models (OpenAI API compatibility)."""
    runtime_model_info = get_model_info()

    if not runtime_model_info["is_ready"]:
        return ModelsResponse(object="list", data=[])

    model_type = runtime_model_info["model_type"]
    model_id = (
        "chatterbox-tts-multilingual-1"
        if runtime_model_info["is_multilingual"]
        else "chatterbox-tts-standard-1"
    )

    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id=model_id,
                object="model",
                created=int(time()),
                owned_by="resemble-ai",
                ready=True,
                device=runtime_model_info["device"],
                model_type=model_type,
                language_count=runtime_model_info["language_count"],
            )
        ],
    )

# Export the base router for the main app to use
__all__ = ["base_router"] 
