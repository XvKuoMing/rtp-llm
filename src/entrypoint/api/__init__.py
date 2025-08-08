from .server_router import router as server_router
from .config_router import router as config_router
from .audio_router import router as audio_router

# dependencies are imported for side effects when needed
from .dependencies import get_server_manager  # noqa: F401

__all__ = ["server_router", "config_router", "audio_router"]
