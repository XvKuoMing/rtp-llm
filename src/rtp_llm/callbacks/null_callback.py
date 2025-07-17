from .base import BaseCallback, ResponseTransformation
from typing import Optional


class NullCallback(BaseCallback):
    """
    A callback that does nothing.
    """

    async def on_response(self, uid: str, text: str) -> Optional[ResponseTransformation]:
        return ResponseTransformation()

    async def on_start(self, uid: str):
        pass

    async def on_error(self, uid: str, error: Exception):
        pass
    
    async def on_finish(self, uid: str):
        pass