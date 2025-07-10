from .base import BaseCallback


class NullCallback(BaseCallback):
    """
    A callback that does nothing.
    """

    async def on_stt(self, uid: str, text: str):
        pass

    async def on_tts(self, uid: str, text: str):
        pass

    async def on_start(self, uid: str):
        pass

    async def on_error(self, uid: str, error: Exception):
        pass
    
    async def on_finish(self, uid: str):
        pass