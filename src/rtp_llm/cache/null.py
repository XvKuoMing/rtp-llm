from .base import BaseAudioCache
from typing import Optional


class NullAudioCache(BaseAudioCache):
    def make_key(self, text: str, tts_footprint: str) -> str:
        return f"{text}_{tts_footprint}"
    
    async def get(self, key: str) -> Optional[bytes]:
        return None
    
    async def set(self, key: str, audio_data: bytes) -> None:
        pass
    
    async def clear(self) -> None:
        pass