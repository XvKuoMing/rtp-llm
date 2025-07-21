"""
In-memory audio cache implementation with LRU eviction.
"""
import asyncio
from typing import Dict, Optional
import logging

from .base import BaseAudioCache

logger = logging.getLogger(__name__)


class InMemoryAudioCache(BaseAudioCache):
    """
    In-memory audio cache with LRU (Least Recently Used) eviction policy.
    """
    
    def __init__(self):
        self._cache: Dict[str, bytes] = {}
        self._lock = asyncio.Lock()
    
    def make_key(self, text: str, tts_footprint: str) -> str:
        return f"{text}_{tts_footprint}"
    
    async def get(self, key: str) -> Optional[bytes]:
        async with self._lock:
            if key in self._cache:
                return self._cache[key]
            return None
    
    async def set(self, key: str, audio_data: bytes) -> None:
        async with self._lock:
            self._cache[key] = audio_data
    
    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()