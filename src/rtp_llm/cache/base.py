from abc import ABC, abstractmethod
from typing import Optional, List

class BaseAudioCache(ABC):
    """Base class for audio caching implementations."""

    @abstractmethod
    def make_key(self, text: str, tts_footprint: str) -> str:
        """
        Make a cache key from the given parameters.
        """
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[List[bytes]]:
        """
        Retrieve cached audio data for the given key.
        
        Args:
            key: The cache key (typically the text that was converted to speech)
            
        Returns:
            A list of audio chunks if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, audio_chunks: List[bytes]) -> None:
        """
        Store audio data in the cache.
        
        Args:
            key: The cache key (typically the text that was converted to speech)
            audio_chunks: List of PCM audio chunks to cache
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached data."""
        pass
