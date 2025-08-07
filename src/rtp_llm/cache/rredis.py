from .base import BaseAudioCache
from redis.asyncio import Redis
from typing import Optional, List
import logging
import json
import base64

logger = logging.getLogger(__name__)

class RedisAudioCache(BaseAudioCache):
    def __init__(self, redis_client: Redis, ttl_seconds: Optional[int] = None):
        self.client = redis_client
        self.ttl_seconds = ttl_seconds

    def make_key(self, text: str, tts_footprint: str) -> str:
        return f"tts_cache:{text}_{tts_footprint}"
    
    async def get(self, key: str) -> Optional[List[bytes]]:
        try:
            result = await self.client.get(key)
            if result is None:
                logger.debug(f"Cache miss for key: {key}")
                return None
            
            # Deserialize JSON and decode base64 chunks
            data = json.loads(result)
            chunks = [base64.b64decode(chunk) for chunk in data]
            logger.debug(f"Cache hit for key: {key}, {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving from Redis cache: {e}")
            return None
    
    async def set(self, key: str, audio_chunks: List[bytes]) -> None:
        try:
            # Serialize chunks as base64-encoded JSON
            data = [base64.b64encode(chunk).decode('utf-8') for chunk in audio_chunks]
            json_data = json.dumps(data)
            
            if self.ttl_seconds is not None:
                # Use setex with TTL
                await self.client.setex(key, self.ttl_seconds, json_data)
                logger.debug(f"Cached {len(audio_chunks)} audio chunks for key: {key} with TTL: {self.ttl_seconds}s")
            else:
                # Use regular set (no expiration)
                await self.client.set(key, json_data)
                logger.debug(f"Cached {len(audio_chunks)} audio chunks for key: {key} (no expiration)")
        except Exception as e:
            logger.error(f"Error storing to Redis cache: {e}")
    
    async def clear(self) -> None:
        try:
            # Only clear TTS cache keys, not all Redis data
            async for key in self.client.scan_iter(match="tts_cache:*"):
                await self.client.delete(key)
            logger.info("Cleared TTS cache")
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    async def close(self) -> None:
        """Close Redis client connections properly"""
        try:
            await self.client.close()
            logger.info("Redis client connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
