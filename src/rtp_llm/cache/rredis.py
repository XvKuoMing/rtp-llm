from .base import BaseAudioCache
from redis.asyncio import Redis
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RedisAudioCache(BaseAudioCache):
    def __init__(self, redis_client: Redis, ttl_seconds: Optional[int] = None):
        self.client = redis_client
        self.ttl_seconds = ttl_seconds

    def make_key(self, text: str, tts_footprint: str) -> str:
        return f"tts_cache:{text}_{tts_footprint}"
    
    async def get(self, key: str) -> Optional[bytes]:
        try:
            result = await self.client.get(key)
            if result is None:
                logger.debug(f"Cache miss for key: {key}")
                return None
            logger.debug(f"Cache hit for key: {key}")
            return result
        except Exception as e:
            logger.error(f"Error retrieving from Redis cache: {e}")
            return None
    
    async def set(self, key: str, audio_data: bytes) -> None:
        try:
            if self.ttl_seconds is not None:
                # Use setex with TTL
                await self.client.setex(key, self.ttl_seconds, audio_data)
                logger.debug(f"Cached audio data for key: {key} with TTL: {self.ttl_seconds}s")
            else:
                # Use regular set (no expiration)
                await self.client.set(key, audio_data)
                logger.debug(f"Cached audio data for key: {key} (no expiration)")
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

async def create_redis_audio_cache(
    host: str = "localhost", 
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    **redis_kwargs
) -> RedisAudioCache:
    """
    Factory function to create a Redis audio cache with proper connection management.
    
    Args:
        host: Redis server host
        port: Redis server port  
        db: Redis database number
        password: Redis password (if required)
        ttl_seconds: Time-to-live for cached items in seconds. If None, keys are stored indefinitely.
        **redis_kwargs: Additional arguments passed to Redis client
    
    Returns:
        Configured RedisAudioCache instance
        
    Example:
        # Basic usage (keys stored indefinitely)
        cache = await create_redis_audio_cache()
        
        # With TTL (keys expire after 2 hours)
        cache = await create_redis_audio_cache(ttl_seconds=7200)
        
        # With custom settings
        cache = await create_redis_audio_cache(
            host="redis.example.com",
            port=6380,
            password="mypassword",
            ttl_seconds=None  # Store indefinitely
        )
    """
    try:
        redis_client = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # We need bytes, not strings
            **redis_kwargs
        )
        
        # Test the connection
        await redis_client.ping()
        logger.info(f"Successfully connected to Redis at {host}:{port}")
        
        return RedisAudioCache(redis_client, ttl_seconds)
        
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {host}:{port}: {e}")
        raise