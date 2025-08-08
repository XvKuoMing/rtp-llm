from pydantic import BaseModel, Field
from typing import Optional, Union, Literal, Dict, Any


class HostServerConfig(BaseModel):
    """fastapi server configuration"""
    host: str
    port: int
    start_port: int
    end_port: int


class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    password: Optional[str] = None
    ttl_seconds: Optional[int] = None


class ReusableComponents(BaseModel):
    providers_config_path: str
    redis: Optional[RedisConfig] = None


class UniAdapterConfig(BaseModel):
    adapter_type: Literal["rtp", "websocket"]
    target_codec: Literal["pcm", "ulaw", "alaw", "opus"]
    peer_ip: Optional[str] = None  # will be ignored for websocket
    peer_port: Optional[int] = None  # will be ignored for websocket


class UniVadConfig(BaseModel):
    vad_type: Literal["webrtc", "silero"]
    min_speech_duration_ms: int
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
