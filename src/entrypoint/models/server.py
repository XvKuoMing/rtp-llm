from pydantic import BaseModel, Field
from typing import Union, Literal, Dict, Any, Optional

from .config import UniAdapterConfig, UniVadConfig


class RestCallbackConfig(BaseModel):
    """Configuration for REST callback endpoints"""
    base_url: str
    on_response_endpoint: Optional[str] = None
    on_start_endpoint: Optional[str] = None
    on_error_endpoint: Optional[str] = None
    on_finish_endpoint: Optional[str] = None


class ServerConfig(BaseModel):
    uid: Union[str, int]
    sample_rate: Literal[8000, 16000, 24000, 48000]
    adapter: UniAdapterConfig
    vad: UniVadConfig
    max_wait_time: int = 10
    chat_limit: int = 10


class RunParams(BaseModel):
    uid: Union[str, int]
    first_message: Optional[str] = None
    allow_interruptions: bool = False
    system_prompt: Optional[str] = None
    tts_gen_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    stt_gen_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tts_volume: float = 1.0
    callback: Optional[RestCallbackConfig] = None


class StopServerRequest(BaseModel):
    uid: Union[str, int]


class UpdateAgentRequest(BaseModel):
    uid: Union[str, int]
    system_prompt: Optional[str] = None
    tts_gen_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    stt_gen_config: Optional[Dict[str, Any]] = Field(default_factory=dict)