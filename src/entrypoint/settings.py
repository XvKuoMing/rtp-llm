import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


def _default_providers_config_path() -> str:
    # Try XDG config directory or user home as fallback
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return os.path.join(xdg, "rtllm", "providers.json")
    home = os.path.expanduser("~")
    return os.path.join(home, ".config", "rtllm", "providers.json")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RTLLM_", case_sensitive=False)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    start_port: int = Field(default=10000)
    end_port: int = Field(default=20000)
    debug: bool = Field(default=False)
    providers_config_path: str = Field(default_factory=_default_providers_config_path)

    # Redis
    redis_enabled: bool = Field(default=False)
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    redis_ttl_seconds: Optional[int] = Field(default=None)

    # Concurrency for audio file parsing
    max_concurrent_files: int = Field(default=50)

    # CORS defaults
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = Field(default=True)


