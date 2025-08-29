import asyncio
import json
from pathlib import Path
from typing import List, Tuple

import pytest
import pytest_asyncio

from rtp_llm.providers import MetaProvider
from rtp_llm.agents import VoiceAgent
from rtp_llm.history import ChatHistoryLimiter


# REQUIRED: providers.json
# uv run pytest tests/test_concurrency.py -vv -o log_cli=true --log-cli-level=DEBUG


def _load_providers_from_parent() -> Tuple[List[object], List[object]]:
    """
    Load STT/TTS providers using MetaProvider from providers.json located in
    the project root (parent of tests/) or fall back to examples/providers.json.
    Mirrors the loading approach used in entrypoint.server_manager.
    """
    project_root = Path(__file__).resolve().parents[1]
    primary_cfg = project_root / "providers.json"

    config = None
    if primary_cfg.exists():
        with primary_cfg.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(
            "providers.json not found in project root or examples/providers.json"
        )

    stt_instances: List[object] = []
    tts_instances: List[object] = []

    if config and "stt_providers" in config:
        for provider_cfg in config["stt_providers"]:
            name = provider_cfg.get("name")
            if not name:
                raise ValueError("STT provider entry missing 'name'")
            params = {k: v for k, v in provider_cfg.items() if k != "name"}
            stt_instances.append(MetaProvider.create_provider_from_config(name, params))

    if config and "tts_providers" in config:
        for provider_cfg in config["tts_providers"]:
            name = provider_cfg.get("name")
            if not name:
                raise ValueError("TTS provider entry missing 'name'")
            params = {k: v for k, v in provider_cfg.items() if k != "name"}
            tts_instances.append(MetaProvider.create_provider_from_config(name, params))

    if not stt_instances or not tts_instances:
        raise RuntimeError("No STT/TTS providers configured")

    return stt_instances, tts_instances


@pytest_asyncio.fixture
async def agents_factory():
    """
    Async fixture returning a factory to create N identical VoiceAgent instances.

    Usage:
        agents = agents_factory(5, chat_limit=10)
    """
    stt_instances, tts_instances = _load_providers_from_parent()

    def _factory(n: int, chat_limit: int = 10) -> List[VoiceAgent]:
        agents: List[VoiceAgent] = []
        for _ in range(n):
            agent = VoiceAgent(
                stt_provider=stt_instances[0],
                tts_provider=tts_instances[0],
                history_manager=ChatHistoryLimiter(limit=chat_limit),
                backup_stt_providers=stt_instances[1:],
                backup_tts_providers=tts_instances[1:],
            )
            agents.append(agent)
        return agents

    return _factory

async def stream_coro_factory():
    # dummy coroutine to yield stream to
    while True:
        _ = yield

@pytest.mark.asyncio
async def test_stream_to_agents(agents_factory):
    agents: List[VoiceAgent] = agents_factory(15)
    coro = stream_coro_factory
    tasks = [agent.tts_stream_to("Hello, how are you?", coro) for agent in agents]
    await asyncio.gather(*tasks)