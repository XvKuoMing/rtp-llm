import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.providers import OpenAIProvider, AstLLmProvider, TextOpenAIMessage, Message
from dotenv import load_dotenv
import logging

load_dotenv(".env.test")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@pytest.fixture
def baudio():
    with open("tests/test_audio/greetings.wav", "rb") as f:
        return f.read()

@pytest.fixture
def openai_provider():
    return OpenAIProvider(
        api_key=os.getenv("OPENAI_AUDIO_LLM_API_KEY"),
        base_url=os.getenv("OPENAI_AUDIO_LLM_BASE_URL"),
    )


@pytest.fixture
def ast_llm_provider():
    return AstLLmProvider(
        ast_model=os.getenv("OPENAI_WHISPER_MODEL"),
        language="en",
        stt_model=os.getenv("OPENAI_LLM_MODEL"),
        tts_model=os.getenv("OPENAI_TTS_MODEL"),
        overwrite_ast_model_api_key=os.getenv("OPENAI_WHISPER_API_KEY"),
        overwrite_ast_model_base_url=os.getenv("OPENAI_WHISPER_BASE_URL"),
        overwrite_stt_model_api_key=os.getenv("OPENAI_LLM_API_KEY"),
        overwrite_stt_model_base_url=os.getenv("OPENAI_LLM_BASE_URL"),
        overwrite_tts_model_api_key=os.getenv("OPENAI_AUDIO_LLM_API_KEY"),
        overwrite_tts_model_base_url=os.getenv("OPENAI_AUDIO_LLM_BASE_URL"),

    )



@pytest.mark.asyncio
async def test_format(openai_provider: OpenAIProvider, baudio: bytes):
    formatted_data = await openai_provider.format(Message(role="user", content=baudio, data_type="audio"))
    assert formatted_data is not None
    assert formatted_data.data_type == "audio"
    assert formatted_data.role == "user"
    # Check that the base64 data exists and is valid base64 (should be decodable)
    base64_data = formatted_data.as_json()["content"][0]["input_audio"]["data"]
    assert len(base64_data) > 0
    # Verify it's valid base64 by attempting to decode it
    import base64
    decoded = base64.b64decode(base64_data)
    assert decoded == baudio

    formatted_data = await openai_provider.format(Message(role="user", content="Hello, how are you?", data_type="text"))
    assert formatted_data is not None
    assert formatted_data.data_type == "text"
    assert formatted_data.role == "user"
    assert formatted_data.content == "Hello, how are you?"


@pytest.mark.asyncio
async def test_openai_provider(openai_provider: OpenAIProvider, baudio: bytes):
    if openai_provider.stt_api_key is None or openai_provider.tts_api_key is None:
        pytest.skip("OPENAI_API_KEY is not set")
    formatted_data = await openai_provider.format(Message(role="user", content=baudio, data_type="audio"))
    response = await openai_provider.stt([formatted_data])
    audio = await openai_provider.tts(response, response_format="wav")
    with open("tests/test_audio/greetings_tts_openai.wav", "wb") as f:
        f.write(audio)
    assert audio is not None


@pytest.mark.asyncio
async def test_ast_of_provider(ast_llm_provider: AstLLmProvider, baudio: bytes):
    if ast_llm_provider.ast_api_key is None:
        pytest.skip("AST_MODEL_API_KEY is not set")
    formatted_data = await ast_llm_provider.format(Message(role="user", content=baudio, data_type="audio"))
    assert isinstance(formatted_data, TextOpenAIMessage)
    assert formatted_data.content is not None
    assert isinstance(formatted_data.content, str)


@pytest.mark.asyncio
async def test_ast_llm_provider(ast_llm_provider: AstLLmProvider, baudio: bytes):
    if ast_llm_provider.ast_api_key is None\
          or ast_llm_provider.stt_api_key is None\
              or ast_llm_provider.tts_api_key is None:
        pytest.skip("AST_MODEL_API_KEY or STT_MODEL_API_KEY or TTS_MODEL_API_KEY is not set")
    formatted_data = await ast_llm_provider.format(Message(role="user", content=baudio, data_type="audio"))
    response = await ast_llm_provider.stt([formatted_data])
    audio = await ast_llm_provider.tts(response, response_format="wav")
    with open("tests/test_audio/greetings_tts_ast_llm.wav", "wb") as f:
        f.write(audio)
    assert audio is not None



