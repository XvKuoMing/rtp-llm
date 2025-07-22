from dataclasses import dataclass
from typing import Optional, List
import os
import argparse

from rtp_llm.providers import GeminiSTTProvider, OpenAIProvider, AstLLmProvider
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.agents import VoiceAgent
from rtp_llm.vad import BaseVAD, WebRTCVAD, SileroVAD

from rtp_llm.server import Server
from rtp_llm.adapters import RTPAdapter
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.cache import InMemoryAudioCache, RedisAudioCache, create_redis_audio_cache, BaseAudioCache


import logging

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    # Common Server configuration
    max_wait_time: int = 5
    chat_limit: int = 10
    vad: str = "webrtc"
    system_prompt: str = "You are a helpful assistant."
    
    # Provider selection
    stt_providers: str = "openai"
    tts_providers: str = "openai"
    
    # Gemini STT configuration
    gemini_stt_api_key: Optional[str] = None
    gemini_stt_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_stt_model: str = "gemini-2.0-flash"
    
    # OpenAI STT configuration
    openai_stt_api_key: Optional[str] = None
    openai_stt_base_url: str = "https://api.openai.com/v1"
    openai_stt_model: str = "gpt-4o-mini-audio-preview"
    
    # OpenAI TTS configuration
    openai_tts_api_key: Optional[str] = None
    openai_tts_base_url: str = "https://api.openai.com/v1"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_pcm_response_format: str = "pcm"
    openai_tts_response_sample_rate: int = 24000
    openai_tts_voice: str = "alloy"
    
    # AST LLM STT configuration
    ast_api_key: Optional[str] = None
    ast_base_url: str = "https://api.openai.com/v1"
    ast_model: str = "openai/whisper-large-v3-turbo"
    ast_language: str = "en"
    llm_model: str = "gpt-4o-mini-audio-preview"
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.openai.com/v1"
    
    # AST LLM TTS configuration
    tts_api_key: Optional[str] = None
    tts_base_url: str = "https://api.openai.com/v1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_pcm_response_format: str = "pcm"
    tts_response_sample_rate: int = 24000
    tts_voice: str = "alloy"


    # redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ttl_seconds: Optional[int] = None
    
    def initialize_redis_audio_cache(self, **redis_kwargs) -> RedisAudioCache:
        return create_redis_audio_cache(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            ttl_seconds=self.redis_ttl_seconds,
            **redis_kwargs
        )

    def initialize_vad(self, min_speech_duration_ms: int = 60, **kwargs) -> BaseVAD:
        if self.vad == "webrtc":
            return WebRTCVAD(min_speech_duration_ms=min_speech_duration_ms, **kwargs)
        elif self.vad == "silero":
            return SileroVAD(min_speech_duration_ms=min_speech_duration_ms, **kwargs)
        else:
            raise ValueError(f"Invalid VAD type: {self.vad}")


    def initialize_agent(self) -> VoiceAgent:
        def parse_provider_list(provider_string: str) -> List[str]:
            """Parse provider string like 'gemini;openai' or 'gemini,openai' into list ['gemini', 'openai']"""
            if not provider_string:
                return []
            # Support both semicolon and comma separators
            if ';' in provider_string:
                separator = ';'
            else:
                separator = ','
            return [p.strip() for p in provider_string.split(separator) if p.strip()]
        
        stt_provider = None
        tts_provider = None
        stt_backup_provider = []
        tts_backup_provider = []
        
        # Parse STT providers
        stt_providers_list = parse_provider_list(self.stt_providers)
        if not stt_providers_list:
            raise ValueError("No STT providers specified")
        
        # Parse TTS providers  
        tts_providers_list = parse_provider_list(self.tts_providers)
        if not tts_providers_list:
            raise ValueError("No TTS providers specified")
        
        # Initialize STT providers
        for i, provider in enumerate(stt_providers_list):
            provider_instance = None
            
            if provider == "gemini":
                if all([self.gemini_stt_api_key, self.gemini_stt_base_url, self.gemini_stt_model]):
                    provider_instance = GeminiSTTProvider(
                        api_key=self.gemini_stt_api_key,
                        base_url=self.gemini_stt_base_url,
                        model=self.gemini_stt_model,
                        system_prompt=self.system_prompt
                    )
            elif provider == "openai":
                if all([self.openai_stt_api_key, self.openai_stt_base_url, self.openai_stt_model]):
                    provider_instance = OpenAIProvider(
                        overwrite_stt_model_api_key=self.openai_stt_api_key,
                        overwrite_stt_model_base_url=self.openai_stt_base_url,
                        stt_model=self.openai_stt_model,
                        system_prompt=self.system_prompt
                    )
            elif provider == "ast_llm":
                if all([self.ast_api_key, self.ast_base_url, self.ast_model]):
                    provider_instance = AstLLmProvider(
                        ast_model=self.ast_model,
                        overwrite_ast_model_api_key=self.ast_api_key,
                        overwrite_ast_model_base_url=self.ast_base_url,
                        language=self.ast_language,
                        stt_model=self.llm_model,
                        overwrite_stt_api_key=self.llm_api_key,
                        overwrite_stt_base_url=self.llm_base_url,
                        system_prompt=self.system_prompt
                    )
            
            if provider_instance:
                if i == 0:  # First provider is primary
                    stt_provider = provider_instance
                else:  # Rest are backups
                    stt_backup_provider.append(provider_instance)
            else:
                logger.warning(f"Failed to initialize STT provider: {provider} (missing configuration)")
        
        # Initialize TTS providers
        for i, provider in enumerate(tts_providers_list):
            provider_instance = None
            
            if provider == "openai":
                if all([self.openai_tts_api_key, self.openai_tts_base_url, self.openai_tts_model]):
                    provider_instance = OpenAIProvider(
                        overwrite_tts_model_api_key=self.openai_tts_api_key,
                        overwrite_tts_model_base_url=self.openai_tts_base_url,
                        tts_model=self.openai_tts_model,
                        pcm_response_format=self.openai_tts_pcm_response_format,
                        response_sample_rate=self.openai_tts_response_sample_rate,
                        tts_voice=self.openai_tts_voice,
                    )
            elif provider == "ast_llm":
                if all([self.tts_api_key, self.tts_base_url, self.tts_model]):
                    provider_instance = OpenAIProvider(
                        overwrite_tts_model_api_key=self.tts_api_key,
                        overwrite_tts_model_base_url=self.tts_base_url,
                        tts_model=self.tts_model,
                        pcm_response_format=self.tts_pcm_response_format,
                        response_sample_rate=self.tts_response_sample_rate,
                        tts_voice=self.tts_voice,
                    )
            
            if provider_instance:
                if i == 0:  # First provider is primary
                    tts_provider = provider_instance
                else:  # Rest are backups
                    tts_backup_provider.append(provider_instance)
            else:
                logger.warning(f"Failed to initialize TTS provider: {provider} (missing configuration)")
        
        if stt_provider is None:
            raise ValueError("No STT provider could be initialized")
        logger.info(f"STT provider initialized: {type(stt_provider).__name__}")
        
        if tts_provider is None:
            raise ValueError("No TTS provider could be initialized")
        logger.info(f"TTS provider initialized: {type(tts_provider).__name__}")
        
        if stt_backup_provider:
            logger.info(f"STT backup providers initialized: {[type(provider).__name__ for provider in stt_backup_provider]}")
        else:
            logger.warning("No STT backup provider found")
        
        if tts_backup_provider:
            logger.info(f"TTS backup providers initialized: {[type(provider).__name__ for provider in tts_backup_provider]}")
        else:
            logger.warning("No TTS backup provider found")

        # Initialize voice agent
        try:
            voice_agent = VoiceAgent(
                stt_provider=stt_provider,
                tts_provider=tts_provider,
                history_manager=ChatHistoryLimiter(limit=self.chat_limit),
                backup_stt_providers=stt_backup_provider,
                backup_tts_providers=tts_backup_provider
            )
            logger.info("Voice agent initialized successfully")
            return voice_agent
        except Exception as e:
            logger.error(f"Failed to initialize voice agent: {e}")
            raise
    

    def initialize_rtp_server(self, 
                          host_ip: str, 
                          host_port: int, 
                          peer_ip: Optional[str] = None, 
                          peer_port: Optional[int] = None,
                          sample_rate: Optional[int] = None,
                          codec: Optional[str] = None,
                          voice_agent: Optional[VoiceAgent] = None,
                          vad: Optional[BaseVAD] = None,
                          audio_cache: Optional[BaseAudioCache] = None,
                          ) -> Server:
        voice_agent = voice_agent or self.initialize_agent()
        vad = vad or self.initialize_vad()
        audio_cache = audio_cache or InMemoryAudioCache()
        
        server = Server(
            adapter=RTPAdapter(
                host_ip=host_ip,
                host_port=host_port,
                peer_ip=peer_ip,
                peer_port=peer_port,
                sample_rate=sample_rate,
                codec=codec,
            ),
            flow_manager=CopyFlowManager(),
            audio_buffer=ArrayBuffer(),
            audio_cache=audio_cache,
            voice_agent=voice_agent,
            vad=vad,
            max_wait_time=self.max_wait_time,
        )
        return server



def get_config_parser(description: str = "RTP LLM base Configuration"):
    parser = argparse.ArgumentParser(description=description)
    # Core configuration
    parser.add_argument("--system-prompt", default="You are a helpful assistant.", help="System prompt for the agent")
    parser.add_argument("--vad", choices=["webrtc", "silero"], default="webrtc", help="VAD type to use")
    parser.add_argument("--max-wait-time", type=int, default=5, help="Maximum wait time for response")
    parser.add_argument("--chat-limit", type=int, default=10, help="Chat history limit")
    
    # Provider configuration
    parser.add_argument("--stt-providers", required=True, help="STT providers in priority order (e.g., 'gemini;openai')")
    parser.add_argument("--tts-providers", required=True, help="TTS providers in priority order (e.g., 'openai')")
    
    # Gemini STT configuration
    parser.add_argument("--gemini-stt-api-key", help="Gemini STT API key")
    parser.add_argument("--gemini-stt-base-url", default="https://generativelanguage.googleapis.com/v1beta", help="Gemini STT base URL")
    parser.add_argument("--gemini-stt-model", default="gemini-2.0-flash", help="Gemini STT model")
    
    # OpenAI STT configuration
    parser.add_argument("--openai-stt-api-key", help="OpenAI STT API key")
    parser.add_argument("--openai-stt-base-url", default="https://api.openai.com/v1", help="OpenAI STT base URL")
    parser.add_argument("--openai-stt-model", default="gpt-4o-mini-audio-preview", help="OpenAI STT model")
    
    # OpenAI TTS configuration
    parser.add_argument("--openai-tts-api-key", help="OpenAI TTS API key")
    parser.add_argument("--openai-tts-base-url", default="https://api.openai.com/v1", help="OpenAI TTS base URL")
    parser.add_argument("--openai-tts-model", default="gpt-4o-mini-tts", help="OpenAI TTS model")
    parser.add_argument("--openai-tts-pcm-response-format", default="pcm", help="OpenAI TTS PCM response format")
    parser.add_argument("--openai-tts-response-sample-rate", type=int, default=24000, help="OpenAI TTS response sample rate")
    parser.add_argument("--openai-tts-voice", default="alloy", help="OpenAI TTS voice")
    
    # AST LLM STT configuration  
    parser.add_argument("--ast-api-key", help="AST API key for speech-to-text")
    parser.add_argument("--ast-base-url", default="https://api.openai.com/v1", help="AST base URL")
    parser.add_argument("--ast-model", default="openai/whisper-large-v3-turbo", help="AST model")
    parser.add_argument("--ast-language", default="en", help="AST language")
    parser.add_argument("--llm-model", default="gpt-4o-mini-audio-preview", help="LLM model for AST STT")
    parser.add_argument("--llm-api-key", help="LLM API key for AST STT")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1", help="LLM base URL for AST STT")
    
    # AST LLM TTS configuration
    parser.add_argument("--tts-api-key", help="TTS API key for AST LLM provider")
    parser.add_argument("--tts-base-url", default="https://api.openai.com/v1", help="TTS base URL for AST LLM provider")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts", help="TTS model for AST LLM provider")
    parser.add_argument("--tts-pcm-response-format", default="pcm", help="TTS PCM response format for AST LLM provider")
    parser.add_argument("--tts-response-sample-rate", type=int, default=24000, help="TTS response sample rate for AST LLM provider")
    parser.add_argument("--tts-voice", default="alloy", help="TTS voice for AST LLM provider")

    # Redis configuration
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument("--redis-password", help="Redis password")
    parser.add_argument("--redis-ttl-seconds", type=int, default=None, help="Redis TTL seconds")

    return parser

def parse_config_from_env(config: Optional[BaseConfig] = None):
    """
    Parse configuration from environment variables.
    If config is None, a new BaseConfig instance is created.
    otherwise, will update the config instance with the environment variables.
    """
    if config is None:
        config = BaseConfig()

    config.max_wait_time = int(os.getenv("MAX_WAIT_TIME", config.max_wait_time))
    config.chat_limit = int(os.getenv("CHAT_LIMIT", config.chat_limit))
    config.vad = os.getenv("VAD", config.vad)
    config.system_prompt = os.getenv("SYSTEM_PROMPT", config.system_prompt)

    config.stt_providers = os.getenv("STT_PROVIDERS", config.stt_providers)
    config.tts_providers = os.getenv("TTS_PROVIDERS", config.tts_providers)

    config.gemini_stt_api_key = os.getenv("GEMINI_STT_API_KEY", config.gemini_stt_api_key)
    config.gemini_stt_base_url = os.getenv("GEMINI_STT_BASE_URL", config.gemini_stt_base_url)
    config.gemini_stt_model = os.getenv("GEMINI_STT_MODEL", config.gemini_stt_model)

    config.openai_stt_api_key = os.getenv("OPENAI_STT_API_KEY", config.openai_stt_api_key)
    config.openai_stt_base_url = os.getenv("OPENAI_STT_BASE_URL", config.openai_stt_base_url)
    config.openai_stt_model = os.getenv("OPENAI_STT_MODEL", config.openai_stt_model)

    config.openai_tts_api_key = os.getenv("OPENAI_TTS_API_KEY", config.openai_tts_api_key)
    config.openai_tts_base_url = os.getenv("OPENAI_TTS_BASE_URL", config.openai_tts_base_url)
    config.openai_tts_model = os.getenv("OPENAI_TTS_MODEL", config.openai_tts_model)
    config.openai_tts_pcm_response_format = os.getenv("OPENAI_TTS_PCM_RESPONSE_FORMAT", config.openai_tts_pcm_response_format)
    config.openai_tts_response_sample_rate = int(os.getenv("OPENAI_TTS_RESPONSE_SAMPLE_RATE", config.openai_tts_response_sample_rate))
    config.openai_tts_voice = os.getenv("OPENAI_TTS_VOICE", config.openai_tts_voice)

    config.ast_api_key = os.getenv("AST_API_KEY", config.ast_api_key)
    config.ast_base_url = os.getenv("AST_BASE_URL", config.ast_base_url)
    config.ast_model = os.getenv("AST_MODEL", config.ast_model)
    config.ast_language = os.getenv("AST_LANGUAGE", config.ast_language)

    config.llm_api_key = os.getenv("LLM_API_KEY", config.llm_api_key)
    config.llm_base_url = os.getenv("LLM_BASE_URL", config.llm_base_url)
    config.llm_model = os.getenv("LLM_MODEL", config.llm_model)

    config.tts_api_key = os.getenv("TTS_API_KEY", config.tts_api_key)
    config.tts_base_url = os.getenv("TTS_BASE_URL", config.tts_base_url)
    config.tts_model = os.getenv("TTS_MODEL", config.tts_model)
    config.tts_pcm_response_format = os.getenv("TTS_PCM_RESPONSE_FORMAT", config.tts_pcm_response_format)
    config.tts_response_sample_rate = int(os.getenv("TTS_RESPONSE_SAMPLE_RATE", config.tts_response_sample_rate))
    config.tts_voice = os.getenv("TTS_VOICE", config.tts_voice)

    config.redis_host = os.getenv("REDIS_HOST", config.redis_host)
    config.redis_port = int(os.getenv("REDIS_PORT", config.redis_port))
    config.redis_db = int(os.getenv("REDIS_DB", config.redis_db))
    config.redis_password = os.getenv("REDIS_PASSWORD", config.redis_password)
    config.redis_ttl_seconds = int(os.getenv("REDIS_TTL_SECONDS", config.redis_ttl_seconds))

    return config
