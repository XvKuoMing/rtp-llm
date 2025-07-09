import asyncio
import fastapi
from fastapi import status
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import traceback
import argparse
import signal
import atexit
from datetime import datetime
from dataclasses import dataclass

from rtp_llm.server import Server
from rtp_llm.adapters import RTPAdapter
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.vad import WebRTCVAD, SileroVAD
from rtp_llm.providers import OpenAIProvider, AstLLmProvider, GeminiSTTProvider
from rtp_llm.agents import VoiceAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)



@dataclass
class Config:
    # Server configuration
    host: str
    port: int
    max_wait_time: int
    chat_limit: int
    vad: str
    system_prompt: str
    
    # Provider selection
    stt_providers: str
    tts_providers: str
    
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

# Global configuration instance
config: Optional[Config] = None


# Global provider variables
stt_provider = None
tts_provider = None
stt_backup_provider = []
tts_backup_provider = []
voice_agent = None

def parse_provider_list(provider_string: str) -> List[str]:
    """Parse provider string like 'gemini;openai' into list ['gemini', 'openai']"""
    if not provider_string:
        return []
    return [p.strip() for p in provider_string.split(';') if p.strip()]

def initialize_agent(system_prompt: Optional[str] = None, chat_limit: int = 10):
    """Initialize STT and TTS providers based on configuration and availability"""
    global stt_provider, tts_provider, stt_backup_provider, tts_backup_provider, voice_agent
    
    # Reset provider variables
    stt_provider = None
    tts_provider = None
    stt_backup_provider = []
    tts_backup_provider = []
    
    # Parse STT providers
    stt_providers_list = parse_provider_list(config.stt_providers)
    if not stt_providers_list:
        raise ValueError("No STT providers specified")
    
    # Parse TTS providers  
    tts_providers_list = parse_provider_list(config.tts_providers)
    if not tts_providers_list:
        raise ValueError("No TTS providers specified")
    
    # Initialize STT providers
    for i, provider in enumerate(stt_providers_list):
        provider_instance = None
        
        if provider == "gemini":
            if all([config.gemini_stt_api_key, config.gemini_stt_base_url, config.gemini_stt_model]):
                provider_instance = GeminiSTTProvider(
                    api_key=config.gemini_stt_api_key,
                    base_url=config.gemini_stt_base_url,
                    model=config.gemini_stt_model,
                    system_prompt=system_prompt or config.system_prompt
                )
        elif provider == "openai":
            if all([config.openai_stt_api_key, config.openai_stt_base_url, config.openai_stt_model]):
                provider_instance = OpenAIProvider(
                    overwrite_stt_model_api_key=config.openai_stt_api_key,
                    overwrite_stt_model_base_url=config.openai_stt_base_url,
                    stt_model=config.openai_stt_model,
                    system_prompt=system_prompt or config.system_prompt
                )
        elif provider == "ast_llm":
            if all([config.ast_api_key, config.ast_base_url, config.ast_model]):
                provider_instance = AstLLmProvider(
                    ast_model=config.ast_model,
                    overwrite_ast_model_api_key=config.ast_api_key,
                    overwrite_ast_model_base_url=config.ast_base_url,
                    language=config.ast_language,
                    stt_model=config.llm_model,
                    overwrite_stt_api_key=config.llm_api_key,
                    overwrite_stt_base_url=config.llm_base_url,
                    system_prompt=system_prompt or config.system_prompt
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
            if all([config.openai_tts_api_key, config.openai_tts_base_url, config.openai_tts_model]):
                provider_instance = OpenAIProvider(
                    overwrite_tts_model_api_key=config.openai_tts_api_key,
                    overwrite_tts_model_base_url=config.openai_tts_base_url,
                    tts_model=config.openai_tts_model,
                    pcm_response_format=config.openai_tts_pcm_response_format,
                    response_sample_rate=config.openai_tts_response_sample_rate,
                )
        elif provider == "ast_llm":
            if all([config.tts_api_key, config.tts_base_url, config.tts_model]):
                provider_instance = OpenAIProvider(
                    overwrite_tts_model_api_key=config.tts_api_key,
                    overwrite_tts_model_base_url=config.tts_base_url,
                    tts_model=config.tts_model,
                    pcm_response_format=config.tts_pcm_response_format,
                    response_sample_rate=config.tts_response_sample_rate,
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
            history_manager=ChatHistoryLimiter(limit=chat_limit),
            backup_stt_providers=stt_backup_provider,
            backup_tts_providers=tts_backup_provider
        )
        logger.info("Voice agent initialized successfully")
        return voice_agent
    except Exception as e:
        logger.error(f"Failed to initialize voice agent: {e}")
        raise

class SingletonServer(Server):
    _instance: Optional["SingletonServer"] = None
    _task: Optional[asyncio.Task] = None
    __static_host_ip: Optional[str] = None
    __static_host_port: Optional[int] = None

    @staticmethod
    def is_running():
        return SingletonServer._task is not None and not SingletonServer._task.done()
    
    @staticmethod
    def set_host_ip(host_ip: str, host_port: int):
        SingletonServer.__static_host_ip = host_ip
        SingletonServer.__static_host_port = host_port
    
    @staticmethod
    def close():
        if SingletonServer._instance:
            SingletonServer._instance.close()
            SingletonServer._instance = None
            SingletonServer._task = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        return cls._instance.post_init(*args, **kwargs)

    def __init__(self, 
                 peer_ip: Optional[str] = None, 
                 peer_port: Optional[int] = None, 
                 target_sample_rate: Optional[int] = None, 
                 target_codec: Optional[str] = None,
                 ):
        vad_type = config.vad.lower()
        if vad_type == "webrtc":
            vad = WebRTCVAD(target_sample_rate, min_speech_duration_ms=60)
        elif vad_type == "silero":
            vad = SileroVAD(target_sample_rate, min_speech_duration_ms=60)
        else:
            raise ValueError(f"Unsupported VAD type: {vad_type}")

        super().__init__(
            adapter=RTPAdapter(
                host_ip=self.__class__.__static_host_ip,
                host_port=self.__class__.__static_host_port,
                peer_ip=peer_ip,
                peer_port=peer_port,
                sample_rate=target_sample_rate,
                target_codec=target_codec
            ),
            audio_buffer=ArrayBuffer(),
            flow_manager=CopyFlowManager(),
            vad=vad,
            agent=voice_agent,
            max_wait_time=config.max_wait_time,
        )
    
    def post_init(self, 
                 peer_ip: Optional[str] = None, 
                 peer_port: Optional[int] = None, 
                 target_sample_rate: Optional[int] = None, 
                 target_codec: Optional[str] = None,
                 ):
        
        self.adapter = RTPAdapter(
            host_ip=self.__class__.__static_host_ip,
            host_port=self.__class__.__static_host_port,
            peer_ip=peer_ip,
            peer_port=peer_port,
            sample_rate=target_sample_rate,
            target_codec=target_codec
        )
                
        return self

    async def run(self, 
                  first_message: Optional[str] = None, 
                  uid: Optional[int | str] = None, 
                  allow_interruptions: bool = False,
                  
                  system_prompt: Optional[str] = None,
                  tts_gen_config: Optional[Dict[str, Any]] = None,
                  stt_gen_config: Optional[Dict[str, Any]] = None,
                  tts_volume: Optional[float] = 1.0,
                  ):
        try:
            self._task = asyncio.create_task(super().run(
                first_message=first_message,
                uid=uid,
                allow_interruptions=allow_interruptions,

                system_prompt=system_prompt,
                tts_gen_config=tts_gen_config,
                stt_gen_config=stt_gen_config,
                volume=tts_volume,
            ))
        except Exception as e:
            logger.error(f"Error running server: {e}")
            raise
    
    def close(self):
        if self._task:
            self._task.cancel()
            self._task = None
        self.close()
    

app = fastapi.FastAPI(
    title="RTP LLM Singleton API",
    description="API for managing a singleton RTP server instance",
    version="0.1.0"
)

class APIResponse(BaseModel):
    message: str
    status: str
    timestamp: datetime
    data: dict = {}

class StartRTPRequest(BaseModel):
    peer_ip: Optional[str] = None
    peer_port: Optional[int] = None
    target_codec: Optional[str] = None
    target_sample_rate: Optional[int] = None
    tts_volume: Optional[float] = 1.0
    first_message: Optional[str] = None
    allow_interruptions: bool = False
    system_prompt: Optional[str] = None
    uid: Optional[int | str] = None
    tts_gen_config: Optional[Dict[str, Any]] = None
    stt_gen_config: Optional[Dict[str, Any]] = None


class StopRTPRequest(BaseModel):
    force: bool = False

@app.get("/status", response_model=APIResponse)
async def ping():
    return APIResponse(
        message="alive",
        status="success",
        timestamp=datetime.now().isoformat(),
        data={"is_running": SingletonServer.is_running()}
    )

@app.post("/start", response_model=APIResponse)
async def start(request: StartRTPRequest):
    
    try:

        if SingletonServer.is_running():
            return APIResponse(
                message="RTP server is already running",
                status="warning",
                timestamp=datetime.now().isoformat(),
                data={"is_running": True}
            )

        server = SingletonServer(
            peer_ip=request.peer_ip,
            peer_port=request.peer_port,
            target_codec=request.target_codec,
            target_sample_rate=request.target_sample_rate,
        )

        await server.run(
            first_message=request.first_message,
            uid=request.uid,
            allow_interruptions=request.allow_interruptions,

            system_prompt=request.system_prompt,
            tts_gen_config=request.tts_gen_config,
            stt_gen_config=request.stt_gen_config,
            tts_volume=request.tts_volume,
        )
        
        return APIResponse(
            message="RTP server started successfully",
            status="success",
            timestamp=datetime.now().isoformat(),
            data={"peer_ip": request.peer_ip, "peer_port": request.peer_port}
        )
    except Exception as e: 
        logger.error(f"Error starting RTP server: {e}")
        return APIResponse(
            message="Failed to start RTP server",
            status="error",
            timestamp=datetime.now().isoformat(),
            data={"error": str(e)}
        )
    
@app.post("/stop", response_model=APIResponse)
async def stop(request: StopRTPRequest):
    try:
        if SingletonServer.is_running():
            if request.force:
                SingletonServer.close()
                return APIResponse(
                    message="RTP server stopped forcefully",
                    status="success",
                    timestamp=datetime.now().isoformat(),
                    data={"is_running": False}
                )
            else:
                return APIResponse(
                    message="RTP server is currently running",
                    status="warning",
                    timestamp=datetime.now().isoformat(),
                    data={"is_running": True}
                )

        SingletonServer.close()
        return APIResponse(
            message="RTP server stopped successfully",
            status="success",
            timestamp=datetime.now().isoformat(),
            data={"is_running": False}
        )
    except Exception as e:
        logger.error(f"Error stopping RTP server: {e}")
        return APIResponse(
            message="Failed to stop RTP server",
            status="error",
            timestamp=datetime.now().isoformat(),
            data={"error": str(e)}
        )



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}")
    logger.error(traceback.format_exc())
    
    return fastapi.responses.JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error",
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "data": {"error": str(exc)}
        }
    )




def cleanup_shutdown():
    """Cleanup function for graceful shutdown"""
    try:
        if SingletonServer.is_running():
            logger.info("Gracefully shutting down RTP server...")
            SingletonServer.close()
            logger.info("RTP server shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    cleanup_shutdown()
    exit(0)


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="RTP LLM Singleton API Server")
    
    # Server configuration
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
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
    
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function to run at exit
    atexit.register(cleanup_shutdown)

    # Set global config from command line arguments
    global config
    config = Config(
        host=args.host,
        port=args.port,
        max_wait_time=args.max_wait_time,
        chat_limit=args.chat_limit,
        vad=args.vad,
        system_prompt=args.system_prompt,
        stt_providers=args.stt_providers,
        tts_providers=args.tts_providers,
        gemini_stt_api_key=args.gemini_stt_api_key,
        gemini_stt_base_url=args.gemini_stt_base_url,
        gemini_stt_model=args.gemini_stt_model,
        openai_stt_api_key=args.openai_stt_api_key,
        openai_stt_base_url=args.openai_stt_base_url,
        openai_stt_model=args.openai_stt_model,
        openai_tts_api_key=args.openai_tts_api_key,
        openai_tts_base_url=args.openai_tts_base_url,
        openai_tts_model=args.openai_tts_model,
        openai_tts_pcm_response_format=args.openai_tts_pcm_response_format,
        openai_tts_response_sample_rate=args.openai_tts_response_sample_rate,
        ast_api_key=args.ast_api_key,
        ast_base_url=args.ast_base_url,
        ast_model=args.ast_model,
        ast_language=args.ast_language,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        tts_api_key=args.tts_api_key,
        tts_base_url=args.tts_base_url,
        tts_model=args.tts_model,
        tts_pcm_response_format=args.tts_pcm_response_format,
        tts_response_sample_rate=args.tts_response_sample_rate,
    )

    # Initialize agent with the configuration
    try:
        initialize_agent(system_prompt=config.system_prompt, chat_limit=config.chat_limit)
        SingletonServer.set_host_ip(args.host, args.port)
        
        logger.info(f"Starting API server on {args.host}:{args.port}")
        logger.info(f"STT providers: {config.stt_providers}")
        logger.info(f"TTS providers: {config.tts_providers}")
        
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        cleanup_shutdown()
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        cleanup_shutdown()

if __name__ == "__main__":
    main() 