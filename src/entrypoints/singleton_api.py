import os
import asyncio
import fastapi
from fastapi import status
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import traceback
import argparse
import signal
import atexit
from datetime import datetime

from rtp_llm.server import Server
from rtp_llm.adapters import RTPAdapter
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.vad import WebRTCVAD, SileroVAD
from rtp_llm.providers import OpenAIProvider, AstLLmProvider, GeminiSTTProvider
from rtp_llm.agents import VoiceAgent
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

logger.info(f"Seeking .env in current directory: {os.getcwd()}")
load_dotenv()


DEBUG = os.getenv("DEBUG", "False").lower() == "true"
if DEBUG:
    logger.setLevel(logging.DEBUG)

SYSTEM = """
You are a helpful assistant.
"""

VAD = os.getenv("VAD", "webrtc").lower()
MAX_WAIT_TIME = int(os.getenv("MAX_WAIT_TIME", 5))
CHAT_LIMIT = int(os.getenv("CHAT_LIMIT", 10))

DEFAULT_TTS_PCM_RESPONSE_FORMAT = "pcm"
DEFAULT_TTS_RESPONSE_SAMPLE_RATE = 24_000


providers = {
    "gemini": {
        "stt": {
            "api_key": os.getenv("GEMINI_STT_API_KEY"),
            "base_url": os.getenv("GEMINI_STT_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            "model": os.getenv("GEMINI_STT_MODEL", "gemini-2.0-flash"),
        },
    },
    "openai": {
        "stt": {
            "overwrite_stt_model_api_key": os.getenv("OPENAI_STT_API_KEY"),
            "overwrite_stt_model_base_url": os.getenv("OPENAI_STT_BASE_URL", "https://api.openai.com/v1"),
            "stt_model": os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-audio-preview")
        },
        "tts": {
            "overwrite_tts_model_api_key": os.getenv("OPENAI_TTS_API_KEY"),
            "overwrite_tts_model_base_url": os.getenv("OPENAI_TTS_BASE_URL", "https://api.openai.com/v1"),
            "tts_model": os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            "pcm_response_format": os.getenv("OPENAI_TTS_PCM_RESPONSE_FORMAT", DEFAULT_TTS_PCM_RESPONSE_FORMAT),
            "response_sample_rate": os.getenv("OPENAI_TTS_RESPONSE_SAMPLE_RATE", DEFAULT_TTS_RESPONSE_SAMPLE_RATE),
        }
    },
    "ast_llm": {
        "stt": {
            "ast_model": os.getenv("AST_MODEL", "openai/whisper-large-v3-turbo"),
            "overwrite_ast_model_api_key": os.getenv("AST_API_KEY"),
            "overwrite_ast_model_base_url": os.getenv("AST_BASE_URL", "https://api.openai.com/v1"),
            "language": os.getenv("AST_LANGUAGE", "en"),
            "stt_model": os.getenv("LLM_MODEL", "gpt-4o-mini-audio-preview"),
            "overwrite_stt_api_key": os.getenv("LLM_API_KEY"),
            "overwrite_stt_base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        },
        "tts": {
            "overwrite_tts_model_api_key": os.getenv("TTS_API_KEY"),
            "overwrite_tts_model_base_url": os.getenv("TTS_BASE_URL", "https://api.openai.com/v1"),
            "tts_model": os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
            "pcm_response_format": os.getenv("TTS_PCM_RESPONSE_FORMAT", DEFAULT_TTS_PCM_RESPONSE_FORMAT),
            "response_sample_rate": os.getenv("TTS_RESPONSE_SAMPLE_RATE", DEFAULT_TTS_RESPONSE_SAMPLE_RATE),
        }
    }
}

# Global provider variables
stt_provider = None
tts_provider = None
stt_backup_provider = []
tts_backup_provider = []
voice_agent = None

def initialize_agent(system_prompt: Optional[str] = None, chat_limit: int = 10):
    """Initialize STT and TTS providers based on configuration and availability"""
    global stt_provider, tts_provider, stt_backup_provider, tts_backup_provider, voice_agent
    
    # Update providers dict with current SYSTEM prompt
    providers["gemini"]["stt"]["system_prompt"] = system_prompt or SYSTEM
    providers["openai"]["stt"]["system_prompt"] = system_prompt or SYSTEM
    providers["ast_llm"]["stt"]["system_prompt"] = system_prompt or SYSTEM
    
    providers_order = os.getenv("PROVIDERS_ORDER", "gemini,openai,ast_llm")
    providers_order = [provider.strip() for provider in providers_order.split(",")]
    
    # Reset provider variables
    stt_provider = None
    tts_provider = None
    stt_backup_provider = []
    tts_backup_provider = []
    
    for provider in providers_order:
        if provider == "gemini" and all(providers["gemini"]["stt"].values()):
            provider_instance = GeminiSTTProvider(**providers["gemini"]["stt"])
            if stt_provider is None:
                stt_provider = provider_instance
            else:
                stt_backup_provider.append(provider_instance)
        if provider == "openai" and all(providers["openai"]["stt"].values()):
            provider_instance = OpenAIProvider(**providers["openai"]["stt"])
            if stt_provider is None:
                stt_provider = provider_instance
            else:
                stt_backup_provider.append(provider_instance)
        if provider == "openai" and all(providers["openai"]["tts"].values()):
            provider_instance = OpenAIProvider(**providers["openai"]["tts"])
            if tts_provider is None:
                tts_provider = provider_instance
            else:
                tts_backup_provider.append(provider_instance)
        if provider == "ast_llm" and all(providers["ast_llm"]["stt"].values()):
            provider_instance = AstLLmProvider(**providers["ast_llm"]["stt"])
            if stt_provider is None:
                stt_provider = provider_instance
            else:
                stt_backup_provider.append(provider_instance)
    
    if stt_provider is None:
        raise ValueError("No STT provider found")
    logger.info(f"STT provider initialized: {type(stt_provider).__name__}")
    if tts_provider is None:
        raise ValueError("No TTS provider found")
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
        if VAD == "webrtc":
            vad = WebRTCVAD(target_sample_rate, min_speech_duration_ms=100)
        elif VAD == "silero":
            vad = SileroVAD(target_sample_rate)

        if self.adapter.sample_rate != self.vad.sample_rate:
            self.vad.sample_rate = self.adapter.sample_rate

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
            max_wait_time=MAX_WAIT_TIME,
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
        if self.adapter.sample_rate != self.vad.sample_rate:
            self.vad.sample_rate = self.adapter.sample_rate
                
        return self

    async def run(self, 
                  first_message: Optional[str] = None, 
                  uid: Optional[int | str] = None, 
                  allow_interruptions: bool = False,
                  
                  system_prompt: Optional[str] = None,
                  tts_pcm_response_format: Optional[str] = "pcm",
                  tts_response_sample_rate: Optional[int] = 24_000,
                  tts_gen_config: Optional[Dict[str, Any]] = None,
                  stt_gen_config: Optional[Dict[str, Any]] = None,
                  ):
        try:
            self._task = asyncio.create_task(super().run(
                first_message=first_message,
                uid=uid,
                allow_interruptions=allow_interruptions,

                system_prompt=system_prompt,
                tts_pcm_response_format=tts_pcm_response_format,
                tts_response_sample_rate=tts_response_sample_rate,
                tts_gen_config=tts_gen_config,
                stt_gen_config=stt_gen_config,
            ))
        except Exception as e:
            logger.error(f"Error running server: {e}")
            raise
    
    def close(self):
        if self._task:
            self._task.cancel()
            self._task = None
        super().close()
    

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
    tts_pcm_response_format: Optional[str] = "pcm"
    tts_response_sample_rate: Optional[int] = 24_000
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
            tts_pcm_response_format=request.tts_pcm_response_format,
            tts_pcm_response_sample_rate=request.tts_response_sample_rate,
            tts_gen_config=request.tts_gen_config,
            stt_gen_config=request.stt_gen_config,
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
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function to run at exit
    atexit.register(cleanup_shutdown)

    initialize_agent(system_prompt=SYSTEM, chat_limit=CHAT_LIMIT)
    SingletonServer.set_host_ip(args.host, args.port)
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        cleanup_shutdown()
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        cleanup_shutdown()

if __name__ == "__main__":
    main() 