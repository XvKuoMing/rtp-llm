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

from entrypoints.config import BaseConfig, get_config_parser

from rtp_llm.server import Server
from rtp_llm.adapters import RTPAdapter
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.vad import WebRTCVAD, SileroVAD
from rtp_llm.providers import OpenAIProvider, AstLLmProvider, GeminiSTTProvider
from rtp_llm.agents import VoiceAgent
from rtp_llm.callbacks import RestCallback, BaseCallback, NullCallback
from rtp_llm.cache import InMemoryAudioCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)



@dataclass
class Config(BaseConfig):
    # Server configuration
    host: str
    port: int

# Global configuration instance
config: Optional[Config] = None


# Global provider variables
vad = None
voice_agent = None


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
    def close_instance():
        if SingletonServer._instance:
            SingletonServer._instance.close()
            SingletonServer._instance = None

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
            audio_cache=InMemoryAudioCache()
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
                  callback: Optional[BaseCallback] = None,
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
                callback=callback,
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


class Callback(BaseModel):
    base_url: str
    on_response_endpoint: Optional[str] = None
    on_start_endpoint: Optional[str] = None
    on_error_endpoint: Optional[str] = None
    on_finish_endpoint: Optional[str] = None


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
    callback: Optional[Callback] = None


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

        if request.callback:
            callback = RestCallback(
                base_url=request.callback.base_url,
                on_response_endpoint=request.callback.on_response_endpoint,
                on_start_endpoint=request.callback.on_start_endpoint,
                on_error_endpoint=request.callback.on_error_endpoint,
                on_finish_endpoint=request.callback.on_finish_endpoint,
            )
        else:
            callback = NullCallback()

        await server.run(
            first_message=request.first_message,
            uid=request.uid,
            allow_interruptions=request.allow_interruptions,

            system_prompt=request.system_prompt,
            tts_gen_config=request.tts_gen_config,
            stt_gen_config=request.stt_gen_config,
            tts_volume=request.tts_volume,
            callback=callback,
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
                SingletonServer.close_instance()
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

        SingletonServer.close_instance()
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
            SingletonServer.close_instance()
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
    parser = get_config_parser(description="RTP LLM Singleton API Server")
    
    # Server configuration
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Core configuration
    
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
    global config, voice_agent, vad
    
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
        openai_tts_voice=args.openai_tts_voice,
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
        tts_voice=args.tts_voice,
    )

    # Initialize agent with the configuration
    try:
        voice_agent = config.initialize_agent()
        vad = config.initialize_vad()
        
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