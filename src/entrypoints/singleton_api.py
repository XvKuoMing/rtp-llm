import os
import asyncio
import fastapi
from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import Optional
import logging
import traceback
import argparse
from datetime import datetime

from rtp_llm.server import Server
from rtp_llm.adapters import RTPAdapter
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.vad import WebRTCVAD
from rtp_llm.providers import OpenAIProvider, AstLLmProvider, GeminiSTTProvider
from rtp_llm.agents import VoiceAgent
from rtp_llm.audio_logger import AudioLogger
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
SYSTEM = """
You are a helpful assistant.
"""
providers = {
    "gemini": {
        "stt": {
            "api_key": os.getenv("GEMINI_STT_API_KEY"),
            "base_url": os.getenv("GEMINI_STT_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            "model": os.getenv("GEMINI_STT_MODEL", "gemini-2.0-flash"),
            "system_prompt": SYSTEM
        },
    },
    "openai": {
        "stt": {
            "overwrite_stt_model_api_key": os.getenv("OPENAI_STT_API_KEY"),
            "overwrite_stt_model_base_url": os.getenv("OPENAI_STT_BASE_URL", "https://api.openai.com/v1"),
            "stt_model": os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-audio-preview"),
            "system_prompt": SYSTEM
        },
        "tts": {
            "overwrite_tts_model_api_key": os.getenv("OPENAI_TTS_API_KEY"),
            "overwrite_tts_model_base_url": os.getenv("OPENAI_TTS_BASE_URL", "https://api.openai.com/v1"),
            "tts_model": os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
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
            "system_prompt": SYSTEM
        },
    }
}

# Global provider variables
stt_provider = None
tts_provider = None
stt_backup_provider = []
tts_backup_provider = []

def initialize_providers():
    """Initialize STT and TTS providers based on configuration and availability"""
    global stt_provider, tts_provider, stt_backup_provider, tts_backup_provider
    
    # Update providers dict with current SYSTEM prompt
    providers["gemini"]["stt"]["system_prompt"] = SYSTEM
    providers["openai"]["stt"]["system_prompt"] = SYSTEM
    providers["ast_llm"]["stt"]["system_prompt"] = SYSTEM
    
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
def create_voice_agent():
    
    try:
        voice_agent = VoiceAgent(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            history_manager=ChatHistoryLimiter(),
            backup_stt_providers=stt_backup_provider,
            backup_tts_providers=tts_backup_provider
        )
        logger.info("Voice agent initialized successfully")
        return voice_agent
    except Exception as e:
        logger.error(f"Failed to initialize voice agent: {e}")
        raise

class SingletonServer:
    _instance = None
    _task = None
    __host_ip = None
    __host_port = None

    @staticmethod
    def get_instance():
        return SingletonServer._instance
    
    @staticmethod
    def set_host_ip(host_ip: str, host_port: int):
        SingletonServer.__host_ip = host_ip
        SingletonServer.__host_port = host_port

    def __new__(cls, channel_id: str | int, **kwargs):
        if cls._instance is None:
            logger.info(f"Creating new SingletonServer instance for channel_id: {channel_id}")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, channel_id: str | int, **kwargs):
        if hasattr(self, '_initialized'):
            return
        
        self.task = None
        self.channel_id = None
        self._initialized = False
        self.server = None
        
        try:
            # Create RTP adapter
            adapter = RTPAdapter(
                host_ip=self.__class__.__host_ip or "0.0.0.0",
                host_port=self.__class__.__host_port or 5000,
                peer_ip=kwargs.get("peer_ip"),
                peer_port=kwargs.get("peer_port"),
                sample_rate=kwargs.get("target_sample_rate", 8000),
                target_codec=kwargs.get("target_codec", "pcm")
            )
            
            # Create voice agent
            voice_agent = create_voice_agent()
            
            # Create server instance
            self.server = Server(
                adapter=adapter,
                audio_buffer=ArrayBuffer(),
                flow_manager=CopyFlowManager(),
                vad=WebRTCVAD(sample_rate=8000, aggressiveness=3, min_speech_duration_ms=500),
                agent=voice_agent,
            )
            
            self.task = None
            self.channel_id = channel_id
            self._initialized = True
            logger.info(f"SingletonServer initialized for channel_id: {channel_id}")
        except Exception as e:
            logger.error(f"Failed to initialize SingletonServer: {e}")
            raise

    @property
    def is_running(self):
        return self.task is not None and not self.task.done()

    def start(self, *args, **kwargs):
        if self.is_running:
            logger.warning("Server is already running")
            return
        
        try:
            logger.info("Starting RTP server...")
            self.task = asyncio.create_task(self.server.run())
            logger.info("RTP server started successfully")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise

    def stop(self):
        if not self.is_running:
            logger.warning("Server is not running")
            return
        
        try:
            logger.info("Stopping RTP server...")
            if self.task:
                self.task.cancel()
                self.task = None
            
            if self.server:
                self.server.close()
            
            logger.info("RTP server stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            raise

# FastAPI app with enhanced configuration
app = fastapi.FastAPI(
    title="RTP LLM Singleton API",
    description="API for managing a singleton RTP server instance",
    version="1.0.0"
)

# Response models
class APIResponse(BaseModel):
    message: str
    status: str
    timestamp: datetime
    data: dict = {}

class StartRTPRequest(BaseModel):
    channel_id: str | int
    peer_ip: Optional[str] = None
    peer_port: Optional[int] = None
    tts_response_format: Optional[str] = None
    tts_codec: Optional[str] = None
    target_codec: Optional[str] = None
    tts_sample_rate: Optional[int] = None
    target_sample_rate: Optional[int] = None
    first_message: Optional[str] = None
    allow_interruptions: bool = False

# Middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    
    logger.info(f"Request started: {method} {url} from {client_ip}")
    
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request completed: {method} {url} - Status: {response.status_code} - Duration: {duration:.3f}s")
        return response
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Request failed: {method} {url} - Error: {str(e)} - Duration: {duration:.3f}s")
        raise

@app.post("/start", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def start_rtp_server(request: StartRTPRequest):
    """Start the RTP server with the given configuration"""
    try:
        logger.info(f"Received start request for channel_id: {request.channel_id}")
        
        # Get existing instance or create new one
        server = SingletonServer.get_instance()
        
        if server and server.is_running:
            logger.warning("Server is already running")
            return APIResponse(
                message="Server is already running",
                status="warning",
                timestamp=datetime.now(),
                data={"channel_id": server.channel_id, "is_running": True}
            )
        
        # Create new server instance
        server = SingletonServer(
            channel_id=request.channel_id,
            peer_ip=request.peer_ip,
            peer_port=request.peer_port,
            tts_response_format=request.tts_response_format,
            tts_codec=request.tts_codec,
            target_codec=request.target_codec,
            tts_sample_rate=request.tts_sample_rate,
            target_sample_rate=request.target_sample_rate,
            allow_interruptions=request.allow_interruptions
        )
        
        # Start the server
        server.start()
        
        logger.info(f"RTP server started successfully for channel_id: {request.channel_id}")
        
        return APIResponse(
            message="RTP server started successfully",
            status="success", 
            timestamp=datetime.now(),
            data={
                "channel_id": request.channel_id,
                "is_running": server.is_running,
                "peer_ip": request.peer_ip,
                "peer_port": request.peer_port
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting RTP server: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start RTP server: {str(e)}"
        )

@app.post("/stop", response_model=APIResponse)
async def stop_rtp_server():
    """Stop the currently running RTP server"""
    try:
        server = SingletonServer.get_instance()
        
        if not server or not server.is_running:
            logger.warning("No server is currently running")
            return APIResponse(
                message="No server is currently running",
                status="warning",
                timestamp=datetime.now(),
                data={"is_running": False}
            )
        
        # Stop the server
        server.stop()
        
        logger.info("RTP server stopped successfully")
        
        return APIResponse(
            message="RTP server stopped successfully",
            status="success",
            timestamp=datetime.now(),
            data={"is_running": False}
        )
        
    except Exception as e:
        logger.error(f"Error stopping RTP server: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop RTP server: {str(e)}"
        )

@app.get("/ping", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    return APIResponse(
        message="API is healthy",
        status="success",
        timestamp=datetime.now(),
        data={"api_version": "1.0.0"}
    )

@app.get("/status", response_model=APIResponse)
async def get_server_status():
    """Get the current status of the RTP server"""
    try:
        server = SingletonServer.get_instance()
        
        if not server:
            return APIResponse(
                message="No server instance found",
                status="info",
                timestamp=datetime.now(),
                data={
                    "server_exists": False,
                    "is_running": False,
                    "channel_id": None
                }
            )
        
        return APIResponse(
            message="Server status retrieved successfully",
            status="success",
            timestamp=datetime.now(),
            data={
                "server_exists": True,
                "is_running": server.is_running,
                "channel_id": server.channel_id,
                "task_status": "running" if server.is_running else "stopped"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get server status: {str(e)}"
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

def load_system_prompt(path_or_prompt: str):
    """Load system prompt from file or use as direct prompt"""
    global SYSTEM
    if os.path.isfile(path_or_prompt):
        with open(path_or_prompt, 'r') as f:
            SYSTEM = f.read().strip()
        logger.info(f"Loaded system prompt from file: {path_or_prompt}")
    else:
        SYSTEM = path_or_prompt
        logger.info("Using provided system prompt")

def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="RTP LLM Singleton API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--system-prompt", help="System prompt or path to file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load system prompt if provided
    if args.system_prompt:
        load_system_prompt(args.system_prompt)
    
    # Set RTP host/port for singleton server
    SingletonServer.set_host_ip(args.host, args.port)
    
    # Initialize providers
    try:
        initialize_providers()
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")
        return
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"RTP server will bind to {args.rtp_host}:{args.rtp_port}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())

if __name__ == "__main__":
    main() 