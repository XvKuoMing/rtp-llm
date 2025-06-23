import os
import asyncio
import fastapi
from fastapi import HTTPException, status
from pydantic import BaseModel
import logging
import traceback
import argparse
from datetime import datetime

from rtp_llm.rtp_server import RTPServer
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
    # Ensure providers are initialized if not already done
    if stt_provider is None or tts_provider is None:
        logger.info("Providers not initialized, initializing now...")
        initialize_providers()
    
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

class SingletonServer(RTPServer):
    _instance = None
    _task = None
    _host_ip = None
    _host_port = None

    @staticmethod
    def get_instance():
        return SingletonServer._instance
    
    @staticmethod
    def set_host_ip(host_ip: str, host_port: int):
        SingletonServer._host_ip = host_ip
        SingletonServer._host_port = host_port

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
        try:
            super().__init__(
                buffer=ArrayBuffer(),
                agent=create_voice_agent(),
                vad=WebRTCVAD(sample_rate=8000, aggressiveness=3, min_speech_duration_ms=500),
                flow=CopyFlowManager(),
                audio_logger=AudioLogger(uid=channel_id),
                host_ip=self.__class__._host_ip,
                host_port=self.__class__._host_port,
                **kwargs
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
            logger.warning(f"Attempted to start server for channel {self.channel_id} while already running")
            return False
        
        try:
            self.task = asyncio.create_task(self.run(*args, **kwargs))
            logger.info(f"Started RTP server for channel {self.channel_id} on {kwargs.get('host_ip', 'unknown')}:{kwargs.get('host_port', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to start server for channel {self.channel_id}: {e}")
            return False

    def stop(self):
        try:
            if self.task:
                self.task.cancel()
                self.task = None
                logger.info(f"Stopped RTP server for channel {self.channel_id}")
            
            self.close()
            SingletonServer._instance = None
            logger.info(f"Cleaned up server instance for channel {self.channel_id}")
            return True
        except Exception as e:
            logger.error(f"Error stopping server for channel {self.channel_id}: {e}")
            return False


# FastAPI app with enhanced configuration
app = fastapi.FastAPI(
    title="RTP Server API",
    description="Singleton RTP Server for Voice Agent Communication",
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
    peer_ip: str
    peer_port: int
    tts_response_format: str
    tts_codec: str
    target_codec: str
    tts_sample_rate: int
    target_sample_rate: int

# Middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request completed: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Request failed: {request.method} {request.url} - Error: {e} - Time: {process_time:.3f}s")
        raise

@app.post("/start", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def start_rtp_server(request: StartRTPRequest):
    """Start the RTP server for voice communication"""
    try:
        server = SingletonServer.get_instance()
        
        # Create new server if none exists
        if server is None:
            logger.info(f"Creating new server instance for channel {request.channel_id}")
            server = SingletonServer(**request.model_dump())
        
        # Check if server is already running
        if server.is_running:
            logger.warning(f"Server already running for channel {request.channel_id}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="RTP server is already running for this process"
            )

        # Start the server
        success = server.start(**request.model_dump())
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start RTP server"
            )

        return APIResponse(
            message=f"RTP server started successfully for channel {request.channel_id}",
            status="success",
            timestamp=datetime.now(),
            data={
                "channel_id": request.channel_id,
                "host": f"{request.host_ip}:{request.host_port}",
                "peer": f"{request.peer_ip}:{request.peer_port}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting RTP server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/stop", response_model=APIResponse)
async def stop_rtp_server():
    """Stop the running RTP server"""
    try:
        server = SingletonServer.get_instance()
        
        if server is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No RTP server instance found"
            )

        success = server.stop()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop RTP server"
            )

        return APIResponse(
            message="RTP server stopped successfully",
            status="success",
            timestamp=datetime.now(),
            data={"channel_id": server.channel_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error stopping RTP server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/ping", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    return APIResponse(
        message="RTP Server API is running",
        status="healthy",
        timestamp=datetime.now()
    )

@app.get("/status", response_model=APIResponse)
async def get_server_status():
    """Get current server status"""
    try:
        server = SingletonServer.get_instance()
        
        if server is None:
            status_info = {
                "server_exists": False,
                "is_running": False,
                "channel_id": None
            }
        else:
            status_info = {
                "server_exists": True,
                "is_running": server.is_running,
                "channel_id": server.channel_id,
                "host": f"{server.host_ip}:{server.host_port}",
                "peer": f"{server.peer_ip}:{server.peer_port}" if server.peer_ip and server.peer_port else None
            }

        return APIResponse(
            message="Server status retrieved successfully",
            status="success",
            timestamp=datetime.now(),
            data=status_info
        )

    except Exception as e:
        logger.error(f"Error retrieving server status: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return fastapi.responses.JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error occurred",
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "data": {}
        }
    )


def load_system_prompt(path_or_prompt: str):
    global SYSTEM
    if os.path.exists(path_or_prompt):
        try:
            with open(path_or_prompt, 'r', encoding="utf-8") as file:
                SYSTEM = file.read()
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode with utf-8 system: {path_or_prompt}")
    else:
        SYSTEM = path_or_prompt
    logger.info(f"System prompt loaded from {path_or_prompt}")

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description='RTP-LLM API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--system', default=SYSTEM, help='Plain system prompt or .txt file path')
    
    args = parser.parse_args()
    load_system_prompt(args.system)
    initialize_providers()
    SingletonServer.set_host_ip(args.host, args.port)
    
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main() 