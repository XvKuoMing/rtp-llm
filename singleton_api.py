import os
import asyncio
import fastapi
from fastapi import HTTPException, status
from pydantic import BaseModel
from src.rtp_server import RTPServer
import logging
import traceback
import argparse
from datetime import datetime

from src.buffer import ArrayBuffer
from src.flow import CopyFlowManager
from src.history import ChatHistoryLimiter
from src.vad import WebRTCVAD
from src.providers import OpenAIProvider, GeminiSTTProvider
from src.agents import VoiceAgent
from src.audio_logger import AudioLogger
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

VOICE = "shimmer"
SYSTEM = """
Ты - ассистент компании "Водовоз".
"""
TTS_RESPONSE_FORMAT = "pcm"
TTS_CODEC = "pcm16"

# Initialize voice agent
try:
    voice_agent = VoiceAgent(
        stt_provider=GeminiSTTProvider(
            api_key=os.getenv("GEMINI_API_KEY"), 
            base_url=os.getenv("GEMINI_BASE_URL"),
            model=os.getenv("GEMINI_MODEL"),
            system_prompt=SYSTEM
        ),
        tts_provider=OpenAIProvider(
            api_key=os.getenv("OPENAI_TTS_API_KEY"), 
            base_url=os.getenv("OPENAI_TTS_BASE_URL"),
            tts_model=os.getenv("OPENAI_TTS_MODEL"),
        ),
        history_manager=ChatHistoryLimiter(limit=7),
        tts_gen_config={"voice": VOICE}
    )
    logger.info("Voice agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize voice agent: {e}")
    raise

class SingletonServer(RTPServer):
    _instance = None
    _task = None

    @staticmethod
    def get_instance():
        return SingletonServer._instance

    def __new__(cls, channel_id: str | int, **kwargs):
        if cls._instance is None:
            logger.info(f"Creating new SingletonServer instance for channel_id: {channel_id}")
            cls._instance = super().__new__(cls, channel_id, **kwargs)
        return cls._instance

    def __init__(self, channel_id: str | int, **kwargs):
        if hasattr(self, '_initialized'):
            return
        
        try:
            super().__init__(
                buffer=ArrayBuffer(),
                voice_agent=VoiceAgent(
                    stt_provider=GeminiSTTProvider(
                        api_key=os.getenv("GEMINI_API_KEY"), 
                        base_url=os.getenv("GEMINI_BASE_URL"),
                        model=os.getenv("GEMINI_MODEL"),
                        system_prompt=SYSTEM
                        ),
                        tts_provider=OpenAIProvider(
                            api_key=os.getenv("OPENAI_TTS_API_KEY"), 
                            base_url=os.getenv("OPENAI_TTS_BASE_URL"),
                            tts_model=os.getenv("OPENAI_TTS_MODEL"),
                        ),
                        history_manager=ChatHistoryLimiter(limit=7),
                        tts_gen_config={"voice": VOICE}
                    ),
                    vad=WebRTCVAD(sample_rate=8000, aggressiveness=3, min_speech_duration_ms=500),
                    flow=CopyFlowManager(),
                    audio_logger=AudioLogger(uid=channel_id),
                    tts_response_format=TTS_RESPONSE_FORMAT,
                    tts_codec=TTS_CODEC,
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
    host_ip: str
    host_port: int
    peer_ip: str
    peer_port: int
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
            server = SingletonServer(channel_id=request.channel_id, **request.model_dump())
        
        # Check if server is already running
        if server.is_running:
            logger.warning(f"Server already running for channel {request.channel_id}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="RTP server is already running for this process"
            )
        
        # Start the server
        success = server.start(
            host_ip=request.host_ip,
            host_port=request.host_port,
            peer_ip=request.peer_ip,
            peer_port=request.peer_port,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start RTP server"
            )
        
        return APIResponse(
            message="RTP server started successfully",
            status="success",
            timestamp=datetime.now(),
            data={
                "channel_id": request.channel_id,
                "host_endpoint": f"{request.host_ip}:{request.host_port}",
                "peer_endpoint": f"{request.peer_ip}:{request.peer_port}",
                "is_running": server.is_running
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting server: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/stop", response_model=APIResponse)
async def stop_rtp_server():
    """Stop the RTP server"""
    try:
        server = SingletonServer.get_instance()
        
        if server is None:
            logger.warning("Attempted to stop server when no instance exists")
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
            data={"is_running": False}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error stopping server: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/ping", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    return APIResponse(
        message="Service is healthy",
        status="success",
        timestamp=datetime.now(),
        data={"service": "rtp-server-api", "version": "1.0.0"}
    )

@app.get("/status", response_model=APIResponse)
async def get_server_status():
    """Get current server status"""
    try:
        server = SingletonServer.get_instance()
        
        if server is None:
            return APIResponse(
                message="No server instance found",
                status="success",
                timestamp=datetime.now(),
                data={
                    "instance_exists": False,
                    "is_running": False,
                    "channel_id": None
                }
            )
        
        return APIResponse(
            message="Server status retrieved successfully",
            status="success",
            timestamp=datetime.now(),
            data={
                "instance_exists": True,
                "is_running": server.is_running,
                "channel_id": getattr(server, 'channel_id', None)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get server status: {str(e)}"
        )

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return fastapi.responses.JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error occurred",
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "data": {}
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="RTP Server API - Stateless voice agent communication service")
    parser.add_argument(
        "--port", 
        type=int, 
        default=30_000,
        help="Port to run the API server on (default: %(default)s)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the API server to (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting RTP Server API on {args.host}:{args.port}...")
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )


