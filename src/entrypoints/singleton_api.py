import os
import asyncio
import fastapi
from fastapi import HTTPException, status
from pydantic import BaseModel
import logging
import traceback
import argparse
from datetime import datetime

from ..rtp_llm.rtp_server import RTPServer
from ..rtp_llm.buffer import ArrayBuffer
from ..rtp_llm.flow import CopyFlowManager
from ..rtp_llm.history import ChatHistoryLimiter
from ..rtp_llm.vad import WebRTCVAD
from ..rtp_llm.providers import OpenAIProvider, GeminiSTTProvider
from ..rtp_llm.agents import VoiceAgent
from ..rtp_llm.audio_logger import AudioLogger

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
def create_voice_agent():
    from dotenv import load_dotenv
    load_dotenv()
    
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
        return voice_agent
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
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, channel_id: str | int, **kwargs):
        if hasattr(self, '_initialized'):
            return
        
        try:
            super().__init__(
                buffer=ArrayBuffer(),
                agent=create_voice_agent(),
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
        
        if not server.is_running:
            logger.warning("Attempted to stop server that was not running")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="RTP server is not currently running"
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

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description='RTP-LLM API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--log-level', default='info', help='Log level')
    
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False
    )

if __name__ == "__main__":
    main() 