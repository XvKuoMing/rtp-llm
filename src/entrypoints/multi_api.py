from entrypoints.config import BaseConfig, parse_config_from_env

from rtp_llm.agents import VoiceAgent
from rtp_llm.vad import BaseVAD
from rtp_llm.cache import RedisAudioCache
from rtp_llm.server import Server
from rtp_llm.adapters.rtp import RTPAdapter
from rtp_llm.callbacks.rest_callback import RestCallback
from rtp_llm.audio_logger import AUDIO_LOGS_DIR

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any, Set, List, Union
import asyncio
import os
import socket
from pydantic import BaseModel
import logging
import threading
import glob
import wave
from datetime import datetime
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

config: Optional[BaseConfig] = None
voice_agent: Optional[VoiceAgent] = None
vad: Optional[BaseVAD] = None
host: str = "0.0.0.0"
start_port: int = 10000
end_port: int = 20000
concurrency_limit: int = -1 # -1 for unlimited
redis_audio_cache: Optional[RedisAudioCache] = None

running_servers: Dict[str, asyncio.Task] = {}
running_servers_instances: Dict[str, Server] = {}
done_servers: Set[str] = set()
used_ports: Set[int] = set()

# Add thread lock for port management
port_lock = threading.Lock()

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """Initialize async components during FastAPI startup"""
    global redis_audio_cache, config
    if config is not None:
        try:
            redis_audio_cache = await config.initialize_redis_audio_cache()
            logger.info("Redis audio cache initialized during startup")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache during startup: {e}")
            # Fallback to in-memory cache
            from rtp_llm.cache import InMemoryAudioCache
            redis_audio_cache = InMemoryAudioCache()
            logger.info("Falling back to in-memory audio cache")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up async components during FastAPI shutdown"""
    global redis_audio_cache
    if redis_audio_cache and hasattr(redis_audio_cache, 'close'):
        try:
            await redis_audio_cache.close()
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")


def get_static_host_ip():
    """Get the current OS static host IP address"""
    return socket.gethostbyname(socket.gethostname())


def get_available_port(uid: str) -> int:
    """Get an available port from the range, with deterministic assignment based on UID"""
    global used_ports
    
    with port_lock:  # Thread-safe port management
        # Try to get a deterministic port based on UID hash
        uid_hash = hash(str(uid)) % (end_port - start_port + 1)
        preferred_port = start_port + uid_hash
        
        # Check if preferred port is available
        if preferred_port not in used_ports and is_port_available(preferred_port):
            used_ports.add(preferred_port)
            return preferred_port
        
        # If preferred port is not available, find next available port
        for port in range(start_port, end_port + 1):
            if port not in used_ports and is_port_available(port):
                used_ports.add(port)
                return port
        
        raise RuntimeError(f"No available ports in range {start_port}-{end_port}")


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True
    except OSError:
        return False


def release_port(port: int):
    """Release a port back to the available pool"""
    global used_ports
    with port_lock:  # Thread-safe port management
        used_ports.discard(port)

class StartRTPRequest(BaseModel):
    uid: Union[int, str]  # Fixed type annotation

    peer_ip: Optional[str] = None
    peer_port: Optional[int] = None
    target_sample_rate: Optional[int] = None
    target_codec: Optional[str] = None


class StartResponse(BaseModel):
    success: bool
    host: str
    port: int

@app.post("/start")
async def start(request: StartRTPRequest):
    global voice_agent, vad, running_servers, host

    logger.info(f"Starting server for UID {request.uid}")

    if concurrency_limit != -1 and len(running_servers) >= concurrency_limit:
        logger.warning(f"Concurrency limit reached ({len(running_servers)}/{concurrency_limit})")
        raise HTTPException(status_code=429, detail="Concurrency limit reached")

    try:
        logger.info(f"Using host IP: {host}")
        
        # Get available port for this UID
        host_port = get_available_port(str(request.uid))
        logger.info(f"Allocated port {host_port} for UID {request.uid}")
        
        server = config.initialize_rtp_server(
            host_ip=host,
            host_port=host_port,
            peer_ip=request.peer_ip,
            peer_port=request.peer_port,
            sample_rate=request.target_sample_rate,
            codec=request.target_codec,
            voice_agent=voice_agent,
            vad=vad,
            audio_cache=redis_audio_cache,
        )

        running_servers_instances[str(request.uid)] = server
        logger.info(f"Server initialized successfully for UID {request.uid} at {host}:{host_port}")
        return StartResponse(success=True, host=host, port=host_port)
    except Exception as e:
        logger.error(f"Error starting server for UID {request.uid}: {e}")
        return StartResponse(success=False, host="", port=0)

class Callback(BaseModel):
    base_url: str
    on_response_endpoint: Optional[str] = None
    on_start_endpoint: Optional[str] = None
    on_error_endpoint: Optional[str] = None
    on_finish_endpoint: Optional[str] = None


class RunRequest(BaseModel):
    uid: Union[int, str]  # Fixed type annotation
    first_message: Optional[str] = None
    allow_interruptions: bool = False
    system_prompt: Optional[str] = None
    tts_gen_config: Optional[Dict[str, Any]] = None
    stt_gen_config: Optional[Dict[str, Any]] = None
    tts_volume: Optional[float] = 1.0
    callback: Optional[Callback] = None

@app.post("/run")
async def run(request: RunRequest):
    global running_servers
    uid_str = str(request.uid)
    if uid_str not in running_servers_instances:
        return StartResponse(success=False, host="", port=0)
    
    server = running_servers_instances[uid_str]

    if isinstance(server.adapter, RTPAdapter):
        server_host = server.adapter.host_ip  # Fixed variable shadowing
        server_port = server.adapter.host_port
    else:
        return StartResponse(success=False, host="", port=0)

    try:
        # Create task with exception handling wrapper
        async def run_with_error_handling():
            try:
                if request.callback:
                    logger.info(f"Using rest callback: {request.callback}")
                    callback = RestCallback(
                        base_url=request.callback.base_url,
                        on_response_endpoint=request.callback.on_response_endpoint,
                        on_start_endpoint=request.callback.on_start_endpoint,
                        on_error_endpoint=request.callback.on_error_endpoint,
                        on_finish_endpoint=request.callback.on_finish_endpoint,
                    )
                else:
                    # no callback configured
                    logger.info(f"No callback configured for UID {request.uid}")
                    callback = None
                logger.info(f"Starting server.run() for UID {request.uid}")
                await server.run(
                    first_message=request.first_message,
                    uid=request.uid,
                    allow_interruptions=request.allow_interruptions,
                    system_prompt=request.system_prompt,
                    tts_gen_config=request.tts_gen_config,
                    stt_gen_config=request.stt_gen_config,
                    volume=request.tts_volume,
                    callback=callback,
                )
            except Exception as e:
                logger.error(f"Server.run() failed for UID {request.uid}: {e}")
                # Clean up on failure
                if uid_str in running_servers:
                    running_servers.pop(uid_str)
                if uid_str in running_servers_instances:
                    running_servers_instances.pop(uid_str)
                raise

        task = asyncio.create_task(run_with_error_handling())

        # Store the task immediately - don't check if done since it just started
        running_servers[uid_str] = task
        logger.info(f"Server task created successfully for UID {request.uid} on {server_host}:{server_port}")
        return StartResponse(success=True, host=server_host, port=server_port)
        
    except Exception as e:
        logger.error(f"Error creating server task for UID {request.uid}: {e}")
        return StartResponse(success=False, host=server_host, port=server_port)


class StopRTPRequest(BaseModel):
    uid: Union[int, str]  # Fixed type annotation


@app.post("/stop")
async def stop(request: StopRTPRequest):
    global running_servers, done_servers

    uid_str = str(request.uid)
    logger.info(f"Stopping server for UID {request.uid}")

    if uid_str not in running_servers_instances:
        logger.warning(f"Stop requested for UID {request.uid} but server was not found")
        return {"message": "Server has not been started"}

    server = running_servers_instances[uid_str]
    running_servers_instances.pop(uid_str)
    done_servers.add(uid_str)

    if isinstance(server.adapter, RTPAdapter):
        port = server.adapter.host_port
        release_port(port)
        logger.info(f"Released port {port} for UID {request.uid}")
    
    if uid_str in running_servers:
        task = running_servers.pop(uid_str)
        task.cancel()
        logger.info(f"Server task cancelled for UID {request.uid}")
        return {"message": "Server stopped"}
    else:
        server.close()
        logger.info(f"Server closed for UID {request.uid} (wasn't running)")
        return {"message": "Server wasn't running, simply closed"}
    

class UpdateRTPRequest(BaseModel):
    uid: Union[int, str]  # Fixed type annotation
    system_prompt: Optional[str] = None


@app.post("/update")
async def update(request: UpdateRTPRequest):
    uid_str = str(request.uid)
    if uid_str not in running_servers_instances:
        return {"message": "Server not found"}
    
    server = running_servers_instances[uid_str]
    if request.system_prompt:
        server.agent.stt_provider.system_prompt = request.system_prompt
    return {"message": "updated"}


class ConfigView(BaseModel):
    # Common Server configuration
    max_wait_time: Optional[int] = None
    chat_limit: Optional[int] = None
    vad: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # Provider selection
    stt_providers: Optional[str] = None
    tts_providers: Optional[str] = None
    
    # Gemini STT configuration
    gemini_stt_api_key: Optional[str] = None
    gemini_stt_base_url: Optional[str] = None
    gemini_stt_model: Optional[str] = None
    
    # OpenAI STT configuration
    openai_stt_api_key: Optional[str] = None
    openai_stt_base_url: Optional[str] = None
    openai_stt_model: Optional[str] = None
    
    # OpenAI TTS configuration
    openai_tts_api_key: Optional[str] = None
    openai_tts_base_url: Optional[str] = None
    openai_tts_model: Optional[str] = None
    openai_tts_pcm_response_format: Optional[str] = None
    openai_tts_response_sample_rate: Optional[int] = None
    openai_tts_voice: Optional[str] = None
    
    # AST LLM STT configuration
    ast_api_key: Optional[str] = None
    ast_base_url: Optional[str] = None
    ast_model: Optional[str] = None
    ast_language: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    
    # AST LLM TTS configuration
    tts_api_key: Optional[str] = None
    tts_base_url: Optional[str] = None
    tts_model: Optional[str] = None
    tts_pcm_response_format: Optional[str] = None
    tts_response_sample_rate: Optional[int] = None
    tts_voice: Optional[str] = None


@app.post("/update_config")
async def update_config(request: ConfigView):
    global config, voice_agent, vad, redis_audio_cache
    
    if config is None:
        return {"message": "Config not initialized", "status": "error"}
    
    try:
        # Update config with non-None values from request
        updated_fields = []
        
        for field_name, field_value in request.model_dump(exclude_none=True).items():
            if hasattr(config, field_name):
                setattr(config, field_name, field_value)
                updated_fields.append(field_name)
                logger.info(f"Updated config.{field_name} = {field_value}")
        
        if not updated_fields:
            return {"message": "No valid fields to update", "status": "warning"}
        
        # Re-initialize VAD if VAD config changed
        if 'vad' in updated_fields:
            try:
                vad = config.initialize_vad()
                logger.info("VAD re-initialized successfully")
            except Exception as e:
                logger.error(f"Failed to re-initialize VAD: {e}")
                return {"message": f"Failed to re-initialize VAD: {e}", "status": "error"}
        
        
        # Re-initialize agent (always do this if any config changed)
        try:
            voice_agent = config.initialize_agent()
            logger.info("Voice agent re-initialized successfully")
        except Exception as e:
            logger.error(f"Failed to re-initialize voice agent: {e}")
            return {"message": f"Failed to re-initialize voice agent: {e}", "status": "error"}
        
        return {
            "message": "Config updated and components re-initialized successfully", 
            "updated_fields": updated_fields,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return {"message": f"Error updating config: {e}", "status": "error"}


@app.get("/config")
async def get_config():
    global config
    
    if config is None:
        raise HTTPException(status_code=404, detail="Config not initialized")
    
    try:
        return ConfigView(
            max_wait_time=config.max_wait_time,
            chat_limit=config.chat_limit,
            vad=config.vad,
            system_prompt=config.system_prompt,
            stt_providers=config.stt_providers,
            tts_providers=config.tts_providers,
            gemini_stt_api_key=config.gemini_stt_api_key,
            gemini_stt_base_url=config.gemini_stt_base_url,
            gemini_stt_model=config.gemini_stt_model,
            openai_stt_api_key=config.openai_stt_api_key,
            openai_stt_base_url=config.openai_stt_base_url,
            openai_stt_model=config.openai_stt_model,
            openai_tts_api_key=config.openai_tts_api_key,
            openai_tts_base_url=config.openai_tts_base_url,
            openai_tts_model=config.openai_tts_model,
            openai_tts_pcm_response_format=config.openai_tts_pcm_response_format,
            openai_tts_response_sample_rate=config.openai_tts_response_sample_rate,
            openai_tts_voice=config.openai_tts_voice,
            ast_api_key=config.ast_api_key,
            ast_base_url=config.ast_base_url,
            ast_model=config.ast_model,
            ast_language=config.ast_language,
            llm_model=config.llm_model,
            llm_api_key=config.llm_api_key,
            llm_base_url=config.llm_base_url,
            tts_api_key=config.tts_api_key,
            tts_base_url=config.tts_base_url,
            tts_model=config.tts_model,
            tts_pcm_response_format=config.tts_pcm_response_format,
            tts_response_sample_rate=config.tts_response_sample_rate,
            tts_voice=config.tts_voice,
        )
    except Exception as e:
        logger.error(f"Error retrieving config: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving config: {e}")


class StatusResponse(BaseModel):
    running_servers: int
    done_servers: int
    free_to_accept: bool


@app.get("/status")
async def status():
    return StatusResponse(
        running_servers=len(running_servers),
        done_servers=len(done_servers),
        free_to_accept=concurrency_limit == -1 or len(running_servers) < concurrency_limit
    )

@app.get("/ping")
async def ping():
    return {"message": "pong"}


class AudioFileInfo(BaseModel):
    filename: str
    uid: str
    conversation_timestamp: float
    file_size: int
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    created_date: str
    file_path: str


class AudioListResponse(BaseModel):
    audio_files: List[AudioFileInfo]
    total_count: int
    page: int
    page_size: int
    total_pages: int


def parse_audio_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse audio filename to extract UID and timestamp"""
    if not filename.endswith('.wav'):
        return None
    
    # Expected format: {uid}_conversation_{timestamp}.wav
    try:
        name_without_ext = filename[:-4]  # Remove .wav
        parts = name_without_ext.split('_conversation_')
        if len(parts) != 2:
            return None
        
        uid = parts[0]
        timestamp = float(parts[1])
        
        return {
            'uid': uid,
            'timestamp': timestamp
        }
    except (ValueError, IndexError):
        return None


async def get_audio_file_info(filepath: str) -> Optional[AudioFileInfo]:
    """Get detailed information about an audio file (async)"""
    try:
        filename = os.path.basename(filepath)
        parsed = parse_audio_filename(filename)
        if not parsed:
            return None
        
        # Use asyncio.to_thread for blocking file I/O operations
        if not await asyncio.to_thread(os.path.exists, filepath):
            return None
        
        file_size = await asyncio.to_thread(os.path.getsize, filepath)
        created_date = datetime.fromtimestamp(parsed['timestamp']).isoformat()
        
        # Try to get audio metadata asynchronously
        duration_seconds = None
        sample_rate = None
        channels = None
        
        try:
            # Run the wave file reading in a thread to avoid blocking
            audio_metadata = await asyncio.to_thread(_read_wave_metadata, filepath)
            if audio_metadata:
                frames, sample_rate, channels = audio_metadata
                duration_seconds = frames / float(sample_rate) if sample_rate > 0 else None
        except Exception as e:
            logger.warning(f"Could not read audio metadata for {filepath}: {e}")
        
        return AudioFileInfo(
            filename=filename,
            uid=parsed['uid'],
            conversation_timestamp=parsed['timestamp'],
            file_size=file_size,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            created_date=created_date,
            file_path=filepath
        )
    except Exception as e:
        logger.error(f"Error getting audio file info for {filepath}: {e}")
        return None


def _read_wave_metadata(filepath: str) -> Optional[tuple]:
    """Synchronous helper function to read wave file metadata"""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            return (frames, sample_rate, channels)
    except Exception:
        return None


@app.get("/audio", response_model=AudioListResponse)
async def list_audio_files(
    uid: Optional[str] = Query(None, description="Filter by UID"),
    date_from: Optional[str] = Query(None, description="Filter by date from (ISO format, e.g., 2024-01-01T00:00:00)"),
    date_to: Optional[str] = Query(None, description="Filter by date to (ISO format, e.g., 2024-01-01T23:59:59)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    sort_by: str = Query("timestamp_desc", description="Sort by: timestamp_asc, timestamp_desc, duration_asc, duration_desc, size_asc, size_desc")
):
    """
    List audio files with optional filtering and pagination.
    
    Returns metadata about audio files stored on disk.
    """
    try:
        # Ensure audio logs directory exists
        if not os.path.exists(AUDIO_LOGS_DIR):
            return AudioListResponse(
                audio_files=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0
            )
        
        # Get all WAV files
        audio_pattern = os.path.join(AUDIO_LOGS_DIR, "*.wav")
        audio_files_paths = await asyncio.to_thread(glob.glob, audio_pattern)
        
        # Parse file information in parallel with a semaphore to limit concurrency
        MAX_CONCURRENT_FILES = 50  # Limit concurrent file operations
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
        
        async def get_file_info_with_semaphore(filepath: str) -> Optional[AudioFileInfo]:
            async with semaphore:
                return await get_audio_file_info(filepath)
        
        # Process all files in parallel
        if audio_files_paths:
            tasks = [get_file_info_with_semaphore(filepath) for filepath in audio_files_paths]
            file_infos = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None values and exceptions
            audio_files_info = []
            for file_info in file_infos:
                if isinstance(file_info, AudioFileInfo):
                    audio_files_info.append(file_info)
                elif isinstance(file_info, Exception):
                    logger.warning(f"Error processing audio file: {file_info}")
        else:
            audio_files_info = []
        
        # Apply filters
        filtered_files = audio_files_info
        
        # Filter by UID
        if uid:
            filtered_files = [f for f in filtered_files if isinstance(f, AudioFileInfo) and f.uid == uid]
        
        # Filter by date range
        if date_from:
            try:
                date_from_ts = datetime.fromisoformat(date_from).timestamp()
                filtered_files = [f for f in filtered_files if isinstance(f, AudioFileInfo) and f.conversation_timestamp >= date_from_ts]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO format: 2024-01-01T00:00:00")
        
        if date_to:
            try:
                date_to_ts = datetime.fromisoformat(date_to).timestamp()
                filtered_files = [f for f in filtered_files if f.conversation_timestamp <= date_to_ts]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO format: 2024-01-01T23:59:59")
        
        # Sort files
        if sort_by == "timestamp_asc":
            filtered_files.sort(key=lambda x: x.conversation_timestamp)
        elif sort_by == "timestamp_desc":
            filtered_files.sort(key=lambda x: x.conversation_timestamp, reverse=True)
        elif sort_by == "duration_asc":
            filtered_files.sort(key=lambda x: x.duration_seconds or 0)
        elif sort_by == "duration_desc":
            filtered_files.sort(key=lambda x: x.duration_seconds or 0, reverse=True)
        elif sort_by == "size_asc":
            filtered_files.sort(key=lambda x: x.file_size)
        elif sort_by == "size_desc":
            filtered_files.sort(key=lambda x: x.file_size, reverse=True)
        else:
            raise HTTPException(status_code=400, detail="Invalid sort_by parameter")
        
        total_count = len(filtered_files)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
        
        # Apply pagination
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_files = filtered_files[start_index:end_index]
        
        return AudioListResponse(
            audio_files=paginated_files,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing audio files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing audio files: {str(e)}")


@app.get("/audio/{filename}")
async def download_audio_file(filename: str):
    """
    Download a specific audio file by filename.
    
    Returns the actual audio file for download.
    """
    try:
        # Validate filename format and prevent directory traversal
        if not filename.endswith('.wav') or '/' in filename or '\\' in filename or '..' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        filepath = os.path.join(AUDIO_LOGS_DIR, filename)
        
        if not await asyncio.to_thread(os.path.exists, filepath):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Verify it's actually an audio file we recognize
        parsed = parse_audio_filename(filename)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid audio filename format")
        
        return FileResponse(
            path=filepath,
            media_type="audio/wav",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading audio file: {str(e)}")


@app.get("/audio/{filename}/info", response_model=AudioFileInfo)
async def get_audio_file_metadata(filename: str):
    """
    Get detailed metadata about a specific audio file.
    """
    try:
        # Validate filename format and prevent directory traversal
        if not filename.endswith('.wav') or '/' in filename or '\\' in filename or '..' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        filepath = os.path.join(AUDIO_LOGS_DIR, filename)
        
        if not await asyncio.to_thread(os.path.exists, filepath):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        file_info = await get_audio_file_info(filepath)
        if not file_info:
            raise HTTPException(status_code=400, detail="Invalid audio file or unable to read metadata")
        
        return file_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audio file metadata for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting audio file metadata: {str(e)}")


@app.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    """
    Delete a specific audio file by filename.
    """
    try:
        # Validate filename format and prevent directory traversal
        if not filename.endswith('.wav') or '/' in filename or '\\' in filename or '..' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        filepath = os.path.join(AUDIO_LOGS_DIR, filename)
        
        if not await asyncio.to_thread(os.path.exists, filepath):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Verify it's actually an audio file we recognize
        parsed = parse_audio_filename(filename)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid audio filename format")
        
        await asyncio.to_thread(os.remove, filepath)
        logger.info(f"Deleted audio file: {filename}")
        
        return {"message": f"Audio file {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting audio file: {str(e)}")



def main():
    import uvicorn
    import argparse

    global voice_agent, vad, concurrency_limit, redis_audio_cache, config, host, start_port, end_port

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-port", type=int, default=10000)
    parser.add_argument("--end-port", type=int, default=20000)
    parser.add_argument("--concurrency-limit", type=int, default=-1)
    parser.add_argument("--env_file", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)

    config = parse_config_from_env()
    logger.info("Configuration parsed from environment")
    
    voice_agent = config.initialize_agent()
    logger.info("Voice agent initialized")
    
    vad = config.initialize_vad()
    logger.info("VAD initialized")
    
    # Redis cache will be initialized during FastAPI startup
    
    concurrency_limit = os.getenv("CONCURRENCY_LIMIT", None) or args.concurrency_limit
    concurrency_limit = int(concurrency_limit) if concurrency_limit is not None else -1

    host = os.getenv("HOST", None) or args.host
    _start_port = os.getenv("START_PORT", None) or args.start_port
    _end_port = os.getenv("END_PORT", None) or args.end_port
    start_port = int(_start_port) if _start_port else 10000
    end_port = int(_end_port) if _end_port else 20000

    port = os.getenv("PORT", None) or args.port
    port = int(port) if port is not None else 8000

    logger.info(f"Starting Multi-RTP API server")
    logger.info(f"Server host: {host}")
    logger.info(f"Server port: {port}")
    logger.info(f"RTP port range: {start_port}-{end_port}")
    logger.info(f"Concurrency limit: {concurrency_limit if concurrency_limit != -1 else 'unlimited'}")
    logger.info(f"Debug logging: {args.debug}")

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
    