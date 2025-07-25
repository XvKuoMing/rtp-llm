from entrypoints.config import BaseConfig, parse_config_from_env

from rtp_llm.agents import VoiceAgent
from rtp_llm.vad import BaseVAD
from rtp_llm.cache import RedisAudioCache
from rtp_llm.server import Server
from rtp_llm.adapters.rtp import RTPAdapter
from rtp_llm.callbacks.rest_callback import RestCallback

from fastapi import FastAPI
from typing import Optional, Dict, Any, Set, List, Union
import asyncio
import os
import socket
from pydantic import BaseModel
import logging
import threading

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
        return {"message": "Concurrency limit reached", "status": "error"}

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
    
    redis_audio_cache = config.initialize_redis_audio_cache()
    logger.info("Redis audio cache initialized")

    concurrency_limit = os.getenv("CONCURRENCY_LIMIT", None) or args.concurrency_limit
    concurrency_limit = int(concurrency_limit) if concurrency_limit is not None else -1

    host = os.getenv("HOST", None) or args.host
    start_port = os.getenv("START_PORT", None) or args.start_port
    end_port = os.getenv("END_PORT", None) or args.end_port

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
    