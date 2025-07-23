from entrypoints.config import BaseConfig, parse_config_from_env

from rtp_llm.agents import VoiceAgent
from rtp_llm.vad import BaseVAD
from rtp_llm.cache import RedisAudioCache
from rtp_llm.server import Server
from rtp_llm.adapters.rtp import RTPAdapter

from fastapi import FastAPI
from typing import Optional, Dict, Any, Set, List
import asyncio
import os
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

config: Optional[BaseConfig] = None
voice_agent: Optional[VoiceAgent] = None
vad: Optional[BaseVAD] = None
concurrency_limit: int = -1 # -1 for unlimited
redis_audio_cache: Optional[RedisAudioCache] = None

running_servers: Dict[str, asyncio.Task] = {}
running_servers_instances: Dict[str, Server] = {}
done_servers: Set[str] = set()

app = FastAPI()


class StartRTPRequest(BaseModel):
    uid: int | str
    host_ip: str
    host_port: int
    peer_ip: Optional[str] = None
    peer_port: Optional[int] = None
    target_sample_rate: Optional[int] = None
    target_codec: Optional[str] = None


class StartResponse(BaseModel):
    success: bool
    host: str
    host: int

@app.post("/start")
async def start(request: StartRTPRequest):
    global voice_agent, vad, running_servers

    if concurrency_limit != -1 and len(running_servers) >= concurrency_limit:
        return {"message": "Concurrency limit reached", "status": "error"}

    try:
        server = config.initialize_rtp_server(
            host_ip=request.host_ip,
            host_port=request.host_port,
            peer_ip=request.peer_ip,
            peer_port=request.peer_port,
            sample_rate=request.target_sample_rate,
            codec=request.target_codec,
            voice_agent=voice_agent,
            vad=vad,
            audio_cache=redis_audio_cache,
        )

        running_servers_instances[request.uid] = server
        return StartResponse(success=True, host=request.host_ip, port=request.host_port)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return StartResponse(success=False, host=request.host_ip, port=request.host_port)

class Callback(BaseModel):
    base_url: str
    on_response_endpoint: Optional[str] = None
    on_start_endpoint: Optional[str] = None
    on_error_endpoint: Optional[str] = None
    on_finish_endpoint: Optional[str] = None


class RunRequest(BaseModel):
    uid: int | str
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
    if not request.uid in running_servers_instances:
        return StartResponse(success=False, host="", port=0)
    
    server = running_servers_instances[request.uid]

    if isinstance(server.adapter, RTPAdapter):
        host = server.adapter.host_ip
        port = server.adapter.host_port
    else:
        return StartResponse(success=False, host="", port=0)


    task = asyncio.create_task(server.run(
        first_message=request.first_message,
        uid=request.uid,
        allow_interruptions=request.allow_interruptions,
        system_prompt=request.system_prompt,
        tts_gen_config=request.tts_gen_config,
        stt_gen_config=request.stt_gen_config,
        tts_volume=request.tts_volume,
        callback=request.callback,
    ))


    if task.done():
        if task.exception():
            logger.error(f"Error running server: {task.exception()}")
        if task.cancelled():
            logger.error(f"Server cancelled almost immediately: {task.cancelled()}")
        return StartResponse(success=False, host=host, port=port)

    running_servers[request.uid] = task
    return StartResponse(success=True, host=host, port=port)


class StopRTPRequest(BaseModel):
    uid: int | str


@app.post("/stop")
async def stop(request: StopRTPRequest):
    global running_servers, done_servers

    if request.uid in running_servers:
        running_servers[request.uid].cancel()
        running_servers.pop(request.uid)
        running_servers_instances.pop(request.uid)
        done_servers.add(request.uid)
        return {"message": "Server stopped"}
    return {"message": "Server not found"}


class UpdateRTPRequest(BaseModel):
    uid: int | str
    system_prompt: Optional[str] = None
    

@app.post("/update")
async def update(request: UpdateRTPRequest):
    if not request.uid in running_servers_instances:
        return {"message": "Server not found"}
    
    server = running_servers_instances[request.uid]
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

    global voice_agent, vad, concurrency_limit, redis_audio_cache, config

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency-limit", type=int, default=-1)
    parser.add_argument("--env_file", type=str, default=None)
    args = parser.parse_args()

    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)

    config = parse_config_from_env()
    
    voice_agent = config.initialize_agent()
    vad = config.initialize_vad()
    redis_audio_cache = config.initialize_redis_audio_cache()

    concurrency_limit = os.getenv("CONCURRENCY_LIMIT", None) or args.concurrency_limit
    concurrency_limit = int(concurrency_limit) if concurrency_limit is not None else -1

    host = os.getenv("HOST", None) or args.host

    port = os.getenv("PORT", None) or args.port
    port = int(port) if port is not None else 8000

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
    