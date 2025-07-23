from entrypoints.config import BaseConfig, parse_config_from_env

from rtp_llm.agents import VoiceAgent
from rtp_llm.vad import BaseVAD
from rtp_llm.cache import RedisAudioCache
from rtp_llm.server import Server

from fastapi import FastAPI
from typing import Optional, Dict, Any, Set
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


class Callback(BaseModel):
    base_url: str
    on_response_endpoint: Optional[str] = None
    on_start_endpoint: Optional[str] = None
    on_error_endpoint: Optional[str] = None
    on_finish_endpoint: Optional[str] = None


class RunServerParams(BaseModel):
    first_message: Optional[str] = None
    allow_interruptions: bool = False
    system_prompt: Optional[str] = None
    tts_gen_config: Optional[Dict[str, Any]] = None
    stt_gen_config: Optional[Dict[str, Any]] = None
    tts_volume: Optional[float] = 1.0

class StartRTPRequest(BaseModel):
    uid: int | str
    host_ip: str
    host_port: int
    peer_ip: Optional[str] = None
    peer_port: Optional[int] = None
    target_sample_rate: Optional[int] = None
    target_codec: Optional[str] = None
    run_server_params: Optional[RunServerParams] = None
    callback: Optional[Callback] = None

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

        task = asyncio.create_task(server.run(
            first_message=request.run_server_params.first_message,
            uid=request.uid,
            allow_interruptions=request.run_server_params.allow_interruptions,
            system_prompt=request.run_server_params.system_prompt,
            tts_gen_config=request.run_server_params.tts_gen_config,
            stt_gen_config=request.run_server_params.stt_gen_config,
            tts_volume=request.run_server_params.tts_volume,
            callback=request.callback,
        ))
        running_servers[request.uid] = task
        running_servers_instances[request.uid] = server
        return {"message": "Server started"}
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return {"message": "Error starting server"}

class StopRTPRequest(BaseModel):
    uid: int | str


@app.post("/stop")
async def stop(request: StopRTPRequest):
    global running_servers, done_servers

    if request.uid in running_servers:
        running_servers[request.uid].cancel()
        running_servers.pop(request.uid)
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

    global voice_agent, vad, concurrency_limit, redis_audio_cache

    config = parse_config_from_env()
    
    voice_agent = config.initialize_agent()
    vad = config.initialize_vad()
    redis_audio_cache = config.initialize_redis_audio_cache()

    concurrency_limit = int(os.getenv("CONCURRENCY_LIMIT", -1))
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    main()
    