from fastapi import APIRouter, Depends
from typing import Union, Dict, Any

from ..models import (
    ServerConfig,
    RunParams,
    Response,
    StartServerResponse,
    StopServerRequest,
    UpdateAgentRequest,
)
from ..server_manager import ServerManager
from .dependencies import get_server_manager


router = APIRouter(prefix="/server", tags=["server"])


@router.post("/start", response_model=StartServerResponse, summary="Start a new server instance")
async def start_server(server_config: ServerConfig, manager: ServerManager = Depends(get_server_manager)):
    """Start a new server instance"""
    host_ip, host_port = manager.start_server(server_config)
    return StartServerResponse(success=True, host_ip=host_ip, host_port=host_port)


@router.post("/run", response_model=Response, summary="Run a server with parameters")
async def run_server(run_params: RunParams, manager: ServerManager = Depends(get_server_manager)):
    """Run a server instance with the specified parameters"""
    manager.run_server(run_params)
    return Response(success=True)


@router.post("/update_agent", response_model=Response, summary="Update agent configuration")
async def update_agent(
    request: UpdateAgentRequest,
    manager: ServerManager = Depends(get_server_manager),
):
    """Update agent configuration for a specific server"""
    manager.update_agent(
        uid=request.uid,
        system_prompt=request.system_prompt,
        tts_gen_config=request.tts_gen_config,
        stt_gen_config=request.stt_gen_config,
    )
    return Response(success=True)


@router.post("/stop", response_model=Response, summary="Stop a server instance")
async def stop_server(request: StopServerRequest, manager: ServerManager = Depends(get_server_manager)):
    """Stop a server instance"""
    manager.stop_server(request.uid)
    return Response(success=True)
