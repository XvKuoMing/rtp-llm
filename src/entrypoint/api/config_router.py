from fastapi import APIRouter, Depends
from typing import Dict, Any

from ..server_manager import ServerManager
from ..models import Response
from .dependencies import get_server_manager


router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("/providers", summary="Get providers configuration")
async def get_providers_config(manager: ServerManager = Depends(get_server_manager)):
    """Get the current providers configuration"""
    return manager.get_providers_config()


@router.post("/providers", response_model=Response, summary="Update providers configuration")
async def update_providers_config(providers_config: Dict[str, Any], manager: ServerManager = Depends(get_server_manager)):
    """Update the providers configuration"""
    manager.update_providers_config(providers_config)
    return Response(success=True)
