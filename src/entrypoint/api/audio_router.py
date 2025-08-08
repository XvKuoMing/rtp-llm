from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from typing import Optional
import asyncio
import os

from ..server_manager import ServerManager
from ..models.audio import AudioFileInfo, AudioListResponse
from .dependencies import get_server_manager


router = APIRouter(prefix="/audio", tags=["audio"])


@router.get("/", response_model=AudioListResponse, summary="List audio files")
async def list_audio_files(
    uid: Optional[str] = Query(None, description="Filter by UID"),
    date_from: Optional[str] = Query(None, description="Filter by date from (ISO format, e.g., 2024-01-01T00:00:00)"),
    date_to: Optional[str] = Query(None, description="Filter by date to (ISO format, e.g., 2024-01-01T23:59:59)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    sort_by: str = Query("timestamp_desc", description="Sort by: timestamp_asc, timestamp_desc, duration_asc, duration_desc, size_asc, size_desc"),
    manager: ServerManager = Depends(get_server_manager),
):
    """
    List audio files with optional filtering and pagination.
    
    Returns metadata about audio files stored on disk.
    """
    return await manager.list_audio_files(
        uid=uid,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
        sort_by=sort_by
    )


@router.get("/{filename}", summary="Download audio file")
async def download_audio_file(filename: str, manager: ServerManager = Depends(get_server_manager)):
    """
    Download a specific audio file by filename.
    
    Returns the actual audio file for download.
    """
    filepath = manager.get_audio_file_path(filename)
    if not await asyncio.to_thread(os.path.exists, filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=filepath,
        media_type="audio/wav",
        filename=filename
    )


@router.get("/{filename}/info", response_model=AudioFileInfo, summary="Get audio metadata")
async def get_audio_file_metadata(filename: str, manager: ServerManager = Depends(get_server_manager)):
    """
    Get detailed metadata about a specific audio file.
    """
    return await manager.get_audio_file_metadata(filename)


@router.delete("/{filename}", summary="Delete audio file")
async def delete_audio_file(filename: str, manager: ServerManager = Depends(get_server_manager)):
    """
    Delete a specific audio file by filename.
    """
    await manager.delete_audio_file(filename)
    return {"message": f"Audio file {filename} deleted successfully"}