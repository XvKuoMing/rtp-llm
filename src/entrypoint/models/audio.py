from pydantic import BaseModel
from typing import Optional, List


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
