from pydantic import BaseModel
from typing import Optional


class Response(BaseModel):
    success: bool
    message: Optional[str] = None


class StartServerResponse(Response):
    host_ip: Optional[str] = None
    host_port: Optional[int] = None
