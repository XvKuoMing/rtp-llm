from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Coroutine, Optional, Any

@dataclass(frozen=True)
class ResponseTransformation:
    text: Optional[str] = None # transformed text
    post_action: Optional[Coroutine[Any, Any, None]] = None # action to be performed after tts


class BaseCallback(ABC):
    # NOTE: used as fire and forget functions, maybe later we will add some control flow
    

    @abstractmethod
    async def on_response(self, uid: str, text: str) -> ResponseTransformation:
        pass

    @abstractmethod
    async def on_start(self, uid: str) -> None:
        pass

    @abstractmethod
    async def on_error(self, uid: str, error: Exception) -> None:
        pass
    
    @abstractmethod
    async def on_finish(self, uid: str) -> None:
        pass