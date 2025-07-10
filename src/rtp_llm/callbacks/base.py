from abc import ABC, abstractmethod



class BaseCallback(ABC):
    # NOTE: used as fire and forget functions, maybe later we will add some control flow
    

    @abstractmethod
    async def on_stt(self, uid: str, text: str) -> None:
        pass

    @abstractmethod
    async def on_tts(self, uid: str, text: str) -> None:
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