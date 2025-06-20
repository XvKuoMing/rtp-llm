from abc import ABC, abstractmethod
from enum import Enum


class VoiceState(Enum):
    SILENCE = "silence" # user is silent
    SPEECH = "speech" # user is speaking 
    
class BaseVAD(ABC):
    
    @abstractmethod
    async def detect(self, pcm16_frame: bytes) -> VoiceState:
        """
        must yield voice state based on the whole pcm16 frame.
        it is assumed that vad make prediction based on the whole pcm16 frame.
        """
        ...
