from abc import ABC, abstractmethod
from enum import Enum


class VoiceState(Enum):
    SILENCE = "silence" # user is silent
    SPEECH = "speech" # user is speaking 
    
class BaseVAD(ABC):

    def __init__(self, sample_rate: int, min_speech_duration_ms: int = 60):
        self.sample_rate = sample_rate
        self.min_speech_duration_ms: int = min_speech_duration_ms
    
    @abstractmethod
    async def detect(self, pcm16_frame: bytes) -> VoiceState:
        """
        must yield voice state based on the whole pcm16 frame.
        it is assumed that vad make prediction based on the whole pcm16 frame.
        """
        ...
