from abc import ABC, abstractmethod
from typing import Union, AsyncGenerator, Any, Optional, Dict
from dataclasses import dataclass




@dataclass(frozen=True)
class Message:
    role: str
    content: Union[str, bytes]
    data_type: str = "text" # audio, video, image.

    def __post_init__(self):
        if self.data_type not in ["text", "audio", "video", "image"]:
            raise ValueError(f"Invalid data type: {self.data_type}")


class BaseTTSProvider(ABC):


    @abstractmethod
    async def tts(self, 
                  text: str, 
                  response_format: str = "pcm",
                  response_sample_rate: int = 24_000,
                  gen_config: Optional[Dict[str, Any]] = None,
                  ) -> bytes:
        """
        Generate audio from text.
        TTS_RESPONSE_FORMATS = ["pcm", "ulaw", "alaw", "opus"]
        TTS_RESPONSE_SAMPLE_RATES = [8000, 16000, 24000, 48000]
        """
        pass


    async def tts_stream(self, 
                         text: str, 
                         response_format: str = "pcm",
                         response_sample_rate: int = 24_000,
                         gen_config: Optional[Dict[str, Any]] = None,
                         ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio from text with streaming.
        """
        raise NotImplementedError("Streaming is not supported for this STT provider")


class BaseSTTProvider(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.system_prompt = None

    @abstractmethod
    async def format(self, message: Message) -> Any:
        """given a messsage, format it to the provider's format"""
        pass

    @abstractmethod
    async def stt(self, 
                  formatted_wav_audio: Any, 
                  gen_config: Optional[Dict[str, Any]] = None,
                  ) -> str:
        """
        Generate text from audio message: a message with audio formatted to the provider's format.
        """
        pass


    async def stt_stream(self,
                         formatted_wav_audio: Any,
                         gen_config: Optional[Dict[str, Any]] = None,
                         ) -> AsyncGenerator[str, None]:
        """
        Generate text from audio message with streaming.
        """
        raise NotImplementedError("Streaming is not supported for this STT provider")
