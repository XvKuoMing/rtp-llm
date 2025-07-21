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
    def __init__(self, 
                 *args,
                 pcm_response_format: Optional[str] = None,
                 response_sample_rate: Optional[int] = None,
                 gen_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        pcm_response_format: The pcm format of the response from the tts_provider; pcm for openai, pcm_24000 for elevenlabs
        response_sample_rate: The sample rate of the response from the tts_provider.
        """
        self.pcm_response_format = pcm_response_format
        self.response_sample_rate = response_sample_rate
        self.tts_gen_config = gen_config or {}
    

    @property
    def tts_footprint(self) -> str:
        """
        A string that uniquely identifies the tts_provider.
        """
        return f"{self.pcm_response_format}_{self.response_sample_rate}_{str(self.tts_gen_config)}"
    

    @abstractmethod
    def validate_tts_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        invalidate wrong params from config, return only valid
        """
        return config


    @abstractmethod
    async def tts(self, 
                  text: str
                  ) -> bytes:
        """
        Generate audio from text.
        """
        pass


    async def tts_stream(self, 
                         text: str
                         ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio from text with streaming.
        """
        raise NotImplementedError("Streaming is not supported for this STT provider")


class BaseSTTProvider(ABC):

    @abstractmethod
    def __init__(self, 
                 *args,
                 system_prompt: Optional[str] = None, 
                 gen_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        self.system_prompt = system_prompt
        self.stt_gen_config = gen_config or {}

    @abstractmethod
    def validate_stt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        invalidate wrong params from config, return only valid
        """
        return config


    @abstractmethod
    async def format(self, message: Message) -> Any:
        """given a messsage, format it to the provider's format"""
        pass

    @abstractmethod
    async def stt(self, 
                  formatted_wav_audio: Any
                  ) -> str:
        """
        Generate text from audio message: a message with audio formatted to the provider's format.
        """
        pass


    async def stt_stream(self,
                         formatted_wav_audio: Any,
                         ) -> AsyncGenerator[str, None]:
        """
        Generate text from audio message with streaming.
        """
        raise NotImplementedError("Streaming is not supported for this STT provider")
