from abc import ABC, abstractmethod
from typing import Optional

class Adapter:
    """
    adapter for recieving and sending audio
    """

    @abstractmethod
    def __init__(self,  
                 *args,
                 sample_rate: int = 8000,
                 target_codec: str | int = "pcm",
                 **kwargs):
        self.sample_rate = sample_rate
        self.target_codec = target_codec

    @property
    def peer_is_configured(self) -> bool:
        """
        check if the peer is configured
        """
        pass

    @abstractmethod
    async def send_audio(self, audio_pcm16: bytes) -> None:
        """
        given audio_pcm16 in self.sample_rate, send audio chunk, must handle conversion from pcm16 to target codec
        """
        ...

    @abstractmethod
    async def receive_audio(self) -> bytes:
        """
        must receive audio and return bytes in pcm16 format
        it is assumed that input audio has the same sample rate as the output audio == self.sample_rate
        """
        ...
    
    @abstractmethod
    def close(self) -> None:
        """
        close the adapter
        """
        ...

    

