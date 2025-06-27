from abc import ABC, abstractmethod
from typing import Optional

class Adapter:
    """
    adapter for recieving and sending audio
    """

    @abstractmethod
    def __init__(self, 
                 host_ip: str, 
                 host_port: int, 
                 peer_ip: Optional[str] = None, 
                 peer_port: Optional[int] = None, 
                 sample_rate: int = 8000,
                 target_codec: str | int = "pcm",
                 **kwargs):
        ...

    @abstractmethod
    async def send_audio(self, audio_pcm16: bytes, sample_rate: int = 24_000) -> None:
        """
        send audio chunk, must handle conversion from pcm16 to target codec and resampling if needed
        sample_rate is the sample rate of the audio chunk
        """
        ...

    @abstractmethod
    async def receive_audio(self) -> bytes:
        """
        must receive audio and return bytes in pcm16 format
        """
        ...
    
    @abstractmethod
    def close(self) -> None:
        """
        close the adapter
        """
        ...

    

