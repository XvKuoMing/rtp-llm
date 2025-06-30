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
        self.host_ip = host_ip
        self.host_port = host_port
        self.peer_ip = peer_ip
        self.peer_port = peer_port
        self.sample_rate = sample_rate
        self.target_codec = target_codec

    @abstractmethod
    async def send_audio(self, audio_pcm16: bytes, audio_sample_rate: int = 24_000) -> None:
        """
        send audio chunk, must handle conversion from pcm16 to target codec and resampling if needed
        audio_sample_rate is the sample rate of the audio chunk
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

    

