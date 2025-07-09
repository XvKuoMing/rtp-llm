# import sounddevice as sd
# from .base import Adapter

# class LocalAdapter(Adapter):
#     # def __init__(self, 
#     #              sample_rate: int = 44100, 
#     #              target_codec: str | int = "pcm", 
#     #              channels: int = 1):
#     #     super().__init__(sample_rate=sample_rate, target_codec=target_codec)
#     #     self.channels = channels

#     # async def send_audio(self, audio: bytes) -> None:
#     #     sd.play(audio, self.sample_rate, self.channels)
#     #     await sd.wait()

#     # async def receive_audio(self) -> bytes:
#     #     audio = sd.rec(len(audio), samplerate=self.sample_rate, channels=self.channels)