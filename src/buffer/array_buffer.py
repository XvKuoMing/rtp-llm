from .base import BaseAudioBuffer


class ArrayBuffer(BaseAudioBuffer):

    def __init__(self):
        self.buffer = bytearray()

    def add_frame(self, frame: bytes):
        self.buffer.extend(frame)

    def get_frames(self) -> bytes:
        return self.buffer
    
    def clear(self):
        self.buffer = bytearray()