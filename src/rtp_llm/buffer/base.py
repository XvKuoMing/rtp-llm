"""
base class for audio buffering
"""

from abc import ABC, abstractmethod
from typing import List


class BaseAudioBuffer(ABC):


    @abstractmethod
    async def add_frame(self, frame: bytes):
        """
        add frame to buffer
        """
        ...

    @abstractmethod
    async def get_frames(self) -> bytes:
        ...
    
    @abstractmethod
    def clear(self):
        """
        clear buffer
        """
        ...