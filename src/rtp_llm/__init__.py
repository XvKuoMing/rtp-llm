"""
RTP-LLM: Real-Time Voice Agent Framework

A powerful, modular framework for building real-time voice agents 
that communicate over RTP/UDP protocols.
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .agents import VoiceAgent
from .rtp_server import RTPServer
from .audio_logger import AudioLogger

# Import submodules
from . import buffer
from . import flow
from . import history
from . import providers
from . import vad
from . import utils

__all__ = [
    "VoiceAgent",
    "RTPServer", 
    "AudioLogger",
    "buffer",
    "flow",
    "history",
    "providers",
    "vad",
    "utils",
] 