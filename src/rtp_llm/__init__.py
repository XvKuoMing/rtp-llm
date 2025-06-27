"""
RTP-LLM: Real-Time Voice Agent Framework

A powerful, modular framework for building real-time voice agents 
that communicate over RTP/UDP protocols.
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .agents import VoiceAgent
from .audio_logger import AudioLogger
from .server import Server

# Import submodules
from . import buffer
from . import flow
from . import history
from . import providers
from . import vad
from . import utils
from . import adapters

__all__ = [
    "VoiceAgent",
    "Server", 
    "AudioLogger",
    "buffer",
    "flow",
    "history",
    "providers",
    "vad",
    "utils",
    "adapters",
] 