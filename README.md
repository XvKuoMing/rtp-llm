# RTP-LLM: Real-Time Voice Agent Framework

A powerful, modular framework for building real-time voice agents that communicate over RTP/UDP protocols. RTP-LLM provides simple building blocks for creating responsive voice AI applications with support for multiple LLM providers, Voice Activity Detection (VAD), audio buffering, and comprehensive backup strategies.

## ✨ Features

### 🎯 **Core Capabilities**
- **Real-time RTP/UDP Communication**: Low-latency voice streaming over standard RTP protocol
- **Multiple LLM Provider Support**: Pluggable architecture supporting OpenAI, Gemini, and Alpaca providers
- **Voice Activity Detection (VAD)**: Smart detection of speech vs silence using WebRTC VAD and Silero models
- **Audio Backup & Logging**: Complete conversation recording with timestamps for debugging and analysis
- **Stateless API Design**: RESTful API for easy integration across different processes and systems

### 🔧 **Advanced Features**
- **Provider Failover**: Automatic backup provider switching for STT/TTS reliability
- **Flexible Audio Processing**: Support for multiple codecs (PCM16, etc.) and sample rates
- **Modular Architecture**: Easily swappable components for buffers, VAD, flow management, and providers
- **Conversation History Management**: Intelligent chat history with configurable limits
- **Audio Format Conversion**: Seamless resampling and format conversion between components

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RTP Server    │◄───│  Voice Agent    │◄───│   Providers     │
│                 │    │                 │    │ ┌─────────────┐ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ │ STT: Gemini │ │
│ │   Buffer    │ │    │ │   History   │ │    │ │ TTS: OpenAI │ │
│ │   VAD       │ │    │ │   Manager   │ │    │ │ Backup: ... │ │
│ │   Flow      │ │    │ └─────────────┘ │    │ └─────────────┘ │
│ └─────────────┘ │    └─────────────────┘    └─────────────────┘
└─────────────────┘                           
        │
        ▼
┌─────────────────┐
│  Audio Logger   │
│ (Logging/Debug)  │
└─────────────────┘
```

### Key Components

- **RTPServer**: Handles RTP packet processing, audio streaming, and protocol management
- **VoiceAgent**: Orchestrates STT/TTS operations and manages conversation flow
- **Providers**: Pluggable STT/TTS implementations (OpenAI, Gemini, Whisper+Openai Compatible LLM)
- **VAD (Voice Activity Detection)**: WebRTC and Silero-based speech detection
- **AudioLogger**: Comprehensive conversation recording
- **Modular Flow Management**: Extendable conversation flow control

## 🚀 Quick Start

### Installation

#### Option 1: Install as a dependency in your project

```bash
# Using uv (recommended)
uv add git+https://github.com/yourusername/rtp-llm

# Or using pip
pip install git+https://github.com/yourusername/rtp-llm

# Or if published to PyPI
uv add rtp-llm
pip install rtp-llm
```

#### Option 2: Install from source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/rtp-llm
cd rtp-llm

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

#### Option 3: Install with optional dependencies

```bash
# Install with test dependencies
uv add "git+https://github.com/yourusername/rtp-llm[tests]"
# or
pip install "git+https://github.com/yourusername/rtp-llm[tests]"
```

### Environment Setup

Create a `.env` file with your API keys:

```env
# STT Provider (Gemini)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_MODEL=gemini-1.5-flash

# TTS Provider (OpenAI)
OPENAI_TTS_API_KEY=your_openai_api_key
OPENAI_TTS_BASE_URL=https://api.openai.com/v1
OPENAI_TTS_MODEL=tts-1
```

### Usage Examples

#### 1. Basic RTP Server (Programmatic Usage)

When installed as a dependency, import and use the components directly:

```python
import asyncio
import os
from dotenv import load_dotenv

# Import from the installed rtp-llm package
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.vad import WebRTCVAD
from rtp_llm.providers import OpenAIProvider, GeminiSTTProvider
from rtp_llm.agents import VoiceAgent
from rtp_llm.rtp_server import RTPServer
from rtp_llm.audio_logger import AudioLogger

load_dotenv()

async def main():
    # Create voice agent with providers
    voice_agent = VoiceAgent(
        stt_provider=GeminiSTTProvider(
            api_key=os.getenv("GEMINI_API_KEY"),
            system_prompt="You are a helpful AI assistant"
        ),
        tts_provider=OpenAIProvider(
            api_key=os.getenv("OPENAI_TTS_API_KEY")
        ),
        history_manager=ChatHistoryLimiter(limit=10)
    )
    
    # Create RTP server
    server = RTPServer(
        buffer=ArrayBuffer(),
        agent=voice_agent,
        vad=WebRTCVAD(sample_rate=8000, aggressiveness=3),
        flow=CopyFlowManager(),
        host_ip="127.0.0.1",
        host_port=5000,
        peer_ip=None,  # Will auto-detect from first packet
        peer_port=None
    )
    
    # Start server
    await server.run(first_message="Hello! How can I help you?")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Using the CLI API Server

After installation, you can run the API server directly:

```bash
# Start the API server
rtp-llm-api --host 0.0.0.0 --port 8000

# Or with custom settings
rtp-llm-api --host localhost --port 9000 --log-level debug
```

Then make HTTP requests to control RTP servers:

```bash
# Start RTP server
curl -X POST "http://localhost:8000/start" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_id": "test_channel",
    "host_ip": "127.0.0.1",
    "host_port": 5000,
    "peer_ip": "127.0.0.1", 
    "peer_port": 5001,
    "target_codec": "pcm16",
    "tts_sample_rate": 24000,
    "target_sample_rate": 8000
  }'

# Check status
curl "http://localhost:8000/status"

# Stop server
curl -X POST "http://localhost:8000/stop"
```

#### 3. Using Just the Voice Agent

For non-RTP use cases, use just the voice processing components:

```python
import asyncio
import os
from rtp_llm.agents import VoiceAgent
from rtp_llm.providers import OpenAIProvider, GeminiSTTProvider
from rtp_llm.history import ChatHistoryLimiter

async def process_audio():
    voice_agent = VoiceAgent(
        stt_provider=GeminiSTTProvider(api_key=os.getenv("GEMINI_API_KEY")),
        tts_provider=OpenAIProvider(api_key=os.getenv("OPENAI_TTS_API_KEY")),
        history_manager=ChatHistoryLimiter(limit=5)
    )
    
    # Process audio bytes (you provide the audio data)
    # audio_bytes = ... your audio data ...
    # response_text = await voice_agent.stt(audio_bytes)
    # response_audio = await voice_agent.tts(response_text)
```

## 📁 Project Structure Recommendations

Based on your current setup, here are some improvements:

### ✅ **Good Aspects of Your Structure**
- **Modular design**: Clear separation of concerns with `buffer/`, `flow/`, `history/`, etc.
- **Comprehensive testing**: Dedicated `tests/` directory
- **Good examples**: `example.py` shows usage patterns

### 🔧 **Improvements Made**

1. **Fixed Package Structure**:
   - Added `src/__init__.py` for proper package imports
   - Updated `pyproject.toml` with proper package configuration
   - Fixed internal imports to use relative imports (`.` instead of `src.`)

2. **Added Entry Points**:
   - CLI command: `rtp-llm-api` for running the API server
   - Proper package metadata in `pyproject.toml`

3. **Better Examples**:
   - `examples/basic_usage.py`: Shows programmatic usage when installed
   - `examples/api_usage.py`: Shows API client usage

### 🎯 **Additional Recommendations**

1. **Documentation**:
   ```
   docs/
   ├── api_reference.md
   ├── user_guide.md
   └── examples/
   ```

2. **Configuration Management**:
   ```python
   # Consider adding src/config.py
   from dataclasses import dataclass
   from typing import Optional

   @dataclass
   class RTPConfig:
       host_ip: str = "127.0.0.1"
       host_port: int = 5000
       sample_rate: int = 8000
       # ... other config options
   ```

3. **More Provider Support**:
   ```
   src/providers/
   ├── __init__.py
   ├── openai.py
   ├── gemini.py
   ├── azure.py        # Future
   ├── elevenlabs.py   # Future
   └── custom.py       # For user extensions
   ```

4. **Better Error Handling**:
   ```python
   # Consider adding src/exceptions.py
   class RTPLLMError(Exception):
       """Base exception for RTP-LLM"""
       pass

   class ProviderError(RTPLLMError):
       """Provider-related errors"""
       pass
   ```

## 🧪 Testing Your Installation

After installing, test that everything works:

```python
# test_installation.py
import rtp_llm

# Test imports
from rtp_llm import VoiceAgent, RTPServer
from rtp_llm.providers import OpenAIProvider
from rtp_llm.buffer import ArrayBuffer

print("✅ rtp-llm installed successfully!")
print(f"📦 Version: {rtp_llm.__version__}")
```

## 📋 Configuration Options

### Voice Agent Configuration

```python
voice_agent = VoiceAgent(
    stt_provider=primary_stt_provider,
    tts_provider=primary_tts_provider,
    history_manager=ChatHistoryLimiter(limit=7),
    
    # Backup providers for reliability
    backup_stt_providers=[backup_stt_provider],
    backup_tts_providers=[backup_tts_provider],
    
    # Generation configs
    stt_gen_config={"temperature": 0.1},
    tts_gen_config={"voice": "nova", "speed": 1.0}
)
```

### RTP Server Configuration

```python
server = RTPServer(
    buffer=ArrayBuffer(),
    agent=voice_agent,
    vad=WebRTCVAD(sample_rate=8000, aggressiveness=3),
    flow=CopyFlowManager(),
    
    # Network settings
    host_ip="127.0.0.1",
    host_port=5000,
    peer_ip="127.0.0.1",  # Optional: auto-detect from first packet
    peer_port=5001,       # Optional: auto-detect from first packet
    
    # Audio settings
    tts_response_format="pcm",
    tts_codec="pcm16",
    target_codec="pcm16",
    tts_sample_rate=24_000,
    target_sample_rate=8_000,
    
    # Optional: Enable audio logging
    audio_logger=AudioLogger(uid="session_1")
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest tests/`
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🚀 What's Next?

- [ ] Add more STT/TTS providers
- [ ] WebRTC integration
- [ ] Docker containerization
- [ ] Kubernetes deployment guides
- [ ] Performance optimization
- [ ] Real-time monitoring dashboard


