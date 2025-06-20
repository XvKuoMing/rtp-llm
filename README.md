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

```bash
# Clone the repository
git clone <your-repo-url>
cd rtp-llm

# Install dependencies
pip install -r requirements.txt
# or using uv
uv sync
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

### Basic Usage

#### 1. Simple RTP Server Example

```python
import asyncio
from src.buffer import ArrayBuffer
from src.flow import CopyFlowManager
from src.history import ChatHistoryLimiter
from src.vad import WebRTCVAD
from src.providers import OpenAIProvider, GeminiSTTProvider
from src.agents import VoiceAgent
from src.rtp_server import RTPServer

async def main():
    # Create voice agent with providers
    voice_agent = VoiceAgent(
        stt_provider=GeminiSTTProvider(
            api_key="your_gemini_key",
            system_prompt="You are a helpful assistant"
        ),
        tts_provider=OpenAIProvider(
            api_key="your_openai_key"
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
        peer_ip="127.0.0.1",
        peer_port=5001
    )
    
    # Start server
    await server.run(first_message="Hello! How can I help you?")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Using the Stateless API

Start the API server:

```bash
python singleton_api.py --port 8000
```

Then make HTTP requests:

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
    # Core components
    buffer=ArrayBuffer(),
    agent=voice_agent,
    vad=WebRTCVAD(sample_rate=8000, aggressiveness=3),
    flow=CopyFlowManager(),
    
    # Network settings
    host_ip="127.0.0.1",
    host_port=5000,
    peer_ip="127.0.0.1", 
    peer_port=5001,
    
    # Audio settings
    tts_response_format="pcm",
    tts_codec="pcm16",
    target_codec="pcm16",
    tts_sample_rate=24_000,
    target_sample_rate=8_000,
    
    # Optional: Audio logging
    audio_logger=AudioLogger(uid="session_123")
)
```

## 🔌 Supported Providers

### Speech-to-Text (STT)
- **Gemini**: Google's Gemini API for speech recognition
- **Alpaca**: Local/self-hosted STT solutions

### Text-to-Speech (TTS)  
- **OpenAI**: OpenAI's TTS API with multiple voice options
- **Alpaca**: Local/self-hosted TTS solutions

### Voice Activity Detection (VAD)
- **WebRTC VAD**: Fast, lightweight voice detection
- **Silero VAD**: AI-powered voice activity detection

## 🛠️ Advanced Features

### Backup Provider System

The framework automatically handles provider failures:

```python
voice_agent = VoiceAgent(
    stt_provider=primary_provider,
    backup_stt_providers=[backup1, backup2],  # Automatic failover
    backup_tts_providers=[tts_backup1, tts_backup2]
)
```

### Audio Logging & Backup

All conversations are automatically logged for debugging:

```python
# Audio logs saved to ./audio_logs/
audio_logger = AudioLogger(uid="session_123", sample_rate=8000)
await audio_logger.log_user(user_audio)
await audio_logger.log_ai(ai_response)
await audio_logger.save()  # Saves complete conversation
```

### Custom Components

Extend the framework with custom implementations:

```python
class CustomVAD(BaseVAD):
    async def detect(self, pcm16_frame: bytes) -> VoiceState:
        # Your custom VAD logic
        return VoiceState.SPEECH or VoiceState.SILENCE

class CustomSTTProvider(BaseSTTProvider):
    async def stt(self, audio: bytes) -> str:
        # Your custom STT implementation
        return transcribed_text
```

## 📁 Project Structure

```
rtp-llm/
├── src/
│   ├── agents.py              # VoiceAgent orchestration
│   ├── rtp_server.py         # RTP protocol handling
│   ├── audio_logger.py       # Audio backup/logging
│   ├── providers/            # STT/TTS provider implementations
│   │   ├── base.py          # Provider base classes
│   │   ├── openai.py        # OpenAI TTS provider
│   │   ├── gemini.py        # Gemini STT provider
│   │   └── alpaca.py        # Alpaca providers
│   ├── vad/                 # Voice Activity Detection
│   │   ├── webrtc.py        # WebRTC VAD implementation
│   │   └── silero.py        # Silero VAD implementation
│   ├── buffer/              # Audio buffering
│   ├── flow/                # Conversation flow management
│   ├── history/             # Chat history management
│   └── utils/               # Audio/RTP utilities
├── tests/                   # Test suite
├── example.py              # Basic usage example
├── singleton_api.py        # RESTful API server
└── requirements.txt        # Dependencies
```

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Test specific providers
python tests/test_providers.py

# Test VAD functionality
python tests/test_vad.py

# Run the example
python example.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `/tests` directory for usage examples
- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Examples**: See `example.py` for a complete working example


