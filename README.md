# RTP-LLM: Real-Time Voice Agent Framework

A framework for building real-time voice agents that communicate over RTP/UDP protocols.

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Running the Singleton API

The `serve-rtp` command starts a FastAPI server that provides REST endpoints to control RTP voice sessions.

#### Basic Usage

```bash
# Start with Gemini STT and OpenAI TTS
serve-rtp \
  --stt-providers "gemini" \
  --tts-providers "openai" \
  --gemini-stt-api-key "your_gemini_api_key" \
  --openai-tts-api-key "your_openai_api_key"
```

#### Multiple Provider Setup with Fallback

```bash
# Use Gemini as primary STT with OpenAI fallback, and OpenAI for TTS
serve-rtp \
  --host "0.0.0.0" \
  --port 8000 \
  --stt-providers "gemini;openai" \
  --tts-providers "openai" \
  --gemini-stt-api-key "your_gemini_api_key" \
  --openai-stt-api-key "your_openai_api_key" \
  --openai-tts-api-key "your_openai_api_key" \
  --vad "webrtc" \
  --system-prompt "You are a helpful assistant."
```

#### AST LLM Provider Example

```bash
# Use AST (Whisper) for STT and OpenAI-compatible endpoint for TTS
serve-rtp \
  --stt-providers "ast_llm" \
  --tts-providers "ast_llm" \
  --ast-api-key "your_ast_api_key" \
  --llm-api-key "your_llm_api_key" \
  --tts-api-key "your_tts_api_key"
```

### API Endpoints

Once the server is running (default: http://localhost:8000), you can use these endpoints:

#### Check Status
```bash
curl http://localhost:8000/status
```

#### Start RTP Session
```bash
curl -X POST http://localhost:8000/start \
  -H "Content-Type: application/json" \
  -d '{
    "peer_ip": "192.168.1.100",
    "peer_port": 5004,
    "target_sample_rate": 24000,
    "first_message": "Hello! How can I help you today?",
    "allow_interruptions": true
  }'
```

#### Stop RTP Session
```bash
curl -X POST http://localhost:8000/stop \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Server host | localhost |
| `--port` | Server port | 8000 |
| `--stt-providers` | STT providers (semicolon-separated) | Required |
| `--tts-providers` | TTS providers (semicolon-separated) | Required |
| `--vad` | VAD type (webrtc/silero) | webrtc |
| `--system-prompt` | System prompt for the agent | "You are a helpful assistant." |
| `--max-wait-time` | Max wait time for response (seconds) | 5 |
| `--chat-limit` | Chat history limit | 10 |

Run `serve-rtp --help` to see all available options.


## example arguments for singleton

```
# Server Configuration
HOST=localhost
PORT=8000
DEBUG=false

# Core Configuration
SYSTEM_PROMPT="You are a helpful AI assistant."
VAD=webrtc
MAX_WAIT_TIME=5
CHAT_LIMIT=10

# Provider Configuration (semicolon-separated for multiple providers)
STT_PROVIDERS=gemini;openai
TTS_PROVIDERS=openai

# Gemini STT Configuration
GEMINI_STT_API_KEY=your_gemini_api_key_here
GEMINI_STT_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_STT_MODEL=gemini-2.0-flash

# OpenAI STT Configuration
OPENAI_STT_API_KEY=your_openai_api_key_here
OPENAI_STT_BASE_URL=https://api.openai.com/v1
OPENAI_STT_MODEL=gpt-4o-mini-audio-preview

# OpenAI TTS Configuration
OPENAI_TTS_API_KEY=your_openai_tts_api_key_here
OPENAI_TTS_BASE_URL=https://api.openai.com/v1
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_PCM_RESPONSE_FORMAT=pcm
OPENAI_TTS_RESPONSE_SAMPLE_RATE=24000

# AST LLM STT Configuration (for Whisper + OpenAI-compatible LLM)
AST_API_KEY=your_ast_api_key_here
AST_BASE_URL=https://api.openai.com/v1
AST_MODEL=openai/whisper-large-v3-turbo
AST_LANGUAGE=en
LLM_MODEL=gpt-4o-mini-audio-preview
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.openai.com/v1

# AST LLM TTS Configuration
TTS_API_KEY=your_tts_api_key_here
TTS_BASE_URL=https://api.openai.com/v1
TTS_MODEL=gpt-4o-mini-tts
TTS_PCM_RESPONSE_FORMAT=pcm
TTS_RESPONSE_SAMPLE_RATE=24000
```


