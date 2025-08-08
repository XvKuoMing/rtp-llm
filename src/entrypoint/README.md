# RTP-LLM Entrypoints

This module contains the main entry points and API components for the RTP-LLM application, organized in a modular architecture for better maintainability and scalability.

## 📁 Project Structure

```
src/entrypoint/
├── README.md              # This file
├── rtllm.py              # Main entry point and FastAPI application
├── server_manager.py     # Server lifecycle management
├── models/               # Pydantic models and configuration classes
│   ├── __init__.py      # Models module initialization
│   ├── audio.py         # Audio-related models
│   ├── config.py        # Configuration models
│   ├── server.py        # Server configuration models
│   └── responses.py     # API response models
├── api/                  # API routers organized by functionality
│   ├── __init__.py      # API module initialization
│   ├── audio_router.py  # Audio file management endpoints
│   ├── config_router.py # Configuration management endpoints
│   └── server_router.py # Server management endpoints
└── utils/               # Utility modules
    ├── audio_logs.py    # Audio logging utilities
    └── port_manager.py  # Port allocation and management
```

## 🚀 Quick Start

### Running the Application

```bash
# Using the installed CLI command
rtllm

# Or using Python module directly
python -m src.entrypoint.rtllm

# With custom configuration
rtllm \
    --host 0.0.0.0 \
    --port 8000 \
    --start-port 10000 \
    --end-port 20000 \
    --providers-config-path ./providers.json \
    --redis-enabled \
    --redis-host localhost \
    --redis-port 6379
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | str | `0.0.0.0` | Host address for the FastAPI server |
| `--port` | int | `8000` | Port for the FastAPI server |
| `--start-port` | int | `10000` | Start of port range for RTP/WebSocket servers |
| `--end-port` | int | `20000` | End of port range for RTP/WebSocket servers |
| `--debug` | flag | `False` | Enable debug logging |
| `--providers-config-path` | str | `./providers.json` | Path to providers configuration file |
| `--redis-enabled` | flag | `False` | Enable Redis caching |
| `--redis-host` | str | `localhost` | Redis server host |
| `--redis-port` | int | `6379` | Redis server port |
| `--redis-db` | int | `0` | Redis database number |
| `--redis-password` | str | `None` | Redis password |
| `--redis-ttl-seconds` | int | `None` | Redis TTL in seconds |

## 📋 Core Components

### 1. Models (`models/`)

Contains all Pydantic models for request/response validation and configuration, organized in separate modules:

#### Configuration Models (`models/config.py`)
- **`HostServerConfig`**: FastAPI server configuration
- **`RedisConfig`**: Redis connection settings
- **`ReusableComponents`**: Shared component configuration
- **`UniAdapterConfig`**: Adapter configuration (RTP/WebSocket)
- **`UniVadConfig`**: Voice Activity Detection configuration

#### Server Models (`models/server.py`)
- **`ServerConfig`**: Complete server instance configuration
- **`RunParams`**: Runtime parameters for server execution

#### Response Models (`models/responses.py`)
- **`Response`**: Standard API response format
- **`StartServerResponse`**: Server start response with connection details

#### Audio Models (`models/audio.py`)
- **`AudioFileInfo`**: Audio file metadata and information
- **`AudioListResponse`**: Response model for audio file listings

### 2. Server Manager (`server_manager.py`)

The `ServerManager` class handles the complete lifecycle of voice servers:

#### Key Responsibilities
- **Provider Management**: Load and manage STT/TTS providers
- **Server Lifecycle**: Create, start, stop, and manage server instances
- **Resource Management**: Port allocation and Redis connections
- **Agent Configuration**: Configure voice agents with providers and history

#### Key Methods
```python
# Server lifecycle
start_server(server_config: ServerConfig) -> Tuple[str, int]
stop_server(uid: Union[str, int])
run_server(run_params: RunParams)

# Configuration management
update_agent(uid, system_prompt, tts_config, stt_config)
update_providers_config(providers_config: Dict[str, Any])

# Resource access
get_agent(chat_limit: int) -> Optional[VoiceAgent]
```

### 3. API Routers (`api/`)

#### Server Router (`api/server_router.py`)
Handles server lifecycle operations:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/server/start` | POST | Start a new server instance |
| `/server/run` | POST | Run server with specific parameters |
| `/server/update_agent` | POST | Update agent configuration |
| `/server/stop` | POST | Stop a server instance |

#### Config Router (`api/config_router.py`)
Manages provider configurations:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config/providers` | GET | Get current providers configuration |
| `/config/providers` | POST | Update providers configuration |

#### Audio Router (`api/audio_router.py`)
Manages audio file operations:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/audio/` | GET | List available audio files with filtering/pagination |
| `/audio/{filename}` | GET | Download specific audio file |
| `/audio/{filename}/info` | GET | Get metadata for a specific audio file |
| `/audio/{filename}` | DELETE | Delete specific audio file |

### 4. Main Entry Point (`rtllm.py`)

The main FastAPI application that:
- Sets up CORS middleware
- Includes API routers
- Handles command-line argument parsing
- Initializes the ServerManager
- Connects components together

## 🔧 Configuration

### Providers Configuration (`providers.json`)

The providers configuration file defines the available STT (Speech-to-Text) and TTS (Text-to-Speech) providers. Each provider corresponds to a provider class in the codebase and uses the `name` field to map to the correct implementation.

Example configuration file:

```json
{
  "stt_providers": [
    {
      "name": "gemini",
      "api_key": "YOUR_GEMINI_API_KEY",
      "base_url": "https://generativelanguage.googleapis.com/v1beta",
      "model": "gemini-2.0-flash-exp",
      "system_prompt": "You are a helpful assistant.",
      "gen_config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
      }
    },
    {
      "name": "openai",
      "api_key": "YOUR_OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1",
      "stt_model": "gpt-4o-mini-audio-preview",
      "tts_model": "gpt-4o-mini-tts",
      "system_prompt": "You are a helpful assistant.",
      "tts_voice": "alloy",
      "stt_gen_config": {
        "temperature": 0.3,
        "top_p": 0.8
      }
    }
  ],
  "tts_providers": [
    {
      "name": "openai",
      "api_key": "YOUR_OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1",
      "tts_model": "gpt-4o-mini-tts",
      "tts_voice": "alloy",
      "tts_gen_config": {
        "speed": 1.0,
        "instructions": "Speak clearly and naturally."
      }
    }
  ]
}
```

#### Available Providers

- **`gemini`**: Google Gemini STT provider
- **`openai`**: OpenAI providers for both STT and TTS 
- **`astllm`**: AST LLM provider (combines ASR + LLM for STT, supports TTS)

### Server Configuration Example

```python
server_config = ServerConfig(
    uid="unique-server-id",
    sample_rate=16000,
    adapter=UniAdapterConfig(
        adapter_type="websocket",
        target_codec="pcm"
    ),
    vad=UniVadConfig(
        vad_type="webrtc",
        min_speech_duration_ms=500,
        config={"aggressiveness": 2}
    ),
    max_wait_time=10,
    chat_limit=10
)
```

## 🌐 API Usage Examples (with example outputs)

### Start Server

Request:
```bash
curl -s -X POST "http://localhost:8000/server/start" \
     -H "Content-Type: application/json" \
     -d '{
       "uid": "test-server",
       "sample_rate": 16000,
       "adapter": {"adapter_type": "websocket", "target_codec": "pcm"},
       "vad": {"vad_type": "webrtc", "min_speech_duration_ms": 500, "config": {"aggressiveness": 2}}
     }'
```
Example success response:
```json
{
  "success": true,
  "message": null,
  "host_ip": "192.168.1.10",
  "host_port": 14532
}
```
Example error (conflict):
```json
{ "detail": "Server with uid test-server already exists" }
```

### Run Server

Request:
```bash
curl -s -X POST "http://localhost:8000/server/run" \
     -H "Content-Type: application/json" \
     -d '{
       "uid": "test-server",
       "first_message": "Hello! How can I help you?",
       "allow_interruptions": true,
       "system_prompt": "You are a helpful assistant.",
       "tts_gen_config": {"voice": "default"},
       "stt_gen_config": {"language": "en"},
       "tts_volume": 1.0
     }'
```
Example success response:
```json
{ "success": true, "message": null }
```
Example error (conflict):
```json
{ "detail": "Server with uid test-server is already running" }
```

### Update Agent

Request body model: `UpdateAgentRequest`

Request:
```bash
curl -s -X POST "http://localhost:8000/server/update_agent" \
     -H "Content-Type: application/json" \
     -d '{
       "uid": "test-server",
       "system_prompt": "You are a helpful assistant.",
       "tts_gen_config": {"voice": "alloy"},
       "stt_gen_config": {"language": "en"}
     }'
```
Example success response:
```json
{ "success": true, "message": null }
```
Example error (not found):
```json
{ "detail": "Server with uid test-server not found" }
```

### Stop Server

Request body model: `StopServerRequest`

Request:
```bash
curl -s -X POST "http://localhost:8000/server/stop" \
     -H "Content-Type: application/json" \
     -d '{"uid": "test-server"}'
```
Example success response:
```json
{ "success": true, "message": null }
```
Example error (not found):
```json
{ "detail": "Server with uid test-server not found" }
```

### Get Providers Configuration

Request:
```bash
curl -s "http://localhost:8000/config/providers"
```
Example response:
```json
{
  "stt_providers": [
    {"name": "gemini", "model": "gemini-2.0-flash-exp", "base_url": "https://..."}
  ],
  "tts_providers": [
    {"name": "openai", "tts_model": "gpt-4o-mini-tts", "base_url": "https://..."}
  ]
}
```

### Update Providers Configuration

Request:
```bash
curl -s -X POST "http://localhost:8000/config/providers" \
     -H "Content-Type: application/json" \
     -d '{
       "stt_providers": [{"name": "gemini", "api_key": "..."}],
       "tts_providers": [{"name": "openai", "api_key": "..."}]
     }'
```
Example success response:
```json
{ "success": true, "message": null }
```
Example error (misconfiguration):
```json
{ "detail": "Failed to create TTS provider 'openai': ..." }
```

### List Audio Files

Request:
```bash
curl -s "http://localhost:8000/audio?page=1&page_size=2&sort_by=timestamp_desc"
```
Example response:
```json
{
  "audio_files": [
    {
      "filename": "test-server_conversation_1736000000.0.wav",
      "uid": "test-server",
      "conversation_timestamp": 1736000000.0,
      "file_size": 12345,
      "duration_seconds": 1.23,
      "sample_rate": 16000,
      "channels": 1,
      "created_date": "2024-12-05T12:00:00",
      "file_path": "/abs/path/audio_logs/test-server_conversation_1736000000.0.wav"
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 2,
  "total_pages": 1
}
```
Example error (bad date):
```json
{ "detail": "Invalid date_from format. Use ISO format: 2024-01-01T00:00:00" }
```

### Get Audio Metadata

Request:
```bash
curl -s "http://localhost:8000/audio/test-server_conversation_1736000000.0.wav/info"
```
Example response:
```json
{
  "filename": "test-server_conversation_1736000000.0.wav",
  "uid": "test-server",
  "conversation_timestamp": 1736000000.0,
  "file_size": 12345,
  "duration_seconds": 1.23,
  "sample_rate": 16000,
  "channels": 1,
  "created_date": "2024-12-05T12:00:00",
  "file_path": "/abs/path/audio_logs/test-server_conversation_1736000000.0.wav"
}
```
Example error (not found):
```json
{ "detail": "Audio file not found" }
```

### Download Audio

Request:
```bash
curl -sOJ "http://localhost:8000/audio/test-server_conversation_1736000000.0.wav"
```
Response: returns the WAV file as attachment.

### Delete Audio

Request:
```bash
curl -s -X DELETE "http://localhost:8000/audio/test-server_conversation_1736000000.0.wav"
```
Example success response:
```json
{ "message": "Audio file test-server_conversation_1736000000.0.wav deleted successfully" }
```
Example error (conflict):
```json
{ "detail": "Cannot delete audio file: Server for UID test-server is currently running" }
```

## 🔍 Health Check

The application provides a simple health check endpoint:

```bash
curl http://localhost:8000/ping
# Response: {"message": "pong"}
```

## 🐛 Debugging

Enable debug mode for detailed logging:

```bash
python -m src.entrypoint.rtllm --debug
```

## 🏗️ Architecture Benefits

### Separation of Concerns
- **Models**: Pure data validation and serialization
- **Server Manager**: Business logic and resource management
- **API Routers**: HTTP request/response handling
- **Main Entry**: Application bootstrap and configuration

### Maintainability
- Each component can be modified independently
- Clear dependencies and interfaces
- Easy to test individual components
- Simple to extend with new functionality

### Scalability
- Modular design allows for easy horizontal scaling
- Router-based API organization supports microservice patterns
- Pluggable provider system for different AI services
- Configurable resource management

## 📝 Development Notes

- All async operations are handled properly
- Error handling includes proper HTTP status codes and unified JSON error body
- CORS is configured for cross-origin requests
- Redis integration is optional and gracefully degrades
- Port management prevents conflicts between server instances

### Error Handling

The API maps domain errors to HTTP status codes:

- Validation errors → 400
- Not-found errors → 404
- Conflicts (e.g. server already running) → 409
- Misconfiguration (e.g. invalid providers) → 422
- Unhandled domain errors → 500

## 🔮 Future Enhancements

- Add authentication and authorization
- Implement server health monitoring
- Add metrics and logging endpoints
- Support for server clustering
- WebSocket management dashboard
