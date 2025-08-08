## RTP‑LLM: Real‑time voice agents (library + full backend)

RTP‑LLM gives you two things:

- Library building blocks in `src/rtp_llm` to build real‑time voice agents (receive audio, detect speech, transcribe, generate replies, speak back).
- A ready‑to‑run backend in `src/entrypoint` (FastAPI + CLI) that manages agent servers over HTTP.

Use the library if you embed it into your own runtime, or the backend if you want a simple service you can deploy and call.

### Library overview (`src/rtp_llm`)

- **`Server`**: Orchestrates the loop. Receives PCM16 frames from an `Adapter`, detects speech via `VAD`, decides when to answer via a `FlowManager`, calls the `VoiceAgent` for STT/LLM, and streams TTS back. Optional audio caching and callbacks.
- **Adapters (`adapters/`)**
  - `RTPAdapter`: UDP RTP in/out. Supports PCM/μ‑law/A‑law/Opus (requires `libopus`).
  - `WebSocketAdapter`: Binary frames carrying raw PCM16. Includes an in‑proc WS server.
- **Providers (`providers/`)**
  - `OpenAIProvider`: STT + TTS (gpt-4o‑mini‑audio‑preview / gpt‑4o‑mini‑tts).
  - `GeminiProvider`: STT via Google GenAI.
  - `AstLLmProvider`: AST (audio → text) + LLM on top of OpenAI APIs.
- **VAD (`vad/`)**
  - `WebRTCVAD`: Classic WebRTC VAD (fast, no model).
  - `SileroVAD`: Neural VAD (requires `silero-vad`, pulls `torch`).
- **Flow (`flow/`)**
  - `CopyFlowManager`: Triggers an answer when speech → silence is detected.
- **History (`history/`)**
  - `ChatHistoryLimiter`: Keeps N formatted messages for the provider.
- **Buffer (`buffer/`)**
  - `ArrayBuffer`: Simple PCM16 byte buffer.
- **Callbacks (`callbacks/`)**
  - `NullCallback`, `RestCallback` (webhooks for events and post‑actions).
- **Cache (`cache/`)**
  - `NullAudioCache`, `InMemoryAudioCache`, `RedisAudioCache` (chunk‑level TTS caching).
- **Utils**: Audio conversion and resampling (`utils/audio_processing.py`), mixed WAV logging (`audio_logger.py`).

### Minimal programmatic usage

```python
import asyncio
from rtp_llm import Server
from rtp_llm.adapters import WebSocketAdapter
from rtp_llm.vad import WebRTCVAD
from rtp_llm.flow import CopyFlowManager
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.providers.alpaca import OpenAIProvider  # OpenAI STT+TTS
from rtp_llm.agents import VoiceAgent

async def main():
    sample_rate = 16000

    adapter = WebSocketAdapter(host="0.0.0.0", port=8765, sample_rate=sample_rate) # start WS server to receive/send PCM16

    vad = WebRTCVAD(sample_rate=sample_rate, aggressiveness=3, min_speech_duration_ms=300)
    flow = CopyFlowManager()
    buffer = ArrayBuffer()

    stt_tts = OpenAIProvider(
        api_key="YOUR_OPENAI_API_KEY",
        stt_model="gpt-4o-mini-audio-preview",
        tts_model="gpt-4o-mini-tts",
        tts_voice="alloy",
    )
    history = ChatHistoryLimiter(limit=10)
    agent = VoiceAgent(stt_provider=stt_tts, tts_provider=stt_tts, history_manager=history)

    server = Server(
        adapter=adapter,
        audio_buffer=buffer,
        flow_manager=flow,
        vad=vad,
        agent=agent,
        max_wait_time=10,
    )

    await server.run(first_message="Hello! How can I help you?")

asyncio.run(main())
```

Notes:
- The server expects PCM16 mono frames at the adapter sample rate. `WebSocketAdapter` uses binary messages as raw PCM16.
- TTS audio is resampled to the adapter rate automatically and volume can be adjusted at runtime via `volume` in `Server.run`.
- Mixed audio logs are written into `audio_logs/` as WAV.

### Full backend (FastAPI + CLI)

If you want a ready‑made API to manage voice servers (start/stop/update, list/download audio logs), use the entrypoint app:

```bash
rtllm --host 0.0.0.0 --port 8000 --providers-config-path ./providers.json
```

See detailed API and examples in `src/entrypoint/README.md`.

### Providers configuration

`providers.json` defines the available STT/TTS providers. An example is in `examples/providers.json`.

Place it anywhere and point the backend with `--providers-config-path` (or env `RTLLM_PROVIDERS_CONFIG_PATH`).

## Docker (uv‑based)

This repo includes a `Dockerfile` that uses uv to create a virtualenv and run the backend.

### Build

```bash
docker build -t rtp-llm:latest .
```

### Run (plain docker)

```bash
docker run --rm -p 8000:8000 \
  -p 10000-10100:10000-10100/udp \
  -p 10000-10100:10000-10100 \
  -v %cd%/audio_logs:/app/audio_logs \
  -v %cd%/examples/providers.json:/data/providers.json:ro \
  -e RTLLM_PROVIDERS_CONFIG_PATH=/data/providers.json \
  --name rtp-llm rtp-llm:latest \
  --host 0.0.0.0 --port 8000 --start-port 10000 --end-port 10100
```

Port notes:
- Map `8000/tcp` for the REST API.
- Map a small range for agent ports: `10000-10100` on TCP (WebSocket) and UDP (RTP). Adjust the range with `RTLLM_START_PORT`/`RTLLM_END_PORT`.

### Docker Compose

The Dockerfile now sets `ENTRYPOINT ["rtllm"]`, so in compose you can pass only args in `command:`. Make sure the mapped port range matches the runtime range.

Example with a narrowed port window (10050–10080) and args-only command:

```yaml
services:
  rtp-llm:
    build: .
    container_name: rtp-llm
    environment:
      - RTLLM_HOST=0.0.0.0
      - RTLLM_PORT=8000
      - RTLLM_START_PORT=10050
      - RTLLM_END_PORT=10080
      - RTLLM_PROVIDERS_CONFIG_PATH=/data/providers.json
      # optional Redis
      - RTLLM_REDIS_ENABLED=true
      - RTLLM_REDIS_HOST=redis
      - RTLLM_REDIS_PORT=6379
    # Only args here because ENTRYPOINT is already `rtllm`
    command: ["--host", "0.0.0.0", "--port", "8000", "--start-port", "10050", "--end-port", "10080"]
    ports:
      - "8000:8000"
      - "10050-10080:10050-10080/udp"   # RTP (UDP)
      - "10050-10080:10050-10080"       # WebSocket (TCP)
    volumes:
      - ./audio_logs:/app/audio_logs
      - ./examples/providers.json:/data/providers.json:ro
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

Run:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000` (`/ping`, `/server/*`, `/config/*`, `/audio/*`).

Notes on ports:
- Align `RTLLM_START_PORT`/`RTLLM_END_PORT` with the compose `ports` mappings.
- If you change the range, update both the env vars (or CLI args) and the `ports:` section accordingly.

### Configuration: environment variables or CLI args

The server reads configuration from both environment variables and CLI arguments:

- Environment variables use the `RTLLM_` prefix (from `src/entrypoint/settings.py`). Common ones:
  - `RTLLM_HOST`, `RTLLM_PORT`
  - `RTLLM_START_PORT`, `RTLLM_END_PORT`
  - `RTLLM_DEBUG`
  - `RTLLM_PROVIDERS_CONFIG_PATH`
  - `RTLLM_REDIS_ENABLED`, `RTLLM_REDIS_HOST`, `RTLLM_REDIS_PORT`, `RTLLM_REDIS_DB`, `RTLLM_REDIS_PASSWORD`, `RTLLM_REDIS_TTL_SECONDS`
  - `RTLLM_MAX_CONCURRENT_FILES`
- CLI arguments (from `src/entrypoint/rtllm.py`) mirror the main options:
  - `--host`, `--port`
  - `--start-port`, `--end-port`
  - `--debug`
  - `--providers-config-path`
  - `--redis-enabled`, `--redis-host`, `--redis-port`, `--redis-db`, `--redis-password`, `--redis-ttl-seconds`

Precedence (as implemented in `rtllm.py`):
- CLI arguments override environment values for non-None options.
- Boolean flags like `--redis-enabled` are always considered (they default to False unless passed).
- If neither env nor CLI provides a value, internal defaults from `AppSettings` apply.

Using a `.env` file with docker compose:

1) Create `.env` in the project root:

```env
RTLLM_HOST=0.0.0.0
RTLLM_PORT=8000
RTLLM_START_PORT=10050
RTLLM_END_PORT=10080
RTLLM_PROVIDERS_CONFIG_PATH=/data/providers.json
```

2) Reference it in compose (either via implicit root `.env` or `env_file:`):

```yaml
services:
  rtp-llm:
    build: .
    env_file: [.env]
    command: ["--start-port", "10050", "--end-port", "10080"]  # optional CLI overrides
    ports:
      - "${RTLLM_PORT}:${RTLLM_PORT}"
      - "${RTLLM_START_PORT}-${RTLLM_END_PORT}:${RTLLM_START_PORT}-${RTLLM_END_PORT}/udp"
      - "${RTLLM_START_PORT}-${RTLLM_END_PORT}:${RTLLM_START_PORT}-${RTLLM_END_PORT}"
```

Alternatively, skip the `.env` and pass everything via `command:` only:

```yaml
services:
  rtp-llm:
    build: .
    command: ["--host", "0.0.0.0", "--port", "8000", "--start-port", "10050", "--end-port", "10080"]
    ports:
      - "8000:8000"
      - "10050-10080:10050-10080/udp"
      - "10050-10080:10050-10080"
```

Note: The container `EXPOSE` lines are advisory. Always publish the same port that the server is configured to use (via env or CLI). If you change the default port from 8000, update both the `command:` and `ports:` mappings accordingly.

### CLI arguments (full list)

All runtime options can be provided via CLI args (shown here) and/or environment variables (prefixed with `RTLLM_`). CLI non-None values override env.

| Argument | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `--host` | str | `0.0.0.0` | `RTLLM_HOST` | API bind host |
| `--port` | int | `8000` | `RTLLM_PORT` | API bind port |
| `--start-port` | int | `10000` | `RTLLM_START_PORT` | Start of dynamic port range used by agent servers (TCP/UDP) |
| `--end-port` | int | `20000` | `RTLLM_END_PORT` | End of dynamic port range used by agent servers (TCP/UDP) |
| `--debug` | flag | `False` | `RTLLM_DEBUG` | Enable debug logging (affects app logger; not the `log_level` below) |
| `--providers-config-path` | str | see note | `RTLLM_PROVIDERS_CONFIG_PATH` | Path to `providers.json`. If absent, a copy from `examples/` may be created under XDG config or `~/.config/rtllm/` |
| `--redis-enabled` | flag | `False` | `RTLLM_REDIS_ENABLED` | Enable Redis-backed TTS audio cache |
| `--redis-host` | str | `localhost` | `RTLLM_REDIS_HOST` | Redis host |
| `--redis-port` | int | `6379` | `RTLLM_REDIS_PORT` | Redis port |
| `--redis-db` | int | `0` | `RTLLM_REDIS_DB` | Redis DB index |
| `--redis-password` | str | `None` | `RTLLM_REDIS_PASSWORD` | Redis password |
| `--redis-ttl-seconds` | int | `None` | `RTLLM_REDIS_TTL_SECONDS` | TTL for cached audio chunks |
| `--log-level` | str | `info` | — | Uvicorn/app log level: `critical` `error` `warning` `info` `debug` `trace` |
| `--cors-allow-origins` | list[str] | `["*"]` | `RTLLM_CORS_ALLOW_ORIGINS` | Allowed origins |
| `--cors-allow-methods` | list[str] | `["*"]` | `RTLLM_CORS_ALLOW_METHODS` | Allowed methods |
| `--cors-allow-headers` | list[str] | `["*"]` | `RTLLM_CORS_ALLOW_HEADERS` | Allowed headers |
| `--cors-allow-credentials` / `--no-cors-allow-credentials` | flag | `True` | `RTLLM_CORS_ALLOW_CREDENTIALS` | Allow credentials in CORS |
| `--max-concurrent-files` | int | `50` | `RTLLM_MAX_CONCURRENT_FILES` | Concurrency limit for audio file metadata parsing |

Notes:
- Defaults are sourced from `AppSettings` unless a CLI value is provided (then CLI wins). Some CLI flags default to `None` to allow env/defaults to apply.
- `providers_config_path` default follows XDG (`$XDG_CONFIG_HOME/rtllm/providers.json`) or `~/.config/rtllm/providers.json`.

### Networking tip: host network

- On Linux, you can run with the host network to avoid per-port publishing:

```bash
docker run --rm --network host \
  -v $PWD/audio_logs:/app/audio_logs \
  -v $PWD/examples/providers.json:/data/providers.json:ro \
  -e RTLLM_PROVIDERS_CONFIG_PATH=/data/providers.json \
  rtp-llm:latest --host 0.0.0.0 --port 8000 --start-port 10050 --end-port 10080
```

Compose variant:

```yaml
services:
  rtp-llm:
    image: rtp-llm:latest
    network_mode: host   # Linux only
    command: ["--host", "0.0.0.0", "--port", "8000", "--start-port", "10050", "--end-port", "10080"]
```

Notes:
- With host networking, you generally do not specify `ports:` mappings.
- Host network mode is fully supported on Linux. On macOS/Windows Docker Desktop, `network_mode: host` is not available for Linux containers.

### Using a registry (push, pull, run)

You can push the image to Docker Hub or GHCR. Replace placeholders with your names.

Docker Hub example:

```bash
# Build local image
docker build -t rtp-llm:latest .

# Tag for Docker Hub
docker tag rtp-llm:latest docker.io/<your-dockerhub-username>/rtp-llm:latest

# Login and push
docker login --username <your-dockerhub-username>
docker push docker.io/<your-dockerhub-username>/rtp-llm:latest
```

GitHub Container Registry (GHCR) example:

```bash
# Build local image
docker build -t rtp-llm:latest .

# Tag for GHCR (use your GitHub username or org)
docker tag rtp-llm:latest ghcr.io/<your-github-username-or-org>/rtp-llm:latest

# Login and push (requires a PAT with package:write)
echo <YOUR_GITHUB_PAT> | docker login ghcr.io -u <your-github-username> --password-stdin
docker push ghcr.io/<your-github-username-or-org>/rtp-llm:latest
```

Then pull and run via docker compose by referencing the pushed image:

```yaml
services:
  rtp-llm:
    image: ghcr.io/<your-github-username-or-org>/rtp-llm:latest  # or docker.io/<user>/rtp-llm:latest
    environment:
      - RTLLM_HOST=0.0.0.0
      - RTLLM_PORT=8000
      - RTLLM_START_PORT=10050
      - RTLLM_END_PORT=10080
      - RTLLM_PROVIDERS_CONFIG_PATH=/data/providers.json
    command: ["--port", "8000", "--start-port", "10050", "--end-port", "10080"]
    ports:
      - "8000:8000"
      - "10050-10080:10050-10080/udp"
      - "10050-10080:10050-10080"
    volumes:
      - ./audio_logs:/app/audio_logs
      - ./examples/providers.json:/data/providers.json:ro
```

Pull and start:

```bash
docker compose pull
docker compose up -d
```

## System requirements and tips

- Python 3.11+ (Dockerfile uses 3.12). If you use `SileroVAD`, ensure `torch` is available (pulled by `silero-vad`).
- For RTP with `opus`, the container installs `libopus`/headers; if you don’t use Opus, you can remove those packages.
- Audio logs are stored in `audio_logs/` (bind‑mount it to persist across containers).
