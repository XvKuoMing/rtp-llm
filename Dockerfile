ARG PYTHON_VERSION=3.11

FROM ghcr.io/astral-sh/uv:0.5.12-python${PYTHON_VERSION}-bookworm AS uv

WORKDIR /app

# System deps for audio codecs (opus) and runtime basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and sources
COPY pyproject.toml uv.lock ./
COPY src ./src

# Create and cache virtualenv with uv
RUN uv venv --seed && \
    . .venv/bin/activate && \
    uv sync --frozen --no-dev

# App runtime
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV RTLLM_HOST=0.0.0.0 \
    RTLLM_PORT=8000 \
    RTLLM_START_PORT=10000 \
    RTLLM_END_PORT=20000

# Create writable dirs
RUN mkdir -p /app/audio_logs /data
VOLUME ["/app/audio_logs", "/data"]

# default ports
EXPOSE 8080/tcp
EXPOSE 10000-10010/tcp
EXPOSE 10000-10010/udp

# Run rtllm as the entrypoint so docker-compose `command:` can pass only the args
ENTRYPOINT ["rtllm"]
# Provide sensible defaults; can be overridden by docker run/compose `command:`
CMD ["--host", "0.0.0.0", "--port", "8000"]


