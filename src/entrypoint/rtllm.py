import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import HostServerConfig, ReusableComponents, RedisConfig
from .server_manager import ServerManager
from .api import server_router, config_router, audio_router
from .settings import AppSettings
from .api.dependencies import set_app_server_manager


logger = logging.getLogger("rtllm")


# FastAPI application with lifespan for DI and graceful shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register exception handlers at startup
    yield
    # On shutdown: gracefully close server manager if present
    try:
        manager = getattr(app.state, "server_manager", None)
        if manager is not None:
            await manager.close()
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)


@app.get("/ping")
async def ping():
    return {"message": "pong"}


# Include routers
app.include_router(server_router)
app.include_router(config_router)
app.include_router(audio_router)


def main():

    parser = argparse.ArgumentParser(description="rtllm cli")
    
    # core configuration
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--start-port", type=int)
    parser.add_argument("--end-port", type=int)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # reusable components
    parser.add_argument("--providers-config-path", help="Providers config path")

    # Redis configuration
    parser.add_argument("--redis-enabled", action="store_true", help="Enable Redis")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument("--redis-password", help="Redis password")
    parser.add_argument("--redis-ttl-seconds", type=int, default=None, help="Redis TTL seconds")

    parser.add_argument("--log-level", default="info", choices=[
        "critical", "error", "warning", "info", "debug", "trace"
    ], help="Logging level for uvicorn and rtllm")

    args = parser.parse_args()

    # Configure logging early
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    logger.info("Starting rtllm server")

    # Load settings from env with CLI overrides
    # Only pass non-None values to allow defaults to take effect
    # For boolean flags, always pass them since they have valid False values
    settings_kwargs = {
        k: v for k, v in {
            'host': args.host,
            'port': args.port,
            'start_port': args.start_port,
            'end_port': args.end_port,
            'providers_config_path': args.providers_config_path,
            'redis_password': args.redis_password,
            'redis_ttl_seconds': args.redis_ttl_seconds,
        }.items() if v is not None
    }
    
    # Always include boolean and defaulted values
    settings_kwargs.update({
        'debug': args.debug,
        'redis_enabled': args.redis_enabled,
        'redis_host': args.redis_host,
        'redis_port': args.redis_port,
        'redis_db': args.redis_db,
    })
    
    settings = AppSettings(**settings_kwargs)

    redis_config = None
    if settings.redis_enabled:
        redis_config = RedisConfig(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            ttl_seconds=settings.redis_ttl_seconds,
        )

    # Initialize server manager
    server_manager = ServerManager(
        host_config=HostServerConfig(
            host=settings.host,
            port=settings.port,
            start_port=settings.start_port,
            end_port=settings.end_port,
            debug=settings.debug,
        ),
        reusable_components=ReusableComponents(
            providers_config_path=settings.providers_config_path,
            redis=redis_config,
        ),
    )

    # Store on app state for DI via dependencies
    set_app_server_manager(app, server_manager)

    # Run the FastAPI application
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level)


if __name__ == "__main__":
    main()