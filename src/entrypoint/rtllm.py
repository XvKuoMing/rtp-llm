import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import HostServerConfig, ReusableComponents, RedisConfig
from .server_manager import ServerManager
from .api import server_router, config_router, audio_router
from .settings import AppSettings
from .utils.logging_utils import sanitize_for_logging, to_json_for_logging
from .api.dependencies import set_app_server_manager
from .exceptions import (
    ValidationError,
    NotFoundError,
    ResourceConflictError,
    ConfigurationError,
    RtllmError,
)


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

# Register exception handlers for domain errors
def _error_body(message: str, code: str | None = None):
    body = {"detail": message}
    if code:
        body["code"] = code
    return body


@app.exception_handler(ValidationError)
async def handle_validation_error(_, exc: ValidationError):
    return JSONResponse(status_code=400, content=_error_body(str(exc), getattr(exc, "code", None)))


@app.exception_handler(NotFoundError)
async def handle_not_found_error(_, exc: NotFoundError):
    return JSONResponse(status_code=404, content=_error_body(str(exc), getattr(exc, "code", None)))


@app.exception_handler(ResourceConflictError)
async def handle_conflict_error(_, exc: ResourceConflictError):
    return JSONResponse(status_code=409, content=_error_body(str(exc), getattr(exc, "code", None)))


@app.exception_handler(ConfigurationError)
async def handle_configuration_error(_, exc: ConfigurationError):
    # Treat misconfiguration as 422 Unprocessable Entity
    return JSONResponse(status_code=422, content=_error_body(str(exc), getattr(exc, "code", None)))


@app.exception_handler(RtllmError)
async def handle_generic_domain_error(_, exc: RtllmError):
    return JSONResponse(status_code=500, content=_error_body(str(exc), getattr(exc, "code", None)))

def _install_cors(app: FastAPI, *, allow_origins, allow_methods, allow_headers, allow_credentials):
    """Install CORS middleware based on provided configuration."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
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
    parser.add_argument("--debug", type=lambda x: x.lower() in ('true', '1', 'yes', 'on'), default=None, help="Enable debug logging (true/false)")
    
    # reusable components
    parser.add_argument("--providers-config-path", help="Providers config path")

    # Redis configuration
    parser.add_argument("--redis-enabled", type=lambda x: x.lower() in ('true', '1', 'yes', 'on'), help="Enable Redis")
    parser.add_argument("--redis-host", help="Redis host")
    parser.add_argument("--redis-port", type=int, help="Redis port")
    parser.add_argument("--redis-db", type=int, help="Redis database")
    parser.add_argument("--redis-password", help="Redis password")
    parser.add_argument("--redis-ttl-seconds", type=int, help="Redis TTL seconds")

    parser.add_argument("--log-level", default="info", choices=[
        "critical", "error", "warning", "info", "debug", "trace"
    ], help="Logging level for uvicorn and rtllm")
    
    # CORS configuration (defaults to *; credentials default to True to match previous behavior)
    parser.add_argument("--cors-allow-origins", nargs="*", default=["*"], help="List of allowed CORS origins (default: *).")
    parser.add_argument("--cors-allow-methods", nargs="*", default=["*"], help="List of allowed CORS methods (default: *).")
    parser.add_argument("--cors-allow-headers", nargs="*", default=["*"], help="List of allowed CORS headers (default: *).")
    cred_group = parser.add_mutually_exclusive_group()
    cred_group.add_argument("--cors-allow-credentials", dest="cors_allow_credentials", action="store_true", help="Enable CORS credentials")
    cred_group.add_argument("--no-cors-allow-credentials", dest="cors_allow_credentials", action="store_false", help="Disable CORS credentials")
    parser.set_defaults(cors_allow_credentials=True)

    # File parsing concurrency (None => use env/defaults)
    parser.add_argument("--max-concurrent-files", type=int, default=None, help="Max concurrent file operations when listing audio (default from env or 50)")

    args = parser.parse_args()

    # Configure logging early
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # Load settings from env with CLI overrides
    # Only pass non-None values to allow defaults to take effect
    settings_kwargs = {
        k: v for k, v in {
            'host': args.host,
            'port': args.port,
            'start_port': args.start_port,
            'end_port': args.end_port,
            'debug': args.debug,
            'providers_config_path': args.providers_config_path,
            'redis_enabled': args.redis_enabled,
            'redis_host': args.redis_host,
            'redis_port': args.redis_port,
            'redis_db': args.redis_db,
            'redis_password': args.redis_password,
            'redis_ttl_seconds': args.redis_ttl_seconds,
        }.items() if v is not None
    }
    
    
    # Optional overrides
    if args.max_concurrent_files is not None:
        settings_kwargs["max_concurrent_files"] = args.max_concurrent_files

    settings = AppSettings(**settings_kwargs)
    # Ensure debug from env/CLI takes effect on logging
    if settings.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Starting rtllm server (log_level=%s, debug=%s)", args.log_level, settings.debug)
    # Log effective application settings (sanitized)
    logger.info("App settings: %s", to_json_for_logging(sanitize_for_logging(settings.model_dump())))

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
        max_concurrent_files=settings.max_concurrent_files,
    )

    # Store on app state for DI via dependencies
    set_app_server_manager(app, server_manager)

    # Install CORS middleware with CLI-provided arguments (default to *)
    _install_cors(
        app,
        allow_origins=args.cors_allow_origins,
        allow_methods=args.cors_allow_methods,
        allow_headers=args.cors_allow_headers,
        allow_credentials=args.cors_allow_credentials,
    )
    logger.info(
        "CORS configured: origins=%s methods=%s headers=%s credentials=%s",
        args.cors_allow_origins,
        args.cors_allow_methods,
        args.cors_allow_headers,
        args.cors_allow_credentials,
    )

    # Run the FastAPI application
    logger.info("Uvicorn starting at http://%s:%s", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level)


if __name__ == "__main__":
    main()