import asyncio
from typing import Optional, Union, Dict, Any, Tuple, List
import json
from pathlib import Path
import os
import glob
import math
import logging
from datetime import datetime

from rtp_llm.providers import MetaProvider
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm import VoiceAgent, Server
from rtp_llm.vad import WebRTCVAD, SileroVAD
from rtp_llm.adapters import RTPAdapter, WebSocketAdapter
from rtp_llm.cache.rredis import RedisAudioCache
from rtp_llm.callbacks import RestCallback
from rtp_llm.audio_logger import AUDIO_LOGS_DIR

from .utils.port_manager import PortManager
from .exceptions import (
    ValidationError,
    NotFoundError,
    ResourceConflictError,
    ConfigurationError,
)
from .utils.audio_logs import get_audio_file_info, parse_audio_filename
from .models import HostServerConfig, ReusableComponents, ServerConfig, RunParams
from .utils.logging_utils import sanitize_for_logging, to_json_for_logging
from .models.audio import AudioFileInfo, AudioListResponse




logger = logging.getLogger(__name__)


class ServerManager:

    def __init__(self, 
                 host_config: HostServerConfig,
                 reusable_components: ReusableComponents,
                 *,
                 max_concurrent_files: int = 50):
        """
        all reusable components are initialized here
        """
        self.host_config = host_config
        self.reusable_components = reusable_components
        self.max_concurrent_files = max_concurrent_files

        self.port_manager = PortManager(
            start_port=host_config.start_port,
            end_port=host_config.end_port,
        )

        self.__providers_config = None
        self.__providers = {
            "stt": [],
            "tts": [],
        }
        self.__redis = None

        self.__servers: Dict[Union[str, int], Server] = {}  # uid -> Server
        self.__running_servers: Dict[Union[str, int], asyncio.Task] = {}  # uid -> asyncio.Task

        self.__lazy_init()

    def __lazy_init(self):
        self.__load_providers()
        self.__load_redis()
    
    def __load_providers(self, config: Optional[Dict[str, Any]] = None):
        target_path = Path(self.reusable_components.providers_config_path)
        if config is None:
            # Try to load from target path; if missing, try bundled example and copy
            if target_path.exists():
                with target_path.open("r", encoding="utf-8") as f:
                    self.__providers_config = json.load(f)
                    logger.debug(
                        "Loaded providers config from %s:\n%s",
                        str(target_path),
                        to_json_for_logging(sanitize_for_logging(self.__providers_config), indent=2),
                    )
            else:
                # Try to locate repo example: ../../examples/providers.json relative to this file
                try_example = (Path(__file__).resolve().parents[2] / "examples" / "providers.json")
                if try_example.exists():
                    # Read from example first, then best-effort persist
                    data = try_example.read_text(encoding="utf-8")
                    try:
                        self.__providers_config = json.loads(data)
                    except Exception:
                        self.__providers_config = None
                    # Best-effort write to target; skip on read-only FS
                    try:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with target_path.open("w", encoding="utf-8") as dst:
                            json.dump(data, dst, ensure_ascii=False, indent=2)
                    except Exception as e:
                        logger.warning(
                            "Could not persist providers config to %s (continuing in-memory only): %s",
                            str(target_path),
                            repr(e),
                        )
                else:
                    # No config available
                    self.__providers_config = None
        else:
            # Config provided: persist and use
            self.__providers_config = config
            # Best-effort persist to disk; skip on failure (e.g., read-only FS)
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with target_path.open("w", encoding="utf-8") as f:
                    json.dump(self.__providers_config, f, ensure_ascii=False, indent=2)
                logger.debug(
                    "Updated providers config at %s:\n%s",
                    str(target_path),
                    to_json_for_logging(sanitize_for_logging(self.__providers_config), indent=2),
                )
            except Exception as e:
                logger.warning(
                    "Could not write providers config to %s (continuing in-memory only): %s. Config: %s",
                    str(target_path),
                    repr(e),
                    to_json_for_logging(sanitize_for_logging(self.__providers_config), indent=2),
                )
        
        if self.__providers_config and "stt_providers" in self.__providers_config:
            stt_instances = []
            for provider_cfg in self.__providers_config["stt_providers"]:
                name = provider_cfg.get("name")
                if not name:
                    raise ConfigurationError("STT provider entry missing 'name'")
                # Work on a shallow copy to avoid mutating source config
                cfg_copy = {k: v for k, v in provider_cfg.items() if k != "name"}
                try:
                    instance = MetaProvider.create_provider_from_config(name, cfg_copy)
                    stt_instances.append(instance)
                    logger.info(
                        "Initialized STT provider: name=%s class=%s",
                        name,
                        instance.__class__.__name__,
                    )
                except Exception as e:
                    raise ConfigurationError(f"Failed to create STT provider '{name}': {e}")
            self.__providers["stt"] = stt_instances
        
        if self.__providers_config and "tts_providers" in self.__providers_config:
            tts_instances = []
            for provider_cfg in self.__providers_config["tts_providers"]:
                name = provider_cfg.get("name")
                if not name:
                    raise ConfigurationError("TTS provider entry missing 'name'")
                cfg_copy = {k: v for k, v in provider_cfg.items() if k != "name"}
                try:
                    instance = MetaProvider.create_provider_from_config(name, cfg_copy)
                    tts_instances.append(instance)
                    logger.info(
                        "Initialized TTS provider: name=%s class=%s",
                        name,
                        instance.__class__.__name__,
                    )
                except Exception as e:
                    raise ConfigurationError(f"Failed to create TTS provider '{name}': {e}")
            self.__providers["tts"] = tts_instances
    
    def __load_redis(self):
        if not self.reusable_components.redis:
            return
        try:
            from redis.asyncio import Redis
            self.__redis = Redis(
                host=self.reusable_components.redis.host,
                port=self.reusable_components.redis.port,
                db=self.reusable_components.redis.db,
                password=self.reusable_components.redis.password,
            )
        except (ModuleNotFoundError, ImportError):
            logger.warning("Redis is not installed, using in-memory cache instead")
            self.__redis = None

    def get_providers_config(self):
        return self.__providers_config

    @property
    def stt_providers(self):
        return self.__providers["stt"]
    
    @property
    def tts_providers(self):
        return self.__providers["tts"]
    
    @property
    def redis(self):
        return self.__redis

    def get_agent(self, chat_limit: int) -> Optional[VoiceAgent]: 
        if not self.__providers["stt"] or not self.__providers["tts"]:
            return None
        
        return VoiceAgent(
            stt_provider=self.stt_providers[0],
            tts_provider=self.tts_providers[0],
            history_manager=ChatHistoryLimiter(
                limit=chat_limit,
            ),
            backup_stt_providers=self.stt_providers[1:] if len(self.stt_providers) > 1 else None,
            backup_tts_providers=self.tts_providers[1:] if len(self.tts_providers) > 1 else None,
        )

    def start_server(self, server_config: ServerConfig) -> Tuple[str, int]:
        # Prevent accidental overwrite of an existing server instance
        if server_config.uid in self.__servers:
            raise ResourceConflictError(f"Server with uid {server_config.uid} already exists")

        try:
            host_port = self.port_manager.get_available_port(server_config.uid)
        except RuntimeError as e:
            raise ResourceConflictError(str(e))
        host_ip = self.port_manager.get_static_host_ip()

        if server_config.adapter.adapter_type == "rtp":
            adapter = RTPAdapter(
                sample_rate=server_config.sample_rate,
                target_codec=server_config.adapter.target_codec,
                host_ip=host_ip,
                host_port=host_port,
                peer_ip=server_config.adapter.peer_ip,
                peer_port=server_config.adapter.peer_port,
            )
        elif server_config.adapter.adapter_type == "websocket":
            adapter = WebSocketAdapter(
                sample_rate=server_config.sample_rate,
                target_codec=server_config.adapter.target_codec,
                host=host_ip,
                port=host_port,
            )
        else:
            raise ValidationError(f"Unsupported adapter type: {server_config.adapter.adapter_type}")
        
        if server_config.vad.vad_type == "webrtc":
            vad = WebRTCVAD(
                sample_rate=server_config.sample_rate,
                min_speech_duration_ms=server_config.vad.min_speech_duration_ms,
                **(server_config.vad.config or {}),
            )
        elif server_config.vad.vad_type == "silero":
            vad = SileroVAD(
                sample_rate=server_config.sample_rate,
                min_speech_duration_ms=server_config.vad.min_speech_duration_ms,
                **(server_config.vad.config or {}),
            )
        else:
            raise ValidationError(f"Unsupported VAD type: {server_config.vad.vad_type}")
        logger.info(
            "Initialized VAD: type=%s min_speech_duration_ms=%s extra=%s",
            server_config.vad.vad_type,
            server_config.vad.min_speech_duration_ms,
            to_json_for_logging(sanitize_for_logging(server_config.vad.config or {})),
        )
        
        agent = self.get_agent(server_config.chat_limit)
        if not agent:
            raise ConfigurationError("No STT/TTS providers configured")
        # Log agent composition
        try:
            stt_cls = agent.stt_provider.__class__.__name__
            tts_cls = agent.tts_provider.__class__.__name__
            logger.info(
                "Initialized Agent: stt=%s tts=%s chat_limit=%s backups(stt=%d, tts=%d)",
                stt_cls,
                tts_cls,
                server_config.chat_limit,
                len(agent.backup_stt_providers),
                len(agent.backup_tts_providers),
            )
        except Exception:
            logger.info("Initialized Agent")
        
        server = Server(
            adapter=adapter,
            audio_buffer=ArrayBuffer(),
            flow_manager=CopyFlowManager(),
            vad=vad,
            agent=agent,
            max_wait_time=server_config.max_wait_time,
            audio_cache=RedisAudioCache(
                redis_client=self.redis,
                ttl_seconds=self.reusable_components.redis.ttl_seconds,
            ) if self.redis else None,
        )

        self.__servers[server_config.uid] = server

        return adapter.host_ip, adapter.host_port

    def stop_server(self, uid: Union[str, int]):
        if uid not in self.__servers:
            raise NotFoundError(f"Server with uid {uid} not found")

        # Cancel running task if exists
        running_task = self.__running_servers.pop(uid, None)
        if running_task and not running_task.done():
            running_task.cancel()
        
        # Close server resources
        try:
            self.__servers[uid].close()
        finally:
            del self.__servers[uid]
            # Release associated port mapping
            self.port_manager.release_uid(uid)

    def run_server(self, run_params: RunParams):
        if run_params.uid not in self.__servers:
            raise NotFoundError(f"Server with uid {run_params.uid} not found")
        if self.is_server_running(run_params.uid):
            raise ResourceConflictError(f"Server with uid {run_params.uid} is already running")
        
        # Create RestCallback if provided
        callback = None
        if run_params.rest_callback:
            callback = RestCallback(
                base_url=run_params.rest_callback.base_url,
                on_response_endpoint=run_params.rest_callback.on_response_endpoint,
                on_start_endpoint=run_params.rest_callback.on_start_endpoint,
                on_error_endpoint=run_params.rest_callback.on_error_endpoint,
                on_finish_endpoint=run_params.rest_callback.on_finish_endpoint,
            )
            logger.info(f"Creating rest callback with params: {str(callback)}")
        
        server = self.__servers[run_params.uid]
        self.__running_servers[run_params.uid] = asyncio.create_task(server.run(
            uid=run_params.uid,
            first_message=run_params.first_message,
            allow_interruptions=run_params.allow_interruptions,
            system_prompt=run_params.system_prompt,
            tts_gen_config=run_params.tts_gen_config,
            stt_gen_config=run_params.stt_gen_config,
            volume=run_params.tts_volume,
            callback=callback,
        ))
        
    def update_agent(self, uid: Union[str, int], system_prompt: str, tts_gen_config: Dict[str, Any], stt_gen_config: Dict[str, Any]):
        if uid not in self.__servers:
            raise NotFoundError(f"Server with uid {uid} not found")
        
        server = self.__servers[uid]
        server.update_agent_config(
            system_prompt=system_prompt,
            tts_gen_config=tts_gen_config,
            stt_gen_config=stt_gen_config,
        )

    def update_providers_config(self, providers_config: Dict[str, Any]):
        self.__load_providers(providers_config)

    def is_server_running(self, uid: Union[str, int]) -> bool:
        """Check if a server is currently running"""
        return uid in self.__running_servers and not self.__running_servers[uid].done()

    async def close(self) -> None:
        """
        Gracefully shutdown all running servers, cancel tasks, close caches and
        release all allocated ports.
        """
        # Cancel running tasks
        for uid, task in list(self.__running_servers.items()):
            if task and not task.done():
                task.cancel()
        # Await task cancellation
        await asyncio.gather(*[t for t in self.__running_servers.values() if t], return_exceptions=True)
        self.__running_servers.clear()

        # Close server instances
        for uid, server in list(self.__servers.items()):
            try:
                server.close()
            except Exception:
                pass
            finally:
                self.port_manager.release_uid(uid)
                self.__servers.pop(uid, None)

        # Close Redis connection if used
        if self.__redis is not None:
            try:
                # redis.asyncio supports await .close() in modern versions
                await self.__redis.close()  # type: ignore[attr-defined]
            except Exception:
                try:
                    # Fallback to connection pool disconnect if available
                    await self.__redis.connection_pool.disconnect()  # type: ignore[attr-defined]
                except Exception:
                    pass
            finally:
                self.__redis = None

        # Finally release any remaining ports
        try:
            self.port_manager.release_all()
        except Exception:
            pass

    async def list_audio_files(
        self,
        uid: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "timestamp_desc"
    ) -> AudioListResponse:
        """
        List audio files with optional filtering and pagination.
        
        Returns metadata about audio files stored on disk.
        """
        # Ensure audio logs directory exists
        if not os.path.exists(AUDIO_LOGS_DIR):
            return AudioListResponse(
                audio_files=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0
            )
        
        # Get all WAV files
        audio_pattern = os.path.join(AUDIO_LOGS_DIR, "*.wav")
        audio_files_paths = await asyncio.to_thread(glob.glob, audio_pattern)
        
        # Parse file information in parallel with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_files)
        
        async def get_file_info_with_semaphore(filepath: str) -> Optional[AudioFileInfo]:
            async with semaphore:
                return await get_audio_file_info(filepath)
        
        # Process all files in parallel
        if audio_files_paths:
            tasks = [get_file_info_with_semaphore(filepath) for filepath in audio_files_paths]
            file_infos = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None values and exceptions
            audio_files_info = []
            for file_info in file_infos:
                if isinstance(file_info, AudioFileInfo):
                    audio_files_info.append(file_info)
                elif isinstance(file_info, Exception):
                    logger.warning(f"Error processing audio file: {file_info}")
        else:
            audio_files_info = []
        
        # Apply filters
        filtered_files = audio_files_info
        
        # Filter by UID
        if uid:
            filtered_files = [f for f in filtered_files if isinstance(f, AudioFileInfo) and f.uid == uid]
        
        # Filter by date range
        if date_from:
            try:
                date_from_ts = datetime.fromisoformat(date_from).timestamp()
                filtered_files = [f for f in filtered_files if isinstance(f, AudioFileInfo) and f.conversation_timestamp >= date_from_ts]
            except ValueError:
                raise ValidationError("Invalid date_from format. Use ISO format: 2024-01-01T00:00:00")
        
        if date_to:
            try:
                date_to_ts = datetime.fromisoformat(date_to).timestamp()
                filtered_files = [f for f in filtered_files if f.conversation_timestamp <= date_to_ts]
            except ValueError:
                raise ValidationError("Invalid date_to format. Use ISO format: 2024-01-01T23:59:59")
        
        # Sort files
        if sort_by == "timestamp_asc":
            filtered_files.sort(key=lambda x: x.conversation_timestamp)
        elif sort_by == "timestamp_desc":
            filtered_files.sort(key=lambda x: x.conversation_timestamp, reverse=True)
        elif sort_by == "duration_asc":
            filtered_files.sort(key=lambda x: x.duration_seconds or 0)
        elif sort_by == "duration_desc":
            filtered_files.sort(key=lambda x: x.duration_seconds or 0, reverse=True)
        elif sort_by == "size_asc":
            filtered_files.sort(key=lambda x: x.file_size)
        elif sort_by == "size_desc":
            filtered_files.sort(key=lambda x: x.file_size, reverse=True)
        else:
            raise ValidationError("Invalid sort_by parameter")
        
        total_count = len(filtered_files)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
        
        # Apply pagination
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_files = filtered_files[start_index:end_index]
        
        return AudioListResponse(
            audio_files=paginated_files,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    async def get_audio_file_metadata(self, filename: str) -> Optional[AudioFileInfo]:
        """Get detailed metadata about a specific audio file."""
        # Validate filename format and prevent directory traversal
        if not filename.endswith('.wav') or '/' in filename or '\\' in filename or '..' in filename:
            raise ValidationError("Invalid filename")
        
        filepath = os.path.join(AUDIO_LOGS_DIR, filename)
        
        if not await asyncio.to_thread(os.path.exists, filepath):
            raise NotFoundError("Audio file not found")
        
        return await get_audio_file_info(filepath)

    async def delete_audio_file(self, filename: str) -> None:
        """
        Delete a specific audio file by filename.
        Raises ValueError if server is running or file is invalid.
        """
        # Validate filename format and prevent directory traversal
        if not filename.endswith('.wav') or '/' in filename or '\\' in filename or '..' in filename:
            raise ValidationError("Invalid filename")
        
        # Parse filename to get UID
        parsed = parse_audio_filename(filename)
        if not parsed:
            raise ValidationError("Invalid audio filename format")
        
        # Check if server is running for this UID
        uid = parsed['uid']
        if self.is_server_running(uid):
            raise ResourceConflictError(f"Cannot delete audio file: Server for UID {uid} is currently running")
        
        filepath = os.path.join(AUDIO_LOGS_DIR, filename)
        
        if not await asyncio.to_thread(os.path.exists, filepath):
            raise NotFoundError("Audio file not found")
        
        await asyncio.to_thread(os.remove, filepath)

    def get_audio_file_path(self, filename: str) -> str:
        """Get the full path to an audio file after validation."""
        # Validate filename format and prevent directory traversal
        if not filename.endswith('.wav') or '/' in filename or '\\' in filename or '..' in filename:
            raise ValidationError("Invalid filename")
        
        # Verify it's actually an audio file we recognize
        parsed = parse_audio_filename(filename)
        if not parsed:
            raise ValidationError("Invalid audio filename format")
        
        return os.path.join(AUDIO_LOGS_DIR, filename)
