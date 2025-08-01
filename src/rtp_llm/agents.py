from .providers import Message, BaseSTTProvider, BaseTTSProvider
from .history import BaseChatHistory
from typing import List, AsyncGenerator, Optional, Any, Dict, Union, Literal, Callable
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VoiceAgent:

    def __init__(self, 
                 stt_provider: BaseSTTProvider,
                 tts_provider: BaseTTSProvider,
                 history_manager: BaseChatHistory,
                 backup_stt_providers: Optional[List[BaseSTTProvider]] = None,
                 backup_tts_providers: Optional[List[BaseTTSProvider]] = None):
        """
        VoiceAgent is a class that represents a voice agent.
        It is responsible for handling the voice input and output, and for managing the chat history.
        It uses the stt_provider and tts_provider to convert between text and voice, and the history_manager to store the chat history.
        It also uses the backup_stt_providers and backup_tts_providers to handle errors in the main providers.
        stt_provider: The primary speech-to-text provider.
        tts_provider: The primary text-to-speech provider.
        history_manager: The chat history manager.
        backup_stt_providers: A list of backup speech-to-text providers; 
        - if the primary provider fails, the agent will switch to the first available backup provider.
        backup_tts_providers: A list of backup text-to-speech providers; 
        - if the primary provider fails, the agent will switch to the first available backup provider.
        """
        self.__stt_provider = stt_provider
        self.__tts_provider = tts_provider
        self.__history_manager = history_manager
        self.backup_stt_providers = backup_stt_providers or []
        self.backup_tts_providers = backup_tts_providers or []

    @property
    def stt_provider(self) -> BaseSTTProvider:
        return self.__stt_provider
    
    @stt_provider.setter
    def stt_provider(self, value: BaseSTTProvider):
        self.__stt_provider = value
    
    @property
    def tts_provider(self) -> BaseTTSProvider:
        return self.__tts_provider
    
    @tts_provider.setter
    def tts_provider(self, value: BaseTTSProvider):
        self.__tts_provider = value
    
    @property
    def history_manager(self) -> BaseChatHistory:
        return self.__history_manager
    

    def update_stt_config(self, config: Dict[str, Any]):
        """
        Update the stt_provider config.
        """
        self.stt_provider.stt_gen_config = self.stt_provider.validate_stt_config(config) or {}
    
    def update_tts_config(self, config: Dict[str, Any]):
        """
        Update the tts_provider config.
        """
        self.tts_provider.tts_gen_config = self.tts_provider.validate_tts_config(config) or {}
    

    async def add_message(self, message: Message, is_user: bool, is_audio: bool = False):
        """
        Add a message to the history.
        """
        role = "user" if is_user else "assistant"
        data_type = "audio" if is_audio else "text"
        await self.history_manager.add_message(Message(role=role, content=message, data_type=data_type))
    

    async def __rotate_provider(self, provider_type: Literal["stt", "tts"]):
        """
        Rotate the provider.
        """
        providers = self.backup_stt_providers if provider_type == "stt" else self.backup_tts_providers
        if not isinstance(providers, list) or len(providers) == 0:
            raise Exception(f"No backup providers for type: {provider_type} exist")
        try:
            if provider_type == "stt":
                self.backup_stt_providers.pop(0)
                self.backup_stt_providers.append(self.__stt_provider)
                new_provider = self.backup_stt_providers[0]
                self.__stt_provider = new_provider
            else:
                self.backup_tts_providers.pop(0)
                self.backup_tts_providers.append(self.__tts_provider)
                new_provider = self.backup_tts_providers[0]
                self.__tts_provider = new_provider
            
            logger.info(f"Rotated {provider_type} provider to {new_provider.__class__.__name__}")
        except IndexError:
            logger.error(f"No backup {provider_type} providers configured")
            raise
        except Exception as e:
            logger.error(f"Error rotating {provider_type} provider: {e}")
            raise
    
    async def rotate_stt_provider(self):
        """
        Rotate the stt_provider.
        """
        await self.__rotate_provider("stt")
    
    async def rotate_tts_provider(self):    
        """
        Rotate the tts_provider.
        """
        await self.__rotate_provider("tts")

    
    async def _stt(self, 
                   audio: bytes, 
                   stream: bool = False,
                   enable_history: bool = True
                   ) -> Optional[Union[str, AsyncGenerator[str, None]]]:
        """
        Convert audio to text using the stt_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        user_message = Message(role="user", content=audio, data_type="audio")
        if enable_history:
            await self.history_manager.add_message(user_message)
        
        messages = await self.history_manager.get_messages(self.stt_provider.format)
        if stream:
            if enable_history:
                return self._stt_stream_with_history(messages)
            else:
                return self.stt_provider.stt_stream(messages)
        else:
            content = await self.stt_provider.stt(messages)
            if enable_history:
                await self.history_manager.add_message(Message(role="assistant", content=content, data_type="text"))
            return content

    async def _stt_stream_with_history(self, messages):
        """
        Stream STT response while accumulating full text for history.
        This yields chunks immediately (low latency) while building complete response.
        """
        accumulated_text = ""
        async for chunk in self.stt_provider.stt_stream(messages):
            accumulated_text += chunk
            yield chunk
        
        await self.history_manager.add_message(
            Message(role="assistant", content=accumulated_text, data_type="text")
        )

    async def _tts(self, 
                   text: str,
                   stream: bool = False, 
                   ) -> Optional[Union[bytes, AsyncGenerator[bytes, None]]]:
        """
        Convert text to audio using the tts_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        if stream:
            return self.tts_provider.tts_stream(text=text)
        else:
            return await self.tts_provider.tts(text=text)
    
    
    async def _fire_with_backup(self,
                           provider: Literal["stt", "tts"],
                          *args,
                          **kwargs
                          ) -> Optional[Union[str, bytes, AsyncGenerator[str, None], AsyncGenerator[bytes, None]]]:
        """
        Fire the backup provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        provider: "stt" or "tts"
        args: arguments for the fire_func
        kwargs: keyword arguments for the fire_func
        """
        backup_providers = self.backup_stt_providers if provider == "stt" else self.backup_tts_providers
        fire_func = self._stt if provider == "stt" else self._tts
        
        # Try primary provider first
        try:
            return await fire_func(*args, **kwargs)
        except Exception as e:
            current_provider = self.stt_provider if provider == "stt" else self.tts_provider
            logger.warning(f"Primary {provider} provider {current_provider.__class__.__name__} failed: {e}")
            
            if not backup_providers:
                raise Exception(f"Primary {provider} provider failed and no backups available")

        # Try backup providers
        rotation_count = 0
        possible_providers_count = len(backup_providers)
        while rotation_count < possible_providers_count:
            try:
                await self.__rotate_provider(provider)
                return await fire_func(*args, **kwargs)
            except Exception as e:
                current_provider = self.stt_provider if provider == "stt" else self.tts_provider
                logger.warning(f"Backup {provider} provider {current_provider.__class__.__name__} failed: {e}")
                rotation_count += 1
                
        raise Exception(f"All {provider} providers (primary + {possible_providers_count} backups) failed")

    
    async def _stt_with_backup(self, audio: bytes, 
                          stream: bool = False, 
                          enable_history: bool = True,
                          ) -> Optional[Union[str, AsyncGenerator[str, None]]]:
        """
        Convert audio to text using the backup stt_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        return await self._fire_with_backup(provider="stt", audio=audio, stream=stream, enable_history=enable_history)


    async def _tts_with_backup(self, 
                          text: str, 
                          stream: bool = False, 
                          ) -> Optional[Union[bytes, AsyncGenerator[bytes, None]]]:
        """
        Convert text to audio using the backup tts_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        return await self._fire_with_backup(provider="tts", text=text, stream=stream)

    async def stt(self, 
                  audio: bytes, 
                  stream: bool = False, 
                  enable_history: bool = True
                  ) -> Optional[Union[str, AsyncGenerator[str, None]]]:
        """
        Convert audio to text using the stt_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        return await self._stt_with_backup(audio=audio, stream=stream, enable_history=enable_history)
    

    async def tts(self, 
                  text: str,
                  stream: bool = False,
                  ) -> Optional[Union[bytes, AsyncGenerator[bytes, None]]]:
        """
        Convert text to audio using the tts_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        return await self._tts_with_backup(text=text, stream=stream)
    

    async def tts_stream_to(self, 
                            text, 
                            coro_factory: Callable[[None], AsyncGenerator[bytes, None]], 
                            try_backup: bool = True):
        """
        Convert text to audio using the tts_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        uses coro to stream generated audio to it;
        coro must accept pcm16 chunks as bytes and output None
        """
        # init coro
        coro = coro_factory()
        if not isinstance(coro, AsyncGenerator):
            raise ValueError("coro_factory must return an AsyncGenerator")
        await coro.asend(None)

        try:
            speech = await self.tts(text=text, stream=True)
            async for chunk in speech:
                await coro.asend(chunk)
        except StopAsyncIteration:
            pass
        except Exception as e:
            if try_backup and self.backup_tts_providers:
                logger.error(f"Error while streaming tts to {coro}: {e}; trying backup")
                await self.rotate_tts_provider()
                coro = coro_factory()
                await self.tts_stream_to(text, coro, try_backup=False)
            else:
                logger.error(f"Error while streaming tts to {coro}: {e}; no backup available")
                raise
        finally:
            await coro.aclose()

       
    