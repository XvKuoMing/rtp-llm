import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.providers import Message, BaseSTTProvider, BaseTTSProvider
from src.history import BaseChatHistory
from typing import List, AsyncGenerator, Optional, Any, Awaitable, Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




class VoiceAgent:

    def __init__(self, 
                 stt_provider: BaseSTTProvider,
                 tts_provider: BaseTTSProvider,
                 history_manager: BaseChatHistory,
                 stt_gen_config: Optional[Dict[str, Any]] = None,
                 tts_gen_config: Optional[Dict[str, Any]] = None,
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
        self.__stt_gen_config = stt_gen_config
        self.__tts_gen_config = tts_gen_config

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
    

    async def _stt(self, 
                   audio: bytes, 
                   stream: bool = False,
                   enable_history: bool = True,
                   gen_config: Optional[Dict[str, Any]] = None,
                   ) -> Optional[str | AsyncGenerator[str, None]]:
        """
        Convert audio to text using the stt_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        gen_config = gen_config or self.__stt_gen_config
        user_message = Message(role="user", content=audio, data_type="audio")
        if enable_history:
            await self.history_manager.add_message(user_message)
        
        messages = await self.history_manager.get_messages(self.stt_provider.format)
        if stream:
            if enable_history:
                return self._stt_stream_with_history(messages, gen_config)
            else:
                return self.stt_provider.stt_stream(messages, gen_config)
        else:
            content = await self.stt_provider.stt(messages, gen_config)
            if enable_history:
                await self.history_manager.add_message(Message(role="assistant", content=content, data_type="text"))
            return content

    async def _stt_stream_with_history(self, messages, gen_config):
        """
        Stream STT response while accumulating full text for history.
        This yields chunks immediately (low latency) while building complete response.
        """
        accumulated_text = ""
        async for chunk in self.stt_provider.stt_stream(messages, gen_config):
            accumulated_text += chunk
            yield chunk
        
        await self.history_manager.add_message(
            Message(role="assistant", content=accumulated_text, data_type="text")
        )

    async def _tts(self, 
                   text: str,
                   stream: bool = False, 
                   response_format: str = "pcm",
                   gen_config: Optional[Dict[str, Any]] = None,
                   ) -> Optional[bytes | AsyncGenerator[bytes, None]]:
        """
        Convert text to audio using the tts_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        gen_config = gen_config or self.__tts_gen_config
        if stream:
            return self.tts_provider.tts_stream(text=text, response_format=response_format, gen_config=gen_config)
        else:
            return await self.tts_provider.tts(text=text, response_format=response_format, gen_config=gen_config)
    
    async def _stt_backup(self, audio: bytes, 
                          stream: bool = False, 
                          enable_history: bool = True,
                          gen_config: Optional[Dict[str, Any]] = None,
                          ) -> Optional[str | AsyncGenerator[str, None]]:
        """
        Convert audio to text using the backup stt_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        if not self.backup_stt_providers:
            raise Exception("No backup STT providers configured")
            
        for backup_provider in self.backup_stt_providers:
            self.stt_provider = backup_provider
            try:
                return await self._stt(audio=audio, stream=stream, enable_history=enable_history, gen_config=gen_config)
            except Exception as e:
                logger.warning(f"Backup STT provider {backup_provider.__class__.__name__} failed: {e}")
                continue
        raise Exception("All backup stt providers failed")

    async def _tts_backup(self, 
                          text: str, 
                          stream: bool = False, 
                          response_format: str = "pcm",
                          gen_config: Optional[Dict[str, Any]] = None,
                          ) -> Optional[bytes | AsyncGenerator[bytes, None]]:
        """
        Convert text to audio using the backup tts_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        if not self.backup_tts_providers:
            raise Exception("No backup TTS providers configured")
            
        for backup_provider in self.backup_tts_providers:
            self.tts_provider = backup_provider
            try:
                return await self._tts(text=text, 
                                       stream=stream, 
                                       response_format=response_format, 
                                       gen_config=gen_config)
            except Exception as e:
                logger.warning(f"Backup TTS provider {backup_provider.__class__.__name__} failed: {e}")
                continue
        raise Exception("All backup tts providers failed")


    async def _call_with_backup(self, func: Awaitable, 
                                backup_func: Awaitable, 
                                *args, 
                                **kwargs) -> Any:
        """
        Call a function with backup providers.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        func_name = getattr(func, '__name__', 'unknown_function')
        backup_func_name = getattr(backup_func, '__name__', 'unknown_backup_function')
        
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error in {func_name}: {e}; switching to backup")
            try:
                return await backup_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error while calling backup {backup_func_name}: {e}")
                raise e


    async def stt(self, 
                  audio: bytes, 
                  stream: bool = False, 
                  enable_history: bool = True,
                  gen_config: Optional[Dict[str, Any]] = None,
                  ) -> Optional[str | AsyncGenerator[str, None]]:
        """
        Convert audio to text using the stt_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        return await self._call_with_backup(self._stt, 
                                            self._stt_backup, 
                                            audio=audio, 
                                            stream=stream, 
                                            enable_history=enable_history,
                                            gen_config=gen_config)
    

    async def tts(self, 
                  text: str, 
                  stream: bool = False, 
                  response_format: str = "pcm",
                  gen_config: Optional[Dict[str, Any]] = None,
                  ) -> Optional[bytes | AsyncGenerator[bytes, None]]:
        """
        Convert text to audio using the tts_provider.
        If the primary provider fails, the agent will switch to the first available backup provider.
        """
        return await self._call_with_backup(self._tts, 
                                            self._tts_backup, 
                                            text=text, 
                                            stream=stream,
                                            response_format=response_format,
                                            gen_config=gen_config)
    