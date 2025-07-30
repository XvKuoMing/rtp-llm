"""
server for communication
"""
import logging
import asyncio

from .adapters import Adapter
from .buffer import BaseAudioBuffer
from .flow import BaseChatFlowManager
from .agents import VoiceAgent
from .vad import BaseVAD
from .utils.audio_processing import pcm2wav, resample_pcm16, adjust_volume_pcm16
from .audio_logger import AudioLogger
from .callbacks import BaseCallback, NullCallback
from .cache import BaseAudioCache, NullAudioCache
from typing import Optional, Dict, Any, List
import time
import uuid


logger = logging.getLogger(__name__)

VAD_INTERVAL = 0.5 # 500ms, 1.0 - sec

class Server:

    def __init__(self,
                 adapter: Adapter,
                 audio_buffer: BaseAudioBuffer,
                 flow_manager: BaseChatFlowManager,
                 vad: BaseVAD,
                 agent: VoiceAgent,
                 max_wait_time: int = 7, # <= 0 means no max wait time
                 vad_interval: float = VAD_INTERVAL,
                 audio_cache: BaseAudioCache = None,
                 ):
        
        # components
        self.adapter = adapter
        self.audio_buffer = audio_buffer
        self.flow_manager = flow_manager
        self.vad = vad
        self.agent = agent
        self.audio_cache = audio_cache or NullAudioCache()
        logger.info(f"Audio cache: {self.audio_cache.__class__.__name__}")

        # Convert VAD interval to samples and bytes for proper unit handling
        self.vad_interval_samples = int(self.adapter.sample_rate * vad_interval)  # samples per interval
        self.vad_interval_bytes = self.vad_interval_samples * 2  # bytes per interval (16-bit PCM)
        
        # Convert to milliseconds for comparison with VAD min duration
        vad_interval_ms = self.vad_interval_samples * 1000 / self.adapter.sample_rate
        
        if vad_interval_ms < self.vad.min_speech_duration_ms:
            raise ValueError(f"VAD interval {vad_interval_ms:.1f}ms is less than min speech duration: {self.vad.min_speech_duration_ms}ms")
        if vad_interval_ms / 2 <= self.vad.min_speech_duration_ms:
            logger.warning(f"Min speech duration {self.vad.min_speech_duration_ms}ms might be too high for given vad interval {vad_interval_ms:.1f}ms, consider increasing vad interval or decreasing min speech duration")

        # state management
        self.last_response_time = None
        self.processed_bytes = 0
        self.speaking: Optional[asyncio.Task]= None
        self.uid: Optional[int | str] = None
        self.callback: Optional[BaseCallback] = None

        # static state management
        self.answer_lock = asyncio.Lock()

        # some params
        self.volume = 1.0 # default volume
        self.max_wait_time = max_wait_time


        # internal
        self.__last_ai_pcm16_chunks = []
    
    def update_agent_config(
            self,
            system_prompt: Optional[str] = None,
            tts_gen_config: Optional[Dict[str, Any]] = None,
            stt_gen_config: Optional[Dict[str, Any]] = None,
    ):
        if system_prompt:
            self.agent.stt_provider.system_prompt = system_prompt
                
        if tts_gen_config:
            self.agent.update_tts_config(tts_gen_config)
        if stt_gen_config:
            self.agent.update_stt_config(stt_gen_config)


    async def run(self, 
                  first_message: Optional[str] = None, 
                  uid: Optional[int | str] = None, 
                  allow_interruptions: bool = False,
                  system_prompt: Optional[str] = None,
                  tts_gen_config: Optional[Dict[str, Any]] = None,
                  stt_gen_config: Optional[Dict[str, Any]] = None,
                  volume: float = 1.0,
                  callback: Optional[BaseCallback] = None,
                  ):
        """
        run the server
        """

        # Store volume setting
        self.volume = volume

        self.update_agent_config(
            system_prompt=system_prompt,
            tts_gen_config=tts_gen_config,
            stt_gen_config=stt_gen_config,
        )

        self.uid = uid or str(uuid.uuid4())
        self.callback = callback or NullCallback()
        self.audio_logger = AudioLogger(
            uid=self.uid, 
            sample_rate=self.adapter.sample_rate)
        
        if first_message is not None and self.adapter.peer_is_configured:
            logger.info(f"Speaking first message: {first_message}")
            self.speaking = asyncio.create_task(self.speak(first_message))
            first_message = None
        

        asyncio.create_task(self.callback.on_start(self.uid)) # fire and forget

        while True:
            try:
                audio = await self.adapter.receive_audio() # produces pcm16
                if audio is None:
                    # Small delay to prevent busy waiting when no audio is received
                    await asyncio.sleep(0.01)  # 10ms delay
                    continue

                async with self.answer_lock:
                    is_speaking = self.speaking is not None and not self.speaking.done()

                    if first_message and self.adapter.peer_is_configured:
                        logger.info(f"Speaking first message: {first_message}")
                        self.speaking = asyncio.create_task(self.speak(first_message))
                        first_message = None
                        continue

                    await self.audio_buffer.add_frame(audio)
                    await self.audio_logger.log_user(audio)

                    buffer_audio = await self.audio_buffer.get_frames()
                    if len(buffer_audio) < self.processed_bytes + self.vad_interval_bytes: # ensure to not check the same interval multiple times
                        continue
                    if is_speaking and not allow_interruptions:
                        #NOTE: user speech will be saved in the buffer
                        continue
                    last_chunk_audio = buffer_audio[self.processed_bytes:self.processed_bytes + self.vad_interval_bytes] # cutting last interval of audio
                    self.processed_bytes += self.vad_interval_bytes

                    vad_state = await self.vad.detect(last_chunk_audio)

                    max_time_reached = self.last_response_time is not None \
                                        and self.max_wait_time > 0 \
                                        and (time.time() - self.last_response_time) > self.max_wait_time
                    need_run_agent = await self.flow_manager.run_agent(vad_state)

                    if max_time_reached or need_run_agent:
                        if need_run_agent:
                            await self.audio_logger.beep() # NOTE: this is a hack to make the user aware that the agent started answering
                        if is_speaking:
                            # allowing interruptions
                            self.speaking.cancel()
                        logger.info(f"Answering to the user; max_time_reached: {max_time_reached}, need_run_agent: {need_run_agent}")
                        self.speaking = asyncio.create_task(self.answer(buffer_audio))
                        self.flow_manager.reset()
                        self.audio_buffer.clear()
                        self.processed_bytes = 0
                    else:
                        pass # later, we will implement silence sending
            except Exception as e:
                logger.error(f"Error in server loop: {e}")
                asyncio.create_task(self.callback.on_error(self.uid, e)) # fire and forget
                raise e
    
    async def answer(self, audio: bytes) -> bool:
        """
        answer the audio
        Returns True if successful, False if failed
        """
        try:
            wav_audio = await pcm2wav(audio, sample_rate=self.adapter.sample_rate)

            logger.info(f"Converted to wav, {len(wav_audio)} bytes")
            response = await self.agent.stt(
                audio=wav_audio,
                stream=False,
                enable_history=True,
            )
            logger.info(f"STT response: {response}")
            response_transformation = await self.callback.on_response(self.uid, response)
            speech_text = response_transformation.text if response_transformation.text else response
            await self.speak(speech_text)
            if response_transformation.post_action:
                # await response_transformation.post_action
                asyncio.create_task(response_transformation.post_action) # fire and forget
            return True
        except asyncio.CancelledError:
            logger.info("Answer was cancelled")
        except Exception as e:
            logger.error(f"Error answering audio: {e}")
            # Notify callback about the error
            asyncio.create_task(self.callback.on_error(self.uid, e))
            return False
    
    async def speak(self, text: str):
        """
        speak the text
        """
        coro = self._speak()
        key = self.audio_cache.make_key(text, self.agent.tts_provider.tts_footprint)
        cached_chunks = await self.audio_cache.get(key)
        
        try:
            if cached_chunks:
                logger.info(f"Using cached audio for text: {text[:50]}... ({len(cached_chunks)} chunks)")
                await coro.asend(None)  # initialize the coroutine
                for chunk in cached_chunks:
                    await coro.asend(chunk)      
            else:
                await self.agent.tts_stream_to(text, coro, try_backup=True)
                logger.info(f"Finished speaking")
                await self.audio_cache.set(key, self.__last_ai_pcm16_chunks)
                logger.info(f"Cached audio for {text}, {len(self.__last_ai_pcm16_chunks)} bytes")
        except asyncio.CancelledError:
            logger.info(f"Speech was cancelled for text: {text[:50]}...")
            
        asyncio.create_task(self.audio_logger.save()) # save in background to not block the main thread
        self.last_response_time = time.time()
    
    async def _speak(self):
        chunk_count = 0
        total_bytes = 0
        buffer = b''
        self.__last_ai_pcm16_chunks = [] # reset the buffer
        while True:
            pcm16_chunk = yield
            chunk_count += 1
            total_bytes += len(pcm16_chunk)

            self.__last_ai_pcm16_chunks.append(pcm16_chunk) # save raw tts chunks

            if buffer:
                logger.debug(f"Joining buffer with pcm16, buffer length: {len(buffer)}")
                pcm16_chunk = buffer + pcm16_chunk
                buffer = b''
            if len(pcm16_chunk) % 2 != 0:
                logger.debug(f"Input pcm16 length is not even ({len(pcm16_chunk)}), appending to deque")
                buffer += pcm16_chunk[-1:]
                pcm16_chunk = pcm16_chunk[:-1]

            logger.debug(f"Sending chunk {chunk_count} ({len(pcm16_chunk)} bytes)")
            if self.agent.tts_provider.response_sample_rate != self.adapter.sample_rate:
                logger.debug(f"Resampling audio from {self.agent.tts_provider.response_sample_rate}Hz to {self.adapter.sample_rate}Hz")
                pcm16_chunk = await resample_pcm16(
                    pcm16=pcm16_chunk,
                    original_sample_rate=self.agent.tts_provider.response_sample_rate,
                    target_sample_rate=self.adapter.sample_rate,
                )

            # Apply volume adjustment
            if self.volume != 1.0:
                logger.debug(f"Applying volume adjustment: {self.volume}")
                pcm16_chunk = await adjust_volume_pcm16(
                    pcm16=pcm16_chunk,
                    volume_factor=self.volume,
                )

            await self.adapter.send_audio(pcm16_chunk)
            await self.audio_logger.log_ai(pcm16_chunk) # log without any processing

    def close(self):
        logger.info("Closing server")
        self.adapter.close()
        self.audio_buffer.clear()
        self.agent.history_manager.clear()
        self.flow_manager.reset()
        self.speaking = None
        self.processed_bytes = 0
        self.last_response_time = None
        if self.uid is not None and self.callback is not None:
            asyncio.create_task(self.callback.on_finish(self.uid)) # fire and forget
            self.uid = None
            self.callback = None



