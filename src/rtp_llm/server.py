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
from typing import Optional, Dict, Any
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
                 ):
        
        # components
        self.adapter = adapter
        self.audio_buffer = audio_buffer
        self.flow_manager = flow_manager
        self.vad = vad
        self.agent = agent

        self.vad_interval = int((self.adapter.sample_rate * 2) * vad_interval) # NOTE: 500ms, do not recommend to change
        if self.vad_interval < self.vad.min_speech_duration_ms:
            raise ValueError(f"VAD interval is less than min speech duration: {self.vad_interval} < {self.vad.min_speech_duration_ms}")
        if self.vad_interval // 2 <= self.vad.min_speech_duration_ms:
            logger.warning(f"Min speech duration might be too high for given vad interval, consider increasing vad interval or decreasing min speech duration")

        # state management
        self.max_wait_time = max_wait_time
        self.last_response_time = None
        self.processed_seconds = 0
        self.speaking: Optional[asyncio.Task]= None
        self.answer_lock = asyncio.Lock()


        self.volume = 1.0 # default volume
    
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


        # Handle first message outside the main loop
        if self.last_response_time is None:
            self.last_response_time = time.time()

        uid = uid or str(uuid.uuid4())
        self.audio_logger = AudioLogger(
            uid=uid, 
            sample_rate=self.adapter.sample_rate)
        
        if first_message is not None and self.adapter.peer_is_configured:
            logger.info(f"Speaking first message: {first_message}")
            self.speaking = asyncio.create_task(self.speak(first_message))
            first_message = None

        while True:
            audio = await self.adapter.receive_audio() # produces pcm16
            if audio is None:
                # Small delay to prevent busy waiting when no audio is received
                await asyncio.sleep(0.01)  # 10ms delay
                continue

            async with self.answer_lock:
                is_speaking = self.speaking is not None and not self.speaking.done()
                if is_speaking:
                    # saving user audio while speaking ## NOTE: as for now simply discard
                    # await self.audio_buffer.add_frame(audio)
                    # await self.audio_logger.log_user(audio)
                    continue

                if first_message and self.adapter.peer_is_configured:
                    logger.info(f"Speaking first message: {first_message}")
                    self.speaking = asyncio.create_task(self.speak(first_message))
                    first_message = None
                    continue

                await self.audio_buffer.add_frame(audio)
                await self.audio_logger.log_user(audio)

                buffer_audio = await self.audio_buffer.get_frames()
                # last_second = self.adapter.sample_rate * 2 # 2 bytes per sample for pcm16
                # last_second = (self.adapter.sample_rate * 2) // 2 #NOTE: 500ms
                if len(buffer_audio) < self.processed_seconds + self.vad_interval: # ensure to not check the same second multiple times
                    continue
                if is_speaking:
                    continue
                last_chunk_audio = buffer_audio[self.processed_seconds:self.processed_seconds + self.vad_interval] # cutting last second of audio
                self.processed_seconds += self.vad_interval

                vad_state = await self.vad.detect(last_chunk_audio)

                max_time_reached = self.max_wait_time > 0 and (time.time() - self.last_response_time) > self.max_wait_time
                need_run_agent = await self.flow_manager.run_agent(vad_state)

                if max_time_reached or need_run_agent:
                    if need_run_agent:
                        await self.audio_logger.beep() # NOTE: this is a hack to make the user aware that the agent started answering
                    logger.info(f"Answering to the user; max_time_reached: {max_time_reached}, need_run_agent: {need_run_agent}")
                    self.speaking = asyncio.create_task(self.answer(buffer_audio))
                    self.flow_manager.reset()
                    self.audio_buffer.clear()
                    self.processed_seconds = 0
                else:
                    pass # later, we will implement silence sending
    
    async def answer(self, audio: bytes):
        """
        answer the audio
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
            await self.speak(response)
        except Exception as e:
            logger.error(f"Error answering audio: {e}")
            return None
    

    async def speak(self, text: str):
        """
        speak the text
        """
        speech = await self.agent.tts(
            text=text,
            stream=True,
        ) # MUST PRODUCE PCM16 AS STATED IN THE DOCS
        logger.info(f"Generated speech for: {text}" if speech else "No speech generated")

        chunk_count = 0
        total_bytes = 0
        
        buffer = b''
        async for chunk in speech:
            chunk_count += 1
            total_bytes += len(chunk)


            if buffer:
                logger.debug(f"Joining buffer with pcm16, buffer length: {len(buffer)}")
                chunk = buffer + chunk
                buffer = b''

            if len(chunk) % 2 != 0:
                logger.debug(f"Input pcm16 length is not even ({len(chunk)}), appending to deque")
                buffer += chunk[-1:]
                chunk = chunk[:-1]


            logger.debug(f"Sending chunk {chunk_count} of {total_bytes} bytes")
            if self.agent.tts_provider.response_sample_rate != self.adapter.sample_rate:
                logger.debug(f"Resampling audio from {self.agent.tts_provider.response_sample_rate}Hz to {self.adapter.sample_rate}Hz")
                chunk = await resample_pcm16(
                    pcm16=chunk,
                    original_sample_rate=self.agent.tts_provider.response_sample_rate,
                    target_sample_rate=self.adapter.sample_rate,
                )

            # Apply volume adjustment
            if self.volume != 1.0:
                logger.debug(f"Applying volume adjustment: {self.volume}")
                chunk = await adjust_volume_pcm16(
                    pcm16=chunk,
                    volume_factor=self.volume,
                )

            await self.adapter.send_audio(chunk)
            await self.audio_logger.log_ai(chunk)

        logger.info(f"Finished speaking: {chunk_count} chunks, total {total_bytes} bytes")
        await self.audio_logger.save()
        self.last_response_time = time.time()
        

    def close(self):
        logger.info("Closing server")
        self.adapter.close()
        self.audio_buffer.clear()
        self.agent.history_manager.clear()
        self.flow_manager.reset()
        self.speaking = None
        self.processed_seconds = 0
        self.last_response_time = None



