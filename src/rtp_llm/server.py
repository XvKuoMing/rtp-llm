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
from .utils.audio_processing import pcm2wav, StreamingResample
from .audio_logger import AudioLogger
from typing import Optional
import random
import time


logger = logging.getLogger(__name__)

TTS_RESPONSE_FORMAT = "pcm"
TTS_SAMPLE_RATE = 24_000


class Server:

    def __init__(self,
                 adapter: Adapter,
                 audio_buffer: BaseAudioBuffer,
                 flow_manager: BaseChatFlowManager,
                 vad: BaseVAD,
                 agent: VoiceAgent,
                 max_wait_time: int = 7, # <= 0 means no max wait time
                 ):
        
        # components
        self.adapter = adapter
        self.audio_buffer = audio_buffer
        self.flow_manager = flow_manager
        self.vad = vad
        self.agent = agent
        
        # state management
        self.max_wait_time = max_wait_time
        self.last_response_time = None
        self.processed_seconds = 0
        self.speaking = False
        self.answer_lock = asyncio.Lock()
    

    async def run(self, 
                  first_message: Optional[str] = None, 
                  uid: Optional[int | str] = None, 
                  system_prompt: Optional[str] = None,
                  allow_interruptions: bool = False):
        """
        run the server
        """
        # Handle first message outside the main loop
        if self.last_response_time is None:
            self.last_response_time = time.time()

        uid = uid or random.randint(0, 1000000)
        self.audio_logger = AudioLogger(
            uid=uid, 
            sample_rate=self.adapter.sample_rate)
        
        if system_prompt:
            self.agent.stt_provider.system_prompt = system_prompt
        
        if first_message is not None and self.adapter.peer_is_configured:
            logger.info(f"Speaking first message: {first_message}")
            asyncio.create_task(self.speak(first_message))
            first_message = None

        while True:
            audio = await self.adapter.receive_audio() # produces pcm16
            if audio is None:
                # Small delay to prevent busy waiting when no audio is received
                await asyncio.sleep(0.01)  # 10ms delay
                continue

            async with self.answer_lock:
                if self.speaking:
                    # saving user audio while speaking ## NOTE: as for now simply discard
                    # await self.audio_buffer.add_frame(audio)
                    # await self.audio_logger.log_user(audio)
                    continue

                if first_message and self.adapter.peer_is_configured:
                    logger.info(f"Speaking first message: {first_message}")
                    asyncio.create_task(self.speak(first_message))
                    first_message = None
                    continue

                await self.audio_buffer.add_frame(audio)
                await self.audio_logger.log_user(audio)

                buffer_audio = await self.audio_buffer.get_frames()
                last_second = self.adapter.sample_rate * 2 # 2 bytes per sample for pcm16
                if len(buffer_audio) < self.processed_seconds + last_second: # ensure to not check the same second multiple times
                    continue
                if self.speaking:
                    continue
                last_second_of_audio = buffer_audio[self.processed_seconds:self.processed_seconds + last_second] # cutting last second of audio
                self.processed_seconds += last_second

                vad_state = await self.vad.detect(last_second_of_audio)

                max_time_reached = self.max_wait_time > 0 and (time.time() - self.last_response_time) > self.max_wait_time
                need_run_agent = await self.flow_manager.run_agent(vad_state)

                if max_time_reached or need_run_agent:
                    logger.info(f"Answering to the user; max_time_reached: {max_time_reached}, need_run_agent: {need_run_agent}")
                    asyncio.create_task(self.answer(buffer_audio))
                    await self.flow_manager.reset()
                    self.audio_buffer.clear()
                    self.processed_seconds = 0
                else:
                    pass # later, we will implement silence sending
    
    async def answer(self, audio: bytes):
        """
        answer the audio
        """
        try:
            self.speaking = True
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
        finally:
            self.speaking = False
    

    async def speak(self, text: str):
        """
        speak the text
        """
        self.speaking = True
        speech = await self.agent.tts(
            text=text,
            stream=True,
            response_format=TTS_RESPONSE_FORMAT,
            response_sample_rate=TTS_SAMPLE_RATE,
        )
        logger.info(f"Generated speech for: {text}" if speech else "No speech generated")

        chunk_count = 0
        total_bytes = 0

        resampler = StreamingResample(
            original_sample_rate=TTS_SAMPLE_RATE, 
            target_sample_rate=self.adapter.sample_rate)
        
        async for chunk in speech:
            chunk_count += 1
            total_bytes += len(chunk)
            logger.debug(f"Sending chunk {chunk_count} of {total_bytes} bytes")
            if TTS_SAMPLE_RATE != self.adapter.sample_rate:
                logger.debug(f"Resampling audio from {TTS_SAMPLE_RATE}Hz to {self.adapter.sample_rate}Hz")
                chunk = await resampler.resample_pcm16(chunk)

            await self.adapter.send_audio(chunk)
            await self.audio_logger.log_ai(chunk)

        logger.info(f"Finished speaking: {chunk_count} chunks, total {total_bytes} bytes")
        await self.audio_logger.save()
        self.last_response_time = time.time()
        self.speaking = False
        

    def close(self):
        logger.info("Closing server")
        self.adapter.close()
        self.audio_buffer.clear()
        self.agent.history_manager.clear()
        self.flow_manager.reset()
        self.speaking = False
        self.processed_seconds = 0
        self.last_response_time = None



