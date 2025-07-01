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
from .utils.audio_processing import pcm2wav, resample_pcm16
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
                 max_wait_time: int = 7,
                 min_wait_time: int = 2
                 ):
        self.adapter = adapter
        self.audio_buffer = audio_buffer
        self.flow_manager = flow_manager
        self.vad = vad
        self.agent = agent
        self.audio_logger = AudioLogger(uid=random.randint(0, 1000000), sample_rate=adapter.sample_rate)
        self.max_wait_time = max_wait_time
        self.min_wait_time = min_wait_time
        self.last_response_time = time.time()
        self.speaking = False
        self.answer_lock = asyncio.Lock()
    

    async def run(self, first_message: Optional[str] = None):
        """
        run the server
        """
        # Handle first message outside the main loop
        # if first_message is not None:
        #     logger.info(f"Speaking first message: {first_message}")
        #     await self.speak(first_message)

        while True:
            audio = await self.adapter.receive_audio()
            if audio is None:
                # Small delay to prevent busy waiting when no audio is received
                await asyncio.sleep(0.01)  # 10ms delay
                continue

            async with self.answer_lock:
                if self.speaking:
                    # discarding user audio while speaking
                    logger.info("Already speaking, discarding user audio")
                    continue
                await self.audio_buffer.add_frame(audio)
                logger.info(f"Added {len(audio)} bytes to buffer")
                await self.audio_logger.log_user(audio)
                buffer_audio = await self.audio_buffer.get_frames()
                logger.info(f"Got {len(buffer_audio)} bytes from buffer")
                last_second = self.adapter.sample_rate * 2 # 2 bytes per sample for pcm16
                logger.info(f"Last second: {last_second}")
                if len(buffer_audio) < last_second:
                    logger.info(f"Not enough audio in buffer, {len(buffer_audio)} bytes, waiting for more")
                    continue
                if self.speaking:
                    logger.info("Already speaking, skipping VAD")
                    continue
                last_second_of_audio = buffer_audio[-last_second:] # cutting last second of audio
                logger.info(f"Last second of audio: {len(last_second_of_audio)} bytes")
                vad_state = await self.vad.detect(last_second_of_audio)
                logger.info(f"VAD state: {vad_state}, speaking: {self.speaking}")
                if (time.time() - self.last_response_time) > self.min_wait_time and \
                    await self.flow_manager.run_agent(vad_state):
                    logger.info("VAD: user speech ended, answering")
                    # await self.answer(buffer_audio)
                    asyncio.create_task(self.answer(buffer_audio))
                    await self.flow_manager.reset()
                elif (time.time() - self.last_response_time) > self.max_wait_time:
                    logger.info("VAD: max wait time reached, answering")
                    # await self.answer(buffer_audio)
                    asyncio.create_task(self.answer(buffer_audio))
                    await self.flow_manager.reset()
                else:
                    pass # later, we will implement silence sending
    
    async def answer(self, audio: bytes):
        """
        answer the audio
        """
        try:
            if self.speaking:
                return
            self.speaking = True
            # Clear buffer immediately to prevent audio contamination
            self.audio_buffer.clear()
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
        if not self.speaking:
            self.speaking = True
        speech = await self.agent.tts(
            text=text,
            stream=True,
            response_format=TTS_RESPONSE_FORMAT
        )
        logger.info(f"Generated speech for: {text}" if speech else "No speech generated")

        chunk_count = 0
        total_bytes = 0
        async for chunk in speech:
            chunk_count += 1
            total_bytes += len(chunk)


            if TTS_SAMPLE_RATE != self.adapter.sample_rate:
                logger.info(f"Resampling audio from {TTS_SAMPLE_RATE}Hz to {self.adapter.sample_rate}Hz")
                chunk = await resample_pcm16(chunk, 
                                             original_sample_rate=TTS_SAMPLE_RATE, 
                                             target_sample_rate=self.adapter.sample_rate)

            await self.adapter.send_audio(chunk)
            await self.audio_logger.log_ai(chunk)
        
        logger.info(f"Finished speaking: {chunk_count} chunks, total {total_bytes} bytes")
        await self.audio_logger.save()
        self.last_response_time = time.time()
        self.speaking = False
        

    def close(self):
        self.adapter.close()



