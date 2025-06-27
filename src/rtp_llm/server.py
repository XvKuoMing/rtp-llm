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
from .utils.audio_processing import pcm2wav
from .audio_logger import AudioLogger
from typing import Optional
import random
import time


logger = logging.getLogger(__name__)

TTS_RESPONSE_FORMAT = "pcm"


class Server:

    def __init__(self,
                 adapter: Adapter,
                 audio_buffer: BaseAudioBuffer,
                 flow_manager: BaseChatFlowManager,
                 vad: BaseVAD,
                 agent: VoiceAgent,
                 max_wait_time: int = 7
                 ):
        self.adapter = adapter
        self.audio_buffer = audio_buffer
        self.flow_manager = flow_manager
        self.vad = vad
        self.agent = agent
        self.audio_logger = AudioLogger(uid=random.randint(0, 1000000), sample_rate=24_000)
        self.max_wait_time = max_wait_time
        self.last_response_time = time.time()
    

    async def run(self, first_message: Optional[str] = None):
        """
        run the server
        """
        # Handle first message outside the main loop
        if first_message is not None:
            logger.info(f"Speaking first message: {first_message}")
            await self.speak(first_message)

        while True:
            audio = await self.adapter.receive_audio()
            if audio is not None:
                await self.audio_buffer.add_frame(audio)
                await self.audio_logger.log_user(audio)
                vad_state = await self.vad.detect(audio)
                if await self.flow_manager.run_agent(vad_state):
                    buffer_audio = await self.audio_buffer.get_frames()
                    await self.answer(buffer_audio)
                elif (time.time() - self.last_response_time) > self.max_wait_time:
                    buffer_audio = await self.audio_buffer.get_frames()
                    await self.speak(buffer_audio)
                    await self.flow_manager.reset()
                else:
                    pass # later, we will implement silence sending
            else:
                # Small delay to prevent busy waiting when no audio is received
                await asyncio.sleep(0.01)  # 10ms delay
    
    async def answer(self, audio: bytes):
        """
        answer the audio
        """
        try:
            wav_audio = await pcm2wav(audio)
            response = await self.agent.stt(
                audio=wav_audio,
                stream=False,
                enable_history=True,
            )
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
            response_format=TTS_RESPONSE_FORMAT
        )
        logger.info(f"Generated speech for: {text}" if speech else "No speech generated")

        chunk_count = 0
        total_bytes = 0
        async for chunk in speech:
            chunk_count += 1
            total_bytes += len(chunk)

            await self.adapter.send_audio(chunk)
            await self.audio_logger.log_ai(chunk)
        
        logger.info(f"Finished sending {chunk_count} chunks, total {total_bytes} bytes")
        await self.audio_logger.save()
        self.last_response_time = time.time()
        

    def close(self):
        self.adapter.close()



