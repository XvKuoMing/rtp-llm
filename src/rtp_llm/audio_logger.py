import os
import wave
import math
import struct
import asyncio
from dataclasses import dataclass
from typing import List
import time

AUDIO_LOGS_DIR = "audio_logs"


@dataclass(frozen=True)
class AudioChunk:
    audio: bytes
    timestamp: float
    is_user: bool

class AudioLogger:

    def __init__(self, uid: str | int, sample_rate: int = 8000):
        os.makedirs(AUDIO_LOGS_DIR, exist_ok=True)
        self.uid = f"{uid}_conversation_{time.time()}"
        self.sample_rate = sample_rate
        self.lock = asyncio.Lock()
        # self.chunks: List[AudioChunk] = []
        self.start_time = None  # Track when logging started
        self.all_chunks: List[AudioChunk] = []  # Keep all chunks ever logged
    

    async def log(self, audio: bytes, is_user: bool):
        audio_chunk = AudioChunk(audio=audio, timestamp=time.time(), is_user=is_user)
        async with self.lock:
            self.all_chunks.append(audio_chunk)
    
    async def log_user(self, pcm16_frames: bytes):
        await self.log(pcm16_frames, is_user=True)
    
    async def log_ai(self, pcm16_frames: bytes):
        await self.log(pcm16_frames, is_user=False)

    async def beep(self):
        # Generate actual beep sound for pcm16
        frequency = 800  # Hz - typical beep frequency
        duration = 0.3   # seconds
        amplitude = 0.3  # 30% of max amplitude to avoid being too loud
        
        # Calculate number of samples
        num_samples = int(self.sample_rate * duration)
        
        # Generate sine wave samples
        beep_samples = []
        for i in range(num_samples):
            # Generate sine wave value (-1 to 1)
            t = i / self.sample_rate
            sample_value = amplitude * math.sin(2 * math.pi * frequency * t)
            
            # Convert to 16-bit signed integer (-32768 to 32767)
            sample_int16 = int(sample_value * 32767)
            
            # Convert to 2 bytes (little endian, signed)
            sample_bytes = sample_int16.to_bytes(2, byteorder='little', signed=True)
            beep_samples.append(sample_bytes)
        
        # Combine all samples into one bytes object
        beep_sound = b''.join(beep_samples)
        await self.log(beep_sound, is_user=False)

    
    async def save(self):
        """Save current state of the audio logger to a single WAV file.

        The heavy work (mixing and file I/O) is executed in a worker thread so
        it doesn't block the event loop, which is critical for real-time audio
        handling (e.g. RTP).
        """
        chunks_copy = list()
        async with self.lock:
            if not self.all_chunks:
                return
            
            # Copy chunks so we can release the lock before heavy processing
            chunks_copy = list(self.all_chunks)
        
        sorted_chunks = sorted(chunks_copy, key=lambda x: x.timestamp)

        filename = f"{self.uid}.wav"
        filepath = os.path.join(AUDIO_LOGS_DIR, filename)
        await self.__append_chunks_to_wav(sorted_chunks, filepath)
    

    async def __append_chunks_to_wav(self, chunks: List[AudioChunk], filepath: str):
        new_audio_data = b''.join(chunk.audio for chunk in chunks)
        # simply ovewrite with appended data
        async with self.lock:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(new_audio_data)

    
    def clear(self):
        self.all_chunks.clear()
        self.all_chunks.clear()
        self.start_time = None