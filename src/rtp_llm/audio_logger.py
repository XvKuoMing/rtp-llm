import os
import wave
import threading
import math
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
        self.lock = threading.Lock()
        self.chunks: List[AudioChunk] = []
    

    async def log(self, audio: bytes, is_user: bool):
        audio_chunk = AudioChunk(audio=audio, timestamp=time.time(), is_user=is_user)
        with self.lock:
            self.chunks.append(audio_chunk)
    
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
        """saves current state of the audio logger to a single WAV file containing both user and AI audio"""
        with self.lock:
            if not self.chunks:
                return
                
            # Use a fixed filename for the user to append to same file
            filename = f"{self.uid}.wav"
            filepath = os.path.join(AUDIO_LOGS_DIR, filename)
            
            # Sort chunks by timestamp to maintain conversation flow
            sorted_chunks = sorted(self.chunks, key=lambda x: x.timestamp)
            await self._append_chunks_to_wav(sorted_chunks, filepath)
            
            # Clear chunks after saving
            self.chunks.clear()
    
    async def _append_chunks_to_wav(self, chunks: List[AudioChunk], filepath: str):
        """Append audio chunks to a WAV file (or create new if doesn't exist)"""
        if not chunks:
            return
            
        try:
            existing_audio_data = b''
            
            # Read existing audio data if file exists
            if os.path.exists(filepath):
                try:
                    with wave.open(filepath, 'rb') as existing_wav:
                        # Verify sample rate matches
                        if existing_wav.getframerate() != self.sample_rate:
                            print(f"Warning: Sample rate mismatch. Existing: {existing_wav.getframerate()}, Current: {self.sample_rate}")
                        
                        # Read all existing frames
                        existing_audio_data = existing_wav.readframes(existing_wav.getnframes())
                except Exception as e:
                    print(f"Error reading existing WAV file {filepath}: {e}")
            
            # Combine existing and new audio data
            new_audio_data = b''.join(chunk.audio for chunk in chunks)
            combined_audio_data = existing_audio_data + new_audio_data
            
            # Write combined audio to file
            with wave.open(filepath, 'wb') as wav_file:
                # Set WAV parameters
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit (2 bytes per sample)
                wav_file.setframerate(self.sample_rate)
                
                # Write all audio data
                wav_file.writeframes(combined_audio_data)
                    
        except Exception as e:
            print(f"Error appending audio to {filepath}: {e}")