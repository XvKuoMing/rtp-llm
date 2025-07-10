import os
import wave
import threading
import math
import struct
from dataclasses import dataclass
from typing import List, Tuple
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
        self.start_time = None  # Track when logging started
    

    async def log(self, audio: bytes, is_user: bool):
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        audio_chunk = AudioChunk(audio=audio, timestamp=current_time, is_user=is_user)
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
    
    def _bytes_to_samples(self, audio_bytes: bytes) -> List[int]:
        """Convert bytes to list of 16-bit signed integers"""
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]  # Remove odd byte
        return list(struct.unpack(f'<{len(audio_bytes)//2}h', audio_bytes))
    
    def _samples_to_bytes(self, samples: List[int]) -> bytes:
        """Convert list of 16-bit signed integers to bytes"""
        return struct.pack(f'<{len(samples)}h', *samples)
    
    def _mix_audio_samples(self, samples1: List[int], samples2: List[int]) -> List[int]:
        """Mix two audio sample lists, handling different lengths"""
        max_len = max(len(samples1), len(samples2))
        mixed = []
        
        for i in range(max_len):
            sample1 = samples1[i] if i < len(samples1) else 0
            sample2 = samples2[i] if i < len(samples2) else 0
            
            # Mix samples with volume reduction to prevent clipping
            mixed_sample = int((sample1 + sample2) * 0.7)  # Reduce volume to prevent clipping
            
            # Clamp to 16-bit range
            mixed_sample = max(-32768, min(32767, mixed_sample))
            mixed.append(mixed_sample)
        
        return mixed
    
    def _create_timeline_audio(self, chunks: List[AudioChunk]) -> bytes:
        """Create a single audio stream with proper timing and mixing for overlapping audio"""
        if not chunks or self.start_time is None:
            return b''
        
        # Calculate the total duration and create a timeline
        end_time = max(chunk.timestamp + len(chunk.audio) / (2 * self.sample_rate) for chunk in chunks)
        total_duration = end_time - self.start_time
        total_samples = int(total_duration * self.sample_rate)
        
        # Initialize timeline with silence
        timeline = [0] * total_samples
        
        for chunk in chunks:
            # Calculate start position in the timeline
            chunk_offset_seconds = chunk.timestamp - self.start_time
            start_sample = int(chunk_offset_seconds * self.sample_rate)
            
            # Convert chunk audio to samples
            chunk_samples = self._bytes_to_samples(chunk.audio)
            
            # Mix chunk samples into timeline
            for i, sample in enumerate(chunk_samples):
                timeline_pos = start_sample + i
                if 0 <= timeline_pos < len(timeline):
                    # Mix with existing audio (simple addition with volume reduction)
                    mixed_sample = int((timeline[timeline_pos] + sample) * 0.8)
                    timeline[timeline_pos] = max(-32768, min(32767, mixed_sample))
        
        return self._samples_to_bytes(timeline)
    
    async def save(self):
        """saves current state of the audio logger to a single WAV file containing mixed user and AI audio"""
        with self.lock:
            if not self.chunks:
                return
                
            # Use a fixed filename for the user to append to same file
            filename = f"{self.uid}.wav"
            filepath = os.path.join(AUDIO_LOGS_DIR, filename)
            
            # Create mixed audio timeline
            mixed_audio_data = self._create_timeline_audio(self.chunks)
            
            if mixed_audio_data:
                await self._append_mixed_audio_to_wav(mixed_audio_data, filepath)
            
            # Clear chunks after saving
            self.chunks.clear()
    
    async def _append_mixed_audio_to_wav(self, new_audio_data: bytes, filepath: str):
        """Append mixed audio data to a WAV file (or create new if doesn't exist)"""
        if not new_audio_data:
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