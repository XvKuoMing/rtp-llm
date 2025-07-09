from .base import BaseVAD, VoiceState
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
import logging

logger = logging.getLogger(__name__)


class SileroVAD(BaseVAD):

    def __init__(self, sample_rate: int = 8000, threshold: float = 0.5, min_speech_duration_ms: int = 60):
        """
        Initialize Silero VAD.
        
        Args:
            sample_rate: Sample rate of audio (must be 8000 or 16000)
            threshold: Speech probability threshold (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms #TODO: actually use it
        
        # Load the Silero VAD model
        self.model = load_silero_vad()
    
    def _pcm16_to_tensor(self, pcm16_frame: bytes) -> torch.Tensor:
        """
        Convert PCM16 bytes to torch tensor.
        
        Args:
            pcm16_frame: Raw PCM16 audio bytes
            
        Returns:
            Normalized audio tensor
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(pcm16_frame, dtype=np.int16)
        
        # Normalize to [-1, 1] range
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_np)
        
        return audio_tensor

    async def detect(self, pcm16_frame: bytes) -> VoiceState:
        """
        Detect voice activity in PCM16 frame.
        
        Args:
            pcm16_frame: Raw PCM16 audio bytes
            
        Returns:
            VoiceState.SPEECH if speech detected, VoiceState.SILENCE otherwise
        """
        try:
            # Convert PCM16 to tensor
            audio_tensor = self._pcm16_to_tensor(pcm16_frame)
            
            # Use get_speech_timestamps to detect speech in the frame
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                return_seconds=False  # Return in samples
            )
            # If any speech timestamps are found, consider it speech
            return VoiceState.SPEECH if len(speech_timestamps) > 0 else VoiceState.SILENCE
            
        except Exception as e:
            # In case of any error, default to silence
            logger.error(f"Error in Silero VAD detection: {e}")
            return VoiceState.SILENCE
