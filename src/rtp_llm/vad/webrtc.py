from .base import BaseVAD, VoiceState
import webrtcvad
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class WebRTCVAD(BaseVAD):

    def __init__(self, sample_rate: int = 8000, aggressiveness: int = 3, min_speech_duration_ms: int = 30):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        
        # Validate sample rate
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"WebRTC VAD only supports sample rates: 8000, 16000, 32000, 48000 Hz, got {self.sample_rate}")
    

    async def detect(self, pcm16_frame: bytes) -> VoiceState:
        """
        Detect voice activity in PCM16 frame.
        
        WebRTC VAD requires frames to be exactly 10, 20, or 30 ms in duration.
        This method will split the input into appropriate frames and return
        SPEECH only if the total speech duration meets min_speech_duration_ms threshold.
        """
        try:
            # Calculate frame size for 30ms (we'll use 30ms frames)
            frame_duration_ms = 30
            bytes_per_sample = 2  # 16-bit = 2 bytes
            frame_size = int(self.sample_rate * frame_duration_ms / 1000) * bytes_per_sample
            
            # Track total speech duration found
            total_speech_duration_ms = 0
            offset = 0
            
            # Process 30ms frames
            while offset + frame_size <= len(pcm16_frame):
                frame = pcm16_frame[offset:offset + frame_size]
                
                # Check this frame with WebRTC VAD
                if self.vad.is_speech(frame, self.sample_rate):
                    total_speech_duration_ms += frame_duration_ms
                    
                    # Early exit if we've already found enough speech
                    if total_speech_duration_ms >= self.speech_duration_ms:
                        return VoiceState.SPEECH
                
                offset += frame_size
            
            # Check remaining partial frame if it exists and is at least 10ms
            if offset < len(pcm16_frame):
                remaining_bytes = len(pcm16_frame) - offset
                min_frame_size = int(self.sample_rate * 10 / 1000) * bytes_per_sample  # 10ms minimum
                
                if remaining_bytes >= min_frame_size:
                    # Determine frame duration for the remaining chunk
                    frame_20ms = int(self.sample_rate * 20 / 1000) * bytes_per_sample
                    if remaining_bytes >= frame_20ms:
                        frame = pcm16_frame[offset:offset + frame_20ms]
                        frame_duration = 20
                    else:
                        # Use 10ms frame
                        frame_10ms = int(self.sample_rate * 10 / 1000) * bytes_per_sample
                        frame = pcm16_frame[offset:offset + frame_10ms]
                        frame_duration = 10
                    
                    if self.vad.is_speech(frame, self.sample_rate):
                        total_speech_duration_ms += frame_duration
            
            # Return SPEECH only if we found enough speech duration
            return VoiceState.SPEECH if total_speech_duration_ms >= self.speech_duration_ms else VoiceState.SILENCE
            
        except Exception as e:
            logger.error(f"Error in WebRTC VAD detection: {e}")
            return VoiceState.SILENCE
