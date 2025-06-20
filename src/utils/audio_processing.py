# audio processing utils
import numpy as np
import io
import wave
import logging
import audioop
import librosa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



async def pcm2wav(pcm16: bytes, sample_rate: int = 8000) -> bytes:
    """convert pcm16 to wav"""
    # First ensure audio quality is good
    pcm16_array = np.frombuffer(pcm16, dtype=np.int16)
    
    # Basic noise gate with better threshold
    rms = np.sqrt(np.mean(pcm16_array.astype(np.float32)**2))
    logger.debug(f"Audio RMS level: {rms}")
    
    # If audio is too quiet, don't process
    if rms < 100:  # Increased threshold to filter out more noise
        logger.warning(f"Audio too quiet (RMS: {rms}), returning empty response")
        return None
        
    # Check for clipping or distortion
    peak_level = np.max(np.abs(pcm16_array))
    if peak_level > 30000:  # Close to 16-bit limit (32767)
        logger.warning(f"Audio may be clipped, peak level: {peak_level}")
    
    # Check for potential noise patterns
    if np.std(pcm16_array.astype(np.float32)) < 10:
        logger.warning(f"Low audio variation detected, possible noise or silent audio")
        return None
            
    # Create WAV from processed audio - do this as efficiently as possible
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16_array.tobytes())
    
    return buf.getvalue()


async def resample_pcm16(pcm16: bytes, original_sample_rate: int = 24000, target_sample_rate: int = 8000) -> bytes:
    """resample pcm16 to 8000hz, using librosa"""
    pcm16_array = np.frombuffer(pcm16, dtype=np.int16)
    
    # Convert to float32 for librosa (normalize to [-1, 1] range)
    float_array = pcm16_array.astype(np.float32) / 32768.0
    
    # Resample using librosa with better quality settings
    resampled_float = librosa.resample(
        float_array, 
        orig_sr=original_sample_rate, 
        target_sr=target_sample_rate,
        res_type='kaiser_best'  # Higher quality resampling
    )
    
    # Convert back to int16 with proper clipping to avoid overflow
    resampled_float = np.clip(resampled_float, -1.0, 1.0)
    resampled_int16 = (resampled_float * 32767.0).astype(np.int16)  # Use 32767 instead of 32768
    
    return resampled_int16.tobytes()


