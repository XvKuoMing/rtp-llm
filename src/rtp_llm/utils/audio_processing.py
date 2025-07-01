# audio processing utils
import numpy as np
import io
import wave
import logging
import librosa
import audioop


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


try:
    import opuslib
except Exception as e:
    logger.error(f"Error importing opuslib: {e}")
    opuslib = None

async def pcm2wav(pcm16: bytes, sample_rate: int = 8000) -> bytes:
    """convert pcm16 to wav"""
    # First ensure audio quality is good
    pcm16_array = np.frombuffer(pcm16, dtype=np.int16)
    
    # # Basic noise gate with better threshold
    # rms = np.sqrt(np.mean(pcm16_array.astype(np.float32)**2))
    # # logger.debug(f"Audio RMS level: {rms}")
    
    # # If audio is too quiet, don't process
    # if rms < 100:  # Increased threshold to filter out more noise
    #     logger.warning(f"Audio too quiet (RMS: {rms}), returning empty response")
    #     return None
        
    # # Check for clipping or distortion
    # peak_level = np.max(np.abs(pcm16_array))
    # if peak_level > 30000:  # Close to 16-bit limit (32767)
    #     logger.warning(f"Audio may be clipped, peak level: {peak_level}")
    
    # # Check for potential noise patterns
    # if np.std(pcm16_array.astype(np.float32)) < 10:
    #     logger.warning(f"Low audio variation detected, possible noise or silent audio")
    #     return None
            
    # Create WAV from processed audio - do this as efficiently as possible
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16_array.tobytes())
    
    return buf.getvalue()


async def pcm2ulaw(pcm16: bytes) -> bytes:
    """convert pcm16 to ulaw"""
    return audioop.lin2ulaw(pcm16, 2)

async def pcm2alaw(pcm16: bytes) -> bytes:
    """convert pcm16 to alaw"""
    return audioop.lin2alaw(pcm16, 2)


async def pcm2opus(pcm16: bytes, sample_rate: int = 8000) -> bytes:
    """convert pcm16 to opus"""
    encoder = opuslib.Encoder(sample_rate, channels=1)
    return encoder.encode(pcm16, frame_size=sample_rate//50)


async def resample_pcm16(pcm16: bytes, original_sample_rate: int = 24000, target_sample_rate: int = 8000) -> bytes:
    """resample pcm16 to target sample rate, using librosa with high quality settings"""
    if len(pcm16) % 2 != 0:
        logger.warning(f"pcm16 length is not even, padding with 0")
        # pcm16 = pcm16 + b'\x00'
        pcm16 = pcm16[:-1] # let's delete the last byte

    pcm16_array = np.frombuffer(pcm16, dtype=np.int16)
    
    # Convert to float32 for librosa (normalize to [-1, 1] range)
    float_array = pcm16_array.astype(np.float32) / 32768.0
    
    # Use higher quality resampling parameters
    resampled_float = librosa.resample(
        float_array, 
        orig_sr=original_sample_rate, 
        target_sr=target_sample_rate,
        res_type='kaiser_best',  # Higher quality resampling
        fix=True,               # Center the filter
        scale=False             # Don't scale to preserve energy
    )
    
    # Apply gentle normalization to prevent clipping while preserving dynamics
    peak = np.max(np.abs(resampled_float))
    if peak > 0.95:
        resampled_float = resampled_float * (0.95 / peak)
    
    # Convert back to int16 with proper clipping
    resampled_int16 = (resampled_float * 32767.0).clip(-32767, 32767).astype(np.int16)
    
    logger.info(f"Resampled from {original_sample_rate}Hz to {target_sample_rate}Hz, "
                f"samples: {len(pcm16_array)} -> {len(resampled_int16)}, "
                f"peak level: {peak:.3f}")
    
    return resampled_int16.tobytes()


async def ulaw2pcm(ulaw: bytes) -> bytes:
    """convert ulaw to pcm16 using audioop"""
    try:

        # Convert μ-law to 16-bit linear PCM using audioop
        # audioop.ulaw2lin(fragment, width) where width=2 for 16-bit
        pcm16_data = audioop.ulaw2lin(ulaw, 2)
        
        # logger.debug(f"Converted {len(ulaw)} μ-law bytes to {len(pcm16_data)} PCM16 bytes")
        return pcm16_data
        
    except Exception as e:
        logger.error(f"Error converting μ-law to PCM with audioop: {e}")


async def alaw2pcm(alaw: bytes) -> bytes:
    """convert alaw to pcm16 using audioop"""
    try:
        if audioop is None:
            raise ImportError("audioop module not available")
        
        # Convert A-law to 16-bit linear PCM using audioop  
        # audioop.alaw2lin(fragment, width) where width=2 for 16-bit
        pcm16_data = audioop.alaw2lin(alaw, 2)
        
        # logger.debug(f"Converted {len(alaw)} A-law bytes to {len(pcm16_data)} PCM16 bytes")
        return pcm16_data
        
    except Exception as e:
        logger.error(f"Error converting A-law to PCM with audioop: {e}")



async def opus2pcm(opus: bytes, sample_rate: int = 8000) -> bytes:
    """convert opus to pcm16"""
    try:
        # Create Opus decoder
        decoder = opuslib.Decoder(sample_rate, channels=1)
        
        # Decode Opus to PCM
        pcm_data = decoder.decode(opus, frame_size=sample_rate//50)  # 20ms frames typical
        
        # Convert to int16 if needed
        if isinstance(pcm_data, bytes):
            return pcm_data
        else:
            # If it's float, convert to int16
            pcm_array = np.array(pcm_data, dtype=np.float32)
            pcm16_array = (pcm_array * 32767).clip(-32768, 32767).astype(np.int16)
            return pcm16_array.tobytes()
            
    except Exception as e:
        logger.error(f"Error converting Opus to PCM: {e}")