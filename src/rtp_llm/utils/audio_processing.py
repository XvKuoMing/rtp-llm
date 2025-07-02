# audio processing utils
import numpy as np
import io
import wave
import logging
import audioop
from math import gcd
from scipy.signal import resample_poly


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
    ulaw_data = audioop.lin2ulaw(pcm16, 2)
    logger.info(f"Converted {len(pcm16)} PCM16 bytes to {len(ulaw_data)} ulaw bytes")
    return ulaw_data

async def pcm2alaw(pcm16: bytes) -> bytes:
    """convert pcm16 to alaw"""
    alaw_data = audioop.lin2alaw(pcm16, 2)
    logger.info(f"Converted {len(pcm16)} PCM16 bytes to {len(alaw_data)} alaw bytes")
    return alaw_data


async def pcm2opus(pcm16: bytes, sample_rate: int = 8000) -> bytes:
    """convert pcm16 to opus"""
    encoder = opuslib.Encoder(sample_rate, channels=1)
    return encoder.encode(pcm16, frame_size=sample_rate//50)



class StreamingResample:

    def __init__(self, original_sample_rate: int, target_sample_rate: int):
        g = gcd(original_sample_rate, target_sample_rate)
        self.up = target_sample_rate // g
        self.down = original_sample_rate // g
        # self.zi = None # not used
        self.buffer = b''
    
    async def resample_pcm16(self, pcm16: bytes):
        
        if len(pcm16) % 2 != 0:
            logger.info(f"Input pcm16 length is not even ({len(pcm16)}), appending to deque")
            self.buffer += pcm16[-1:]
            pcm16 = pcm16[:-1]
        
        if self.buffer:
            logger.info(f"Joining buffer with pcm16, buffer length: {len(self.buffer)}")
            pcm16 = self.buffer + pcm16
            self.buffer = b''
        
        if not pcm16:
            logger.info("No valid audio data, returning empty bytes")
            return b''

        pcm16_array = np.frombuffer(pcm16, dtype=np.int16)
        pcm16_array = pcm16_array.astype(np.float32) / 32768.0 # convert and normalize to [-1, 1]
        out = resample_poly(
            pcm16_array, 
            self.up, 
            self.down, 
            padtype='line', 
            axis=0, 
            window=('kaiser', 14)
            )
        if isinstance(out, np.ndarray):
            out_i16 = (out.clip(-1, 1) * 32767).astype(np.int16) # convert back to int16
            return out_i16.tobytes()
        else:
            logger.warning("Resampled output is not a numpy array, returning empty bytes")
            return b''


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