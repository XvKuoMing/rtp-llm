import asyncio
import os
import wave
from typing import Optional, Dict, Any
from datetime import datetime

from ..models.audio import AudioFileInfo


def parse_audio_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse audio filename to extract UID and timestamp"""
    if not filename.endswith('.wav'):
        return None
    
    # Expected format: {uid}_conversation_{timestamp}.wav
    try:
        name_without_ext = filename[:-4]  # Remove .wav
        parts = name_without_ext.split('_conversation_')
        if len(parts) != 2:
            return None
        
        uid = parts[0]
        timestamp = float(parts[1])
        
        return {
            'uid': uid,
            'timestamp': timestamp
        }
    except (ValueError, IndexError):
        return None


async def get_audio_file_info(filepath: str) -> Optional[AudioFileInfo]:
    """Get detailed information about an audio file (async)"""
    try:
        filename = os.path.basename(filepath)
        parsed = parse_audio_filename(filename)
        if not parsed:
            return None
        
        # Use asyncio.to_thread for blocking file I/O operations
        if not await asyncio.to_thread(os.path.exists, filepath):
            return None
        
        file_size = await asyncio.to_thread(os.path.getsize, filepath)
        created_date = datetime.fromtimestamp(parsed['timestamp']).isoformat()
        
        # Try to get audio metadata asynchronously
        duration_seconds = None
        sample_rate = None
        channels = None
        
        try:
            # Run the wave file reading in a thread to avoid blocking
            audio_metadata = await asyncio.to_thread(_read_wave_metadata, filepath)
            if audio_metadata:
                frames, sample_rate, channels = audio_metadata
                duration_seconds = frames / float(sample_rate) if sample_rate > 0 else None
        except Exception as e:
            # logger would be available through server_manager if needed
            pass
        
        return AudioFileInfo(
            filename=filename,
            uid=parsed['uid'],
            conversation_timestamp=parsed['timestamp'],
            file_size=file_size,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            created_date=created_date,
            file_path=filepath
        )
    except Exception as e:
        return None


def _read_wave_metadata(filepath: str) -> Optional[tuple]:
    """Synchronous helper function to read wave file metadata"""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            return (frames, sample_rate, channels)
    except Exception:
        return None
