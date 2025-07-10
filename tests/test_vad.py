#TODO: write appropriate tests


# import sys
# import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import pytest
# import pytest_asyncio
# import contextlib
# import wave

# from src.rtp_llm.vad import VoiceState, WebRTCVAD, SileroVAD
# from src.rtp_llm.utils.audio_processing import resample_pcm16
# from typing import Tuple

# @pytest.fixture
# def webrtc_vad() -> WebRTCVAD:
#     return WebRTCVAD(sample_rate=8000, aggressiveness=3, min_speech_duration_ms=500)


# @pytest.fixture
# def silero_vad() -> SileroVAD:
#     return SileroVAD(sample_rate=8000, threshold=0.5)


# @pytest_asyncio.fixture
# async def speech_data() -> Tuple[bytes, int]:
#     """Reads a .wav file.

#     Takes the path, and returns (PCM audio data, sample rate).
#     """
#     with contextlib.closing(wave.open("tests/test_audio/greetings.wav", 'rb')) as wf:
#         num_channels = wf.getnchannels()
#         assert num_channels == 1
#         sample_width = wf.getsampwidth()
#         assert sample_width == 2
#         sample_rate = wf.getframerate()
#         pcm_data = wf.readframes(wf.getnframes())
#         if sample_rate not in (8000, 16000):
#             resampler = StreamingResample(sample_rate, 8000)
#             pcm_data = resampler.resample(pcm_data)
#         return pcm_data

# @pytest.mark.asyncio
# async def test_webrtc_vad(webrtc_vad: WebRTCVAD, speech_data: bytes):
#     vad_state = await webrtc_vad.detect(speech_data)
#     assert vad_state == VoiceState.SPEECH

# @pytest.mark.asyncio
# async def test_silero_vad(silero_vad: SileroVAD, speech_data: bytes):
#     vad_state = await silero_vad.detect(speech_data)
#     assert vad_state == VoiceState.SPEECH


