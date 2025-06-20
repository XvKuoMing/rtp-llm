import os
import asyncio
import logging
from src.buffer import ArrayBuffer
from src.flow import CopyFlowManager
from src.history import ChatHistoryLimiter
from src.vad import WebRTCVAD
from src.providers import OpenAIProvider, GeminiSTTProvider
from src.agents import VoiceAgent
from src.rtp_server import RTPServer
from src.audio_logger import AudioLogger
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(".env.test")

async def create_rtp_server(host_ip: str, host_port: int, peer_ip: str, peer_port: int, name: str, voice: str = "nova"):
    """Create and configure an RTP server instance"""
    voice_agent = VoiceAgent(
        stt_provider=GeminiSTTProvider(
            api_key=os.getenv("GEMINI_API_KEY"), 
            base_url=os.getenv("GEMINI_BASE_URL"),
            model=os.getenv("GEMINI_MODEL"),
            system_prompt=f"You are {name}"
        ),
        tts_provider=OpenAIProvider(
            api_key=os.getenv("OPENAI_TTS_API_KEY"), 
            base_url=os.getenv("OPENAI_TTS_BASE_URL"),
            tts_model=os.getenv("OPENAI_TTS_MODEL"),
        ),
        history_manager=ChatHistoryLimiter(limit=7),
        tts_gen_config={"voice": voice}
    )

    server = RTPServer(
        buffer=ArrayBuffer(),
        agent=voice_agent,
        vad=WebRTCVAD(sample_rate=8000, aggressiveness=3, min_speech_duration_ms=500),
        flow=CopyFlowManager(),
        host_ip=host_ip,
        host_port=host_port,
        peer_ip=peer_ip,
        peer_port=peer_port,
        tts_response_format="pcm",
        tts_codec="pcm16",
        target_codec="pcm16",
        tts_sample_rate=24_000,
        target_sample_rate=8_000,
        audio_logger=AudioLogger(uid=random.randint(4, 8))
    )
    
    logger.info(f"Created {name} RTP server on {host_ip}:{host_port} with peer {peer_ip}:{peer_port}")
    return server

async def main():
    # Create two RTP servers that will communicate with each other
    server1 = await create_rtp_server(
        host_ip="127.0.0.1",
        host_port=5000,
        peer_ip="127.0.0.1",
        peer_port=5001,
        name="Server 1",
        voice="nova"
    )
    
    server2 = await create_rtp_server(
        host_ip="127.0.0.1",
        host_port=5001,
        peer_ip="127.0.0.1",
        peer_port=5000,
        name="Server 2",
        voice="shimmer"
    )
    
    try:
        # Start both servers
        logger.info("Starting RTP servers...")
        await asyncio.gather(
            server1.run(first_message="Hello from Server 1! I'm waiting for you to say something."
                        ),
            server2.run(first_message=None)
        )
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
        server1.close()
        server2.close()
        logger.info("Servers stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test terminated by user") 