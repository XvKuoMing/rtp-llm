import os
import asyncio
import logging
from rtp_llm.adapters import RTPAdapter
from rtp_llm.buffer import ArrayBuffer
from rtp_llm.flow import CopyFlowManager
from rtp_llm.history import ChatHistoryLimiter
from rtp_llm.vad import WebRTCVAD
from rtp_llm.providers import OpenAIProvider, GeminiSTTProvider
from rtp_llm.agents import VoiceAgent
from rtp_llm.server import Server


# Suppress numba debug logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(".env.test")

async def create_rtp_server(host_ip: str, host_port: int, peer_ip: str, peer_port: int, name: str, voice: str = "nova"):
    """Create and configure an RTP server instance"""
    
    # Create RTP adapter
    adapter = RTPAdapter(
        host_ip=host_ip,
        host_port=host_port,
        peer_ip=peer_ip,
        peer_port=peer_port,
        sample_rate=8_000,  # Match VAD sample rate
        target_codec="pcm"
    )
    
    # Create voice agent
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

    # Create server with all required components
    server = Server(
        adapter=adapter,
        audio_buffer=ArrayBuffer(),
        flow_manager=CopyFlowManager(),
        vad=WebRTCVAD(sample_rate=8000, aggressiveness=3, min_speech_duration_ms=500),
        agent=voice_agent,
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
            server1.run(first_message="Hello, how can I help you today?"),
            server2.run()
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