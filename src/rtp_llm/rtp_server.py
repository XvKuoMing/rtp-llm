import asyncio
import socket
import time
from typing import Optional
import logging
import struct

# Suppress numba debug logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

from .utils.rtp_processing import RTPPacket, AudioCodec
from .utils.audio_processing import resample_pcm16, pcm2wav
from .buffer.array_buffer import BaseAudioBuffer
from .vad import BaseVAD
from .flow import BaseChatFlowManager
from .agents import VoiceAgent
from .audio_logger import AudioLogger

DEFAULT_SAMPLE_RATE = 8000
DEFAULT_MIN_PACK_DURATION = 1 # seconds
DEFAULT_MAX_PACK_DURATION = 9 # seconds
MAX_PAYLOAD_SIZE = 1024 # bytes


class RTPServer:
    

    def __init__(self, 
                 buffer: BaseAudioBuffer,
                 vad: BaseVAD,
                 flow: BaseChatFlowManager,
                 agent: VoiceAgent,
                 host_ip: str, host_port: int, 
                 peer_ip: Optional[str] = None, peer_port: Optional[int] = None,
                 tts_response_format: str = "pcm",
                 tts_codec: str | AudioCodec = "pcm16",
                 target_codec: str | AudioCodec = "pcm16",
                 tts_sample_rate: int = 24_000,
                 target_sample_rate: int = 24_000,
                 silence_interval: float = 0.02,  # Send silence for 20ms if if no response
                 audio_logger: Optional[AudioLogger] = None
                 ):
        """
        rtp server that can send and receive rtp packets
        it is meant to have only one peer
        """
        # agent settings
        self.buffer = buffer
        self.vad = vad
        self.flow = flow
        self.agent = agent
        self.audio_logger = audio_logger

        # tts settings
        self.tts_response_format = tts_response_format
        self.tts_codec = tts_codec
        self.target_codec = target_codec
        self.tts_sample_rate = tts_sample_rate
        self.target_sample_rate = target_sample_rate

        # silence settings
        self.silence_interval = silence_interval
        self.agent_task = None

        # network settings
        self.host_ip = host_ip
        self.host_port = host_port
        self.peer_ip = peer_ip
        self.peer_port = peer_port
        self.sock = None # will be initialized in _init_socket
        self.__init_socket()

        # rtp sending settings
        self.__ssrc = int(time.time()) & 0xFFFFFFFF
        self.__last_seq_num = 0

    def __init_socket(self):
        """Initialize socket with error handling"""
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                # Increase socket receive buffer to prevent drops
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB
                
                # Try to bind to the port
                self.sock.bind((self.host_ip, self.host_port))
                self.sock.setblocking(False)
                
                logger.info(f"Successfully initialized RTP server on {self.host_ip}:{self.host_port}")
                logger.debug(f"Socket buffer size: {self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)} bytes")
                logger.debug(f"Socket blocking mode: {self.sock.getblocking()}")
                logger.debug(f"Socket family: {self.sock.family}, type: {self.sock.type}")
                return
                
            except OSError as e:
                logger.warning(f"Socket bind attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise OSError(f"Failed to bind to port {self.host_port} after {max_retries} attempts")

    async def run(self, 
                  first_message: str = None,
                  min_frame_size: int = DEFAULT_SAMPLE_RATE * DEFAULT_MIN_PACK_DURATION * 2, # 2 = 16-bit samples
                  max_frame_size: int = DEFAULT_SAMPLE_RATE * DEFAULT_MAX_PACK_DURATION * 2, # 2 = 16-bit samples
                  allow_interruptions: bool = False
                  ):
        """
        main rtp loop for sending and receiving packets

        start_first_packet - if True and peer_ip and peer_port are set, the server will send a first packet to the peer
        if peer_ip and peer_port are not set, the server will consider firstly encountered peer (first packet's source)
        """
        logger.info(f"Starting RTP server on {self.host_ip}:{self.host_port} with min_frame_size={min_frame_size}")
        logger.debug(f"Peer configured as {self.peer_ip}:{self.peer_port}")
        
        if first_message:
            if (self.peer_ip is None) or (self.peer_port is None):
                logger.warning("peer_ip or peer_port is not set, will try to send first message on discovery")
            else:
                logger.info(f"Sending first message to peer {self.peer_ip}:{self.peer_port}")
                self.agent_task = asyncio.create_task(self._speak(first_message))
                first_message = None
        
        while True:
            
            try:
                # sending silence packet if not speaking
                if self.agent_task is None or self.agent_task.done():
                    await self._send_silence_packet()
                
                # receiving packets using recvfrom for UDP
                try:
                    data, address = await asyncio.wait_for(
                        asyncio.get_event_loop().sock_recvfrom(self.sock, MAX_PAYLOAD_SIZE),
                        timeout=0.1  # 100ms timeout
                    )
                    if not data:
                        logger.debug("Received empty packet")
                        continue
                    logger.debug(f"*** PACKET RECEIVED *** size={len(data)} bytes from {address} on {self.host_ip}:{self.host_port}")
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                except (OSError, ValueError) as e:
                    logger.error(f"Error receiving data: {e}")
                    await asyncio.sleep(0.1)
                    continue
                
                peer_host, peer_port = address
                
                logger.debug(f"Received packet of size {len(data)} bytes from {peer_host}:{peer_port}")

                # setting peer host and port if we have not set them yet
                if self.peer_ip is None:
                    logger.info(f"First packet received from {peer_host}:{peer_port}, setting as peer")
                    self.peer_ip = peer_host
                if self.peer_port is None:
                    self.peer_port = peer_port
                
                # checking if the packet is from the correct peer
                if peer_host == "127.0.0.1" and self.peer_ip == "localhost":
                    peer_host = "localhost"
                if peer_host == "localhost" and self.peer_ip == "127.0.0.1":
                    peer_host = "127.0.0.1"
                if peer_port != self.peer_port or peer_host != self.peer_ip:
                    logger.warning(f"Ignoring packet from unexpected peer {peer_host}:{peer_port}, expected {self.peer_ip}:{self.peer_port}")
                    continue
                
                if first_message:
                    logger.info("Sending first message to newly discovered peer")
                    self.agent_task = asyncio.create_task(self._speak(first_message))
                    first_message = None
                    
                # Process packet in background to avoid blocking the receive loop
                asyncio.create_task(self._handle_packet(data, min_frame_size, max_frame_size, allow_interruptions))
                
            except Exception as e:
                logger.error(f"Error in main RTP loop on {self.host_ip}:{self.host_port}: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Prevent tight loop on error
    
    async def _handle_packet(self, data: bytes, min_frame_size: int, max_frame_size: int, allow_interruptions: bool):
        """
        handle received packet
        """
        try:
            rtp_packet = RTPPacket.from_bytes(data)
            logger.debug(f"Processing RTP packet: seq={rtp_packet.header.sequence_number}, timestamp={rtp_packet.header.timestamp}, ssrc={rtp_packet.header.ssrc}")
            
            try:
                pcm16_frame = rtp_packet.enforce_pcm16()
            except ValueError as e:
                logger.warning(f"Error enforcing PCM16 audio data: {e}; dropping packet")
                return
            
            # Always add frame to buffer first
            self.buffer.add_frame(pcm16_frame)
            
            # Get accumulated frames from buffer
            pcm16_frames = self.buffer.get_frames()
            logger.debug(f"Buffer now contains {len(pcm16_frames)} bytes of audio data")
            
            # Only process if we have enough accumulated data
            if len(pcm16_frames) < min_frame_size:
                logger.debug(f"Not enough data yet: {len(pcm16_frames)} bytes < {min_frame_size} min size")
                return
            
            if len(pcm16_frames) > max_frame_size:
                logger.warning(f"Received packet of size {len(pcm16_frames)} bytes, which is greater than max_frame_size={max_frame_size}")
                asyncio.create_task(self._run_agent(pcm16_frames))
                self.buffer.clear()
                await self.flow.reset()
                return
            
            # Run VAD on the accumulated frames (last second of audio)
            pcm16_frames = pcm16_frames[min_frame_size:] if len(pcm16_frames) > min_frame_size else pcm16_frames
            voice_state = await self.vad.detect(pcm16_frames)
            logger.debug(f"VAD state: {voice_state}")
            if (self.agent_task is None or self.agent_task.done()) or allow_interruptions:
                start_agent = await self.flow.run_agent(voice_state)
            else:
                start_agent = False

            if start_agent:
                if self.agent_task and not self.agent_task.done():
                    self.agent_task.cancel()
                    self.agent_task = None
                    logger.info("Agent task cancelled due to new agent task")
                logger.info("Starting agent processing")
                self.agent_task = asyncio.create_task(self._run_agent(pcm16_frames))
                self.buffer.clear()
                
        except Exception as e:
            logger.error(f"Error handling packet: {e}", exc_info=True)
    

    async def _run_agent(self, pcm16_frames: bytes):
        """
        Run agent: STT -> TTS -> RTP transmission
        
        This function:
        1. Converts received audio to text using STT
        2. Generates TTS response as PCM chunks
        3. Converts PCM to μ-law and sends via RTP
        4. Manages sequence numbers and timestamps correctly
        """
        try:
            # Log user audio when starting agent
            if self.audio_logger:
                await self.audio_logger.log_user(pcm16_frames)
            
            # Step 1: Speech-to-Text
            wav_bytes = await pcm2wav(pcm16_frames)
            if not wav_bytes:
                logger.warning("Empty WAV bytes, skipping STT")
                return
            stt_response = await self.agent.stt(
                wav_bytes, 
                stream=False, 
                enable_history=True
            )
            logger.info(f"STT response: {stt_response}")
            
            if not stt_response or not stt_response.strip():
                logger.warning("Empty STT response, skipping TTS")
                return
            
            # Step 2: Text-to-Speech streaming
            await self._speak(stt_response)
                       
        except Exception as e:
            logger.error(f"Error in _run_agent: {e}")
        
    
    async def _speak(self, 
                     message: str):
        send_packet = self.__start_transmission(format=self.tts_codec, target=self.target_codec)
        send_packet.send(None) # initializing the coroutine

        chunk_count = 0
        tts_stream = await self.agent.tts(message, stream=True, response_format=self.tts_response_format)
        async for chunk in tts_stream:
            if not chunk:
                continue
            
            try:
                
                if self.tts_sample_rate != self.target_sample_rate:
                    if self.tts_codec == "pcm16":
                        # Ensure chunk size is even (2 bytes per sample for PCM16)
                        if len(chunk) % 2 != 0:
                            logger.warning(f"Odd-sized audio chunk detected: {len(chunk)} bytes, padding with zero")
                            chunk = chunk + b'\x00'  # Pad with a zero byte instead of truncating
                        
                        # Only resample if we have actual audio data
                        if len(chunk) > 0:
                            chunk = await resample_pcm16(chunk, self.tts_sample_rate, self.target_sample_rate)
                        else:
                            logger.warning("Empty audio chunk after size check, skipping")
                            continue
                    else:
                        logger.warning(f"Resampling of {self.tts_codec} to {self.target_codec} is not supported yet")
                        continue
                
                # Log AI audio chunk (use resampled chunk if available, otherwise original)
                if self.audio_logger:
                    await self.audio_logger.log_ai(chunk)
                
                send_packet.send(chunk)
                await asyncio.sleep(0.02) # Increased delay to reduce network congestion
                chunk_count += 1
            
            except Exception as e:
                logger.error(f"Error processing TTS chunk: {e}")
                continue
        
        # Store last sequence number for next agent run
        self.__last_seq_num -= 1
        
        # Save audio logs after speaking is complete (checkpoint)
        if self.audio_logger:
            try:
                await self.audio_logger.save()
                logger.info("Audio logs saved successfully")
            except Exception as e:
                logger.error(f"Error saving audio logs: {e}")
        
        logger.info(f"Agent completed: sent {chunk_count} RTP packets")

    async def _send_silence_packet(self):
        """Send a silence RTP packet to maintain connection"""
        if not self.peer_ip or not self.peer_port:
            return
            
        # Create silence audio data (zeros) - 20ms at target sample rate
        samples_per_packet = int(self.target_sample_rate * self.silence_interval)
        silence_data = struct.pack('<' + 'h' * samples_per_packet, *([0] * samples_per_packet))
        
        send_packet = self.__start_transmission(format="pcm16", 
                                                target=self.target_codec)
        send_packet.send(None)  # Initialize the coroutine
        send_packet.send(silence_data)
        logger.debug(f"Sent silence packet: {len(silence_data)} bytes")
        await asyncio.sleep(self.silence_interval)



    def __start_transmission(self, 
                     format: str | AudioCodec, 
                     target: str | AudioCodec
                     ):
        """
        send rtp packet corotine
        accepts format and target as strings [pcm16, ulaw, alaw] or AudioCodec enums [AudioCodec.L16_1CH, AudioCodec.PCMU, AudioCodec.PCMA]
        """
        # initializing RTP transmission state
        sequence_number = self.__last_seq_num + 1
        timestamp = int(time.time() * DEFAULT_SAMPLE_RATE)  # Convert to sample units
        ssrc = self.__ssrc
        first_chunk = True

        total_samples_sent = 0

        while True:
            chunk = yield

            # Split large chunks into smaller packets to avoid UDP fragmentation
            chunk_offset = 0
            while chunk_offset < len(chunk):
                # Calculate chunk size for this packet
                remaining_data = len(chunk) - chunk_offset
                current_chunk_size = min(remaining_data, MAX_PAYLOAD_SIZE)
                current_chunk = chunk[chunk_offset:chunk_offset + current_chunk_size]

                rtp = RTPPacket.enforce_rtp(
                    data=current_chunk,
                    format=format, target=target,
                    sequence_number=sequence_number & 0xFFFF,  # 16-bit sequence number
                    timestamp=(timestamp + total_samples_sent) & 0xFFFFFFFF,  # 32-bit timestamp
                    ssrc=ssrc,
                    marker=first_chunk  # Mark first packet of utterance
                )

                # Send the packet
                if self.peer_ip and self.peer_port:
                    logger.debug(f"*** SENDING RTP PACKET *** to {self.peer_ip}:{self.peer_port}, size={len(rtp.as_bytes)} bytes, seq={sequence_number}")
                    self.sock.sendto(rtp.as_bytes, (self.peer_ip, self.peer_port))
                else:
                    logger.warning("Cannot send RTP packet: peer address not set")
                    break

                sequence_number += 1
                chunk_offset += current_chunk_size
                
                # Update total samples sent
                current_samples = len(current_chunk) // 2 \
                                if format == "pcm16" or format in [AudioCodec.L16_1CH, AudioCodec.L16_2CH] \
                                else len(current_chunk)
                total_samples_sent += current_samples
                
                first_chunk = False



    def close(self):
        """Close the RTP server and cleanup resources"""
        logger.info("Closing RTP server")
        try:
            if self.sock:
                self.sock.close()
                logger.info("Socket closed successfully")
        except Exception as e:
            logger.error(f"Error closing socket: {e}")