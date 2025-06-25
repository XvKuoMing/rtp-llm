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
    level=logging.INFO,  # Default to INFO level, will be overridden by RTPServer debug param
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

from .utils.rtp_processing import RTPPacket, AudioCodec
from .utils.audio_processing import resample_pcm16, pcm2wav, wav2pcm
from .buffer.array_buffer import BaseAudioBuffer
from .vad import BaseVAD
from .flow import BaseChatFlowManager
from .agents import VoiceAgent
from .audio_logger import AudioLogger
from .providers import Message

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
                 peer_ip: Optional[str] = None, 
                 peer_port: Optional[int] = None,
                 tts_response_format: str = "pcm",
                 tts_codec: str | AudioCodec = "pcm16",
                 target_codec: str | AudioCodec = "pcm16",
                 tts_sample_rate: int = 24_000,
                 target_sample_rate: int = 24_000,
                 silence_interval: float = 0.02,  # Send silence for 20ms if no response
                 audio_logger: Optional[AudioLogger] = None,
                 debug: bool = False
                 ):
        """
        rtp server that can send and receive rtp packets
        it is meant to have only one peer
        """
        # Configure logging level based on debug parameter
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.info("=== DEBUG MODE ENABLED ===")
        else:
            logger.setLevel(logging.INFO)
        
        logger.info("=== INITIALIZING RTP SERVER ===")
        logger.debug(f"INIT DEBUG - Debug mode: {self.debug}")
        
        # agent settings
        self.buffer = buffer
        self.vad = vad
        self.flow = flow
        self.agent = agent
        self.audio_logger = audio_logger
        
        # Debug component initialization
        logger.debug(f"COMPONENT DEBUG - Buffer type: {type(buffer).__name__}")
        logger.debug(f"COMPONENT DEBUG - VAD type: {type(vad).__name__}")
        logger.debug(f"COMPONENT DEBUG - Flow type: {type(flow).__name__}")
        logger.debug(f"COMPONENT DEBUG - Agent type: {type(agent).__name__}")
        logger.debug(f"COMPONENT DEBUG - Audio logger: {'Enabled' if audio_logger else 'Disabled'}")

        # tts settings
        self.tts_response_format = tts_response_format
        self.tts_codec = tts_codec
        self.target_codec = target_codec
        self.tts_sample_rate = tts_sample_rate
        self.target_sample_rate = target_sample_rate
        
        # Debug codec configuration
        logger.info("=== CODEC CONFIGURATION DEBUG ===")
        logger.info(f"CODEC DEBUG - TTS Response Format: {self.tts_response_format}")
        logger.info(f"CODEC DEBUG - TTS Codec: {self.tts_codec} (type: {type(self.tts_codec)})")
        logger.info(f"CODEC DEBUG - Target Codec: {self.target_codec} (type: {type(self.target_codec)})")
        logger.info(f"CODEC DEBUG - TTS Sample Rate: {self.tts_sample_rate}Hz")
        logger.info(f"CODEC DEBUG - Target Sample Rate: {self.target_sample_rate}Hz")
        logger.info(f"CODEC DEBUG - Sample Rate Conversion Required: {self.tts_sample_rate != self.target_sample_rate}")
        
        # Calculate frame size information for debugging
        self.bytes_per_sample = 2 if (self.target_codec == "pcm16" or isinstance(self.target_codec, AudioCodec) and self.target_codec in [AudioCodec.L16_1CH, AudioCodec.L16_2CH]) else 1
        logger.info(f"CODEC DEBUG - Bytes per sample for target codec: {self.bytes_per_sample}")
        logger.info(f"CODEC DEBUG - 20ms frame size at {self.target_sample_rate}Hz: {int(self.target_sample_rate * 0.02) * self.bytes_per_sample} bytes")

        # silence settings
        self.silence_interval = silence_interval
        self.agent_task = None
        logger.debug(f"SILENCE DEBUG - Silence interval: {self.silence_interval}s")

        # network settings
        self.host_ip = host_ip
        self.host_port = host_port
        self.peer_ip = peer_ip
        self.peer_port = peer_port
        self.sock = None # will be initialized in _init_socket
        
        logger.info("=== NETWORK CONFIGURATION DEBUG ===")
        logger.info(f"NETWORK DEBUG - Host: {self.host_ip}:{self.host_port}")
        logger.info(f"NETWORK DEBUG - Peer: {self.peer_ip}:{self.peer_port} ({'configured' if self.peer_ip and self.peer_port else 'will be discovered'})")
        
        self.__init_socket()

        # RTP sending settings - FIXED: Initialize proper RTP state for continuous stream
        self.__ssrc = int(time.time()) & 0xFFFFFFFF
        self.__sequence_number = 0  # Start from 0 for compatibility
        self.__timestamp_base = 0   # Initialize to 0, will be set on first packet
        self.__total_samples_sent = 0  # Track total samples for continuous timestamps
        self.__rtp_stream_active = False  # Track if we have an active RTP stream
        
        logger.info("=== RTP CONFIGURATION DEBUG ===")
        logger.info(f"RTP DEBUG - SSRC: 0x{self.__ssrc:08X} ({self.__ssrc})")
        logger.info(f"RTP DEBUG - Initial sequence number: {self.__sequence_number}")
        logger.info(f"RTP DEBUG - Max payload size: {MAX_PAYLOAD_SIZE} bytes")
        logger.info("=== RTP SERVER INITIALIZATION COMPLETE ===")

    def __init_socket(self):
        """Initialize socket with error handling"""
        logger.info("=== SOCKET INITIALIZATION DEBUG ===")
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"SOCKET DEBUG - Attempt {attempt + 1}/{max_retries}")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                logger.debug(f"SOCKET DEBUG - Created UDP socket: {self.sock}")
                
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                logger.debug("SOCKET DEBUG - Set SO_REUSEADDR option")
                
                # Increase socket receive buffer to prevent drops
                old_buffer_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB
                new_buffer_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                logger.debug(f"SOCKET DEBUG - Buffer size changed: {old_buffer_size} -> {new_buffer_size} bytes")
                
                # Try to bind to the port
                logger.debug(f"SOCKET DEBUG - Attempting to bind to {self.host_ip}:{self.host_port}")
                self.sock.bind((self.host_ip, self.host_port))
                logger.debug(f"SOCKET DEBUG - Successfully bound to {self.host_ip}:{self.host_port}")
                
                self.sock.setblocking(False)
                logger.debug("SOCKET DEBUG - Set socket to non-blocking mode")
                
                logger.info(f"SOCKET DEBUG - Successfully initialized RTP server on {self.host_ip}:{self.host_port}")
                logger.debug(f"SOCKET DEBUG - Final socket buffer size: {self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)} bytes")
                logger.debug(f"SOCKET DEBUG - Socket blocking mode: {self.sock.getblocking()}")
                logger.debug(f"SOCKET DEBUG - Socket family: {self.sock.family}, type: {self.sock.type}")
                return
                
            except OSError as e:
                logger.warning(f"SOCKET DEBUG - Bind attempt {attempt + 1}/{max_retries} failed: {e}")
                logger.debug(f"SOCKET DEBUG - Error details: type={type(e)}, errno={getattr(e, 'errno', 'unknown')}")
                
                if self.sock:
                    try:
                        self.sock.close()
                        logger.debug("SOCKET DEBUG - Closed failed socket")
                    except:
                        pass
                    self.sock = None
                
                if attempt < max_retries - 1:
                    logger.debug(f"SOCKET DEBUG - Waiting {retry_delay}s before retry")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"SOCKET DEBUG - Failed to bind after all attempts")
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
        logger.info("=== STARTING RTP SERVER MAIN LOOP ===")
        logger.info(f"RTP LOOP DEBUG - Host: {self.host_ip}:{self.host_port}")
        logger.debug(f"RTP LOOP DEBUG - Peer configured as {self.peer_ip}:{self.peer_port}")
        logger.info(f"RTP LOOP DEBUG - Frame sizes: min={min_frame_size} bytes, max={max_frame_size} bytes")
        logger.info(f"RTP LOOP DEBUG - Target sample rate: {self.target_sample_rate}Hz")
        logger.info(f"RTP LOOP DEBUG - Allow interruptions: {allow_interruptions}")
        
        # Calculate timing information
        min_duration_ms = (min_frame_size / 2) / (self.target_sample_rate / 1000)
        max_duration_ms = (max_frame_size / 2) / (self.target_sample_rate / 1000)
        logger.info(f"RTP LOOP DEBUG - Frame durations: min={min_duration_ms:.1f}ms, max={max_duration_ms:.1f}ms")
        
        if first_message:
            if (self.peer_ip is None) or (self.peer_port is None):
                logger.warning("FIRST MESSAGE DEBUG - Peer not configured, will send on discovery")
            else:
                logger.info(f"FIRST MESSAGE DEBUG - Sending to peer {self.peer_ip}:{self.peer_port}")
                logger.debug(f"FIRST MESSAGE DEBUG - Message: '{first_message}'")
                self.agent_task = asyncio.create_task(self._speak(first_message))
                await self.agent.history_manager.add_message(Message(role="user", content=first_message))
                first_message = None
        
        packet_count = 0
        silence_count = 0
        
        while True:
            
            try:
                # FIXED: Only send silence packet if agent is not running AND we have a peer
                # This prevents silence packets from interfering with speech
                agent_running = self.agent_task is not None and not self.agent_task.done()
                if not agent_running and self.peer_ip and self.peer_port:
                    await self._send_silence_packet()
                    silence_count += 1
                    if silence_count % 50 == 0:  # Log every 50 silence packets (every second)
                        logger.debug(f"SILENCE DEBUG - Sent {silence_count} silence packets")
                
                # receiving packets using recvfrom for UDP
                try:
                    data, address = await asyncio.wait_for(
                        asyncio.get_event_loop().sock_recvfrom(self.sock, MAX_PAYLOAD_SIZE),
                        timeout=0.1  # 100ms timeout
                    )
                    if not data:
                        logger.debug("RECEIVE DEBUG - Received empty packet")
                        continue
                    
                    packet_count += 1
                    logger.debug(f"*** PACKET RECEIVED #{packet_count} *** size={len(data)} bytes from {address} on {self.host_ip}:{self.host_port}")
                    logger.debug(f"PACKET DEBUG - Raw data preview: {data[:min(20, len(data))].hex()}")
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                except (OSError, ValueError) as e:
                    logger.error(f"RECEIVE DEBUG - Error receiving data: {e}")
                    await asyncio.sleep(0.1)
                    continue
                
                peer_host, peer_port = address
                
                logger.debug(f"PEER DEBUG - Packet from {peer_host}:{peer_port}")

                # setting peer host and port if we have not set them yet
                if self.peer_ip is None:
                    logger.info(f"PEER DISCOVERY DEBUG - First packet from {peer_host}:{peer_port}, setting as peer")
                    self.peer_ip = peer_host
                if self.peer_port is None:
                    logger.info(f"PEER DISCOVERY DEBUG - Setting peer port to {peer_port}")
                    self.peer_port = peer_port
                
                # checking if the packet is from the correct peer
                peer_host_normalized = peer_host
                expected_host_normalized = self.peer_ip
                
                if peer_host == "127.0.0.1" and self.peer_ip == "localhost":
                    peer_host_normalized = "localhost"
                if peer_host == "localhost" and self.peer_ip == "127.0.0.1":
                    peer_host_normalized = "127.0.0.1"
                if self.peer_ip == "127.0.0.1" and peer_host == "localhost":
                    expected_host_normalized = "localhost"
                if self.peer_ip == "localhost" and peer_host == "127.0.0.1":
                    expected_host_normalized = "127.0.0.1"
                
                logger.debug(f"PEER VALIDATION DEBUG - Normalized: received={peer_host_normalized}:{peer_port}, expected={expected_host_normalized}:{self.peer_port}")
                
                if peer_port != self.peer_port or peer_host_normalized != expected_host_normalized:
                    logger.warning(f"PEER VALIDATION DEBUG - Ignoring packet from unexpected peer {peer_host}:{peer_port}, expected {self.peer_ip}:{self.peer_port}")
                    continue
                
                if first_message:
                    logger.info("FIRST MESSAGE DEBUG - Sending to newly discovered peer")
                    logger.debug(f"FIRST MESSAGE DEBUG - Message: '{first_message}'")
                    self.agent_task = asyncio.create_task(self._speak(first_message))
                    await self.agent.history_manager.add_message(Message(role="user", content=first_message))
                    first_message = None
                    
                # Process packet in background to avoid blocking the receive loop
                logger.debug(f"PACKET PROCESSING DEBUG - Creating background task for packet #{packet_count}")
                asyncio.create_task(self._handle_packet(data, min_frame_size, max_frame_size, allow_interruptions))
                
            except Exception as e:
                logger.error(f"RTP LOOP DEBUG - Error in main loop on {self.host_ip}:{self.host_port}: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Prevent tight loop on error
    
    async def _handle_packet(self, data: bytes, min_frame_size: int, max_frame_size: int, allow_interruptions: bool):
        """
        handle received packet
        """
        logger.debug("=== PACKET HANDLING DEBUG ===")
        try:
            logger.debug(f"PACKET PARSE DEBUG - Processing {len(data)} bytes")
            rtp_packet = RTPPacket.from_bytes(data)
            
            # Debug RTP header information
            logger.debug("=== RTP HEADER DEBUG ===")
            logger.debug(f"RTP DEBUG - Version: {rtp_packet.header.version}")
            logger.debug(f"RTP DEBUG - Padding: {rtp_packet.header.padding}")
            logger.debug(f"RTP DEBUG - Extension: {rtp_packet.header.extension}")
            logger.debug(f"RTP DEBUG - CSRC count: {rtp_packet.header.csrc_count}")
            logger.debug(f"RTP DEBUG - Marker: {rtp_packet.header.marker}")
            logger.debug(f"RTP DEBUG - Payload type: {rtp_packet.header.payload_type}")
            logger.debug(f"RTP DEBUG - Sequence: {rtp_packet.header.sequence_number}")
            logger.debug(f"RTP DEBUG - Timestamp: {rtp_packet.header.timestamp}")
            logger.debug(f"RTP DEBUG - SSRC: 0x{rtp_packet.header.ssrc:08X}")
            logger.debug(f"RTP DEBUG - Payload size: {len(rtp_packet.payload)} bytes")
            
            try:
                logger.debug("CODEC DEBUG - Attempting to enforce PCM16 format")
                logger.debug(f"CODEC DEBUG - Original payload size: {len(rtp_packet.payload)} bytes")
                logger.debug(f"CODEC DEBUG - Original payload preview: {rtp_packet.payload[:min(10, len(rtp_packet.payload))].hex()}")
                
                pcm16_frame = rtp_packet.enforce_pcm16()
                
                logger.debug(f"CODEC DEBUG - PCM16 conversion successful")
                logger.debug(f"CODEC DEBUG - PCM16 frame size: {len(pcm16_frame)} bytes")
                logger.debug(f"CODEC DEBUG - PCM16 sample count: {len(pcm16_frame) // 2}")
                logger.debug(f"CODEC DEBUG - PCM16 preview: {pcm16_frame[:min(10, len(pcm16_frame))].hex()}")
                
            except ValueError as e:
                logger.warning(f"CODEC DEBUG - Error enforcing PCM16: {e}")
                logger.debug(f"CODEC DEBUG - Failed payload type: {rtp_packet.header.payload_type}")
                logger.debug(f"CODEC DEBUG - Failed payload size: {len(rtp_packet.payload)}")
                return
            
            # Buffer operations debugging
            logger.debug("=== BUFFER DEBUG ===")
            buffer_size_before = len(self.buffer.get_frames()) if hasattr(self.buffer, 'get_frames') else "unknown"
            logger.debug(f"BUFFER DEBUG - Size before adding frame: {buffer_size_before} bytes")
            
            # Always add frame to buffer first
            self.buffer.add_frame(pcm16_frame)
            logger.debug(f"BUFFER DEBUG - Added frame of {len(pcm16_frame)} bytes")
            
            # Get accumulated frames from buffer
            pcm16_frames = self.buffer.get_frames()
            logger.debug(f"BUFFER DEBUG - Total accumulated: {len(pcm16_frames)} bytes")
            logger.debug(f"BUFFER DEBUG - Sample count: {len(pcm16_frames) // 2}")
            duration_ms = (len(pcm16_frames) // 2) / (self.target_sample_rate / 1000)
            logger.debug(f"BUFFER DEBUG - Duration: {duration_ms:.1f}ms")
            
            # Only process if we have enough accumulated data
            if len(pcm16_frames) < min_frame_size:
                logger.debug(f"BUFFER DEBUG - Not enough data: {len(pcm16_frames)} < {min_frame_size} (need {min_frame_size - len(pcm16_frames)} more bytes)")
                return
            
            if len(pcm16_frames) > max_frame_size:
                logger.warning(f"BUFFER DEBUG - Exceeded max size: {len(pcm16_frames)} > {max_frame_size} bytes")
                logger.info("BUFFER DEBUG - Force processing due to size limit")
                asyncio.create_task(self._run_agent(pcm16_frames))
                self.buffer.clear()
                logger.debug("BUFFER DEBUG - Buffer cleared after force processing")
                await self.flow.reset()
                logger.debug("FLOW DEBUG - Flow reset after buffer overflow")
                return
            
            # VAD debugging
            logger.debug("=== VAD DEBUG ===")
            vad_frames = pcm16_frames[-min_frame_size:] if len(pcm16_frames) > min_frame_size else pcm16_frames
            logger.debug(f"VAD DEBUG - Processing {len(vad_frames)} bytes for voice detection")
            logger.debug(f"VAD DEBUG - VAD frame sample count: {len(vad_frames) // 2}")
            
            voice_state = await self.vad.detect(vad_frames)
            logger.debug(f"VAD DEBUG - Detection result: {voice_state}")
            
            # Flow management debugging
            logger.debug("=== FLOW DEBUG ===")
            agent_running = self.agent_task is not None and not self.agent_task.done()
            logger.debug(f"FLOW DEBUG - Agent currently running: {agent_running}")
            logger.debug(f"FLOW DEBUG - Allow interruptions: {allow_interruptions}")
            logger.debug(f"FLOW DEBUG - Can start agent: {not agent_running or allow_interruptions}")
            
            if (self.agent_task is None or self.agent_task.done()) or allow_interruptions:
                start_agent = await self.flow.run_agent(voice_state)
                logger.debug(f"FLOW DEBUG - Flow decision: start_agent={start_agent}")
            else:
                start_agent = False
                logger.debug("FLOW DEBUG - Agent blocked (running and interruptions disabled)")

            if start_agent:
                if self.agent_task and not self.agent_task.done():
                    logger.info("AGENT DEBUG - Cancelling existing agent task for new processing")
                    self.agent_task.cancel()
                    self.agent_task = None
                
                logger.info("AGENT DEBUG - Starting agent processing")
                logger.debug(f"AGENT DEBUG - Processing {len(pcm16_frames)} bytes of audio")
                self.agent_task = asyncio.create_task(self._run_agent(pcm16_frames))
                self.buffer.clear()
                logger.debug("BUFFER DEBUG - Buffer cleared after starting agent")
                
        except Exception as e:
            logger.error(f"PACKET HANDLING DEBUG - Error: {e}", exc_info=True)
            logger.debug(f"PACKET HANDLING DEBUG - Failed packet size: {len(data)} bytes")
            logger.debug(f"PACKET HANDLING DEBUG - Failed packet preview: {data[:min(20, len(data))].hex()}")
    

    async def _run_agent(self, pcm16_frames: bytes):
        """
        Run agent: STT -> TTS -> RTP transmission
        
        This function:
        1. Converts received audio to text using STT
        2. Generates TTS response as PCM chunks
        3. Converts PCM to μ-law and sends via RTP
        4. Manages sequence numbers and timestamps correctly
        """
        logger.info("=== AGENT PROCESSING DEBUG ===")
        try:
            logger.debug(f"AGENT DEBUG - Processing {len(pcm16_frames)} bytes of PCM16 audio")
            logger.debug(f"AGENT DEBUG - Sample count: {len(pcm16_frames) // 2}")
            duration_ms = (len(pcm16_frames) // 2) / (self.target_sample_rate / 1000)
            logger.debug(f"AGENT DEBUG - Audio duration: {duration_ms:.1f}ms")
            
            # Log user audio when starting agent
            if self.audio_logger:
                logger.debug("AUDIO LOGGER DEBUG - Logging user audio")
                await self.audio_logger.log_user(pcm16_frames)
                logger.debug("AUDIO LOGGER DEBUG - User audio logged successfully")
            
            # Step 1: Speech-to-Text
            logger.debug("=== STT PROCESSING DEBUG ===")
            logger.debug("STT DEBUG - Converting PCM16 to WAV")
            wav_bytes = await pcm2wav(pcm16_frames)
            
            if not wav_bytes:
                logger.warning("STT DEBUG - Empty WAV bytes, skipping STT")
                self.buffer.clear() # clearing buffer to avoid accumulation of empty frames
                logger.debug("BUFFER DEBUG - Buffer cleared due to empty WAV")
                return
            
            logger.debug(f"STT DEBUG - WAV conversion successful, size: {len(wav_bytes)} bytes")
            logger.info("STT DEBUG - Sending audio to speech-to-text service")
            
            stt_response = await self.agent.stt(
                wav_bytes, 
                stream=False, 
                enable_history=True
            )
            logger.info(f"STT DEBUG - Response received: '{stt_response}'")
            logger.debug(f"STT DEBUG - Response length: {len(stt_response) if stt_response else 0} characters")
            
            if not stt_response or not stt_response.strip():
                logger.warning("STT DEBUG - Empty or whitespace-only response, skipping TTS")
                return
            
            # Step 2: Text-to-Speech streaming
            logger.info("TTS DEBUG - Starting text-to-speech processing")
            await self._speak(stt_response)
                       
        except Exception as e:
            logger.error(f"AGENT DEBUG - Error in _run_agent: {e}", exc_info=True)
        
    
    async def _send_silence_packet(self):
        """Send a silence RTP packet to maintain connection using continuous RTP stream"""
        if not self.peer_ip or not self.peer_port:
            return
        
        logger.debug("=== SILENCE PACKET DEBUG ===")
        
        # Create silence audio data (zeros) - 20ms at target sample rate
        samples_per_packet = int(self.target_sample_rate * self.silence_interval)
        
        # FIXED: Generate appropriate silence data based on target codec
        if self.target_codec == "ulaw" or (isinstance(self.target_codec, AudioCodec) and self.target_codec == AudioCodec.PCMU):
            # For ulaw, silence is 0xFF (127 in linear, encoded as 0xFF in ulaw)
            silence_data = bytes([0xFF] * samples_per_packet)
        elif self.target_codec == "alaw" or (isinstance(self.target_codec, AudioCodec) and self.target_codec == AudioCodec.PCMA):
            # For alaw, silence is 0xD5 (0 in linear, encoded as 0xD5 in alaw)
            silence_data = bytes([0xD5] * samples_per_packet)
        else:
            # For PCM16, silence is zeros
            silence_data = struct.pack('<' + 'h' * samples_per_packet, *([0] * samples_per_packet))
        
        logger.debug(f"SILENCE DEBUG - Generated {len(silence_data)} bytes ({samples_per_packet} samples) for {self.silence_interval * 1000}ms")
        
        # FIXED: Use continuous RTP stream instead of creating new transmission context
        await self._send_rtp_packet(silence_data, marker=False)
        logger.debug(f"SILENCE DEBUG - Sent silence packet: {len(silence_data)} bytes to {self.peer_ip}:{self.peer_port}")
        await asyncio.sleep(self.silence_interval)

    async def _send_rtp_packet(self, audio_data: bytes, marker: bool = False, source_format: str = "pcm16"):
        """Send a single RTP packet with continuous timestamp management"""
        if not self.peer_ip or not self.peer_port:
            logger.warning("TRANSMISSION DEBUG - Cannot send: peer address not set")
            return
        
        # Initialize timestamp base on first packet
        if not self.__rtp_stream_active:
            self.__timestamp_base = int(time.time() * self.target_sample_rate) & 0xFFFFFFFF
            self.__total_samples_sent = 0
            self.__rtp_stream_active = True
            logger.debug(f"RTP STREAM DEBUG - Initialized timestamp base: {self.__timestamp_base}")
        
        # Calculate current timestamp (continuous from base + total samples sent)
        current_timestamp = (self.__timestamp_base + self.__total_samples_sent) & 0xFFFFFFFF
        
        # FIXED: Use existing enforce_rtp for codec conversion
        rtp = RTPPacket.enforce_rtp(
            data=audio_data,
            format=source_format,
            target=self.target_codec,
            sequence_number=self.__sequence_number & 0xFFFF,
            timestamp=current_timestamp,
            ssrc=self.__ssrc,
            marker=marker
        )
        
        logger.debug(f"RTP PACKET DEBUG - Created packet:")
        logger.debug(f"  - Sequence: {self.__sequence_number & 0xFFFF}")
        logger.debug(f"  - Timestamp: {current_timestamp}")
        logger.debug(f"  - Marker: {marker}")
        logger.debug(f"  - Source format: {source_format}")
        logger.debug(f"  - Target codec: {self.target_codec}")
        logger.debug(f"  - Payload size: {len(rtp.payload)} bytes")
        
        # Send the packet
        try:
            self.sock.sendto(rtp.as_bytes, (self.peer_ip, self.peer_port))
            logger.debug(f"*** SENT RTP PACKET #{self.__sequence_number} *** to {self.peer_ip}:{self.peer_port}")
            
            # Update RTP state
            self.__sequence_number += 1
            # FIXED: Calculate samples correctly based on source format (input to this function)
            if source_format in ["ulaw", "alaw"]:
                samples_in_packet = len(audio_data)  # 1 byte per sample for ulaw/alaw
            else:
                samples_in_packet = len(audio_data) // 2  # 2 bytes per sample for PCM16
            
            self.__total_samples_sent += samples_in_packet
            logger.debug(f"RTP STREAM DEBUG - Samples in packet: {samples_in_packet}, total sent: {self.__total_samples_sent}")
            
        except Exception as e:
            logger.error(f"TRANSMISSION DEBUG - Error sending packet: {e}")

    async def _speak(self, 
                     message: str):
        logger.info("=== TTS STREAMING DEBUG ===")
        logger.info(f"TTS DEBUG - Speaking message: '{message}'")
        logger.debug(f"TTS DEBUG - Message length: {len(message)} characters")

        chunk_count = 0
        total_bytes_processed = 0
        first_chunk = True
        
        logger.info(f"TTS DEBUG - Starting TTS stream (format: {self.tts_response_format})")
        tts_stream = await self.agent.tts(message, stream=True, response_format=self.tts_response_format)
        
        async for chunk in tts_stream:
            if not chunk:
                logger.debug("TTS DEBUG - Received empty chunk, skipping")
                continue
            
            try:
                logger.debug(f"TTS DEBUG - Processing chunk #{chunk_count + 1}, size: {len(chunk)} bytes")
                original_chunk_size = len(chunk)
                
                # Handle WAV format TTS response
                if self.tts_response_format == "wav":
                    logger.debug("CODEC DEBUG - Converting WAV chunk to PCM16")
                    try:
                        chunk = await wav2pcm(chunk)
                        logger.debug(f"CODEC DEBUG - WAV->PCM16 conversion: {original_chunk_size} -> {len(chunk)} bytes")
                    except Exception as e:
                        logger.error(f"CODEC DEBUG - WAV->PCM16 conversion failed: {e}")
                        continue
                
                # Handle resampling for PCM16 format
                if self.tts_sample_rate != self.target_sample_rate:
                    logger.debug(f"CODEC DEBUG - Resampling required: {self.tts_sample_rate}Hz -> {self.target_sample_rate}Hz")
                    
                    if self.tts_codec == "pcm16" or self.tts_response_format == "wav":
                        # Ensure chunk size is even (2 bytes per sample for PCM16)
                        if len(chunk) % 2 != 0:
                            logger.warning(f"CODEC DEBUG - Odd-sized chunk: {len(chunk)} bytes, padding with zero")
                            chunk = chunk + b'\x00'  # Pad with a zero byte instead of truncating
                        
                        # Only resample if we have actual audio data
                        if len(chunk) > 0:
                            pre_resample_samples = len(chunk) // 2
                            chunk = await resample_pcm16(chunk, self.tts_sample_rate, self.target_sample_rate)
                            post_resample_samples = len(chunk) // 2
                            logger.debug(f"CODEC DEBUG - Resampling: {pre_resample_samples} -> {post_resample_samples} samples")
                        else:
                            logger.warning("CODEC DEBUG - Empty chunk after size check, skipping")
                            continue
                    else:
                        logger.warning(f"CODEC DEBUG - Resampling {self.tts_codec} to {self.target_codec} not supported")
                        continue
                else:
                    logger.debug("CODEC DEBUG - No resampling required")
                
                logger.debug(f"TTS DEBUG - Final chunk size: {len(chunk)} bytes")
                
                # Log AI audio chunk (use resampled chunk if available, otherwise original)
                if self.audio_logger:
                    await self.audio_logger.log_ai(chunk)
                    logger.debug("AUDIO LOGGER DEBUG - AI audio chunk logged")
                
                # FIXED: Let enforce_rtp handle codec conversion instead of manual conversion
                # Send packets in appropriate sizes to avoid UDP fragmentation
                chunk_offset = 0
                while chunk_offset < len(chunk):
                    remaining_data = len(chunk) - chunk_offset
                    current_chunk_size = min(remaining_data, MAX_PAYLOAD_SIZE)
                    current_chunk = chunk[chunk_offset:chunk_offset + current_chunk_size]
                    
                    # Mark first packet of utterance
                    marker = first_chunk and chunk_offset == 0
                    # Pass PCM16 data directly - enforce_rtp will handle codec conversion
                    await self._send_rtp_packet(current_chunk, marker=marker, source_format="pcm16")
                    
                    chunk_offset += current_chunk_size
                    first_chunk = False
                
                total_bytes_processed += len(chunk)
                await asyncio.sleep(0.02) # Increased delay to reduce network congestion
                chunk_count += 1
                
                if chunk_count % 10 == 0:  # Log every 10 chunks
                    logger.debug(f"TTS DEBUG - Processed {chunk_count} chunks, {total_bytes_processed} bytes total")
            
            except Exception as e:
                logger.error(f"TTS DEBUG - Error processing chunk #{chunk_count + 1}: {e}")
                continue
        
        # Save audio logs after speaking is complete (checkpoint)
        if self.audio_logger:
            try:
                logger.debug("AUDIO LOGGER DEBUG - Saving audio logs")
                await self.audio_logger.save()
                logger.info("AUDIO LOGGER DEBUG - Audio logs saved successfully")
            except Exception as e:
                logger.error(f"AUDIO LOGGER DEBUG - Error saving audio logs: {e}")
        
        logger.info(f"TTS DEBUG - Agent completed: sent {chunk_count} RTP packets, {total_bytes_processed} bytes total")

    def close(self):
        """Close the RTP server and cleanup resources"""
        logger.info("=== CLOSING RTP SERVER DEBUG ===")
        try:
            if self.sock:
                logger.debug("CLEANUP DEBUG - Closing socket")
                self.sock.close()
                logger.info("CLEANUP DEBUG - Socket closed successfully")
            else:
                logger.debug("CLEANUP DEBUG - No socket to close")
        except Exception as e:
            logger.error(f"CLEANUP DEBUG - Error closing socket: {e}")
        
        logger.info("=== RTP SERVER CLOSED ===")