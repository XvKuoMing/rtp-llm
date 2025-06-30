import socket
import struct
import asyncio
import time

from .base import Adapter
from ..utils.audio_processing import (
    resample_pcm16, 
    ulaw2pcm, 
    alaw2pcm, 
    opus2pcm,
    pcm2ulaw,
    pcm2alaw,
    pcm2opus
)
from typing import Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Separate constants for sending and receiving
# RTP_CHUNK_SIZE = 1024  # Size for outgoing audio chunks
RTP_MAX_PACKET_SIZE = 8192  # Maximum size for incoming RTP packets (8KB should handle most cases)
RTP_INTER_PACKET_DELAY = 0.02  # 20ms delay between packets

logger = logging.getLogger(__name__)



class AudioCodec(Enum):
    PCM = 10
    ULAW = 0
    ALAW = 8
    OPUS = 111

    @classmethod
    def from_codec(cls, codec: str | int) -> 'AudioCodec':
        if isinstance(codec, int):
            # Try to find by value
            for c in cls:
                if c.value == codec:
                    return c
            raise ValueError(f"Unsupported audio codec value: {codec}")
        elif isinstance(codec, str):
            codec = codec.lower()
            if codec == "pcm":
                return cls.PCM
            elif codec == "ulaw" or codec == "mulaw":
                return cls.ULAW
            elif codec == "alaw":
                return cls.ALAW
            elif codec == "opus":
                return cls.OPUS
            else:
                raise ValueError(f"Unsupported audio codec: {codec}")


@dataclass
class RTPHeader:
    version: int
    padding: bool
    extension: bool
    csrc_count: int
    marker: bool
    payload_type: AudioCodec
    sequence_number: int
    timestamp: int
    ssrc: int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'RTPHeader':
        if len(data) < 12:
            raise ValueError("RTP header must be at least 12 bytes")
        
        # Unpack the first 12 bytes of RTP header
        # Format: >BBHII (big-endian: byte, byte, short, int, int)
        byte1, byte2, seq_num, timestamp, ssrc = struct.unpack('>BBHII', data[:12])
        
        # Parse first byte: V(2) P(1) X(1) CC(4)
        version = (byte1 >> 6) & 0x3
        padding = bool((byte1 >> 5) & 0x1)
        extension = bool((byte1 >> 4) & 0x1)
        csrc_count = byte1 & 0xF
        
        # Parse second byte: M(1) PT(7)
        marker = bool((byte2 >> 7) & 0x1)
        payload_type_val = byte2 & 0x7F
        
        try:
            payload_type = AudioCodec.from_codec(payload_type_val)
        except ValueError:
            # Default to PCM if unknown
            logger.warning(f"Unknown payload type {payload_type_val}, defaulting to PCM")
            payload_type = AudioCodec.PCM
        
        return cls(
            version=version,
            padding=padding,
            extension=extension,
            csrc_count=csrc_count,
            marker=marker,
            payload_type=payload_type,
            sequence_number=seq_num,
            timestamp=timestamp,
            ssrc=ssrc
        )

    @property
    def as_bytes(self) -> bytes:
        # Construct first byte: V(2) P(1) X(1) CC(4)
        byte1 = (self.version << 6) | (int(self.padding) << 5) | (int(self.extension) << 4) | self.csrc_count
        
        # Construct second byte: M(1) PT(7)
        byte2 = (int(self.marker) << 7) | self.payload_type.value
        
        # Pack the header
        return struct.pack('>BBHII', 
                          byte1, 
                          byte2, 
                          self.sequence_number, 
                          self.timestamp, 
                          self.ssrc)


@dataclass
class RTPPacket:
    header: RTPHeader
    payload: bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> 'RTPPacket':
        if len(data) < 12:
            raise ValueError("RTP packet must be at least 12 bytes")
            
        header_bytes = data[:12]
        payload_bytes = data[12:]
        header = RTPHeader.from_bytes(header_bytes)
        payload = payload_bytes
        return cls(header, payload)
    
    @property
    def as_bytes(self) -> bytes:
        return self.header.as_bytes + self.payload
    
    async def convert_to_pcm16(self) -> bytes:
        if self.header.payload_type == AudioCodec.PCM:
            return self.payload
        elif self.header.payload_type == AudioCodec.ULAW:
            return await ulaw2pcm(self.payload)
        elif self.header.payload_type == AudioCodec.ALAW:
            return await alaw2pcm(self.payload)
        elif self.header.payload_type == AudioCodec.OPUS:
            return await opus2pcm(self.payload)
        else:
            raise ValueError(f"Unsupported audio codec: {self.header.payload_type}")

class RTPAdapter(Adapter):

    def __init__(self, 
                 host_ip: str, 
                 host_port: int, 
                 peer_ip: Optional[str] = None, 
                 peer_port: Optional[int] = None,
                 sample_rate: int = 8000,
                 target_codec: str | int = "pcm"
                 ):
        self.host_ip = host_ip
        self.host_port = host_port
        self.peer_ip = peer_ip
        self.peer_port = peer_port
        self.sample_rate = sample_rate
        self.frame_size = self.sample_rate * RTP_INTER_PACKET_DELAY # must be multiple of 160 for 8kHz

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Increase socket buffer sizes to handle larger RTP packets
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB receive buffer
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # 256KB send buffer
        self.socket.bind((self.host_ip, self.host_port))
        self.socket.setblocking(False)

        # RTP state
        self.__ssrc = int(time.time() * 1000) & 0xFFFFFFFF  # Use timestamp as SSRC
        self.__sequence_number = 0
        self.__timestamp = 0
        
        # For calculating proper timestamps
        self.samples_per_second = sample_rate
        self.target_codec = AudioCodec.from_codec(target_codec)
        self.bytes_per_sample = 2 if self.target_codec == AudioCodec.PCM else 1  # PCM16 = 2 bytes per sample

    async def send_audio(self, audio: bytes, audio_sample_rate: int = 24_000) -> None:
        if self.peer_ip is None or self.peer_port is None:
            logger.warning("Peer IP and port are not set, skipping send")
            return

        if audio_sample_rate != self.sample_rate:
            logger.info(f"Resampling audio from {audio_sample_rate}Hz to {self.sample_rate}Hz")
            audio = await resample_pcm16(
                audio, 
                original_sample_rate=audio_sample_rate, 
                target_sample_rate=self.sample_rate)
        
        # Convert to target codec if necessary
        if self.target_codec != AudioCodec.PCM:
            if self.target_codec == AudioCodec.ULAW:
                logger.info("Converting to ulaw")
                audio = await pcm2ulaw(audio)
            elif self.target_codec == AudioCodec.ALAW:
                logger.info("Converting to alaw")
                audio = await pcm2alaw(audio)
            elif self.target_codec == AudioCodec.OPUS:
                logger.info("Converting to opus")
                audio = await pcm2opus(audio)
            else:
                logger.warning("Keeping audio as pcm, since codec has not been detected")
        
        first_chunk = True
        total_samples_sent = 0
        chunk_count = 0
        
        while len(audio) > 0:
            # chunk = audio[:RTP_CHUNK_SIZE]
            chunk = audio[:self.frame_size]
            
            # Calculate actual samples in this chunk
            samples_in_chunk = len(chunk) // self.bytes_per_sample
            
            rtp_packet = RTPPacket(
                header=RTPHeader(
                    version=2,
                    padding=False,
                    extension=False,
                    csrc_count=0,
                    marker=first_chunk,
                    payload_type=self.target_codec,
                    sequence_number=self.__sequence_number,
                    timestamp=self.__timestamp,
                    ssrc=self.__ssrc
                ),
                payload=chunk
            )
            first_chunk = False
            
            try:
                self.socket.sendto(rtp_packet.as_bytes, (self.peer_ip, self.peer_port))
                
                # Use a more precise timing based on actual samples
                chunk_duration = samples_in_chunk / self.sample_rate
                await asyncio.sleep(min(chunk_duration, RTP_INTER_PACKET_DELAY))
                
                logger.debug(f"Sent RTP packet: seq={rtp_packet.header.sequence_number}, ts={rtp_packet.header.timestamp}, len={len(chunk)} bytes, samples={samples_in_chunk}, to {self.peer_ip}:{self.peer_port}")
            except Exception as e:
                logger.error(f"Error sending RTP packet: {e}")
                break

            # Update sequence number and timestamp correctly
            self.__sequence_number = (self.__sequence_number + 1) % 65536
            self.__timestamp = (self.__timestamp + self.frame_size) >> 0
            # Timestamp should increment by number of samples, not bytes
            # samples_sent += len(chunk) // self.bytes_per_sample
            # self.__timestamp = (self.__timestamp + samples_sent) & 0xFFFFFFFF
            audio = audio[self.frame_size:]
            chunk_count += 1
            total_samples_sent += samples_in_chunk
            
        logger.info(f"Sent {chunk_count} chunks, {total_samples_sent} samples, duration: {total_samples_sent/self.sample_rate:.2f}s")

    async def receive_audio(self) -> Optional[bytes]:
        try:
            data, address = self.socket.recvfrom(RTP_MAX_PACKET_SIZE)
            host_ip, host_port = address

            # Auto-configure peer if not set
            if self.peer_ip is None:
                logger.info(f"Received packet from {address}, setting it as peer ip")
                self.peer_ip = host_ip
            if self.peer_port is None:
                logger.info(f"Received packet from {address}, setting it as peer port")
                self.peer_port = host_port

            # Validate source
            if host_ip != self.peer_ip or host_port != self.peer_port:
                logger.warning(f"Received packet from unexpected source: {address}")
                return None
            
            rtp_packet = RTPPacket.from_bytes(data)
            logger.debug(f"Received RTP packet: seq={rtp_packet.header.sequence_number}, ts={rtp_packet.header.timestamp}, codec={rtp_packet.header.payload_type}")
            
            # Convert to PCM16
            pcm16_frame = await rtp_packet.convert_to_pcm16()
            return pcm16_frame

        except socket.timeout:
            return None
        except socket.error as e:
            if e.errno == 10035:  # WSAEWOULDBLOCK on Windows
                return None
            elif e.errno == 11:  # EAGAIN/EWOULDBLOCK on Unix/Linux
                return None
            elif e.errno == 10040:  # WSAEMSGSIZE on Windows - message too large
                logger.warning(f"Received RTP packet larger than buffer size ({RTP_MAX_PACKET_SIZE} bytes). Consider increasing RTP_MAX_PACKET_SIZE.")
                return None
            logger.error(f"Socket error in receive_audio: {e}")
            return None
        except Exception as e:
            logger.error(f"Error receiving RTP packet: {e}")
            return None
    
    def close(self):
        """Close the socket connection"""
        if self.socket:
            self.socket.close()
            logger.info("RTP socket closed")



