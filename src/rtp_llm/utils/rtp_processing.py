import struct
from dataclasses import dataclass
from typing import Optional
from enum import IntEnum
import audioop

# Optional Opus support
try:
    import opuslib
    OPUS_AVAILABLE = True
except Exception as e:
    print(f"Opus library not found: {e}")
    OPUS_AVAILABLE = False
    opuslib = None


class AudioCodec(IntEnum):
    """Common audio codec payload types (RFC 3551)"""
    PCMU = 0      # μ-law (G.711)
    GSM = 3       # GSM
    G723 = 4      # G.723
    DVI4_8000 = 5 # DVI4 8kHz
    DVI4_16000 = 6 # DVI4 16kHz
    LPC = 7       # LPC
    PCMA = 8      # A-law (G.711)
    G722 = 9      # G.722
    L16_2CH = 10  # Linear PCM 16-bit stereo
    L16_1CH = 11  # Linear PCM 16-bit mono
    QCELP = 12    # QCELP
    CN = 13       # Comfort Noise
    MPA = 14      # MPEG Audio
    G728 = 15     # G.728
    DVI4_11025 = 16 # DVI4 11.025kHz
    DVI4_22050 = 17 # DVI4 22.05kHz
    G729 = 18     # G.729
    OPUS = 111    # Opus (RFC 7587) - dynamic payload type


@dataclass(frozen=True)
class RTPHeader:
    version: int
    padding: bool
    extension: bool
    csrc_count: int
    marker: bool
    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'RTPHeader':
        """
        Parse RTP header from bytes according to RFC 3550
        
        RTP Header format:
         0                   1                   2                   3
         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |V=2|P|X|  CC   |M|     PT      |       sequence number         |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                           timestamp                           |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |           synchronization source (SSRC) identifier          |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        """
        if len(data) < 12:
            raise ValueError("RTP header must be at least 12 bytes")
        header = data[:12] # slice header
        # Unpack first 12 bytes of RTP header
        byte0, byte1, seq_num, timestamp, ssrc = struct.unpack('!BBHII', header)
        
        # Parse first byte: V(2), P(1), X(1), CC(4)
        version = (byte0 >> 6) & 0x3
        padding = bool((byte0 >> 5) & 0x1)
        extension = bool((byte0 >> 4) & 0x1)
        csrc_count = byte0 & 0xF
        
        # Parse second byte: M(1), PT(7)
        marker = bool((byte1 >> 7) & 0x1)
        payload_type = byte1 & 0x7F
        
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
        """
        Convert RTP header to bytes
        """
        # Build first byte: V(2), P(1), X(1), CC(4)
        byte0 = (self.version << 6) | (int(self.padding) << 5) | (int(self.extension) << 4) | self.csrc_count
        
        # Build second byte: M(1), PT(7)
        byte1 = (int(self.marker) << 7) | self.payload_type
        
        # Pack header into bytes
        return struct.pack('!BBHII', byte0, byte1, self.sequence_number, self.timestamp, self.ssrc)


@dataclass(frozen=True)
class RTPPacket:
    header: RTPHeader
    payload: bytes


    @staticmethod
    def detect_codec(codec: str | AudioCodec) -> AudioCodec:
        """
        Detect audio codec from string
        """
        if isinstance(codec, AudioCodec):
            return codec
        assert codec in ["pcm16", "ulaw", "alaw", "opus"], "Unsupported codec"
        if codec == "pcm16":
            return AudioCodec.L16_1CH
        elif codec == "ulaw":
            return AudioCodec.PCMU
        elif codec == "alaw":
            return AudioCodec.PCMA
        elif codec == "opus":
            return AudioCodec.OPUS
        else:
            raise ValueError(f"Unsupported codec: {codec}")
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'RTPPacket':
        """
        Parse RTP packet from bytes
        """
        header = RTPHeader.from_bytes(data[:12])
        payload = data[12:]
        return cls(header=header, payload=payload)

    @classmethod
    def simple_rtp(cls, data: bytes, payload_type: AudioCodec, sequence_number: int, timestamp: int, ssrc: int, marker: bool) -> 'RTPPacket':
        """
        simplified rtp creation
        """
        #TODO: check pt, sn, ts, ssrc are valid
        header = RTPHeader(
            version=2,  # RTP version 2
            padding=False,
            extension=False,
            csrc_count=0,
            marker=marker,
            payload_type=payload_type.value,
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc
        )
        return cls(header=header, payload=data)


    @classmethod
    def ulaw_rtp(cls, ulaw_data: bytes,
                  sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0,
                  marker: bool = False
                  ) -> 'RTPPacket':     
        """
        Create RTP packet from μ-law audio data
        """
        #TODO: check if ulaw_data is valid
        return cls.simple_rtp(ulaw_data, AudioCodec.PCMU, sequence_number, timestamp, ssrc, marker)
    
    @classmethod
    def alaw_rtp(cls, alaw_data: bytes, sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, marker: bool = False) -> 'RTPPacket':
        """
        Create RTP packet from A-law audio data
        """
        return cls.simple_rtp(alaw_data, AudioCodec.PCMA, sequence_number, timestamp, ssrc, marker)
    
    @classmethod
    def enforce_ulaw_from_pcm16(cls, pcm16_data: bytes, sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, marker: bool = False) -> 'RTPPacket':
        """
        Convert PCM16 to μ-law and create RTP packet
        """
        ulaw_data = audioop.lin2ulaw(pcm16_data, 2)
        return cls.ulaw_rtp(ulaw_data, sequence_number, timestamp, ssrc, marker)
    
    @classmethod
    def enforce_alaw_from_pcm16(cls, pcm16_data: bytes, sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, marker: bool = False) -> 'RTPPacket':
        """
        Convert PCM16 to A-law and create RTP packet
        """
        alaw_data = audioop.lin2alaw(pcm16_data, 2)
        return cls.alaw_rtp(alaw_data, sequence_number, timestamp, ssrc, marker)
    
    @classmethod
    def opus_rtp(cls, opus_data: bytes, sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, marker: bool = False) -> 'RTPPacket':
        """
        Create RTP packet from Opus audio data
        """
        return cls.simple_rtp(opus_data, AudioCodec.OPUS, sequence_number, timestamp, ssrc, marker)
    
    @classmethod
    def enforce_opus_from_pcm16(cls, pcm16_data: bytes, sample_rate: int = 48000, channels: int = 1, 
                               sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, marker: bool = False) -> 'RTPPacket':
        """
        Convert PCM16 to Opus and create RTP packet
        """
        if not OPUS_AVAILABLE:
            raise RuntimeError("Opus codec is not available. Install the Opus library to use Opus encoding.")
        
        # Create Opus encoder
        encoder = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_VOIP)
        
        # Encode PCM16 to Opus
        opus_data = encoder.encode(pcm16_data, frame_size=sample_rate // 50)  # 20ms frame
        
        return cls.opus_rtp(opus_data, sequence_number, timestamp, ssrc, marker)
    
    @classmethod
    def enforce_rtp(cls, data: bytes, 
                    format: str | AudioCodec, target: str | AudioCodec, 
                    sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, marker: bool = False,
                    sample_rate: int = 48000, channels: int = 1) -> 'RTPPacket':
        format_codec = cls.detect_codec(format)
        target_codec = cls.detect_codec(target)
        
        if format_codec == target_codec:
            return cls.simple_rtp(data, format_codec, sequence_number, timestamp, ssrc, marker)
        
        else:
            if format_codec in [AudioCodec.L16_1CH, AudioCodec.L16_2CH] and target_codec == AudioCodec.PCMU:
                return cls.enforce_ulaw_from_pcm16(data, sequence_number, timestamp, ssrc, marker)
            elif format_codec in [AudioCodec.L16_1CH, AudioCodec.L16_2CH] and target_codec == AudioCodec.PCMA:
                return cls.enforce_alaw_from_pcm16(data, sequence_number, timestamp, ssrc, marker)
            elif format_codec in [AudioCodec.L16_1CH, AudioCodec.L16_2CH] and target_codec == AudioCodec.OPUS:
                return cls.enforce_opus_from_pcm16(data, sample_rate, channels, sequence_number, timestamp, ssrc, marker)
            else:
                raise ValueError(f"Unsupported format or target: {format}, {target}")

    @property
    def as_bytes(self) -> bytes:
        """
        Convert RTP packet to bytes
        """
        return self.header.as_bytes + self.payload

    @property
    def is_audio(self) -> bool:
        """Check if this packet contains audio data"""
        return self.header.payload_type in [codec.value for codec in AudioCodec]
    
    @property
    def audio_codec(self) -> Optional[AudioCodec]:
        """Get the audio codec if this is an audio packet"""
        if not self.is_audio:
            return None
        try:
            return AudioCodec(self.header.payload_type)
        except ValueError:
            return None
    

    @property
    def is_ulaw(self) -> bool:
        """
        Convert RTP packet to μ-law audio data
        """
        if not self.is_audio or self.audio_codec != AudioCodec.PCMU:
            return False
        return True
    
    @property
    def is_alaw(self) -> bool:
        """
        Convert RTP packet to A-law audio data
        """
        if not self.is_audio or self.audio_codec != AudioCodec.PCMA:
            return False
        return True
        
    @property
    def is_linear_pcm(self) -> bool:
        """
        Convert RTP packet to linear PCM audio data
        """
        if not self.is_audio or self.audio_codec not in [AudioCodec.L16_1CH, AudioCodec.L16_2CH]:
            return False
        return True
    
    @property
    def is_comfort_noise(self) -> bool:
        """
        Check if this packet contains comfort noise (CN)
        """
        return self.header.payload_type == AudioCodec.CN.value
    
    @property
    def is_opus(self) -> bool:
        """
        Check if this packet contains Opus audio data
        """
        if not self.is_audio or self.audio_codec != AudioCodec.OPUS:
            return False
        return True
    
    def enforce_pcm16(self, sample_rate: int = 48000, channels: int = 1) -> bytes:
        """
        Convert RTP packet to 16-bit linear PCM audio data
        """
        if self.is_ulaw:
            return audioop.ulaw2lin(self.payload, 2) # 2 = 16-bit samples
        elif self.is_alaw:
            return audioop.alaw2lin(self.payload, 2) # 2 = 16-bit samples
        elif self.is_linear_pcm:
            return self.payload
        elif self.is_opus:
            if not OPUS_AVAILABLE:
                raise RuntimeError("Opus codec is not available. Install the Opus library to use Opus decoding.")
            
            # Create Opus decoder
            decoder = opuslib.Decoder(sample_rate, channels)
            
            # Decode Opus to PCM16
            pcm16_data = decoder.decode(self.payload, frame_size=sample_rate // 50)  # 20ms frame
            return pcm16_data
        else:
            raise ValueError("Unsupported audio codec or data format")
