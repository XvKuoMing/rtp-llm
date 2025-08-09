import asyncio
import logging
import websockets
from typing import Optional
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.server import WebSocketServerProtocol

from .base import Adapter

logger = logging.getLogger(__name__)

class WebSocketAdapter(Adapter):
    """
    WebSocket adapter for real-time audio streaming.
    Designed for mobile clients (Android/iOS) that need reliable audio transmission.
    
    Protocol:
    - Binary messages only: Raw audio data (PCM16)
    - No control messages - pure audio streaming
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8765,
                 sample_rate: int = 8000,
                 target_codec: str = "pcm",
                 **kwargs):
        super().__init__(sample_rate=sample_rate, target_codec=target_codec)
        self.host = host
        self.port = port
        self.websocket_server = None
        self.connected_client: Optional[WebSocketServerProtocol] = None
        self.audio_queue = asyncio.Queue()
        self.is_running = False
        self.chunk_size = int(sample_rate * 0.02 * 2)  # 20ms chunks in bytes (PCM16 = 2 bytes per sample)
        # Autostart the websocket server in the background
        asyncio.get_running_loop()
        asyncio.create_task(self.__start_server())
        logger.info(f"WebSocketAdapter initialized: {host}:{port}, sample_rate={sample_rate}, chunk_size={self.chunk_size}")

    # Compatibility with RTPAdapter interface used by ServerManager
    @property
    def host_ip(self) -> str:
        return self.host

    @property
    def host_port(self) -> int:
        return self.port

    @property
    def peer_is_configured(self) -> bool:
        """
        Check if a WebSocket client is connected and ready
        """
        return (self.connected_client is not None and 
                not self.connected_client.closed and 
                self.is_running)

    async def __start_server(self):
        """Start the WebSocket server and wait for client connections"""
        if self.websocket_server is not None:
            logger.warning("WebSocket server is already running")
            return
            
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        # websockets>=12 passes a single argument (connection/websocket) to the handler.
        # Older versions passed (websocket, path). We accept a single argument for
        # compatibility with newer versions; the path is not used by this adapter.
        async def client_handler(websocket):
            """Handle individual client connections"""
            client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f"New WebSocket client connected from {client_address}")
            
            # If there's already a connected client, disconnect the old one
            if self.connected_client is not None:
                logger.info("Disconnecting previous client")
                await self.connected_client.close()
            
            self.connected_client = websocket
            
            try:
                # Handle incoming binary messages (audio data only)
                async for message in websocket:
                    if isinstance(message, bytes):
                        # Audio data received
                        await self.audio_queue.put(message)
                        logger.debug(f"Received {len(message)} bytes of audio data")
                    else:
                        # Ignore non-binary messages
                        logger.warning(f"Received non-binary message, ignoring: {type(message)}")
                            
            except ConnectionClosed:
                logger.info(f"Client {client_address} disconnected")
            except Exception as e:
                logger.error(f"Error handling client {client_address}: {e}")
            finally:
                if self.connected_client == websocket:
                    self.connected_client = None
                logger.info(f"Client {client_address} connection cleaned up")
        
        try:
            self.websocket_server = await websockets.serve(
                client_handler,
                self.host,
                self.port,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                close_timeout=10   # Wait 10 seconds for close
            )
            self.is_running = True
            logger.info(f"WebSocket server started successfully on ws://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def send_audio(self, audio_pcm16: bytes) -> None:
        """
        Send audio data to the connected WebSocket client.
        Audio is expected to be in PCM16 format at the configured sample rate.
        """
        if not self.peer_is_configured:
            logger.debug("No client connected, skipping audio send")
            return
            
        if not audio_pcm16:
            logger.debug("No audio data to send")
            return
            
        try:
            # Send raw audio bytes directly (binary message)
            await self.connected_client.send(audio_pcm16)
                
            logger.debug(f"Sent {len(audio_pcm16)} bytes of audio to WebSocket client")
            
        except ConnectionClosed:
            logger.info("Client disconnected during audio send")
            self.connected_client = None
        except Exception as e:
            logger.error(f"Error sending audio to WebSocket client: {e}")
            # Try to close the connection gracefully
            if self.connected_client:
                try:
                    await self.connected_client.close()
                except:
                    pass
                self.connected_client = None

    async def receive_audio(self) -> Optional[bytes]:
        """
        Receive audio data from the connected WebSocket client.
        Returns audio in PCM16 format at the configured sample rate.
        """
        if not self.is_running:
            return None
            
        try:
            # Wait for audio data with a reasonable timeout
            audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            
            # Return raw audio data as-is (assuming client sends PCM16)
            logger.debug(f"Received {len(audio_data)} bytes of audio from WebSocket client")
            return audio_data
                
        except asyncio.TimeoutError:
            # No audio data available, which is normal
            return None
        except Exception as e:
            logger.error(f"Error receiving audio from WebSocket client: {e}")
            return None

    def close(self):
        """Close the WebSocket server and all connections"""
        logger.info("Closing WebSocket adapter")
        self.is_running = False
        
        async def async_close():
            try:
                # Close client connection
                if self.connected_client:
                    await self.connected_client.close()
                    self.connected_client = None
                
                # Close server
                if self.websocket_server:
                    self.websocket_server.close()
                    await self.websocket_server.wait_closed()
                    self.websocket_server = None
                    
                logger.info("WebSocket server closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebSocket server: {e}")
        
        # If we're in an async context, run the close coroutine
        try:
            asyncio.create_task(async_close())
        except RuntimeError:
            # If no event loop is running, we can't close gracefully
            logger.warning("No event loop running, WebSocket server may not close gracefully")