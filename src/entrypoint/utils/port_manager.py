import hashlib
import socket
import threading
from typing import Optional, Dict, Set, Union


class PortManager:
    """
    Manages allocation of TCP ports within a configured range, maintaining a stable
    mapping between a server UID and an allocated port. Thread-safe.
    """

    def __init__(self, start_port: int, end_port: int):
        if start_port <= 0 or end_port <= 0 or end_port < start_port:
            raise ValueError("Invalid port range")

        self.start_port = start_port
        self.end_port = end_port
        self.uid_to_port: Dict[Union[str, int], int] = {}
        self.allocated_ports: Set[int] = set()
        self.port_lock = threading.Lock()

    def get_static_host_ip(self) -> str:
        """
        Determine the primary local IP address using the UDP "connect trick".
        Falls back to hostname resolution and finally 127.0.0.1.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))  # No packets sent
                ip = s.getsockname()[0]
            finally:
                s.close()
            if ip:
                return ip
        except Exception:
            pass

        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

    def _stable_port_for_uid(self, uid: Union[str, int]) -> int:
        """Compute a stable preferred port for a UID using SHA-256 hashing."""
        uid_bytes = str(uid).encode("utf-8")
        digest = hashlib.sha256(uid_bytes).digest()
        # Convert first 8 bytes to int for modulo math
        hashed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        span = self.end_port - self.start_port + 1
        return self.start_port + (hashed % span)

    def get_available_port(self, uid: Union[str, int]) -> int:
        """
        Get an available port for a given UID. If the UID already has a port, return it.
        Otherwise, choose a stable preferred port; if not available, pick the first free one.
        """
        with self.port_lock:
            if uid in self.uid_to_port:
                return self.uid_to_port[uid]

            preferred_port = self._stable_port_for_uid(uid)
            if preferred_port not in self.allocated_ports and self.is_port_available(preferred_port):
                self.allocated_ports.add(preferred_port)
                self.uid_to_port[uid] = preferred_port
                return preferred_port

            for port in range(self.start_port, self.end_port + 1):
                if port not in self.allocated_ports and self.is_port_available(port):
                    self.allocated_ports.add(port)
                    self.uid_to_port[uid] = port
                    return port

            raise RuntimeError(f"No available ports in range {self.start_port}-{self.end_port}")

    def get_uid_port(self, uid: Union[str, int]) -> Optional[int]:
        """Return the allocated port for the given UID, if any."""
        return self.uid_to_port.get(uid)

    def is_port_available(self, port: int) -> bool:
        """Check if a TCP port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
            return True
        except OSError:
            return False

    def release_uid(self, uid: Union[str, int]) -> None:
        """Release the port allocated to a specific UID."""
        with self.port_lock:
            port = self.uid_to_port.pop(uid, None)
            if port is not None:
                self.allocated_ports.discard(port)

    def release_port(self, port: int) -> None:
        """Release a port back to the available pool."""
        with self.port_lock:
            if port in self.allocated_ports:
                self.allocated_ports.discard(port)
                # Remove any UID mapping pointing to this port
                for mapped_uid, mapped_port in list(self.uid_to_port.items()):
                    if mapped_port == port:
                        self.uid_to_port.pop(mapped_uid, None)

    def release_all(self) -> None:
        """Release all allocated ports and clear mappings."""
        with self.port_lock:
            self.uid_to_port.clear()
            self.allocated_ports.clear()

