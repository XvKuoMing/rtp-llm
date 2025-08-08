from typing import Optional


class RtllmError(Exception):
    """Base class for all rtllm entrypoint errors."""

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


class ValidationError(RtllmError):
    """Raised when input parameters are invalid (400)."""


class NotFoundError(RtllmError):
    """Raised when a requested resource cannot be found (404)."""


class ResourceConflictError(RtllmError):
    """Raised when an operation conflicts with current state (409)."""


class ConfigurationError(RtllmError):
    """Raised when server/provider configuration is invalid (422/500)."""


class DependencyUnavailableError(RtllmError):
    """Raised when optional dependencies (e.g., Redis) are unavailable (503)."""


