import json
from typing import Any, Dict, Iterable


SENSITIVE_KEYS: set[str] = {
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "client_secret",
    "authorization",
    "auth",
}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in SENSITIVE_KEYS:
        return True
    # catch common substrings like *_key, *_token, *_secret, *_password
    for needle in ("key", "token", "secret", "password", "auth"):
        if lowered.endswith(needle) or needle in lowered:
            return True
    return False


def _mask(value: Any) -> str:
    try:
        s = str(value)
    except Exception:
        return "***"
    if not s:
        return "***"
    # Keep only prefix to hint presence without leaking value
    return (s[:2] + "***") if len(s) > 2 else "***"


def sanitize_for_logging(obj: Any) -> Any:
    """Recursively sanitize a structure for safe logging by masking sensitive values.

    - Dict keys that look sensitive are masked
    - Lists/tuples are sanitized element-wise
    - Primitive values are returned as-is
    """
    if isinstance(obj, dict):
        sanitized: Dict[str, Any] = {}
        for k, v in obj.items():
            if _is_sensitive_key(str(k)):
                sanitized[k] = _mask(v)
            else:
                sanitized[k] = sanitize_for_logging(v)
        return sanitized
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_logging(v) for v in obj]
    return obj


def to_json_for_logging(data: Any, *, indent: int | None = None) -> str:
    try:
        if indent is not None:
            return json.dumps(data, ensure_ascii=False, indent=indent)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Fallback to repr
        return repr(data)


