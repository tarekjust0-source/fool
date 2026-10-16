"""Secret redaction helpers.

Any text that might be shown to the user, printed to the console or stored
in history passes through :func:`redact` first.  Known secret values are
registered at runtime; common key formats are redacted by pattern too.
"""

from __future__ import annotations

import re
import threading

_lock = threading.Lock()
_known: set[str] = set()

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),            # OpenAI style
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),           # Google API keys
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)(x-goog-api-key[\"']?\s*[:=]\s*[\"']?)[^\s\"',]+"),
    re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\s\"',&]+"),
    re.compile(r"(?i)([?&]key=)[^&\s]+"),
]


def register_secret(value: str | None):
    if value and len(value) >= 6:
        with _lock:
            _known.add(value)


def forget_secret(value: str | None):
    with _lock:
        _known.discard(value or "")


def redact(text) -> str:
    if text is None:
        return ""
    s = str(text)
    with _lock:
        known = sorted(_known, key=len, reverse=True)
    for k in known:
        if k in s:
            s = s.replace(k, "[REDACTED]")
    for p in _PATTERNS:
        if p.groups:
            s = p.sub(lambda m: m.group(1) + "[REDACTED]", s)
        else:
            s = p.sub("[REDACTED]", s)
    return s
