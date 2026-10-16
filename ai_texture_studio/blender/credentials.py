"""API key resolution with explicit storage modes.

Modes (chosen per provider in the add-on preferences):

* ENV      – read from an environment variable (recommended; nothing stored
             by Blender).
* SESSION  – kept in memory for this Blender session only (lost on quit).
* KEYRING  – OS credential store via the optional ``keyring`` package
             (only offered when that package is importable).
* PREFS    – stored in Blender's user preferences. Blender saves this in
             userpref.blend as PLAIN TEXT; least secure option.

Keys are registered with the redactor so they never appear in messages.
"""

from __future__ import annotations

import os
import time

from ..core.secrets import forget_secret, register_secret

KEYRING_SERVICE = "blender-ai-texture-studio"
_session: dict[str, str] = {}
_keyring_cache: dict[str, tuple] = {}


def keyring_available() -> bool:
    try:
        import keyring  # noqa: F401
        return True
    except Exception:
        return False


def set_session_key(provider_id: str, key: str):
    old = _session.get(provider_id)
    if old:
        forget_secret(old)
    if key:
        _session[provider_id] = key.strip()
        register_secret(key.strip())
    else:
        _session.pop(provider_id, None)


def set_keyring_key(provider_id: str, key: str):
    import keyring
    keyring.set_password(KEYRING_SERVICE, provider_id, key.strip())
    _keyring_cache.pop(provider_id, None)
    register_secret(key.strip())


def clear_keyring_key(provider_id: str):
    _keyring_cache.pop(provider_id, None)
    try:
        import keyring
        keyring.delete_password(KEYRING_SERVICE, provider_id)
    except Exception:
        pass


def get_key(provider_id: str, settings) -> str:
    """Resolve the key for ``provider_id`` from its ProviderSettings."""
    mode = settings.key_mode
    key = ""
    if mode == "ENV":
        key = os.environ.get(settings.env_var or "", "")
    elif mode == "SESSION":
        key = _session.get(provider_id, "")
    elif mode == "KEYRING":
        hit = _keyring_cache.get(provider_id)
        if hit and time.monotonic() - hit[0] < 30:
            key = hit[1]
        else:
            try:
                import keyring
                key = keyring.get_password(KEYRING_SERVICE, provider_id) or ""
            except Exception:
                key = ""
            _keyring_cache[provider_id] = (time.monotonic(), key)
    elif mode == "PREFS":
        key = settings.api_key
    key = (key or "").strip()
    register_secret(key)
    return key


def key_status(provider_id: str, settings) -> str:
    """Human readable status that never reveals the key."""
    mode = settings.key_mode
    has = bool(get_key(provider_id, settings))
    where = {"ENV": f"environment variable {settings.env_var or '(unset)'}", "SESSION": "this session only",
             "KEYRING": "system keyring", "PREFS": "Blender preferences (plain text)"}.get(mode, mode)
    return f"Key {'found' if has else 'NOT set'} – {where}"
