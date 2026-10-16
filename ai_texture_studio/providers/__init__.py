"""Provider registry."""

from __future__ import annotations

import os

from .base import Provider, ProviderConfig, ProviderError, ImageRequest, ImageResult  # noqa: F401
from .comfyui_provider import ComfyUIProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .stability_provider import StabilityProvider

__all__ = ["Provider", "ProviderConfig", "ProviderError", "ImageRequest", "ImageResult", "get", "all_providers",
           "register", "test_provider_enabled"]

_REGISTRY: dict[str, Provider] = {}


def _register_defaults():
    for cls in (OpenAIProvider, StabilityProvider, GeminiProvider, ComfyUIProvider):
        p = cls()
        _REGISTRY[p.id] = p
    if test_provider_enabled():
        from .mock_provider import MockProvider
        _REGISTRY["MOCK"] = MockProvider()


def test_provider_enabled() -> bool:
    return os.environ.get("ATS_ENABLE_TEST_PROVIDER") == "1"


def get(provider_id: str) -> Provider:
    if not _REGISTRY:
        _register_defaults()
    try:
        return _REGISTRY[provider_id]
    except KeyError:
        raise ProviderError("config", f"Unknown or unavailable provider '{provider_id}'.", retryable=False) from None


def all_providers() -> list[Provider]:
    if not _REGISTRY:
        _register_defaults()
    return list(_REGISTRY.values())


def register(provider: Provider):
    """Extension point: third-party code can add providers at runtime."""
    _REGISTRY[provider.id] = provider
