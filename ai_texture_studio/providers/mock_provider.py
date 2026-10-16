"""Deterministic offline provider – FOR AUTOMATED TESTS ONLY.

It is registered only when the environment variable
``ATS_ENABLE_TEST_PROVIDER=1`` is set, and its output is an obviously
synthetic pattern (it is not a substitute for real generation).

``options['fail_mode']`` simulates failures: network, timeout, auth,
rate_limit, invalid_image, blank_image, content_policy, fail_after:N.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np

from ..core import imageio
from . import base as b


class MockProvider(b.Provider):
    id = "MOCK"
    name = "Test Provider (offline, synthetic)"
    description = "Synthetic deterministic output for automated tests. Not an AI model."
    is_local = True
    requires_key = False
    default_model = "mock"
    data_notice = "Nothing is sent anywhere."

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def capabilities(self, cfg):
        return {b.TEXT_TO_IMAGE, b.IMAGE_EDIT, b.MASK, b.REFERENCE, b.NEGATIVE_PROMPT, b.SEED, b.STRENGTH}

    def _color(self, text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return np.array([h[0], h[1], h[2]], np.float32) / 255.0 * 0.7 + 0.15

    def _generate(self, req, cfg, cancel=None):
        with self._lock:
            self.calls.append(req)
            n = len(self.calls)
        mode = cfg.options.get("fail_mode", "")
        if mode.startswith("fail_after:") and n > int(mode.split(":")[1]):
            raise b.ProviderError("server", "Mock provider: simulated failure after N calls")
        if mode == "network":
            raise b.ProviderError("network", "Mock provider: simulated network error")
        if mode == "timeout":
            raise b.ProviderError("timeout", "Mock provider: simulated timeout")
        if mode == "auth":
            raise b.ProviderError("auth", "Mock provider: simulated invalid API key", retryable=False)
        if mode == "rate_limit":
            raise b.ProviderError("rate_limit", "Mock provider: simulated rate limit")
        if mode == "content_policy":
            raise b.ProviderError("content_policy", "Mock provider: simulated refusal", retryable=False)
        if mode == "invalid_image":
            return b.ImageResult(b.decode_result_bytes(b"<html>not an image</html>", self.name), self.id, "mock")
        if mode == "blank_image":
            blank = np.full((64, 64, 3), 7, np.uint8)
            return b.ImageResult(b.decode_result_bytes(imageio.encode_png(blank), self.name), self.id, "mock")

        h, w = req.height, req.width
        base = self._color(req.prompt.split("Art style:")[0])
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        stripes = 0.12 * np.sin(xx / 9.0) * np.cos(yy / 13.0)
        img = np.clip(base[None, None, :] + stripes[..., None], 0, 1)
        if req.image is not None:
            src = imageio.to_float(imageio.rgb(req.image))
            if src.shape[:2] != (h, w):
                src = imageio.resize(src, w, h)
            # keep the conditioning image's shading so structure is visible
            shade = imageio.luminance(src)[..., None]
            img = np.clip(img * (0.5 + 0.6 * shade), 0, 1)
            if req.mask is not None:
                m = req.mask
                if m.shape != (h, w):
                    m = imageio.resize_nearest(m, w, h)
                img = np.where(m[..., None], img, src)
        if req.references:
            ref = imageio.to_float(imageio.rgb(req.references[0])).reshape(-1, 3).mean(axis=0)
            img = np.clip(img * 0.6 + ref[None, None, :] * 0.4, 0, 1)
        data = imageio.encode_png(imageio.to_uint8(img))
        arr = b.decode_result_bytes(data, self.name)
        return b.ImageResult(arr, self.id, "mock", req.seed, {"call": n})

    def _test_connection(self, cfg):
        return "Mock provider ready (synthetic output, tests only)."
