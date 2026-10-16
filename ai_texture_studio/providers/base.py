"""Provider abstraction.

A provider turns an :class:`ImageRequest` (prompt + conditioning render +
optional mask/reference images) into an image.  Providers are pure Python:
they never touch ``bpy`` and run inside worker threads.

Capabilities are declared explicitly so the UI/pipeline can tell users which
controls a provider actually honours, instead of silently ignoring them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import http, imageio
from ..core.secrets import redact

# Capability flags
TEXT_TO_IMAGE = "text_to_image"      # generate without an input image
IMAGE_EDIT = "image_edit"            # edit / condition on an input image
MASK = "mask"                        # honours an explicit inpainting mask
SOFT_MASK = "soft_mask"              # mask is a hint; model may repaint outside
REFERENCE = "reference"              # extra reference image(s) for style/design
NEGATIVE_PROMPT = "negative_prompt"  # native negative prompt field
SEED = "seed"                        # deterministic seed
STRENGTH = "strength"                # numeric control of input adherence
STYLE_PRESET = "style_preset"        # native style preset enum
DEPTH_CONTROL = "depth_control"      # uses a depth map for structure (ControlNet)
MAY_MISALIGN = "may_misalign"        # output can drift spatially; enable auto-align

ERROR_HINTS = {
    "auth": "Check the API key in Preferences > Add-ons > AI Texture Studio.",
    "config": "Check the provider settings (model name, URL, workflow).",
    "rate_limit": "The provider is rate limiting requests. Wait a moment and press Retry.",
    "quota": "Your account has no remaining credit/quota for this provider.",
    "network": "Could not reach the provider. Check your internet connection / proxy and Blender's "
               "'Allow Online Access' setting.",
    "timeout": "The provider took too long. Press Retry or raise the timeout in Preferences.",
    "content_policy": "The provider refused the request (content policy). Rephrase the prompt.",
    "invalid_response": "The provider returned an unusable result. Press Retry.",
    "server": "The provider reported a server error. Press Retry later.",
    "unsupported": "This provider/model does not support the requested operation.",
    "cancelled": "",
}


class ProviderError(Exception):
    """Error with a user-facing classification. Messages are redacted."""

    def __init__(self, kind: str, message: str, *, retryable: bool | None = None, status: int | None = None):
        super().__init__(redact(message))
        self.kind = kind
        self.status = status
        if retryable is None:
            retryable = kind in ("rate_limit", "network", "timeout", "server", "invalid_response")
        self.retryable = retryable

    def user_message(self) -> str:
        hint = ERROR_HINTS.get(self.kind, "")
        msg = str(self)
        return f"{msg} {hint}".strip() if hint and hint not in msg else msg


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str = ""
    purpose: str = "first_view"      # first_view | next_view | refine | refine_uv | tile
    image: np.ndarray | None = None  # (H,W,3) uint8 conditioning / context image
    mask: np.ndarray | None = None   # (H,W) bool, True = paint here
    depth: np.ndarray | None = None  # (H,W) uint8 depth (near = white)
    references: list = field(default_factory=list)  # list of (H,W,3) uint8
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    strength: float = 0.7            # 0 = adhere to input, 1 = follow prompt freely
    style_preset: str | None = None

    def summary(self) -> dict:
        """What will be transmitted (for consent dialogs / history)."""
        return {
            "purpose": self.purpose,
            "prompt_chars": len(self.prompt),
            "has_image": self.image is not None,
            "has_mask": self.mask is not None,
            "has_depth": self.depth is not None,
            "references": len(self.references),
            "size": f"{self.width}x{self.height}",
        }


@dataclass
class ImageResult:
    image: np.ndarray               # (H,W,3|4) uint8
    provider: str
    model: str
    seed: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Plain data handed to providers (resolved on the main thread)."""

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout: float = 180.0
    max_retries: int = 2
    options: dict = field(default_factory=dict)

    def __repr__(self):  # never print the key
        return (f"ProviderConfig(model={self.model!r}, base_url={self.base_url!r}, api_key="
                f"{'<set>' if self.api_key else '<empty>'}, timeout={self.timeout}, options={self.options!r})")


class Provider:
    id = "base"
    name = "Base"
    description = ""
    is_local = False
    requires_key = True
    default_base_url = ""
    default_model = ""
    model_presets: tuple = ()
    data_notice = ""
    key_env_var = ""

    def capabilities(self, cfg: ProviderConfig) -> set:
        return set()

    def validate(self, cfg: ProviderConfig) -> list[str]:
        problems = []
        if self.requires_key and not cfg.api_key:
            problems.append(f"No API key configured for {self.name}.")
        if not (cfg.model or self.default_model):
            problems.append("No model selected.")
        return problems

    def generate(self, req: ImageRequest, cfg: ProviderConfig, cancel=None) -> ImageResult:
        """Generate an image. Every transport/parse failure becomes a ProviderError."""
        try:
            return self._generate(req, cfg, cancel)
        except http.HTTPError as err:
            raise classify_http_error(err, self.name) from None

    def test_connection(self, cfg: ProviderConfig) -> str:
        try:
            return self._test_connection(cfg)
        except http.HTTPError as err:
            raise classify_http_error(err, self.name) from None

    # Subclasses implement these two.
    def _generate(self, req: ImageRequest, cfg: ProviderConfig, cancel=None) -> ImageResult:
        raise NotImplementedError

    def _test_connection(self, cfg: ProviderConfig) -> str:
        raise NotImplementedError

    # -- helpers -------------------------------------------------------------
    def base(self, cfg: ProviderConfig) -> str:
        return (cfg.base_url or self.default_base_url).rstrip("/")

    def model(self, cfg: ProviderConfig) -> str:
        return cfg.model or self.default_model

    def retry_policy(self, cfg: ProviderConfig) -> http.RetryPolicy:
        return http.RetryPolicy(max_retries=int(cfg.max_retries))

    def check_problems(self, cfg: ProviderConfig):
        problems = self.validate(cfg)
        if problems:
            kind = "auth" if any("API key" in p for p in problems) else "config"
            raise ProviderError(kind, " ".join(problems), retryable=False)


def classify_http_error(err: http.HTTPError, provider_name: str) -> ProviderError:
    """Map transport/HTTP failures onto ProviderError kinds."""
    if err.kind == "cancelled":
        return ProviderError("cancelled", "Cancelled", retryable=False)
    if err.kind == "timeout":
        return ProviderError("timeout", f"{provider_name}: request timed out.")
    if err.kind == "network":
        return ProviderError("network", f"{provider_name}: {err}")
    if err.kind == "too_large":
        return ProviderError("invalid_response", f"{provider_name}: response too large.")
    if err.kind == "invalid":
        return ProviderError("invalid_response", f"{provider_name}: unexpected response ({err}).")
    detail = _error_detail(err)
    s = err.status or 0
    base = f"{provider_name} error {s}" + (f": {detail}" if detail else "")
    if s in (401, 403):
        if "quota" in detail.lower() or "credit" in detail.lower() or "billing" in detail.lower():
            return ProviderError("quota", base, retryable=False, status=s)
        return ProviderError("auth", base, retryable=False, status=s)
    if s == 402:
        return ProviderError("quota", base, retryable=False, status=s)
    if s == 429:
        low = detail.lower()
        if "quota" in low or "billing" in low or "insufficient" in low:
            return ProviderError("quota", base, retryable=False, status=s)
        return ProviderError("rate_limit", base, status=s)
    if s in (400, 404, 413, 415, 422):
        low = detail.lower()
        if any(w in low for w in ("safety", "moderation", "content policy", "content_policy", "nsfw", "blocked")):
            return ProviderError("content_policy", base, retryable=False, status=s)
        return ProviderError("config" if s in (404,) else "unsupported" if s == 415 else "config", base,
                             retryable=False, status=s)
    if s >= 500:
        return ProviderError("server", base, status=s)
    return ProviderError("server", base, status=s)


def _error_detail(err: http.HTTPError) -> str:
    data = err.json()
    msg = ""
    if isinstance(data, dict):
        e = data.get("error")
        if isinstance(e, dict):
            msg = e.get("message") or e.get("code") or ""
        elif isinstance(e, str):
            msg = e
        if not msg and isinstance(data.get("errors"), list):
            msg = "; ".join(str(x) for x in data["errors"][:3])
        if not msg:
            msg = data.get("message") or data.get("name") or ""
    if not msg:
        msg = err.text(300)
    return redact(str(msg))[:500]


def decode_result_bytes(data: bytes, provider_name: str, decoder=None) -> np.ndarray:
    """Decode and validate image bytes from a provider.

    ``decoder`` is an optional fallback (bytes -> array) used for formats the
    pure-Python decoder cannot read (supplied by the Blender bridge).
    """
    try:
        arr = imageio.decode_image(data)
    except imageio.ImageDecodeError:
        if decoder is None:
            raise ProviderError("invalid_response", f"{provider_name} returned an image format that "
                                                    "could not be decoded.") from None
        try:
            arr = decoder(data)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError("invalid_response", f"{provider_name}: could not decode image ({exc}).") from None
    except imageio.InvalidImageError as exc:
        raise ProviderError("invalid_response", f"{provider_name}: {exc}") from None
    try:
        return imageio.validate_image_array(arr)
    except imageio.InvalidImageError as exc:
        raise ProviderError("invalid_response", f"{provider_name}: {exc}") from None


# Optional fallback decoder installed by the Blender bridge at register time.
FALLBACK_DECODER = None


def png_bytes(arr: np.ndarray) -> bytes:
    return imageio.encode_png(arr, level=6)


def mask_png_white(mask: np.ndarray) -> bytes:
    """Greyscale PNG, white = paint (Stability / ComfyUI convention)."""
    return imageio.encode_png((mask.astype(np.uint8) * 255))


def mask_png_alpha(mask: np.ndarray, image: np.ndarray) -> bytes:
    """RGBA PNG of ``image`` with alpha 0 where ``mask`` (OpenAI convention)."""
    rgba = np.concatenate([image[..., :3], np.where(mask, 0, 255).astype(np.uint8)[..., None]], axis=2)
    return imageio.encode_png(rgba)
