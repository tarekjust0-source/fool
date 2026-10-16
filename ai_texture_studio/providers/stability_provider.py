"""Stability AI provider (REST v2beta "Stable Image" endpoints).

Endpoint selection:

* first view, no reference -> ``/v2beta/stable-image/control/structure``
  (keeps the render's structure; ``control_strength`` from Prompt Strength)
* first view with reference -> ``/v2beta/stable-image/control/style-transfer``
  (``init_image`` = render, ``style_image`` = reference)
* following views / local refinement -> ``/v2beta/stable-image/edit/inpaint``
  (greyscale mask, white = repaint; unmasked pixels are preserved)
* tile / text only -> ``/v2beta/stable-image/generate/core``

All requests are multipart; ``Accept: image/*`` returns raw image bytes.
The ``finish-reason`` response header reports ``CONTENT_FILTERED``.
"""

from __future__ import annotations

from ..core import http, imageio
from . import base as b

STYLE_PRESETS = ("3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance",
                 "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk",
                 "origami", "photographic", "pixel-art", "tile-texture")


class StabilityProvider(b.Provider):
    id = "STABILITY"
    name = "Stability AI"
    description = "Hosted Stable Image API: structure control, style transfer and true masked inpainting."
    default_base_url = "https://api.stability.ai"
    default_model = "stable-image"
    model_presets = ("stable-image",)
    key_env_var = "STABILITY_API_KEY"
    # Only the style-transfer endpoint accepts a reference image.
    reference_purposes = {"first_view"}
    data_notice = ("Sends view renders of the selected object, masks, your prompt and any reference image to "
                   "Stability AI (api.stability.ai). Requests consume Stability credits; see Stability AI's "
                   "pricing and terms.")

    def capabilities(self, cfg):
        return {b.TEXT_TO_IMAGE, b.IMAGE_EDIT, b.MASK, b.REFERENCE, b.NEGATIVE_PROMPT, b.SEED, b.STRENGTH,
                b.STYLE_PRESET}

    def _headers(self, cfg):
        return {"Authorization": f"Bearer {cfg.api_key}", "Accept": "image/*"}

    def _common(self, req, *, style=True):
        f = [http.MultipartField("prompt", req.prompt[:10000]), http.MultipartField("output_format", "png")]
        if req.negative_prompt:
            f.append(http.MultipartField("negative_prompt", req.negative_prompt[:10000]))
        if req.seed is not None:
            f.append(http.MultipartField("seed", str(int(req.seed) % 4294967294)))
        if style and req.style_preset in STYLE_PRESETS:
            f.append(http.MultipartField("style_preset", req.style_preset))
        return f

    def endpoint_for(self, req: b.ImageRequest) -> str:
        if req.image is None:
            return "generate/core"
        if req.purpose == "first_view":
            return "control/style-transfer" if req.references else "control/structure"
        return "edit/inpaint"

    def _generate(self, req, cfg, cancel=None):
        self.check_problems(cfg)
        ep = self.endpoint_for(req)
        fields = []
        if ep == "generate/core":
            fields = self._common(req) + [http.MultipartField("aspect_ratio", "1:1")]
        elif ep == "control/structure":
            fields = self._common(req) + [
                http.MultipartField("image", b.png_bytes(imageio.rgb(req.image)), "render.png", "image/png"),
                http.MultipartField("control_strength", f"{max(0.0, min(1.0, 1.0 - req.strength * 0.6)):.3f}"),
            ]
        elif ep == "control/style-transfer":
            fields = self._common(req, style=False) + [
                http.MultipartField("init_image", b.png_bytes(imageio.rgb(req.image)), "render.png", "image/png"),
                http.MultipartField("style_image", b.png_bytes(imageio.rgb(req.references[0])), "style.png",
                                    "image/png"),
                http.MultipartField("composition_fidelity", f"{max(0.2, 1.0 - req.strength * 0.5):.3f}"),
            ]
        else:
            mask = req.mask if req.mask is not None else None
            if mask is None:
                raise b.ProviderError("unsupported", "Stability inpaint requires a mask.", retryable=False)
            fields = self._common(req) + [
                http.MultipartField("image", b.png_bytes(imageio.rgb(req.image)), "render.png", "image/png"),
                http.MultipartField("mask", b.mask_png_white(mask), "mask.png", "image/png"),
                http.MultipartField("grow_mask", str(int(cfg.options.get("grow_mask", 5)))),
            ]
        url = f"{self.base(cfg)}/v2beta/stable-image/{ep}"
        try:
            resp = http.post_multipart(url, fields, headers=self._headers(cfg), timeout=cfg.timeout,
                                       retry=self.retry_policy(cfg), cancel=cancel)
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None
        if resp.headers.get("finish-reason", "").upper() == "CONTENT_FILTERED":
            raise b.ProviderError("content_policy", f"{self.name}: result was blocked by the content filter.",
                                  retryable=False)
        if resp.content_type.startswith("application/json"):
            raise b.ProviderError("invalid_response", f"{self.name}: expected an image, received JSON.")
        arr = b.decode_result_bytes(resp.body, self.name, b.FALLBACK_DECODER)
        seed = resp.headers.get("seed")
        return b.ImageResult(arr, self.id, ep, int(seed) if seed and seed.isdigit() else req.seed,
                             {"endpoint": ep})

    def _test_connection(self, cfg):
        self.check_problems(cfg)
        try:
            resp = http.get(f"{self.base(cfg)}/v1/user/balance", headers={"Authorization": f"Bearer {cfg.api_key}"},
                            timeout=30, retry=http.RetryPolicy(max_retries=0))
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None
        data = resp.json()
        credits = data.get("credits") if isinstance(data, dict) else None
        return f"Connected. Remaining credits: {credits}" if credits is not None else "Connected."
