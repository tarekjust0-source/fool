"""OpenAI GPT Image provider (``/v1/images/generations`` and ``/v1/images/edits``).

Request formats follow the public OpenAI Images API reference:

* edits: multipart form with ``model``, ``prompt``, repeated ``image[]``
  files (the first is the conditioning render, further ones are references),
  optional ``mask`` (PNG, same size as the first image, alpha 0 = edit),
  ``size``, ``quality``, ``n`` and ``input_fidelity`` (GPT Image 1 / 1.5).
* generations: JSON with ``model``, ``prompt``, ``size``, ``quality``, ``n``.
* responses carry base64 images in ``data[i].b64_json``.

Limitations (documented in the UI): no seed, no native negative prompt (it is
appended as an "Avoid:" sentence), the mask is a soft guide and the model may
redraw the whole frame slightly offset - the pipeline therefore aligns the
result and only projects masked pixels.
"""

from __future__ import annotations

import base64

from ..core import http, imageio
from . import base as b

SIZES = ("1024x1024", "1536x1024", "1024x1536")
FIDELITY_MODELS = ("gpt-image-1", "gpt-image-1.5")


class OpenAIProvider(b.Provider):
    id = "OPENAI"
    name = "OpenAI GPT Image"
    description = "Hosted GPT Image models. Strong prompt understanding and reference-image guidance."
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-image-1.5"
    model_presets = ("gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini", "gpt-image-2")
    key_env_var = "OPENAI_API_KEY"
    data_notice = ("Sends view renders of the selected object, masks, your prompt and any reference image to "
                   "OpenAI (api.openai.com). Usage is billed per image by OpenAI; see OpenAI's pricing and "
                   "data-usage policies.")

    def capabilities(self, cfg):
        return {b.TEXT_TO_IMAGE, b.IMAGE_EDIT, b.SOFT_MASK, b.REFERENCE, b.MAY_MISALIGN, b.STRENGTH}

    def _headers(self, cfg):
        h = {"Authorization": f"Bearer {cfg.api_key}"}
        org = cfg.options.get("organization")
        if org:
            h["OpenAI-Organization"] = org
        return h

    def _prompt(self, req: b.ImageRequest) -> str:
        p = req.prompt
        if req.negative_prompt:
            p += f" Avoid: {req.negative_prompt}."
        return p[:30000]

    def _size(self, req):
        s = f"{req.width}x{req.height}"
        return s if s in SIZES else "1024x1024"

    def _generate(self, req: b.ImageRequest, cfg: b.ProviderConfig, cancel=None) -> b.ImageResult:
        self.check_problems(cfg)
        model = self.model(cfg)
        quality = cfg.options.get("quality", "high")
        try:
            if req.image is None and not req.references:
                payload = {"model": model, "prompt": self._prompt(req), "size": self._size(req),
                           "quality": quality, "n": 1}
                resp = http.post_json(f"{self.base(cfg)}/images/generations", payload, headers=self._headers(cfg),
                                      timeout=cfg.timeout, retry=self.retry_policy(cfg), cancel=cancel)
            else:
                resp = self._edit(req, cfg, model, quality, cancel, with_fidelity=True)
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None
        return self._parse(resp, model, cancel, cfg)

    def _edit(self, req, cfg, model, quality, cancel, *, with_fidelity):
        first = req.image if req.image is not None else req.references[0]
        refs = req.references if req.image is not None else req.references[1:]
        first = imageio.rgb(first)
        fields = [
            http.MultipartField("model", model),
            http.MultipartField("prompt", self._prompt(req)),
            http.MultipartField("n", "1"),
            http.MultipartField("size", self._size(req)),
            http.MultipartField("quality", quality),
            http.MultipartField("image[]", b.png_bytes(first), "render.png", "image/png"),
        ]
        for i, ref in enumerate(refs[:8]):
            fields.append(http.MultipartField("image[]", b.png_bytes(imageio.rgb(ref)), f"reference_{i}.png",
                                              "image/png"))
        if req.mask is not None and req.image is not None:
            m = req.mask
            if m.shape != first.shape[:2]:
                m = imageio.resize_nearest(m, first.shape[1], first.shape[0])
            fields.append(http.MultipartField("mask", b.mask_png_alpha(m, first), "mask.png", "image/png"))
        fidelity = None
        if with_fidelity and any(model == fm for fm in FIDELITY_MODELS):
            fidelity = "high" if req.strength <= 0.6 or req.references else "low"
            fields.append(http.MultipartField("input_fidelity", fidelity))
        try:
            return http.post_multipart(f"{self.base(cfg)}/images/edits", fields, headers=self._headers(cfg),
                                       timeout=cfg.timeout, retry=self.retry_policy(cfg), cancel=cancel)
        except http.HTTPError as err:
            # Some model versions reject optional parameters: retry once without.
            if fidelity and err.status == 400 and "input_fidelity" in err.text(2000):
                return self._edit(req, cfg, model, quality, cancel, with_fidelity=False)
            raise

    def _parse(self, resp: http.Response, model: str, cancel, cfg) -> b.ImageResult:
        data = resp.json()
        items = data.get("data") if isinstance(data, dict) else None
        if not items:
            raise b.ProviderError("invalid_response", f"{self.name}: response contained no image.")
        item = items[0]
        raw = None
        if item.get("b64_json"):
            try:
                raw = base64.b64decode(item["b64_json"], validate=False)
            except Exception:
                raise b.ProviderError("invalid_response", f"{self.name}: invalid base64 image data.") from None
        elif item.get("url", "").startswith("https://"):
            try:
                raw = http.get(item["url"], timeout=cfg.timeout, cancel=cancel).body
            except http.HTTPError as err:
                raise b.classify_http_error(err, self.name) from None
        if not raw:
            raise b.ProviderError("invalid_response", f"{self.name}: response contained no image data.")
        arr = b.decode_result_bytes(raw, self.name, b.FALLBACK_DECODER)
        meta = {"revised_prompt": item.get("revised_prompt", "")}
        if isinstance(data.get("usage"), dict):
            meta["usage"] = data["usage"]
        return b.ImageResult(arr, self.id, model, None, meta)

    def _test_connection(self, cfg):
        self.check_problems(cfg)
        try:
            resp = http.get(f"{self.base(cfg)}/models/{self.model(cfg)}", headers=self._headers(cfg),
                            timeout=30, retry=http.RetryPolicy(max_retries=0))
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None
        info = resp.json()
        return f"Connected. Model '{info.get('id', self.model(cfg))}' is available."
