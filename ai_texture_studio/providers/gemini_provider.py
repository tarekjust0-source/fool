"""Google Gemini image models via the Generative Language REST API.

``POST {base}/v1beta/models/{model}:generateContent`` with header
``x-goog-api-key``.  Images are sent as ``inline_data`` parts and returned
as ``inlineData`` parts (base64).  There is no mask parameter: the mask is
sent as an extra black/white image and described in the prompt, and the
pipeline only projects masked pixels back, so locality is still enforced.
"""

from __future__ import annotations

import base64

from ..core import http, imageio
from . import base as b

BLOCK_REASONS = ("SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "BLOCKLIST", "SPII", "RECITATION")


class GeminiProvider(b.Provider):
    id = "GEMINI"
    name = "Google Gemini Image"
    description = "Hosted Gemini image models; multi-image input for consistent design across views."
    default_base_url = "https://generativelanguage.googleapis.com"
    default_model = "gemini-2.5-flash-image"
    model_presets = ("gemini-2.5-flash-image", "gemini-3-pro-image-preview")
    key_env_var = "GEMINI_API_KEY"
    mask_as_image = True
    data_notice = ("Sends view renders of the selected object, masks, your prompt and any reference image to "
                   "Google (generativelanguage.googleapis.com). Usage is billed by Google; free-tier data may "
                   "be used to improve Google products - see Google's Gemini API terms.")

    def capabilities(self, cfg):
        return {b.TEXT_TO_IMAGE, b.IMAGE_EDIT, b.SOFT_MASK, b.REFERENCE, b.MAY_MISALIGN}

    @staticmethod
    def _part(arr):
        return {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(b.png_bytes(arr)).decode("ascii")}}

    def build_payload(self, req: b.ImageRequest, with_image_config=True) -> dict:
        text = req.prompt
        if req.negative_prompt:
            text += f" Avoid: {req.negative_prompt}."
        parts = [{"text": text}]
        if req.image is not None:
            parts.append({"text": "Image 1: the render to paint."})
            parts.append(self._part(imageio.rgb(req.image)))
            if req.mask is not None:
                parts.append({"text": "Image 2: mask (white = region to paint, black = keep unchanged)."})
                parts.append(self._part((req.mask.astype("uint8") * 255)))
        for i, ref in enumerate(req.references[:6]):
            parts.append({"text": f"Reference image {i + 1} for design, colours and materials:"})
            parts.append(self._part(imageio.rgb(ref)))
        gen_cfg = {"responseModalities": ["TEXT", "IMAGE"]}
        if with_image_config:
            gen_cfg["imageConfig"] = {"aspectRatio": "1:1"}
        return {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen_cfg}

    def _generate(self, req, cfg, cancel=None):
        self.check_problems(cfg)
        model = self.model(cfg)
        url = f"{self.base(cfg)}/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": cfg.api_key}
        try:
            try:
                resp = http.post_json(url, self.build_payload(req), headers=headers, timeout=cfg.timeout,
                                      retry=self.retry_policy(cfg), cancel=cancel)
            except http.HTTPError as err:
                if err.status == 400 and "imageConfig" in err.text(2000):
                    resp = http.post_json(url, self.build_payload(req, False), headers=headers,
                                          timeout=cfg.timeout, retry=self.retry_policy(cfg), cancel=cancel)
                else:
                    raise
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None
        return self.parse(resp.json(), model)

    def parse(self, data, model):
        if not isinstance(data, dict):
            raise b.ProviderError("invalid_response", f"{self.name}: malformed response.")
        fb = data.get("promptFeedback") or {}
        if fb.get("blockReason"):
            raise b.ProviderError("content_policy", f"{self.name}: prompt blocked ({fb['blockReason']}).",
                                  retryable=False)
        cands = data.get("candidates") or []
        texts = []
        for cand in cands:
            reason = (cand.get("finishReason") or "").upper()
            for part in (cand.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    try:
                        raw = base64.b64decode(inline["data"])
                    except Exception:
                        raise b.ProviderError("invalid_response", f"{self.name}: invalid base64 image.") from None
                    arr = b.decode_result_bytes(raw, self.name, b.FALLBACK_DECODER)
                    return b.ImageResult(arr, self.id, model, None, {"finish_reason": reason})
                if part.get("text"):
                    texts.append(part["text"])
            if reason in BLOCK_REASONS:
                raise b.ProviderError("content_policy", f"{self.name}: generation blocked ({reason}).",
                                      retryable=False)
        hint = (" Model said: " + " ".join(texts)[:300]) if texts else ""
        raise b.ProviderError("invalid_response", f"{self.name}: response contained no image.{hint}")

    def _test_connection(self, cfg):
        self.check_problems(cfg)
        try:
            resp = http.get(f"{self.base(cfg)}/v1beta/models/{self.model(cfg)}",
                            headers={"x-goog-api-key": cfg.api_key}, timeout=30,
                            retry=http.RetryPolicy(max_retries=0))
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None
        info = resp.json()
        return f"Connected. Model '{info.get('displayName') or info.get('name') or self.model(cfg)}' is available."
