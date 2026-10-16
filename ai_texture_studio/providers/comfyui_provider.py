"""ComfyUI provider (local or self-hosted inference, no data leaves your network).

Uses the standard ComfyUI server routes:

* ``POST /upload/image``   multipart upload into ComfyUI's input folder
* ``POST /prompt``         queue an API-format workflow -> ``prompt_id``
* ``GET  /history/{id}``   poll for completion and output file names
* ``GET  /view``           download an output image
* ``POST /interrupt``      cancel the running prompt

A built-in workflow (core nodes only) is generated for the selected
checkpoint: masked img2img via ``SetLatentNoiseMask`` with optional depth
ControlNet built from the real mesh depth render.  Alternatively users can
point to their own API-format workflow JSON containing placeholders:
``{{PROMPT}}``, ``{{NEGATIVE}}``, ``{{SEED}}``, ``{{DENOISE}}``, ``{{STEPS}}``,
``{{CFG}}``, ``{{IMAGE}}``, ``{{MASK}}``, ``{{DEPTH}}``, ``{{REFERENCE}}``,
``{{WIDTH}}``, ``{{HEIGHT}}``.
"""

from __future__ import annotations

import copy
import json
import random
import time
import urllib.parse
import uuid

from ..core import http, imageio
from . import base as b


def build_workflow(*, checkpoint: str, prompt: str, negative: str, seed: int, steps: int, cfg: float,
                   sampler: str, scheduler: str, denoise: float, image_name: str | None, mask_name: str | None,
                   depth_name: str | None, controlnet: str | None, control_strength: float,
                   width: int, height: int) -> dict:
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ATS"}},
    }
    if image_name:
        wf["4"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        wf["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}}
        latent = ["6", 0]
        if mask_name:
            wf["5"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
            wf["7"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["6", 0], "mask": ["5", 0]}}
            latent = ["7", 0]
    else:
        wf["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["6", 0]
        denoise = 1.0
    pos, neg = ["2", 0], ["3", 0]
    if controlnet and depth_name:
        wf["11"] = {"class_type": "LoadImage", "inputs": {"image": depth_name}}
        wf["12"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}}
        wf["13"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": pos, "negative": neg, "control_net": ["12", 0], "image": ["11", 0],
            "strength": control_strength, "start_percent": 0.0, "end_percent": 1.0}}
        pos, neg = ["13", 0], ["13", 1]
    wf["8"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": pos, "negative": neg, "latent_image": latent, "seed": seed,
        "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise}}
    return wf


def fill_placeholders(obj, values: dict):
    """Recursively replace "{{KEY}}" strings in a workflow."""
    if isinstance(obj, dict):
        return {k: fill_placeholders(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill_placeholders(v, values) for v in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith("{{") and stripped.endswith("}}") and stripped[2:-2] in values:
            return values[stripped[2:-2]]  # keep native type (int/float)
        for k, v in values.items():
            obj = obj.replace("{{" + k + "}}", str(v))
        return obj
    return obj


class ComfyUIProvider(b.Provider):
    id = "COMFYUI"
    name = "ComfyUI (local)"
    description = "Your own ComfyUI server (local GPU). Free, private, true masked inpainting + depth ControlNet."
    is_local = True
    requires_key = False
    default_base_url = "http://127.0.0.1:8188"
    default_model = ""
    data_notice = ("Sends renders/masks/prompt only to the ComfyUI server URL you configured. With the default "
                   "127.0.0.1 address nothing leaves your computer.")

    def capabilities(self, cfg):
        caps = {b.TEXT_TO_IMAGE, b.IMAGE_EDIT, b.MASK, b.NEGATIVE_PROMPT, b.SEED, b.STRENGTH}
        if cfg.options.get("controlnet"):
            caps.add(b.DEPTH_CONTROL)
        if cfg.options.get("workflow_json") and "{{REFERENCE}}" in cfg.options.get("workflow_json", ""):
            caps.add(b.REFERENCE)
        return caps

    def validate(self, cfg):
        problems = []
        if not cfg.options.get("workflow_json") and not cfg.model:
            problems.append("Set the ComfyUI checkpoint file name (e.g. 'sd_xl_base_1.0.safetensors').")
        if cfg.options.get("workflow_json"):
            try:
                json.loads(cfg.options["workflow_json"])
            except ValueError as exc:
                problems.append(f"Custom workflow is not valid JSON: {exc}")
        return problems

    def _headers(self, cfg):
        return {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}

    def _upload(self, cfg, data: bytes, name: str, cancel):
        fields = [http.MultipartField("image", data, name, "image/png"),
                  http.MultipartField("type", "input"), http.MultipartField("overwrite", "true")]
        resp = http.post_multipart(f"{self.base(cfg)}/upload/image", fields, headers=self._headers(cfg),
                                   timeout=60, retry=self.retry_policy(cfg), cancel=cancel)
        info = resp.json()
        fname = info.get("name", name)
        sub = info.get("subfolder") or ""
        return f"{sub}/{fname}" if sub else fname

    def _generate(self, req, cfg, cancel=None):
        self.check_problems(cfg)
        o = cfg.options
        seed = req.seed if req.seed is not None else random.randint(0, 2 ** 31 - 1)
        tag = uuid.uuid4().hex[:8]
        try:
            img_name = mask_name = depth_name = ref_name = None
            if req.image is not None:
                img_name = self._upload(cfg, b.png_bytes(imageio.rgb(req.image)), f"ats_{tag}_image.png", cancel)
            if req.mask is not None:
                mask_name = self._upload(cfg, b.mask_png_white(req.mask), f"ats_{tag}_mask.png", cancel)
            if req.depth is not None and (o.get("controlnet") or o.get("workflow_json")):
                depth_name = self._upload(cfg, imageio.encode_png(req.depth), f"ats_{tag}_depth.png", cancel)
            if req.references and o.get("workflow_json"):
                ref_name = self._upload(cfg, b.png_bytes(imageio.rgb(req.references[0])), f"ats_{tag}_ref.png", cancel)

            # Strength -> denoise: first view repaints clay entirely (structure
            # comes from ControlNet if configured), later views keep context.
            if req.purpose == "first_view":
                denoise = 1.0 if o.get("controlnet") else 0.55 + 0.4 * req.strength
            else:
                denoise = 0.6 + 0.4 * req.strength
            steps = int(o.get("steps", 28))
            cfg_scale = float(o.get("cfg", 6.5))
            if o.get("workflow_json"):
                values = {"PROMPT": req.prompt, "NEGATIVE": req.negative_prompt, "SEED": seed,
                          "DENOISE": round(denoise, 3), "STEPS": steps, "CFG": cfg_scale,
                          "IMAGE": img_name or "", "MASK": mask_name or "", "DEPTH": depth_name or "",
                          "REFERENCE": ref_name or "", "WIDTH": req.width, "HEIGHT": req.height}
                wf = fill_placeholders(copy.deepcopy(json.loads(o["workflow_json"])), values)
            else:
                wf = build_workflow(
                    checkpoint=cfg.model, prompt=req.prompt, negative=req.negative_prompt, seed=seed, steps=steps,
                    cfg=cfg_scale, sampler=o.get("sampler", "euler"), scheduler=o.get("scheduler", "normal"),
                    denoise=round(min(1.0, denoise), 3), image_name=img_name, mask_name=mask_name,
                    depth_name=depth_name, controlnet=o.get("controlnet") or None,
                    control_strength=float(o.get("control_strength", 0.8)), width=req.width, height=req.height)
            client_id = uuid.uuid4().hex
            resp = http.post_json(f"{self.base(cfg)}/prompt", {"prompt": wf, "client_id": client_id},
                                  headers=self._headers(cfg), timeout=60, retry=self.retry_policy(cfg), cancel=cancel)
            data = resp.json()
            if data.get("node_errors"):
                raise b.ProviderError("config", f"ComfyUI rejected the workflow: "
                                                f"{json.dumps(data['node_errors'])[:600]}", retryable=False)
            pid = data.get("prompt_id")
            if not pid:
                raise b.ProviderError("invalid_response", "ComfyUI did not return a prompt id.")
            outputs = self._wait(cfg, pid, cancel)
            img = self._first_image(outputs)
            if img is None:
                raise b.ProviderError("invalid_response", "ComfyUI workflow produced no image output.")
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            raw = http.get(f"{self.base(cfg)}/view?{q}", headers=self._headers(cfg), timeout=60,
                           retry=self.retry_policy(cfg), cancel=cancel).body
        except http.HTTPError as err:
            if err.kind == "http" and err.status == 400:
                detail = err.json()
                msg = json.dumps(detail)[:600] if detail else err.text()
                raise b.ProviderError("config", f"ComfyUI rejected the request: {msg}", retryable=False) from None
            raise b.classify_http_error(err, self.name) from None
        arr = b.decode_result_bytes(raw, self.name, b.FALLBACK_DECODER)
        return b.ImageResult(arr, self.id, cfg.model or "custom-workflow", seed, {"prompt_id": pid})

    def _wait(self, cfg, pid, cancel):
        deadline = time.monotonic() + cfg.timeout
        delay = 0.5
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                try:
                    http.request("POST", f"{self.base(cfg)}/interrupt", headers=self._headers(cfg), body=b"",
                                 timeout=10, retry=http.RetryPolicy(max_retries=0))
                except http.HTTPError:
                    pass
                raise b.ProviderError("cancelled", "Cancelled", retryable=False)
            hist = http.get(f"{self.base(cfg)}/history/{pid}", headers=self._headers(cfg), timeout=30,
                            retry=self.retry_policy(cfg), cancel=cancel).json()
            entry = hist.get(pid) if isinstance(hist, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message", "") if msgs and len(msgs[0]) > 1 else ""
                    raise b.ProviderError("server", f"ComfyUI execution failed: {detail}"[:600], retryable=False)
                if entry.get("outputs") and (status.get("completed", True)):
                    return entry["outputs"]
            time.sleep(delay)
            delay = min(2.0, delay * 1.5)
        raise b.ProviderError("timeout", "ComfyUI did not finish within the timeout.")

    @staticmethod
    def _first_image(outputs: dict):
        for node_id in sorted(outputs, key=lambda k: (len(k), k)):
            imgs = outputs[node_id].get("images") or []
            for im in imgs:
                if im.get("type", "output") == "output":
                    return im
        for node_id in outputs:
            imgs = outputs[node_id].get("images") or []
            if imgs:
                return imgs[0]
        return None

    def node_options(self, cfg, node: str, field: str) -> list[str]:
        """Choices of a node input, from ComfyUI's /object_info (checkpoints, ControlNets...)."""
        data = http.get(f"{self.base(cfg)}/object_info/{node}", headers=self._headers(cfg), timeout=20,
                        retry=http.RetryPolicy(max_retries=0)).json()
        spec = (((data.get(node) or {}).get("input") or {}).get("required") or {}).get(field)
        if not spec:
            return []
        first = spec[0]
        if isinstance(first, list):                         # classic format: [[names...], {...}]
            return [str(x) for x in first]
        if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):   # newer: ["COMBO", {"options": [...]}]
            return [str(x) for x in spec[1].get("options", [])]
        return []

    def _list_models(self, cfg):
        return self.node_options(cfg, "CheckpointLoaderSimple", "ckpt_name")

    def list_controlnets(self, cfg) -> list[str]:
        try:
            return self.node_options(cfg, "ControlNetLoader", "control_net_name")
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None

    def _test_connection(self, cfg):
        try:
            resp = http.get(f"{self.base(cfg)}/system_stats", headers=self._headers(cfg), timeout=15,
                            retry=http.RetryPolicy(max_retries=0))
        except http.HTTPError as err:
            raise b.classify_http_error(err, self.name) from None
        info = resp.json()
        ver = (info.get("system") or {}).get("comfyui_version", "")
        return f"Connected to ComfyUI {ver}".strip()
