"""Generation pipelines (run inside worker threads; no bpy).

Each pipeline reports progress via the Job and emits results as events,
which the Blender bridge turns into layers on the main thread.  Nothing is
ever emitted for a failed request: a failed view raises, and earlier views
that did succeed remain as real layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import imageio, masks
from .camera import VIEW_LABELS, ViewCamera
from .jobs import Job
from .meshdata import MeshData
from .projection import (ProjectionSettings, TextureState, ViewPlan, finalize_fill, plan_view, prepare_generated,
                         project_image_from_camera, project_view)
from .prompting import StyleSettings, compose
from .render import pixel_attributes, render_view


class PipelineError(Exception):
    """User-facing pipeline problem (not a provider failure)."""


@dataclass
class GenerationSpec:
    prompt: str
    negative: str = ""
    style: StyleSettings = field(default_factory=StyleSettings)
    seed: int | None = None
    references: list = field(default_factory=list)   # (H,W,3) uint8
    request_size: int = 1024


def _caps(provider, cfg):
    from ..providers import base as pb
    caps = provider.capabilities(cfg)
    return caps, pb


def _build_request(provider, cfg, spec: GenerationSpec, *, purpose, view_label, image, mask, depth):
    caps, pb = _caps(provider, cfg)
    refs = list(spec.references) if pb.REFERENCE in caps else []
    ref_scope = getattr(provider, "reference_purposes", None)
    if ref_scope is not None and purpose not in ref_scope:
        refs = []
    mask_as_image = getattr(provider, "mask_as_image", False) and mask is not None
    cp = compose(spec.prompt, spec.negative, spec.style, purpose=purpose, view_name=view_label,
                 has_reference=bool(refs), has_mask_image=mask_as_image)
    size = spec.request_size
    img = None if image is None else imageio.to_uint8(image)
    msk = mask
    dep = None if depth is None else imageio.to_uint8(depth)
    if img is not None and img.shape[:2] != (size, size):
        img = imageio.resize(img, size, size)
        if msk is not None:
            msk = imageio.resize_nearest(msk, size, size)
        if dep is not None:
            dep = imageio.resize(dep, size, size)
    return pb.ImageRequest(
        prompt=cp.prompt, negative_prompt=cp.negative, purpose=purpose, image=img, mask=msk, depth=dep,
        references=refs, width=size, height=size, seed=spec.seed, strength=cp.strength,
        style_preset=cp.style_preset if pb.STYLE_PRESET in caps else None,
    ), cp


def run_multiview(job: Job, mesh: MeshData, cameras: list[ViewCamera], state: TextureState, provider, cfg,
                  spec: GenerationSpec, ps: ProjectionSettings, *, object_name: str = "",
                  progress_range=(0.0, 1.0)) -> dict:
    caps, pb = _caps(provider, cfg)
    may_misalign = pb.MAY_MISALIGN in caps
    n = max(len(cameras), 1)
    p0, p1 = progress_range
    summary = {"views": [], "skipped": [], "calls": 0}

    def prog(i, frac, msg):
        job.set_progress(p0 + (p1 - p0) * (i + frac) / n, msg)

    for i, cam in enumerate(cameras):
        job.check_cancel()
        label = cam.meta.get("label") or VIEW_LABELS.get(cam.name, cam.name)
        prog(i, 0.0, f"{object_name} {label}: rendering view ({i + 1}/{n})")
        plan: ViewPlan = plan_view(mesh, cam, state, ps)
        if plan.skip:
            summary["skipped"].append(label)
            job.emit("view_skipped", {"object": object_name, "view": cam.name,
                                      "view_key": cam.meta.get("key", cam.name), "label": label,
                                      "untextured_fraction": plan.untextured_fraction})
            continue
        purpose = "first_view" if plan.first else "next_view"
        req, cp = _build_request(provider, cfg, spec, purpose=purpose, view_label=label.lower(),
                                 image=plan.render.composite, mask=plan.ai_mask, depth=plan.render.depth_image)
        prog(i, 0.15, f"{object_name} {label}: waiting for {provider.name} ({i + 1}/{n})")
        result = provider.generate(req, cfg, cancel=job.cancel_event)
        summary["calls"] += 1
        if "first_generated" not in summary:
            summary["first_generated"] = imageio.to_uint8(imageio.rgb(result.image))
        job.check_cancel()
        prog(i, 0.8, f"{object_name} {label}: projecting onto UVs")
        gen, info = prepare_generated(plan, result.image, ps, may_misalign)
        layer, stats = project_view(mesh, plan, gen, state, ps)
        state.apply_layer(layer)
        job.emit("view_done", {
            "object": object_name, "view": cam.name, "view_key": cam.meta.get("key", cam.name), "label": label,
            "camera": cam.to_dict(),
            "state_rgb": state.rgb, "state_alpha": state.alpha,
            "layer": imageio.to_uint8(layer), "stats": stats, "align": info, "purpose": purpose,
            "prompt": cp.as_dict(), "provider": provider.id, "model": result.model, "seed": result.seed,
            "debug": {
                "render": imageio.to_uint8(plan.render.composite),
                "mask": (plan.ai_mask.astype(np.uint8) * 255),
                "generated": imageio.to_uint8(imageio.rgb(result.image)),
                "aligned": imageio.to_uint8(gen),
                "depth": imageio.to_uint8(plan.render.depth_image),
            },
        })
        summary["views"].append(label)
    job.check_cancel()
    if summary["calls"] == 0:
        # Nothing was generated: existing low-angle / fill layers stay valid as they are.
        summary.update({"fallback_fraction": None, "unseen_fraction": None})
        summary["textured_fraction"] = state.textured_fraction()
        return summary
    grazing, fill, fstats = finalize_fill(state, ps)
    summary.update(fstats)
    # Both sit *below* the generated layers, so only still-unpainted texels change.
    for kind, layer in (("grazing_done", grazing), ("fill_done", fill)):
        if layer is None:
            continue
        top_a = state.alpha[..., None]               # existing paint stays on top
        has = layer[..., 3:4] > 0
        state.rgb = np.where(has, state.rgb * top_a + layer[..., :3] * (1 - top_a), state.rgb)
        state.alpha = np.maximum(state.alpha, layer[..., 3])
        job.emit(kind, {"object": object_name, "layer": imageio.to_uint8(layer), "stats": fstats,
                        "state_rgb": state.rgb, "state_alpha": state.alpha})
    summary["textured_fraction"] = state.textured_fraction()
    return summary


def run_refine_view(job: Job, mesh: MeshData, cam: ViewCamera, state: TextureState, uv_mask: np.ndarray,
                    provider, cfg, spec: GenerationSpec, *, grow_px: int = 6, object_name: str = "") -> dict:
    """Local correction seen through a camera (usually the current viewport)."""
    caps, pb = _caps(provider, cfg)
    job.set_progress(0.05, "Rendering current view")
    col, a = state.display(4)
    vr = render_view(mesh, cam, col, a)
    m, uv, _ = pixel_attributes(mesh, vr.geo)
    size = uv_mask.shape[0]
    screen = np.zeros(vr.mask.shape, bool)
    if m.any():
        vals = imageio.sample_nearest(uv_mask, uv[:, 0] * size, (1.0 - uv[:, 1]) * size)
        screen[m] = vals > 0.5
    if screen.sum() < 16:
        raise PipelineError("The masked area is not visible from this view. Orbit the viewport so the "
                            "region faces the camera, or use UV-space refinement.")
    ai_mask = masks.dilate(screen, grow_px) & vr.mask
    req, cp = _build_request(provider, cfg, spec, purpose="refine", view_label="current",
                             image=vr.composite, mask=ai_mask, depth=vr.depth_image)
    job.set_progress(0.2, f"Waiting for {provider.name}")
    result = provider.generate(req, cfg, cancel=job.cancel_event)
    job.check_cancel()
    job.set_progress(0.85, "Projecting correction")
    plan = ViewPlan(vr, ai_mask, screen, False, 0.0, False)
    ps = ProjectionSettings(texture_size=size, view_size=cam.width)
    gen, info = prepare_generated(plan, result.image, ps, pb.MAY_MISALIGN in caps)
    rgb, w, stats = project_image_from_camera(mesh, cam, gen, uv_mask > 0.01, size)
    alpha = np.clip(uv_mask, 0, 1) * w
    layer = np.concatenate([rgb, alpha[..., None]], axis=2)
    payload = {"object": object_name, "layer": imageio.to_uint8(layer), "stats": stats, "align": info,
               "prompt": cp.as_dict(), "provider": provider.id, "model": result.model,
               "debug": {"render": imageio.to_uint8(vr.composite), "mask": ai_mask.astype(np.uint8) * 255,
                         "generated": imageio.to_uint8(imageio.rgb(result.image)), "aligned": imageio.to_uint8(gen)}}
    job.emit("refine_done", payload)
    return stats


def run_refine_uv(job: Job, state: TextureState, uv_mask: np.ndarray, provider, cfg, spec: GenerationSpec, *,
                  object_name: str = "") -> dict:
    """Local correction performed directly on the UV atlas."""
    size = state.size
    sel = uv_mask > 0.01
    bb = masks.bbox(sel)
    if bb is None:
        raise PipelineError("The mask is empty.")
    y0, y1, x0, x1 = bb
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max(y1 - y0, x1 - x0) * 0.65 + 16
    half = min(half, size / 2)
    y0 = int(max(0, min(size - 2 * half, cy - half)))
    x0 = int(max(0, min(size - 2 * half, cx - half)))
    side = int(2 * half)
    y1, x1 = min(size, y0 + side), min(size, x0 + side)
    col, _ = state.display(16)
    crop = col[y0:y1, x0:x1]
    mcrop = uv_mask[y0:y1, x0:x1] > 0.5
    req, cp = _build_request(provider, cfg, spec, purpose="refine_uv", view_label="uv",
                             image=crop, mask=masks.dilate(mcrop, 4), depth=None)
    job.set_progress(0.2, f"Waiting for {provider.name}")
    result = provider.generate(req, cfg, cancel=job.cancel_event)
    job.check_cancel()
    gen = imageio.resize(imageio.to_float(imageio.rgb(result.image)), x1 - x0, y1 - y0)
    layer = np.zeros((size, size, 4), np.float32)
    layer[y0:y1, x0:x1, :3] = gen
    layer[y0:y1, x0:x1, 3] = np.clip(uv_mask[y0:y1, x0:x1], 0, 1)
    covered = state.uv.covered_mask()
    layer[..., 3] *= masks.dilate(covered, 2)
    stats = {"crop": [y0, y1, x0, x1], "masked_texels": int(sel.sum())}
    job.emit("refine_done", {"object": object_name, "layer": imageio.to_uint8(layer), "stats": stats,
                             "prompt": cp.as_dict(), "provider": provider.id, "model": result.model,
                             "debug": {"render": imageio.to_uint8(crop), "mask": mcrop.astype(np.uint8) * 255,
                                       "generated": imageio.to_uint8(imageio.rgb(result.image))}})
    return stats


def make_seamless(img: np.ndarray) -> np.ndarray:
    """Blend an image with its half-offset copy so it tiles without seams."""
    f = imageio.to_float(img)[..., :3]
    h, w = f.shape[:2]
    off = np.roll(np.roll(f, h // 2, axis=0), w // 2, axis=1)
    y = np.abs(np.linspace(-1, 1, h))[:, None]
    x = np.abs(np.linspace(-1, 1, w))[None, :]
    wgt = np.clip(1.0 - np.maximum(x, y), 0, 1) ** 0.7
    wgt = np.clip(wgt * 2.0, 0, 1)[..., None]
    return f * wgt + off * (1 - wgt)


def run_tile(job: Job, mesh: MeshData, state: TextureState, provider, cfg, spec: GenerationSpec, *,
             scale: float = 4.0, object_name: str = "") -> dict:
    req, cp = _build_request(provider, cfg, spec, purpose="tile", view_label="tile", image=None, mask=None,
                             depth=None)
    job.set_progress(0.2, f"Waiting for {provider.name}")
    result = provider.generate(req, cfg, cancel=job.cancel_event)
    job.check_cancel()
    tile = make_seamless(result.image)
    th, tw = tile.shape[:2]
    uvs = state.uv
    uv = mesh.uv_points(uvs.tri, uvs.bary)
    u = np.mod(uv[:, 0] * scale, 1.0)
    v = np.mod(uv[:, 1] * scale, 1.0)
    col = imageio.sample_bilinear(tile, u * tw, (1.0 - v) * th)
    size = state.size
    layer = np.zeros((size * size, 4), np.float32)
    layer[uvs.index, :3] = col
    layer[uvs.index, 3] = 1.0
    job.emit("tile_done", {"object": object_name, "layer": imageio.to_uint8(layer.reshape(size, size, 4)),
                           "prompt": cp.as_dict(), "provider": provider.id, "model": result.model,
                           "debug": {"generated": imageio.to_uint8(imageio.rgb(result.image)),
                                     "tile": imageio.to_uint8(tile)}})
    return {"scale": scale}
