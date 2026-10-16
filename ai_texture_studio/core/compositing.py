"""Non-destructive layer compositing (numpy, straight alpha, sRGB values)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLEND_MODES = ("NORMAL", "MULTIPLY", "SCREEN", "OVERLAY", "SOFT_LIGHT", "ADD", "DARKEN", "LIGHTEN")


@dataclass
class LayerInput:
    pixels: np.ndarray    # (H,W,4) uint8 or float, straight alpha
    opacity: float = 1.0
    blend: str = "NORMAL"
    mask: np.ndarray | None = None  # optional (H,W) float 0..1 multiplier


def _blend(mode: str, base: np.ndarray, top: np.ndarray) -> np.ndarray:
    if mode == "NORMAL":
        return top
    if mode == "MULTIPLY":
        return base * top
    if mode == "SCREEN":
        return 1 - (1 - base) * (1 - top)
    if mode == "OVERLAY":
        return np.where(base < 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
    if mode == "SOFT_LIGHT":
        return (1 - 2 * top) * base * base + 2 * top * base
    if mode == "ADD":
        return np.minimum(base + top, 1.0)
    if mode == "DARKEN":
        return np.minimum(base, top)
    if mode == "LIGHTEN":
        return np.maximum(base, top)
    raise ValueError(f"Unknown blend mode {mode}")


def _as_float(px: np.ndarray, rows: slice) -> np.ndarray:
    chunk = px[rows]
    if chunk.dtype == np.uint8:
        return chunk.astype(np.float32) * (1.0 / 255.0)
    return chunk.astype(np.float32, copy=False)


def composite(layers: list[LayerInput], size: int, *, rows_per_chunk: int = 256):
    """Composite bottom-to-top. Returns (rgb float32 (S,S,3), alpha float32 (S,S))."""
    out_rgb = np.zeros((size, size, 3), np.float32)
    out_a = np.zeros((size, size), np.float32)
    for r0 in range(0, size, rows_per_chunk):
        rows = slice(r0, min(size, r0 + rows_per_chunk))
        c = out_rgb[rows]
        a = out_a[rows]
        for layer in layers:
            if layer.opacity <= 0:
                continue
            if layer.pixels.shape[0] != size or layer.pixels.shape[1] != size:
                raise ValueError("Layer size mismatch; resize layers before compositing")
            px = _as_float(layer.pixels, rows)
            la = px[..., 3] * float(layer.opacity)
            if layer.mask is not None:
                la = la * layer.mask[rows]
            if not la.any():
                continue
            top = px[..., :3]
            if layer.blend == "NORMAL":
                blended = top
            else:
                # Where nothing is underneath, blend modes act like Normal.
                blended = np.where(a[..., None] > 0, _blend(layer.blend, c, top), top)
            na = la + a * (1 - la)
            safe = np.maximum(na, 1e-8)[..., None]
            c[...] = (blended * la[..., None] + c * a[..., None] * (1 - la[..., None])) / safe
            a[...] = na
    return out_rgb, out_a


def merge_two(lower: LayerInput, upper: LayerInput, size: int) -> np.ndarray:
    """Merge ``upper`` into ``lower`` and return a float RGBA layer."""
    lo = LayerInput(lower.pixels, lower.opacity, "NORMAL", lower.mask)
    rgb, a = composite([lo, upper], size)
    return np.concatenate([rgb, a[..., None]], axis=2)
