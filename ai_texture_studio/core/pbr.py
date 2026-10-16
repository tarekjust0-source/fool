"""Derivation of auxiliary material maps from a base colour texture.

IMPORTANT – provenance.  Image-generation models produce colour images.  None
of the supported providers returns physically measured roughness, metallic or
normal maps, so the maps below are clearly labelled approximations:

* height    – APPROXIMATION: high-pass luminance (assumes darker = recessed).
* normal    – APPROXIMATION: tangent-space normals derived from that height.
* roughness – APPROXIMATION: material preset base value modulated by detail.
* metallic  – USER-AUTHORED: a preset/uniform value, optionally masked.
* AO        – GEOMETRIC: baked from the actual mesh by Blender/Cycles (see
              blender/bake.py), not derived here.

The add-on's UI shows these labels next to every map.
"""

from __future__ import annotations

import numpy as np

from . import imageio, masks

MAP_PROVENANCE = {
    "BASE_COLOR": "AI-generated",
    "ROUGHNESS": "Approximation (derived from base colour + material preset)",
    "METALLIC": "User-authored (preset value, optional mask)",
    "NORMAL": "Approximation (derived from estimated height)",
    "HEIGHT": "Approximation (high-pass luminance)",
    "AO": "Baked from mesh geometry (Cycles)",
}

# appearance -> (base roughness, metallic, roughness variation sign)
MATERIAL_PRESETS = {
    "AUTO": (0.6, 0.0, 1.0),
    "METAL": (0.35, 1.0, 1.0),
    "PAINTED_METAL": (0.5, 0.0, 1.0),
    "WOOD": (0.7, 0.0, 1.0),
    "LEATHER": (0.55, 0.0, 1.0),
    "STONE": (0.85, 0.0, 1.0),
    "FABRIC": (0.9, 0.0, 0.5),
    "SKIN": (0.5, 0.0, 0.5),
    "PLASTIC": (0.4, 0.0, 0.5),
    "CERAMIC": (0.25, 0.0, 0.5),
}


def height_from_color(rgb: np.ndarray, valid: np.ndarray | None = None, *, radius: int = 24,
                      detail: float = 1.0) -> np.ndarray:
    """High-pass luminance mapped to 0..1 (0.5 = mid)."""
    f = imageio.to_float(rgb)[..., :3]
    lum = imageio.luminance(f)
    if valid is not None:
        lum, _ = masks.fill_gutter(lum, valid, 16)
    low = masks.box_blur(masks.box_blur(lum, max(1, radius // 2)), max(1, radius // 2))
    hp = lum - low
    scale = np.percentile(np.abs(hp), 98) + 1e-6
    h = 0.5 + 0.5 * np.clip(hp / scale * float(detail), -1, 1)
    return h.astype(np.float32)


def normal_from_height(height: np.ndarray, strength: float = 2.0, *, directx: bool = False) -> np.ndarray:
    """Tangent-space normal map (0..1 RGB) from a 0..1 height map.

    Image rows go top->bottom while V goes up, so dh/dv = -(dh/drow).
    OpenGL convention (Blender, glTF) has +Y = +V; DirectX flips green.
    """
    h = height.astype(np.float32)
    s = float(strength) * h.shape[0] / 512.0
    dhdx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    dhdrow = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    dhdv = -dhdrow
    nx = -dhdx * s
    ny = -dhdv * s
    nz = np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    if directx:
        n[..., 1] *= -1
    return (n * 0.5 + 0.5).astype(np.float32)


def roughness_map(rgb: np.ndarray, preset: str = "AUTO", *, base: float | None = None,
                  variation: float = 0.25, wear: float = 0.0) -> np.ndarray:
    pr = MATERIAL_PRESETS.get(preset, MATERIAL_PRESETS["AUTO"])
    b = pr[0] if base is None else float(base)
    f = imageio.to_float(rgb)[..., :3]
    lum = imageio.luminance(f)
    local = masks.box_blur(lum, 8)
    detail = lum - local
    # Brighter local detail on worn surfaces (edges, scratches) tends to be
    # polished; darker grime is rougher.  Purely heuristic.
    r = b - detail * variation * 2.0 * pr[2] + (0.5 - lum) * variation * 0.3
    r = r + wear * 0.1
    return np.clip(r, 0.02, 1.0).astype(np.float32)


def metallic_map(size: int, value: float, mask: np.ndarray | None = None) -> np.ndarray:
    m = np.full((size, size), float(value), np.float32)
    if mask is not None:
        mk = mask.astype(np.float32)
        if mk.shape != (size, size):
            mk = imageio.resize(mk, size, size)
        m = m * mk
    return m


def pack_orm(ao: np.ndarray | None, rough: np.ndarray, metal: np.ndarray) -> np.ndarray:
    """glTF/Unreal style channel pack: R=AO, G=Roughness, B=Metallic."""
    size = rough.shape[0]
    a = np.ones((size, size), np.float32) if ao is None else ao.astype(np.float32)
    if a.shape != rough.shape:
        a = imageio.resize(a, rough.shape[1], rough.shape[0])
    return np.stack([a, rough, metal], axis=-1)


def pixelate(rgb: np.ndarray, block: int, colors: int = 0) -> np.ndarray:
    """Pixel-art post process: block averaging + optional colour quantisation."""
    f = imageio.to_float(rgb)[..., :3]
    h, w = f.shape[:2]
    b = max(1, int(block))
    small = imageio.resize(f, max(1, w // b), max(1, h // b))
    if colors and colors > 1:
        levels = float(colors - 1)
        small = np.round(small * levels) / levels
    return imageio.resize_nearest(small, w, h).astype(np.float32)


def quantize_to_palette(rgb: np.ndarray, palette: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Pull colours towards the nearest palette entry (``amount`` 0..1)."""
    f = imageio.to_float(rgb)[..., :3]
    pal = np.asarray(palette, np.float32).reshape(-1, 3)
    if pal.size == 0 or amount <= 0:
        return f
    flat = f.reshape(-1, 3)
    out = np.empty_like(flat)
    for s in range(0, len(flat), 1 << 20):
        chunk = flat[s:s + (1 << 20)]
        d = ((chunk[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        out[s:s + len(chunk)] = pal[np.argmin(d, axis=1)]
    res = flat * (1 - amount) + out * amount
    return res.reshape(f.shape).astype(np.float32)
