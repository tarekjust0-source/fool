"""Small compatibility helpers across Blender 4.2 - 5.x."""

from __future__ import annotations

import bpy
import numpy as np

NON_COLOR_NAMES = ("Non-Color", "Generic Data", "Raw", "Linear")
SRGB_NAMES = ("sRGB", "sRGB IEC61966-2.1")


def set_colorspace(image: bpy.types.Image, non_color: bool):
    names = NON_COLOR_NAMES if non_color else SRGB_NAMES
    for n in names:
        try:
            image.colorspace_settings.name = n
            return
        except TypeError:
            continue
    # Fallback for renamed configurations: data images just need is_data.
    if non_color:
        try:
            image.colorspace_settings.is_data = True
        except AttributeError:
            pass


def ensure_nodes(mat):
    """Materials always use nodes from Blender 5.0 ('use_nodes' is deprecated)."""
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    return mat.node_tree


def uv_vectors(uv_layer) -> np.ndarray:
    n = len(uv_layer.uv) if hasattr(uv_layer, "uv") else len(uv_layer.data)
    out = np.empty(n * 2, np.float32)
    if hasattr(uv_layer, "uv"):
        uv_layer.uv.foreach_get("vector", out)
    else:  # pragma: no cover - very old API
        uv_layer.data.foreach_get("uv", out)
    return out.reshape(-1, 2)


def corner_normals(mesh) -> np.ndarray:
    n = len(mesh.loops)
    out = np.empty(n * 3, np.float32)
    if hasattr(mesh, "corner_normals") and len(mesh.corner_normals) == n:
        mesh.corner_normals.foreach_get("vector", out)
    else:  # pragma: no cover - < 4.1
        mesh.calc_normals_split()
        mesh.loops.foreach_get("normal", out)
    return out.reshape(-1, 3)


def progress(layout, factor: float, text: str):
    if hasattr(layout, "progress"):
        layout.progress(factor=factor, type="BAR", text=text)
    else:  # pragma: no cover
        layout.label(text=f"{text} ({factor:.0%})")


def new_group_input(group, name: str, socket_type: str = "NodeSocketFloat"):
    if hasattr(group, "interface"):
        return group.interface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
    return group.inputs.new(socket_type, name)  # pragma: no cover


def image_to_array(image: bpy.types.Image) -> np.ndarray:
    """Float RGBA (H,W,4), row 0 = top."""
    w, h = image.size
    buf = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)[::-1]


def array_to_image(image: bpy.types.Image, arr: np.ndarray):
    """Write float/uint8 (H,W,C) (row 0 = top) into ``image`` (same size)."""
    h, w = arr.shape[:2]
    if tuple(image.size) != (w, h):
        image.scale(w, h)
    a = arr
    if a.dtype == np.uint8:
        a = a.astype(np.float32) * (1.0 / 255.0)
    if a.ndim == 2:
        a = a[..., None]
    c = a.shape[2]
    if c == 1:
        a = np.concatenate([np.repeat(a, 3, axis=2), np.ones((h, w, 1), np.float32)], axis=2)
    elif c == 3:
        a = np.concatenate([a, np.ones((h, w, 1), np.float32)], axis=2)
    image.pixels.foreach_set(np.ascontiguousarray(a[::-1], dtype=np.float32).reshape(-1))
    image.update()
