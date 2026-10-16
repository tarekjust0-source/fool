"""Layer stack management, compositing to the base colour image, undo/redo.

The layer stack lives in ``obj.ats.layers``; pixels live in PNG files.
Compositing produces ``<object>_BaseColor`` which the material uses.  Layer
operations push JSON snapshots of the stack (files are never deleted by
layer operations), giving reliable undo/redo independent of Blender's
global undo, which does not track image pixels.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import bpy
import numpy as np

from ..core import compositing, imageio, masks, pbr
from . import compat, storage
from .props import NON_CONFIDENT_KINDS, PROTECTED_KINDS

UNDO_DEPTH = 40
_pending_recomposite: set[str] = set()


# ---------------------------------------------------------------------------
# snapshots / undo
# ---------------------------------------------------------------------------

_LAYER_FIELDS = ("uid", "name", "kind", "visible", "opacity", "blend", "filepath", "pending", "created", "info",
                 "history_uid", "view_key", "protected")


def snapshot(obj) -> list:
    return [{f: getattr(ly, f) for f in _LAYER_FIELDS} for ly in obj.ats.layers]


def restore_snapshot(obj, snap: list):
    st = obj.ats
    st.layers.clear()
    for d in snap:
        ly = st.layers.add()
        for f in _LAYER_FIELDS:
            if f in d:
                setattr(ly, f, d[f])
    st.active_layer = min(st.active_layer, max(0, len(st.layers) - 1))


def push_undo(obj, label: str):
    st = obj.ats
    stack = json.loads(st.undo_stack or "[]")
    stack.append({"label": label, "layers": snapshot(obj), "active": st.active_layer})
    st.undo_stack = json.dumps(stack[-UNDO_DEPTH:])
    st.redo_stack = "[]"


def can_undo(obj) -> bool:
    return len(json.loads(obj.ats.undo_stack or "[]")) > 0


def can_redo(obj) -> bool:
    return len(json.loads(obj.ats.redo_stack or "[]")) > 0


def undo(obj) -> str | None:
    st = obj.ats
    stack = json.loads(st.undo_stack or "[]")
    if not stack:
        return None
    entry = stack.pop()
    redo = json.loads(st.redo_stack or "[]")
    redo.append({"label": entry["label"], "layers": snapshot(obj), "active": st.active_layer})
    restore_snapshot(obj, entry["layers"])
    st.active_layer = min(entry.get("active", 0), max(0, len(st.layers) - 1))
    st.undo_stack = json.dumps(stack)
    st.redo_stack = json.dumps(redo[-UNDO_DEPTH:])
    recomposite(obj)
    return entry["label"]


def redo(obj) -> str | None:
    st = obj.ats
    redo_s = json.loads(st.redo_stack or "[]")
    if not redo_s:
        return None
    entry = redo_s.pop()
    stack = json.loads(st.undo_stack or "[]")
    stack.append({"label": entry["label"], "layers": snapshot(obj), "active": st.active_layer})
    restore_snapshot(obj, entry["layers"])
    st.active_layer = min(entry.get("active", 0), max(0, len(st.layers) - 1))
    st.undo_stack = json.dumps(stack[-UNDO_DEPTH:])
    st.redo_stack = json.dumps(redo_s)
    recomposite(obj)
    return entry["label"]


# ---------------------------------------------------------------------------
# layer CRUD
# ---------------------------------------------------------------------------

def texture_size(obj, scene_settings=None) -> int:
    st = obj.ats
    if st.texture_size:
        return st.texture_size
    if scene_settings is not None:
        return int(scene_settings.texture_size)
    return 2048


def add_layer(obj, pixels: np.ndarray | None, *, name: str, kind: str, info: str = "", filepath: str = "",
              pending: bool = False, history_uid: str = "", index: int | None = None, push: bool = True,
              view_key: str = "", protected: bool | None = None):
    """Add a layer. ``pixels`` (uint8 RGBA) is written to disk unless ``filepath`` given."""
    st = obj.ats
    if push:
        push_undo(obj, f"Add {name}")
    uid = uuid.uuid4().hex[:10]
    if not filepath:
        filepath = os.path.join(storage.object_dir(obj), f"layer_{uid}.png")
        storage.write_png(filepath, pixels)
    ly = st.layers.add()
    ly.uid = uid
    ly.name = name
    ly.kind = kind
    ly.filepath = storage.to_stored(filepath)
    ly.info = info[:1000]
    ly.pending = pending
    ly.created = time.strftime("%Y-%m-%d %H:%M:%S")
    ly.history_uid = history_uid
    ly.view_key = view_key
    ly.protected = (kind in PROTECTED_KINDS) if protected is None else protected
    if index is not None:
        st.layers.move(len(st.layers) - 1, index)
        st.active_layer = index
    else:
        st.active_layer = len(st.layers) - 1
    return st.layers[st.active_layer]


def find_layer(obj, uid: str):
    for i, ly in enumerate(obj.ats.layers):
        if ly.uid == uid:
            return i, ly
    return -1, None


def remove_layer(obj, index: int, push: bool = True):
    st = obj.ats
    if not (0 <= index < len(st.layers)):
        return
    if push:
        push_undo(obj, f"Delete {st.layers[index].name}")
    st.layers.remove(index)
    st.active_layer = min(max(0, index - 1), max(0, len(st.layers) - 1))


def duplicate_layer(obj, index: int):
    st = obj.ats
    src = st.layers[index]
    push_undo(obj, f"Duplicate {src.name}")
    px = storage.read_png(src.filepath)
    ly = add_layer(obj, px, name=f"{src.name} copy", kind=src.kind, info=src.info, push=False, index=index + 1)
    ly.opacity = src.opacity
    ly.blend = src.blend
    ly.visible = src.visible
    return ly


def move_layer(obj, index: int, direction: int):
    st = obj.ats
    new = index + direction
    if not (0 <= index < len(st.layers)) or not (0 <= new < len(st.layers)):
        return
    push_undo(obj, f"Move {st.layers[index].name}")
    st.layers.move(index, new)
    st.active_layer = new


def layer_pixels(obj, ly, size: int) -> np.ndarray:
    px = storage.read_png(ly.filepath)
    if px.shape[0] != size or px.shape[1] != size:
        px = imageio.resize(px, size, size)
    return px


def merge_down(obj, index: int):
    st = obj.ats
    if index <= 0 or index >= len(st.layers):
        return None
    size = texture_size(obj)
    upper, lower = st.layers[index], st.layers[index - 1]
    push_undo(obj, f"Merge {upper.name}")
    merged = compositing.merge_two(
        compositing.LayerInput(layer_pixels(obj, lower, size), lower.opacity if lower.visible else 0.0),
        compositing.LayerInput(layer_pixels(obj, upper, size), upper.opacity if upper.visible else 0.0, upper.blend),
        size)
    name = lower.name
    st.layers.remove(index)
    st.layers.remove(index - 1)
    ly = add_layer(obj, imageio.to_uint8(merged), name=name, kind="MERGED", push=False, index=index - 1)
    recomposite(obj)
    return ly


def flatten(obj):
    st = obj.ats
    if len(st.layers) < 2:
        return None
    size = texture_size(obj)
    push_undo(obj, "Flatten")
    rgb, a = compositing.composite(visible_inputs(obj, size), size)
    st.layers.clear()
    ly = add_layer(obj, imageio.to_uint8(np.concatenate([rgb, a[..., None]], axis=2)), name="Flattened",
                   kind="MERGED", push=False)
    recomposite(obj)
    return ly


def visible_inputs(obj, size: int, *, include_pending: bool = True, exclude_uid: str = "") -> list:
    out = []
    for ly in obj.ats.layers:
        if not ly.visible or ly.opacity <= 0 or ly.uid == exclude_uid:
            continue
        if ly.pending and not include_pending:
            continue
        try:
            px = layer_pixels(obj, ly, size)
        except FileNotFoundError:
            continue
        out.append(compositing.LayerInput(px, ly.opacity, ly.blend))
    return out


def missing_layer_files(obj) -> list[str]:
    return [ly.name for ly in obj.ats.layers if not os.path.exists(storage.resolve(ly.filepath))]


def current_texture(obj, size: int, *, include_pending=True):
    """Composite of visible layers -> (rgb float (S,S,3), alpha float (S,S))."""
    return compositing.composite(visible_inputs(obj, size, include_pending=include_pending), size)


def planning_texture(obj, size: int, *, exclude_view_key: str = "", protected_only: bool = False):
    """What a new generation should treat as already painted.

    * low-confidence layers (grazing-angle pixels, colour-extension fill) are
      left out, so a better or newly added view can repaint those texels;
    * layers of ``exclude_view_key`` are left out (single-view regeneration);
    * ``protected_only`` keeps only protected layers ("Start From Scratch").

    Returns (rgb, alpha, protected, grazing): ``protected`` (S,S) bool marks
    texels of protected layers, which projection never writes; ``grazing`` is
    the RGBA of low-angle AI layers (or None), kept unless a view improves them.
    """
    inputs, prot_inputs, grazing_inputs = [], [], []
    for ly in obj.ats.layers:
        if not ly.visible or ly.opacity <= 0 or ly.pending:
            continue
        if ly.kind == "GRAZING" and not protected_only:
            try:
                grazing_inputs.append(compositing.LayerInput(layer_pixels(obj, ly, size), ly.opacity))
            except FileNotFoundError:
                pass
            continue
        if ly.kind in NON_CONFIDENT_KINDS:
            continue
        if exclude_view_key and ly.view_key == exclude_view_key:
            continue
        if protected_only and not ly.protected:
            continue
        try:
            px = layer_pixels(obj, ly, size)
        except FileNotFoundError:
            continue
        li = compositing.LayerInput(px, ly.opacity, ly.blend)
        inputs.append(li)
        if ly.protected:
            prot_inputs.append(li)
    rgb, alpha = compositing.composite(inputs, size)
    if prot_inputs:
        _, pa = compositing.composite(prot_inputs, size)
        protected = pa > 0.5
    else:
        protected = np.zeros((size, size), bool)
    grazing = None
    if grazing_inputs:
        g_rgb, g_a = compositing.composite(grazing_inputs, size)
        grazing = np.concatenate([g_rgb, g_a[..., None]], axis=2)
    return rgb, alpha, protected, grazing


def covered_mask(obj, size: int) -> np.ndarray:
    """Texels covered by the object's working UV map (for tests/diagnostics)."""
    from ..core import raster
    from . import meshio
    mesh = meshio.extract(obj, uv_name=obj.ats.uv_map or None, use_modifiers=False)
    return raster.rasterize_uv(mesh.tri_uvs, size).covered_mask()


# ---------------------------------------------------------------------------
# output image
# ---------------------------------------------------------------------------

def base_image_name(obj) -> str:
    return f"ATS_{obj.name}_BaseColor"


def recomposite(obj, *, save: bool = True, palette=None) -> bpy.types.Image | None:
    """Composite visible layers into the base colour image (main thread)."""
    st = obj.ats
    size = texture_size(obj)
    if not st.layers:
        return st.base_image
    rgb, a = current_texture(obj, size)
    valid = a > 0.02
    if st.pixelate:
        rgb = pbr.pixelate(rgb, st.pixel_size, st.pixel_colors)
    if st.palette_lock > 0:
        pal = palette
        if pal is None:
            sc = bpy.context.scene.ats if hasattr(bpy.context, "scene") and bpy.context.scene else None
            if sc is not None and sc.use_palette:
                from .props import palette as pal_fn
                pal = pal_fn(sc)
        if pal:
            rgb = pbr.quantize_to_palette(rgb, np.array(pal, np.float32), st.palette_lock) * valid[..., None] + \
                rgb * (~valid[..., None])
    # Padding around islands, then fill the rest so mips never see black.
    if valid.any():
        rgb, v2 = masks.fill_gutter(rgb, valid, st.gutter)
        if not v2.all():
            mean = rgb[v2].mean(axis=0)
            rgb[~v2] = mean
    img = st.base_image
    if img is None or img.name not in bpy.data.images:
        img = storage.get_or_create_image(base_image_name(obj), size)
        img["ats_map"] = "BASE_COLOR"
        st.base_image = img
    elif tuple(img.size) != (size, size):
        img.scale(size, size)
    compat.array_to_image(img, rgb)
    if save:
        path = os.path.join(storage.object_dir(obj), f"{storage.safe_name(obj.name)}_BaseColor.png")
        storage.save_image_datablock(img, path)
    return img


def write_preview(obj, rgb_u8: np.ndarray) -> bpy.types.Image:
    """Fast path during generation: show a worker-computed composite."""
    st = obj.ats
    size = rgb_u8.shape[0]
    img = st.base_image
    if img is None or img.name not in bpy.data.images:
        img = storage.get_or_create_image(base_image_name(obj), size)
        img["ats_map"] = "BASE_COLOR"
        st.base_image = img
    compat.array_to_image(img, rgb_u8)
    return img


def schedule_recomposite(obj, delay: float = 0.15):
    """Debounced recomposite (property sliders fire many updates)."""
    name = obj.name
    if name in _pending_recomposite:
        return
    _pending_recomposite.add(name)

    def run():
        _pending_recomposite.discard(name)
        o = bpy.data.objects.get(name)
        if o is not None:
            try:
                recomposite(o, save=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[AI Texture Studio] recomposite failed: {exc}")
        return None

    if bpy.app.background:
        run()
    else:
        bpy.app.timers.register(run, first_interval=delay)
