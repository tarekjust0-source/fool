"""File storage for layers, maps and debug images.

Layers are PNG files (source of truth) plus an in-memory uint8 cache.
Location:
* saved .blend -> ``//ATS_textures/<object>_<uid>/`` next to the .blend
* unsaved      -> a temp folder; moved next to the .blend on first save
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import OrderedDict

import bpy
import numpy as np

from ..core import imageio
from . import compat

_SESSION_DIR = os.path.join(tempfile.gettempdir(), "ai_texture_studio", uuid.uuid4().hex[:8])
_cache_lock = threading.Lock()
_cache: "OrderedDict[str, tuple[float, np.ndarray]]" = OrderedDict()
CACHE_LIMIT_BYTES = 1536 * 1024 * 1024


def prune_debug_runs(obj_dir: str, keep: int = 40) -> int:
    """Keep only the newest ``keep`` intermediate-image folders of an object."""
    d = os.path.join(obj_dir, "views")
    if not os.path.isdir(d):
        return 0
    runs = sorted((e for e in os.scandir(d) if e.is_dir()), key=lambda e: e.stat().st_mtime, reverse=True)
    for e in runs[keep:]:
        shutil.rmtree(e.path, ignore_errors=True)
    return max(0, len(runs) - keep)


def cleanup_stale_sessions(max_age_days: float = 14.0) -> int:
    """Delete session folders of unsaved files older than ``max_age_days``.

    Recent ones are kept because Blender's recovery files may reference them.
    """
    import time
    parent = os.path.dirname(_SESSION_DIR)
    if not os.path.isdir(parent):
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for e in os.scandir(parent):
        if e.is_dir() and e.path != _SESSION_DIR and e.stat().st_mtime < cutoff:
            shutil.rmtree(e.path, ignore_errors=True)
            removed += 1
    return removed


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "object"


def ensure_uid(obj) -> str:
    if not obj.ats.uid:
        obj.ats.uid = uuid.uuid4().hex[:8]
    return obj.ats.uid


def root_dir() -> str:
    if bpy.data.is_saved:
        return bpy.path.abspath("//ATS_textures")
    return _SESSION_DIR


def object_dir(obj) -> str:
    d = os.path.join(root_dir(), f"{safe_name(obj.name)}_{ensure_uid(obj)}")
    os.makedirs(d, exist_ok=True)
    return d


def to_stored(path: str) -> str:
    """Store paths relative to the .blend when possible."""
    if bpy.data.is_saved:
        try:
            return bpy.path.relpath(path)
        except ValueError:  # different drive on Windows
            return path
    return path


def resolve(path: str) -> str:
    return bpy.path.abspath(path) if path else ""


def is_temp(path: str) -> bool:
    return resolve(path).startswith(_SESSION_DIR)


def write_png(path: str, arr: np.ndarray, level: int = 6):
    """Thread-safe PNG write (pure Python encoder)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = imageio.encode_png(imageio.to_uint8(arr), level=level)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    with _cache_lock:
        _cache.pop(os.path.abspath(path), None)


def read_png(path: str) -> np.ndarray:
    """uint8 RGBA (H,W,4) with caching; falls back to Blender's loader."""
    ap = os.path.abspath(resolve(path))
    try:
        mtime = os.path.getmtime(ap)
    except OSError:
        raise FileNotFoundError(f"Layer file missing: {ap}") from None
    with _cache_lock:
        hit = _cache.get(ap)
        if hit and hit[0] == mtime:
            _cache.move_to_end(ap)
            return hit[1]
    with open(ap, "rb") as fh:
        data = fh.read()
    try:
        arr = imageio.decode_png(data)
    except imageio.ImageDecodeError:
        arr = decode_with_blender(data)
    arr = imageio.rgba(arr)
    with _cache_lock:
        _cache[ap] = (mtime, arr)
        total = sum(v[1].nbytes for v in _cache.values())
        while total > CACHE_LIMIT_BYTES and len(_cache) > 1:
            _, (_, old) = _cache.popitem(last=False)
            total -= old.nbytes
    return arr


def clear_cache():
    with _cache_lock:
        _cache.clear()


def decode_with_blender(data: bytes) -> np.ndarray:
    """Decode any image format Blender understands. MAIN THREAD ONLY."""
    fd, tmp = tempfile.mkstemp(suffix=".img", prefix="ats_")
    os.close(fd)
    img = None
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        img = bpy.data.images.load(tmp, check_existing=False)
        if img.size[0] == 0 or img.size[1] == 0:
            raise imageio.InvalidImageError("Blender could not decode the image")
        arr = compat.image_to_array(img)
        return imageio.to_uint8(arr)
    finally:
        if img is not None:
            bpy.data.images.remove(img)
        try:
            os.remove(tmp)
        except OSError:
            pass


def prepare_save(target_blend: str) -> int:
    """``save_pre``: make every add-on path valid for the file being written.

    * files still in the session temp folder are copied to
      ``<blend dir>/ATS_textures/<object>/``;
    * add-on path properties (which Blender does not remap itself) are stored
      relative to the *new* .blend location, so "Save As" elsewhere works;
    * images are given absolute paths here and made relative in ``finish_save``
      (Blender's own relative-remap only rebases paths that are already relative).
    """
    if not target_blend:
        return 0
    new_dir = os.path.dirname(os.path.abspath(target_blend))

    def new_rel(path_abs: str) -> str:
        try:
            return "//" + os.path.relpath(path_abs, new_dir).replace(os.sep, "/")
        except ValueError:  # other drive (Windows)
            return path_abs

    moved = 0
    for obj in bpy.data.objects:
        st = getattr(obj, "ats", None)
        if st is None or not st.uid:
            continue
        dest = os.path.join(new_dir, "ATS_textures", f"{safe_name(obj.name)}_{st.uid}")

        def relocate(src_abs: str) -> str:
            nonlocal moved
            if not is_temp(src_abs) or not os.path.exists(src_abs):
                return src_abs
            os.makedirs(dest, exist_ok=True)
            dst = os.path.join(dest, os.path.basename(src_abs))
            if not os.path.exists(dst):
                shutil.copy2(src_abs, dst)
            moved += 1
            return dst

        for item in list(st.layers) + list(st.history):
            if item.filepath:
                item.filepath = new_rel(relocate(os.path.abspath(resolve(item.filepath))))
        imgs = [st.base_image, st.mask_image]
        imgs += [bpy.data.images.get(n) for n in _json_values(st.maps)]
        for img in imgs:
            if img is not None and img.filepath_raw and is_temp(img.filepath_raw):
                img.filepath_raw = relocate(os.path.abspath(resolve(img.filepath_raw)))
    return moved


def finish_save():
    """``save_post``: store add-on image paths relative to the saved .blend."""
    if not bpy.data.is_saved:
        return
    base = os.path.dirname(bpy.data.filepath)
    for img in bpy.data.images:
        p = img.filepath_raw
        if p and not p.startswith("//") and (img.get("ats_map") or img.name.startswith("ATS_")):
            ap = os.path.abspath(p)
            if ap.startswith(os.path.join(base, "ATS_textures")):
                try:
                    img.filepath_raw = bpy.path.relpath(ap)
                except ValueError:
                    pass


def _json_values(s: str):
    import json
    try:
        return list(json.loads(s or "{}").values())
    except ValueError:
        return []


def save_image_datablock(img: bpy.types.Image, path: str):
    """Write a Blender image to PNG at ``path`` and make it file-backed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    img.filepath_raw = to_stored(path)


def get_or_create_image(name: str, size: int, *, non_color: bool = False, alpha: bool = False) -> bpy.types.Image:
    img = bpy.data.images.get(name)
    if img is None or img.size[0] == 0:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, size, size, alpha=alpha, float_buffer=False)
    elif tuple(img.size) != (size, size):
        img.scale(size, size)
    compat.set_colorspace(img, non_color)
    return img


def load_debug_image(path: str, name: str) -> bpy.types.Image | None:
    ap = resolve(path)
    if not os.path.exists(ap):
        return None
    img = bpy.data.images.get(name)
    if img is not None and bpy.path.abspath(img.filepath) == ap:
        img.reload()
        return img
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.load(ap, check_existing=False)
    img.name = name
    return img
