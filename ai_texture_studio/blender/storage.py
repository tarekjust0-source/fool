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


def migrate_temp_files():
    """After the first save, move session files next to the .blend."""
    if not bpy.data.is_saved:
        return
    moved = 0
    for obj in bpy.data.objects:
        st = getattr(obj, "ats", None)
        if st is None or not st.uid:
            continue
        paths = [ly for ly in st.layers]
        if not any(is_temp(ly.filepath) for ly in paths) and not any(is_temp(h.filepath) for h in st.history):
            continue
        dest = object_dir(obj)
        for item in list(st.layers) + list(st.history):
            if item.filepath and is_temp(item.filepath):
                src = resolve(item.filepath)
                if os.path.exists(src):
                    dst = os.path.join(dest, os.path.basename(src))
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    item.filepath = to_stored(dst)
                    moved += 1
        for img in (st.base_image, st.mask_image):
            if img is not None and img.filepath_raw and is_temp(img.filepath_raw):
                src = resolve(img.filepath_raw)
                if os.path.exists(src):
                    dst = os.path.join(dest, os.path.basename(src))
                    shutil.copy2(src, dst)
                    img.filepath_raw = to_stored(dst)
    for img in bpy.data.images:
        if img.get("ats_map") and img.filepath_raw and is_temp(img.filepath_raw):
            src = resolve(img.filepath_raw)
            if os.path.exists(src):
                obj_dir = os.path.join(root_dir(), "maps")
                os.makedirs(obj_dir, exist_ok=True)
                dst = os.path.join(obj_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                img.filepath_raw = to_stored(dst)
    return moved


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
