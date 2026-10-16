"""Texture saving, packing and model export."""

from __future__ import annotations

import os
import shutil

import bpy
import numpy as np

from ..core import imageio, pbr
from . import compat, storage

MAP_SUFFIX = {"BASE_COLOR": "BaseColor", "ROUGHNESS": "Roughness", "METALLIC": "Metallic",
              "NORMAL": "Normal_OpenGL", "HEIGHT": "Height", "AO": "AO"}


def save_textures(obj, maps: dict, folder: str, *, orm: bool, dx_normal: bool) -> list[str]:
    folder = bpy.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    base = storage.safe_name(obj.name)
    written = []
    arrays = {}
    for kind, img in maps.items():
        dst = os.path.join(folder, f"{base}_{MAP_SUFFIX.get(kind, kind.title())}.png")
        src = bpy.path.abspath(img.filepath_raw) if img.filepath_raw else ""
        if src and os.path.exists(src) and not img.is_dirty:
            shutil.copy2(src, dst)
        else:
            arr = compat.image_to_array(img)
            storage.write_png(dst, arr[..., :3] if kind == "BASE_COLOR" else arr[..., :3])
        written.append(dst)
        if kind in ("AO", "ROUGHNESS", "METALLIC", "NORMAL"):
            arrays[kind] = compat.image_to_array(img)
    if orm and ("ROUGHNESS" in arrays or "METALLIC" in arrays):
        size = next(iter(arrays.values())).shape[0]
        rough = arrays.get("ROUGHNESS", np.full((size, size, 4), 0.5, np.float32))[..., 0]
        metal = arrays.get("METALLIC", np.zeros((size, size, 4), np.float32))[..., 0]
        ao = arrays.get("AO")
        ao = ao[..., 0] if ao is not None else None
        if metal.shape != rough.shape:
            metal = imageio.resize(metal, rough.shape[1], rough.shape[0])
        dst = os.path.join(folder, f"{base}_ORM.png")
        storage.write_png(dst, pbr.pack_orm(ao, rough, metal))
        written.append(dst)
    if dx_normal and "NORMAL" in arrays:
        n = arrays["NORMAL"][..., :3].copy()
        n[..., 1] = 1.0 - n[..., 1]
        dst = os.path.join(folder, f"{base}_Normal_DirectX.png")
        storage.write_png(dst, n)
        written.append(dst)
    return written


def pack_images(maps: dict) -> int:
    n = 0
    for img in maps.values():
        if img.packed_file is None:
            img.pack()
            n += 1
    return n


def export_model(context, obj, path: str, fmt: str) -> str:
    path = bpy.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prev_sel = [o for o in context.view_layer.objects if o.select_get()]
    prev_active = context.view_layer.objects.active
    prev_mode = obj.mode
    try:
        if prev_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.view_layer.objects:
            o.select_set(o == obj)
        context.view_layer.objects.active = obj
        if fmt == "GLB":
            if not path.lower().endswith(".glb"):
                path += ".glb"
            bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True)
        else:
            if not path.lower().endswith(".fbx"):
                path += ".fbx"
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, path_mode="COPY", embed_textures=True,
                                     mesh_smooth_type="FACE", add_leaf_bones=False)
    finally:
        for o in context.view_layer.objects:
            o.select_set(o in prev_sel)
        context.view_layer.objects.active = prev_active
        if prev_mode != "OBJECT" and prev_active == obj:
            try:
                bpy.ops.object.mode_set(mode=prev_mode)
            except RuntimeError:
                pass
    return path
